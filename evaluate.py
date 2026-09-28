# -*- coding: utf-8 -*-
"""
evaluate.py —— 连续记录推理与事件级评估

评估方式与临床使用一致：**按整段连续 EEG 逐窗打分，还原为采样点级概率序列，
经平滑与阈值化后形成"报警事件"，再与真实发作做事件级匹配。**

主要终点：
    · 事件级灵敏度 = 检出的真实发作数 / 真实发作总数
    · 假阳性率 FA/24h = 假报警次数 / 记录总时长 × 24
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from preprocess import read_continuous
from utils.metrics import aggregate_records, evaluate_continuous


# ------------------------------------------------------------------ #
def load_continuous(entry: dict, cfg, logger=None):
    """优先读缓存，否则现场重跑预处理（不写窗口文件）。"""
    cache = entry.get("cache_path")
    if cache and Path(cache).exists():
        d = np.load(cache)
        return (d["X"].astype(np.float32), d["y"].astype(np.uint8),
                float(np.atleast_1d(d["sfreq"])[0]), entry.get("seizures", []))
    rec = {"edf": entry["edf"], "record": entry["record"],
           "patient": entry["patient"], "seizures": entry.get("seizures", [])}
    cont = read_continuous(rec, cfg, logger)
    if cont is None:
        return None
    return cont["X"], cont["y_sample"], cont["sfreq"], cont["seizures"]


@torch.no_grad()
def predict_record(model, X: np.ndarray, cfg, device, channel_idx=None,
                   batch_size: int = 256):
    """
    对一条记录做逐窗推理并还原采样点级概率。
    返回 (prob_sample(T,), p_win(n,), y_win(n,))
    """
    model.eval()
    sfreq = float(cfg.sfreq)
    win = int(round(cfg.win_sec * sfreq))
    hop = int(round(cfg.hop_sec * sfreq))
    if channel_idx is not None:
        X = X[list(channel_idx)]
    C, T = X.shape
    if T < win:
        return np.zeros(T, dtype=np.float32), np.zeros(0), np.zeros(0, np.uint8)
    starts = np.arange(0, T - win + 1, hop, dtype=np.int64)
    W = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        W[i] = X[:, s:s + win]

    probs = []
    for i in range(0, len(W), batch_size):
        b = torch.from_numpy(W[i:i + batch_size]).to(device)
        logit = model(b)
        if isinstance(logit, tuple):
            logit = logit[0]
        probs.append(torch.sigmoid(logit).detach().cpu().numpy())
    p_win = np.concatenate(probs).astype(np.float32)

    # 窗口概率 -> 采样点概率（重叠处取最大值）
    prob = np.full(T, -np.inf, dtype=np.float32)
    idx = starts[:, None] + np.arange(win, dtype=np.int64)[None, :]
    np.maximum.at(prob, idx.ravel(), np.repeat(p_win, win))
    prob[~np.isfinite(prob)] = 0.0
    return prob, p_win, None


def evaluate_patient(model, entries, cfg, device, channel_idx=None,
                     logger=None, batch_size=256, return_arrays=True):
    """
    对一位患者的所有记录做评估。
    返回 dict{ 'records': [...], 'agg': {...}, 'y_true': ..., 'y_prob': ... }
    """
    rec_results, ys, ps = [], [], []
    for e in entries:
        loaded = load_continuous(e, cfg, logger)
        if loaded is None:
            continue
        X, y_sample, sfreq, seizures = loaded
        prob, p_win, _ = predict_record(model, X, cfg, device, channel_idx, batch_size)
        y_win = y_sample.reshape(1, -1)
        # 窗口级标签（用于 ROC）
        win = int(round(cfg.win_sec * sfreq))
        if len(p_win):
            csum = np.concatenate([[0], np.cumsum(y_sample)])
            st = np.arange(0, len(y_sample) - win + 1, int(round(cfg.hop_sec * sfreq)))
            yw = ((csum[st + win] - csum[st]) / win >= cfg.ictal_overlap).astype(np.uint8)
            ys.append(yw[:len(p_win)])
            ps.append(p_win)
        r = evaluate_continuous(y_sample, prob, sfreq, cfg)
        r.update(record=e.get("record"), patient=e.get("patient"), sfreq=sfreq)
        rec_results.append(r)
        if logger:
            logger.info(f"    {e.get('record')}: 发作 {r['n_detected']}/{r['n_seizures']}，"
                        f"假报警 {r['n_false_alarms']}，{r['hours']:.1f} h")

    agg = aggregate_records(rec_results)
    out = {"records": rec_results, "agg": agg}
    if return_arrays and ys:
        yy = np.concatenate(ys)
        pp = np.concatenate(ps)
        # 控制用于 ROC 的样本规模：保留全部正样本 + 最多 20 万负样本
        if len(yy) > 200_000:
            rng = np.random.default_rng(cfg.seed)
            pos = np.where(yy == 1)[0]
            neg = rng.choice(np.where(yy == 0)[0], 200_000 - len(pos), replace=False)
            sel = np.concatenate([pos, neg])
            yy, pp = yy[sel], pp[sel]
        out["y_true"], out["y_prob"] = yy, pp
    return out


def tune_threshold_event_level(model, entries, cfg, device, channel_idx=None,
                               budget_fa=None, grid=None, logger=None):
    """
    在验证患者的**连续记录**上按事件级指标标定概率阈值。

    为什么不直接用窗级 F1 最优阈值？
        窗级 F1 是在"已抽样的平衡验证窗口"上算的，而真实使用是连续推理——
        一段 1 h 记录有约 1800 个窗口，其中仅个位数是发作。
        窗级最优阈值（常在 0.2~0.4）放到连续序列上会把孤立尖峰也算成报警，
        实测 FA/24h 可达上百次，临床不可用。

    本函数直接在连续序列上扫描阈值，取"FA/24h 不超过预算"约束下灵敏度最高者；
    若所有阈值都超预算，则取 FA 最低者（保证结果始终有定义）。

    返回 (best_threshold, info)
    """
    from utils.metrics import event_counts_fast, movavg_same

    budget = float(cfg.fa_budget if budget_fa is None else budget_fa)
    # 先做一次连续推理，供所有阈值复用（避免重复前向传播）
    recs = []
    for e in entries:
        loaded = load_continuous(e, cfg, logger)
        if loaded is None:
            continue
        X, y_sample, sfreq, seizures = loaded
        prob, _, _ = predict_record(model, X, cfg, device, channel_idx)
        if len(prob):
            recs.append((y_sample, prob, sfreq))
    if not recs:
        return float(cfg.threshold), {"note": "验证集为空，沿用默认阈值"}

    if grid is None:
        # 低阈值区在连续推理下 FA 必然爆表，但保留到 0.30 以便在"模型已充分
        # 学到发作间期分布"时不至于人为抬高风险；高阈值区对 FA 极敏感，加密采样。
        hi = float(getattr(cfg, "thr_grid_hi", 0.999))
        grid = np.unique(np.round(np.concatenate([
            np.linspace(0.30, 0.90, 25), np.linspace(0.91, 0.98, 15),
            np.linspace(0.985, hi, 6)]), 4))
    # 平滑只与 (记录, smooth_sec) 有关，与阈值无关 → 每条记录只算一次
    smoothed = [(y, movavg_same(p, max(1, int(round(cfg.smooth_sec * sf)))),
                 sf) for y, p, sf in recs]
    rows = []
    for t in grid:
        ts = td = tf = 0
        hrs = 0.0
        for y, ps, sf in smoothed:
            a, b, c, h = event_counts_fast(
                y, ps, sf, float(t), cfg.merge_gap_sec, cfg.min_event_sec,
                cfg.min_overlap_sec, prob_smoothed=ps)
            ts += a; td += b; tf += c; hrs += h
        rows.append({
            "threshold": float(t),
            "sensitivity": (td / ts) if ts else 0.0,
            "fa_per_24h": (tf / hrs * 24.0) if hrs > 0 else 0.0,
            "n_seizures": ts, "n_detected": td, "n_false_alarms": tf,
        })
    feas = [r for r in rows if r["fa_per_24h"] <= budget]
    if feas:
        # 同灵敏度时取 FA 更低者
        best = max(feas, key=lambda r: (round(r["sensitivity"], 6), -r["fa_per_24h"]))
    else:
        best = min(rows, key=lambda r: (r["fa_per_24h"], -round(r["sensitivity"], 6)))
    return float(best["threshold"]), {"budget_fa_per_24h": budget, "grid": rows,
                                      "chosen": best}


def evaluate_patients(model, entries_by_patient: dict, cfg, device,
                      channel_idx=None, logger=None, batch_size=256):
    """对多位患者逐一评估，返回 {patient: result}"""
    out = {}
    for p, entries in entries_by_patient.items():
        if logger:
            logger.info(f"  评估患者 {p}（{len(entries)} 条记录）")
        out[p] = evaluate_patient(model, entries, cfg, device, channel_idx, logger, batch_size)
    return out
