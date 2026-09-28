# -*- coding: utf-8 -*-
"""
metrics.py —— 评价指标

本项目的核心终点指标（与《癫痫杂志》论著"结果"部分一一对应）：

【共同主要终点】
  1. 事件级灵敏度 (event-level sensitivity) = 被检出的真实发作次数 / 真实发作总数
  2. 每 24 小时假阳性数 (false alarms per 24 h, FA/24h)

【次要终点】
  - 样本/窗级 AUC-ROC、AUPRC、F1、灵敏度、特异度
  - 患者级灵敏度（≥50% 发作被检出的患者比例）
  - 检出延迟 latency（报警起点 - 发作起点，中位数）
  - 发作时长覆盖率（真阳性重叠时长 / 发作总时长）

注：癫痫发作检测领域公认"窗级准确度"会因极端类别不平衡而虚高，
因此必须以"事件级"指标作为主要终点，窗级指标仅作补充。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def movavg_same(x: np.ndarray, w: int) -> np.ndarray:
    """等价于 np.convolve(x, np.ones(w)/w, mode='same')，但用累积和实现 O(T)。

    直接卷积在 w=1024（8 s × 128 Hz）、T≈46 万（1 h 记录）时慢到秒级，
    而阈值网格扫描要把它重复执行"阈值数 × 记录数"次（每折上千次），
    是整个 LOPO 流程的主要瓶颈。numpy 的 mode='same' 定义为
    full[(w-1)//2 : (w-1)//2 + len(x)]，此处按同一偏移用前缀和实现。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if w <= 1:
        return x.copy()
    c = np.concatenate([[0.0], np.cumsum(x)])          # c[k] = sum(x[:k])
    idx = (w - 1) // 2 + np.arange(n)
    hi = np.clip(idx, 0, n - 1)
    lo = np.clip(idx - w + 1, 0, n - 1)
    return (c[hi + 1] - c[lo]) / float(w)


def event_counts_fast(y_sample: np.ndarray, prob_sample: np.ndarray,
                      sfreq: float, threshold: float, merge_gap_sec: float,
                      min_dur_sec: float, min_overlap_sec: float,
                      smooth_sec: float = 0.0, prob_smoothed=None):
    """只算事件级四元组 (n_true, n_detected, n_false_alarms, hours)。

    与 evaluate_continuous 的事件判定逻辑完全一致，但跳过 ROC/AUC 等
    采样点级计算，用于阈值网格扫描（快 10~20 倍）。
    prob_smoothed 传入可复用已平滑的序列，避免重复计算。
    """
    y = np.asarray(y_sample).astype(int).ravel()
    n = y.size
    if prob_smoothed is None:
        p = np.asarray(prob_sample, dtype=np.float64).ravel()[:n]
        w = max(1, int(round(smooth_sec * sfreq)))
        p = movavg_same(p, w) if w > 1 else p
    else:
        p = np.asarray(prob_smoothed, dtype=np.float64).ravel()[:n]
    pred = merge_events(events_from_mask(p >= float(threshold), sfreq),
                        gap_sec=merge_gap_sec, min_dur_sec=min_dur_sec)
    true = events_from_mask(y, sfreq)
    pairs, fp_idx, _ = match_events(pred, true, min_overlap_sec)
    return len(true), len(pairs), len(fp_idx), n / float(sfreq) / 3600.0


def _run_bounds(mask: np.ndarray):
    """二值序列 -> (starts, ends) 采样点下标；ends 为末个 True 之后的位置。"""
    pad = np.concatenate([[0], np.asarray(mask).astype(np.int8).ravel(), [0]])
    d = np.diff(pad)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def grid_event_counts(y_sample: np.ndarray, prob_smoothed: np.ndarray,
                      sfreq: float, thresholds, min_durs,
                      merge_gap_sec: float, min_overlap_sec: float):
    """一次性算出「阈值 × 最短事件时长」整张网格的事件级四元组。

    返回 (n_thr, n_md, 4)，四个通道依次为
        [0] n_true  [1] n_detected  [2] n_false_alarms  [3] hours

    与逐个调用 event_counts_fast 的结果严格一致，但：
      - 合并用 reduceat 向量化，匹配用广播向量化（原为 Python 双重循环）；
      - 每个阈值只做一次阈值化与游程提取，多个 min_dur 复用同一批事件。
    实测在阈值网格扫描场景（数十阈值 × 上百条长记录）快 1~2 个数量级。
    """
    y = np.asarray(y_sample).astype(int).ravel()
    p = np.asarray(prob_smoothed, dtype=np.float64).ravel()[:y.size]
    sfreq = float(sfreq)
    hours = y.size / sfreq / 3600.0

    true = events_from_mask(y, sfreq)
    n_t = len(true)
    thr = np.atleast_1d(np.asarray(thresholds, dtype=float))
    mds = np.atleast_1d(np.asarray(min_durs, dtype=float))
    out = np.zeros((thr.size, mds.size, 4))
    out[:, :, 0] = float(n_t)
    out[:, :, 3] = hours
    gap_s = float(merge_gap_sec) * sfreq

    for ti, t in enumerate(thr):
        mask = p >= float(t)
        if not mask.any():
            continue                                  # 无报警：det=0, fa=0
        starts, ends = _run_bounds(mask)
        if starts.size == 0:
            continue
        if starts.size > 1:
            new_grp = np.concatenate([[True], (starts[1:] - ends[:-1]) > gap_s])
            off = np.flatnonzero(new_grp)
            gs = np.minimum.reduceat(starts, off)
            ge = np.maximum.reduceat(ends, off)
        else:
            gs, ge = starts, ends
        dur = (ge - gs) / sfreq
        ps, pe = gs / sfreq, ge / sfreq
        if n_t:
            ov = np.maximum(0.0, np.minimum(pe[:, None], true[None, :, 1])
                            - np.maximum(ps[:, None], true[None, :, 0]))
        for di, md in enumerate(mds):
            keep = np.flatnonzero(dur >= float(md))
            if keep.size == 0:
                continue
            if not n_t:
                out[ti, di, 2] = keep.size             # 无发作记录：全部为假报警
                continue
            sub = ov[keep]
            ii, jj = np.nonzero(sub >= float(min_overlap_sec))
            if ii.size == 0:
                out[ti, di, 2] = keep.size
                continue
            order = np.argsort(-sub[ii, jj], kind="stable")
            used_p, used_t = set(), set()
            n_det = 0
            for k in order:
                i, j = int(ii[k]), int(jj[k])
                if i in used_p or j in used_t:
                    continue
                used_p.add(i)
                used_t.add(j)
                n_det += 1
            out[ti, di, 1] = n_det
            out[ti, di, 2] = keep.size - len(used_p)
    return out


# ------------------------------------------------------------------ #
# 一、窗级 / 样本级指标
# ------------------------------------------------------------------ #
def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    out = {"threshold": float(threshold), "n_pos": int(y_true.sum()),
           "n_neg": int((y_true == 0).sum())}
    if y_true.sum() == 0 or (y_true == 0).sum() == 0:
        # 单类情况无法计算 AUC，返回 NaN，聚合时自动忽略
        out.update({"auc": np.nan, "auprc": np.nan})
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    if y_true.sum() == 0 and y_pred.sum() == 0:
        out.update(sensitivity=np.nan, specificity=1.0, precision=np.nan,
                   f1=np.nan, accuracy=1.0)
        return out
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                       labels=[0, 1], zero_division=0)
    out.update(
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        sensitivity=float(tp / (tp + fn)) if (tp + fn) else np.nan,
        specificity=float(tn / (tn + fp)) if (tn + fp) else np.nan,
        precision=float(prec[1]),
        npv=float(tn / (tn + fn)) if (tn + fn) else np.nan,
        f1=float(f1[1]),
        accuracy=float((tp + tn) / len(y_true)),
    )
    return out


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        metric: str = "f1", grid=None) -> Tuple[float, float]:
    """在验证集上搜索最优概率阈值。"""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if grid is None:
        # 下限设 0.10：过低阈值虽能提高 F1，却会让 FA/24h 失控，不符合临床可用性
        grid = np.unique(np.round(np.linspace(0.10, 0.95, 86), 3))
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        m = binary_metrics(y_true, y_prob, t)
        v = m.get(metric, np.nan)
        if v is not None and np.isfinite(v) and v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t, best_v


# ------------------------------------------------------------------ #
# 二、事件（发作）级指标
# ------------------------------------------------------------------ #
def events_from_mask(mask: np.ndarray, sfreq: float) -> np.ndarray:
    """二值序列 -> 事件区间数组 (n,2)，单位秒。"""
    mask = np.asarray(mask).astype(int).ravel()
    if mask.size == 0:
        return np.zeros((0, 2))
    pad = np.concatenate([[0], mask, [0]])
    diff = np.diff(pad)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    ev = np.stack([starts, ends], axis=1).astype(float) / float(sfreq)
    return ev


def merge_events(events: np.ndarray, gap_sec: float = 4.0,
                 min_dur_sec: float = 2.0) -> np.ndarray:
    """合并间隔 < gap 的事件，并丢弃短于 min_dur 的事件。

    向量化实现：先按起点排序，用「间隔 > gap」标记新组，再用 reduceat
    取每组的最小起点与最大终点。与原 Python 逐事件循环结果完全一致，
    但在低阈值（预测事件上千）时快 1~2 个数量级。
    """
    ev = np.asarray(events, dtype=float).reshape(-1, 2)
    if len(ev) == 0:
        return np.zeros((0, 2))
    ev = ev[np.argsort(ev[:, 0], kind="stable")]
    starts, ends = ev[:, 0], ev[:, 1]
    if starts.size > 1:
        new_grp = np.concatenate([[True], (starts[1:] - ends[:-1]) > float(gap_sec)])
        off = np.flatnonzero(new_grp)
        gs = np.minimum.reduceat(starts, off)
        ge = np.maximum.reduceat(ends, off)
    else:
        gs, ge = starts, ends
    keep = (ge - gs) >= float(min_dur_sec)
    out = np.stack([gs[keep], ge[keep]], axis=1)
    return out if out.size else np.zeros((0, 2))


def match_events(pred_events: np.ndarray, true_events: np.ndarray,
                 min_overlap_sec: float = 1.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    将预测事件与真实发作做一对一匹配。
    匹配条件：重叠时长 ≥ min_overlap_sec。采用"重叠最长优先"的贪心匹配。
    返回: (匹配对 [(pred_idx, true_idx)], 未匹配的预测 idx, 未匹配的真实 idx)
    """
    pred_events = np.asarray(pred_events).reshape(-1, 2)
    true_events = np.asarray(true_events).reshape(-1, 2)
    if len(pred_events) == 0 or len(true_events) == 0:
        return [], list(range(len(pred_events))), list(range(len(true_events)))

    # 重叠矩阵：ov[i,j] = max(0, min(pe_i,te_j) - max(ps_i,ts_j))，广播替代双重 Python 循环
    ov = np.maximum(0.0, np.minimum(pred_events[:, 1:2], true_events[None, :, 1])
                    - np.maximum(pred_events[:, 0:1], true_events[None, :, 0]))
    # 只有达到最小重叠的候选才参与贪心排序，避免对全矩阵排序
    ii, jj = np.nonzero(ov >= float(min_overlap_sec))
    pairs, used_p, used_t = [], set(), set()
    if ii.size:
        order = np.argsort(-ov[ii, jj], kind="stable")
        for k in order:
            i, j = int(ii[k]), int(jj[k])
            if i in used_p or j in used_t:
                continue
            pairs.append((i, j))
            used_p.add(i)
            used_t.add(j)
    fp = [i for i in range(len(pred_events)) if i not in used_p]
    fn = [j for j in range(len(true_events)) if j not in used_t]
    return pairs, fp, fn


def evaluate_continuous(y_true_sample: np.ndarray, prob_sample: np.ndarray,
                        sfreq: float, cfg) -> Dict:
    """
    对**一整段连续记录**做事件级评估。
    y_true_sample: (T,) 采样点级真实标签(1=发作)
    prob_sample  : (T,) 采样点级预测概率
    """
    y_true_sample = np.asarray(y_true_sample).astype(int).ravel()
    prob_sample = np.asarray(prob_sample, dtype=float).ravel()
    n = min(len(y_true_sample), len(prob_sample))
    y_true_sample, prob_sample = y_true_sample[:n], prob_sample[:n]

    win = max(1, int(round(cfg.smooth_sec * sfreq)))
    if win > 1:  # 滑动平均平滑，抑制孤立尖峰报警（与 np.convolve(...,'same') 等价）
        prob_s = movavg_same(prob_sample, win)
    else:
        prob_s = prob_sample

    # 阈值：默认绝对阈值；mode="quantile" 时改用「本条记录平滑概率的分位数」，
    # 并取绝对下限兜底——用于抵消跨患者/跨数据库的输出分布漂移。
    if getattr(cfg, "threshold_mode", "absolute") == "quantile" and prob_s.size:
        thr_eff = float(np.quantile(prob_s, float(getattr(cfg, "thr_quantile", 0.995))))
        thr_eff = max(thr_eff, float(getattr(cfg, "thr_floor", 0.0)))
    else:
        thr_eff = float(cfg.threshold)

    alarm = (prob_s >= thr_eff).astype(int)
    pred_events = merge_events(events_from_mask(alarm, sfreq),
                               gap_sec=cfg.merge_gap_sec,
                               min_dur_sec=cfg.min_event_sec)
    true_events = merge_events(events_from_mask(y_true_sample, sfreq),
                               gap_sec=0.0, min_dur_sec=0.0)
    # 真发作不做合并（避免相邻发作被并成一次），但保留原始区间
    true_events = events_from_mask(y_true_sample, sfreq)

    pairs, fp_idx, fn_idx = match_events(pred_events, true_events, cfg.min_overlap_sec)

    delays = []
    overlap_sec, seizure_sec = 0.0, 0.0
    for pi, ti in pairs:
        ps, pe = pred_events[pi]
        ts, te = true_events[ti]
        delays.append(ps - ts)
        overlap_sec += max(0.0, min(pe, te) - max(ps, ts))
    for ts, te in true_events:
        seizure_sec += (te - ts)

    hours = n / sfreq / 3600.0
    res = {
        "n_seizures": int(len(true_events)),
        "n_detected": int(len(pairs)),
        "n_false_alarms": int(len(fp_idx)),
        "hours": float(hours),
        "delay_median_sec": float(np.median(delays)) if delays else np.nan,
        "delay_iqr_sec": float(np.percentile(delays, 75) - np.percentile(delays, 25)) if delays else np.nan,
        "seizure_coverage": float(overlap_sec / seizure_sec) if seizure_sec > 0 else np.nan,
        # 报警事件区间（秒），供可视化与人工复核使用
        "alarms": [(float(a), float(b)) for a, b in pred_events],
        "seizure_events": [(float(a), float(b)) for a, b in true_events],
    }
    # 窗级 / 采样点级指标（补充）
    res.update({("sample_" + k if k in ("auc", "auprc", "sensitivity", "specificity",
                                        "precision", "f1", "accuracy") else k): v
                for k, v in binary_metrics(y_true_sample, prob_sample, cfg.threshold).items()})
    return res


def aggregate_records(records: List[Dict]) -> Dict:
    """把多条记录的结果汇总为"患者级 / 总体"指标。"""
    if not records:
        return {}
    n_seiz = sum(r["n_seizures"] for r in records)
    n_det = sum(r["n_detected"] for r in records)
    hours = sum(r["hours"] for r in records)
    n_fa = sum(r["n_false_alarms"] for r in records)
    aucs = [r.get("sample_auc", np.nan) for r in records]
    aucs = [a for a in aucs if a is not None and np.isfinite(a)]
    return {
        "n_records": len(records),
        "n_seizures": int(n_seiz),
        "n_detected": int(n_det),
        "sensitivity": float(n_det / n_seiz) if n_seiz else np.nan,
        "n_false_alarms": int(n_fa),
        "total_hours": float(hours),
        "fa_per_24h": float(n_fa / hours * 24.0) if hours > 0 else np.nan,
        "sample_auc_mean": float(np.nanmean(aucs)) if aucs else np.nan,
        "delay_median_sec": float(np.nanmedian([r["delay_median_sec"] for r in records])),
        "seizure_coverage": float(np.nanmean([r["seizure_coverage"] for r in records])),
    }


# ------------------------------------------------------------------ #
# 三、统计推断：以"患者"为单位的 Bootstrap / 配对比较
# ------------------------------------------------------------------ #
def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 1000,
                      alpha: float = 0.05, seed: int = 42,
                      stat=np.nanmean) -> Tuple[float, float, float]:
    """患者级聚类 Bootstrap：返回 (点估计, 下限, 上限)。"""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(v)), float(lo), float(hi)


def paired_bootstrap_test(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 2000, seed: int = 42) -> Dict:
    """
    配对 Bootstrap 检验（用于消融实验：全导联 vs 精简导联）。
    返回 差值均值、95%CI、双侧经验 p 值。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # 双侧经验 p 值：以 0 为界
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p))}


def wilcoxon_safe(a: Sequence[float], b: Sequence[float]) -> float:
    """配对 Wilcoxon 符号秩检验；样本不足时返回 NaN。"""
    try:
        from scipy.stats import wilcoxon
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5 or np.allclose(a[m], b[m]):
            return np.nan
        return float(wilcoxon(a[m], b[m]).pvalue)
    except Exception:
        return np.nan
