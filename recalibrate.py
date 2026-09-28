# -*- coding: utf-8 -*-
"""
recalibrate.py —— 事后全局工作点重标定（pooled cross-validation operating point）

背景
----
train.py 中每折的阈值是在**单一验证患者**上标定的。单一患者只有几十小时记录、
个位数发作，标定出的阈值方差极大：实测 Fold 1 在验证患者上 FA/24h=6.81（满足
"≤10"预算），换到测试患者 chb01 上却变成 24.86。这是"小样本标定、大样本泛化"
的失配，不是模型本身的问题。

本脚本的做法
------------
把 24 折的**验证患者预测全部池化**（24 例独立患者的连续记录汇总，约 900 h），
在池化数据上一次性搜索 (概率阈值, 最短报警时长) 的工作点，取
"FA/24h ≤ 预算" 约束下事件级灵敏度最高者；随后把这个**唯一全局工作点**
应用到全部 24 例测试患者上。

优点：
  1. 标定数据从 1 例扩展到 24 例，阈值稳定性大幅提升；
  2. 全程只用验证患者，**不触碰任何测试数据**，无信息泄漏；
  3. 推理结果按"模型_角色_患者"缓存到本地，换预算重新扫描只需秒级。

用法
----
    python recalibrate.py                              # 先推理并缓存，再按默认预算标定
    python recalibrate.py --budgets 3,5,10,15          # 同时输出多个预算的对比
    python recalibrate.py --rescan --budgets 5         # 跳过推理，用缓存重扫（秒级）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                                        # noqa: E402
from evaluate import load_continuous, predict_record             # noqa: E402
from models.seizure_net import build_model                       # noqa: E402
from utils.common import get_logger, select_device, set_seed     # noqa: E402
from utils.metrics import (bootstrap_mean_ci, grid_event_counts,  # noqa: E402
                           movavg_same)

FOLD_RE = re.compile(
    r"=====\s*Fold\s+(\d+)/(\d+)\s+测试患者=(\S+)\s+验证患者=(\S+)\s*=====")

DEFAULT_WORK = Path(r"E:\临时存储夹\人工智能（AI）与癫痫\ai_epilepsy_project\work")


# ------------------------------------------------------------------ #
def _check_movavg():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    for w in (1, 3, 8, 1024):
        a = np.convolve(x, np.ones(w) / w, mode="same")
        b = movavg_same(x, w)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-10), f"w={w} 不一致"
    print("movavg_same 与 np.convolve(mode='same') 一致 ✓")


# ------------------------------------------------------------------ #
def parse_folds(log_path: Path):
    """从 train.log 解析 (折号, 测试患者, 验证患者)。

    ★ 陷阱：train.log 是"活"文件 —— ablation.py 等脚本会以同一 work_dir 追加
    写入，导致重标定把消融的 6 epoch 折也当成主实验折（曾解析出 25 折）。
    因此只取**首次出现 "全部完成" 之前**的内容，并在解析后按折号去重。
    更稳妥的做法是优先使用冻结快照 work/folds_lopo.json（见 load_folds）。
    """
    out, seen = [], set()
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "全部完成" in line:
            break
        m = FOLD_RE.search(line)
        if m:
            idx = int(m.group(1))
            if idx in seen:          # 重跑/追加时同号折只取第一次
                continue
            seen.add(idx)
            out.append((idx, m.group(3), m.group(4)))
    return out


FOLDS_SNAPSHOT = Path(__file__).resolve().parent / "work" / "folds_lopo.json"


def load_folds(log_path: Path, logger=None, snapshot=None):
    """优先用冻结快照；没有快照才回退到解析 train.log。"""
    snap = snapshot or FOLDS_SNAPSHOT
    if snap.exists():
        try:
            items = json.loads(snap.read_text(encoding="utf-8"))
            out = [(int(it["idx"]), it["test"], it["val"]) for it in items]
            if logger:
                logger.info(f"使用冻结折快照 {snap.name}（{len(out)} 折）")
            return out
        except Exception as e:                                  # noqa: BLE001
            if logger:
                logger.warning(f"折快照读取失败（{e}），回退解析 train.log")
    return parse_folds(log_path)


def ensure_cache(model, entries, acfg, device, channel_idx, cache_path: Path,
                 role: str, logger):
    """推理一次并按 npz 缓存；已存在则直接复用。不返回数组，避免占内存。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return True
    payload, n = {}, 0
    for e in entries:
        loaded = load_continuous(e, acfg, logger)
        if loaded is None:
            continue
        X, y_sample, sfreq, _ = loaded
        prob, _, _ = predict_record(model, X, acfg, device, channel_idx,
                                    batch_size=256)
        if prob.size == 0:
            continue
        payload[f"{n}_y"] = y_sample.astype(np.uint8)
        payload[f"{n}_p"] = prob.astype(np.float32)
        payload[f"{n}_fs"] = np.array([float(sfreq)])
        n += 1
    if n == 0:
        return False
    np.savez(cache_path, **payload)
    logger.info(f"    缓存 {role}：{n} 条 → {cache_path.name}")
    return True


def iter_cache(cache_path: Path):
    """逐条读取缓存，返回 (y, prob, sfreq) 生成器，避免一次性载入整位患者。"""
    d = np.load(cache_path)
    i = 0
    while f"{i}_p" in d:
        yield d[f"{i}_y"], d[f"{i}_p"], float(d[f"{i}_fs"][0])
        i += 1
    d.close()


def accumulate(recs_iter, cfg, grid_thr, grid_md, acc):
    """把一组记录的事件级计数按网格累加到 acc[n_thr, n_md, 4]。

    逐条记录调用 grid_event_counts（向量化合并 + 广播匹配），
    整张阈值×时长网格一次算完，比逐格调用快约 25 倍。
    """
    for y, p, sf in recs_iter:
        ps = movavg_same(p, max(1, int(round(cfg.smooth_sec * sf))))
        g = grid_event_counts(y, ps, sf, grid_thr, grid_md,
                              cfg.merge_gap_sec, cfg.min_overlap_sec)
        acc += g
    return acc


# ------------------------------------------------------------------ #
def accumulate_quantile(recs_iter, cfg, qs, floors, grid_md, acc):
    """逐记录分位阈值扫描：阈值 = max(该记录平滑概率的 q 分位, floor)。

    设计动机：跨患者场景下模型输出存在显著的患者级（乃至记录级）分布漂移
    ——实测部分患者的发作间期概率中位数可达 0.96，而另一些仅 0.02。
    固定绝对阈值无法同时适配这两类记录；改用**记录内秩（分位）**定阈，
    相当于比较"该记录内部相对异常程度"，对漂移天然免疫。
    floor 用于给"整条记录都很干净"的情况兜底，避免在极低概率上触发报警。

    acc 形状 (n_q * n_floor, n_md, 4)；同一 (q, floor) 下的所有 min_dur
    在一次 grid_event_counts 调用内复用，故每条记录只需 n_q*n_floor 次调用。
    """
    n_f = len(floors)
    for y, p, sf in recs_iter:
        ps = movavg_same(p, max(1, int(round(cfg.smooth_sec * sf))))
        if ps.size == 0:
            continue
        for qi, q in enumerate(qs):
            thr_q = float(np.quantile(ps, q))
            if not np.isfinite(thr_q):
                continue
            for fi, fl in enumerate(floors):
                thr = max(thr_q, float(fl))
                # grid_event_counts 返回 (1, n_md, 4)，取第 0 维对齐 acc 的 (n_md, 4)
                acc[qi * n_f + fi] += grid_event_counts(
                    y, ps, sf, [thr], grid_md,
                    cfg.merge_gap_sec, cfg.min_overlap_sec)[0]
    return acc


def pick_index(acc, budget):
    """在 acc[:, :, :] 上按「FA ≤ 预算下灵敏度最高；同灵敏度取 FA 最低」选点。"""
    tot_true = acc[:, :, 0]
    hrs = np.maximum(acc[:, :, 3], 1e-9)
    sens = np.where(tot_true > 0, acc[:, :, 1] / np.maximum(tot_true, 1), 0.0)
    fa = np.where(acc[:, :, 3] > 0, acc[:, :, 2] / hrs * 24.0, 0.0)
    feas = fa <= budget + 1e-12
    if feas.any():
        top = round(float(sens[feas].max()), 6)
        same = [(i, j) for i in range(sens.shape[0]) for j in range(sens.shape[1])
                if feas[i, j] and round(sens[i, j], 6) >= top - 1e-9]
        i, j = min(same, key=lambda ij: (fa[ij], -ij[0]))
        note = "可行"
    else:
        i, j = min(((i, j) for i in range(sens.shape[0]) for j in range(sens.shape[1])),
                   key=lambda ij: (fa[ij], -round(sens[ij], 6)))
        note = "无工作点满足预算，取 FA 最低者"
    return (int(i), int(j)), float(sens[i, j]), float(fa[i, j]), note


def pick_point(acc_val, grid_thr, grid_md, budget):
    """在池化验证集上选工作点：FA≤预算下灵敏度最高；满灵敏度时取 FA 最低者。"""
    tot_true = acc_val[:, :, 0]
    sens = np.where(tot_true > 0, acc_val[:, :, 1] / np.maximum(tot_true, 1), 0.0)
    fa = np.where(acc_val[:, :, 3] > 0,
                  acc_val[:, :, 2] / np.maximum(acc_val[:, :, 3], 1e-9) * 24.0, 0.0)
    feas = fa <= budget + 1e-12
    rows = [{"threshold": float(grid_thr[i]), "min_event_sec": float(grid_md[j]),
             "val_sensitivity": float(sens[i, j]), "val_fa_per_24h": float(fa[i, j]),
             "val_n_seizures": int(acc_val[i, j, 0]),
             "val_n_detected": int(acc_val[i, j, 1]),
             "val_n_false_alarms": int(acc_val[i, j, 2]),
             "val_hours": float(acc_val[i, j, 3])}
            for i in range(sens.shape[0]) for j in range(sens.shape[1])]
    if feas.any():
        cand = [(round(sens[i, j], 6), -fa[i, j]) for i, j in
                zip(*np.where(feas))]
        best = max(zip(*np.where(feas)), key=lambda ij: (round(sens[ij], 6),
                                                         -fa[ij]))
        # 灵敏度已达上界时，改为在同等灵敏度里取 FA 最低者，给临床留最大余量
        top = round(sens[best], 6)
        same = [(i, j) for i in range(sens.shape[0]) for j in range(sens.shape[1])
                if feas[i, j] and round(sens[i, j], 6) >= top - 1e-9]
        best = min(same, key=lambda ij: (fa[ij], -ij[0]))
        note = "可行"
    else:
        best = min(((i, j) for i in range(sens.shape[0])
                    for j in range(sens.shape[1])),
                   key=lambda ij: (fa[ij], -round(sens[ij], 6)))
        note = "无工作点满足预算，取 FA 最低者"
    i, j = best
    return {"threshold": float(grid_thr[i]), "min_event_sec": float(grid_md[j]),
            "val_sensitivity": float(sens[i, j]), "val_fa_per_24h": float(fa[i, j]),
            "note": note}, rows


def table_from_acc(acc, ti, di, patient=None):
    row = dict(n_seizures=int(acc[ti, di, 0]), n_detected=int(acc[ti, di, 1]),
               n_false_alarms=int(acc[ti, di, 2]), hours=float(acc[ti, di, 3]))
    row["sensitivity"] = row["n_detected"] / row["n_seizures"] if row["n_seizures"] else np.nan
    row["fa_per_24h"] = row["n_false_alarms"] / row["hours"] * 24 if row["hours"] else np.nan
    if patient:
        row["patient"] = patient
    return row


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chbmit")
    ap.add_argument("--channel_set", default="full18")
    ap.add_argument("--budgets", default="10", help="验证池化 FA/24h 上限，逗号分隔")
    ap.add_argument("--work_dir", default=str(DEFAULT_WORK))
    ap.add_argument("--device", default=None)
    ap.add_argument("--rescan", action="store_true", help="跳过推理，仅用缓存重扫")
    ap.add_argument("--min_durs", default="4,6,8,10,15,20")
    ap.add_argument("--quantiles", default="0.98,0.99,0.995",
                    help="逐记录分位阈值的分位点（策略 B）")
    ap.add_argument("--floors", default="0.0,0.5,0.8",
                    help="分位阈值的绝对下限（策略 B），防止干净记录误报")
    ap.add_argument("--selftest", action="store_true", help="仅校验快速平滑实现后退出")
    ap.add_argument("--train_log", default=None, help="默认 <work_dir>/train.log")
    ap.add_argument("--cache_dir", default=None, help="默认 <work_dir>/recalib_cache")
    ap.add_argument("--limit_folds", type=int, default=0, help=">0 时只取前 N 折（冒烟测试用）")
    ap.add_argument("--folds_json", default=None,
                    help="折快照路径（默认 work/folds_lopo.json）")
    ap.add_argument("--result_dir", default=None,
                    help="结果输出目录（默认 work/results）")
    ap.add_argument("--net", default="seizurenet",
                    help="网络类型，用于匹配 ckpt 文件名")
    ap.add_argument("--merge_json", default=None,
                    help="受试者合并映射 JSON（如 {\"chb01\":[\"chb21\"]}），"
                         "使合并折的测试/验证记录包含同源记录")
    args = ap.parse_args()

    if args.selftest:
        _check_movavg()
        return

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    logger = get_logger("recalib", "INFO", work / "recalibrate.log")
    set_seed(0)
    device = select_device(args.device)

    cfg = Config()
    cfg.work_root = str(work)
    cfg.dataset = args.dataset

    from data.montage import CHANNEL_SETS
    channel_idx = CHANNEL_SETS[args.channel_set]

    # 受试者合并映射（同源记录归入同一受试者做测试）
    merge_map = {}
    if args.merge_json and Path(args.merge_json).exists():
        merge_map = json.loads(Path(args.merge_json).read_text(encoding="utf-8"))
    # 双向展开：chb21->chb01 且 chb01->chb21
    _expand = {}
    for k, vs in merge_map.items():
        grp = [k] + list(vs)
        for g in grp:
            _expand[g] = grp
    def expand_subject(p):
        return _expand.get(p, [p])

    manifest = json.loads((work / "index" / args.dataset / "manifest.json")
                          .read_text(encoding="utf-8"))
    by_patient = {}
    for m in manifest:
        if m.get("ok"):
            by_patient.setdefault(m["patient"], []).append(m)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (work / "recalib_cache")
    folds = load_folds(Path(args.train_log) if args.train_log
                       else (work / "train.log"), logger,
                       snapshot=(Path(args.folds_json) if args.folds_json else None))
    if args.limit_folds > 0:
        folds = folds[:args.limit_folds]
    if not folds:
        raise SystemExit("未能解析出折，确认 LOPO 已跑过至少一折")
    logger.info(f"解析到 {len(folds)} 折｜设备 {device}｜重扫={args.rescan}")
    usable = []

    for idx, te_p, val_p in folds:
        ck = work / "ckpt" / f"{args.dataset}_{args.channel_set}_{args.net}_{te_p}.pt"
        if not ck.exists():
            logger.warning(f"  Fold {idx}: {te_p} 权重缺失，跳过")
            continue
        cv = cache_dir / f"{te_p}__val_{val_p}.npz"
        ct = cache_dir / f"{te_p}__test_{te_p}.npz"
        if args.rescan:
            if cv.exists() and ct.exists():
                usable.append((idx, te_p, val_p))
            continue
        if cv.exists() and ct.exists():
            usable.append((idx, te_p, val_p))
            continue

        data = torch.load(ck, map_location=device, weights_only=False)
        acfg = Config()
        acfg.__dict__.update(data["cfg"])
        acfg.n_channels = len(data.get("channel_idx", channel_idx))
        model = build_model(acfg).to(device)
        model.load_state_dict(data["state_dict"])
        model.eval()

        logger.info(f"  Fold {idx:>2}/{len(folds)}  模型={te_p}  验证={val_p}")
        test_recs = [m for q in expand_subject(te_p) for m in by_patient.get(q, [])]
        val_recs = [m for q in expand_subject(val_p) for m in by_patient.get(q, [])]
        ok_v = ensure_cache(model, val_recs, acfg, device,
                            channel_idx, cv, f"val:{val_p}", logger)
        ok_t = ensure_cache(model, test_recs, acfg, device,
                            channel_idx, ct, f"test:{te_p}", logger)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        if ok_v and ok_t:
            usable.append((idx, te_p, val_p))

    logger.info(f"可用折数 {len(usable)}/{len(folds)}；开始网格累加")

    grid_thr = np.unique(np.round(np.concatenate([
        np.linspace(0.05, 0.90, 35),
        np.linspace(0.91, 0.98, 15),
        np.array([0.985, 0.99, 0.993, 0.995, 0.997, 0.999]),
    ]), 4))
    grid_md = [float(x) for x in args.min_durs.split(",")]
    nT, nD = len(grid_thr), len(grid_md)

    acc_val = np.zeros((nT, nD, 4))
    acc_te = {}
    for idx, te_p, val_p in usable:
        acc_val = accumulate(iter_cache(cache_dir / f"{te_p}__val_{val_p}.npz"),
                             cfg, grid_thr, grid_md, acc_val)
        acc_te[te_p] = accumulate(iter_cache(cache_dir / f"{te_p}__test_{te_p}.npz"),
                                  cfg, grid_thr, grid_md,
                                  np.zeros((nT, nD, 4)))

    # ---- 策略 B：逐记录分位阈值 ----
    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]
    floors = [float(x) for x in args.floors.split(",") if x.strip()]
    params = [(q, fl) for q in qs for fl in floors]
    nP = len(params)
    q_val = np.zeros((nP, nD, 4))
    q_te = {}
    for idx, te_p, val_p in usable:
        q_val = accumulate_quantile(iter_cache(cache_dir / f"{te_p}__val_{val_p}.npz"),
                                    cfg, qs, floors, grid_md, q_val)
        q_te[te_p] = accumulate_quantile(
            iter_cache(cache_dir / f"{te_p}__test_{te_p}.npz"),
            cfg, qs, floors, grid_md, np.zeros((nP, nD, 4)))
    logger.info("策略 B（分位阈值）网格累加完成")

    val_hours = float(acc_val[0, 0, 3])
    val_seiz = int(acc_val[0, 0, 0])
    logger.info(f"验证池 {val_hours:.1f} h / {val_seiz} 次发作；测试 {len(acc_te)} 例")

    outdir = Path(args.result_dir) if args.result_dir else (work / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    rows_summary, print_rows = [], []

    def eval_on_test(acc_dict, ti, di, budget, strategy, thr_label):
        """给定工作点索引，在测试患者上汇总指标（测试集只在此处被读取一次）。"""
        per = [table_from_acc(acc_dict[p], ti, di, p) for p in sorted(acc_dict)]
        df = pd.DataFrame(per)
        pooled_sens = df.n_detected.sum() / max(1, df.n_seizures.sum())
        pooled_fa = df.n_false_alarms.sum() / df.hours.sum() * 24
        # bootstrap_mean_ci 返回 (点估计, 下限, 上限)；CI 字段要取 [1],[2]
        _, s_lo, s_hi = bootstrap_mean_ci(df.sensitivity.dropna().values, 2000)
        _, f_lo, f_hi = bootstrap_mean_ci(df.fa_per_24h.dropna().values, 2000)
        return df, {
            "strategy": strategy, "budget_fa_per_24h": budget,
            "threshold": thr_label, "min_event_sec": float(grid_md[di]),
            "smooth_sec": cfg.smooth_sec, "merge_gap_sec": cfg.merge_gap_sec,
            "min_overlap_sec": cfg.min_overlap_sec,
            "n_patients": int(len(df)),
            "total_seizures": int(df.n_seizures.sum()),
            "total_detected": int(df.n_detected.sum()),
            "pooled_sensitivity": float(pooled_sens),
            "sensitivity_mean": float(df.sensitivity.mean()),
            "sensitivity_median": float(df.sensitivity.median()),
            "sensitivity_ci95": [float(s_lo), float(s_hi)],
            "total_false_alarms": int(df.n_false_alarms.sum()),
            "total_hours": float(df.hours.sum()),
            "pooled_fa_per_24h": float(pooled_fa),
            "fa_mean": float(df.fa_per_24h.mean()),
            "fa_median": float(df.fa_per_24h.median()),
            "fa_ci95": [float(f_lo), float(f_hi)],
        }

    for budget in [float(b) for b in args.budgets.split(",") if b.strip()]:
        # --- 策略 A：全局固定阈值 ---
        point, rows = pick_point(acc_val, grid_thr, grid_md, budget)
        ti = int(np.argmin(np.abs(grid_thr - point["threshold"])))
        di = grid_md.index(point["min_event_sec"])
        df, summ = eval_on_test(acc_te, ti, di, budget, "A 全局阈值",
                                round(point["threshold"], 4))
        summ.update({"note": point["note"], "val_hours": val_hours,
                     "val_n_seizures": val_seiz,
                     "val_sensitivity": point["val_sensitivity"],
                     "val_fa_per_24h": point["val_fa_per_24h"]})
        rows_summary.append(summ)
        tag = f"_b{str(budget).replace('.', 'p')}"
        df.to_csv(outdir / f"recalibA_per_patient{tag}.csv", index=False,
                  encoding="utf-8-sig")
        pd.DataFrame(rows).to_csv(outdir / f"recalibA_grid_validation{tag}.csv",
                                  index=False, encoding="utf-8-sig")

        # --- 策略 B：逐记录分位阈值 ---
        (pi, bi), v_sens, v_fa, note_b = pick_index(q_val, budget)
        q, fl = params[pi]
        df_b, summ_b = eval_on_test(q_te, pi, bi, budget, "B 分位阈值",
                                    f"q={q}/floor={fl}")
        summ_b.update({"note": note_b, "val_hours": val_hours,
                       "val_n_seizures": val_seiz,
                       "val_sensitivity": v_sens, "val_fa_per_24h": v_fa})
        rows_summary.append(summ_b)
        df_b.to_csv(outdir / f"recalibB_per_patient{tag}.csv", index=False,
                    encoding="utf-8-sig")
        pd.DataFrame([{"quantile": q, "floor": fl,
                       "min_event_sec": float(grid_md[bi]),
                       "val_sensitivity": v_sens, "val_fa_per_24h": v_fa}
                      for (q, fl) in params]).to_csv(
            outdir / f"recalibB_grid_validation{tag}.csv", index=False,
            encoding="utf-8-sig")

        print_rows.append((budget, summ))
        print_rows.append((budget, summ_b))
        for s in (summ, summ_b):
            (outdir / f"recalib_summary_{s['strategy'][0]}_b"
                      f"{str(budget).replace('.', 'p')}.json").write_text(
                json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(rows_summary).to_csv(outdir / "recalib_budget_comparison.csv",
                                      index=False, encoding="utf-8-sig")

    print("\n=== 工作点重标定（阈值仅在验证患者上标定，未触碰测试数据） ===")
    print(f"验证池：{val_hours:.1f} h / {val_seiz} 次发作 / {len(usable)} 折")
    print(f"{'策略':<10} {'预算':>5} {'工作点':>22} {'最短报警':>7} "
          f"{'验证灵敏度':>9} {'验证FA':>7} | {'测试灵敏度':>9} {'测试FA':>8}")
    for b, s in print_rows:
        thr = (f"{s['threshold']:.3f}" if isinstance(s["threshold"], float)
               else str(s["threshold"]))
        print(f"{s['strategy']:<10} {b:>5.1f} {thr:>22} "
              f"{s['min_event_sec']:>6g}s {s['val_sensitivity']:>9.3f} "
              f"{s['val_fa_per_24h']:>7.2f} | {s['pooled_sensitivity']:>9.3f} "
              f"{s['pooled_fa_per_24h']:>8.2f}")
    logger.info(f"结果已写入 {outdir}")


if __name__ == "__main__":
    main()
