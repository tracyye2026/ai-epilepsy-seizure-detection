# -*- coding: utf-8 -*-
"""
eval_trad_recalib.py —— 从 RF 分数缓存免推理计算策略 B 结果（表 8 第 5 行）

前置：run_trad_scores.py 已生成 work/trad_scores/{patient}_{test,val}.npz。
输出：
  1. 策略 B @ 与 SeizureNet 主终点相同工作点 (q=0.99, floor=0.0, min_event=15 s)：
     逐例 + 汇总（灵敏度 / FA/24h / 精确率 / F1），直接可入表 8；
  2. 策略 B 网格敏感性简表（q × floor × min_event 的池化汇总）；
  3. 窗级 AUC（2 s 步长子采样，逐折均值 + 池化）。
用法：python eval_trad_recalib.py [--cache_dir work/trad_scores]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                                            # noqa: E402
from recalibrate import (accumulate_quantile, iter_cache,            # noqa: E402
                         table_from_acc)
from utils.metrics import bootstrap_mean_ci                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--work_dir", default=r"E:\临时存储夹\人工智能（AI）与癫痫\ai_epilepsy_project\work")
    ap.add_argument("--quantiles", default="0.98,0.99,0.995")
    ap.add_argument("--floors", default="0.0,0.5,0.8")
    ap.add_argument("--min_durs", default="4,6,8,10,15,20")
    ap.add_argument("--primary", default="0.99,0.0,15",
                    help="主工作点 q,floor,md（与 SeizureNet 策略 B 主终点一致）")
    args = ap.parse_args()

    work = Path(args.work_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (work / "trad_scores")
    tests = sorted(cache_dir.glob("*_test.npz"))
    if not tests:
        raise SystemExit(f"未找到缓存：{cache_dir}")
    patients = [f.name.replace("_test.npz", "") for f in tests]
    print(f"测试患者 {len(patients)}：{', '.join(patients)}")

    cfg = Config()
    qs = [float(x) for x in args.quantiles.split(",")]
    floors = [float(x) for x in args.floors.split(",")]
    mds = [float(x) for x in args.min_durs.split(",")]
    q0, f0, md0 = (float(x) for x in args.primary.split(","))
    nP, nD = len(qs) * len(floors), len(mds)
    qi0 = qs.index(q0) * len(floors) + floors.index(f0)
    di0 = mds.index(md0)

    # ---- 策略 B：逐折网格累加 ----
    per_rows, pooled_acc = [], np.zeros((nP, nD, 4))
    y_all, p_all = [], []          # 窗级 AUC（2 s 子采样）
    for p in patients:
        acc = accumulate_quantile(iter_cache(cache_dir / f"{p}_test.npz"),
                                  cfg, qs, floors, mds, np.zeros((nP, nD, 4)))
        pooled_acc += acc
        per_rows.append(table_from_acc(acc, qi0, di0, patient=p))

        d = np.load(cache_dir / f"{p}_test.npz")
        fy, fp = [], []
        k = 0
        while f"{k}_p" in d:
            y = d[f"{k}_y"].astype(int)
            prob = d[f"{k}_p"].astype(float)
            sf = float(d[f"{k}_fs"][0])
            step = max(1, int(round(2.0 * sf)))
            off = step // 2
            fy.append(y[off::step])
            fp.append(prob[off::step])
            k += 1
        d.close()
        y_all.append(np.concatenate(fy))
        p_all.append(np.concatenate(fp))

    df = pd.DataFrame(per_rows)
    df["precision"] = df.apply(
        lambda r: r.n_detected / (r.n_detected + r.n_false_alarms)
        if (r.n_detected + r.n_false_alarms) > 0 else np.nan, axis=1)
    df["f1"] = df.apply(
        lambda r: 2 * r.precision * r.sensitivity / (r.precision + r.sensitivity)
        if np.isfinite(r.precision) and (r.precision + r.sensitivity) > 0
        else np.nan, axis=1)

    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    from sklearn.metrics import average_precision_score, roc_auc_score
    fold_aucs = [roc_auc_score(yy, pp) for yy, pp in zip(y_all, p_all)
                 if 0 < yy.sum() < len(yy)]

    outdir = work / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "recalibB_trad_per_patient.csv",
              index=False, encoding="utf-8-sig")

    # ---- 汇总（主工作点） ----
    row = table_from_acc(pooled_acc, qi0, di0)
    s_pt, s_lo, s_hi = bootstrap_mean_ci(df.sensitivity.dropna().values, 1000, seed=0)
    f_pt, f_lo, f_hi = bootstrap_mean_ci(df.fa_per_24h.dropna().values, 1000, seed=0)
    prec_mean = df.precision.mean()
    f1_mean = df.f1.mean()
    print("\n===== RF + 策略 B（q=0.99, floor=0.0, min_event=15 s） =====")
    print(f"逐例灵敏度 {s_pt:.3f} (95%CI {s_lo:.3f}~{s_hi:.3f})｜中位 {df.sensitivity.median():.3f}")
    print(f"汇总灵敏度 {row['n_detected']}/{row['n_seizures']} = "
          f"{row['sensitivity']:.3f}")
    print(f"逐例 FA/24h {f_pt:.2f} (95%CI {f_lo:.2f}~{f_hi:.2f})｜中位 {df.fa_per_24h.median():.2f}")
    print(f"汇总 FA/24h {row['n_false_alarms']}/{row['hours']:.1f} h × 24 = "
          f"{row['fa_per_24h']:.2f}")
    print(f"逐例精确率均值 {prec_mean:.3f}｜F1 均值 {f1_mean:.3f}")
    print(f"窗级 AUC：逐折均值 {np.mean(fold_aucs):.4f}（最小 {np.min(fold_aucs):.3f}）"
          f"｜池化 {roc_auc_score(y, p):.4f}"
          f"｜AUPRC {average_precision_score(y, p):.4f}")

    # ---- 网格敏感性简表 ----
    grid_rows = []
    for pi, (q, fl) in enumerate([(q, fl) for q in qs for fl in floors]):
        for dj, md in enumerate(mds):
            r = table_from_acc(pooled_acc, pi, dj)
            grid_rows.append({"q": q, "floor": fl, "min_event_sec": md,
                              "sensitivity": round(r["sensitivity"], 3),
                              "fa_per_24h": round(r["fa_per_24h"], 2),
                              "n_detected": r["n_detected"],
                              "n_seizures": r["n_seizures"]})
    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(outdir / "recalibB_trad_grid.csv", index=False, encoding="utf-8-sig")
    print("\n网格敏感性（池化）已存 recalibB_trad_grid.csv；主点邻近预览：")
    print(gdf[(gdf.q == q0)].to_string(index=False))

    summary = {
        "strategy": "B", "q": q0, "floor": f0, "min_event_sec": md0,
        "n_patients": int(len(df)),
        "sens_mean": round(float(s_pt), 4), "sens_ci": [round(float(s_lo), 4), round(float(s_hi), 4)],
        "sens_median": round(float(df.sensitivity.median()), 4),
        "fa_mean": round(float(f_pt), 4), "fa_ci": [round(float(f_lo), 4), round(float(f_hi), 4)],
        "fa_median": round(float(df.fa_per_24h.median()), 4),
        "pooled_sensitivity": round(float(row["sensitivity"]), 4),
        "pooled_n_detected": int(row["n_detected"]),
        "pooled_n_seizures": int(row["n_seizures"]),
        "pooled_fa_per_24h": round(float(row["fa_per_24h"]), 4),
        "pooled_hours": round(float(row["hours"]), 1),
        "precision_mean": round(float(prec_mean), 4),
        "f1_mean": round(float(f1_mean), 4),
        "window_auc_foldmean": round(float(np.mean(fold_aucs)), 4),
        "window_auc_pooled": round(float(roc_auc_score(y, p)), 4),
        "window_auprc_pooled": round(float(average_precision_score(y, p)), 4),
    }
    (outdir / "recalibB_trad_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果：{outdir / 'recalibB_trad_per_patient.csv'} 与 recalibB_trad_summary.json")


if __name__ == "__main__":
    main()
