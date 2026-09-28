# -*- coding: utf-8 -*-
"""
compare_baselines.py —— 跨患者基线对比 + 逐例精确率/F1 汇总

读取多个"逐例"CSV（含 n_seizures, n_detected, n_false_alarms 列），统一计算：
  患者均值灵敏度(95%CI)、汇总灵敏度、患者均值 FA/24h、汇总 FA/24h、
  患者均值精确率、汇总精确率、患者均值 F1、汇总 F1。
逐例精确率 = n_detected/(n_detected+n_false_alarms)；F1 = 2PR/(P+R)。

用法：
  python compare_baselines.py \
    --files work/results/lopo_chbmit_full18_23subj.csv:深度SeizureNet \
            work/results/recalibB_per_patient_b10p0.csv:深度SeizureNet(策略B) \
            work/results/lopo_chbmit_full18_trad23subj.csv:传统特征RF \
            work/results/lopo_chbmit_full18_resnet23subj.csv:同量级网络ResNet1D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.metrics import bootstrap_mean_ci  # noqa: E402


def summarize(path: str, label: str, n_boot: int = 2000) -> dict:
    df = pd.read_csv(path)
    df = df.dropna(subset=["sensitivity"])
    df = df[df.n_seizures > 0]  # 仅统计含发作受试者
    df = df.copy()
    if "total_hours" not in df.columns and "hours" in df.columns:
        df["total_hours"] = df["hours"]
    df["precision"] = df.apply(
        lambda r: r.n_detected / (r.n_detected + r.n_false_alarms)
        if (r.n_detected + r.n_false_alarms) > 0 else np.nan, axis=1)
    df["f1"] = df.apply(
        lambda r: 2 * r.precision * r.sensitivity / (r.precision + r.sensitivity)
        if np.isfinite(r.precision) and (r.precision + r.sensitivity) > 0 else np.nan,
        axis=1)

    tot_seiz = int(df.n_seizures.sum()); tot_det = int(df.n_detected.sum())
    tot_fa = int(df.n_false_alarms.sum()); tot_h = float(df.total_hours.sum())
    s_pt, s_lo, s_hi = bootstrap_mean_ci(df.sensitivity.values, n_boot, seed=42)
    f_pt, f_lo, f_hi = bootstrap_mean_ci(df.fa_per_24h.dropna().values, n_boot, seed=42)
    return {
        "method": label,
        "n_subjects": int(len(df)),
        "total_seizures": tot_seiz, "total_detected": tot_det,
        "pooled_sensitivity": tot_det / max(tot_seiz, 1),
        "sens_mean": s_pt, "sens_ci95": f"[{s_lo:.3f},{s_hi:.3f}]",
        "pooled_fa": tot_fa / tot_h * 24 if tot_h else np.nan,
        "fa_mean": f_pt, "fa_ci95": f"[{f_lo:.2f},{f_hi:.2f}]",
        "pooled_precision": tot_det / (tot_det + tot_fa) if (tot_det + tot_fa) else np.nan,
        "precision_mean": float(df.precision.mean()),
        "pooled_f1": (2 * (tot_det / (tot_det + tot_fa)) * (tot_det / max(tot_seiz, 1))
                      / ((tot_det / (tot_det + tot_fa)) + (tot_det / max(tot_seiz, 1))))
                      if (tot_det + tot_fa) and tot_seiz else np.nan,
        "f1_mean": float(df.f1.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True,
                    help="path:label 逗号分隔，可多个用分号分隔")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    rows = []
    for item in args.files.split(";"):
        p, lab = item.split(":", 1)
        rows.append(summarize(p.strip(), lab.strip(), args.n_boot))

    out = pd.DataFrame(rows)[
        ["method", "n_subjects", "total_seizures", "total_detected",
         "pooled_sensitivity", "sens_mean", "sens_ci95",
         "pooled_fa", "fa_mean", "fa_ci95",
         "pooled_precision", "precision_mean", "pooled_f1", "f1_mean"]]
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(out.to_string(index=False))
    out.to_csv("work/results/baseline_comparison.csv", index=False,
               encoding="utf-8-sig")
    print("\n已写出 work/results/baseline_comparison.csv")


if __name__ == "__main__":
    main()
