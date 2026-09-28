#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finalize_primary.py — 由重标定产物确定"主要终点"，并修正预算对比表里的 CI 列

背景（踩过的坑）：
  recalibrate.py 13:19 版曾把 bootstrap_mean_ci 的返回值当成 (lo, hi)，
  实际函数返回 (点估计, 下限, 上限) —— 导致 recalib_budget_comparison.csv 中
  sensitivity_ci95 / fa_ci95 存成了 [均值, 下限]。脚本已在 13:41 修正，
  但重标定本身要跑 70+ 分钟，不值得为 CI 重跑。
  逐例 CSV（recalib?_per_patient_b*.csv）是原始逐患者数字、未受影响，
  因此本脚本直接由逐例 CSV 重算所有汇总量与 CI，产出"权威版"主要终点。

输出：
  work/results/primary_endpoint.json            主要终点（冻结工作点，策略B）
  work/results/primary_endpoint_per_patient.csv 主要终点逐例
  work/results/recalib_budget_comparison.csv    就地修正 CI 两列
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.metrics import bootstrap_mean_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "work" / "results"


def summarize(df: pd.DataFrame) -> dict:
    """由逐例表计算汇总指标，CI 一律取 bootstrap_mean_ci 的 [1],[2]。"""
    df = df.dropna(subset=["sensitivity"])
    n_seiz = int(df.n_seizures.sum())
    n_det = int(df.n_detected.sum())
    hours = float(df.hours.sum())
    n_fa = int(df.n_false_alarms.sum())
    _, s_lo, s_hi = bootstrap_mean_ci(df.sensitivity.values, 2000)
    _, f_lo, f_hi = bootstrap_mean_ci(df.fa_per_24h.dropna().values, 2000)
    return {
        "n_patients": int(len(df)),
        "total_seizures": n_seiz,
        "total_detected": n_det,
        "pooled_sensitivity": n_det / max(n_seiz, 1),
        "sensitivity_mean": float(df.sensitivity.mean()),
        "sensitivity_median": float(df.sensitivity.median()),
        "sensitivity_ci95": [float(s_lo), float(s_hi)],
        "total_false_alarms": n_fa,
        "total_hours": hours,
        "pooled_fa_per_24h": n_fa / max(hours, 1e-9) * 24,
        "fa_mean": float(df.fa_per_24h.mean()),
        "fa_median": float(df.fa_per_24h.median()),
        "fa_ci95": [float(f_lo), float(f_hi)],
        "patients_sensitivity_ge50pct": float((df.sensitivity >= 0.5).mean()),
    }


def main() -> int:
    op = RES / "operating_point.json"
    if not op.exists():
        print("缺少 operating_point.json，请先运行 pick_operating_point.py")
        return 1
    opj = json.loads(op.read_text(encoding="utf-8"))
    budget = float(opj.get("budget", 10.0))
    tag = f"_b{str(budget).replace('.', 'p')}"

    # ---- 1) 主要终点（策略 B） ----
    per_b = RES / f"recalibB_per_patient{tag}.csv"
    per_a = RES / f"recalibA_per_patient{tag}.csv"
    if not per_b.exists():
        print(f"缺少 {per_b}")
        return 1
    df_b = pd.read_csv(per_b)
    df_a = pd.read_csv(per_a) if per_a.exists() else None

    primary = {
        "endpoint_definition": "主要终点 = 事件级灵敏度 与 假报警率(FA/24h)",
        "strategy": opj.get("strategy", "B 分位阈值"),
        "threshold_mode": opj.get("threshold_mode"),
        "thr_quantile": opj.get("thr_quantile"),
        "thr_floor": opj.get("thr_floor"),
        "min_event_sec": opj.get("min_event_sec"),
        "budget_fa_per_24h": budget,
        "calibrated_on": "验证患者池（阈值与工作点未使用任何测试患者数据）",
        **summarize(df_b),
    }
    if df_a is not None:
        primary["control_A_global_threshold"] = summarize(df_a)

    # 窗级 AUC / 延迟：阈值无关或来自主 LOPO 运行，标注来源
    sm = RES / "summary_chbmit_full18.json"
    if sm.exists():
        s0 = json.loads(sm.read_text(encoding="utf-8"))
        primary["window_auc"] = s0.get("mean_auc")
        primary["window_auc_note"] = "窗级 AUC 与阈值无关，取自同一批 LOPO 模型"
        primary["median_delay_sec_default_thr"] = s0.get("median_delay_sec")
        primary["sensitivity_ge50pct_default_thr"] = s0.get("patient_level_sensitivity_ge50pct")
        primary["control_default_threshold"] = {
            "note": "未标定（默认固定阈值）的原始 LOPO 输出，仅作对照",
            "pooled_sensitivity": s0.get("pooled_sensitivity"),
            "sensitivity_mean": s0.get("patient_mean_sensitivity"),
            "pooled_fa_per_24h": s0.get("pooled_fa_per_24h"),
            "fa_mean": s0.get("patient_mean_fa_per_24h"),
        }

    (RES / "primary_endpoint.json").write_text(
        json.dumps(primary, ensure_ascii=False, indent=2), encoding="utf-8")
    df_b.to_csv(RES / "primary_endpoint_per_patient.csv", index=False,
                encoding="utf-8-sig")

    # ---- 2) 修正预算对比表 CI ----
    cmp_csv = RES / "recalib_budget_comparison.csv"
    if cmp_csv.exists():
        cmp_df = pd.read_csv(cmp_csv)
        fixed = 0
        for i, r in cmp_df.iterrows():
            strat = str(r["strategy"])
            b = float(r["budget_fa_per_24h"])
            t = f"_b{str(b).replace('.', 'p')}"
            f = RES / f"recalib{strat[0]}_per_patient{t}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f).dropna(subset=["sensitivity"])
            _, s_lo, s_hi = bootstrap_mean_ci(d.sensitivity.values, 2000)
            _, f_lo, f_hi = bootstrap_mean_ci(d.fa_per_24h.dropna().values, 2000)
            cmp_df.at[i, "sensitivity_ci95"] = str([float(s_lo), float(s_hi)])
            cmp_df.at[i, "fa_ci95"] = str([float(f_lo), float(f_hi)])
            fixed += 1
        cmp_df.to_csv(cmp_csv, index=False, encoding="utf-8-sig")
        print(f"已修正预算对比表 CI：{fixed} 行")

    # ---- 3) 打印 ----
    print("\n=== 主要终点（冻结工作点，策略 B，预算 %.0f/24h） ===" % budget)
    print(f"逐例灵敏度 均值 {primary['sensitivity_mean']*100:.1f}% "
          f"(95%CI {primary['sensitivity_ci95'][0]*100:.1f}~"
          f"{primary['sensitivity_ci95'][1]*100:.1f}%)，"
          f"中位 {primary['sensitivity_median']*100:.1f}%")
    print(f"汇总灵敏度 {primary['pooled_sensitivity']*100:.1f}% "
          f"({primary['total_detected']}/{primary['total_seizures']})")
    print(f"逐例 FA/24h 均值 {primary['fa_mean']:.2f} "
          f"(95%CI {primary['fa_ci95'][0]:.2f}~{primary['fa_ci95'][1]:.2f})，"
          f"中位 {primary['fa_median']:.2f}")
    print(f"汇总 FA/24h {primary['pooled_fa_per_24h']:.2f} "
          f"({primary['total_false_alarms']} 次 / {primary['total_hours']:.1f} h)")
    if df_a is not None:
        ca = primary["control_A_global_threshold"]
        print(f"[对照·策略A全局阈值] 灵敏度 {ca['sensitivity_mean']*100:.1f}%，"
              f"FA/24h {ca['fa_mean']:.2f}")
    print(f"\n已写出: {RES/'primary_endpoint.json'}")
    print(f"已写出: {RES/'primary_endpoint_per_patient.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
