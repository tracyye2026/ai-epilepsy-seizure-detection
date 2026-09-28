# -*- coding: utf-8 -*-
"""
ablation.py —— 导联精简消融实验（本研究的临床转化亮点）

科学问题：**基层医院常用的 8/4 导简化脑电，还能不能跑 AI 辅助判读？**
做法：在完全相同的训练/验证/测试划分下，仅改变输入导联集合，重跑 LOPO，
      用**配对 Bootstrap**（以患者为配对单位）检验性能差异。

用法（--sets 的顺序即论文表格顺序）：
    python ablation.py --sets full18 temporal8 temporal4 cz2 --epochs 20
    python ablation.py --sets full18 temporal8 --folds 5 --epochs 2   # 快速试跑
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, add_config_args, config_from_args    # noqa: E402
from data.montage import CHANNEL_SETS                           # noqa: E402
from utils.common import get_logger                             # noqa: E402
from utils.metrics import bootstrap_mean_ci, paired_bootstrap_test  # noqa: E402
from utils.viz import plot_ablation                             # noqa: E402

# 论文表格中的展示名
PRETTY = {"full18": "18 导（全导联）", "temporal8": "8 导（双颞链）",
          "central4": "4 导（中央区）", "temporal4": "4 导（颞区）", "cz2": "2 导（中线）"}


def run_one(cfg, ch_set, args, logger):
    """调用 train.py 训练并评估某一导联配置。"""
    out_csv = cfg.result_dir / f"lopo_{cfg.dataset}_{ch_set}.csv"
    cmd = [sys.executable, "train.py",
           "--dataset", cfg.dataset,
           "--data_root", str(cfg.data_root),
           "--work_root", str(cfg.work_root),
           "--channel_set", ch_set,
           "--epochs", str(cfg.epochs),
           "--max_train_windows", str(cfg.max_train_windows),
           "--formats", args.formats]
    if args.folds:
        cmd += ["--folds", str(args.folds)]
    logger.info("执行: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
    if r.returncode != 0:
        logger.warning(f"配置 {ch_set} 运行失败（returncode={r.returncode}）")
        return None
    return pd.read_csv(out_csv) if out_csv.exists() else None


def main():
    ap = argparse.ArgumentParser(description="导联精简消融实验")
    add_config_args(ap)
    ap.add_argument("--sets", nargs="+", default=["full18", "temporal8", "temporal4", "cz2"])
    # 注意：--epochs 由 add_config_args 统一提供，此处不可重复声明
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--formats", default="png,svg")
    ap.add_argument("--skip_run", action="store_true",
                    help="跳过训练，直接用已有 CSV 汇总（改图/改表时用）")
    args = ap.parse_args()

    cfg = config_from_args(args)
    logger = get_logger("ablation", cfg.log_level, cfg.work_dir / "ablation.log")

    dfs = {}
    for ch_set in args.sets:
        if ch_set not in CHANNEL_SETS:
            logger.warning(f"未知导联集合 {ch_set}，跳过")
            continue
        csv_path = cfg.result_dir / f"lopo_{cfg.dataset}_{ch_set}.csv"
        if args.skip_run and csv_path.exists():
            dfs[ch_set] = pd.read_csv(csv_path)
            continue
        df = run_one(cfg, ch_set, args, logger)
        if df is not None:
            dfs[ch_set] = df
    if not dfs:
        raise RuntimeError("没有可用的实验结果")

    # ---------------- 汇总 ---------------- #
    rows, per_patient = [], {}
    for ch_set, df in dfs.items():
        n_ch = len(CHANNEL_SETS[ch_set])
        sens = df["sensitivity"].dropna().values
        fa = df["fa_per_24h"].dropna().values
        s_pt, s_lo, s_hi = bootstrap_mean_ci(sens, cfg.n_boot, seed=cfg.seed)
        f_pt, f_lo, f_hi = bootstrap_mean_ci(fa, cfg.n_boot, seed=cfg.seed)
        per_patient[ch_set] = df.set_index("patient")["sensitivity"]
        rows.append({
            "导联配置": PRETTY.get(ch_set, ch_set), "配置名": ch_set, "导联数": n_ch,
            "检出/总数": f"{int(df['n_detected'].sum())}/{int(df['n_seizures'].sum())}",
            "汇总灵敏度": round(float(df["n_detected"].sum() / max(1, df["n_seizures"].sum())), 3),
            "患者平均灵敏度": round(s_pt, 3),
            "灵敏度95%CI": f"{s_lo:.3f}~{s_hi:.3f}",
            "平均FA/24h": round(f_pt, 2),
            "FA/24h 95%CI": f"{f_lo:.2f}~{f_hi:.2f}",
            "平均窗级AUC": round(float(df["sample_auc"].mean(skipna=True)), 3),
            "中位延迟(s)": round(float(df["delay_median_sec"].median(skipna=True)), 2),
            "ci_low": s_lo, "ci_high": s_hi,
        })
    res = pd.DataFrame(rows).sort_values("导联数", ascending=False)

    # ---------------- 与全导联的配对比较 ---------------- #
    base = "full18"
    if base in per_patient:
        diffs, ps = [], []
        for _, r in res.iterrows():
            cs = r["配置名"]
            if cs == base:
                diffs.append("—"); ps.append("—"); continue
            common = per_patient[base].index.intersection(per_patient[cs].index)
            a = per_patient[base].loc[common].values
            b = per_patient[cs].loc[common].values
            t = paired_bootstrap_test(a, b, n_boot=2000, seed=cfg.seed)
            diffs.append(round(a.mean() - b.mean(), 3))
            ps.append(round(t["p_value"], 4))
        res["Δ灵敏度(vs全导联)"] = diffs
        res["P值(配对Bootstrap)"] = ps

    out_csv = cfg.result_dir / "ablation_channels.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"消融结果已保存: {out_csv}")

    plot_ablation(res, cfg.figure_dir / "ablation_channels",
                  x_col="导联数", y_col="患者平均灵敏度",
                  formats=args.formats.split(","), logger=logger,
                  ylabel="患者平均发作检出率",
                  title="导联精简对发作检出率的影响（含 95%CI）")

    print("\n" + "=" * 78)
    print("导联精简消融结果")
    print("=" * 78)
    cols = [c for c in ["导联配置", "导联数", "检出/总数", "患者平均灵敏度",
                        "灵敏度95%CI", "平均FA/24h", "Δ灵敏度(vs全导联)",
                        "P值(配对Bootstrap)"] if c in res.columns]
    print(res[cols].to_string(index=False))
    print(f"\n结果: {out_csv}")
    print("解读要点：若 18→8 导的 Δ 灵敏度 < 0.05 且 P > 0.05，即支持"
          "'基层 8~16 导设备可部署'的结论。")


if __name__ == "__main__":
    main()
