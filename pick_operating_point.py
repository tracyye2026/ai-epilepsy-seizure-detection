"""从 recalibrate.py 的产物中提取"预先指定的工作点"，供外部验证沿用。

外部验证的诚实协议是：**工作点必须在源库（CHB-MIT）验证患者上标定完成并冻结**，
再原样应用到目标库（Siena），不得在目标库上回看调整。本脚本即负责冻结这一步。

用法:
  python -u pick_operating_point.py --budget 10
  python -u pick_operating_point.py --budget 10 --format shell   # 输出 Q/F/MD 供 bash 读取
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--strategy", default="B", choices=["A", "B"])
    ap.add_argument("--format", default="shell", choices=["shell", "json"])
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    res = Path(args.work_dir) if args.work_dir else (here / "work" / "results")
    res = res / "results" if not res.name == "results" else res
    cmp_csv = res / "recalib_budget_comparison.csv"
    if not cmp_csv.exists():
        raise SystemExit(f"未找到 {cmp_csv}，请先运行 recalibrate.py")

    df = pd.read_csv(cmp_csv)
    df = df[df["strategy"].astype(str).str.startswith(args.strategy)]
    if df.empty:
        raise SystemExit(f"策略 {args.strategy} 无结果")
    budgets = df["budget_fa_per_24h"].astype(float).tolist()
    b = min(budgets, key=lambda x: abs(x - args.budget))
    row = df[df["budget_fa_per_24h"].astype(float) == b].iloc[0]

    thr = str(row["threshold"])
    md = float(row["min_event_sec"])
    if args.strategy == "B" and "q=" in thr:
        # 形如 "q=0.995/floor=0.5"
        q = float(thr.split("q=")[1].split("/")[0])
        f = float(thr.split("floor=")[1])
        mode = "quantile"
    else:
        q, f, mode = 0.995, float(thr), "absolute"

    out = {"strategy": str(row["strategy"]), "budget": b, "threshold_mode": mode,
           "thr_quantile": q, "thr_floor": f, "threshold": f if mode == "absolute" else q,
           "min_event_sec": md,
           "val_sensitivity": float(row["val_sensitivity"]),
           "val_fa_per_24h": float(row["val_fa_per_24h"]),
           "test_sensitivity": float(row["pooled_sensitivity"]),
           "test_fa_per_24h": float(row["pooled_fa_per_24h"])}
    (res / "operating_point.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.format == "shell":
        print(f"{q} {f} {md:g}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
