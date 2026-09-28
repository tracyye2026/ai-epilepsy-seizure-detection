# -*- coding: utf-8 -*-
"""
external_validate.py —— 跨数据库外部验证（本研究最重要的原创性来源）

设计：
    源库（训练集）：CHB-MIT 儿童头皮 EEG（23 例有发作）
    目标库（外部验证集）：Siena 成人头皮 EEG（14 例，参考导联、512 Hz、不同人种/设备/年龄）

只有完成"跨库外部验证"，才能回答审稿人最关心的问题：
    "你的模型换个医院、换个设备、换个年龄段还能用吗？"

前置步骤：
    python preprocess.py --dataset chbmit
    python preprocess.py --dataset siena        # 需先下载 Siena
    python external_validate.py --source chbmit --target siena
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, add_config_args, config_from_args     # noqa: E402
from data.dataset import WindowDataset                           # noqa: E402
from evaluate import evaluate_patients                           # noqa: E402
from train import train_one_fold                                 # noqa: E402
from utils.common import get_logger, select_device, set_seed      # noqa: E402
from utils.metrics import binary_metrics, bootstrap_mean_ci       # noqa: E402
from utils.viz import plot_per_patient, plot_roc_pr               # noqa: E402


def load_ok_manifest(cfg):
    mf = cfg.index_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"未找到 {mf}，请先运行 python preprocess.py --dataset {cfg.dataset}")
    return [m for m in json.loads(mf.read_text(encoding="utf-8")) if m.get("ok")]


def main():
    ap = argparse.ArgumentParser(description="跨数据库外部验证")
    add_config_args(ap)
    ap.add_argument("--source", default="chbmit")
    ap.add_argument("--target", default="siena")
    # 注：--epochs 等 Config 字段已由 add_config_args 自动注册，此处勿重复添加
    ap.add_argument("--formats", default="png,svg")
    args = ap.parse_args()

    base = config_from_args(args)
    logger = get_logger("extval", base.log_level, base.work_dir / "external_validate.log")
    set_seed(base.seed)
    device = select_device(base.device)

    # ---------------- 1. 在源库全体患者上训练最终模型 ---------------- #
    src_cfg = replace(base, dataset=args.source)
    src_cfg.ensure_dirs()
    src_manifest = load_ok_manifest(src_cfg)
    patients = sorted({m["patient"] for m in src_manifest if m.get("n_seizures", 0) > 0})
    if len(patients) < 3:
        raise RuntimeError("源库患者数不足，无法训练")
    val_p = patients[-1]
    tr_p = patients[:-1]
    by_p = {}
    for m in src_manifest:
        by_p.setdefault(m["patient"], []).append(m)
    tr_entries = [m for p in tr_p for m in by_p[p] if m.get("n_windows", 0) > 0]
    val_entries = [m for m in by_p[val_p] if m.get("n_windows", 0) > 0]

    logger.info(f"源库 {args.source}: {len(patients)} 例，训练 {len(tr_p)} 例 / 验证 {val_p}")
    if args.epochs:
        src_cfg.epochs = args.epochs
    model, thr, _ = train_one_fold(src_cfg, tr_entries, val_entries, list(range(18)),
                                   device, logger, tag="external-final")
    ck = src_cfg.ckpt_dir / f"final_{args.source}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": src_cfg.__dict__,
                "threshold": thr}, ck)
    logger.info(f"最终模型已保存: {ck}")

    # ---------------- 2. 在目标库上做外部验证 ---------------- #
    tgt_cfg = replace(base, dataset=args.target)
    tgt_cfg.ensure_dirs()
    tgt_manifest = load_ok_manifest(tgt_cfg)
    tgt_by_p = {}
    for m in tgt_manifest:
        tgt_by_p.setdefault(m["patient"], []).append(m)
    logger.info(f"目标库 {args.target}: {len(tgt_by_p)} 例患者待评估，阈值 {thr:.3f}")

    ecfg = replace(tgt_cfg, threshold=thr)
    results = evaluate_patients(model, tgt_by_p, ecfg, device, None, logger)

    rows, ys, ps = [], [], []
    for p, r in results.items():
        a = r["agg"]
        rows.append({"patient": p, "n_records": a.get("n_records", 0),
                     "n_seizures": a.get("n_seizures", 0), "n_detected": a.get("n_detected", 0),
                     "sensitivity": a.get("sensitivity", np.nan),
                     "n_false_alarms": a.get("n_false_alarms", 0),
                     "total_hours": round(a.get("total_hours", 0), 2),
                     "fa_per_24h": round(a.get("fa_per_24h", np.nan), 3),
                     "delay_median_sec": round(a.get("delay_median_sec", np.nan), 2)})
        if "y_true" in r:
            ys.append(r["y_true"]); ps.append(r["y_prob"])
    df = pd.DataFrame(rows)
    df.to_csv(tgt_cfg.result_dir / f"external_{args.source}2{args.target}.csv",
              index=False, encoding="utf-8-sig")

    sens = df["sensitivity"].dropna().values
    fa = df["fa_per_24h"].dropna().values
    sp, slo, shi = bootstrap_mean_ci(sens, base.n_boot, seed=base.seed)
    fp, flo, fhi = bootstrap_mean_ci(fa, base.n_boot, seed=base.seed)
    auc = binary_metrics(np.concatenate(ys), np.concatenate(ps), thr)["auc"] if ys else np.nan

    summary = {
        "source": args.source, "target": args.target, "n_patients": len(df),
        "threshold": thr,
        "pooled_sensitivity": float(df["n_detected"].sum() / max(1, df["n_seizures"].sum())),
        "patient_mean_sensitivity": sp, "sensitivity_ci95": [slo, shi],
        "pooled_fa_per_24h": float(df["n_false_alarms"].sum() /
                                   max(1e-6, df["total_hours"].sum()) * 24),
        "patient_mean_fa_per_24h": fp, "fa_per_24h_ci95": [flo, fhi],
        "window_auc": float(auc),
        "total_seizures": int(df["n_seizures"].sum()),
        "total_detected": int(df["n_detected"].sum()),
        "total_hours": float(df["total_hours"].sum()),
    }
    # 与内部验证（LOPO）对比，量化泛化差距
    internal = base.result_dir / f"summary_{args.source}_full18.json"
    if internal.exists():
        d = json.loads(internal.read_text(encoding="utf-8"))
        summary["internal_lopo_sensitivity"] = d.get("patient_mean_sensitivity")
        summary["generalization_gap_sensitivity"] = (
            d.get("patient_mean_sensitivity", np.nan) - sp
            if d.get("patient_mean_sensitivity") is not None else np.nan)
    (tgt_cfg.result_dir / f"external_summary_{args.source}2{args.target}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fmts = args.formats.split(",")
    plot_per_patient(df, tgt_cfg.figure_dir / f"external_{args.source}2{args.target}",
                     formats=fmts, logger=logger)
    if ys:
        plot_roc_pr(np.concatenate(ys), np.concatenate(ps),
                    tgt_cfg.figure_dir / f"roc_pr_{args.source}2{args.target}",
                    title=f"外部验证 {args.source} → {args.target}", formats=fmts, logger=logger)

    print("\n" + "=" * 72)
    print(f"【外部验证】{args.source} 训练 → {args.target} 测试")
    print("=" * 72)
    print(f"患者数 {summary['n_patients']}，发作 {summary['total_seizures']} 次，"
          f"检出 {summary['total_detected']} 次")
    print(f"汇总灵敏度 {summary['pooled_sensitivity']:.3f}；"
          f"患者平均 {sp:.3f} (95%CI {slo:.3f}~{shi:.3f})")
    print(f"汇总 FA/24h {summary['pooled_fa_per_24h']:.2f}；"
          f"患者平均 {fp:.2f} (95%CI {flo:.2f}~{fhi:.2f})")
    print(f"窗级 AUC = {auc:.3f}")
    if "generalization_gap_sensitivity" in summary:
        print(f"与内部 LOPO 的灵敏度差距 = {summary['generalization_gap_sensitivity']:+.3f}")
    print(f"结果: {tgt_cfg.result_dir / f'external_{args.source}2{args.target}.csv'}")


if __name__ == "__main__":
    main()
