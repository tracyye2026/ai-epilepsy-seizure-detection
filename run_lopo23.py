# -*- coding: utf-8 -*-
"""
run_lopo23.py —— 23 受试者合并的留一患者法（LOPO）训练与评估

与 train.py 的唯一区别：把 chb01 / chb21 视为**同一受试者**（CHB-MIT 官方注释
chb21 为 chb01 的同源后续记录），因此 LOPO 折数由 24 降为 23，且测试该合并受试者时
chb01 与 chb21 的**全部记录同时移出训练集**，彻底消除此前"留一患者"方案里的受试者级泄漏。

复用策略（省算且统计成立）：
  · 对于非合并受试者 X，其旧 24 折训练集与 23 受试者方案训练集完全一致
    （都包含 chb01+chb21 两条记录），故可直接复用既有权重；
  · 仅以下三折的训练集在新旧方案下有差异，必须重训：
      - chb01（合并折：测试 chb01+chb21，二者同时移出）
      - chb20（旧方案验证患者=chb21，故旧权重训练时排除了 chb21）
      - chb24（旧方案验证患者=chb01（环绕），故旧权重训练时排除了 chb01）
  · 其余 22 折复用 work/ckpt/{model}_full18_{te}.pt 既有权重。

输出：
  work/results/lopo_chbmit_full18_23subj.csv            23 受试者逐例（事件级）
  work/results/lopo_chbmit_full18_23subj_detail.csv     逐记录明细（含 chb01/chb21 各自）
  work/folds_lopo23.json                                23 折快照（供 recalibrate 使用）
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, add_config_args, config_from_args       # noqa: E402
from evaluate import evaluate_patient, tune_threshold_event_level  # noqa: E402
from models.seizure_net import FocalLoss, build_model, count_parameters  # noqa: E402
from utils.common import get_logger, select_device, set_seed        # noqa: E402
from utils.metrics import bootstrap_mean_ci                          # noqa: E402
from train import make_loader, train_one_fold, build_folds           # noqa: E402


# chb21 同源 chb01 → 合并为同一受试者
MERGE = {"chb21": "chb01"}


def remap(p: str) -> str:
    return MERGE.get(p, p)


def main():
    ap = argparse.ArgumentParser(description="23 受试者合并 LOPO")
    add_config_args(ap)
    ap.add_argument("--net", default="seizurenet", help="seizurenet | resnet1d")
    ap.add_argument("--channel_set", default="full18")
    ap.add_argument("--folds", type=int, default=0, help=">0 时只跑前 N 折（冒烟测试）")
    ap.add_argument("--retrain_subjects", default="chb01,chb20,chb24",
                    help="强制重训的受试者（其余复用既有权重）")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = config_from_args(args)
    cfg.model = args.net
    logger = get_logger("lopo23", cfg.log_level, cfg.work_dir / "lopo23.log")
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    logger.info(f"计算设备: {device}｜模型: {cfg.model}")

    from data.montage import CHANNEL_SETS
    channel_idx = CHANNEL_SETS[args.channel_set]
    cfg.n_channels = len(channel_idx)

    manifest = json.loads((cfg.index_dir / "manifest.json").read_text(encoding="utf-8"))
    ok = [m for m in manifest if m.get("ok")]

    # ---- 按"受试者"聚合记录（chb21 并入 chb01）----
    by_subject = {}
    for m in ok:
        by_subject.setdefault(remap(m["patient"]), []).append(m)
    subjects = sorted(by_subject)                     # 23 个受试者
    logger.info(f"受试者数（合并后）: {len(subjects)} -> {subjects}")

    folds = build_folds(subjects)                     # 23 折
    if args.folds:
        folds = folds[:args.folds]
    logger.info(f"共 {len(folds)} 折")

    force = {x.strip() for x in args.retrain_subjects.split(",") if x.strip()}
    logger.info(f"强制重训受试者: {sorted(force)}；其余复用既有权重")

    rows, detail_rows = [], []
    for i, (tr_s, val_s, te_s) in enumerate(folds, 1):
        logger.info(f"===== Fold {i}/{len(folds)}  测试受试者={te_s}  验证={val_s} =====")
        te_entries = by_subject[te_s]
        tr_entries = [m for s in tr_s for m in by_subject[s]
                      if m.get("n_windows", 0) > 0]
        val_entries = [m for m in by_subject[val_s] if m.get("n_windows", 0) > 0]

        ck = cfg.ckpt_dir / f"{cfg.dataset}_{args.channel_set}_{args.net}_{te_s}.pt"
        acfg = replace(cfg, threshold=cfg.threshold)
        if ck.exists() and te_s not in force:
            logger.info(f"  复用既有权重 {ck.name}")
            data = torch.load(ck, map_location=device, weights_only=False)
            acfg.__dict__.update(data["cfg"])
            acfg.n_channels = len(data.get("channel_idx", channel_idx))
            model = build_model(acfg).to(device)
            model.load_state_dict(data["state_dict"])
            model.eval()
            thr = float(data.get("threshold", cfg.threshold))
        else:
            logger.info(f"  重训（{te_s}）训练集窗口来源记录数={len(tr_entries)}")
            model, thr, hist = train_one_fold(
                cfg, tr_entries, val_entries, channel_idx, device, logger,
                tag=f"fold{i}")
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                        "channel_idx": channel_idx, "threshold": thr}, ck)

        ecfg = replace(cfg, threshold=thr)
        res = evaluate_patient(model, te_entries, ecfg, device, channel_idx, logger)
        agg = res["agg"]
        # 逐记录明细
        for r in res["records"]:
            detail_rows.append({
                "fold": i, "subject": te_s, "record": r.get("record"),
                "patient_label": r.get("patient"),
                "n_seizures": r.get("n_seizures", 0),
                "n_detected": r.get("n_detected", 0),
                "n_false_alarms": r.get("n_false_alarms", 0),
                "hours": round(r.get("hours", 0), 3),
                "sensitivity": (r.get("n_detected", 0) / r["n_seizures"]
                                if r.get("n_seizures", 0) else np.nan),
                "fa_per_24h": (r.get("n_false_alarms", 0) / r["hours"] * 24
                               if r.get("hours", 0) > 0 else np.nan),
            })
        rows.append({
            "fold": i, "patient": te_s,
            "n_records": agg.get("n_records", 0),
            "n_seizures": agg.get("n_seizures", 0),
            "n_detected": agg.get("n_detected", 0),
            "sensitivity": agg.get("sensitivity", np.nan),
            "n_false_alarms": agg.get("n_false_alarms", 0),
            "total_hours": round(agg.get("total_hours", 0), 2),
            "fa_per_24h": round(agg.get("fa_per_24h", np.nan), 3),
            "sample_auc": round(agg.get("sample_auc_mean", np.nan), 4),
            "delay_median_sec": round(agg.get("delay_median_sec", np.nan), 2),
            "seizure_coverage": round(agg.get("seizure_coverage", np.nan), 3),
            "threshold": round(thr, 3),
        })
        logger.info(f"  >>> {te_s}: 灵敏度 {rows[-1]['sensitivity']:.3f} "
                    f"({rows[-1]['n_detected']}/{rows[-1]['n_seizures']})，"
                    f"FA/24h = {rows[-1]['fa_per_24h']:.2f}")

    df = pd.DataFrame(rows)
    # 精确率 / F1（事件级，逐受试者）
    df["precision"] = df.apply(
        lambda r: (r.n_detected / (r.n_detected + r.n_false_alarms))
        if (r.n_detected + r.n_false_alarms) > 0 else np.nan, axis=1)
    df["f1"] = df.apply(
        lambda r: (2 * r.precision * r.sensitivity / (r.precision + r.sensitivity))
        if (r.precision is not None and np.isfinite(r.precision)
            and (r.precision + r.sensitivity) > 0) else np.nan, axis=1)

    base = f"lopo_{cfg.dataset}_{args.channel_set}_{args.net}_23subj"
    out = args.out_csv or str(cfg.result_dir / f"{base}.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(
        str(cfg.result_dir / f"{base}_detail.csv"),
        index=False, encoding="utf-8-sig")

    # 折快照（供 recalibrate）
    snap = cfg.work_dir / "folds_lopo23.json"
    snap.write_text(json.dumps(
        [{"idx": i, "test": te, "val": val} for i, (_, val, te) in enumerate(folds, 1)],
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    sens = df["sensitivity"].dropna().values
    fa = df["fa_per_24h"].dropna().values
    tot_seiz = int(df.n_seizures.sum()); tot_det = int(df.n_detected.sum())
    tot_fa = int(df.n_false_alarms.sum()); tot_h = float(df.total_hours.sum())
    s_pt, s_lo, s_hi = bootstrap_mean_ci(sens, cfg.n_boot, seed=cfg.seed)
    f_pt, f_lo, f_hi = bootstrap_mean_ci(fa, cfg.n_boot, seed=cfg.seed)
    print("\n===== 23 受试者合并 LOPO 汇总 =====")
    print(f"受试者数 {len(df)}｜发作 {tot_seiz}｜检出 {tot_det}｜汇总灵敏度 "
          f"{tot_det/max(tot_seiz,1):.3f}")
    print(f"逐例灵敏度均值 {s_pt:.3f} (95%CI {s_lo:.3f}~{s_hi:.3f})")
    print(f"假阳性 {tot_fa} 次 / {tot_h:.1f} h → 汇总 FA/24h {tot_fa/tot_h*24:.2f}")
    print(f"逐例 FA/24h 均值 {f_pt:.2f} (95%CI {f_lo:.2f}~{f_hi:.2f})")
    print(f"逐例精确率均值 {df.precision.mean():.3f}｜逐例 F1 均值 {df.f1.mean():.3f}")
    print(f"结果: {out}")


if __name__ == "__main__":
    main()
