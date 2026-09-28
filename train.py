# -*- coding: utf-8 -*-
"""
train.py —— 留一患者法（Leave-One-Patient-Out, LOPO）训练与评估主程序

为什么必须用"留一患者法"而不是随机划分？
    随机划分窗口会让同一位患者的相邻片段同时出现在训练集和测试集，
    造成严重的"信息泄漏"，使 AUC 虚高到 0.99 而在临床上完全失效。
    以**患者**为单位划分，才能真实反映模型面对新患者时的泛化能力。

运行：
    python train.py --dataset chbmit --epochs 30
    python train.py --dataset chbmit --folds 3 --epochs 2      # 快速试跑
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
from data.dataset import WindowDataset                             # noqa: E402
from evaluate import evaluate_patient, tune_threshold_event_level   # noqa: E402
from models.seizure_net import FocalLoss, build_model, count_parameters  # noqa: E402
from utils.common import get_logger, select_device, set_seed        # noqa: E402
from utils.metrics import binary_metrics, bootstrap_mean_ci, find_best_threshold  # noqa: E402
from utils.viz import plot_per_patient, plot_roc_pr, plot_training_curves  # noqa: E402


# ------------------------------------------------------------------ #
def build_folds(patients):
    """每位患者轮流做测试集；再取其后一位患者做验证集（用于早停与阈值搜索）。"""
    folds = []
    n = len(patients)
    for i, test_p in enumerate(patients):
        val_p = patients[(i + 1) % n] if n > 2 else patients[(i + 1) % n]
        train_p = [p for p in patients if p not in (test_p, val_p)]
        if len(train_p) == 0:
            continue
        folds.append((train_p, val_p, test_p))
    return folds


def make_loader(entries, channel_idx, cfg, augment=False, shuffle=True,
                max_windows=None, batch_size=None, seed=0):
    ds = WindowDataset(entries, channel_idx=channel_idx, augment=augment, cfg=cfg,
                       max_windows=max_windows, seed=seed)
    if len(ds) == 0:
        return None, ds
    loader = DataLoader(ds, batch_size=batch_size or cfg.batch_size, shuffle=shuffle,
                        num_workers=cfg.num_workers, drop_last=False,
                        pin_memory=torch.cuda.is_available())
    return loader, ds


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit = model(xb)
        if isinstance(logit, tuple):
            logit = logit[0]
        ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(yb.numpy())
    if not ys:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def train_one_fold(cfg, train_entries, val_entries, channel_idx, device,
                   logger, tag="fold"):
    set_seed(cfg.seed)
    train_loader, train_ds = make_loader(train_entries, channel_idx, cfg, augment=True,
                                         shuffle=True, max_windows=cfg.max_train_windows)
    val_loader, val_ds = make_loader(val_entries, channel_idx, cfg, augment=False,
                                     shuffle=False, batch_size=512)
    if train_loader is None:
        raise RuntimeError(f"{tag}: 训练集为空")

    n_pos, n_neg = train_ds.class_counts()
    logger.info(f"  {tag} 训练集窗口 {len(train_ds)}（正 {n_pos} / 负 {n_neg}），"
                f"验证集 {0 if val_loader is None else len(val_ds)}")

    model = build_model(cfg).to(device)
    logger.info(f"  模型 {cfg.model}，参数量 {count_parameters(model):,}")

    if cfg.loss == "focal":
        criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    else:
        pw = cfg.pos_weight if cfg.pos_weight > 0 else train_ds.pos_weight
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    use_amp = bool(cfg.amp and device.type == "cuda")
    try:                                        # torch >= 2.4 推荐写法
        scaler = torch.amp.GradScaler(str(device.type), enabled=use_amp)
        def _autocast():
            return torch.amp.autocast(str(device.type), enabled=use_amp)
    except (AttributeError, TypeError):         # 兼容旧版
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        def _autocast():
            return torch.cuda.amp.autocast(enabled=use_amp)

    history = {"train_loss": [], "val_auc": [], "val_f1": []}
    best_auc, best_state, bad = -np.inf, None, 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            opt.zero_grad(set_to_none=True)
            with _autocast():
                logit = model(xb)
                if isinstance(logit, tuple):
                    logit = logit[0]
                loss = criterion(logit, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item()); nb += 1
        sched.step()
        history["train_loss"].append(tot / max(1, nb))

        if val_loader is not None and len(val_ds) > 0:
            y, p = collect_probs(model, val_loader, device)
            m = binary_metrics(y, p, 0.5) if len(np.unique(y)) > 1 else {"auc": np.nan, "f1": np.nan}
            auc = m.get("auc", np.nan)
            f1 = m.get("f1", np.nan)
            history["val_auc"].append(float(auc) if np.isfinite(auc) else 0.0)
            history["val_f1"].append(float(f1) if np.isfinite(f1) else 0.0)
            logger.info(f"    epoch {ep:3d}/{cfg.epochs}  loss={history['train_loss'][-1]:.4f}  "
                        f"val_AUC={auc if np.isfinite(auc) else float('nan'):.4f}  val_F1={f1:.4f}")
            if np.isfinite(auc) and auc > best_auc:
                best_auc, bad = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if cfg.early_stop > 0 and bad >= cfg.early_stop:
                    logger.info(f"    早停于 epoch {ep}")
                    break
        else:
            logger.info(f"    epoch {ep:3d}/{cfg.epochs}  loss={history['train_loss'][-1]:.4f}")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 阈值标定：在验证患者的**连续记录**上按事件级指标搜索 ----
    # （窗级 F1 最优阈值放到连续推理上会引发上百次/天的假报警，临床不可用）
    thr = cfg.threshold
    if cfg.tune_threshold and val_entries:
        thr, info = tune_threshold_event_level(model, val_entries, cfg, device,
                                               channel_idx)
        ch = info.get("chosen")
        if ch:
            logger.info(f"  事件级阈值标定（FA预算 ≤{cfg.fa_budget}/24h）→ 阈值={thr:.3f} "
                        f"验证集灵敏度={ch['sensitivity']:.3f} FA/24h={ch['fa_per_24h']:.2f}")
        else:
            logger.info(f"  阈值标定跳过（{info.get('note')}），沿用 {thr:.3f}")
    return model, thr, history


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="留一患者法训练与事件级评估")
    add_config_args(ap)
    ap.add_argument("--folds", type=int, default=None, help="只跑前 N 折（快速试跑）")
    ap.add_argument("--test_patients", default=None,
                    help="只跑指定测试患者所在的折，逗号分隔（如 chb06,chb12,chb13）")
    # 注：--aug_scale 等 Config 字段已由 add_config_args 自动注册，此处勿重复添加
    ap.add_argument("--channel_set", default="full18",
                    help="导联子集: full18 / temporal8 / central4 / temporal4 / cz2")
    ap.add_argument("--formats", default="png,svg", help="图片导出格式")
    args = ap.parse_args()

    cfg = config_from_args(args)
    logger = get_logger("train", cfg.log_level, cfg.work_dir / "train.log")
    logger.info("配置: " + cfg.summary())
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    logger.info(f"计算设备: {device}")

    from data.montage import CHANNEL_SETS
    channel_idx = CHANNEL_SETS[args.channel_set]
    cfg.n_channels = len(channel_idx)
    logger.info(f"导联子集 {args.channel_set}（{cfg.n_channels} 导）: {channel_idx}")

    mf = cfg.index_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"未找到索引 {mf}，请先运行 python preprocess.py --dataset {cfg.dataset}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    ok = [m for m in manifest if m.get("ok")]
    usable = [m for m in ok if m.get("n_windows", 0) > 0]
    patients = sorted({m["patient"] for m in usable if m.get("n_seizures", 0) > 0})
    fp_patients = sorted({m["patient"] for m in ok if m.get("n_seizures", 0) == 0})
    logger.info(f"可训练患者 {len(patients)} 例；无发作患者（用于纯假阳性评估）{len(fp_patients)} 例")

    folds = build_folds(patients)
    if args.test_patients:
        want = {x.strip() for x in args.test_patients.split(",") if x.strip()}
        folds = [f for f in folds if f[2] in want]
        missing = want - {f[2] for f in folds}
        if missing:
            logger.warning(f"以下患者不在折列表中：{sorted(missing)}")
    if args.folds:
        folds = folds[:args.folds]
    logger.info(f"共 {len(folds)} 折"
                + (f"（测试患者：{', '.join(f[2] for f in folds)}）" if folds else ""))

    # 患者 -> 记录列表
    by_patient = {}
    for m in ok:
        by_patient.setdefault(m["patient"], []).append(m)

    rows, all_y, all_p, last_model = [], [], [], None
    for i, (tr_p, val_p, te_p) in enumerate(folds, 1):
        logger.info(f"===== Fold {i}/{len(folds)}  测试患者={te_p}  验证患者={val_p} =====")
        tr_entries = [m for p in tr_p for m in by_patient.get(p, []) if m.get("n_windows", 0) > 0]
        val_entries = [m for m in by_patient.get(val_p, []) if m.get("n_windows", 0) > 0]
        te_entries = by_patient.get(te_p, [])

        model, thr, hist = train_one_fold(cfg, tr_entries, val_entries, channel_idx,
                                          device, logger, tag=f"fold{i}")
        last_model = model
        ck = cfg.ckpt_dir / f"{cfg.dataset}_{args.channel_set}_{te_p}.pt"
        torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                    "channel_idx": channel_idx, "threshold": thr}, ck)

        if i == 1:
            plot_training_curves(hist, cfg.figure_dir / "training_curves",
                                 formats=args.formats.split(","))

        ecfg = replace(cfg, threshold=thr)
        res = evaluate_patient(model, te_entries, ecfg, device, channel_idx, logger)
        agg = res["agg"]
        row = {
            "fold": i, "patient": te_p,
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
        }
        rows.append(row)
        logger.info(f"  >>> {te_p}: 灵敏度 {row['sensitivity']:.3f} "
                    f"({row['n_detected']}/{row['n_seizures']})，FA/24h = {row['fa_per_24h']:.2f}")
        if "y_true" in res:
            all_y.append(res["y_true"]); all_p.append(res["y_prob"])

    df = pd.DataFrame(rows)
    df.to_csv(cfg.result_dir / f"lopo_{cfg.dataset}_{args.channel_set}.csv",
              index=False, encoding="utf-8-sig")

    # ---------------- 汇总统计 ---------------- #
    sens = df["sensitivity"].dropna().values
    fa = df["fa_per_24h"].dropna().values
    tot_seiz = int(df["n_seizures"].sum()); tot_det = int(df["n_detected"].sum())
    tot_fa = int(df["n_false_alarms"].sum()); tot_h = float(df["total_hours"].sum())
    s_pt, s_lo, s_hi = bootstrap_mean_ci(sens, cfg.n_boot, seed=cfg.seed)
    f_pt, f_lo, f_hi = bootstrap_mean_ci(fa, cfg.n_boot, seed=cfg.seed)

    summary = {
        "dataset": cfg.dataset, "channel_set": args.channel_set, "n_folds": len(df),
        "pooled_sensitivity": tot_det / tot_seiz if tot_seiz else np.nan,
        "patient_mean_sensitivity": s_pt, "sensitivity_ci95": [s_lo, s_hi],
        "pooled_fa_per_24h": tot_fa / tot_h * 24 if tot_h else np.nan,
        "patient_mean_fa_per_24h": f_pt, "fa_per_24h_ci95": [f_lo, f_hi],
        "mean_auc": float(df["sample_auc"].mean(skipna=True)),
        "median_delay_sec": float(df["delay_median_sec"].median(skipna=True)),
        "patient_level_sensitivity_ge50pct": float((sens >= 0.5).mean()),
        "total_seizures": tot_seiz, "total_detected": tot_det,
        "total_false_alarms": tot_fa, "total_hours": round(tot_h, 1),
    }
    # 无发作患者的纯假阳性评估
    if fp_patients and last_model is not None:
        fp_res = {}
        for p in fp_patients:
            r = evaluate_patient(last_model, by_patient[p], replace(cfg, threshold=thr),
                                 device, channel_idx, None)
            fp_res[p] = r["agg"]
        summary["seizure_free_patients"] = {k: {"hours": round(v["total_hours"], 2),
                                                "fa_per_24h": round(v["fa_per_24h"], 3)}
                                            for k, v in fp_res.items()}
    (cfg.result_dir / f"summary_{cfg.dataset}_{args.channel_set}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 绘图 ---------------- #
    fmts = args.formats.split(",")
    if all_y:
        plot_roc_pr(np.concatenate(all_y), np.concatenate(all_p),
                    cfg.figure_dir / "roc_pr", title=f"{cfg.dataset} · LOPO", formats=fmts)
    plot_per_patient(df, cfg.figure_dir / "per_patient", formats=fmts)

    print("\n" + "=" * 72)
    print(f"【{cfg.dataset} · {args.channel_set}】留一患者法结果")
    print("=" * 72)
    print(f"发作总数 {tot_seiz}，检出 {tot_det}，汇总灵敏度 {summary['pooled_sensitivity']:.3f}")
    print(f"患者水平平均灵敏度 {s_pt:.3f} (95%CI {s_lo:.3f}~{s_hi:.3f})")
    print(f"假阳性 {tot_fa} 次 / {tot_h:.1f} h  →  汇总 FA/24h = {summary['pooled_fa_per_24h']:.2f}")
    print(f"患者水平平均 FA/24h {f_pt:.2f} (95%CI {f_lo:.2f}~{f_hi:.2f})")
    print(f"平均窗级 AUC = {summary['mean_auc']:.3f}；中位检出延迟 = {summary['median_delay_sec']:.1f} s")
    print(f"结果已保存至: {cfg.result_dir}")
    logger.info("全部完成。")


if __name__ == "__main__":
    main()
