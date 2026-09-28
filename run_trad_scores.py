# -*- coding: utf-8 -*-
"""
run_trad_scores.py —— RF 基线连续分数缓存（供策略 B/C 免推理重算）

背景：三审 B 项要求补"RF + 策略 B"对照行，使表 8 的"模型×阈值策略"归因解耦。
原 baseline_trad.py 每折约 37.5 min（步长 1 采样点连续推理 × RF 预测是瓶颈），
本脚本把每折的推理结果**只做一次**就落盘缓存，之后策略 A/B/C 全部免推理重算：
  · 每折：训练 RF（与 baseline_trad 完全同参）→ 对验证患者与测试患者的全部
    连续记录各推理一次 → 采样点级概率存 NPZ（work/trad_scores/{patient}_{role}.npz）；
  · 断点续跑：某折两个缓存均已存在则整折跳过，中断后重跑只补缺；
  · 每折结束打印累计耗时与预计剩余，便于挂机监控。

评估：用 eval_trad_recalib.py 从缓存计算（秒级）。
用法：
    python run_trad_scores.py                # 全部 23 折
    python run_trad_scores.py --limit_folds 1   # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_trad import TradModel, remap, window_features            # noqa: E402
from config import Config, add_config_args, config_from_args           # noqa: E402
from data.dataset import load_window_arrays                            # noqa: E402
from recalibrate import ensure_cache                                   # noqa: E402
from train import build_folds                                          # noqa: E402
from utils.common import get_logger, select_device, set_seed           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RF 基线连续分数缓存（23 受试者 LOPO）")
    add_config_args(ap)
    ap.add_argument("--net", default="trad_rf")
    ap.add_argument("--channel_set", default="full18")
    ap.add_argument("--n_trees", type=int, default=300)
    ap.add_argument("--trad_max_windows", type=int, default=20000)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--limit_folds", type=int, default=0, help=">0 时只跑前 N 折（冒烟）")
    args = ap.parse_args()

    cfg = config_from_args(args)
    logger = get_logger("tradscore", cfg.log_level, cfg.work_dir / "trad_scores.log")
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    from data.montage import CHANNEL_SETS
    channel_idx = CHANNEL_SETS[args.channel_set]
    cfg.n_channels = len(channel_idx)

    manifest = json.loads((cfg.index_dir / "manifest.json").read_text(encoding="utf-8"))
    ok = [m for m in manifest if m.get("ok")]
    by_subject = {}
    for m in ok:
        by_subject.setdefault(remap(m["patient"]), []).append(m)
    subjects = sorted(by_subject)
    folds = build_folds(subjects)
    if args.limit_folds:
        folds = folds[:args.limit_folds]

    cache_dir = (Path(args.cache_dir) if args.cache_dir
                 else cfg.work_dir / "trad_scores")
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"折数 {len(folds)}｜缓存目录 {cache_dir}｜设备 {device}")

    t0 = time.time()
    for i, (tr_s, val_s, te_s) in enumerate(folds, 1):
        cv = cache_dir / f"{te_s}_test.npz"
        vv = cache_dir / f"{val_s}_val.npz"
        if cv.exists() and vv.exists():
            logger.info(f"Fold {i:>2}/{len(folds)} 测试={te_s} 验证={val_s}｜缓存齐全，跳过")
            continue
        tf = time.time()
        tr_entries = [m for s in tr_s for m in by_subject[s]
                      if m.get("n_windows", 0) > 0]
        val_entries = [m for m in by_subject[val_s] if m.get("n_windows", 0) > 0]
        te_entries = by_subject[te_s]

        X, y = load_window_arrays(tr_entries, channel_idx,
                                  max_windows=args.trad_max_windows, seed=cfg.seed)
        if len(X) == 0:
            logger.warning(f"  Fold {i} 训练窗口为空，跳过")
            continue
        Xtr = window_features(X.astype(np.float32))
        rf = RandomForestClassifier(n_estimators=args.n_trees,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=cfg.seed)
        rf.fit(Xtr, y.astype(int))
        model = TradModel(rf)
        logger.info(f"Fold {i:>2}/{len(folds)} 测试={te_s} 验证={val_s}"
                    f"｜RF 训练 {time.time() - tf:.0f}s"
                    f"（窗 {len(X)}，正 {int((y == 1).sum())}）")

        ensure_cache(model, val_entries, cfg, device, channel_idx, vv, "val", logger)
        ensure_cache(model, te_entries, cfg, device, channel_idx, cv, "test", logger)

        el = time.time() - t0
        remain = el / i * (len(folds) - i)
        logger.info(f"Fold {i} 完成｜累计 {el / 60:.1f} min｜预计剩余 {remain / 60:.0f} min"
                    f"｜ETA {time.strftime('%H:%M', time.localtime(time.time() + remain))}")

    logger.info(f"全部完成，总耗时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
