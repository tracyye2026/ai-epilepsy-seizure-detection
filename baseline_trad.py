# -*- coding: utf-8 -*-
"""
baseline_trad.py —— 跨患者基线（一）：传统手工特征 + 随机森林

在 23 受试者合并 LOPO 下，与深度模型完全相同的"连续推理→事件级"评估口径：
  · 训练：对每个训练窗口提取手工特征（每导联 5 个时域 + 5 个频带功率），训练 RandomForest；
  · 评估：对测试记录的连续信号滑窗提取特征→预测概率→还原采样点级概率→
          evaluate_continuous 做平滑/阈值/事件匹配，输出事件级灵敏度、FA/24h、精确率、F1；
  · 阈值：在验证患者的连续记录上按事件级指标标定（与深度模型策略 C 一致）。

目的：回应评审"缺跨患者基线"——证明深度架构相对经典 EEG 特征方法的增益（或持平）。
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
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, add_config_args, config_from_args       # noqa: E402
from evaluate import evaluate_patient, tune_threshold_event_level  # noqa: E402
from utils.common import get_logger, select_device, set_seed        # noqa: E402
from utils.metrics import bootstrap_mean_ci                          # noqa: E402
from train import build_folds                                        # noqa: E402
from data.dataset import load_window_arrays                         # noqa: E402


MERGE = {"chb21": "chb01"}


def remap(p: str) -> str:
    return MERGE.get(p, p)


# 频带（Hz），与预处理低通 45 Hz 对应
BANDS = [(0.5, 4.0, "delta"), (4.0, 8.0, "theta"), (8.0, 13.0, "alpha"),
         (13.0, 30.0, "beta"), (30.0, 45.0, "gamma")]
SF = 128.0
WIN = int(round(4.0 * SF))   # 512


def band_bins():
    freqs = np.fft.rfftfreq(WIN, 1.0 / SF)
    edges = []
    for lo, hi, _ in BANDS:
        edges.append((np.searchsorted(freqs, lo, "left"),
                      np.searchsorted(freqs, hi, "right")))
    return edges


_BINS = band_bins()


def window_features(X: np.ndarray) -> np.ndarray:
    """X: (N, C, T) -> (N, C*(4 时域 + 5 频带))。

    时域：均值、标准差、线长度、过零率、峰峰值；频域：5 个频带功率（|rFFT|^2 求和）。
    """
    X = np.asarray(X, dtype=np.float32)
    N, C, T = X.shape
    # 时域
    mean = X.mean(-1)
    std = X.std(-1)
    ll = np.abs(np.diff(X, axis=-1)).sum(-1)
    zc = (X[..., 1:] * X[..., :-1] < 0).sum(-1).astype(np.float32)
    rng = X.max(-1) - X.min(-1)
    time_feat = np.stack([mean, std, ll, zc, rng], axis=-1)  # (N,C,5)
    # 频域
    Xf = np.fft.rfft(X, axis=-1)                            # (N,C,F)
    P = (Xf.real ** 2 + Xf.imag ** 2)                       # 功率谱
    band = []
    for (a, b) in _BINS:
        band.append(P[..., a:b].sum(-1))
    band = np.stack(band, axis=-1)                          # (N,C,5)
    feat = np.concatenate([time_feat, band], axis=-1)       # (N,C,10)
    return feat.reshape(N, -1)                              # (N, C*10=180)


class TradModel:
    """把 RandomForest 包成 evaluate 可用的 "model"（forward 返回 logits）。"""
    def __init__(self, rf: RandomForestClassifier):
        self.rf = rf

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        return self.forward(xb)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        x = xb.detach().cpu().numpy().astype(np.float32)
        feat = window_features(x)
        proba = self.rf.predict_proba(feat)[:, 1]
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        logit = np.log(proba / (1.0 - proba))
        return torch.from_numpy(logit).float().to(xb.device).unsqueeze(1)


def main():
    ap = argparse.ArgumentParser(description="传统特征基线（23 受试者 LOPO）")
    add_config_args(ap)
    ap.add_argument("--net", default="trad_rf")
    ap.add_argument("--channel_set", default="full18")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--n_trees", type=int, default=300)
    ap.add_argument("--trad_max_windows", type=int, default=20000)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = config_from_args(args)
    logger = get_logger("trad", cfg.log_level, cfg.work_dir / "trad.log")
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
    if args.folds:
        folds = folds[:args.folds]
    logger.info(f"受试者 {len(subjects)}｜折 {len(folds)}")

    rows = []
    for i, (tr_s, val_s, te_s) in enumerate(folds, 1):
        logger.info(f"===== Fold {i}/{len(folds)} 测试={te_s} 验证={val_s} =====")
        tr_entries = [m for s in tr_s for m in by_subject[s]
                      if m.get("n_windows", 0) > 0]
        val_entries = [m for m in by_subject[val_s] if m.get("n_windows", 0) > 0]
        te_entries = by_subject[te_s]

        # 训练特征
        X, y = load_window_arrays(tr_entries, channel_idx,
                                  max_windows=args.trad_max_windows, seed=cfg.seed)
        if len(X) == 0:
            logger.warning(f"  Fold {i} 训练窗口为空，跳过"); continue
        # 含发作记录不足时仍需保留无发作记录（负样本）
        Xtr = window_features(X.astype(np.float32))
        rf = RandomForestClassifier(n_estimators=args.n_trees,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=cfg.seed)
        rf.fit(Xtr, y.astype(int))
        logger.info(f"  训练窗口 {len(X)}（正 {int((y==1).sum())}）｜特征维 {Xtr.shape[1]}")
        model = TradModel(rf)

        # 阈值标定（验证患者连续记录，事件级）
        thr = cfg.threshold
        if args.tune_threshold and val_entries:
            thr, info = tune_threshold_event_level(model, val_entries, cfg, device,
                                                   channel_idx)
            ch = info.get("chosen")
            if ch:
                logger.info(f"  事件级阈值标定→ {thr:.3f} 验证灵敏度 {ch['sensitivity']:.3f} "
                            f"FA {ch['fa_per_24h']:.2f}")

        ecfg = replace(cfg, threshold=thr)
        res = evaluate_patient(model, te_entries, ecfg, device, channel_idx, logger)
        agg = res["agg"]
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
            "threshold": round(thr, 3),
        })
        logger.info(f"  >>> {te_s}: 灵敏度 {rows[-1]['sensitivity']:.3f} "
                    f"({rows[-1]['n_detected']}/{rows[-1]['n_seizures']})，"
                    f"FA/24h {rows[-1]['fa_per_24h']:.2f}")

    df = pd.DataFrame(rows)
    df["precision"] = df.apply(
        lambda r: r.n_detected/(r.n_detected+r.n_false_alarms)
        if (r.n_detected+r.n_false_alarms) > 0 else np.nan, axis=1)
    df["f1"] = df.apply(
        lambda r: 2*r.precision*r.sensitivity/(r.precision+r.sensitivity)
        if (r.precision is not None and np.isfinite(r.precision)
            and (r.precision+r.sensitivity) > 0) else np.nan, axis=1)

    out = args.out_csv or str(cfg.result_dir / "lopo_chbmit_full18_trad23subj.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    sens = df.sensitivity.dropna().values
    fa = df.fa_per_24h.dropna().values
    tot_seiz = int(df.n_seizures.sum()); tot_det = int(df.n_detected.sum())
    tot_fa = int(df.n_false_alarms.sum()); tot_h = float(df.total_hours.sum())
    s_pt, s_lo, s_hi = bootstrap_mean_ci(sens, cfg.n_boot, seed=cfg.seed)
    f_pt, f_lo, f_hi = bootstrap_mean_ci(fa, cfg.n_boot, seed=cfg.seed)
    print("\n===== 传统特征基线（23 受试者 LOPO） =====")
    print(f"受试者 {len(df)}｜发作 {tot_seiz}｜检出 {tot_det}｜汇总灵敏度 {tot_det/max(tot_seiz,1):.3f}")
    print(f"逐例灵敏度均值 {s_pt:.3f} (95%CI {s_lo:.3f}~{s_hi:.3f})")
    print(f"汇总 FA/24h {tot_fa/tot_h*24:.2f}｜逐例 FA/24h 均值 {f_pt:.2f}")
    print(f"逐例精确率均值 {df.precision.mean():.3f}｜F1 均值 {df.f1.mean():.3f}")
    print(f"结果: {out}")


if __name__ == "__main__":
    main()
