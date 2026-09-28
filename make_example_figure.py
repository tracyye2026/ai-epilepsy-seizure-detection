#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_example_figure.py —— 生成"典型案例检出"图（论文附图）

内容：一段连续脑电（截取若干导联） + 真实发作区间阴影
      + 模型输出的逐样本概率曲线 + 阈值线 + 检出的报警区间。

用法：
  python make_example_figure.py                       # 自动挑一例成功检出
  python make_example_figure.py --patient chb01 --record chb01_03 --center 1080 --span 600
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config                                        # noqa: E402
from evaluate import load_continuous, predict_record             # noqa: E402
from models.seizure_net import build_model                       # noqa: E402
from utils.common import select_device                           # noqa: E402
from utils.metrics import evaluate_continuous                    # noqa: E402
from utils.viz import save_fig                                   # noqa: E402

from data.montage import CANONICAL_18                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--center", type=float, default=None, help="截取窗口中心(s)；默认取发作中心")
    ap.add_argument("--span", type=float, default=600.0, help="截取时长(s)")
    ap.add_argument("--max_ch", type=int, default=6)
    ap.add_argument("--formats", default="png,svg")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config().ensure_dirs()
    device = select_device(cfg.device)
    manifest = json.loads((cfg.index_dir / "manifest.json").read_text(encoding="utf-8"))
    ok = [m for m in manifest if m.get("ok")]

    # ---- 选定记录 ----
    if args.record:
        cand = [m for m in ok if m["record"] == args.record]
    elif args.patient:
        cand = [m for m in ok if m["patient"] == args.patient and m["n_seizures"] > 0]
    else:
        # 自动挑：有发作、模型置信度高的记录
        lopo = cfg.result_dir / "lopo_chbmit_full18.csv"
        pref = []
        if lopo.exists():
            import pandas as pd
            df = pd.read_csv(lopo)
            df = df.sort_values("seizure_coverage", ascending=False)
            pref = [p for p in df["patient"]]
        pool = pref or sorted({m["patient"] for m in ok})
        cand = []
        for p in pool:
            c = [m for m in ok if m["patient"] == p and m["n_seizures"] > 0]
            if c:
                cand = c
                break
    if not cand:
        raise SystemExit("未找到符合条件的记录")
    entry = cand[0]
    patient = entry["patient"]
    print(f"选定记录 {entry['record']}（患者 {patient}，发作 {entry['n_seizures']} 次）")

    # ---- 载入该患者对应的 LOPO 模型 ----
    ck = cfg.ckpt_dir / f"chbmit_full18_{patient}.pt"
    if not ck.exists():
        raise SystemExit(f"未找到模型 {ck}，请先运行 train.py")
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    c2 = Config(**{k: v for k, v in ckpt["cfg"].items() if k in Config().__dict__})
    c2.data_root, c2.work_root = cfg.data_root, cfg.work_root
    thr = float(ckpt.get("threshold", c2.threshold))
    model = build_model(c2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    loaded = load_continuous(entry, c2)
    if loaded is None:
        raise SystemExit("记录读取失败")
    X, y_sample, sfreq, seizures = loaded
    prob, _, _ = predict_record(model, X, c2, device, None)
    ecfg = replace(c2, threshold=thr)
    res = evaluate_continuous(y_sample, prob, sfreq, ecfg)
    alarms = res.get("alarms", []) or []

    # ---- 截取片段 ----
    if args.center is not None:
        center = args.center
    elif seizures:
        center = float(np.mean(seizures[0]))
    else:
        center = len(y_sample) / sfreq / 2
    half = args.span / 2
    i0 = max(0, int((center - half) * sfreq))
    i1 = min(len(y_sample), int((center + half) * sfreq))
    Xs = X[:, i0:i1]
    ys = y_sample[i0:i1]
    ps = prob[i0:i1]
    t = np.arange(Xs.shape[-1]) / sfreq

    # ---- 绘图 ----
    n_ax = args.max_ch + 1
    fig, axes = plt.subplots(n_ax, 1, figsize=(11.5, 1.25 * n_ax + 1.0), sharex=True)
    step = max(1, Xs.shape[0] // args.max_ch)
    scale = np.nanpercentile(np.abs(Xs), 98) + 1e-6
    for i, ax in enumerate(axes[:-1]):
        ch = min(i * step, Xs.shape[0] - 1)
        ax.plot(t, Xs[ch] / scale + i, lw=0.5, color="#1a1a1a")
        ax.set_ylabel(CANONICAL_18[ch], fontsize=7.5, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        m = ys.astype(bool)
        if m.any():
            ax.fill_between(t, i - 0.42, i + 0.42, where=m, color="#E74C3C", alpha=0.28, lw=0)
        for a, b in alarms:
            if b < t[0] + 0 or a > t[-1]:
                continue
            ax.axvspan(max(a, t[0]), min(b, t[-1]), color="#2E86C1", alpha=0.10, lw=0)
    axes[0].set_title(
        f"案例：{entry['record']}（患者 {patient}）连续脑电与模型输出概率。"
        f"红色阴影 = 真实发作；蓝色底纹 = 模型报警；阈值 = {thr:.3f}", fontsize=9.5)

    ax = axes[-1]
    ax.plot(t, ps, lw=1.1, color="#C0392B")
    ax.axhline(thr, ls="--", lw=0.9, color="#7F8C8D")
    ax.text(t[-1], min(thr + 0.03, 0.97), f" 阈值 {thr:.3f}", fontsize=7, color="#7F8C8D",
            ha="right", va="bottom")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("发作概率", fontsize=8)
    ax.set_xlabel("时间 (s，片段内相对时间)", fontsize=8.5)
    for a, b in alarms:
        ax.axvspan(max(a, t[0]), min(b, t[-1]), color="#2E86C1", alpha=0.10, lw=0)
    fig.tight_layout()

    out = Path(args.out) if args.out else (cfg.figure_dir / f"fig_example_{patient}")
    save_fig(fig, out, args.formats.split(","))
    print(f"已导出 {out}.png / .svg")
    print(f"该记录：真实发作 {res['n_seizures']} 次，检出 {res['n_detected']} 次，"
          f"假报警 {res['n_false_alarms']} 次，时长 {res['hours']:.2f} h")


if __name__ == "__main__":
    main()
