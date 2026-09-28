# -*- coding: utf-8 -*-
"""calc_brier.py —— 从 recalib_cache 预测缓存计算窗级 Brier 分数（校准证据）

三审报告 5.3 D 项建议补充校准分析。本脚本从 23 折测试预测（窗级等效采样，
与 redraw_fig45.py 同一口径）计算：
  · 池化 Brier 分数与零基准（恒定预测患病率）比较 → Brier 技巧分数；
  · 逐折 Brier 分布 → 跨患者概率尺度漂移的直接证据。

用法：python calc_brier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "work" / "recalib_cache"
STEP_SEC = 2  # 窗级等效采样步长（s），与分析窗步长一致


def load_window_level():
    ys, ps = [], []
    for f in sorted(CACHE.glob("*__test_*.npz")):
        d = np.load(f, allow_pickle=True)
        idxs = sorted({int(k.split("_")[0]) for k in d.keys()})
        fy, fp = [], []
        for i in idxs:
            p = d[f"{i}_p"].astype(float)
            y = d[f"{i}_y"].astype(int)
            fs = float(np.asarray(d[f"{i}_fs"]).ravel()[0])
            step = int(round(STEP_SEC * fs))
            off = step // 2
            fy.append(y[off::step])
            fp.append(p[off::step])
        ys.append(np.concatenate(fy))
        ps.append(np.concatenate(fp))
    return ys, ps


def main():
    ys, ps = load_window_level()
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    brier_pool = float(np.mean((p - y) ** 2))
    prev = float(y.mean())
    brier_null = float(np.mean((prev - y) ** 2))
    per_fold = [float(np.mean((pp - yy) ** 2)) for yy, pp in zip(ys, ps)]
    print(f"池化 Brier = {brier_pool:.4f}（窗级正例率 {prev:.4f}）")
    print(f"零基准 Brier（恒定预测患病率）= {brier_null:.4f}")
    print(f"Brier 技巧分数 = {1 - brier_pool / brier_null:.1f}")
    print(f"逐折 Brier: 均值 {np.mean(per_fold):.4f}，"
          f"最小 {np.min(per_fold):.4f}，最大 {np.max(per_fold):.4f}（{len(per_fold)} 折）")


if __name__ == "__main__":
    main()
