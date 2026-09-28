# -*- coding: utf-8 -*-
"""redraw_fig45.py —— 从 recalib_cache 预测缓存免重训重绘论文图 4 / 图 5（23 折口径）

背景：work/figures/roc_pr.png（图4）与 per_patient.png（图5）由旧 24 折 LOPO 运行
生成（2026-09-21），早于 chb21→chb01 合并的 23 折重跑（2026-09-22/23），
图内标注（AUC=0.956）与正文新口径（0.907）不一致。

做法：
  图4  汇总全部 *__test_*.npz（chb01 文件含 chb01+chb21 全部记录）的逐采样点
       概率/标签，按 2 s 步长（256 采样点）取窗中心点构成窗级等效序列，
       池化后调用 utils.viz.plot_roc_pr（与 train.py L309 同一池化口径）。
  图5  直接用 23 折主终点逐例结果 recalibB_per_patient_b10p0.csv（列名与
       plot_per_patient 默认参数一致）。

用法：python redraw_fig45.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.viz import plot_per_patient, plot_roc_pr  # noqa: E402

CACHE = ROOT / "work" / "recalib_cache"
RESULTS = ROOT / "work" / "results" / "lopo23"
FIG = ROOT / "work" / "figures"

STEP_SEC = 2  # 窗级等效采样步长（s），与分析窗步长一致


def load_window_level():
    """读取全部测试折 NPZ，返回池化的窗级等效 (y, p)。"""
    ys, ps = [], []
    files = sorted(CACHE.glob("*__test_*.npz"))
    if not files:
        raise FileNotFoundError(f"未找到测试折缓存: {CACHE}")
    for f in files:
        d = np.load(f, allow_pickle=True)
        idxs = sorted({int(k.split("_")[0]) for k in d.keys()})
        for i in idxs:
            p = d[f"{i}_p"].astype(float)
            y = d[f"{i}_y"].astype(int)
            fs = float(np.asarray(d[f"{i}_fs"]).ravel()[0])
            step = int(round(STEP_SEC * fs))
            if step <= 1 or step > len(p):
                raise ValueError(f"{f.name} 记录 {i}: 步长异常 step={step}")
            off = step // 2  # 窗中心采样点
            if off + step * ((len(p) - off) // step) > len(p):
                pass  # 末尾不足一窗的部分自然截断
            ys.append(y[off::step])
            ps.append(p[off::step])
    return np.concatenate(ys), np.concatenate(ps), len(files)


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    # ---- 图4：池化窗级 ROC/PR（23 折测试预测） ----
    y, p, n_files = load_window_level()
    print(f"图4 输入: {n_files} 个测试折文件, 窗级等效样本 {len(y):,} (发作窗 {int(y.sum()):,})")
    # 逐折均值 AUC（正文"窗级 AUC 平均 0.907"的口径）取自主结果表 sample_auc 列
    per_fold = pd.read_csv(ROOT / "work" / "results" /
                           "lopo_chbmit_full18_seizurenet_23subj.csv", encoding="utf-8-sig")
    mean_auc = float(per_fold["sample_auc"].mean())
    note = f"逐折均值 AUC = {mean_auc:.3f}（正文口径）"
    print(f"逐折均值 AUC = {mean_auc:.4f}（{len(per_fold)} 折）")
    plot_roc_pr(y, p, FIG / "roc_pr", title="CHB-MIT · LOPO (23 受试者)", note=note)
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"池化窗级 AUC = {roc_auc_score(y, p):.4f}  AUPRC = {average_precision_score(y, p):.4f}")

    # ---- 图5：逐例事件级灵敏度 / FA/24h（策略 B 主终点） ----
    csv = RESULTS / "recalibB_per_patient_b10p0.csv"
    df = pd.read_csv(csv, encoding="utf-8-sig")
    df = df.sort_values("patient").reset_index(drop=True)
    print(f"图5 输入: {csv.name}, {len(df)} 名受试者")
    plot_per_patient(df, FIG / "per_patient")
    print(f"图4/图5 已重绘并覆盖 {FIG}\\roc_pr.(png|svg) 与 per_patient.(png|svg)")


if __name__ == "__main__":
    main()
