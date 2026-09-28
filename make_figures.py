#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figures.py —— 生成论文中不由训练脚本自动产出的配图

  fig1_flowchart   研究设计与数据流图（含纳入记录数、窗口数、 fold 划分方式）
  fig2_arch        SeizureNet 网络结构示意

其余配图（ROC/PR、逐患者结果、训练曲线、导联消融、外部验证逐患者）
由 train.py / ablation.py / external_validate.py 自动导出至 work/figures/。

用法：
  python make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config                                   # noqa: E402
from utils.viz import save_fig                              # noqa: E402

cfg = Config().ensure_dirs()
FIG = cfg.figure_dir
FMT = ("png", "svg")

C_BOX = "#EAF2F8"
C_BOX2 = "#FDF2E9"
C_BOX3 = "#E8F6F3"
C_EDGE = "#2C3E50"
C_ARROW = "#34495E"


def _box(ax, x, y, w, h, text, fc=C_BOX, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor=C_EDGE, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.45)


def _arrow(ax, x1, y1, x2, y2, label="", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.2, color=C_ARROW, zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label,
                ha="center", va="bottom", fontsize=7.5, color=C_ARROW)


# ------------------------------------------------------------------ #
def fig1_flowchart(n_records_chb=685, n_sz_chb=198,
                   n_pat_chb=23, n_sess_chb=24, n_no_sz_chb=545,
                   n_win=1766780, n_pos=5785,
                   n_records_si=41, n_pat_si=14, n_sz_si=47):
    """纵向线性流程：数据 → 预处理 → 分窗 → 模型 → 双路径验证 → 后处理 → 终点

    数字口径与正文 1.3/2.1 及表 1 一致：23 名受试者（24 个会话，chb21 并入 chb01）、
    685 个记录文件、198 次发作、1 766 780 窗（发作窗 5 785，占 0.33%）、
    545 条无发作记录、留一患者法 23 折（每折 1 测试 + 1 验证 + 21 训练）。
    """
    fig, ax = plt.subplots(figsize=(9.0, 7.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    L, R = 0.02, 0.98
    MID = 0.5

    # ---- Row1 数据来源（左右） ----
    _box(ax, L, 0.885, 0.44, 0.10,
         f"CHB-MIT 头皮脑电数据库（ODC-By 1.0）\n"
         f"{n_pat_chb} 例（{n_sess_chb} 个会话）/ {n_records_chb} 条记录 / {n_sz_chb} 次发作\n"
         f"→ 训练集 + 内部验证", fc=C_BOX, bold=True, fs=8.6)
    _box(ax, 0.54, 0.885, 0.44, 0.10,
         f"Siena 头皮脑电数据库（CC BY 4.0）\n"
         f"{n_pat_si} 例成人 / {n_records_si} 条记录 / {n_sz_si} 次发作\n"
         f"→ 外部验证（不参与训练）", fc=C_BOX, bold=True, fs=8.6)

    # ---- Row2 统一预处理（通栏） ----
    _box(ax, L, 0.755, 0.96, 0.095,
         "统一预处理（两库参数完全一致）：标准 18 导双极蒙太奇 ｜ 重采样 128 Hz ｜ "
         "0.5–45 Hz 零相位带通\n稳健 Z 分数标准化（以发作间期为基准，±10 截断）",
         fc=C_BOX2, fs=8.6)
    _arrow(ax, 0.24, 0.882, 0.24, 0.852)
    _arrow(ax, 0.76, 0.882, 0.76, 0.852)

    # ---- Row3 分窗与训练集构成 ----
    _box(ax, L, 0.615, 0.55, 0.105,
         f"4 s 分析窗 / 步长 2 s（50% 重叠）\n"
         f"发作重叠 ≥50% 记为发作期；发作后 30 s 内窗不计入负样本\n"
         f"→ 共 {n_win:,} 窗（发作期 {n_pos:,}，占 {n_pos / n_win * 100:.2f}%）",
         fc=C_BOX2, fs=8.4)
    _box(ax, 0.61, 0.615, 0.37, 0.105,
         "训练集负样本覆盖全部记录\n"
         f"（含 {n_no_sz_chb} 条无发作记录）\n"
         "负∶正上限 6∶1 ｜ 每折 ≤36 000 窗",
         fc="#FDEDEC", fs=8.4, bold=True)
    _arrow(ax, 0.24, 0.752, 0.24, 0.722)
    _arrow(ax, 0.795, 0.752, 0.795, 0.722)
    _arrow(ax, 0.565, 0.6675, 0.605, 0.6675)

    # ---- Row4 模型（通栏） ----
    _box(ax, L, 0.492, 0.96, 0.088,
         "SeizureNet：深度可分离卷积 ×3（512 → 8 时间步） + 双向 GRU（隐层 64） "
         "+ 时间注意力池化 + 分类头\n参数量 72 386 ｜ 可在普通 CPU 上实时推理",
         fc=C_BOX3, fs=8.6, bold=True)
    _arrow(ax, 0.24, 0.612, 0.24, 0.583)

    # ---- Row5 验证策略（左右） ----
    _box(ax, L, 0.345, 0.44, 0.105,
         "内部验证：留一患者法（23 折）\n"
         "每折 1 例测试 / 1 例验证 / 21 例训练\n"
         "阈值在验证患者上按事件级标定",
         fc=C_BOX3, fs=8.4, bold=True)
    _box(ax, 0.54, 0.345, 0.44, 0.105,
         "外部验证：Siena 全库 14 例\n"
         "直接套用 CHB-MIT 训练所得模型\n"
         "量化跨中心泛化差距",
         fc=C_BOX3, fs=8.4, bold=True)
    _arrow(ax, 0.24, 0.489, 0.24, 0.453)
    _arrow(ax, 0.76, 0.489, 0.76, 0.453)

    # ---- Row6 后处理（通栏） ----
    _box(ax, L, 0.205, 0.96, 0.088,
         "事件级后处理：概率滑动平均 8 s ｜ 间隔 <10 s 的报警合并为一次事件 ｜ "
         "丢弃时长 <15 s 的短事件（外部部署为 10 s）\n"
         "（依据发作时长先验：本组发作时长中位数 39 s，而伪迹多为秒级孤立尖峰）",
         fc=C_BOX2, fs=8.3)
    _arrow(ax, 0.24, 0.342, 0.24, 0.296)
    _arrow(ax, 0.76, 0.342, 0.76, 0.296)

    # ---- Row7 终点 ----
    _box(ax, L, 0.055, 0.96, 0.095,
         "主要终点：事件级灵敏度 ｜ 每 24 小时假报警数（FA/24h）\n"
         "次要终点：窗级 ROC-AUC ｜ 中位检出延迟 ｜ 患者水平一致性",
         fc="#F4ECF7", fs=8.8, bold=True)
    _arrow(ax, MID, 0.202, MID, 0.153)

    save_fig(fig, FIG / "fig1_flowchart", FMT)


# ------------------------------------------------------------------ #
def fig2_arch():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    cols = {
        "in": "#D6EAF8", "conv": "#D1F2EB", "gru": "#FCF3CF",
        "attn": "#FADBD8", "head": "#E8DAEF",
    }
    stages = [
        ("输入\n18 导 × 4 s\n(18 × 512)", cols["in"], 1.0),
        ("卷积干\nConv1d k=7\n+ BN + ELU", cols["conv"], 1.0),
        ("深度可分离块 ×3\nDW Conv k=15 → PW Conv\nBN + ELU + AvgPool + Drop\n512 → 8 时间步", cols["conv"], 1.6),
        ("双向 GRU\n隐层 64 ｜ 1 层\n输出 128 维", cols["gru"], 1.0),
        ("时间注意力\n池化", cols["attn"], 0.8),
        ("分类头\nLayerNorm-Dropout\n-Linear-ReLU-Linear\n输出发作概率", cols["head"], 1.0),
    ]
    x = 0.025
    gap = 0.018
    total_w = 0.95 - gap * (len(stages) - 1)
    widths = [total_w * s[2] / sum(t[2] for t in stages) for s in stages]
    for (text, c, _), w in zip(stages, widths):
        _box(ax, x, 0.18, w, 0.5, text, fc=c, fs=8.5)
        x += w + gap

    # 底部尺度标注
    ax.annotate("", xy=(0.03, 0.12), xytext=(0.30, 0.12),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=C_ARROW))
    ax.text(0.165, 0.075, "时频特征提取", ha="center", fontsize=8, color=C_ARROW)
    ax.annotate("", xy=(0.42, 0.12), xytext=(0.60, 0.12),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=C_ARROW))
    ax.text(0.51, 0.075, "时序建模", ha="center", fontsize=8, color=C_ARROW)
    ax.annotate("", xy=(0.63, 0.12), xytext=(0.95, 0.12),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=C_ARROW))
    ax.text(0.79, 0.075, "决策", ha="center", fontsize=8, color=C_ARROW)

    ax.text(0.5, 0.85, "SeizureNet：18 导输入，参数量 72 386，CPU 可实时推理",
            ha="center", fontsize=10.5, fontweight="bold")
    save_fig(fig, FIG / "fig2_architecture", FMT)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_flowchart()
    fig2_arch()
    print(f"配图已导出至 {FIG}（PNG 300 dpi + SVG）")


if __name__ == "__main__":
    main()
