# -*- coding: utf-8 -*-
"""
viz.py —— 论文用图绘制

所有图片默认同时导出 PNG(300 dpi) 与 **矢量格式 SVG/PDF**，
可直接交付《癫痫杂志》排版老师使用（--formats png,svg,pdf）。
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# 中文字体：优先 Windows 自带雅黑/黑体，退回系统默认
_CJK = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "DejaVu Sans"]
_available = {f.name for f in font_manager.fontManager.ttflist}
for _f in _CJK:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

# 期刊友好配色（色盲安全）
COLORS = ["#1f4e79", "#c0504d", "#4f81bd", "#9bbb59", "#8064a2", "#f79646", "#4bacc6"]


def save_fig(fig, out_base, formats=("png", "svg"), logger=None):
    """保存图片，out_base 不带扩展名。"""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_base.with_suffix("." + fmt)
        fig.savefig(p, bbox_inches="tight", format=fmt)
        paths.append(str(p))
    plt.close(fig)
    if logger:
        logger.info(f"已保存图片: {', '.join(paths)}")
    return paths


def plot_roc_pr(y_true, y_prob, out_base, title="", formats=("png", "svg"), logger=None,
                note=""):
    """图3：ROC 与 PR 曲线（合并为一张双面板图）。note 为附加说明文本（显示于 ROC 图例）。"""
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    if y_true.sum() > 0 and (y_true == 0).sum() > 0:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, color=COLORS[0], lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        axes[1].plot(rec, prec, color=COLORS[1], lw=2, label=f"AUPRC = {auc(rec, prec):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    if note:
        axes[0].plot([], [], " ", label=note)  # 仅图例文本，不画线
    for ax, xl, yl in zip(axes, ["1 - 特异度 (FPR)", "灵敏度 (Recall)"],
                          ["灵敏度 (TPR)", "精确率 (Precision)"]):
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(loc="lower right"); ax.grid(alpha=.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    axes[0].set_title("ROC 曲线"); axes[1].set_title("Precision-Recall 曲线")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return save_fig(fig, out_base, formats, logger)


def plot_per_patient(df, out_base, sens_col="sensitivity", fa_col="fa_per_24h",
                     patient_col="patient", formats=("png", "svg"), logger=None):
    """图4：逐患者灵敏度与 FA/24h（双轴柱状+折线）。"""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(df))
    ax.bar(x - .2, df[sens_col] * 100, width=.4, color=COLORS[0], label="发作检出率 (%)")
    ax.set_ylabel("发作检出率 (%)", color=COLORS[0])
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.bar(x + .2, df[fa_col], width=.4, color=COLORS[1], alpha=.85, label="假阳性/24h")
    ax2.set_ylabel("假阳性数 / 24 h", color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(df[patient_col], rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("患者编号（留一法测试集）")
    ax.grid(axis="y", alpha=.3)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left")
    fig.tight_layout()
    return save_fig(fig, out_base, formats, logger)


def plot_ablation(df, out_base, x_col="n_channels", y_col="sensitivity",
                  hue_col=None, formats=("png", "svg"), logger=None,
                  ylabel="发作检出率", title="通道精简消融实验"):
    """图5：通道数消融曲线（含 95%CI 误差带）。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if hue_col and hue_col in df.columns:
        for i, (k, g) in enumerate(df.groupby(hue_col)):
            g = g.sort_values(x_col)
            ax.errorbar(g[x_col], g[y_col],
                        yerr=[g[y_col] - g.get("ci_low", g[y_col]),
                              g.get("ci_high", g[y_col]) - g[y_col]],
                        marker="o", lw=2, capsize=4, color=COLORS[i % len(COLORS)], label=str(k))
        ax.legend(title=hue_col)
    else:
        g = df.sort_values(x_col)
        ax.errorbar(g[x_col], g[y_col],
                    yerr=[g[y_col] - g.get("ci_low", g[y_col]),
                          g.get("ci_high", g[y_col]) - g[y_col]],
                    marker="o", lw=2, capsize=4, color=COLORS[0])
    ax.set_xlabel("输入导联数"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=.3)
    fig.tight_layout()
    return save_fig(fig, out_base, formats, logger)


def plot_channel_importance(names, values, out_base, formats=("png", "svg"),
                            logger=None, title="导联贡献度（遮挡法）"):
    """图6：导联遮挡后性能下降幅度，反映各导联贡献。"""
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = np.asarray(values)[order]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(range(len(names))[::-1], values, color=COLORS[2])
    ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("遮挡该导联后的性能下降（ΔAUC 或 Δ灵敏度）")
    ax.set_title(title); ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    return save_fig(fig, out_base, formats, logger)


def plot_training_curves(history, out_base, formats=("png", "svg"), logger=None):
    """训练曲线（loss / val AUC / val F1）。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(history.get("train_loss", []), color=COLORS[0]); axes[0].set_title("训练损失")
    axes[1].plot(history.get("val_auc", []), color=COLORS[2]); axes[1].set_title("验证集 AUC")
    axes[2].plot(history.get("val_f1", []), color=COLORS[3]); axes[2].set_title("验证集 F1")
    for ax, yl in zip(axes, ["BCE / Focal loss", "AUC", "F1"]):
        ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.grid(alpha=.3)
    fig.tight_layout()
    return save_fig(fig, out_base, formats, logger)


def plot_detection_example(X_win, prob, y_true, sfreq, out_base,
                           ch_names=None, max_ch=6, formats=("png", "svg"), logger=None):
    """图2/附图：一段连续 EEG + 真实发作区间 + 模型输出概率曲线。"""
    T = X_win.shape[-1]
    t = np.arange(T) / sfreq
    fig, axes = plt.subplots(max_ch + 1, 1, figsize=(12, 1.15 * (max_ch + 1.4)), sharex=True)
    step = max(1, X_win.shape[0] // max_ch)
    scale = np.nanpercentile(np.abs(X_win), 98) + 1e-6
    for i, ax in enumerate(axes[:-1]):
        ch = min(i * step, X_win.shape[0] - 1)
        ax.plot(t, X_win[ch] / scale + i, lw=.5, color="k")
        ax.set_ylabel(ch_names[ch] if ch_names else f"ch{ch}", fontsize=7, rotation=0, ha="right")
        ax.set_yticks([])
        m = y_true.astype(bool)
        if m.any():
            ax.fill_between(t, i - .5, i + .5, where=m, color=COLORS[1], alpha=.25)
    axes[-1].plot(t, prob, lw=1.2, color=COLORS[0])
    axes[-1].axhline(0.5, ls="--", lw=.8, color="gray")
    axes[-1].set_ylabel("预测概率", fontsize=8)
    axes[-1].set_xlabel("时间 (s)")
    axes[-1].set_ylim(0, 1)
    axes[0].set_title("示例：连续 EEG（红色阴影=真实发作）与模型输出概率")
    fig.tight_layout()
    return save_fig(fig, out_base, formats, logger)
