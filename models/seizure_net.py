# -*- coding: utf-8 -*-
"""
seizure_net.py —— 轻量化发作检测模型

设计目标（面向基层部署）：
  · 参数量 ~0.15 M，单条 4 s 18 导窗口在 CPU 上推理 < 5 ms；
  · 深度可分离卷积(EEGNet 思想) 提取时-空特征，参数与算力都远低于标准 CNN；
  · 双向 GRU + 时间注意力池化，捕捉发作期演化的时序动态；
  · 同时提供 ResNet-1D 基线模型用于方法学对照。

输入: (B, C, T)，C=导联数(18/8/4/2)，T=win_sec × sfreq(默认 4 s × 128 Hz = 512)
输出: (B,) 未归一化的 logit
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
class DepthwiseSeparableBlock(nn.Module):
    """深度可分离卷积块：逐通道时域卷积 + 1×1 通道融合 + 池化。"""
    def __init__(self, in_ch, out_ch, kernel=15, pool=4, dropout=0.3):
        super().__init__()
        pad = kernel // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel, padding=pad, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.pool = nn.AvgPool1d(pool) if pool and pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.elu(self.bn1(self.depthwise(x)))
        x = F.elu(self.bn2(self.pointwise(x)))
        return self.drop(self.pool(x))


class TemporalAttention(nn.Module):
    """时间注意力池化：让模型自动关注发作期最关键的片段。"""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        score = self.w(torch.tanh(x)).squeeze(-1)      # (B, T)
        attn = torch.softmax(score, dim=1)
        return (attn.unsqueeze(-1) * x).sum(1), attn   # (B, D), (B, T)


class SeizureNet(nn.Module):
    def __init__(self, n_channels: int = 18, n_timesteps: int = 512,
                 base_filters: int = 32, hidden: int = 64, gru_layers: int = 1,
                 dropout: float = 0.3, use_attention: bool = True):
        super().__init__()
        f = base_filters
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, f, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(f), nn.ELU(),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(f, f, kernel=15, pool=4, dropout=dropout),      # T -> T/4
            DepthwiseSeparableBlock(f, f * 2, kernel=15, pool=4, dropout=dropout),  # -> T/16
            DepthwiseSeparableBlock(f * 2, f * 2, kernel=15, pool=4, dropout=dropout),  # -> T/64
        )
        self.gru = nn.GRU(f * 2, hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.use_attention = use_attention
        if use_attention:
            self.attn = TemporalAttention(hidden * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self._feat_cache = None

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)          # (B, 2f, T/64)
        return x.permute(0, 2, 1)   # (B, T', 2f)

    def forward(self, x, return_attn=False):
        h = self.forward_features(x)         # (B, T', 2f)
        h, _ = self.gru(h)                   # (B, T', 2*hidden)
        if self.use_attention:
            pooled, attn = self.attn(h)
        else:
            pooled, attn = h.mean(1), None
        logit = self.head(pooled).squeeze(-1)
        return (logit, attn) if return_attn else logit


# ------------------------------------------------------------------ #
class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel=7, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False), nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False), nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class ResNet1DBaseline(nn.Module):
    """方法学对照基线：纯卷积残差网络 + 全局平均池化。"""
    def __init__(self, n_channels=18, n_timesteps=512, base_filters=32,
                 hidden=64, dropout=0.3, **kw):
        super().__init__()
        f = base_filters
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, f, 7, padding=3, bias=False), nn.BatchNorm1d(f), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(ResBlock1D(f), nn.AvgPool1d(4))
        self.up1 = nn.Conv1d(f, f * 2, 1)
        self.stage2 = nn.Sequential(ResBlock1D(f * 2), nn.AvgPool1d(4))
        self.up2 = nn.Conv1d(f * 2, f * 4, 1)
        self.stage3 = nn.Sequential(ResBlock1D(f * 4), nn.AvgPool1d(4))
        self.stage4 = nn.Sequential(ResBlock1D(f * 4), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(f * 4, 1))

    def forward(self, x, return_attn=False):
        x = self.stem(x)
        x = self.up1(self.stage1(x))
        x = self.up2(self.stage2(x))
        x = self.stage3(x)
        x = self.stage4(x).squeeze(-1)
        return (self.head(x).squeeze(-1), None) if return_attn else self.head(x).squeeze(-1)


# ------------------------------------------------------------------ #
def build_model(cfg) -> nn.Module:
    n_t = int(round(cfg.win_sec * cfg.sfreq))
    common = dict(n_channels=cfg.n_channels, n_timesteps=n_t,
                  base_filters=cfg.base_filters, hidden=cfg.hidden,
                  dropout=cfg.dropout)
    if cfg.model == "resnet1d":
        return ResNet1DBaseline(**common)
    return SeizureNet(gru_layers=cfg.gru_layers, use_attention=cfg.use_attention, **common)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """处理极端类别不平衡（发作窗口通常 < 10%）。"""
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha, self.gamma, self.pos_weight = alpha, gamma, pos_weight

    def forward(self, logit, target):
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none",
                                                 pos_weight=self.pos_weight)
        p = torch.sigmoid(logit)
        pt = torch.where(target == 1, p, 1 - p)
        a = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (a * (1 - pt) ** self.gamma * bce).mean()


if __name__ == "__main__":
    for name, m in [("SeizureNet", SeizureNet()), ("ResNet1D", ResNet1DBaseline())]:
        x = torch.randn(8, 18, 512)
        y, attn = m(x, return_attn=True)
        print(f"{name:12s} params={count_parameters(m):,}  out={tuple(y.shape)}  attn={None if attn is None else tuple(attn.shape)}")
