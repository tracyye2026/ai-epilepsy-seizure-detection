# -*- coding: utf-8 -*-
"""
dataset.py —— 窗口级数据集与在线数据增强
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_window_arrays(entries, channel_idx=None, max_windows=None, seed=0,
                       max_neg_pos_ratio=None):
    """
    把若干记录的窗口 npz 拼接为一个大数组（float16 存内存，取用时转 float32）。
    entries: manifest 中的记录字典列表
    max_neg_pos_ratio: 负:正 上限。负样本远多于正样本时先按此比例裁剪，
                       再套用 max_windows——否则按比例抽样会连正样本一起砍掉，
                       导致每折正样本只剩一千余个，灵敏度受损。
    """
    Xs, ys = [], []
    for e in entries:
        p = e.get("windows_path")
        if not p or not Path(p).exists():
            continue
        d = np.load(p)
        X, y = d["X"], d["y"]
        if len(y) == 0:
            continue
        if channel_idx is not None:
            X = X[:, list(channel_idx), :]
        Xs.append(X)
        ys.append(y)
    if not Xs:
        return np.zeros((0, 18, 512), dtype=np.float16), np.zeros((0,), dtype=np.uint8)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # 先把 负:正 压到上限（保留全部正样本）
    if max_neg_pos_ratio and max_neg_pos_ratio > 0:
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if len(pos) > 0 and len(neg) > max_neg_pos_ratio * len(pos):
            rng0 = np.random.default_rng(seed)
            keep_neg = rng0.choice(neg, size=int(max_neg_pos_ratio * len(pos)),
                                   replace=False)
            sel0 = np.concatenate([pos, keep_neg])
            sel0.sort()
            X, y = X[sel0], y[sel0]

    if max_windows is not None and len(y) > max_windows:
        rng = np.random.default_rng(seed)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos = min(len(pos), int(max_windows * len(pos) / max(1, len(y))))
        n_neg = max_windows - n_pos
        sel = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, n_neg, replace=False)])
        sel.sort()
        X, y = X[sel], y[sel]
    return X, y


def augment_window(x: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    """在线数据增强：时间平移、加噪、幅度抖动、随机导联失活。"""
    C, T = x.shape
    if cfg.aug_shift > 0:
        k = int(rng.integers(-cfg.aug_shift, cfg.aug_shift + 1))
        if k:
            x = np.roll(x, k, axis=-1)
    if cfg.aug_noise > 0:
        x = x + rng.normal(0, cfg.aug_noise, size=x.shape).astype(x.dtype)
    if cfg.aug_scale > 0:
        x = x * float(rng.uniform(1 - cfg.aug_scale, 1 + cfg.aug_scale))
    if cfg.aug_chan_drop > 0 and C > 2:
        mask = rng.random(C) > cfg.aug_chan_drop
        if mask.sum() >= 2:
            x = x * mask[:, None]
    return x


class WindowDataset(Dataset):
    def __init__(self, entries, channel_idx=None, augment=False, cfg=None,
                 max_windows=None, seed=0):
        self.cfg = cfg
        self.augment = augment
        ratio = getattr(cfg, "max_neg_pos_ratio", None) if cfg is not None else None
        self.X, self.y = load_window_arrays(entries, channel_idx, max_windows, seed,
                                            max_neg_pos_ratio=ratio)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)
        if self.augment and self.cfg is not None:
            x = augment_window(x, self.cfg, self.rng)
        return torch.from_numpy(x), float(self.y[i])

    @property
    def pos_weight(self) -> float:
        n_pos = float((self.y == 1).sum())
        n_neg = float((self.y == 0).sum())
        return (n_neg / n_pos) if n_pos > 0 else 1.0

    def class_counts(self):
        return int((self.y == 1).sum()), int((self.y == 0).sum())
