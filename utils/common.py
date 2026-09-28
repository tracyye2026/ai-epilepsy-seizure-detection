# -*- coding: utf-8 -*-
"""通用工具：随机种子、日志、计时、设备选择。"""
from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_logger(name: str = "eeg", level: str = "INFO", log_file=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@contextmanager
def timer(desc: str = "", logger=None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    msg = f"{desc} 耗时 {dt:.1f}s" if desc else f"耗时 {dt:.1f}s"
    if logger:
        logger.info(msg)


def select_device(pref: str = "auto"):
    """返回 torch.device。无 GPU 时自动退回 CPU。"""
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        print("[warn] 未检测到 CUDA，自动退回 CPU")
    return torch.device("cpu")


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def human_readable(n: float, unit: str = "B") -> str:
    for u in ["", "K", "M", "G", "T"]:
        if abs(n) < 1024:
            return f"{n:.1f}{u}{unit}"
        n /= 1024
    return f"{n:.1f}P{unit}"


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
