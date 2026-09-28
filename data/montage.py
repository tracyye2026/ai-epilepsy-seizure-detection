# -*- coding: utf-8 -*-
"""
montage.py —— 跨数据库导联统一（canonical 18 导双极蒙太奇）

为什么需要这一步？
  CHB-MIT 已经是 18 导**双极**蒙太奇（FP1-F7、F7-T7 …）；
  Siena 是 10-20 **参考**导联（Fp1、F3、C3 …）。
  要让同一个模型跨库复用，必须把两者映射到同一套导联定义上：
      · 若源库本身就有同名的双极导联 -> 直接取用；
      · 若源库是参考导联          -> 用相邻电极相减构造同名双极导联；
      · 若某一导联无法构造          -> 置零，并用 presence 掩码标记。

这是"跨库外部验证"能否成立的方法学关键，建议在论文"资料与方法"中明确写出。
"""
from __future__ import annotations

import re
from typing import List, Tuple

# ---------------- 标准 18 导双极蒙太奇（CHB-MIT 默认顺序） ---------------- #
CANONICAL_18: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ",  "CZ-PZ",
]

# 电极命名别名（旧 10-20 命名 -> 新命名）
ELECTRODE_ALIAS = {
    "T7": ["T7", "T3"], "P7": ["P7", "T5"],
    "T8": ["T8", "T4"], "P8": ["P8", "T6"],
    "FZ": ["FZ", "FZ"], "CZ": ["CZ", "CZ"], "PZ": ["PZ", "PZ"],
    "FP1": ["FP1", "FP1"], "FP2": ["FP2", "FP2"],
    "F3": ["F3"], "F4": ["F4"], "C3": ["C3"], "C4": ["C4"],
    "P3": ["P3"], "P4": ["P4"], "O1": ["O1"], "O2": ["O2"],
    "F7": ["F7"], "F8": ["F8"],
}

# 明确非 EEG 通道（排除，避免混入心电/呼吸/肌电）
NON_EEG_TOKENS = {"ECG", "EKG", "EMG", "RESP", "SPO2", "PLETH", "ART", "ABDO", "CHIN",
                  "EOG", "DC", "TRIG", "STIM", "MARKER", "EVENT", "ANN", "STATUS", "TEMP",
                  # Siena 额外的非脑电生理通道：HR=心率、PULSE、OSAT 等
                  "HR", "PULSE", "OSAT", "SAO2", "ETCO2", "AIRFLOW", "THOR", "SNORE",
                  "POSITION", "LIGHT", "OXIM", "CAPNO"}


# ---------------- 通道名归一化 ---------------- #
def norm_name(name: str) -> str:
    s = str(name).upper().strip()
    s = re.sub(r"-REF$|-AVG$|\.+$", "", s)
    s = s.replace("EEG ", "").replace("EEG", "")
    s = re.sub(r"[\s_]+", "", s)
    # CHB-MIT 对同一双极导联重复记录时会加序号后缀（T8-P8-0 / T8-P8-1），
    # 剥掉尾部 "-数字" 才能匹配到标准名 T8-P8，否则该导联会被误判为缺失。
    s = re.sub(r"-\d+$", "", s)
    return s


def is_non_eeg(name: str) -> bool:
    n = norm_name(name)
    return any(tok in n for tok in NON_EEG_TOKENS)


# ---------------- 建立映射方案 ---------------- #
def plan_channel_mapping(ch_names: List[str],
                         canonical: List[str] = CANONICAL_18
                         ) -> Tuple[List[Tuple[str, int, int]], List[int]]:
    """
    返回:
      plan     : 每个 canonical 导联的取数方式
                 ("direct", i, -1) 直接取第 i 通道
                 ("diff",   i, j) 取第 i 通道减第 j 通道
                 ("zero",  -1, -1) 缺失
      presence : 0/1，标记该导联是否真实存在
    """
    norm = [norm_name(c) for c in ch_names]
    idx: dict = {}
    for i, n in enumerate(norm):
        idx.setdefault(n, i)

    plan, presence = [], []
    for pair in canonical:
        full = norm_name(pair)
        if full in idx:                      # 已是同名双极导联
            plan.append(("direct", idx[full], -1)); presence.append(1); continue
        a, b = pair.split("-")
        ia = next((idx[norm_name(x)] for x in ELECTRODE_ALIAS.get(a, [a])
                   if norm_name(x) in idx), -1)
        ib = next((idx[norm_name(x)] for x in ELECTRODE_ALIAS.get(b, [b])
                   if norm_name(x) in idx), -1)
        if ia >= 0 and ib >= 0:
            plan.append(("diff", ia, ib)); presence.append(1)
        else:
            plan.append(("zero", -1, -1)); presence.append(0)
    return plan, presence


def apply_mapping(data: "object", plan) -> "object":
    """data: (n_ch, T) ndarray -> (n_canonical, T)"""
    import numpy as np
    out = np.zeros((len(plan), data.shape[1]), dtype=data.dtype)
    for k, (mode, i, j) in enumerate(plan):
        if mode == "direct":
            out[k] = data[i]
        elif mode == "diff":
            out[k] = data[i] - data[j]
    return out


# ---------------- 通道精简子集（消融实验） ---------------- #
# 索引对应 CANONICAL_18
CHANNEL_SETS = {
    "full18":   list(range(18)),
    "temporal8": [0, 1, 2, 3, 12, 13, 14, 15],   # 双侧颞-枕链
    "central4":  [5, 6, 9, 10],                  # F3-C3 / C3-P3 / F4-C4 / C4-P4
    "temporal4": [1, 2, 13, 14],                 # F7-T7 / T7-P7 / F8-T8 / T8-P8
    "cz2":       [16, 17],                       # 中线 FZ-CZ / CZ-PZ
}


def subset_names(key: str) -> List[str]:
    return [CANONICAL_18[i] for i in CHANNEL_SETS[key]]


if __name__ == "__main__":
    demo = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
            "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6",
            "Fz", "Cz", "ECG"]
    plan, pres = plan_channel_mapping(demo)
    for name, (mode, i, j), p in zip(CANONICAL_18, plan, pres):
        src = ""
        if mode == "direct":
            src = demo[i]
        elif mode == "diff":
            src = f"{demo[i]}-{demo[j]}"
        print(f"{name:8s} -> {src if p else '[缺失]':20s} presence={p}")
