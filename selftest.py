# -*- coding: utf-8 -*-
"""
selftest.py —— 全流程自检（不依赖任何真实数据库）

用途：在下载 40 GB 的 CHB-MIT 之前，先用**合成的 EDF 数据**跑通
      「预处理 → 留一患者法训练 → 事件级评估」整条流水线，
      确认环境与代码没有问题，避免浪费几天时间在错误的配置上。

本脚本自带一个极简 EDF 写入器（纯 Python，无需 pyedflib）：
      生成 3 位"虚拟患者"共 6 段 8 分钟脑电，其中埋入节律性发作样放电。

运行：
    python selftest.py
    python selftest.py --epochs 3 --folds 2
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SELF_DATA = HERE / "_selftest" / "data_raw"
SELF_WORK = HERE / "_selftest" / "work"

CH_REF = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
          "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "Fz", "Cz"]


# ------------------------------------------------------------------ #
# 极简 EDF 写入器（EDF 16-bit，1 秒/数据记录）
# ------------------------------------------------------------------ #
def _pad(s: str, n: int) -> bytes:
    return s.encode("ascii", "replace")[:n].ljust(n, b" ")


def _fmt8(x: float) -> str:
    """把数值格式化为不超过 8 字符（EDF 头字段定长 8 字节）。"""
    for fmt in ("%.2f", "%.1f", "%.1e"):
        s = fmt % x
        if len(s) <= 8:
            return s
    return ("%.1e" % x)[:8]


def write_edf(path: Path, data: np.ndarray, sfreq: int, ch_names: list,
              patient: str = "X"):
    """data: (n_ch, n_samples)，物理单位 µV。"""
    n_ch, n_s = data.shape
    spr = int(sfreq)
    n_rec = n_s // spr
    data = data[:, : n_rec * spr]
    pmin = data.min(axis=1)
    pmax = data.max(axis=1)
    rng = np.where(pmax - pmin < 1e-9, 1.0, pmax - pmin)
    dmin, dmax = -32768, 32767

    h = _pad("0", 8)                      # 版本号字段固定 8 字节
    h += _pad(patient, 80) + _pad("selftest", 80)
    h += _pad("01.01.26", 8) + _pad("00.00.00", 8)
    h += _pad(str(256 + 256 * n_ch), 8)
    h += b" " * 44
    h += _pad(str(n_rec), 8) + _pad("1", 8) + _pad(str(n_ch), 4)
    # EDF 头部是"按字段分块"存储：先导联名 ×ns，再换能器 ×ns，再单位 ×ns …
    h += b"".join(_pad(nm, 16) for nm in ch_names)          # 导联名
    h += b"".join(_pad("", 80) for _ in ch_names)           # 换能器类型
    h += b"".join(_pad("uV", 8) for _ in ch_names)          # 物理单位
    h += b"".join(_pad(_fmt8(pmin[i]), 8) for i in range(n_ch))   # 物理最小值
    h += b"".join(_pad(_fmt8(pmax[i]), 8) for i in range(n_ch))   # 物理最大值
    h += b"".join(_pad(str(dmin), 8) for _ in ch_names)     # 数字最小值
    h += b"".join(_pad(str(dmax), 8) for _ in ch_names)     # 数字最大值
    h += b"".join(_pad("", 80) for _ in ch_names)           # 预滤波
    h += b"".join(_pad(str(spr), 8) for _ in ch_names)      # 每记录采样点数
    h += b"".join(b" " * 32 for _ in ch_names)              # 保留字段
    assert len(h) == 256 + 256 * n_ch, f"header {len(h)}"

    dig = np.round((data - pmin[:, None]) / rng[:, None] * (dmax - dmin) + dmin)
    dig = np.clip(dig, dmin, dmax).astype("<i2")
    rec = dig.reshape(n_ch, n_rec, spr).transpose(1, 0, 2).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(h)
        f.write(rec.tobytes())


# ------------------------------------------------------------------ #
# 合成 EEG：背景噪声 + 发作期节律性放电（定位在颞区）
# ------------------------------------------------------------------ #
def synth_record(sfreq=256, duration=480.0, n_seiz=2, seed=0):
    rng = np.random.default_rng(seed)
    n_s = int(sfreq * duration)
    t = np.arange(n_s) / sfreq
    n_ch = len(CH_REF)
    # 1/f 样背景
    X = rng.normal(0, 1, (n_ch, n_s))
    for i in range(n_ch):
        b, a = 0.2, 0.96
        for _ in range(3):
            X[i] = np.convolve(X[i], [b], mode="same") + a * np.roll(X[i], 1)
    X = X / (np.abs(X).max() + 1e-6) * 20.0
    # 慢波漂移
    X += 8 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3, (n_ch, 1)) * t)[:, :]
    X += 5 * rng.normal(0, 1, (n_ch, n_s))

    seizures = []
    for k in range(n_seiz):
        dur = float(rng.uniform(25, 45))
        st = float(rng.uniform(20, max(21.0, duration - dur - 20)))
        seizures.append((st, st + dur))
        i0, i1 = int(st * sfreq), int((st + dur) * sfreq)
        tt = t[i0:i1] - st
        # 3 Hz -> 1.5 Hz 演化的节律性放电
        freq = np.linspace(3.0, 1.5, len(tt))
        phase = 2 * np.pi * np.cumsum(freq) / sfreq
        discharge = np.sin(phase) * (1 + 0.5 * np.sin(2 * np.pi * 0.6 * tt))
        amp = 120 * np.hanning(len(tt)) ** 0.3
        w = {"T3": 1.0, "T4": 0.9, "F7": 0.5, "F8": 0.45,
             "T5": 0.4, "T6": 0.35, "C3": 0.2, "C4": 0.2, "Fz": 0.25, "Cz": 0.25}
        for name, coef in w.items():
            j = CH_REF.index(name)
            X[j, i0:i1] += amp * discharge * coef
        # 发作后抑制
        i2 = min(n_s, i1 + int(10 * sfreq))
        X[:, i1:i2] *= 0.5
    return X.astype(np.float64), sorted(seizures)


def make_dataset(root: Path, n_patients=3, n_records=2, sfreq=256, duration=480.0):
    if root.exists():
        shutil.rmtree(root)
    lines_all = []
    for pi in range(1, n_patients + 1):
        pdir = root / f"chb{pi:02d}"
        pdir.mkdir(parents=True, exist_ok=True)
        lines = [f"Patient: chb{pi:02d}", ""]
        for ri in range(1, n_records + 1):
            X, seizures = synth_record(sfreq, duration, n_seiz=2, seed=pi * 100 + ri)
            rec = f"chb{pi:02d}_{ri:02d}"
            write_edf(pdir / f"{rec}.edf", X, sfreq, CH_REF, patient=f"chb{pi:02d}")
            lines += [f"File Name: {rec}.edf",
                      f"File Start Time: {ri:02d}:00:00",
                      f"File End Time: {ri:02d}:08:00",
                      f"Number of Seizures in File: {len(seizures)}"]
            for s, e in seizures:
                lines += [f"Seizure Start Time: {int(s)} seconds",
                          f"Seizure End Time: {int(e)} seconds"]
            lines.append("")
        (pdir / f"chb{pi:02d}-summary.txt").write_text("\n".join(lines), encoding="utf-8")
        lines_all += lines
    print(f"合成数据已生成: {root}（{n_patients} 例 × {n_records} 段 × {duration/60:.0f} min）")


# ------------------------------------------------------------------ #
def run(cmd: list):
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit(f"命令失败: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="全流程自检")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--keep", action="store_true", help="保留合成数据与中间结果")
    args = ap.parse_args()

    print("=" * 72)
    print("全流程自检：合成数据 -> 预处理 -> 留一患者法训练 -> 事件级评估")
    print("=" * 72)
    make_dataset(SELF_DATA)

    py = sys.executable
    run([py, "preprocess.py", "--dataset", "chbmit",
         "--data_root", str(SELF_DATA), "--work_root", str(SELF_WORK)])
    run([py, "train.py", "--dataset", "chbmit",
         "--data_root", str(SELF_DATA), "--work_root", str(SELF_WORK),
         "--epochs", str(args.epochs), "--folds", str(args.folds),
         "--batch_size", "32", "--formats", "png"])

    print("\n" + "=" * 72)
    print("自检通过 ✓  环境、依赖与代码链路均正常。")
    print("下一步：申请 PhysioNet 认证账号后运行 "
          "python data/download.py --dataset chbmit --data_root ./data_raw")
    print("=" * 72)
    if not args.keep:
        shutil.rmtree(HERE / "_selftest", ignore_errors=True)
        print("已清理临时合成数据。")


if __name__ == "__main__":
    main()
