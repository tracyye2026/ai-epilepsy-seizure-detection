# -*- coding: utf-8 -*-
"""
preprocess.py —— EEG 读取、标准化、滤波、导联统一、窗口化与缓存

流程（对应论文"资料与方法 - 数据预处理"）：
  EDF 读取 -> 剔除非 EEG 通道 -> 重采样至 128 Hz -> 0.5~45 Hz 带通
  -> 映射到标准 18 导双极蒙太奇 -> 基于发作间期的稳健 Z 分数标准化
  -> 4 s 窗 / 2 s 步长滑窗 -> 发作期/发作间期标签 -> 负样本按比例抽样 -> 存盘

输出：
  work/index/{dataset}/manifest.json   记录级索引（患者、发作区间、缓存路径…）
  work/windows/{dataset}/*.npz         训练用窗口（float16，省磁盘）
  work/cache/{dataset}/*.npz           可选：整段连续信号缓存（--cache_continuous）

用法：
  python preprocess.py --dataset chbmit --data_root ./data_raw
  python preprocess.py --dataset siena  --data_root ./data_raw --cache_continuous
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, add_config_args, config_from_args          # noqa: E402
from data.annotations import discover_records, summarize, save_records  # noqa: E402
from data.montage import CANONICAL_18, apply_mapping, is_non_eeg, plan_channel_mapping  # noqa: E402
from utils.common import get_logger                                    # noqa: E402


# ------------------------------------------------------------------ #
# 1. 读取 EDF 并做信号级预处理
# ------------------------------------------------------------------ #
def read_and_filter(edf_path, cfg, logger=None):
    """
    返回 (raw, 报错信息)。失败时 raw=None。

    实现要点（重要）：
      Siena 库存在超长记录（PN01-1 达 13.5 h @ 512 Hz，35 通道），
      一次性 load_data 需要 6.5 GB(float64)，在本机内存下必然 MemoryError。
      因此这里采用**流式分块**策略：
        只读取头信息 -> 分 5 min 段按需 get_data -> 立即做导联映射与重采样
        -> 拼接为 18 导 @128 Hz 后再整体带通滤波。
      峰值内存从数 GB 降到 ~200 MB 量级，且两库统一走同一条代码路径。
    """
    import mne
    from scipy.signal import butter, sosfiltfilt, resample_poly
    from fractions import Fraction

    try:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR",
                                      infer_types=True)
        except TypeError:  # 老版本 mne 无 infer_types 参数
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        # 非 EEG 通道标记为 misc，便于剔除
        non = [ch for ch in raw.ch_names if is_non_eeg(ch)]
        if non:
            raw.set_channel_types({c: "misc" for c in non})
        try:                                    # mne >= 1.5 推荐写法
            raw.pick(picks="eeg", exclude=[])
        except TypeError:                       # 兼容旧版
            raw.pick_types(eeg=True, misc=False, exclude=[])
        if len(raw.ch_names) == 0:
            return None, "未找到 EEG 通道"

        # ---- 导联映射方案（先算出，避免加载后才发现不可用）----
        plan, presence = plan_channel_mapping(raw.ch_names, CANONICAL_18)
        if sum(presence) < cfg.min_present_channels:
            return None, f"可用导联仅 {sum(presence)} 个 < {cfg.min_present_channels}"

        sf0 = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        if n_times <= 0:
            return None, "记录为空"

        # ---- 重采样比例（有理数近似，保证 resample_poly 因子为整数）----
        # 注意方向：resample_poly(x, up, down) 的输出率 = 输入率 * up/down，
        # 因此 up/down 必须等于 **目标/源**（= cfg.sfreq / sf0），
        # 写成 sf0/cfg.sfreq 会变成升采样（4 倍数据膨胀 -> 直接撑爆内存）。
        if abs(sf0 - cfg.sfreq) > 1e-6:
            fr = Fraction(float(cfg.sfreq) / sf0).limit_denominator(64)
            up, down = fr.numerator, fr.denominator
        else:
            up, down = 1, 1

        # ---- 分块读取 + 映射 + 重采样 ----
        seg_sec = 300.0            # 单块 5 分钟（原始采样率下）
        pad_sec = 4.0              # 前后各多读 4 s 作为边界上下文，处理后裁掉
        seg_n = max(1, int(seg_sec * sf0))
        pad_n = int(pad_sec * sf0)
        parts = []
        for s in range(0, n_times, seg_n):
            a = max(0, s - pad_n)
            b = min(n_times, s + seg_n + pad_n)
            d = raw.get_data(start=a, stop=b)                  # (n_ch, n) float64
            blk = apply_mapping(d, plan).astype(np.float32)     # -> (18, n)
            if down != 1 or up != 1:
                blk = resample_poly(blk, up, down, axis=-1).astype(np.float32)
            # 裁掉上下文（换算到目标采样率下的点数）
            cut_l = int(round((s - a) * cfg.sfreq / sf0))
            cut_r = int(round((b - min(n_times, s + seg_n)) * cfg.sfreq / sf0))
            end = blk.shape[1] - (cut_r if cut_r > 0 else 0)
            parts.append(blk[:, cut_l:end])
        X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        del parts

        # ---- 带通滤波（逐通道 sosfiltfilt：零相位，且避免整块 float64 复制撑爆内存）----
        nyq = float(cfg.sfreq) / 2.0
        hi = min(float(cfg.bandpass_hi), nyq * 0.94)
        lo = max(float(cfg.bandpass_lo), 0.01)
        if hi <= lo:
            return None, f"滤波频带非法 ({lo}~{hi} Hz)"
        sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        for k in range(X.shape[0]):
            X[k] = sosfiltfilt(sos, X[k]).astype(np.float32)

        # ---- 封装回 RawArray，保持与上层接口一致 ----
        info = mne.create_info(ch_names=CANONICAL_18, sfreq=float(cfg.sfreq),
                               ch_types="eeg", verbose="ERROR")
        raw_out = mne.io.RawArray(X, info, verbose="ERROR")
        # mne 的 Info 不允许直接设自定义键，临时字段须放进 info["temp"]
        raw_out.info["temp"] = {"_presence": presence, "_src_sfreq": sf0}
        return raw_out, None
    except MemoryError:
        return None, "MemoryError（记录过长，分块后仍超出内存）"
    except Exception as e:  # noqa
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def robust_zscore(X: np.ndarray, interictal_mask: np.ndarray) -> np.ndarray:
    """
    以"发作间期"为基线做稳健标准化：
        z = (x - median) / (IQR / 1.349)
    相比均值-标准差，中位数/IQR 对发作期高幅放电与伪迹不敏感。
    """
    if interictal_mask is not None and interictal_mask.sum() > 64:
        base = X[:, interictal_mask]
    else:
        base = X
    med = np.median(base, axis=1, keepdims=True)
    q75 = np.percentile(base, 75, axis=1, keepdims=True)
    q25 = np.percentile(base, 25, axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349
    scale[~np.isfinite(scale) | (scale <= 0)] = np.std(base, axis=1, keepdims=True)[
        ~np.isfinite(scale) | (scale <= 0)]
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return (X - med) / scale


# ------------------------------------------------------------------ #
# 2. 记录级处理：连续信号 + 窗口
# ------------------------------------------------------------------ #
def read_continuous(rec: dict, cfg: Config, logger=None):
    """
    读取一条记录并返回预处理后的**整段连续信号**（供评估/推理使用）。
    返回 dict{X(18,T) float32, y_sample(T,) uint8, sfreq, seizures, presence, duration, start}
    失败返回 None。
    """
    sfreq = float(cfg.sfreq)
    raw, err = read_and_filter(rec["edf"], cfg, logger)
    if raw is None:
        if logger:
            logger.warning(f"跳过 {rec['record']}: {err}")
        return None

    # read_and_filter 已完成"导联映射 + 重采样 + 带通"，这里直接取 18 导矩阵
    presence = list((raw.info.get("temp") or {}).get("_presence")
                    or [1] * len(CANONICAL_18))
    X = raw.get_data().astype(np.float32)          # (18, T)
    T = X.shape[1]
    duration = T / sfreq

    # ---- 采样点级发作标签 ----
    y_sample = np.zeros(T, dtype=np.uint8)
    seizures = []
    for s, e in rec.get("seizures", []):
        s2, e2 = max(0.0, float(s)), min(duration, float(e))
        if e2 <= s2:
            continue
        y_sample[int(s2 * sfreq): int(e2 * sfreq)] = 1
        seizures.append((s2, e2))

    # ---- 稳健标准化 + 截断 ----
    X = robust_zscore(X, (y_sample == 0))
    np.clip(X, -cfg.clip_z, cfg.clip_z, out=X)
    X[~np.isfinite(X)] = 0.0
    return {"X": X, "y_sample": y_sample, "sfreq": sfreq, "seizures": seizures,
            "presence": presence, "duration": float(duration), "record": rec["record"],
            "patient": rec["patient"], "edf": rec["edf"]}


def process_record(rec: dict, cfg: Config, logger=None,
                   save_windows=True, windows_dir=None, cache_dir=None) -> dict:
    cont = read_continuous(rec, cfg, logger)
    if cont is None:
        return {"record": rec["record"], "patient": rec["patient"], "ok": False,
                "error": "读取或通道不足", "n_windows": 0, "n_ictal": 0,
                "seizures": rec.get("seizures", [])}

    X, y_sample = cont["X"], cont["y_sample"]
    sfreq, T = cont["sfreq"], X.shape[1]
    duration, seizures, presence = cont["duration"], cont["seizures"], cont["presence"]

    # ---- 窗口化 ----
    win = int(round(cfg.win_sec * sfreq))
    hop = int(round(cfg.hop_sec * sfreq))
    if T < win:
        return {"record": rec["record"], "patient": rec["patient"], "ok": False,
                "error": f"记录过短 ({duration:.1f}s)", "n_windows": 0,
                "n_ictal": 0, "seizures": seizures}

    W, y_win, starts = make_windows(X, y_sample, win, hop, cfg.ictal_overlap)
    meta = {
        "patient": rec["patient"], "record": rec["record"], "edf": rec["edf"],
        "sfreq": sfreq, "duration": float(duration), "win": win, "hop": hop,
        "seizures": seizures, "n_seizures": len(seizures),
        "presence": presence, "channels": CANONICAL_18,
        "n_windows_total": int(len(y_win)),
        "n_ictal_total": int(y_win.sum()),
    }

    # ---- 负样本抽样（训练集用） ----
    if save_windows and windows_dir is not None:
        ictal_idx = np.where(y_win == 1)[0]
        # 发作结束后 guard 期内的窗口不纳入负样本
        guard = int(cfg.guard_after_sec * sfreq)
        ext = y_sample.copy()
        if guard > 0 and len(seizures):
            for s, e in seizures:
                ext[int(e * sfreq): min(T, int(e * sfreq) + guard)] = 1
        csum = np.concatenate([[0], np.cumsum(ext)])
        ov_guard = csum[starts + win] - csum[starts]
        guard_flag = (ov_guard > 0) & (y_win == 0)
        inter_idx = np.where((y_win == 0) & (~guard_flag))[0]

        n_ictal = len(ictal_idx)
        if n_ictal > 0:
            n_inter = int(cfg.neg_pos_ratio * n_ictal)
        else:
            # 无发作记录：按固定条数抽样负样本。
            # 若此处取 0，则全部无发作记录都不会进入训练集，模型学不到
            # "长时程正常脑电"的分布，跨患者泛化时假报警会爆炸（实测 FA/24h >200）。
            n_inter = int(getattr(cfg, "neg_windows_per_record", 120))
        n_inter = min(n_inter, len(inter_idx))
        rng = np.random.default_rng(cfg.seed)
        sel = np.concatenate([
            ictal_idx,
            rng.choice(inter_idx, size=n_inter, replace=False) if n_inter else np.array([], int),
        ])
        sel.sort()
        Ws = W[sel]
        ys = y_win[sel]
        windows_dir.mkdir(parents=True, exist_ok=True)
        wpath = windows_dir / f"{rec['patient']}_{rec['record']}.npz"
        np.savez_compressed(wpath, X=Ws.astype(np.float16), y=ys.astype(np.uint8))
        meta.update(windows_path=str(wpath), n_windows=int(len(sel)), n_ictal=int(n_ictal))
    else:
        meta.update(windows_path=None, n_windows=0, n_ictal=int(y_win.sum()))

    # ---- 可选：缓存整段连续信号（加速反复评估） ----
    if cfg.cache_continuous and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / f"{rec['patient']}_{rec['record']}_cont.npz"
        np.savez_compressed(cpath, X=X.astype(np.float16), y=y_sample.astype(np.uint8),
                            sfreq=np.array([sfreq]))
        meta["cache_path"] = str(cpath)
    meta["ok"] = True

    # ---- 落盘记录级元信息（含参数指纹，供增量处理判断是否需要重跑） ----
    if windows_dir is not None:
        meta["fingerprint"] = param_fingerprint(cfg)
        (windows_dir / f"{rec['patient']}_{rec['record']}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


FP_KEYS = ("sfreq", "bandpass_lo", "bandpass_hi", "notch_freq", "win_sec", "hop_sec",
           "ictal_overlap", "guard_after_sec", "neg_pos_ratio", "clip_z",
           "min_present_channels", "neg_windows_per_record")


def param_fingerprint(cfg: Config) -> dict:
    """
    影响预处理结果的关键参数指纹。
    用途：分批次下载数据后重跑 preprocess 时，指纹未变的记录直接复用，
    不必重复耗时（CHB-MIT 全量预处理通常 2~6 小时）。
    """
    return {k: getattr(cfg, k) for k in FP_KEYS if hasattr(cfg, k)}


def make_windows(X: np.ndarray, y_sample: np.ndarray, win: int, hop: int,
                 ictal_overlap: float = 0.5):
    """滑窗切分。返回 (W(n,C,win) float32, y(n,) uint8, starts(n,))"""
    C, T = X.shape
    starts = np.arange(0, T - win + 1, hop, dtype=np.int64)
    csum = np.concatenate([[0], np.cumsum(y_sample)])
    ov = csum[starts + win] - csum[starts]
    y = (ov / float(win) >= ictal_overlap).astype(np.uint8)
    W = np.zeros((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        W[i] = X[:, s:s + win]
    return W, y, starts


# ------------------------------------------------------------------ #
# 3. 构建索引
# ------------------------------------------------------------------ #
def build_manifest(cfg: Config, logger=None, patients=None) -> list:
    cfg.ensure_dirs()
    records = discover_records(cfg.data_dir, cfg.dataset, patients=patients)
    if logger:
        s = summarize(records)
        logger.info(f"发现 {s['n_patients']} 例患者 / {s['n_records']} 条记录 / "
                    f"{s['n_seizures']} 次发作（发作总时长 {s['seizure_hours']:.2f} h）")

    windows_dir = cfg.windows_dir
    cache_dir = cfg.work_dir / "cache" / cfg.dataset
    windows_dir.mkdir(parents=True, exist_ok=True)
    fp = param_fingerprint(cfg)
    out, t0, n_skip = [], time.time(), 0
    for i, rec in enumerate(records, 1):
        stem = f"{rec['patient']}_{rec['record']}"
        mpath, wpath = windows_dir / f"{stem}.meta.json", windows_dir / f"{stem}.npz"
        # ---- 增量处理：参数未变且产物已存在则直接复用 ----
        if cfg.skip_existing and mpath.exists() and wpath.exists():
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
                if m.get("ok") and m.get("fingerprint") == fp:
                    out.append(m); n_skip += 1
                    continue
            except Exception:
                pass
        m = process_record(rec, cfg, logger, save_windows=True,
                           windows_dir=windows_dir, cache_dir=cache_dir)
        out.append(m)
        if logger and (i % 10 == 0 or i == len(records)):
            el = time.time() - t0
            logger.info(f"  已处理 {i}/{len(records)} 条记录，累计 {el/60:.1f} min，"
                        f"预计剩余 {el/i*(len(records)-i)/60:.1f} min")
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    mf = cfg.index_dir / "manifest.json"
    mf.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if logger:
        if n_skip:
            logger.info(f"增量处理：复用已处理记录 {n_skip} 条，"
                        f"新处理 {len(records) - n_skip} 条")
        ok = [m for m in out if m.get("ok")]
        logger.info(f"索引已写入 {mf}；成功 {len(ok)} 条，"
                    f"发作窗口 {sum(m.get('n_ictal', 0) for m in ok)} 个 "
                    f"（实际保留 {sum(m.get('n_windows', 0) for m in ok)} 个窗口含负样本）")
    return out


def main():
    ap = argparse.ArgumentParser(description="EEG 预处理与窗口化")
    add_config_args(ap)
    ap.add_argument("--patients", nargs="*", default=None, help="只处理指定患者，如 chb01 chb02")
    args = ap.parse_args()
    cfg = config_from_args(args)
    logger = get_logger("preprocess", cfg.log_level, cfg.work_dir / "preprocess.log")
    logger.info("配置: " + cfg.summary())
    build_manifest(cfg, logger, patients=args.patients)


if __name__ == "__main__":
    main()
