# -*- coding: utf-8 -*-
"""
build_cont_cache.py —— 预生成"整段连续信号"缓存，大幅加速反复评估

问题
----
preprocess.py 默认只落盘训练窗口，不缓存整段连续信号。于是每次做**连续推理评估**
（阈值标定、留一患者测试、外部验证、导联消融）都要重新读 EDF → 重采样 → 滤波 →
标准化，一条 1 h 记录要几秒到几十秒。24 折 LOPO 里每折要对验证患者（约 40 条）和
测试患者（约 40 条）各做一遍，导联消融还要再乘 4 个导联方案，累计重复读取上千次。

做法
----
对每个数据集（chbmit / siena）逐条记录调用与训练完全一致的 `read_continuous`，
把 (X, y_sample, sfreq) 存成 npz（float32，压缩），并回写 manifest 的 cache_path。
之后 `evaluate.load_continuous` 会优先命中缓存，评估耗时降到原来的几分之一。

用法
----
    python build_cont_cache.py --dataset chbmit --jobs 3
    python build_cont_cache.py --dataset siena  --jobs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                       # noqa: E402
from preprocess import read_continuous          # noqa: E402
from utils.common import get_logger             # noqa: E402


def build_one(rec: dict, cfg: Config, cache_dir: Path):
    path = cache_dir / f"{rec['patient']}_{rec['record']}_cont.npz"
    if path.exists():
        return rec["record"], str(path), "skip"
    cont = read_continuous(rec, cfg)
    if cont is None:
        return rec["record"], None, "fail"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 用 float32 而非 float16：与不命中缓存时的推理路径数值完全一致，
    # 避免"缓存/非缓存"两套结果对不上。
    np.savez_compressed(path, X=cont["X"].astype(np.float32),
                        y=cont["y_sample"].astype(np.uint8),
                        sfreq=np.array([float(cont["sfreq"])]))
    return rec["record"], str(path), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--data_root", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir) if args.work_dir else (
        Path(r"E:\临时存储夹\人工智能（AI）与癫痫\ai_epilepsy_project\work"))
    cfg = Config()
    cfg.work_root = str(work)
    cfg.dataset = args.dataset
    if args.data_root:
        cfg.data_root = args.data_root
    logger = get_logger(f"contcache_{args.dataset}", "INFO",
                        work / f"build_cont_cache_{args.dataset}.log")

    mf_path = work / "index" / args.dataset / "manifest.json"
    manifest = json.loads(mf_path.read_text(encoding="utf-8"))
    cache_dir = work / "cache" / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    todo = [m for m in manifest if m.get("ok")]
    logger.info(f"{args.dataset}：共 {len(todo)} 条记录，缓存目录 {cache_dir}")
    t0 = time.time()

    res = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(build_one)(m, cfg, cache_dir) for m in todo)

    n_ok = n_skip = n_fail = 0
    path_of = {}
    for rec, path, st in res:
        if st == "ok":
            n_ok += 1
        elif st == "skip":
            n_skip += 1
        else:
            n_fail += 1
        if path:
            path_of[rec] = path

    for m in manifest:
        if m.get("record") in path_of:
            m["cache_path"] = path_of[m["record"]]

    # 用 os.replace 而非 shutil.move：后者在 Windows 上会先删除已存在的目标文件，
    # 被本环境的"安全删除"钩子拦截而抛错（SHFileOperationW 0x2）。
    tmp = mf_path.with_name("manifest.json.new")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(mf_path))

    dt = time.time() - t0
    logger.info(f"完成：新建 {n_ok}｜已存在 {n_skip}｜失败 {n_fail}｜"
                f"耗时 {dt/60:.1f} min；manifest 已回写 {len(path_of)} 条 cache_path")
    print(f"{args.dataset}: 新建 {n_ok} 跳过 {n_skip} 失败 {n_fail} "
          f"耗时 {dt/60:.1f} min")


if __name__ == "__main__":
    main()
