# -*- coding: utf-8 -*-
"""
annotations.py —— 公共数据库目录发现与"发作区间"标注解析

支持：
  · CHB-MIT Scalp EEG Database  -> 每个患者目录下的 chbXX-summary.txt
  · Siena Scalp EEG Database    -> 每个患者目录下的 Seizures-list-PNXX.txt

解析结果统一为：
  Record = {
     "patient": "chb01",          # 患者编号（留一法划分的最小单位）
     "record" : "chb01_01",       # 记录编号
     "edf"    : Path(...),        # EDF 文件路径
     "seizures": [(start_sec, end_sec), ...]   # 相对本记录起点的发作区间
  }
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# ------------------------------------------------------------------ #
# 正则表达式（同时兼容 CHB-MIT 与 Siena 两种文本格式）
# ------------------------------------------------------------------ #
RE_FILE = re.compile(r"File\s*Name\s*:\s*(\S+\.edf)", re.I)
RE_NSEIZ = re.compile(r"Number\s+of\s+[Ss]eizures\s+in\s+[Ff]ile\s*:\s*(\d+)", re.I)
RE_START = re.compile(r"Seizure\s*\d*\s*[Ss]tart\s*[Tt]ime\s*:\s*(\d+)\s*seconds?", re.I)
RE_END = re.compile(r"Seizure\s*\d*\s*[Ee]nd\s*[Tt]ime\s*:\s*(\d+)\s*seconds?", re.I)
RE_CHB_PATIENT = re.compile(r"^(chb\d{2})$", re.I)
RE_SIENA_PATIENT = re.compile(r"^(PN\d{2})$", re.I)


def _parse_summary_txt(txt_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """返回 {文件名: [(start_sec, end_sec), ...]}"""
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    out: Dict[str, List[Tuple[float, float]]] = {}
    cur_file, n_seiz, starts, ends = None, 0, [], []

    def flush():
        """把当前文件收集到的起止时间配对入库。"""
        if not cur_file:
            return
        # 少数标注只写了起始时间（缺结束时间），按 30 s 补齐，避免整条发作被丢弃
        ee = list(ends) + [s + 30.0 for s in starts[len(ends):]]
        for s, e in zip(starts, ee):
            if e > s:
                out[cur_file].append((s, e))
        starts.clear(); ends.clear()

    for line in text.splitlines():
        m = RE_FILE.search(line)
        if m:
            flush()
            cur_file = Path(m.group(1)).name
            out.setdefault(cur_file, [])
            n_seiz = 0
            continue
        m = RE_NSEIZ.search(line)
        if m:
            n_seiz = int(m.group(1))
            continue
        m = RE_START.search(line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = RE_END.search(line)
        if m:
            ends.append(float(m.group(1)))
    flush()
    return out


RE_DUP_SUFFIX = re.compile(r"^(?P<stem>.+)\.v\d+$", re.I)


def drop_duplicate_edfs(found: List[Path]) -> List[Path]:
    """
    剔除下载器因"重名不覆盖"策略产生的副本文件（xxx.v2.edf / xxx.v3.edf …）。

    为什么要剔除：
      这些副本与原文件是**同一段脑电**，若一并纳入，会导致同一条记录被统计两次——
      发作次数、记录时长、假报警数全部翻倍，事件级指标与 FA/24h 直接失真。
    判定规则：若去掉 .vN 后缀后存在同名文件，则副本丢弃。
    """
    stems = {p.stem for p in found}
    out = []
    for p in found:
        m = RE_DUP_SUFFIX.match(p.stem)
        if m and m.group("stem") in stems:
            continue
        out.append(p)
    return out


def discover_records(data_root, dataset: str = "chbmit",
                     patients=None) -> List[Dict]:
    """
    扫描 data_root，返回 Record 列表。
    patients: 可选，限定患者编号列表（如 ["chb01","chb02"]），便于分批处理。
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {root}")

    pat_re = RE_SIENA_PATIENT if dataset.lower() == "siena" else RE_CHB_PATIENT
    records: List[Dict] = []

    # 1) 找到所有患者目录
    if dataset.lower() == "siena":
        subject_dirs = [d for d in sorted(root.rglob("*"))
                        if d.is_dir() and pat_re.match(d.name)]
    else:
        subject_dirs = [d for d in sorted(root.rglob("*"))
                        if d.is_dir() and pat_re.match(d.name)]

    if not subject_dirs:  # 兼容"扁平"解压结构
        subject_dirs = [root]

    # 2) Siena 走专用解析器：标注是绝对时钟 + 多注册段，通用正则解析不到
    siena_map: Dict[str, List[Tuple[float, float]]] = {}
    if dataset.lower() == "siena":
        try:
            from .siena_annotations import load_dataset_annotations
            known = {p.name for p in root.rglob("*.edf")}
            siena_map = load_dataset_annotations(root, known_files=known)
        except Exception as e:  # 解析失败不应静默成"零发作"，需显式报错
            raise RuntimeError(f"Siena 发作标注解析失败: {type(e).__name__}: {e}")

    for sdir in subject_dirs:
        patient = sdir.name if pat_re.match(sdir.name) else sdir.name
        # 3) 解析该患者目录下所有标注文本
        ann_map: Dict[str, List[Tuple[float, float]]] = {}
        if dataset.lower() == "siena":
            for k, v in siena_map.items():
                if k.upper().startswith(patient.upper()):
                    ann_map[k] = v
        else:
            for txt in sorted(sdir.rglob("*.txt")):
                name = txt.name.lower()
                if ("summary" in name) or ("seizures" in name) or ("seizure" in name):
                    for k, v in _parse_summary_txt(txt).items():
                        ann_map[k] = v
        # 4) 逐个 EDF 建立记录（先剔除 .vN 副本，避免同一段脑电被统计两次）
        for edf in drop_duplicate_edfs(sorted(sdir.rglob("*.edf"))):
            rec_id = edf.stem
            seiz = ann_map.get(edf.name, [])
            if not seiz:  # 某些标注以"不带后缀"或大小写不一致给出
                for k, v in ann_map.items():
                    if Path(k).stem.lower() == rec_id.lower():
                        seiz = v
                        break
            records.append({
                "patient": patient,
                "record": rec_id,
                "edf": str(edf),
                "seizures": [tuple(map(float, s)) for s in seiz],
            })

    if patients:
        want = {p.lower() for p in patients}
        records = [r for r in records if r["patient"].lower() in want]

    records.sort(key=lambda r: (r["patient"], r["record"]))
    return records


def summarize(records: List[Dict]) -> Dict:
    """输出数据库概览（论文表 1 用）。"""
    pat = {}
    for r in records:
        p = pat.setdefault(r["patient"], {"n_records": 0, "n_seizures": 0, "seizure_sec": 0.0})
        p["n_records"] += 1
        p["n_seizures"] += len(r["seizures"])
        p["seizure_sec"] += sum(e - s for s, e in r["seizures"])
    return {
        "n_patients": len(pat),
        "n_records": len(records),
        "n_seizures": sum(p["n_seizures"] for p in pat.values()),
        "seizure_hours": sum(p["seizure_sec"] for p in pat.values()) / 3600.0,
        "per_patient": pat,
    }


def save_records(records: List[Dict], path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_records(path) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
