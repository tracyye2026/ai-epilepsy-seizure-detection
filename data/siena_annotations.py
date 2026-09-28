# -*- coding: utf-8 -*-
"""
Siena Scalp EEG 数据库（physionet.org/content/siena-scalp-eeg/1.0.0）发作标注解析

与 CHB-MIT 的关键差异（这也是本项目外部验证环节最容易出错的地方）：

1. 标注文件是 Seizures-list-PNxx.txt，不在 RECORDS 索引里（需从 SHA256SUMS 反推）
2. 时间不是"秒偏移"，而是**绝对时钟** HH.MM.SS / HH:MM:SS（分隔符混用两种）
3. 每例参数都存在**跨零点**的长程记录（记录可长达 13 h），
   因此必须用 Registration start time 作基准换算，并对负值回绕 +24 h
4. 一个物理 EDF 文件可能包含多个连续"注册段"（如 PN10-4.5.6.edf 含 3 段），
   每段各有自己的 Registration start；按文件换算偏移时需要特别处理
5. 起止时间字段可能带括号批注，例如：
       Seizure start time: 15.43.53 (CLINICAL ONSET); 15.43.59 (ELECTRIC ONSET)
       Seizure end time: 11.41.04 opure 11.40.43

输出结构：
    { file_name: { "segments": [...], "seizures": [(start_sec, end_sec), ...] } }

校验依据：dataset 根目录的 subject_info.csv 中 number_seizures 列，
        可与本模块统计的发作次数逐例比对。
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RE_FILE = re.compile(r"^\s*File name:\s*(\S+)", re.I)
RE_REG_START = re.compile(r"^\s*Registration start time:\s*(.+)", re.I)
RE_REG_END = re.compile(r"^\s*Registration end time:\s*(.+)", re.I)
RE_SZ_HEAD = re.compile(r"^\s*Seizure\s*n\s*(\d+)\s*(\([^)]*\))?\s*:?", re.I)
# 起止时间有两种写法：'Seizure start time:' 与 'Start time:'（PN01 属后者）
RE_START = re.compile(r"^\s*(?:Seizure\s+)?Start\s+time\s*:\s*(.+)", re.I)
RE_END = re.compile(r"^\s*(?:Seizure\s+)?End\s+time\s*:\s*(.+)", re.I)
RE_CLOCK = re.compile(r"(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})")

DAY = 24 * 3600.0


def _to_sec(text: str) -> Optional[float]:
    """从可能带批注的字符串中取出第一个时钟时间并转为秒

    例： '15.43.53 (CLINICAL ONSET); 15.43.59 (ELECTRIC ONSET)' -> 571*60+53
        '11.41.04 opure 11.40.43'                              -> 取第一个
    若字符串中标注了 ELECTRIC ONSET，优先取电生理起始（与实际脑电改变同步）。
    """
    if not text:
        return None
    # 原始文本存在数字间误插空格的情况（如 PN10 的 '1 6.49.25' 实为 16.49.25），
    # 先把数字之间的空白压掉，否则会把 16 点误读成 6 点。
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    # 电生理优先：标注区分了临床起始与电生理起始时，取电生理
    m = re.search(r"(\d{1,2}[.:]\d{1,2}[.:]\d{1,2})\s*\(ELECTRIC", text, re.I)
    if m:
        text_body = m.group(1)
    else:
        m0 = RE_CLOCK.search(text)
        if not m0:
            return None
        text_body = m0.group(0)
    h, mi, s = (int(x) for x in re.split(r"[.:]", text_body))
    if h > 23 or mi > 59 or s > 59:
        return None
    return h * 3600 + mi * 60 + s


def _offset(event_sec: float, ref_sec: float) -> float:
    """相对 ref 的偏移秒数；跨零点时回绕一天"""
    d = event_sec - ref_sec
    if d < 0:
        d += DAY
    return d


def parse_patient_file(path: Path) -> dict:
    """
    解析单个 Seizures-list-PNxx.txt

    返回 { edf文件名: {"segments":[(reg_start_sec, reg_end_sec)],
                        "seizures":[(start_sec, end_sec)],
                        "labels":[str]} }
    其中 start_sec 是相对**该段** registration start 的秒偏移。
    若同一文件存在多个注册段，segments 会有多项，换算整体文件偏移时需注意顺序。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, dict] = {}

    cur_file = None
    cur_reg = None
    cur_label = None
    for line in text.splitlines():
        m = RE_FILE.search(line)
        if m:
            fname = Path(m.group(1)).name
            # 同一个物理文件被反复列出时（如 PN12-1.2.edf 内含多个注册段），
            # 后续段往往不再重复写 Registration start time，需要继承上一段的基准。
            # 只有切换到**另一个文件**时才清空基准。
            if fname != cur_file:
                cur_file = fname
                cur_reg = None
                out.setdefault(cur_file, {"segments": [], "seizures": [], "labels": []})
            continue
        m = RE_REG_START.search(line)
        if m and cur_file:
            cur_reg = _to_sec(m.group(1))
            out[cur_file]["segments"].append(cur_reg)
            continue
        m = RE_SZ_HEAD.search(line)
        if m:
            cur_label = m.group(1)
            continue
        m = RE_START.search(line)
        if m and cur_file and cur_reg is not None:
            t = _to_sec(m.group(1))
            if t is not None:
                out[cur_file]["seizures"].append([_offset(t, cur_reg), None])
                out[cur_file]["labels"].append(cur_label)
            continue
        m = RE_END.search(line)
        if m and cur_file and cur_reg is not None:
            t = _to_sec(m.group(1))
            if t is not None and out[cur_file]["seizures"]:
                out[cur_file]["seizures"][-1][1] = _offset(t, cur_reg)
            continue

    # 清理：丢弃缺少起始或结束的残缺条目，转为 tuple
    for fname, info in list(out.items()):
        pairs = [(float(a), float(b)) for a, b in info["seizures"] if a is not None and b is not None]
        keep = [i for i, (a, b) in enumerate(info["seizures"]) if a is not None and b is not None]
        info["seizures"] = pairs
        info["labels"] = [info["labels"][i] for i in keep]
        info.pop("segments", None) if False else None
    return out


def _canon_name(name: str, known: set) -> str:
    """
    标注文件里的文件名与磁盘上的真实文件名可能不一致：
    实测 PN06 的标注写成了 'PNO6-1.edf'——把数字 0 写成了字母 O。
    此类笔误若不修正，会静默丢失整段发作标注。
    """
    if name in known:
        return name
    stem = Path(name).stem
    # 实测 PN11 的标注写成了 'PN11-.edf'（分段编号空缺），需先去掉尾部连字符，
    # 否则下面的"省去分段编号"回退匹配会被 "-" not in stem 条件挡掉，静默丢失发作。
    stem = re.sub(r"[-_]+$", "", stem)
    cand = stem.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    for suf in (Path(name).suffix, ".edf"):
        c = cand + suf
        if c in known:
            return c
    lower = {k.lower(): k for k in known}
    if (cand + Path(name).suffix).lower() in lower:
        return lower[(cand + Path(name).suffix).lower()]

    # 标注里可能省去分段编号：'PN01.edf' 实际对应 'PN01-1.edf'
    pid = re.match(r"([A-Za-z]+[0-9OolI]*)", stem)
    if pid and "-" not in stem:
        pid0 = pid.group(1).replace("O", "0").replace("o", "0")
        cands = sorted(k for k in known
                       if k.startswith(pid0 + "-") and k.lower().endswith(".edf"))
        if len(cands) == 1:
            return cands[0]
    return name


def load_dataset_annotations(root: Path,
                             known_files: set = None) -> Dict[str, List[Tuple[float, float]]]:
    """
    扫描整个 Siena 目录，返回 {edf文件名: [(start_sec, end_sec), ...]}

    known_files 用于修正标注中的文件名笔误；缺省时自动读取根目录下的 RECORDS。
    """
    root = Path(root)
    if known_files is None:
        known_files = set()
        rf = root / "RECORDS"
        if rf.exists():
            known_files = {Path(l.strip()).name
                           for l in rf.read_text(encoding="utf-8",
                                                 errors="replace").splitlines()
                           if l.strip()}
    merged: Dict[str, List[Tuple[float, float]]] = {}
    for p in sorted(root.glob("PN*/Seizures-list-PN*.txt")):
        for fname, info in parse_patient_file(p).items():
            fname = _canon_name(fname, known_files)
            merged.setdefault(fname, []).extend(info["seizures"])
    for k in merged:
        merged[k] = sorted(merged[k])
    return merged


def read_subject_info(root: Path) -> dict:
    """读取 subject_info.csv -> {patient_id: {...}}"""
    import csv
    root = Path(root)
    f = root / "subject_info.csv"
    out = {}
    if not f.exists():
        return out
    with open(f, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("patient_id") or "").strip()
            if pid:
                out[pid] = {k.strip(): (v or "").strip() for k, v in row.items()}
    return out


if __name__ == "__main__":
    import sys
    from collections import defaultdict

    r = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"F:\研究数据库\physionet_siena_v1.0.0_20260915")
    ann = load_dataset_annotations(r)
    info = read_subject_info(r)

    print(f"Siena 解析结果（目录 {r.name}）")
    print("-" * 62)
    print(f"{'患者':6s} {'标注发作':>8s} {'csv 记录':>8s} {'含发作文件':>10s}  一致性")
    per = defaultdict(int)
    files_per = defaultdict(set)
    for fn, segs in ann.items():
        pid = fn.split("-")[0]
        per[pid] += len(segs)
        if segs:
            files_per[pid].add(fn)
    total = 0
    for pid in sorted(info):
        n_csv = int(info[pid].get("number_seizures") or 0)
        n_par = per.get(pid, 0)
        total += n_par
        flag = "OK" if n_csv == n_par else f"差异 {n_par - n_csv:+d}"
        print(f"{pid:6s} {n_par:>8d} {n_csv:>8d} {len(files_per.get(pid,())):>10d}  {flag}")
    print("-" * 62)
    print(f"合计：解析到 {total} 次发作，分布在 {sum(len(v) for v in ann.values() if v)} 个文件中")
    print(f"       csv 记录总数：{sum(int(v.get('number_seizures') or 0) for v in info.values())}")
