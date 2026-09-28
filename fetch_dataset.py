# -*- coding: utf-8 -*-
"""
fetch_dataset.py —— 公共脑电数据库稳健下载器（断点续传 / 重试 / 校验 / 分批）

面向大批量下载设计，解决朴素 urlretrieve 的四个致命问题：
  1) 断网后从头重来      -> HTTP Range 断点续传，进度落盘
  2) 偶发错误直接失败    -> 指数退避重试 + 抖动，可配置次数
  3) 下载损坏无法发现    -> SHA256SUMS.txt 逐文件校验，不符自动重下
  4) 大文件无进度可见性  -> 分批 + 实时速度/ETA/百分比

【命名规则】
  根目录：  <目标目录>/<来源>_<数据集>_v<版本>_<YYYYMMDD>/
            例：F:/研究数据库/physionet_chbmit_v1.0.0_20260914/
  目录内部：完全保留 PhysioNet 原始相对路径与文件名（chb01/chb01_01.edf），
            不做任何改名，以保证 preprocess.py 等下游脚本可直接读取。

【命令行示例】
  # 下载 CHB-MIT 与 Siena 全量到 F:/研究数据库
  python fetch_dataset.py --dest "F:/研究数据库" --datasets chbmit siena

  # 只下部分患者（分批由脚本自动按患者目录切分）
  python fetch_dataset.py --dest "F:/研究数据库" --datasets chbmit --patients chb01 chb02

  # 提高并发（带宽好时）
  python fetch_dataset.py --dest "F:/研究数据库" --datasets chbmit --workers 8

  # 重跑：已完整且校验通过的文件自动跳过，只补缺失/损坏的部分
  python fetch_dataset.py --dest "F:/研究数据库" --datasets chbmit siena
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa
    pass

USER_AGENT = "Mozilla/5.0 (research-data-fetcher; epilepsy-eeg-study)"

# ---------------------------------------------------------------- 数据源定义
DATASETS = {
    "chbmit": {
        "slug": "chbmit/1.0.0",
        "version": "1.0.0",
        "title": "CHB-MIT Scalp EEG Database",
        "source": "physionet",
    },
    "siena": {
        "slug": "siena-scalp-eeg/1.0.0",
        "version": "1.0.0",
        "title": "Siena Scalp EEG Database",
        "source": "physionet",
    },
}
BASE = "https://physionet.org/files"

CHUNK = 1 << 20          # 1 MB 读写块
_print_lock = threading.Lock()

# CSV 日志里的中文状态标签
STATUS_CN = {"ok": "成功", "skipped": "跳过（已存在）", "failed": "失败"}


def log(msg: str, endl: str = "\n") -> None:
    with _print_lock:
        sys.stdout.write(msg + endl)
        sys.stdout.flush()


# ---------------------------------------------------------------- 基础 HTTP
SIZE_CACHE: dict = {}       # rel -> 远端大小，减少对服务端的 HEAD 探测请求
CONNECT_JITTER: float = 0.0  # 每次建立新连接前的随机抖动上限（秒）


def _build_opener() -> urllib.request.OpenerDirector:
    """显式禁用系统代理继承导致的偶发失败，并统一 UA。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_OPENER = _build_opener()


def http_open(url: str, headers: dict = None, timeout: int = 60):
    # 弱网/按连接限速的链路上，并发握手突发会大量触发 WinError 10060。
    # 每次建连前插入随机抖动，把握手在时间上摊开。
    if CONNECT_JITTER > 0:
        time.sleep(random.uniform(0.05, CONNECT_JITTER))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return _OPENER.open(req, timeout=timeout)


def fetch_text(url: str, retries: int = 5, timeout: int = 60) -> str:
    last = None
    for attempt in range(1, retries + 1):
        try:
            with http_open(url, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:  # noqa
            last = e
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
    raise RuntimeError(f"获取失败 {url}: {last}")


def remote_size(url: str, retries: int = 5) -> int:
    """取远端文件大小。优先 HEAD，不支持时用 Range 兜底。"""
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}, method="HEAD")
            if CONNECT_JITTER > 0:
                time.sleep(random.uniform(0.05, CONNECT_JITTER))
            with _OPENER.open(req, timeout=60) as r:
                n = int(r.headers.get("Content-Length", -1))
                if n >= 0:
                    return n
                raise ValueError("no Content-Length")
        except Exception as e:  # noqa
            last = e
            try:
                with http_open(url, headers={"Range": "bytes=0-0"}, timeout=60) as r:
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        return int(cr.split("/")[-1])
            except Exception:  # noqa
                pass
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
    raise RuntimeError(f"无法获取远端大小 {url}: {last}")


# ---------------------------------------------------------------- 目标盘预检
def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TB"


def preflight(dest_root: Path, need_bytes: int = 0) -> dict:
    """
    下载前的目标目录体检：是否存在/能否创建/能否写入/剩余空间是否充足。
    返回 dict，含错误与处理建议。
    """
    info = {"path": str(dest_root), "exists": False, "created": False,
            "writable": False, "free": None, "error": "", "hint": ""}
    # 1) 存在性 / 自动创建
    try:
        if not dest_root.exists():
            dest_root.mkdir(parents=True, exist_ok=True)
            info["created"] = True
        info["exists"] = True
    except Exception as e:  # noqa
        info["error"] = f"{type(e).__name__}: {e}"
        info["hint"] = "目标目录无法创建：路径非法、磁盘写保护或卷处于损坏/只读状态。"
        return info
    # 2) 写入探测（写入 -> 读回 -> 删除）
    probe = dest_root / "__write_probe__.tmp"
    try:
        probe.write_bytes(b"ok")
        if probe.read_bytes() != b"ok":
            raise IOError("读回内容与写入不一致")
        # 判定依据是"写入且读回一致"，删除探针文件只是清理，失败不应影响判定
        info["writable"] = True
    except Exception as e:  # noqa
        info["error"] = f"{type(e).__name__}: {e}"
        if isinstance(e, OSError) and getattr(e, "errno", None) == 22:
            info["hint"] = (
                "写入返回 EINVAL（Invalid argument）。Windows 上典型成因是 exFAT/FAT 卷的"
                "“脏标记”置位（fsutil dirty query 显示“已损坏”），系统随即拒绝写入。\n"
                "        处理顺序建议：① 桌面右下角“安全删除硬件”弹出后拔下、重新插拔"
                "（多数情况会自动清除脏标记）；\n"
                "        ② 若仍不行，以管理员身份运行 chkdsk <盘符>: /F ——注意此操作改动文件系统、"
                "存在数据丢失风险，务必先备份盘上重要资料；\n"
                "        ③ 或先用 --dest 指定另一个可写目录，稍后再复制过去。")
        else:
            info["hint"] = "目标目录不可写，请检查权限，或改用其他目录。"
        try:
            probe.unlink(missing_ok=True)
        except Exception:  # noqa
            pass
        return info
    # 清理探针文件：删除失败（如安全删除/回收站拦截）不影响可写性判定
    try:
        probe.unlink(missing_ok=True)
    except Exception:  # noqa
        pass
    # 3) 空间
    import shutil
    _, _, free = shutil.disk_usage(dest_root)
    info["free"] = free
    if need_bytes and free < need_bytes * 1.1:
        info["hint"] = (f"剩余空间不足：预计需要 {human(need_bytes)}，当前可用 {human(free)}"
                        f"（建议保留 10% 余量）。")
    return info


def print_preflight(info: dict) -> bool:
    log("=" * 74)
    log("步骤 1 / 4　目标目录预检")
    log("=" * 74)
    if info["created"]:
        log(f"  目录：{info['path']}  （原不存在，已自动创建）")
    else:
        log(f"  目录：{info['path']}  （已存在）")
    ok = info["writable"]
    log(f"  可写：{'通过' if ok else '不通过'}"
        + (f"  -> {info['error']}" if info["error"] else ""))
    if info["free"] is not None:
        log(f"  可用空间：{human(info['free'])}")
    if info["hint"]:
        for line in info["hint"].split("\n"):
            log(f"  {'建议：' if line is info['hint'].split(chr(10))[0] else '      '}{line.strip()}")
    if not ok:
        log("\n  [中止] 目标盘当前不可写入，下载无法开始。请按上述建议处理后重跑本脚本。")
    return ok


def sha256_file(path: Path, stop=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            if stop and stop.is_set():
                return ""
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 追溯日志
class CsvLogger:
    """
    把每条下载记录追加写入 CSV（带 UTF-8 BOM，Excel 双击打开不乱码）。
    字段：序号 / 完成时间 / 数据集 / 来源地址 / 原始文件名 / 保存路径 /
          大小(字节) / 大小 / 状态 / SHA256 / 重试次数 / 备注
    """
    FIELDS = ["序号", "完成时间", "数据集", "来源地址", "原始文件名",
              "保存路径", "大小(字节)", "大小", "状态", "SHA256", "重试次数", "备注"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                f.write("\ufeff")                      # BOM，供 Excel 正确识别中文
                csv.writer(f).writerow(self.FIELDS)
        else:
            try:
                with open(self.path, "r", encoding="utf-8-sig", newline="") as f:
                    self._seq = max(0, sum(1 for _ in csv.reader(f)) - 1)
            except Exception:  # noqa
                self._seq = 0

    def write(self, **kw) -> int:
        with self._lock:
            self._seq += 1
            size = kw.get("bytes", 0) or 0
            row = {
                "序号": self._seq,
                "完成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "数据集": kw.get("dataset", ""),
                "来源地址": kw.get("url", ""),
                "原始文件名": kw.get("file", ""),
                "保存路径": kw.get("save_path", ""),
                "大小(字节)": size,
                "大小": human(size) if size else "",
                "状态": kw.get("status", ""),
                "SHA256": kw.get("sha256", ""),
                "重试次数": kw.get("attempt", ""),
                "备注": kw.get("note", ""),
            }
            with open(self.path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([row[k] for k in self.FIELDS])
            return self._seq


def _next_seq_file(p: Path) -> Path:
    """为已存在的同名文件生成 name.v2 / name.v3 ... 新路径，绝不覆盖原文件。"""
    i = 2
    while True:
        cand = p.with_name(f"{p.stem}.v{i}{p.suffix}")
        if not cand.exists():
            return cand
        i += 1


# ---------------------------------------------------------------- 单文件下载
def download_file(url: str, dest: Path, expect_size: int, expect_sha: str = None,
                  retries: int = 3, on_conflict: str = "unique",
                  verify: bool = True, timeout: int = 120) -> dict:
    """
    下载单文件，返回 {status, path, bytes, sha256, attempt, note}

    on_conflict:
        unique（默认）同名且内容一致 -> 跳过；同名但内容不一致 -> **追加序号另存**，
                     绝不覆盖已存在的文件
        overwrite          覆盖重新下载
        skip               同名即跳过（不校验内容）
    断点续传：无论哪种策略，未完成的 .part 一律复用续传。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    original = dest
    part = dest.with_suffix(dest.suffix + ".part")
    result = {"url": url, "path": str(dest), "status": "", "bytes": 0,
              "sha256": "", "attempt": 0, "note": ""}

    # ---- 同名文件判定：一致则跳过，不一致则追加序号，绝不覆盖 ----
    if dest.exists() and on_conflict in ("unique", "skip"):
        cur = dest.stat().st_size
        same_size = (expect_size > 0 and cur == expect_size)
        if on_conflict == "skip" and same_size:
            result.update(status="skipped", bytes=cur, note="同名文件已存在，跳过（未校验）")
            return result
        ok = False
        if same_size:
            if verify and expect_sha:
                ok = sha256_file(dest) == expect_sha
            else:
                ok = True
        if ok:
            result.update(status="skipped", bytes=cur,
                          sha256=expect_sha or "", note="同名文件已存在且 SHA256 一致，跳过")
            return result
        dest = _next_seq_file(original)
        part = dest.with_suffix(dest.suffix + ".part")
        old_part = original.with_suffix(original.suffix + ".part")
        if old_part.exists() and not part.exists():
            try:
                old_part.rename(part)      # 把旧断点续传文件接过来，不浪费已下载字节
            except Exception:  # noqa
                pass
        result["path"] = str(dest)
        result["note"] = f"同名文件已存在且内容不一致，另存为 {dest.name}（原文件保留）"

    # ---- 断点续传起点 ----
    start = 0
    if part.exists():
        start = part.stat().st_size
        if start > expect_size > 0:        # 残留异常，重来
            part.unlink()
            start = 0
    elif on_conflict == "overwrite" and dest.exists():
        dest.unlink()

    # ---- 重试循环（上限 retries 次） ----
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={start}-"} if start > 0 else {}
            with http_open(url, headers=headers, timeout=timeout) as resp:
                if start > 0 and resp.status != 206:      # 服务端未返回续传 -> 重来
                    start = 0
                    part.open("wb").close()
                # 未在事前取到远端大小时，从响应头推断总长度（省去一次 HEAD 往返）
                if expect_size <= 0:
                    cr = resp.headers.get("Content-Range", "")
                    if "/" in cr:
                        try:
                            expect_size = int(cr.split("/")[-1])
                        except ValueError:
                            pass
                    if expect_size <= 0:
                        cl = resp.headers.get("Content-Length")
                        if start == 0 and cl:
                            expect_size = int(cl)
                mode = "ab" if start > 0 else "wb"
                with open(part, mode) as f:
                    while True:
                        buf = resp.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
            got = part.stat().st_size
            if expect_size > 0 and got != expect_size:
                raise IOError(f"大小不符：本地 {got} vs 远端 {expect_size}")
            if verify and expect_sha and sha256_file(part) != expect_sha:
                raise IOError("SHA256 校验失败")
            part.replace(dest)
            result.update(status="ok", bytes=got, sha256=expect_sha or "",
                          attempt=attempt,
                          note=(result["note"] or ("续传完成" if start > 0 else "下载完成")))
            return result

        except Exception as e:  # noqa
            last_err = e
            if part.exists() and (expect_size <= 0 or part.stat().st_size <= expect_size):
                start = part.stat().st_size        # 保留已下载部分，下轮续传
            elif part.exists():
                part.unlink()
                start = 0
            if attempt < retries:
                sleep_s = min(2 ** attempt, 30) + random.uniform(0, 2)
                log(f"      [重试 {attempt}/{retries}] {dest.name}: {e}；{sleep_s:.0f}s 后再试")
                time.sleep(sleep_s)

    # ---- 彻底失败（保留 .part，重跑可续传） ----
    result.update(status="failed",
                  bytes=(part.stat().st_size if part.exists() else 0),
                  attempt=retries, note=str(last_err))
    return result


# ---------------------------------------------------------------- 索引获取
def get_index(slug: str) -> tuple:
    """返回 (文件相对路径列表, {rel: sha256})"""
    log(f"  获取文件索引 RECORDS ...")
    records = [l.strip() for l in fetch_text(f"{BASE}/{slug}/RECORDS").splitlines() if l.strip()]
    log(f"    索引文件数：{len(records)}")
    sums = {}
    try:
        txt = fetch_text(f"{BASE}/{slug}/SHA256SUMS.txt")
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                # PhysioNet 格式：<sha256>  ./rel/path  或 <sha256> *rel/path
                sums[parts[-1].lstrip("./").lstrip("*")] = parts[0]
        log(f"    校验和条目：{len(sums)}")
    except Exception as e:  # noqa
        log(f"    [警告] 未获取到 SHA256SUMS.txt（{e}），将只做大小校验")

    # ★ 关键：RECORDS 只列出数据主体（如 *.edf），
    #   而标注文件（chbNN-summary.txt、*.edf.seizures、LICENSE.txt 等）
    #   并不在 RECORDS 中，却存在于 SHA256SUMS.txt。
    #   这里用校验表反推出这部分"隐性文件"，否则数据下完也没有标注，无法训练。
    if sums:
        extra = sorted(set(sums) - set(records))
        if extra:
            log(f"    [发现] 校验表中另有 {len(extra)} 个不在 RECORDS 里的标注/说明文件，一并纳入")
            for r in extra[:5]:
                log(f"           {r}")
            records = records + extra
    return records, sums


# ---------------------------------------------------------------- 主流程
def resolve_folder(dest_root: Path, meta: dict, key: str,
                   reuse: bool = True, explicit: str = None) -> tuple:
    """
    确定数据集保存目录。

    命名规则：<来源>_<数据集>_v<版本>_<YYYYMMDD>
    由于全量数据往往需要跨天续传，日期目录不能每天新建，否则换天后重跑会在
    新目录里从头重下。因此默认 **复用已存在的同名前缀目录**（取日期后缀最大的那个）。
    返回 (目录, 是否复用了已有目录)。
    """
    if explicit:
        return dest_root / explicit, True
    prefix = f"{meta['source']}_{key}_v{meta['version']}"
    if reuse:
        cands = sorted([c for c in dest_root.glob(prefix + "_*") if c.is_dir()])
        if cands:
            return cands[-1], True
    return dest_root / f"{prefix}_{datetime.now().strftime('%Y%m%d')}", False


def run_dataset(key: str, dest_root: Path, patients: list = None,
                workers: int = 12, retries: int = 3, on_conflict: str = "unique",
                verify: bool = True, subdir: str = None, logger: CsvLogger = None,
                rounds: int = 1) -> dict:
    meta = DATASETS[key]
    slug = meta["slug"]
    folder, reused = resolve_folder(dest_root, meta, key, explicit=subdir)
    folder.mkdir(parents=True, exist_ok=True)
    global SIZE_CACHE
    _sc = folder / "_remote_sizes.json"
    if _sc.exists():
        try:
            SIZE_CACHE = json.loads(_sc.read_text(encoding="utf-8"))
        except Exception:  # noqa
            SIZE_CACHE = {}

    log("=" * 74)
    log(f"数据集：{meta['title']}  ({meta['source']}/{key} v{meta['version']})")
    log(f"保存目录：{folder}{'  （检测到已有目录，续传模式）' if reused else '  （新建）'}")
    log("=" * 74)

    records, sums = get_index(slug)

    if patients:
        ps = [p.strip().lower() for p in patients]
        records = [r for r in records if r.lower().split("/")[0] in ps]
        log(f"  按患者筛选 -> {len(records)} 个文件（{', '.join(ps)}）")

    # ---- 分批：按顶层患者目录 ----
    buckets = {}
    for rel in records:
        top = rel.split("/")[0] if "/" in rel else "_root"
        buckets.setdefault(top, []).append(rel)
    batch_names = sorted(buckets)
    log(f"  分批计划：{len(batch_names)} 批（按患者目录），并发 {workers}")
    log("")

    manifest = {
        "dataset": key,
        "title": meta["title"],
        "source": meta["source"],
        "version": meta["version"],
        "dest": str(folder),
        "started": datetime.now().isoformat(timespec="seconds"),
        "concurrency": workers,
        "files": [],
    }

    overall = {"ok": 0, "skipped": 0, "failed": 0, "bytes_ok": 0, "bytes_total": 0}
    failures = []
    t0 = time.time()

    def handle(rel: str) -> dict:
        global SIZE_CACHE
        url = f"{BASE}/{slug}/{rel}"
        dest = folder / rel
        # 优先用缓存的大小，避免每文件一次 HEAD 往返（弱网环境下这是主要失败源）
        size = SIZE_CACHE.get(rel, -1)
        if size <= 0:
            try:
                size = remote_size(url, retries=retries)
                SIZE_CACHE[rel] = size
            except Exception:  # noqa
                # 取不到远端大小不再直接判失败：交给下载环节从响应头推断
                size = -1
        sha = sums.get(rel)
        res = download_file(url, dest, size, sha, retries=retries,
                            on_conflict=on_conflict, verify=verify)
        res["origin"] = rel
        # 逐条写入追溯日志（CSV）
        if logger is not None:
            try:
                logger.write(dataset=f"{meta['source']}/{key} v{meta['version']}",
                             url=url, file=Path(rel).name, save_path=res.get("path", ""),
                             bytes=res.get("bytes", 0), status=STATUS_CN.get(res.get("status"), res.get("status")),
                             sha256=res.get("sha256", ""), attempt=res.get("attempt", 0),
                             note=res.get("note", ""))
            except Exception as e:  # noqa
                log(f"      [警告] 写入 CSV 失败 {rel}: {e}")
        return res

    total_files = len(records)
    t0 = time.time()

    pending = list(records)          # 本轮待处理集合
    round_no = 0
    while pending and round_no < rounds:
        round_no += 1
        if rounds > 1:
            log(f"\n  ---- 第 {round_no}/{rounds} 轮：待处理 {len(pending)} 个文件 ----")
        failures = []
        done_files = 0
        ok_this = skip_this = fail_this = 0
        for bi, bname in enumerate(batch_names, 1):
            rels = [r for r in buckets[bname] if r in set(pending)]
            if not rels:
                continue
            # 批次内先取小文件（标注/说明），再取大体积 EDF：
            # 这样每个患者一开批就能拿到 -summary.txt 与 *.edf.seizures，
            # 便于边下边做标注核对，也为后续"先到先处理"留出条件。
            rels.sort(key=lambda r: (0 if not r.lower().endswith(".edf") else 1, r))
            log(f"  [{bi}/{len(batch_names)}] 批次 {bname} —— {len(rels)} 个文件"
                + (f"（第 {round_no} 轮）" if rounds > 1 else ""))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(handle, r): r for r in rels}
                for fut in as_completed(futs):
                    rel = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa
                        res = {"path": str(folder / rel), "origin": rel, "status": "failed",
                               "bytes": 0, "sha256": "", "attempt": 0, "note": str(e)}
                    done_files += 1
                    st = res.get("status")
                    if st == "ok":
                        overall["ok"] += 1
                        overall["bytes_ok"] += res["bytes"]
                        ok_this += 1
                    elif st == "skipped":
                        overall["skipped"] += 1
                        overall["bytes_ok"] += res["bytes"]
                        skip_this += 1
                    else:
                        overall["failed"] += 1
                        fail_this += 1
                        failures.append({"dataset": key, "file": rel,
                                         "path": res.get("path", ""),
                                         "note": res.get("note", "")})
                    if st == "failed":
                        log(f"      [失败] {rel}: {res.get('note', '')[:110]}")
                    overall["bytes_total"] += res.get("bytes", 0)
                    if done_files % 20 == 0 or done_files == len(pending):
                        el = time.time() - t0
                        spd = overall["bytes_total"] / el / 1048576 if el > 0 else 0
                        pct = done_files / len(pending) * 100
                        eta = (len(pending) - done_files) * el / done_files if done_files else 0
                        log(f"      进度 {done_files}/{len(pending)} ({pct:5.1f}%) | "
                            f"已完成 {human(overall['bytes_total'])} | {spd:.2f} MB/s | "
                            f"预计剩余 {eta/60:.1f} min")
            # 每批结束即落盘大小缓存：
            # 看门狗会在 180 min 强杀本进程，若只在全部跑完时才保存，
            # 缓存将永远丢失，导致每次重启都要对已完成文件重复 HEAD 探测。
            try:
                (folder / "_remote_sizes.json").write_text(
                    json.dumps(SIZE_CACHE, ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa
                pass

        if not failures:
            log(f"  第 {round_no} 轮无失败，数据集 {key} 处理完毕。")
            break
        pending = [f["file"] for f in failures]
        if round_no < rounds:
            log(f"  ---- 第 {round_no} 轮结束：成功 {ok_this} / 跳过 {skip_this} / "
                f"失败 {fail_this}；剩余 {len(pending)} 个进入下一轮补齐 ----")
            time.sleep(5)
        else:
            log(f"  ---- 已达最大补齐轮次 {rounds}，仍 {len(pending)} 个未完成；"
                f"重跑同一命令可继续补齐（已下载字节不会丢失）----")

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    # 落盘远端大小缓存，下次重跑免于重复 HEAD 探测
    try:
        (folder / "_remote_sizes.json").write_text(
            json.dumps(SIZE_CACHE, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa
        pass
    manifest["summary"] = {**overall, "total_files": total_files}
    manifest["failures"] = failures
    mpath = folder / "_download_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log("")
    log(f"  ---- {key} 完成 ----")
    log(f"  新增 {overall['ok']} | 跳过 {overall['skipped']} | 失败 {overall['failed']} | 共 {total_files}")
    log(f"  目录字节数 {human(overall['bytes_total'])} | 清单 {mpath}")
    if failures:
        log(f"  [注意] 失败清单已写入 manifest，直接重跑本命令即可续传补齐")
    return {"key": key, "folder": folder, "manifest": mpath, "summary": overall,
            "total_files": total_files, "failures": failures}


def plan_dry(key: str, dest_root: Path, patients: list = None) -> None:
    """演练模式：只读取索引并输出将要生成的结构与分批计划，不下载任何数据。"""
    meta = DATASETS[key]
    folder, reused = resolve_folder(dest_root, meta, key)
    log("=" * 74)
    log(f"演练 · {meta['title']}（{meta['source']}/{key} v{meta['version']}）")
    log("=" * 74)
    log(f"  计划保存路径：{folder}{'  （已存在的目录，将续传而非重下）' if reused else '  （新建）'}")
    records, sums = get_index(meta["slug"])
    if patients:
        ps = [p.strip().lower() for p in patients]
        records = [r for r in records if r.lower().split("/")[0] in ps]
        log(f"  按患者筛选后：{len(records)} 个文件")
    buckets = {}
    for rel in records:
        top = rel.split("/")[0] if "/" in rel else "_root"
        buckets.setdefault(top, []).append(rel)
    log(f"  文件总数：{len(records)}　|　分批数：{len(buckets)}（按患者目录）")
    log(f"  可校验文件（SHA256SUMS 命中）：{sum(1 for r in records if r in sums)}")
    for i, (b, rels) in enumerate(sorted(buckets.items()), 1):
        if i <= 8 or i == len(buckets):
            log(f"    批次 {i:>2}/{len(buckets)}  {b:<12s} {len(rels)} 个文件")
        elif i == 9:
            log("    ...（其余批次已省略，正式运行时逐个显示）")
    log("")


def main():
    ap = argparse.ArgumentParser(description="公共脑电数据库稳健下载器")
    ap.add_argument("--dest", required=True, help="目标根目录，如 F:/研究数据库")
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=["chbmit"])
    ap.add_argument("--patients", nargs="*", default=None, help="只下载指定患者，如 chb01 chb02")
    ap.add_argument("--workers", type=int, default=12,
                    help="并发下载线程数（默认 12；网络被按连接限速，并发是关键提速手段）")
    ap.add_argument("--retries", type=int, default=3, help="单文件最大重试次数（默认 3）")
    ap.add_argument("--rounds", type=int, default=1,
                    help="补齐轮次：一轮跑完后自动对仍未完成的文件再跑一轮，最多 N 轮（默认 1）。"
                         "弱网下建议 3~5；每轮内每个文件的重试次数仍受 --retries 限制")
    ap.add_argument("--connect-jitter", type=float, default=0.0,
                    help="每次建立新连接前的随机抖动上限（秒）。弱网下设为 1~3 可显著降低 WinError 10060")
    ap.add_argument("--on-conflict", choices=["unique", "overwrite", "skip"],
                    default="unique",
                    help="同名文件处理：unique=一致跳过/不一致追加序号（默认），overwrite=覆盖，skip=跳过")
    ap.add_argument("--no-verify", action="store_true", help="跳过 SHA256 校验（更快但不安全）")
    ap.add_argument("--log", default=None,
                    help="追溯日志 CSV 路径（默认 <目标目录>/download_log.csv）")
    ap.add_argument("--dry-run", action="store_true",
                    help="演练：读取索引并输出目录结构/分批计划，不下载任何数据")
    ap.add_argument("--subdir", default=None,
                    help="显式指定数据集子目录名（默认按命名规则并复用已有日期目录）")
    ap.add_argument("--skip-preflight", action="store_true", help="跳过目标盘可写性预检")
    ap.add_argument("--need-gb", type=float, default=0.0, help="预计需要的 GB 数，用于空间校验")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in DATASETS.items():
            print(f"[{k}] {v['title']} v{v['version']}  ({v['source']})")
        return

    # ---- 步骤 1：目标目录预检 ----
    dest_root = Path(args.dest)
    need = int(args.need_gb * 2 ** 30) if args.need_gb else 0
    pf = preflight(dest_root, need)
    if not print_preflight(pf) and not args.skip_preflight:
        sys.exit(2)

    global CONNECT_JITTER
    CONNECT_JITTER = max(0.0, args.connect_jitter)

    log("")
    log(f"[策略] 冲突处理={args.on_conflict} | 并发={args.workers} | "
        f"重试={args.retries} | 校验={'关' if args.no_verify else '开'} | "
        f"连接抖动={CONNECT_JITTER}s")
    log(f"[命名] <目标>/<来源>_<数据集>_v<版本>_<日期>/，内部完整保留原始相对路径与文件名")

    if args.dry_run:
        log("\n（演练模式：不下载任何数据）\n")
        for ds in args.datasets:
            plan_dry(ds, dest_root, args.patients)
        log("演练结束。确认无误后去掉 --dry-run 即可正式下载。")
        return

    # ---- 追溯日志 ----
    log_path = Path(args.log) if args.log else dest_root / "download_log.csv"
    logger = CsvLogger(log_path)

    results = []
    for ds in args.datasets:
        results.append(run_dataset(
            ds, dest_root, patients=args.patients, workers=args.workers,
            retries=args.retries, on_conflict=args.on_conflict,
            verify=not args.no_verify, subdir=args.subdir, logger=logger,
            rounds=args.rounds))

    # ---- 汇总 ----
    log("\n" + "=" * 74)
    log("步骤 4 / 4　下载结果汇总")
    log("=" * 74)
    tot_f = tot_b = tot_fail = 0
    for r in results:
        s = r["summary"]
        folder: Path = r["folder"]
        files = [p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("_")]
        actual = sum(p.stat().st_size for p in files)
        log(f"  数据集 {r['key']}")
        log(f"    保存路径：{folder}")
        log(f"    文件数量：{len(files)}（本次新增 {s['ok']}，已存在跳过 {s['skipped']}，"
            f"失败 {s['failed']}，索引登记 {r['total_files']}）")
        log(f"    占用空间：{human(actual)}")
        log(f"    明细清单：{r['manifest']}")
        exts = {}
        for p in files:
            exts[p.suffix.lower()] = exts.get(p.suffix.lower(), 0) + 1
        log(f"    文件类型：{', '.join(f'{k} ×{v}' for k, v in sorted(exts.items(), key=lambda x: -x[1]))}")
        tot_f += len(files)
        tot_b += actual
        tot_fail += s["failed"]
    log("-" * 74)
    log(f"  合计：{tot_f} 个文件，{human(tot_b)}，失败 {tot_fail} 个")
    log(f"  追溯日志：{log_path}")

    # ---- 失败/成功清单明细 ----
    all_fail = [f for r in results for f in r["failures"]]
    if all_fail:
        log("\n" + "-" * 74)
        log(f"  失败文件清单（{len(all_fail)} 个）：")
        for i, f in enumerate(all_fail[:50], 1):
            log(f"    {i:>2}. [{f.get('dataset','')}] {f.get('file','')}")
            log(f"        保存路径 {f.get('path','')}  原因 {str(f.get('note',''))[:100]}")
        if len(all_fail) > 50:
            log(f"    ...另有 {len(all_fail)-50} 个，见各数据集的 _download_manifest.json")
        log("\n  处理办法：原样再跑一次相同命令，缺失/失败文件会自动续传补齐；")
        log("            若某文件反复失败，可降低 --workers 后重试。")
    else:
        log("\n  本次运行无失败文件。")
    log("=" * 74)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n[中断] 用户中止。已下载部分保留为 .part，重跑同命令可继续。")
        sys.exit(130)
