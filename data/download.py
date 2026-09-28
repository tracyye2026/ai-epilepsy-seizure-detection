# -*- coding: utf-8 -*-
"""
download.py —— 公共数据库获取

【重要·已核实 2026-09-14】CHB-MIT 与 Siena 在 PhysioNet 上均为 **开放访问（Open Access）**：
   · CHB-MIT  v1.0.0：Access Policy = "Anyone can access the files"；
                        License = Open Data Commons Attribution License (ODC-By) v1.0；
                        未压缩 42.6 GB
   · Siena    v1.0.0：License = Creative Commons Attribution 4.0 International (CC BY 4.0)；
                        未压缩 20.3 GB（zip 13.0 GB）
  => **不需要 PhysioNet 账号、不需要 CITI 培训、不需要 Credentialing、不需要推荐人**，
     可直接匿名下载（ wget / AWS S3 --no-sign-request / 本脚本 ）。

  注：PhysioNet 自 2025-04-08 起实施的美国司法部数据安全计划(DOJ DSP)限制，
      仅针对 credentialed / restricted 层级的受控数据集（如 MIMIC 系列），
      开放访问的 CHB-MIT 与 Siena 不在限制范围内。

  是否需要伦理培训，取决于**本单位伦理委员会**的要求；
  若单位或期刊索要，最省事的免费路径是：
      CITI Program 选择 affiliation = "Massachusetts Institute of Technology Affiliates"
      + 课程 "Data or Specimens Only Research"（详见项目根目录 CITI 说明文档）。

命令行示例（匿名直连，无需账号）：
   python data/download.py --dataset chbmit --data_root ./data_raw
   python data/download.py --dataset chbmit --patients chb01 chb02 chb03   # 只下部分患者
   python data/download.py --dataset siena  --data_root ./data_raw
   # 若确需账号（如下载其他受控资源）：
   python data/download.py --dataset chbmit --data_root ./data_raw --username 你的邮箱
   # 若已手动下载 zip：
   python data/download.py --dataset chbmit --zip chb-mit-scalp-eeg-database-1.0.0.zip --data_root ./data_raw
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "chbmit": {
        "title": "CHB-MIT Scalp EEG Database v1.0.0",
        "url": "https://physionet.org/static/published-projects/chbmit/chb-mit-scalp-eeg-database-1.0.0.zip",
        "size": "未压缩 42.6 GB（zip 约 43 GB）",
        "patients": "22 名受试者 23 例（含同一人二次入院 chb21），664 个 edf，198 次发作；"
                    "其中 129 个文件含发作；23 导双极 256 Hz",
        "need": "开放访问，匿名可直接下载（ODC-By v1.0）；引用即可",
        "citation": "Shoeb AH. Application of Machine Learning to Epileptic Seizure Onset "
                    "Detection and Treatment. PhD Thesis, MIT, 2009.",
    },
    "siena": {
        "title": "Siena Scalp EEG Database v1.0.0",
        "url": "https://physionet.org/static/published-projects/siena-scalp-eeg/siena-scalp-eeg-database-1.0.0.zip",
        "size": "未压缩 20.3 GB（zip 13.0 GB）",
        "patients": "14 例成人（9 男 25-71 岁，5 女 20-58 岁），47 次发作，约 128 小时；"
                    "29 通道 10-20 参考导联 512 Hz，含 1-2 路 EKG",
        "need": "开放访问，匿名可直接下载（CC BY 4.0）；引用即可",
        "citation": "Detti P, Vatti G, et al. Siena Scalp EEG Database. PhysioNet, 2020.",
    },
    "tusz": {
        "title": "TUH EEG Seizure Corpus (TUSZ) v2.0.0",
        "url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
        "size": "约 70 GB",
        "patients": "675 例，1476 小时，4029 次发作；≥17 导(多数 19)，≥250 Hz，双极 TCP 蒙太奇",
        "need": "需向 Temple NEDC 申请账号 + 签署 DUA（非 PhysioNet 直接下载）",
        "citation": "Shah V, et al. The Temple University Hospital Seizure Detection Corpus. "
                    "Front Neuroinform, 2018.",
    },
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = downloaded / total_size * 100 if total_size else 0
    mb = downloaded / 1024 / 1024
    sys.stdout.write(f"\r  已下载 {mb:8.1f} MB ({pct:5.1f}%)")
    sys.stdout.flush()


# PhysioNet 文件浏览路径（用于按患者目录逐个下载，避免整个 43 GB 压缩包）
FILE_SLUGS = {
    "chbmit": "chbmit/1.0.0",
    "siena": "siena-scalp-eeg/1.0.0",
}


def _install_auth(username: str, password: str) -> None:
    pwd_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pwd_mgr.add_password(None, "https://physionet.org/", username, password or "")
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(pwd_mgr)))


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def download_patients(dataset: str, patients: list, data_root: Path,
                      username: str = None, password: str = None) -> Path:
    """
    只下载指定患者的目录（磁盘/时间紧张时的首选）。

    做法：先取 PhysioNet 的 RECORDS 索引（几十 KB），筛出目标患者的文件逐个下载。
    例：python data/download.py --dataset chbmit --patients chb01 chb02 chb03
    """
    slug = FILE_SLUGS.get(dataset)
    if not slug:
        raise ValueError(f"{dataset} 不支持按患者下载，请使用整库下载")
    if username:
        _install_auth(username, password)
    print(f"获取文件索引: https://physionet.org/files/{slug}/RECORDS")
    files = [l.strip() for l in _get_text(
        f"https://physionet.org/files/{slug}/RECORDS").splitlines() if l.strip()]
    wanted = []
    for p in patients:
        p = p.strip()
        hit = [f for f in files if f.lower().startswith(p.lower() + "/")]
        # 补上标注文件（RECORDS 有时不含 .txt）
        ann = f"{p}/{p}-summary.txt" if p.lower().startswith("chb") else None
        if ann and ann not in hit:
            hit.append(ann)
        if not hit:
            print(f"  [警告] 未找到患者 {p} 的文件")
        wanted += hit
    wanted = sorted(set(wanted))
    print(f"共 {len(wanted)} 个文件待下载（患者：{', '.join(patients)}）")

    out_root = data_root / dataset
    for i, rel in enumerate(wanted, 1):
        dest = out_root / rel
        if dest.exists():
            print(f"  [{i}/{len(wanted)}] 已存在，跳过 {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [{i}/{len(wanted)}] {rel}")
        try:
            urllib.request.urlretrieve(f"https://physionet.org/files/{slug}/{rel}", str(dest))
        except Exception as e:
            print(f"    下载失败：{e}")
    print(f"\n完成，数据位于 {out_root}")
    return out_root


def download_file(url: str, dest: Path, username: str = None, password: str = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if username:
        pwd_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pwd_mgr.add_password(None, "https://physionet.org/", username, password or "")
        handler = urllib.request.HTTPBasicAuthHandler(pwd_mgr)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)
    print(f"开始下载: {url}\n保存到: {dest}")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    except Exception as e:  # noqa
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"下载失败：{e}\n"
            "常见原因：① 未通过 PhysioNet 认证（需先完成 CITI 课程并获批）；\n"
            "        ② 账号/密码错误；③ 网络代理限制。\n"
            "也可直接在浏览器登录 https://physionet.org 手动下载 zip，再用 --zip 参数解压。"
        )
    print("\n下载完成。")
    return dest


def unzip(zip_path: Path, data_root: Path) -> Path:
    print(f"正在解压 {zip_path.name} ...（大文件，请耐心等待）")
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_root)
    print(f"解压完成 -> {data_root}")
    return data_root


def main():
    ap = argparse.ArgumentParser(description="下载 PhysioNet 脑电数据库")
    ap.add_argument("--dataset", choices=list(DATASETS), default="chbmit")
    ap.add_argument("--data_root", default="./data_raw")
    ap.add_argument("--username", default=None, help="PhysioNet 账号(邮箱)")
    ap.add_argument("--password", default=None, help="PhysioNet 密码")
    ap.add_argument("--zip", default=None, help="已下载好的本地 zip 路径（跳过下载）")
    ap.add_argument("--patients", nargs="*", default=None,
                    help="只下载指定患者目录，如 --patients chb01 chb02（磁盘/时间紧张时用）")
    ap.add_argument("--list", action="store_true", help="仅列出可用数据库信息")
    args = ap.parse_args()

    if args.list:
        for k, v in DATASETS.items():
            print(f"\n[{k}] {v['title']}")
            for kk in ("size", "patients", "need", "citation"):
                print(f"    {kk:9s}: {v[kk]}")
        return

    if args.username and args.password is None:
        args.password = getpass.getpass("PhysioNet 密码(仅受控资源需要): ")
    meta = DATASETS[args.dataset]
    print(f"数据库: {meta['title']}\n{meta['patients']}\n获取条件: {meta['need']}\n")
    data_root = Path(args.data_root).resolve()

    if args.zip:
        unzip(Path(args.zip).resolve(), data_root)
    elif args.patients:
        # CHB-MIT / Siena 为开放访问，默认匿名；仅在显式提供 --username 时才加认证
        download_patients(args.dataset, args.patients, data_root,
                          args.username, args.password)
        print("\n完成。下一步：python preprocess.py --dataset " + args.dataset +
              " （仅处理已下载的患者）")
    else:
        if args.dataset == "tusz":
            print("TUSZ 需先签署 DUA，请访问：", meta["url"], " 按官方指引申请。")
            return
        user, pwd = args.username, args.password     # 开放数据集无需账号
        zip_path = data_root / Path(meta["url"]).name
        download_file(meta["url"], zip_path, user, pwd)
        unzip(zip_path, data_root)
        # 可选：删除 zip 节省磁盘
        if input("是否删除 zip 以释放磁盘空间？(y/N): ").strip().lower() == "y":
            zip_path.unlink()
    print("\n完成。下一步：python preprocess.py --dataset " + args.dataset)


if __name__ == "__main__":
    main()
