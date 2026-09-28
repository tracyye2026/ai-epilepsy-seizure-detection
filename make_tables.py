#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_tables.py — 由实验结果自动生成论文表格（Markdown + CSV）

输出（work/tables/）：
  table1_dataset.md / .csv        数据集与患者特征
  table2_lopo.md / .csv           留一患者法逐例结果（主结果）
  table3_summary.md               主要终点汇总（灵敏度 / FA/24h / 延迟）
  table4_ablation.md / .csv       导联精简消融
  table5_external.md / .csv       Siena 外部验证
  tables_all.md                   合并版（可直接粘进投稿系统）

用法：
  python make_tables.py
  python make_tables.py --prefix chbmit      # 只看主实验
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG  # noqa: E402


# ---------------------------------------------------------------- 工具
def _dur(m: dict) -> float:
    """记录时长(秒)。manifest 里字段名为 duration（不是 duration_sec）。"""
    v = m.get("duration", m.get("duration_sec", 0)) or 0
    return float(v)


def _sz_sec(m: dict) -> float:
    """记录内发作总时长(秒)，由 seizures 区间求和。"""
    return float(sum(max(0.0, float(e) - float(s)) for s, e in m.get("seizures", [])))


def _fmt(x, nd=3, dash="—"):
    """数值格式化；NaN / None 显示为中文破折号"""
    if x is None:
        return dash
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return dash
    except TypeError:
        pass
    return f"{x:.{nd}f}"


def _pct(x, nd=1):
    return _fmt(100 * x, nd) + "%" if x is not None and not (
        isinstance(x, float) and np.isnan(x)) else "—"


def md_table(rows, header):
    """list[list] -> Markdown 表格"""
    out = ["| " + " | ".join(str(c) for c in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def save_pair(df: pd.DataFrame, path_stem: Path):
    """同时写出 CSV(utf-8-sig) 与 Markdown"""
    df.to_csv(path_stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    path_stem.with_suffix(".md").write_text(
        md_table(df.astype(str).values.tolist(), list(df.columns)), encoding="utf-8")
    return df


def ci_str(ci, nd=3):
    if not ci or len(ci) < 2:
        return "—"
    return f"{ci[0]:.{nd}f}~{ci[1]:.{nd}f}"


# ---------------------------------------------------------------- 表1
def build_table1(cfg, manifests: dict) -> pd.DataFrame:
    rows = []
    for ds, mf in manifests.items():
        ok = [m for m in mf if m.get("ok")]
        pats = sorted({m["patient"] for m in ok})
        seiz = sum(m.get("n_seizures", 0) for m in ok)
        hours = sum(_dur(m) for m in ok) / 3600.0
        # 真实窗口数取 n_windows_total / n_ictal_total（n_windows 是负样本抽样后的建模窗口数）
        win = sum(m.get("n_windows_total", m.get("n_windows", 0)) for m in ok)
        swe = sum(m.get("n_ictal_total", 0) for m in ok)
        win_used = sum(m.get("n_windows", 0) for m in ok)
        rows.append({
            "数据集": "CHB-MIT" if ds == "chbmit" else ("Siena" if ds == "siena" else ds),
            "患者数": len(pats),
            "记录数": len(ok),
            "总时长(h)": round(hours, 1),
            "发作次数": seiz,
            "窗口总数": int(win),
            "发作窗口数": int(swe),
            "发作窗口占比(%)": round(100 * swe / max(win, 1), 2),
            "建模窗口数(抽样后)": int(win_used),
            "采样率(Hz)": 128,
            "导联数": 18,
        })
    return pd.DataFrame(rows)


def build_table1_patients(cfg, mf, ds_label: str, lopo_csv: Path | None) -> pd.DataFrame:
    """逐例特征（可与主结果合并）"""
    ok = [m for m in mf if m.get("ok")]
    by = {}
    for m in ok:
        by.setdefault(m["patient"], []).append(m)
    det = {}
    if lopo_csv and lopo_csv.exists():
        df = pd.read_csv(lopo_csv)
        det = dict(zip(df["patient"], df["n_seizures"]))
    rows = []
    for p in sorted(by):
        recs = by[p]
        rows.append({
            "患者": p,
            "记录数": len(recs),
            "总时长(h)": round(sum(_dur(r) for r in recs) / 3600, 1),
            "发作次数": sum(r.get("n_seizures", 0) for r in recs),
            "发作总时长(min)": round(sum(_sz_sec(r) for r in recs) / 60, 1),
            "主结果纳入发作": det.get(p, "—"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 表2/3
def build_table2(cfg, lopo_csv: Path, aux_csv: Path | None = None) -> pd.DataFrame:
    """逐例结果表。

    lopo_csv：主结果（优先 primary_endpoint_per_patient.csv，冻结工作点）
    aux_csv ：可选，原始 lopo_*.csv，用于补齐记录数/窗级AUC（阈值无关量）
    """
    df = pd.read_csv(lopo_csv)
    has = lambda c: c in df.columns  # noqa: E731

    def series(name, default=np.nan):
        return df[name] if has(name) else pd.Series([default] * len(df))

    auc = series("sample_auc")
    delay = series("delay_median_sec")
    thr = series("threshold")
    nrec = series("n_records")

    if aux_csv is not None and Path(aux_csv).exists():
        ax = pd.read_csv(aux_csv)
        if {"patient", "sample_auc"}.issubset(ax.columns) and not has("sample_auc"):
            auc = pd.Series(df["patient"].map(
                dict(zip(ax["patient"], ax["sample_auc"]))).values)
        if {"patient", "n_records"}.issubset(ax.columns) and not has("n_records"):
            nrec = pd.Series(df["patient"].map(
                dict(zip(ax["patient"], ax["n_records"]))).values)

    d = pd.DataFrame({
        "测试患者": df["patient"],
        "记录数": nrec,
        "发作次数": df["n_seizures"],
        "检出次数": df["n_detected"],
        "事件灵敏度": df["sensitivity"].map(lambda v: _pct(v, 1)),
        "假报警(次)": df["n_false_alarms"].astype(int),
        "记录时长(h)": (df["total_hours"] if has("total_hours") else df["hours"]
                        ).map(lambda v: _fmt(v, 1)),
        "FA/24h": df["fa_per_24h"].map(lambda v: _fmt(v, 2)),
        "窗级AUC": auc.map(lambda v: _fmt(v, 3)),
        "中位检出延迟(s)": delay.map(lambda v: _fmt(v, 1)),
        "阈值": thr.map(lambda v: _fmt(v, 2)),
    })
    # 合计行
    tot_h = (df["total_hours"] if has("total_hours") else df["hours"]).sum()
    d.loc[len(d)] = [
        "合计/汇总",
        int(pd.to_numeric(nrec, errors="coerce").fillna(0).sum()) or "—",
        int(df["n_seizures"].sum()), int(df["n_detected"].sum()),
        _pct(df["n_detected"].sum() / max(df["n_seizures"].sum(), 1), 1),
        int(df["n_false_alarms"].sum()), _fmt(float(tot_h), 1),
        _fmt(df["n_false_alarms"].sum() / max(tot_h, 1e-9) * 24, 2),
        _fmt(pd.to_numeric(auc, errors="coerce").mean(), 3),
        _fmt(pd.to_numeric(delay, errors="coerce").median(), 1), "—",
    ]
    # 全空列不展示（冻结工作点版没有逐例延迟/阈值）
    for c in list(d.columns):
        if set(d[c].astype(str).unique()) <= {"—", "nan", "None"}:
            d = d.drop(columns=[c])
    return d


def build_table3(summary: dict) -> pd.DataFrame:
    s = summary
    rows = [
        ["汇总事件灵敏度", _pct(s.get("pooled_sensitivity"), 1),
         f"{int(s.get('total_detected', 0))}/{int(s.get('total_seizures', 0))}"],
        ["患者水平平均灵敏度", _pct(s.get("patient_mean_sensitivity"), 1),
         f"95%CI {ci_str(s.get('sensitivity_ci95'), 3)}"],
        ["汇总 FA/24h", _fmt(s.get("pooled_fa_per_24h"), 2),
         f"{int(s.get('total_false_alarms', 0))} 次 / {s.get('total_hours', 0)} h"],
        ["患者水平平均 FA/24h", _fmt(s.get("patient_mean_fa_per_24h"), 2),
         f"95%CI {ci_str(s.get('fa_per_24h_ci95'), 2)}"],
        ["窗级平均 AUC", _fmt(s.get("mean_auc"), 3), "次要终点"],
        ["中位检出延迟(s)", _fmt(s.get("median_delay_sec"), 1), "事件级"],
        ["灵敏度≥50% 的患者比例", _pct(s.get("patient_level_sensitivity_ge50pct"), 1), ""],
    ]
    return pd.DataFrame(rows, columns=["指标", "数值", "备注"])


def build_table3_primary(p: dict) -> pd.DataFrame:
    """主要终点表（冻结工作点版）：主策略 + 两种对照。"""
    s_ci, f_ci = p.get("sensitivity_ci95", [np.nan, np.nan]), p.get("fa_ci95", [np.nan, np.nan])
    rows = [
        ["事件灵敏度（逐例均值）", _pct(p.get("sensitivity_mean"), 1),
         f"95%CI {_pct(s_ci[0], 1)}~{_pct(s_ci[1], 1)}；中位 {_pct(p.get('sensitivity_median'), 1)}"],
        ["事件灵敏度（汇总）", _pct(p.get("pooled_sensitivity"), 1),
         f"{p.get('total_detected')}/{p.get('total_seizures')} 次发作"],
        ["FA/24h（逐例均值）", _fmt(p.get("fa_mean"), 2),
         f"95%CI {_fmt(f_ci[0], 2)}~{_fmt(f_ci[1], 2)}；中位 {_fmt(p.get('fa_median'), 2)}"],
        ["FA/24h（汇总）", _fmt(p.get("pooled_fa_per_24h"), 2),
         f"{p.get('total_false_alarms')} 次 / {_fmt(p.get('total_hours'), 1)} h"],
        ["灵敏度≥50% 的患者比例", _pct(p.get("patients_sensitivity_ge50pct"), 1), ""],
        ["窗级 AUC", _fmt(p.get("window_auc"), 3), "次要终点（与阈值无关）"],
        ["工作点", f"q={p.get('thr_quantile')}/下限{p.get('thr_floor')}，"
                  f"最短报警 {_fmt(p.get('min_event_sec'), 0)} s",
         f"仅在验证患者上标定，FA 预算 {_fmt(p.get('budget_fa_per_24h'), 0)}/24h"],
    ]
    ca = p.get("control_A_global_threshold")
    if ca:
        rows.append(["对照：策略A 全局固定阈值",
                     f"灵敏度 {_pct(ca.get('sensitivity_mean'), 1)}",
                     f"FA/24h {_fmt(ca.get('fa_mean'), 2)}（同预算）"])
    cd = p.get("control_default_threshold")
    if cd:
        rows.append(["对照：未标定默认阈值",
                     f"灵敏度 {_pct(cd.get('sensitivity_mean'), 1)}",
                     f"FA/24h {_fmt(cd.get('fa_mean'), 2)}（FA 超标）"])
    return pd.DataFrame(rows, columns=["指标", "数值", "备注"])


# ---------------------------------------------------------------- 表4 消融
def build_table4(cfg, ab_csv: Path | None, summaries: dict) -> pd.DataFrame | None:
    """导联精简消融。ablation.py 输出为中文列名，此处做列名兼容映射。"""
    rows = []
    if ab_csv and ab_csv.exists():
        df = pd.read_csv(ab_csv)

        def col(*names):
            for n in names:
                if n in df.columns:
                    return n
            return None

        c_name = col("导联配置", "channel_set")
        c_nch = col("导联数", "n_channels")
        c_sens = col("患者平均灵敏度", "patient_mean_sensitivity", "sensitivity")
        c_ci = col("灵敏度95%CI", "sensitivity_ci95")
        c_fa = col("平均FA/24h", "patient_mean_fa_per_24h", "fa_per_24h")
        c_auc = col("平均窗级AUC", "mean_auc")
        c_det = col("检出/总数", "detected_over_total")
        c_d = col("Δ灵敏度(vs全导联)", "delta_sensitivity")
        c_p = col("P值(配对Bootstrap)", "p_value")
        c_delay = col("中位延迟(s)", "median_delay_sec")

        for _, r in df.iterrows():
            rows.append([
                r.get(c_name, "?") if c_name else "?",
                r.get(c_nch, "?") if c_nch else "?",
                r.get(c_det, "—") if c_det else "—",
                _pct(r.get(c_sens, np.nan), 1) if c_sens else "—",
                (str(r.get(c_ci, "—")) if c_ci else "—"),
                _fmt(r.get(c_fa, np.nan), 2) if c_fa else "—",
                _fmt(r.get(c_auc, np.nan), 3) if c_auc else "—",
                _fmt(r.get(c_delay, np.nan), 1) if c_delay else "—",
                (str(r.get(c_d, "—")) if c_d else "—"),
                (str(r.get(c_p, "—")) if c_p else "—"),
            ])
    else:
        for key, s in summaries.items():
            if key.startswith("full18") or "_" not in key:
                continue
            rows.append([
                s.get("channel_set", key), s.get("n_channels", "?"), "—",
                _pct(s.get("patient_mean_sensitivity"), 1), "—",
                _fmt(s.get("patient_mean_fa_per_24h"), 2),
                _fmt(s.get("mean_auc"), 3), "—", "—", "—",
            ])
    if not rows:
        return None
    return pd.DataFrame(rows, columns=[
        "导联方案", "导联数", "检出/总数", "平均灵敏度", "灵敏度95%CI",
        "平均FA/24h", "窗级AUC", "中位延迟(s)", "Δ灵敏度(vs全导联)", "P值(配对Bootstrap)"])


# ---------------------------------------------------------------- 表5 外部验证
def build_table5(cfg, ext_json: Path | None, ext_csv: Path | None) -> pd.DataFrame | None:
    """外部验证汇总表。工作点已在 CHB-MIT 上冻结，未用 Siena 任何数据调参。"""
    if ext_json and ext_json.exists():
        s = json.loads(ext_json.read_text(encoding="utf-8"))
        ci_s = s.get("sensitivity_ci95") or [np.nan, np.nan]
        ci_f = s.get("fa_per_24h_ci95") or [np.nan, np.nan]
        gap = s.get("generalization_gap_sensitivity")
        rows = [
            ["外部验证患者数", s.get("n_patients", "—"), ""],
            ["发作总数", s.get("total_seizures", "—"), "事件级"],
            ["检出发作数", s.get("total_detected", "—"), "事件级"],
            ["汇总事件灵敏度", _pct(s.get("pooled_sensitivity"), 1), "事件级"],
            ["患者平均灵敏度", _pct(s.get("patient_mean_sensitivity"), 1),
             f"95%CI {_pct(ci_s[0], 1)}~{_pct(ci_s[1], 1)}"],
            ["汇总 FA/24h", _fmt(s.get("pooled_fa_per_24h"), 2), "事件级"],
            ["患者平均 FA/24h", _fmt(s.get("patient_mean_fa_per_24h"), 2),
             f"95%CI {_fmt(ci_f[0], 2)}~{_fmt(ci_f[1], 2)}"],
            ["窗级 AUC", _fmt(s.get("window_auc", s.get("mean_auc")), 3), "次要终点"],
            ["记录总时长(h)", _fmt(s.get("total_hours"), 1), ""],
            ["内部 LOPO 灵敏度（同工作点）", _pct(s.get("internal_lopo_sensitivity"), 1), "对照"],
            ["泛化落差（外部−内部）", _fmt(gap, 3) if gap is not None else "—", "对照"],
        ]
        return pd.DataFrame(rows, columns=["指标", "数值", "备注"])
    if ext_csv and ext_csv.exists():
        return pd.read_csv(ext_csv)
    return None


def build_table5b(ext_csv: Path | None) -> pd.DataFrame | None:
    """外部验证逐例结果（列名中文化）。"""
    if not (ext_csv and ext_csv.exists()):
        return None
    df = pd.read_csv(ext_csv)
    out = pd.DataFrame({
        "患者": df["patient"],
        "记录数": df["n_records"],
        "发作数": df["n_seizures"],
        "检出": df["n_detected"],
        "灵敏度": (df["sensitivity"].astype(float) * 100).round(1),
        "假报警(次)": df["n_false_alarms"],
        "时长(h)": df["total_hours"].astype(float).round(2),
        "FA/24h": df["fa_per_24h"].astype(float).round(2),
        "中位延迟(s)": pd.to_numeric(df.get("delay_median_sec"), errors="coerce").round(1),
    })
    return out


# ---------------------------------------------------------------- 表6 全局工作点重标定
def _tag(b):
    return f"_b{str(float(b)).replace('.', 'p')}"


def build_table6(res: Path, prefer_budget: float = 10.0) -> tuple | None:
    """读取 recalibrate.py 的产物：预算对比 + 选定工作点下的逐例结果。"""
    cmp_csv = res / "recalib_budget_comparison.csv"
    if not cmp_csv.exists():
        return None
    cmp_df = pd.read_csv(cmp_csv)

    budgets = cmp_df["budget_fa_per_24h"].astype(float).tolist()
    b = min(budgets, key=lambda x: abs(x - prefer_budget))
    tag = _tag(b)
    # 主要策略为 B（逐记录秩自适应阈值）；A 为对照
    per_csv = res / f"recalibB_per_patient{tag}.csv"
    if not per_csv.exists():
        per_csv = res / f"recalibB_per_patient{_tag(budgets[0])}.csv"
        b = budgets[0]
    if not per_csv.exists():                      # 兼容旧版单一文件名
        per_csv = res / f"recalib_per_patient{tag}.csv"
    per = pd.read_csv(per_csv)

    # (1) 各预算下、两种策略的工作点与测试集表现（全部只用验证数据标定）
    c1 = pd.DataFrame({
        "标定策略": cmp_df["strategy"] if "strategy" in cmp_df else "—",
        "验证集FA预算(/24h)": cmp_df["budget_fa_per_24h"].map(lambda v: _fmt(v, 1)),
        "工作点": cmp_df["threshold"].map(
            lambda v: _fmt(v, 3) if isinstance(v, (int, float)) else str(v)),
        "最短报警(s)": cmp_df["min_event_sec"].map(lambda v: _fmt(v, 0)),
        "验证灵敏度": cmp_df["val_sensitivity"].map(lambda v: _pct(v, 1)),
        "验证FA/24h": cmp_df["val_fa_per_24h"].map(lambda v: _fmt(v, 2)),
        "测试灵敏度(池化)": cmp_df["pooled_sensitivity"].map(lambda v: _pct(v, 1)),
        "测试灵敏度(逐例均值)": cmp_df["sensitivity_mean"].map(lambda v: _pct(v, 1)),
        "测试FA/24h(池化)": cmp_df["pooled_fa_per_24h"].map(lambda v: _fmt(v, 2)),
        "测试FA/24h(逐例均值)": cmp_df["fa_mean"].map(lambda v: _fmt(v, 2)),
    })

    # (2) 选定预算下的逐例结果
    c2 = pd.DataFrame({
        "测试患者": per["patient"],
        "发作次数": per["n_seizures"].astype(int),
        "检出次数": per["n_detected"].astype(int),
        "事件灵敏度": per["sensitivity"].map(lambda v: _pct(v, 1)),
        "假报警(次)": per["n_false_alarms"].astype(int),
        "记录时长(h)": per["hours"].map(lambda v: _fmt(v, 1)),
        "FA/24h": per["fa_per_24h"].map(lambda v: _fmt(v, 2)),
    })
    tp, td, tf, th = (int(per["n_seizures"].sum()), int(per["n_detected"].sum()),
                      int(per["n_false_alarms"].sum()), float(per["hours"].sum()))
    c2.loc[len(c2)] = ["合计/汇总", tp, td, _pct(td / max(tp, 1), 1), tf,
                       _fmt(th, 1), _fmt(tf / max(th, 1e-9) * 24, 2)]
    return c1, c2, b


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--main_csv", default=None, help="主实验逐例CSV路径")
    args = ap.parse_args()

    cfg = CFG.config_from_args(argparse.Namespace(
        dataset="chbmit", work_root=args.work_dir, channel_set="full18",
        epochs=None, batch_size=None, lr=None, folds=None, device=None,
        log_level="INFO", formats="png,svg", max_train_windows=None,
    )) if hasattr(CFG, "config_from_args") else CFG.Config()

    res = Path(cfg.result_dir)
    tbl = Path(cfg.work_dir) / "tables"
    tbl.mkdir(parents=True, exist_ok=True)

    # 读取清单
    manifests = {}
    for ds in ("chbmit", "siena"):
        mf = Path(cfg.work_dir) / "index" / ds / "manifest.json"
        if mf.exists():
            manifests[ds] = json.loads(mf.read_text(encoding="utf-8"))

    # 读取各结果
    summary_files = sorted(res.glob("summary_*.json"))
    summaries = {}
    for f in summary_files:
        key = f.stem.replace("summary_", "")
        try:
            summaries[key] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print("读取失败", f, e)

    main_csv = Path(args.main_csv) if args.main_csv else (res / "lopo_chbmit_full18.csv")
    if not main_csv.exists():
        cands = sorted(res.glob("lopo_*.csv"))
        main_csv = cands[0] if cands else None

    blocks = []

    # 表1
    if manifests:
        t1 = build_table1(cfg, manifests)
        save_pair(t1, tbl / "table1_dataset")
        blocks.append(("表1 数据集基本特征", t1))
        if "chbmit" in manifests:
            t1b = build_table1_patients(cfg, manifests["chbmit"], "CHB-MIT", main_csv)
            save_pair(t1b, tbl / "table1b_chbmit_patients")
            blocks.append(("表1b CHB-MIT 逐例特征", t1b))

    # 表2 / 表3
    # 主要终点必须来自"冻结工作点"（策略B，预算10/24h）的重标定产物：
    # primary_endpoint_per_patient.csv；lopo_*.csv 是默认固定阈值的原始输出，仅作对照。
    prim_csv = res / "primary_endpoint_per_patient.csv"
    prim_json = res / "primary_endpoint.json"
    t2_csv = prim_csv if prim_csv.exists() else main_csv
    if t2_csv and t2_csv.exists():
        t2 = build_table2(cfg, t2_csv, aux_csv=main_csv if main_csv and main_csv.exists() else None)
        save_pair(t2, tbl / "table2_lopo")
        blocks.append(("表2 留一患者法逐例结果（冻结工作点，策略B）", t2))

    if prim_json.exists():
        t3 = build_table3_primary(json.loads(prim_json.read_text(encoding="utf-8")))
        save_pair(t3, tbl / "table3_summary")
        blocks.append(("表3 主要终点汇总（冻结工作点，策略B）", t3))
    elif main_csv and main_csv.exists():
        key = main_csv.stem.replace("lopo_", "")
        s = summaries.get(key, {})
        if s:
            t3 = build_table3(s)
            save_pair(t3, tbl / "table3_summary")
            blocks.append(("表3 主要终点汇总", t3))

    # 表4
    t4 = build_table4(cfg, res / "ablation_channels.csv", summaries)
    if t4 is not None:
        save_pair(t4, tbl / "table4_ablation")
        blocks.append(("表4 导联精简消融", t4))

    # 表5（外部验证：实际文件名为 external_summary_<src>2<tgt>.json / external_<src>2<tgt>.csv）
    ext_json, ext_csv = None, None
    for p in sorted(res.glob("external_summary_*.json")):
        ext_json = p
        break
    for p in sorted(res.glob("external_*.csv")):
        ext_csv = p
        break
    if ext_json is None and (res / "external_validation.json").exists():
        ext_json = res / "external_validation.json"
    if ext_csv is None and (res / "external_validation.csv").exists():
        ext_csv = res / "external_validation.csv"

    t5 = build_table5(cfg, ext_json, ext_csv)
    if t5 is not None:
        save_pair(t5, tbl / "table5_external")
        blocks.append(("表5 Siena 外部验证（工作点冻结）", t5))
    t5b = build_table5b(ext_csv)
    if t5b is not None:
        save_pair(t5b, tbl / "table5b_external_per_patient")
        blocks.append(("表5b 外部验证逐例结果", t5b))

    # 表6 全局工作点重标定
    t6 = build_table6(res, prefer_budget=10.0)
    if t6 is not None:
        c1, c2, b = t6
        save_pair(c1, tbl / "table6a_recalib_budgets")
        blocks.append((f"表6a 全局工作点标定（预算敏感性，选定预算 {b:g}/24h）", c1))
        save_pair(c2, tbl / "table6b_recalib_per_patient")
        blocks.append((f"表6b 全局工作点下留一患者法逐例结果（预算 {b:g}/24h）", c2))

    # 合并
    lines = ["# 论文表格汇总（自动生成）", ""]
    for title, df in blocks:
        lines += [f"## {title}", "",
                  md_table(df.astype(str).values.tolist(), list(df.columns)), ""]
    (tbl / "tables_all.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"已生成 {len(blocks)} 张表 -> {tbl}")
    for title, df in blocks:
        print(f"  · {title}：{len(df)} 行")


if __name__ == "__main__":
    main()
