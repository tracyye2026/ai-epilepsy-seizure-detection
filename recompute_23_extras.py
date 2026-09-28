# -*- coding: utf-8 -*-
"""recompute_23_extras.py —— 23 受试者口径的补充统计：
  A. "未标定默认阈值"(thr=0.5) 对照（用于表 3 新增行）
  B. 汇总（pooled）指标的以患者为单位 Bootstrap 95%CI
  C. 四种方法的汇总精确率 / F1（用于表 8 新增列）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.metrics import movavg_same, event_counts_fast  # noqa: E402

work = ROOT / "work"
CACHE = work / "recalib_cache"
SMOOTH, MERGE_GAP, MIN_DUR, MIN_OVERLAP = 8.0, 10.0, 10.0, 1.0
DEFAULT_THR = 0.5


def iter_cache(p):
    d = np.load(p)
    i = 0
    while f"{i}_p" in d:
        yield d[f"{i}_y"], d[f"{i}_p"], float(d[f"{i}_fs"][0])
        i += 1
    d.close()


def per_patient_default():
    """未标定默认阈值（0.5）下逐受试者事件级计数。"""
    rows = []
    for f in sorted(CACHE.glob("*__test_*.npz")):
        subj = f.name.split("__test_")[0]
        nt = nd = nf = 0
        hrs = 0.0
        for y, p, sf in iter_cache(f):
            ps = movavg_same(p, max(1, int(round(SMOOTH * sf))))
            a, b, c, h = event_counts_fast(y, p, sf, DEFAULT_THR, MERGE_GAP,
                                           MIN_DUR, MIN_OVERLAP,
                                           smooth_sec=SMOOTH, prob_smoothed=ps)
            nt += a
            nd += b
            nf += c
            hrs += h
        rows.append({"patient": subj, "n_seizures": nt, "n_detected": nd,
                     "n_false_alarms": nf, "hours": hrs,
                     "sensitivity": nd / nt if nt else np.nan,
                     "fa_per_24h": nf / hrs * 24 if hrs else np.nan})
    return pd.DataFrame(rows)


def boot_ci(df, n_boot=1000, seed=42):
    """以患者为单位重抽样：pooled 灵敏度 与 pooled FA/24h 的 95%CI。"""
    rng = np.random.default_rng(seed)
    n = len(df)
    s = df["n_seizures"].to_numpy(float)
    d = df["n_detected"].to_numpy(float)
    f = df["n_false_alarms"].to_numpy(float)
    h = df["hours"].to_numpy(float)
    sens, fa = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ss, dd, ff, hh = s[idx].sum(), d[idx].sum(), f[idx].sum(), h[idx].sum()
        if ss > 0:
            sens.append(dd / ss)
        if hh > 0:
            fa.append(ff / hh * 24)
    return (float(np.percentile(sens, 2.5)), float(np.percentile(sens, 97.5)),
            float(np.percentile(fa, 2.5)), float(np.percentile(fa, 97.5)))


def pooled_precision_f1(df):
    nd = float(df["n_detected"].sum())
    nf = float(df["n_false_alarms"].sum())
    ns = float(df["n_seizures"].sum())
    prec = nd / (nd + nf) if (nd + nf) else np.nan
    sens = nd / ns if ns else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else np.nan
    return round(prec, 3), round(f1, 3)


def main():
    out = {}

    # ---- A/B: 默认阈值对照 + 策略 B 汇总 CI ----
    dfl = per_patient_default()
    dfl.to_csv(work / "results" / "lopo23" / "default_thr_per_patient.csv",
               index=False, encoding="utf-8-sig")
    nt, nd, nf, hrs = (dfl["n_seizures"].sum(), dfl["n_detected"].sum(),
                       dfl["n_false_alarms"].sum(), dfl["hours"].sum())
    out["default_threshold_0p5"] = {
        "n_patients": len(dfl),
        "pooled_sensitivity": round(nd / nt, 3), "detected": int(nd),
        "total_seizures": int(nt),
        "pooled_fa_per_24h": round(nf / hrs * 24, 2),
        "fa_mean": round(float(dfl["fa_per_24h"].mean()), 2),
        "fa_median": round(float(dfl["fa_per_24h"].median()), 2),
        "sensitivity_mean": round(float(dfl["sensitivity"].dropna().mean()), 3),
        "ci95": [round(x, 3) for x in boot_ci(dfl)],
    }

    stratB = (work / "results" / "lopo23" / "recalibB_per_patient_b10p0.csv")
    rb = pd.read_csv(stratB, encoding="utf-8-sig")
    out["strategy_B_pooled_ci"] = [round(x, 3) for x in boot_ci(rb)]
    out["strategy_B_pooled_prec_f1"] = pooled_precision_f1(rb)

    stratA = work / "results" / "lopo23" / "recalibA_per_patient_b10p0.csv"
    if stratA.exists():
        ra = pd.read_csv(stratA, encoding="utf-8-sig")
        out["strategy_A_pooled_ci"] = [round(x, 3) for x in boot_ci(ra)]
        out["strategy_A_pooled_prec_f1"] = pooled_precision_f1(ra)

    ext = work / "results" / "external_chbmit2siena.csv"
    if ext.exists():
        de = pd.read_csv(ext, encoding="utf-8-sig")
        cols = {c.lower(): c for c in de.columns}
        kd = cols.get("n_detected"); kf = cols.get("n_false_alarms")
        ks = cols.get("n_seizures"); kh = cols.get("total_hours") or cols.get("hours")
        de2 = pd.DataFrame({"n_seizures": de[ks], "n_detected": de[kd],
                            "n_false_alarms": de[kf], "hours": de[kh]})
        out["external_pooled_ci"] = [round(x, 3) for x in boot_ci(de2.dropna())]
        out["external_pooled_prec_f1"] = pooled_precision_f1(de2.dropna())

    # ---- C: 四法汇总精确率 / F1 ----
    files = {
        "SeizureNet_策略C": work / "results" / "lopo_chbmit_full18_seizurenet_23subj.csv",
        "SeizureNet_策略B": stratB,
        "传统特征RF": work / "results" / "lopo_chbmit_full18_trad23subj.csv",
        "ResNet1D": work / "results" / "lopo_chbmit_full18_resnet1d_23subj.csv",
    }
    tab = {}
    for k, f in files.items():
        d = pd.read_csv(f, encoding="utf-8-sig")
        prec, f1 = pooled_precision_f1(d)
        tab[k] = {"pooled_precision": prec, "pooled_f1": f1,
                  "pooled_sensitivity": round(float(d["n_detected"].sum()
                                                   / d["n_seizures"].sum()), 3),
                  "pooled_fa_per_24h": round(float(
                      (d["n_false_alarms"].sum()
                       / (d["total_hours"].sum() if "total_hours" in d.columns
                          else d["hours"].sum())) * 24), 2)}
    out["pooled_precision_f1"] = tab

    (work / "results" / "extras_23_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n-> work/results/extras_23_summary.json")


if __name__ == "__main__":
    main()
