# -*- coding: utf-8 -*-
"""build_heterogeneity_23.py —— 按 23 受试者合并口径（chb21→chb01）重算异质性表。

输出：
  work/tables/tableX_heterogeneity_23.csv
  work/results/heterogeneity_23_summary.json
仅 chb01（合并 75 条记录）需重算患者内 AUC；其余受试者记录集未变，沿用
spectral_probe.csv（患者内 AUC 只依赖该患者自身记录，与模型无关）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from probe_spectral import WIN, HOP, features_batch  # noqa: E402

work = ROOT / "work"
MERGE = {"chb21": "chb01"}


def subj(p):
    return MERGE.get(p, p)


def load_manifest():
    return json.loads((work / "index" / "chbmit" / "manifest.json")
                      .read_text(encoding="utf-8"))


def records_of(mf, p):
    return [m for m in mf
            if m.get("ok") and m.get("cache_path") and subj(m["patient"]) == p]


def amplitude_ratio(recs):
    vals = []
    for m in recs:
        z = np.load(m["cache_path"])
        X = np.asarray(z["X"])
        y = np.asarray(z["y"])
        i1 = np.where(y == 1)[0]
        i0 = np.where(y == 0)[0]
        if len(i1) == 0 or len(i0) == 0:
            continue
        a1 = float(np.abs(X[:, i1]).mean())
        a0 = float(np.abs(X[:, i0[::5]]).mean())
        vals.append(a1 / max(a0, 1e-9))
    return float(np.mean(vals)) if vals else np.nan


def build_records(recs, sf=128.0):
    win, hop = int(WIN * sf), int(HOP * sf)
    data = []
    for m in recs:
        z = np.load(m["cache_path"])
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"]).astype(int)
        starts = np.arange(0, X.shape[-1] - win + 1, hop)
        ict, itc = [], []
        for s in starts:
            frac = y[s:s + win].mean()
            if frac >= 0.98:
                ict.append(s)
            elif frac == 0.0:
                itc.append(s)
        if not ict:
            continue
        rng = np.random.default_rng(0)
        n_neg = min(len(itc), 5 * len(ict))
        neg = rng.choice(itc, size=n_neg, replace=False) if itc else []
        idxs = list(ict) + list(neg)
        labels = np.r_[np.ones(len(ict)), np.zeros(len(neg))]
        W = np.stack([X[:, s:s + win] for s in idxs])
        data.append((m["record"], features_batch(W, sf), labels))
    return data


def within_auc(recs, sf=128.0):
    data = build_records(recs, sf)
    aucs, n_pos = [], []
    for i, (rec, F, ylab) in enumerate(data):
        tr = [d for j, d in enumerate(data) if j != i]
        if not tr:
            continue
        Xtr = np.concatenate([d[1] for d in tr])
        ytr = np.concatenate([d[2] for d in tr])
        if len(np.unique(ytr)) < 2 or len(np.unique(ylab)) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr, ytr)
        aucs.append(roc_auc_score(ylab, clf.predict_proba(F)[:, 1]))
        n_pos.append(int(ylab.sum()))
    return (float(np.mean(aucs)) if aucs else np.nan,
            int(sum(n_pos)), len(data))


def main():
    mf = load_manifest()
    subjects = sorted({subj(m["patient"]) for m in mf if m.get("ok")})
    lopo = pd.read_csv(work / "results" / "lopo_chbmit_full18_seizurenet_23subj.csv",
                       encoding="utf-8-sig")
    stratB = pd.read_csv(work / "results" / "lopo23" / "recalibB_per_patient_b10p0.csv",
                         encoding="utf-8-sig")
    auc_cross = dict(zip(lopo["patient"], lopo["sample_auc"]))
    sens_b = dict(zip(stratB["patient"], stratB["sensitivity"]))
    fa_b = dict(zip(stratB["patient"], stratB["fa_per_24h"]))
    nsz = dict(zip(stratB["patient"], stratB["n_seizures"]))

    probe_old = pd.read_csv(work / "results" / "spectral_probe.csv", encoding="utf-8-sig")
    probe_map = dict(zip(probe_old["患者"], probe_old["留一记录AUC均值"]))

    rows = []
    for p in subjects:
        recs = records_of(mf, p)
        amp = amplitude_ratio(recs)
        if p == "chb01":                      # 合并受试者：重算患者内上界
            w_auc, npos, nrec = within_auc(recs)
            print(f"  chb01(合并) 重算患者内 AUC = {w_auc:.3f} "
                  f"(含发作记录 {nrec}，发作窗 {npos})", flush=True)
        else:
            w_auc, npos, nrec = probe_map.get(p, np.nan), None, None
        rows.append({
            "受试者": p,
            "记录数": len(recs),
            "幅度比": round(amp, 2) if not np.isnan(amp) else None,
            "跨患者窗级AUC": round(float(auc_cross[p]), 3),
            "患者内AUC上界": round(float(w_auc), 3),
            "策略B灵敏度": round(float(sens_b[p]), 3),
            "策略B_FA24h": round(float(fa_b[p]), 2),
            "发作次数": int(nsz[p]),
        })
    df = pd.DataFrame(rows).sort_values("幅度比", ascending=False)
    df["泛化缺口"] = (df["患者内AUC上界"] - df["跨患者窗级AUC"]).round(3)
    out = work / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tableX_heterogeneity_23.csv", index=False, encoding="utf-8-sig")

    d = df.dropna(subset=["幅度比", "跨患者窗级AUC"])
    rho, pval = spearmanr(d["幅度比"], d["跨患者窗级AUC"])
    n_all = len(d)
    grp1 = d[(d["跨患者窗级AUC"] < 0.80)]
    grp2 = d[(d["跨患者窗级AUC"] >= 0.80) & (d["策略B灵敏度"] < 0.5)]
    summary = {
        "n_subjects": int(n_all),
        "spearman_rho": round(float(rho), 3),
        "spearman_p": float(pval),
        "within_auc_mean": round(float(d["患者内AUC上界"].mean()), 3),
        "within_auc_min": round(float(d["患者内AUC上界"].min()), 3),
        "within_auc_ge_0944": int((d["患者内AUC上界"] >= 0.944).sum()),
        "cross_auc_mean": round(float(d["跨患者窗级AUC"].mean()), 3),
        "cross_auc_min": round(float(d["跨患者窗级AUC"].min()), 3),
        "cross_auc_lt_080": {"subjects": list(grp1["受试者"]),
                             "seizures": int(grp1["发作次数"].sum())},
        "auc_ge080_sens_lt_050": {"subjects": list(grp2["受试者"]),
                                  "seizures": int(grp2["发作次数"].sum())},
        "failure_total_seizures": int(grp1["发作次数"].sum() + grp2["发作次数"].sum()),
        "seizures_total": int(d["发作次数"].sum()),
        "sens_ge_050": int((d["策略B灵敏度"] >= 0.5).sum()),
        "sens_eq_100": int((d["策略B灵敏度"] >= 0.999).sum()),
        "amp_ge2_auc_min": round(float(d[d["幅度比"] >= 2]["跨患者窗级AUC"].min()), 3),
        "amp_le14_auc_range": [round(float(d[d["幅度比"] <= 1.4]["跨患者窗级AUC"].min()), 3),
                               round(float(d[d["幅度比"] <= 1.4]["跨患者窗级AUC"].max()), 3)],
    }
    (work / "results" / "heterogeneity_23_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n", df.to_string(index=False))
    print("\n", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n-> {out/'tableX_heterogeneity_23.csv'}")


if __name__ == "__main__":
    main()
