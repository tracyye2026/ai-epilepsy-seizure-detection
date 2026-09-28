# -*- coding: utf-8 -*-
"""within_patient_loso.py —— 患者内对照模型的划分口径敏感性分析。

对每位受试者（chb21 并入 chb01），用同一套窗级特征（频谱/线长/方差/峰度/
Hjorth）与逻辑回归，比较两种划分：
  (a) 留一记录（leave-one-record-out）：留出 1 条记录的全部窗作测试，
      训练集只含该患者其余记录 —— 即本研究正文所报告的口径；
  (b) 窗级随机划分（10 折）：把该患者全部窗随机分折 —— 文献中最常见、
      也最容易因 2 s 步长 / 50% 重叠而产生相邻窗泄漏的划分。
两种口径的 AUC 差值说明"患者内 AUC 上界"在多大程度上依赖窗级随机划分，
用于回答审稿人"上界是否源于窗级随机划分（泄漏）"的质疑。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from probe_spectral import WIN, HOP, features_batch  # noqa: E402

work = ROOT / "work"
MERGE = {"chb21": "chb01"}
RNG = np.random.default_rng(0)


def subj(p):
    return MERGE.get(p, p)


def seizure_runs(y, win, hop):
    """把采样点级标签转成窗级的发作事件编号（每个发作事件一组窗起点）。"""
    m = (y == 1).astype(int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    starts_win = np.arange(0, y.size - win + 1, hop)
    ev = np.full(starts_win.size, -1, dtype=int)
    for k, (s, e) in enumerate(zip(starts, ends)):
        mid = starts_win + win / 2.0
        ev[(mid >= s) & (mid < e)] = k
    return starts_win, ev


def collect(subjects):
    mf = json.loads((work / "index" / "chbmit" / "manifest.json")
                    .read_text(encoding="utf-8"))
    win, hop = int(WIN * 128), int(HOP * 128)
    out = {}
    for p in subjects:
        recs = [m for m in mf if m.get("ok") and m.get("cache_path")
                and subj(m["patient"]) == p]
        data = []
        for m in recs:
            z = np.load(m["cache_path"])
            X = np.asarray(z["X"], dtype=np.float32)
            y = np.asarray(z["y"]).astype(int)
            starts, ev = seizure_runs(y, win, hop)
            ict = [i for i, e in enumerate(ev) if e >= 0]
            itc = [i for i, e in enumerate(ev) if e < 0 and y[starts[i]:starts[i] + win].mean() == 0]
            if not ict:
                continue
            n_neg = min(len(itc), 5 * len(ict))
            neg = RNG.choice(itc, size=n_neg, replace=False) if itc else []
            idxs = np.array(ict + list(neg))
            labels = np.r_[np.ones(len(ict)), np.zeros(len(neg))].astype(int)
            ev_ids = np.r_[ev[idxs[:len(ict)]], np.full(len(neg), -1)].astype(int)
            W = np.stack([X[:, starts[i]:starts[i] + win] for i in idxs])
            F = features_batch(W, 128.0)
            data.append({"record": m["record"], "F": F, "y": labels, "ev": ev_ids})
        if data:
            out[p] = data
    return out


def loo_record(data):
    aucs = []
    for i, d in enumerate(data):
        tr = [x for j, x in enumerate(data) if j != i]
        if not tr:
            continue
        Xtr = np.concatenate([x["F"] for x in tr])
        ytr = np.concatenate([x["y"] for x in tr])
        if len(np.unique(ytr)) < 2 or len(np.unique(d["y"])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr, ytr)
        aucs.append(roc_auc_score(d["y"], clf.predict_proba(d["F"])[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def split_random(data, n_splits=10, seed=0):
    """对照口径：把该患者全部窗随机分成 n_splits 折（窗级随机划分）。

    这是文献中最常见、也最容易产生信息泄漏的划分方式：同一记录的相邻
    窗（2 s 步长、50% 重叠）会同时出现在训练集与测试集。若患者内 AUC
    上界主要由该泄漏造成，则它与留一记录法的差值应当很大。
    """
    rng = np.random.default_rng(seed)
    F = np.concatenate([d["F"] for d in data])
    y = np.concatenate([d["y"] for d in data])
    idx = rng.permutation(len(y))
    aucs = []
    for te in np.array_split(idx, n_splits):
        tr = np.setdiff1d(np.arange(len(y)), te, assume_unique=False)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(F[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def main():
    mf = json.loads((work / "index" / "chbmit" / "manifest.json")
                    .read_text(encoding="utf-8"))
    subjects = sorted({subj(m["patient"]) for m in mf if m.get("ok")})
    want = sys.argv[1:] or subjects
    data = collect([p for p in subjects if p in want])
    rows = []
    for p, d in data.items():
        r = loo_record(d)              # 留一记录（本研究方法）
        q = split_random(d)            # 窗级随机划分（对照，易泄漏）
        rows.append({"患者": p, "记录数": len(d),
                     "留一记录AUC": round(r, 3) if not np.isnan(r) else None,
                     "窗级随机AUC": round(q, 3) if not np.isnan(q) else None})
        print(f"{p:>7}  留一记录 {r:.3f}   窗级随机 {q:.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = work / "results" / "within_patient_split_sensitivity.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    sub = df.dropna()
    print(f"\nn={len(sub)}  留一记录 均值 {sub['留一记录AUC'].mean():.3f} "
          f"(min {sub['留一记录AUC'].min():.3f})  "
          f"窗级随机 均值 {sub['窗级随机AUC'].mean():.3f} "
          f"(min {sub['窗级随机AUC'].min():.3f})  "
          f"Δ={(sub['窗级随机AUC'] - sub['留一记录AUC']).mean():+.3f}")
    for p in ("chb12", "chb13", "chb14"):
        if p in sub["患者"].values:
            r = sub[sub["患者"] == p].iloc[0]
            print(f"  {p}: 留一记录 {r['留一记录AUC']}  窗级随机 {r['窗级随机AUC']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
