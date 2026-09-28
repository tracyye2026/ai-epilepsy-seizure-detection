""" build_heterogeneity_table.py —— 汇总"跨患者泛化差异"的证据表。

每位患者一行，四列证据：
  1. 发作/间期 幅度比    |X_sz| / |X_inter|   （EEG 层面的可检测性）
  2. 患者内 上界 AUC      留一记录法频谱 logistic（probe_spectral.py 产物）
  3. 跨患者 窗级 AUC      LOPO 模型在该患者测试记录上的排序能力
  4. LOPO 事件级 灵敏度 / FA/24h
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
work = ROOT / "work"


def amplitude_ratio(patient: str, mf):
    recs = [m for m in mf if m["patient"] == patient
            and m.get("ok") and m.get("cache_path")]
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


def main():
    mf = json.loads((work / "index" / "chbmit" / "manifest.json")
                    .read_text(encoding="utf-8"))
    rows = []
    for m in mf:
        pass
    patients = sorted({m["patient"] for m in mf if m.get("ok")})

    # 1) 幅度比
    amp = {p: amplitude_ratio(p, mf) for p in patients}

    # 2) 患者内上界
    probe = work / "results" / "spectral_probe.csv"
    p2 = {}
    if probe.exists():
        d = pd.read_csv(probe)
        p2 = dict(zip(d["患者"], d["留一记录AUC均值"]))

    # 3)(4) LOPO 结果
    lopo = work / "results" / "lopo_chbmit_full18.csv"
    sens = fa = auc = {}
    if lopo.exists():
        d = pd.read_csv(lopo)
        col = "patient" if "patient" in d.columns else d.columns[0]
        sens = dict(zip(d[col], d.get("sensitivity", pd.Series(dtype=float))))
        fa = dict(zip(d[col], d.get("fa_per_24h", pd.Series(dtype=float))))
        # LOPO 已逐患者记录窗级 AUC（列名 sample_auc）
        if "sample_auc" in d.columns:
            auc = {k: (round(v, 3) if not np.isnan(v) else None)
                   for k, v in zip(d[col], d["sample_auc"])}
        else:
            auc = dict(zip(d[col], d.get("auc", pd.Series(dtype=float))))

    for p in patients:
        rows.append({
            "患者": p,
            "幅度比": round(amp[p], 2) if not np.isnan(amp[p]) else None,
            "跨患者窗级AUC": auc.get(p),
            "患者内AUC上界": p2.get(p),
            "LOPO灵敏度": sens.get(p),
            "LOPO_FA24h": round(fa[p], 1) if p in fa and not np.isnan(fa[p]) else None,
        })
    df = pd.DataFrame(rows)
    df["泛化缺口"] = (df["患者内AUC上界"] - df["跨患者窗级AUC"]).round(3)

    out = work / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tableX_heterogeneity.csv", index=False, encoding="utf-8-sig")
    df.to_markdown(out / "tableX_heterogeneity.md", index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\n-> {out/'tableX_heterogeneity.csv'}")
    return df


if __name__ == "__main__":
    main()
