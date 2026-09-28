# -*- coding: utf-8 -*-
"""measure_delay_23.py —— 在**主终点工作点**（策略 B：q=0.99, 下限 0.0,
最短报警 15 s）下测量 23 受试者的事件级检出延迟，使内部与外部（Siena，
冻结工作点）的延迟口径可比。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.metrics import (movavg_same, events_from_mask, merge_events,  # noqa: E402
                           match_events)

work = ROOT / "work"
CACHE = work / "recalib_cache"
Q, FLOOR, MIN_DUR = 0.99, 0.0, 15.0
SMOOTH, MERGE_GAP, OVERLAP = 8.0, 10.0, 1.0


def iter_cache(p):
    d = np.load(p)
    i = 0
    while f"{i}_p" in d:
        yield d[f"{i}_y"], d[f"{i}_p"], float(d[f"{i}_fs"][0])
        i += 1
    d.close()


def main():
    rows = []
    all_delays = []
    for f in sorted(CACHE.glob("*__test_*.npz")):
        subj = f.name.split("__test_")[0]
        delays = []
        n_det = n_true = 0
        for y, p, sf in iter_cache(f):
            ps = movavg_same(p, max(1, int(round(SMOOTH * sf))))
            thr = max(float(np.quantile(ps, Q)), FLOOR)
            pred = merge_events(events_from_mask(ps >= thr, sf),
                               gap_sec=MERGE_GAP, min_dur_sec=MIN_DUR)
            true = events_from_mask(y, sf)
            pairs, _, _ = match_events(pred, true, OVERLAP)
            n_det += len(pairs)
            n_true += len(true)
            for i, j in pairs:
                delays.append(pred[i, 0] - true[j, 0])
        if delays:
            rows.append({"patient": subj, "n_detected": n_det, "n_seizures": n_true,
                         "delay_median": float(np.median(delays))})
            all_delays.extend(delays)
    df = pd.DataFrame(rows)
    out = {
        "operating_point": {"quantile": Q, "floor": FLOOR,
                            "min_event_sec": MIN_DUR, "smooth_sec": SMOOTH},
        "n_patients": len(df),
        "n_matched_events": int(len(all_delays)),
        "delay_median_sec": round(float(np.median(all_delays)), 1),
        "delay_q1q3": [round(float(np.percentile(all_delays, 25)), 1),
                       round(float(np.percentile(all_delays, 75)), 1)],
        "per_patient_median_median": round(float(df["delay_median"].median()), 1),
    }
    (work / "results" / "delay23_strategyB.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # 外部：按患者中位数再取汇总（与论文 15.1 s 口径核对）
    ext = pd.read_csv(work / "results" / "external_chbmit2siena.csv",
                      encoding="utf-8-sig")
    d = ext["delay_median_sec"].dropna()
    print(f"\n外部 Siena：逐例中位延迟的均值 {d.mean():.1f} s，"
          f"中位数 {d.median():.1f} s（n={len(d)}）")


if __name__ == "__main__":
    main()
