# Lightweight Deep Learning for Scalp-EEG Seizure Detection (SeizureNet)

Code and result files for the manuscript:

> 轻量化深度学习模型在头皮脑电癫痫发作自动检测中的应用：基于留一患者法与跨数据库外部验证

The model (**SeizureNet**, ~72k parameters) is trained and evaluated under a strict
leave-one-patient-out (LOPO, 23 subjects after merging the two CHB-MIT recording
batches of the same patient) protocol on **CHB-MIT**, with external validation on
**Siena Scalp EEG**. The primary endpoints are event-level sensitivity and false
alarms per 24 hours (FA/24h), using a per-record rank-adaptive (quantile)
thresholding strategy ("Strategy B") that requires no labels at test time.

## Data access

Both datasets are openly available; **no patient data is distributed in this repository**:

| Dataset | Source | License | Used for |
|---|---|---|---|
| CHB-MIT Scalp EEG | PhysioNet <https://physionet.org/content/chbmit/1.0.0/> | ODC-By 1.0 | Training + internal LOPO validation |
| Siena Scalp EEG | PhysioNet <https://physionet.org/content/siena-scalp-eeg/1.0.0/> | CC BY 4.0 | External validation |

`fetch_dataset.py` downloads and SHA256-verifies the files via PhysioNet's HTTP API.

## Repository layout

```
├── config.py                 # all hyperparameters / paths (single source of truth)
├── preprocess.py             # EDF → canonical 18-channel bipolar montage → 4-s windows
├── train.py / train_lopo.py  # model training (per-fold), LOPO driver
├── run_lopo23.py             # 23-subject merged LOPO (chb21 → chb01)
├── evaluate.py               # continuous inference, smoothing, event-level matching
├── recalibrate.py            # post-hoc operating-point recalibration (Strategies A/B/C)
├── finalize_primary.py       # primary-endpoint table (Strategy B, budget 10 FA/24h)
├── external_validate.py      # CHB-MIT → Siena transfer with frozen operating point
├── ablation.py               # channel-set ablation
├── baseline_trad.py          # 180-dim hand-crafted features + Random Forest baseline
├── run_trad_scores.py        # RF continuous-score caching (one pass per fold)
├── eval_trad_recalib.py      # label-free RF + Strategy B evaluation from cached scores
├── calc_brier.py             # calibration analysis (pooled/per-fold Brier scores)
├── build_heterogeneity_23.py, within_patient_loso.py, measure_delay_23.py,
│   recompute_23_extras.py    # heterogeneity, within-patient upper bound, latency
├── measure_latency.py        # single-thread CPU inference latency benchmark
├── make_figures.py, redraw_fig45.py, make_tables.py   # figures / tables
├── data/  models/  utils/    # data pipeline, SeizureNet, metrics & plotting
└── work/results/             # all per-patient CSVs and summary JSONs reported in the paper
```

Key result files (all under `work/results/`):

- `lopo_chbmit_full18_seizurenet_23subj.csv` — per-fold LOPO results (23 subjects)
- `recalibB_*_b10p0.*` — primary endpoint (Strategy B, q=0.99, floor=0.0, min-event 15 s)
- `recalibB_trad_per_patient.csv` / `recalibB_trad_summary.json` — RF + Strategy B comparison
- `external_chbmit2siena.csv` / `external_summary_chbmit2siena.json` — external validation

## Environment

- Python ≥ 3.10, PyTorch (CPU is sufficient), numpy, scipy, pandas, scikit-learn, matplotlib, tqdm, joblib
- No GPU required: training one LOPO fold takes ~1 min/epoch on a modern laptop CPU.

## Reproduction pipeline (condensed)

```bash
python fetch_dataset.py                # download + verify CHB-MIT / Siena
python preprocess.py                   # windows + manifest
python build_cont_cache.py             # continuous-signal cache for evaluation
python run_lopo23.py                   # 23-subject LOPO training
python recalibrate.py                  # Strategy A/B/C operating points
python finalize_primary.py             # primary endpoint
python external_validate.py            # Siena external validation
python run_trad_scores.py              # RF baseline score cache (~2.5 min/fold)
python eval_trad_recalib.py            # RF + Strategy B row of Table 8
python calc_brier.py                   # calibration (Brier) analysis
python measure_latency.py              # CPU latency benchmark
```

## License

Code is released under the MIT License (see `LICENSE`).
EEG data remain under the licenses of their respective sources (see table above).
