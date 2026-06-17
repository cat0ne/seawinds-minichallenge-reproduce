# Compliance & Reproducibility Statement

**Author:** Guillaume Hochard, PhD (alias *Hegoa*) · Codabench #13566

This document shows that the winning submission (`predictions.csv`) is produced **only from the
provided competition data** and is **fully reproducible from scratch**.

---

## 1. The rule

Mini-Challenge FAQ Rule #3 — permitted inputs: *the provided mini-challenge dataset (buoy
observations, HRES forecasts, reanalysis 0.25° patches, ETOPO) and foundation models. **No
other external data.*** Rule #4 — d14 is predicted from buoy history / climatology built from
the provided 2020–21 observations.

## 2. Full traceability: provided dataset → predictions

`reproduce.py` rebuilds **every** scored column from `mini_challenge_dataset/` and asserts the
result is byte-identical to `predictions.csv`. There is **no pre-built intermediate file**.
Each component and its provided-data source:

| Column | Built from | Type |
|---|---|---|
| d1 speed | HRES (5 m) + per-station 5/50/95 quantiles of `(obs − HRES_5m)`, provided 2020–21 | deterministic |
| d1 dir | provided HRES direction + global −2.5° | deterministic |
| d7 speed | starting-kit LightGBM (9 feats) trained on provided 2020 HRES+obs + storm widen | deterministic |
| d7 dir | 0.40·HRES ⊕ 0.40·ML(sin/cos on `DIR_FEATURES`) ⊕ 0.20·speed-weighted clim-median (provided obs) | **trained model** + deterministic blend |
| d14 speed | provided 2020–21 seasonal climatology + widen + storm boost | deterministic |
| d14 dir | speed-weighted circular-median climatology from provided observations | deterministic |
| d14 dir @42001 (w6-8) | pooled provided evidence: buoy obs + bias-corrected provided ERA5 patch + provided window contexts | deterministic |

Run it:

```bash
python reproduce.py --data /path/to/mini_challenge_dataset
# -> d1/d7/d14: max|Δspeed|=0.0000  max|Δdir|=0.000
# -> RESULT: BYTE-IDENTICAL to the official submission ✓
```

## 3. The single trained model

The **only** machine-learned component is the **d7-direction** GBM ensemble
(`models/dir_models.joblib`). It is trained exclusively on `DIR_FEATURES`, which contains
**no external data** (HRES fields, the provided reanalysis patch, buoy context, day-of-year).
Its training code is `train_direction_models` in `code/src/pipeline.py`. The committed weights
are loaded for exact reproduction; the deterministic 0.40/0.40/0.20 vector blend that turns them
into the final d7 direction lives in `reproduce.py`.

`python reproduce.py --retrain` rebuilds this model from the provided data. Note: LightGBM
boosters with early stopping are **not bit-identical across machines/library builds**, so a
retrain reproduces the d7 *direction* to within a few degrees rather than to the bit (all
eight other column-groups remain byte-identical). For exact reproduction of the scored
submission the trained weights are shipped; for provenance, `--retrain` proves they come
from the provided data and the documented code. (No other column depends on a trained model:
d1 speed is conformal, d14 speed/direction are climatology, d7 speed is the notebook model.)

## 4. No external data — verifiable two ways

1. **Nothing external ships.** This package contains no NDBC multi-decade archive, no network
   data, and no download/scrape code. A scan of `code/` for `requests`, `urllib`, `wget`,
   `http(s)://`, `noaa.gov`, `open-meteo`, etc. returns nothing live.
2. **It runs offline and still reproduces.** `reproduce.py` makes no network calls; run it with
   networking disabled — it still yields the byte-identical submission.

### Note on legacy code hooks
`code/src/pipeline.py` is the project's real pipeline module and retains, for transparency,
**disabled** hooks from earlier development that once referenced an external NDBC long-record
archive (`_load_ndbc_long_clim`, `_ndbc_speed_clim_lookup`, and a d14-direction branch in
`predict_direction`). In this package the first two **hard-`return None` on their first line**
(the code below is unreachable), the d14-direction branch is guarded by the existence of an
external file that is **not present**, and the submission's d14 speed/direction instead use the
**provided-data** climatology (`compliant_clim.py`, `d14_dir_climatology.py`). They have no
effect on the reproduced output — `reproduce.py` never calls them, and removing them changes
nothing.

## 5. Environment

`requirements.txt` pins the exact versions that yield byte-identical output, on **Python
3.12.6**: numpy 1.26.4, pandas 3.0.3, lightgbm 4.6.0, scikit-learn 1.8.0, scipy 1.17.1,
joblib 1.5.3, pyarrow 24.0.0. `checksums.sha256` records the SHA-256 of `predictions.csv` and
the model file; verify with `shasum -a 256 -c checksums.sha256`.

---

*Summary: the submission uses only the provided competition data; it is rebuilt from that
data, from scratch, by `reproduce.py`, and the result is byte-identical to the file scored on
Codabench. The one trained model uses external-data-free features and is reproducible from the
provided data.*
