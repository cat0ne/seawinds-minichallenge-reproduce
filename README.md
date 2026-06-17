# Sea Winds Mini Challenge — Winning Submission, Reproducible From Scratch

**Author:** Guillaume Hochard, PhD — competition alias **Hegoa**
**Competition:** [Codabench #13566](https://www.codabench.org/competitions/13566/) — Sea Winds Mini Challenge (Capgemini Invent Lab)
**This repository:** the **first-place** submission (`predictions.csv`) together with everything
needed to **rebuild it byte-for-byte from the provided competition data alone** — no external
data, no pre-built intermediate, one committed model.

`predictions.csv` is 576 rows = 8 windows × 6 buoys × 3 horizons (+1/+7/+14 days) × 4 hours,
each row giving a wind-speed interval (`q05`, `q50`, `q95`) and a direction (`dir_50`).

---

## For the organizers — what you asked for

| Requested | Provided here |
|---|---|
| **Code** (repo link or archive) | This repository (public). |
| **Environment** (Python version + pinned deps) | `requirements.txt` — every dependency pinned to an exact version; **Python 3.12.6**. These are the versions that yield byte-identical output. |
| **README** (setup + run end-to-end, with the reproduce command) | The **TL;DR** below: create the env, get the dataset, run one command to regenerate `predictions.csv`. |
| **Trained models** (weights, so you can run inference directly) | `models/dir_models.joblib` is committed (4.2 MB). The default `reproduce.py` run **loads these weights and runs inference** — you do *not* need to retrain. (Retraining the one model from the provided data is optional, via `--retrain`.) |

The single input is the organizers' dataset (Zenodo 20317029); nothing else is downloaded or used.

---

## TL;DR — reproduce from scratch

```bash
# 1. Clone and create the exact Python environment (Python 3.12.6)
git clone https://github.com/cat0ne/seawinds-minichallenge-reproduce.git
cd seawinds-minichallenge-reproduce
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the organizers' dataset (the ONLY input) from Zenodo and extract it
#    https://zenodo.org/records/20317029
#    -> extract so that <dir>/buoy_metadata.json exists
#    place it at ./mini_challenge_dataset  (or pass --data <dir> below)

# 3. Rebuild the submission from the dataset and verify it is byte-identical
python reproduce.py --data /path/to/mini_challenge_dataset
```

Expected final lines:

```
  d1 : max|Δspeed|=0.0000  max|Δdir|=0.000
  d7 : max|Δspeed|=0.0000  max|Δdir|=0.000
  d14: max|Δspeed|=0.0000  max|Δdir|=0.000
RESULT: BYTE-IDENTICAL to the official submission ✓
```

The script writes `predictions_reproduced.csv` and asserts it equals the shipped
`predictions.csv`. You can confirm independently:

```bash
cmp predictions_reproduced.csv predictions.csv && echo OK   # no output + "OK"
shasum -a 256 -c checksums.sha256                           # predictions.csv: OK / model: OK
```

---

## Full traceability: dataset → predictions

This is the whole point. **Every** scored column is recomputed in `reproduce.py` from
`mini_challenge_dataset/`. There is no cached intermediate submission anywhere in the chain.

| Horizon | Speed interval `q05/q50/q95` | Direction `dir_50` |
|---|---|---|
| **d1**  | conformal — HRES (5 m) + per-station 5/50/95 quantiles of `(obs − HRES_5m)` on provided 2020–21 | raw HRES direction − 2.5° |
| **d7**  | starting-kit notebook LightGBM (9 feats, train-2020, default seed) + storm widen | **3-vector blend: 0.40·HRES + 0.40·ML(sin/cos) + 0.20·speed-weighted clim-median** ← the winning lever |
| **d14** | provided 2020–21 seasonal climatology + asym widen (+0.3/+3.0) + extreme boost | speed-weighted circular-**median** climatology (provided obs); buoy **42001** fall windows use a pooled-evidence median |

The single machine-learned component is the **d7-direction** sin/cos gradient-boosted ensemble.
Its trained weights are committed at `models/dir_models.joblib` (the organizers require the model
artifact), and they are themselves rebuildable from the provided data:

```bash
python reproduce.py --data /path/to/mini_challenge_dataset --retrain
```

`--retrain` rebuilds the d7-direction GBM from the provided dataset before predicting. All eight
deterministic column-groups stay byte-identical; the d7 direction matches to within a few degrees
(LightGBM boosters with early stopping are not bit-identical across machines — see
`COMPLIANCE.md`). This proves the one model carries no external data.

See **`METHODOLOGY.md`** for *why* each choice was made and **`COMPLIANCE.md`** for the
provided-data-only / no-external-data proof.

---

## What's in this repository

```
predictions.csv          THE winning submission (q05,q50,q95,dir_50 per window/buoy/horizon/hour)
reproduce.py             provided dataset -> predictions.csv, asserts byte-identity (the source of truth)
requirements.txt         exact pinned versions for byte-identical reproduction (Python 3.12.6)
checksums.sha256         SHA-256 of predictions.csv and the model file
METHODOLOGY.md           the approach: per-horizon design and the key ideas
COMPLIANCE.md            data-provenance proof — only the provided dataset is used
models/
  dir_models.joblib      the ONLY trained model — d7-direction GBM (external-data-free features)
code/src/                the library modules reproduce.py imports (auditable)
  pipeline.py              features, the d7-direction model, training + prediction helpers
  features.py              feature engineering (HRES / reanalysis / buoy context / season)
  compliant_clim.py        d14 speed seasonal climatology from provided 2020–21 obs
  d14_dir_climatology.py   speed-weighted circular-median direction climatology
  __init__.py
mini_challenge_dataset/  NOT included — download from Zenodo (it is the organizers' data)
```

`code/src/` contains exactly the modules `reproduce.py` imports — nothing dead, nothing with a
broken import. `reproduce.py` is the single, self-contained driver and the authoritative record
of how the submission is built.

---

## The winning move, in one paragraph

The runner-up chain set the d7 direction to a 50/50 vector mean of the ML ensemble and raw HRES.
Direction is scored by **circular MAE**, whose loss-minimizing estimate is the **circular
median** — so folding the speed-weighted circular-median direction *climatology* into the d7
blend (weight 0.20, a robust cross-year plateau between 0.15 and 0.25) nudges the estimate toward
the metric's optimum. This touches **only the 192 d7 `dir_50` cells**; every speed/Winkler column
and the d1/d14 directions are unchanged. It took the d7 circular-MAE dimension to its rank-1
position and the submission to first place overall.

---

*Reproduced and verified on macOS / Python 3.12.6 with the pinned `requirements.txt`. Questions:
open an issue.*
