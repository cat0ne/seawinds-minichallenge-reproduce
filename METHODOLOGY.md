# Methodology — Sea Winds Mini Challenge (winning submission)

**Author:** Guillaume Hochard, PhD (alias *Hegoa*) · Codabench #13566

A plain-language explanation of what the submission does and why, followed by the
per-component detail. Everything here is built from the **provided** competition data only.

---

## 1. The problem

We are given the raw ECMWF **HRES** wind forecast (interpolated to 6 Gulf-of-Mexico /
Caribbean buoys) plus a local reanalysis patch and the buoys' recent history, and must turn
it into a **calibrated probabilistic** forecast: at +1, +7 and +14 days, output a wind-speed
interval (`q05`, `q50`, `q95`) and a wind direction (`dir_50`). We train on 2020–2021 and
are scored on 8 two-week windows across 2022.

## 2. How it is scored (this shapes every choice)

The final standing is the **average rank over 9 dimensions** = {Winkler interval score,
circular MAE, extreme Winkler} × {d1, d7, d14}.

- **Winkler (90% interval):** `width + (2/α)·shortfall`, α=0.10 → a miss outside the band
  costs **20×** the distance. The optimal band is *the narrowest that still covers ~90%*.
- **Circular MAE:** direction error on the compass (the short way around). Its
  loss-minimizing point estimate is the **circular median**, not the mean — a fact we exploit.
- **Extreme Winkler:** the same interval score but only over storm observations (>13 m/s);
  high-variance, dominated by a handful of points.

## 3. The one big idea

> **Match each horizon to the tool whose skill survives that far out, and match each
> estimator to the metric it is scored by.**

- **+1 day:** HRES is excellent → trust it, add an honest empirical error margin (conformal).
- **+7 days:** HRES has decayed but context still helps → a light gradient-boosted model,
  and for direction blend in the climatological median (see 4.4).
- **+14 days:** HRES is gone → fall back to **climatology** ("what is normal here, now?").

And because direction is graded by circular MAE, the long-range direction estimators are
**circular medians**, not means.

---

## 4. The components (each is rebuilt from provided data in `reproduce.py`)

### 4.1 d1 speed — conformal prediction
For each buoy, take every `(observed − HRES_5m)` error on 2020–2021 and use its empirical
5/50/95 percentiles as the interval offsets around today's HRES. The band covers ~90% by
construction. We verified (cross-year) this sits at the Winkler optimum at d1 — no ML beats
it. *(HRES 10 m is downscaled to the 5 m anemometer height; offsets are per-station.)*

### 4.2 d1 direction — raw HRES − 2.5°
HRES direction is highly skilled at +24 h; ML only adds noise. HRES carries a tiny, *stable*
clockwise bias year-over-year, so we apply a single global **−2.5°** correction.
(Per-station offsets were tried and rejected — they swing 5–7° between years and don't transfer.)

### 4.3 d7 speed — the starting-kit model + storm widen
The decisive lesson of this competition: our heavily engineered d7 pipeline was
**over-wide** for the calmer 2022. The deterministic **starting-kit notebook model** (a
9-feature LightGBM, train-2020, default seed — exactly what an unmodified notebook produces)
has ~38% narrower bands and **wins** the 2022 Winkler. We use it directly, and keep one
piece of insurance: where HRES forecasts >9 m/s we raise `q95` to ≥14 m/s to cover the rare
storm (protects the extreme-Winkler dimension).

### 4.4 d7 direction — 3-vector blend: HRES + ML + climatological median ← *the winning lever*
The base estimate is a small LightGBM **sin/cos** ensemble (3 seeds, early-stopped) trained on
`DIR_FEATURES` (HRES, reanalysis patch, buoy context, season — **no external data**). This is the
**only machine-learned model** in the submission; its weights ship in `models/dir_models.joblib`
and `--retrain` rebuilds it.

The submission's d7 direction is a **unit-vector blend of three sources**:

```
dir_d7 = arg( 0.40 · û(HRES) + 0.40 · û(ML) + 0.20 · û(clim-median) )
```

where `clim-median` is the **speed-weighted circular-median** direction climatology (the same
estimator used at d14, §4.6) evaluated at the target day-of-year. Rationale: d7 direction is
scored by circular MAE, whose optimum is the circular median; the ML+HRES vector mean drifts off
that optimum, and folding in the climatological median pulls it back. The weight **0.20** is the
measured cross-year plateau (gains hold flat for 0.15–0.25; both pure-blend and heavier-climatology
are worse). This changes **only the 192 d7 `dir_50` cells** — every speed/Winkler column and the
d1/d14 directions are byte-identical to the runner-up chain — and it is what moved d7 circular MAE
to rank 1 and the submission to first place.

### 4.5 d14 speed — provided seasonal climatology
Per (buoy, day-of-year, hour) empirical 5/50/95 wind-speed quantiles from the **provided
2020–2021** observations, with an asymmetric widen (`q05 +0.3`, `q95 +3.0`) and a
live-signal storm boost (`q95` up when pressure is low *and* falling, or recent winds are
high — never on static climatology alone, which fires on calm trade-wind days).

### 4.6 d14 direction — speed-weighted circular-MEDIAN climatology
Because d14 direction is scored by circular MAE, the right estimate is the **circular
median**, not the mean. We build a per-(buoy, day-of-year) **speed-weighted circular-median**
direction climatology from the provided observations, over **all six** buoys (the gain comes
precisely from the volatile Gulf buoys, where a 2-year ML model overfits worst). This moved
cMAE_d14 from 40.1° (ML) to 37.6°.

### 4.7 d14 direction at buoy 42001 — pooled-evidence median (the data-hole fix)
Buoy 42001 went offline on 2021-07-25 and returned in Aug-2022, so its three scored windows
(Sep/Nov/Dec) rest on a **single year (2020)** of training data — at a regime-flipping Gulf
site. We replace those cells with a speed-weighted circular median over **pooled provided
evidence**: the buoy's own 6-hourly history **+** the bias-corrected ERA5 reanalysis point
direction (the provided 0.25° patch, rotated to remove its small measured offset vs the buoy)
**+** the 2022 window contexts up to each target. This took cMAE_d14 from 37.6° to **35.1°**.
*(All three sources are provided data; the reanalysis patch is explicitly allowed.)*

---

## 5. How decisions were validated

Local "wins" on 2020–2021 repeatedly **failed to transfer** to 2022 (a forward 2020→2021
holdout was even anti-predictive). The reliable rule that emerged: prefer the **simplest
robust estimator**, match the estimator to the metric (median for MAE), cover every unit
including the hard ones, and let the held-out year decide. The biggest gains —
the starting-kit d7 speed, the median d14 direction, and the climatology-blended d7 direction —
all came from *removing* over-engineering and aligning the estimator with the metric, not from
adding signal.

---

*Appendix — code map (`code/src/`):* `pipeline.py` (features, the d7-direction sin/cos model,
training + prediction helpers), `features.py` (feature engineering), `compliant_clim.py`
(d14 speed climatology from provided obs), `d14_dir_climatology.py` (speed-weighted
circular-median direction climatology — used by both the d7 blend and d14). The end-to-end
driver is `reproduce.py` at the package root; it is self-contained over these modules and is the
authoritative record of how every column is built.
