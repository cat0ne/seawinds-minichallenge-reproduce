"""Median-matched & per-buoy direction climatology for d14 (provided-data only).

The d14 metric is circular MAE, whose loss-minimizing point estimate is the
circular MEDIAN — not the mean. Every prior d14-direction method predicted the
mean (or ML regressing toward it). This module builds per-(station, day-of-year)
circular-median direction climatologies from the 2020-21 buoy history, optionally
speed-weighted, and predicts with a circular nearest-doy fallback.

No tuned hyperparameters: doy half-width fixed at 15 days by robustness, grid
search resolution fixed. Nothing is selected on 2020-21 cMAE (which is
anti-predictive for this dim — see feedback_validation_cannot_guard_2022).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

import numpy as np
import pandas as pd

DOY_HALFWIDTH = 15          # +/- days pooled around the target doy
GRID_COARSE = 1.0           # degrees, first pass
GRID_FINE = 0.1             # degrees, refine around coarse winner
MIN_N = 20                  # need this many obs in window, else widen


def _circ_dist(a_deg, b_deg):
    """Absolute circular distance in degrees, vectorised over a_deg."""
    d = np.abs(a_deg - b_deg) % 360.0
    return np.minimum(d, 360.0 - d)


def circular_median(deg, weights=None):
    """Angle minimizing the (weighted) sum of absolute circular distances.

    This is the Bayes-optimal point estimate for circular MAE. Solved by a
    coarse 1-degree grid search refined to 0.1 degree around the winner.
    """
    deg = np.asarray(deg, float)
    m = np.isfinite(deg)
    deg = deg[m]
    if len(deg) == 0:
        return None
    if weights is None:
        w = np.ones(len(deg))
    else:
        w = np.asarray(weights, float)[m]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() <= 0:
            w = np.ones(len(deg))

    def cost(cands):
        # cands: (G,), deg: (N,) -> (G,)
        dd = _circ_dist(cands[:, None], deg[None, :])  # (G, N)
        return (dd * w[None, :]).sum(axis=1)

    coarse = np.arange(0.0, 360.0, GRID_COARSE)
    c0 = coarse[np.argmin(cost(coarse))]
    fine = np.arange(c0 - GRID_COARSE, c0 + GRID_COARSE + GRID_FINE, GRID_FINE) % 360.0
    best = fine[np.argmin(cost(fine))]
    return float(best % 360.0)


def _doy_circ_dist(doy_a, doy_b, period=365.25):
    d = np.abs(doy_a - doy_b) % period
    return np.minimum(d, period - d)


def build_dir_climatology(buoy_obs: dict[str, pd.DataFrame],
                          doy_halfwidth: int = DOY_HALFWIDTH,
                          speed_weighted: bool = False) -> pd.DataFrame:
    """Per-(station, doy) circular-median direction table.

    buoy_obs: {station -> raw obs df with time, wind_direction[, wind_speed]}.
    Returns df [station, doy, dir_median, n].
    """
    rows = []
    for stn, o in buoy_obs.items():
        o = o.copy()
        o['time'] = pd.to_datetime(o['time'])
        o = o.dropna(subset=['wind_direction'])
        if len(o) == 0:
            continue
        doy = o['time'].dt.dayofyear.values.astype(float)
        wd = o['wind_direction'].values.astype(float)
        sp = (o['wind_speed'].values.astype(float)
              if speed_weighted and 'wind_speed' in o.columns else None)
        for target_doy in range(1, 367):
            dd = _doy_circ_dist(doy, target_doy)
            mask = dd <= doy_halfwidth
            hw = doy_halfwidth
            while mask.sum() < MIN_N and hw < 183:
                hw += 10
                mask = dd <= hw
            if mask.sum() == 0:
                continue
            w = sp[mask] if sp is not None else None
            med = circular_median(wd[mask], weights=w)
            if med is None:
                continue
            rows.append((str(stn), int(target_doy), med, int(mask.sum())))
    return pd.DataFrame(rows, columns=['station', 'doy', 'dir_median', 'n'])


def predict_dir_clim(table: pd.DataFrame, stations, doys) -> np.ndarray:
    """Look up median direction per (station, doy) with circular nearest-doy fallback."""
    stations = np.asarray(stations).astype(str)
    doys = np.asarray(doys).astype(int)
    out = np.full(len(stations), np.nan)
    by_stn = {s: g.set_index('doy')['dir_median'].to_dict()
              for s, g in table.groupby('station')}
    stn_doys = {s: np.array(sorted(g.keys())) for s, g in by_stn.items()}
    for i, (s, d) in enumerate(zip(stations, doys)):
        lut = by_stn.get(s)
        if not lut:
            continue
        if d in lut:
            out[i] = lut[d]
        else:
            avail = stn_doys[s]
            j = int(np.argmin(_doy_circ_dist(avail.astype(float), float(d))))
            out[i] = lut[int(avail[j])]
    return out
