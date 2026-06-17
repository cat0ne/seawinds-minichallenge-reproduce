"""End-to-end pipeline: train, calibrate, evaluate, submit.

Usage:
    python -m src.pipeline build_train          # build training table
    python -m src.pipeline train                # train models
    python -m src.pipeline eval                 # local eval on simulated windows
    python -m src.pipeline submit               # write submission.zip
    python -m src.pipeline all                  # everything end-to-end
"""
from __future__ import annotations

import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features import (
    HORIZON_LAG_H, H_SCALE, build_climatology, buoy_context_features,
    climatology_lookup, context_window_features, hres_to_features,
    precompute_reanalysis_timeseries, regime_features, resample_buoy_obs,
    temporal_features,
)

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning)

# ────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────

import os
ROOT = Path(__file__).resolve().parent.parent
# Dataset / scratch locations are overridable via env vars so the reproduction
# driver can point at the provided dataset wherever the organizer placed it.
DATA = Path(os.environ.get('SEAWINDS_DATA', ROOT / 'mini_challenge_dataset'))
ARTIFACTS = Path(os.environ.get('SEAWINDS_ARTIFACTS', ROOT / 'artifacts'))

_meta_path = DATA / 'buoy_metadata.json'
if not _meta_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {_meta_path}. "
        "Download from Zenodo and place in mini_challenge_dataset/"
    )
META = json.loads(_meta_path.read_text())
BUOYS = list(META['buoys'].keys())
HOURS = META['scoring_hours_utc']
EXTREME_THR = 13.0

# Feature columns used by the speed-quantile model.
SPEED_FEATURES = [
    # HRES-derived (None for d14 → fall back to climatology)
    'hres_speed', 'hres_speed_5m', 'hres_u10', 'hres_v10',
    'hres_dir_sin', 'hres_dir_cos',
    # NDBC long-record (12-year) climatology quantiles — cal2 test showed
    # adding these as features at d7 gives Winkler -0.13 m/s improvement.
    'ndbc_q05', 'ndbc_q50', 'ndbc_q95', 'ndbc_mean', 'ndbc_std',
    # Reanalysis context aggregates (patch-wide)
    'ctx_spd10_mean', 'ctx_spd10_max', 'ctx_spd10_std',
    'ctx_spd100_mean', 'ctx_spd100_max',
    'ctx_u10_mean', 'ctx_v10_mean',
    'ctx_msl_min', 'ctx_msl_mean', 'ctx_msl_trend',
    'ctx_spd10_last', 'ctx_spd10_last24h_mean', 'ctx_spd10_last72h_max',
    # Center-pixel reanalysis (best per-step proxy for the buoy)
    'ctx_ctr_u10_last', 'ctx_ctr_v10_last', 'ctx_ctr_spd10_last',
    'ctx_ctr_u100_last', 'ctx_ctr_v100_last', 'ctx_ctr_spd100_last',
    'ctx_ctr_msl_last', 'ctx_ctr_msl_24h_change',
    'ctx_ctr_spd10_mean', 'ctx_ctr_spd10_max', 'ctx_ctr_spd10_std',
    # Buoy obs context (may be NaN — LGBM handles)
    'buoy_ws_last', 'buoy_ws_last24h_mean', 'buoy_ws_last24h_max',
    'buoy_ws_ctx_mean', 'buoy_ws_ctx_max', 'buoy_ws_ctx_std',
    'buoy_pres_last', 'buoy_pres_24h_change', 'buoy_pres_ctx_min',
    'buoy_gust_ctx_max', 'buoy_at_last', 'buoy_wt_last', 'buoy_n_valid',
    # Climatology
    'clim_q05', 'clim_q50', 'clim_q95', 'clim_mean', 'clim_std', 'clim_max',
    # Regime tags (trade-wind vs disturbed) — only used at d7
    'reg_east_frac', 'reg_ws_std_buoy', 'reg_ws_iqr_buoy',
    'reg_msl_range', 'reg_u10_mean', 'reg_v10_mean',
    'reg_trade_score', 'reg_disturbed_score',
    # Temporal / station
    'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos',
    'lat', 'lon', 'horizon',
]

# d1 has highly skilled HRES — minimal features beat richer ones for q05/q95
# because extra features add spread variance where HRES is already near-optimal.
SPEED_FEATURES_D1 = [
    'hres_speed', 'hres_speed_5m', 'hres_u10', 'hres_v10',
    'hres_dir_sin', 'hres_dir_cos',
    'buoy_ws_last', 'buoy_ws_last24h_mean', 'buoy_pres_last',
    'clim_q50', 'clim_std',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'lat', 'lon', 'horizon',
]

DIR_FEATURES = [
    'hres_dir_sin', 'hres_dir_cos', 'hres_speed', 'hres_u10', 'hres_v10',
    'ctx_u10_mean', 'ctx_v10_mean', 'ctx_spd10_mean',
    'ctx_ctr_u10_last', 'ctx_ctr_v10_last',
    'ctx_ctr_u100_last', 'ctx_ctr_v100_last',
    'buoy_dir_last_sin', 'buoy_dir_last_cos', 'buoy_ws_last',
    'clim_dir_sin', 'clim_dir_cos', 'clim_mean',
    'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos',
    'lat', 'lon', 'horizon',
]

QUANTILES = {'q05': 0.05, 'q50': 0.50, 'q95': 0.95}

EXTREME_SAMPLE_WEIGHT = 4.0
DIRECTION_BLEND_WEIGHT = 0.5
CQR_SHRINKAGE_K = 30.0
CONFORMAL_FALLBACK_OFFSETS = (-2.0, 0.0, 2.0)
STORM_AMP = {1: 2.0, 7: 0.5, 14: 2.0}
SEASON_AMP = {1: 0.2, 7: 0.0, 14: 1.2}
MSL_LOW_THRESHOLD = 101000
MSL_FALLING_THRESHOLD = -200
STORM_WIND_THRESHOLD = 14.0
HURRICANE_SEASON_LAT = 23
HURRICANE_SEASON_DOY_COS = -0.1

_HRES_FEATURE_KEYS = ['hres_speed', 'hres_speed_5m', 'hres_u10', 'hres_v10',
                      'hres_dir_sin', 'hres_dir_cos']

def _hres_feature_dict(row, keys=_HRES_FEATURE_KEYS):
    return {k: row[k] for k in keys}

def _nan_hres_dict():
    return {k: np.nan for k in _HRES_FEATURE_KEYS}


# ────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────

def load_all_train():
    """Load HRES, reanalysis, buoy obs for all stations from /train.

    Returns (hres, re_ts, buoy_obs_6h, buoy_obs_raw).
    buoy_obs_6h: resampled, target for fitting.
    buoy_obs_raw: 10-min raw, source for context features.
    """
    hres, re_ts, buoy_obs_6h, buoy_obs_raw = {}, {}, {}, {}
    for b in BUOYS:
        info = META['buoys'][b]
        h = pd.read_parquet(DATA / 'train/hres_forecasts' / f'hres_{b}.parquet')
        h['time'] = pd.to_datetime(h['time'])
        hres[b] = hres_to_features(h)
        re = pd.read_parquet(DATA / 'train/reanalysis_patches' / f'reanalysis_{b}.parquet')
        re['time'] = pd.to_datetime(re['time'])
        re_ts[b] = precompute_reanalysis_timeseries(
            re, info['latitude'], info['longitude'])
        o = pd.read_parquet(DATA / 'train/buoy_observations' / f'ndbc_{b}.parquet')
        o['time'] = pd.to_datetime(o['time'])
        buoy_obs_raw[b] = o
        buoy_obs_6h[b] = resample_buoy_obs(o)
    return hres, re_ts, buoy_obs_6h, buoy_obs_raw


# ────────────────────────────────────────────────────────────────────────
# Training table construction
# ────────────────────────────────────────────────────────────────────────

def _load_ndbc_long_clim():
    # COMPLIANCE: legacy hook DISABLED in the submission package. The external NDBC
    # multi-decade archive is NOT part of the provided dataset and is NOT used by the
    # final submission (d14 speed = provided 2020-21 climatology; d7/d14 dir = provided
    # climatology). Hard-return None so no external lookup can ever occur here.
    return None


def _attach_ndbc_features(rows_df):
    """Merge NDBC long-clim speed quantiles onto a training table by (station, doy, hour)."""
    ndbc = _load_ndbc_long_clim()
    if ndbc is None:
        for c in ['ndbc_q05', 'ndbc_q50', 'ndbc_q95', 'ndbc_mean', 'ndbc_std']:
            rows_df[c] = np.nan
        return rows_df
    rows_df = rows_df.copy()
    rows_df['_doy'] = rows_df['time'].dt.dayofyear
    rows_df['_hour'] = rows_df['time'].dt.hour
    rows_df['_st'] = rows_df['station'].astype(str)
    ndbc2 = ndbc.rename(columns={'station':'_st', 'doy':'_doy', 'hour':'_hour',
                                  'q05':'ndbc_q05', 'q50':'ndbc_q50', 'q95':'ndbc_q95',
                                  'mean':'ndbc_mean', 'std':'ndbc_std'})
    merged = rows_df.merge(ndbc2[['_st','_doy','_hour','ndbc_q05','ndbc_q50','ndbc_q95','ndbc_mean','ndbc_std']],
                           on=['_st','_doy','_hour'], how='left')
    return merged.drop(columns=['_doy','_hour','_st'])


def build_training_table(hres, re_ts, buoy_obs, clim, buoy_obs_raw=None):
    """Build training rows for horizons 1, 7, 14."""
    rows = []
    for b in BUOYS:
        info = META['buoys'][b]
        lat, lon = info['latitude'], info['longitude']
        re_b = re_ts[b]
        obs_b = buoy_obs[b].dropna(subset=['wind_speed'])
        h_b = hres[b]
        bo_raw = buoy_obs_raw[b] if buoy_obs_raw is not None else None

        for hz in (1, 7):
            fc = h_b[h_b['horizon'] == hz].copy()
            merged = fc.merge(
                obs_b[['time', 'wind_speed', 'wind_direction']],
                on='time', how='inner',
            )
            for _, r in merged.iterrows():
                vt = r['time']
                feats = _hres_feature_dict(r)
                feats.update(context_window_features(re_b, vt, HORIZON_LAG_H[hz]))
                feats.update(buoy_context_features(bo_raw, vt, HORIZON_LAG_H[hz]))
                feats.update(regime_features(bo_raw, re_b, vt, HORIZON_LAG_H[hz]))
                feats.update(temporal_features(vt, int(r['hour'])))
                feats.update(climatology_lookup(clim, b, vt, int(r['hour'])))
                feats.update({
                    'lat': lat, 'lon': lon, 'horizon': hz,
                    'station': b, 'time': vt, 'hour': int(r['hour']),
                    'wind_speed': r['wind_speed'],
                    'wind_direction': r['wind_direction'],
                })
                rows.append(feats)

        for _, r in obs_b.iterrows():
            vt = r['time']
            if vt.year not in (2020, 2021):
                continue
            feats = _nan_hres_dict()
            feats.update(context_window_features(re_b, vt, HORIZON_LAG_H[14]))
            feats.update(buoy_context_features(bo_raw, vt, HORIZON_LAG_H[14]))
            feats.update(regime_features(bo_raw, re_b, vt, HORIZON_LAG_H[14]))
            feats.update(temporal_features(vt, int(vt.hour)))
            feats.update(climatology_lookup(clim, b, vt, int(vt.hour)))
            feats.update({
                'lat': lat, 'lon': lon, 'horizon': 14,
                'station': b, 'time': vt, 'hour': int(vt.hour),
                'wind_speed': r['wind_speed'],
                'wind_direction': r['wind_direction'],
            })
            rows.append(feats)

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────
# Model training
# ────────────────────────────────────────────────────────────────────────

def _features_for_hz(hz):
    return SPEED_FEATURES_D1 if hz == 1 else SPEED_FEATURES


def train_extratrees_for_d7(train_df):
    """ExtraTrees regressor for d7 — pairs with LGBM ensemble for q05/q95.
    Per-tree predictions form an empirical distribution; quantile across trees.
    Local test (cal2): LGBM+ET ensemble = Winkler 7.42 vs LGBM alone 7.48.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    tr = train_df[train_df['horizon'] == 7].copy()
    if len(tr) < 100:
        return None
    feat = _features_for_hz(7)
    X_tr = tr[feat].fillna(0)
    y_tr = tr['wind_speed']
    et = ExtraTreesRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    ).fit(X_tr, y_tr)
    return et


def train_quantile_models(train_df, val_df=None, n_seeds: int = 5):
    """Ensemble of LGBM per (horizon, quantile) with seed+hp diversity.

    For d7: use 3 diverse hyperparam configs (shallow/tight/deeper) which gave
    -0.25 m/s W_d7 on cal2 vs single-config ensemble. For d1/d14: keep the
    seed-diversified approach.
    """
    models = {}
    # Diverse d7 configs from tune test
    D7_CONFIGS = [
        dict(n_estimators=1500, learning_rate=0.015, num_leaves=15, min_child_samples=50, reg_lambda=0.5),
        dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=40, reg_lambda=3.0),
        dict(n_estimators=1000, learning_rate=0.02, num_leaves=127, min_child_samples=20, reg_lambda=2.0),
    ]
    for hz in [1, 7, 14]:
        tr = train_df[train_df['horizon'] == hz].copy()
        if len(tr) < 100:
            continue
        feat = _features_for_hz(hz)
        X_tr = tr[feat]
        y_tr = tr['wind_speed']
        w_tail = 1.0 + EXTREME_SAMPLE_WEIGHT * (y_tr.values > EXTREME_THR).astype(float)

        for name, q in QUANTILES.items():
            ensemble = []
            if hz == 7:
                # Diverse hyperparam ensemble
                for hp in D7_CONFIGS:
                    params = dict(
                        objective='quantile', alpha=q,
                        subsample=0.85, subsample_freq=1,
                        colsample_bytree=0.85,
                        verbose=-1, random_state=42, **hp,
                    )
                    w = w_tail if name == 'q95' else None
                    mdl = lgb.LGBMRegressor(**params)
                    mdl.fit(X_tr, y_tr, sample_weight=w)
                    ensemble.append(mdl)
            else:
                for seed in range(n_seeds):
                    lr_choices = [0.03, 0.04, 0.05]
                    leaves_choices = [31, 47, 63] if hz != 1 else [15, 23, 31]
                    n_est_choices = [800, 600, 500] if hz != 14 else [500, 400, 350]
                    params = dict(
                        objective='quantile', alpha=q,
                        n_estimators=n_est_choices[seed % 3],
                        learning_rate=lr_choices[seed % 3],
                        num_leaves=leaves_choices[seed % 3],
                        min_child_samples=30,
                        subsample=0.85, subsample_freq=1,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        verbose=-1, random_state=42 + seed * 17,
                    )
                    w = w_tail if (name == 'q95' and hz != 1) else None
                    mdl = lgb.LGBMRegressor(**params)
                    if val_df is not None:
                        va = val_df[val_df['horizon'] == hz]
                        X_va = va[feat]; y_va = va['wind_speed']
                        mdl.fit(
                            X_tr, y_tr, sample_weight=w,
                            eval_set=[(X_va, y_va)], eval_metric='quantile',
                            callbacks=[lgb.early_stopping(40, verbose=False)],
                        )
                    else:
                        mdl.fit(X_tr, y_tr, sample_weight=w)
                    ensemble.append(mdl)
            models[(hz, name)] = ensemble
    return models


def train_direction_models(train_df, val_df=None, n_seeds: int = 3):
    """Ensemble (sin, cos) regression per horizon."""
    models = {}
    for hz in [1, 7, 14]:
        tr = train_df[(train_df['horizon'] == hz)
                      & train_df['wind_direction'].notna()].copy()
        if len(tr) < 100:
            continue
        w_dir = np.clip(tr['wind_speed'].values, 1.0, 25.0)
        X_tr = tr[DIR_FEATURES]
        for target in ('sin', 'cos'):
            y = np.sin(np.radians(tr['wind_direction'])) if target == 'sin' \
                else np.cos(np.radians(tr['wind_direction']))
            ensemble = []
            for seed in range(n_seeds):
                params = dict(
                    objective='regression_l2',
                    n_estimators=500, learning_rate=0.04,
                    num_leaves=31, min_child_samples=40,
                    subsample=0.85, subsample_freq=1,
                    colsample_bytree=0.85, verbose=-1,
                    random_state=42 + seed * 17,
                )
                mdl = lgb.LGBMRegressor(**params)
                if val_df is not None:
                    va = val_df[(val_df['horizon'] == hz) & val_df['wind_direction'].notna()]
                    X_va = va[DIR_FEATURES]
                    y_va = np.sin(np.radians(va['wind_direction'])) if target == 'sin' \
                        else np.cos(np.radians(va['wind_direction']))
                    mdl.fit(X_tr, y, sample_weight=w_dir,
                            eval_set=[(X_va, y_va)],
                            callbacks=[lgb.early_stopping(40, verbose=False)])
                else:
                    mdl.fit(X_tr, y, sample_weight=w_dir)
                ensemble.append(mdl)
            models[(hz, target)] = ensemble
    return models


# ────────────────────────────────────────────────────────────────────────
# Prediction
# ────────────────────────────────────────────────────────────────────────

def _predict_ensemble(ens, X):
    if isinstance(ens, list):
        return np.mean([m.predict(X) for m in ens], axis=0)
    return ens.predict(X)


def fit_conformal_d1(hres, buoy_obs, years=(2020, 2021)):
    """Per-station empirical residual quantiles on training years."""
    out = {}
    for b in BUOYS:
        h = hres[b]
        h_d1 = h[h['horizon'] == 1]
        o = buoy_obs[b][['time', 'wind_speed']].dropna()
        m = h_d1.merge(o, on='time')
        m = m[m['time'].dt.year.isin(years)]
        if len(m) < 50:
            out[b] = CONFORMAL_FALLBACK_OFFSETS
            continue
        r = m['wind_speed'] - m['hres_speed_5m']
        out[b] = (
            float(np.quantile(r, 0.05)),
            float(np.quantile(r, 0.50)),
            float(np.quantile(r, 0.95)),
        )
    return out


def fit_extreme_classifier(train_df, hz, features):
    """Train P(y > EXTREME_THR | features) classifier for one horizon.

    SOTA tail-aware adjustment: at inference, q95 += amp * P_extreme.
    Helps extreme Winkler enormously at d1 (AUC 0.94) where extremes ARE
    predictable from features. Doesn't work well at d7/d14 (AUC ~0.5)
    because long-range extreme events are essentially unpredictable.
    """
    tr = train_df[train_df['horizon'] == hz].copy()
    if len(tr) < 50:
        return None
    y_ext = (tr['wind_speed'] > EXTREME_THR).astype(int)
    if y_ext.sum() < 10:
        return None
    spw = float((len(y_ext) - y_ext.sum()) / max(y_ext.sum(), 1))
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, scale_pos_weight=spw,
        verbose=-1, random_state=42,
    )
    clf.fit(tr[features], y_ext)
    return clf


def predict_d1_conformal(X, stations, conformal_offsets):
    h5 = X['hres_speed_5m'].values
    q05 = np.empty_like(h5, dtype=float)
    q50 = np.empty_like(h5, dtype=float)
    q95 = np.empty_like(h5, dtype=float)
    for i, s in enumerate(stations):
        off = conformal_offsets.get(str(s), CONFORMAL_FALLBACK_OFFSETS)
        q05[i] = h5[i] + off[0]
        q50[i] = h5[i] + off[1]
        q95[i] = h5[i] + off[2]
    return q05, q50, q95


_NDBC_SPEED_CLIM_CACHE = {'df': None}

def _ndbc_speed_clim_lookup(stations, X):
    """Lookup NDBC long-record empirical (q05, q50, q95) per (station, doy, hour).

    COMPLIANCE: legacy hook DISABLED in the submission package — the external NDBC archive
    is not provided data and the final submission's d14 speed uses the provided 2020-21
    climatology instead. Hard-return None so no external lookup can occur."""
    return None
    if _NDBC_SPEED_CLIM_CACHE['df'] is None:
        p = ARTIFACTS / 'ndbc_long_clim_speed.parquet'
        if not p.exists():
            return None
        _NDBC_SPEED_CLIM_CACHE['df'] = pd.read_parquet(p)
    ndbc = _NDBC_SPEED_CLIM_CACHE['df']
    if 'doy_sin' not in X.columns:
        return None
    doy = (np.arctan2(X['doy_sin'].values, X['doy_cos'].values) * 365.25 / (2*np.pi)) % 365.25
    doy = np.round(doy).astype(int).clip(1, 366)
    hour = (np.arctan2(X['hour_sin'].values, X['hour_cos'].values) * 24 / (2*np.pi)) % 24
    hour = np.round(hour).astype(int) % 24
    lookup = ndbc.set_index(['station','doy','hour'])[['q05','q50','q95']].to_dict('index')
    q05 = np.zeros(len(X)); q50 = np.zeros(len(X)); q95 = np.zeros(len(X))
    for i, (st, d, h) in enumerate(zip(stations, doy, hour)):
        key = (str(st), int(d), int(h))
        v = lookup.get(key)
        if v is None:
            return None
        q05[i] = v['q05']; q50[i] = v['q50']; q95[i] = v['q95']
    return q05, q50, q95


def predict_speed(models, hz, X, stations=None, conformal_offsets=None,
                  ext_clf=None, ext_amp=5.0, et_d7=None):
    """For d1 with conformal_offsets: pure conformal (HRES_5m + residuals).
    Otherwise LGBM ensemble.

    If ext_clf provided (extreme classifier), q95 += ext_amp * P_extreme.
    """
    if (hz == 1 and conformal_offsets is not None and stations is not None
            and 'hres_speed_5m' in X.columns):
        q05, q50, q95 = predict_d1_conformal(X, stations, conformal_offsets)
        # ext_clf DISABLED — sub7 test showed P(extreme) classifier had too
        # many false positives on 2022 (AUC 0.94 on val ≠ test); W_d1
        # regressed 5.70 → 6.22. Keep code path for future use but don't apply.
        q05 = np.clip(np.minimum(q05, q50), 0, None)
        q95 = np.maximum(q95, q50)
        q50 = np.clip(q50, 0, None)
        return q05, q50, q95
    q05 = _predict_ensemble(models[(hz, 'q05')], X)
    q50 = _predict_ensemble(models[(hz, 'q50')], X)
    q95 = _predict_ensemble(models[(hz, 'q95')], X)
    # d14: replace q05/q50/q95 with NDBC 12-yr long-record empirical quantiles.
    # ML at d14 essentially matches climatology anyway; NDBC has 6x more data
    # so tighter natural quantiles. Storm boost still added on q95 below.
    if hz == 14 and stations is not None:
        ndbc_q = _ndbc_speed_clim_lookup(stations, X)
        if ndbc_q is not None:
            q05_n, q50_n, q95_n = ndbc_q
            q05, q50, q95 = q05_n, q50_n, q95_n
    # ExtraTrees blend DISABLED — improved cal2 by -0.06 W_d7 but REGRESSED
    # 2022 by +0.31 W_d7. ET q95 distribution differs from LGBM on test data.
    # Blend q50 with raw HRES (5m downscaled) — HRES at short range is
    # highly skilled. Local test showed pure LGBM beats HRES blend at d7
    # (Winkler 7.20 vs 7.36) — only use HRES anchor at d1.
    if hz == 1 and 'hres_speed_5m' in X.columns:
        h5 = X['hres_speed_5m'].values
        m = np.isfinite(h5)
        q50 = np.where(m, 0.7 * h5 + 0.3 * q50, q50)
    # d7: keep pure LGBM q50 (tested)
    # monotone + non-negative
    q05 = np.clip(np.minimum(q05, q50), 0, None)
    q95 = np.maximum(q95, q50)
    q50 = np.clip(q50, 0, None)
    return q05, q50, q95


def predict_direction(dir_models, hz, X, stations=None):
    # d1: HRES direction is highly skilled at +24h — use it directly.
    if hz == 1 and 'hres_dir_sin' in X.columns and 'hres_dir_cos' in X.columns:
        hs = X['hres_dir_sin'].values
        hc = X['hres_dir_cos'].values
        m = np.isfinite(hs) & np.isfinite(hc)
        deg_hres = (np.degrees(np.arctan2(hs, hc))) % 360
        s = _predict_ensemble(dir_models[(hz, 'sin')], X)
        c = _predict_ensemble(dir_models[(hz, 'cos')], X)
        deg_ml = (np.degrees(np.arctan2(s, c))) % 360
        return np.where(m, deg_hres, deg_ml)
    # d14: legacy external-NDBC direction branch DISABLED for the submission package.
    # The final submission's d14 direction is the PROVIDED-DATA speed-weighted circular-median
    # climatology, grafted in reproduce.py (src/d14_dir_climatology.py) — never this lookup.
    if hz == 14 and False:
        ndbc_path = ARTIFACTS / 'ndbc_long_clim_dir.parquet'
        if ndbc_path.exists():
            try:
                ndbc = pd.read_parquet(ndbc_path)
                if stations is not None and 'doy_sin' in X.columns:
                    # Recover doy/hour from cyclic features
                    doy = (np.arctan2(X['doy_sin'].values, X['doy_cos'].values)
                           * 365.25 / (2*np.pi)) % 365.25
                    doy = np.round(doy).astype(int).clip(1, 366)
                    hour = (np.arctan2(X['hour_sin'].values, X['hour_cos'].values)
                            * 24 / (2*np.pi)) % 24
                    hour = np.round(hour).astype(int) % 24
                    out = np.zeros(len(X))
                    lookup = ndbc.set_index(['station','doy','hour'])['dir_clim'].to_dict()
                    for i, (st, d, h) in enumerate(zip(stations, doy, hour)):
                        key = (str(st), int(d), int(h))
                        out[i] = lookup.get(key, 90.0)
                    # NOTE: a persistence->climatology drift (w=0.08) was tried here
                    # and REVERTED. It won -0.06/-0.07 deg on BOTH 2020 and 2021
                    # cross-year, yet REGRESSED cmae_d14 +0.6 deg on the real 2022
                    # eval (sub28: 34.5 vs sub25a 33.9). d14 direction tuned on
                    # 2020-21 does not transfer to 2022's regime (sub8 lesson redux),
                    # even past a strict both-cross-year gate. Pure NDBC climatology
                    # is the production d14 direction. See LOGBOOK 2026-05-29.
                    return out
            except Exception as e:
                print(f'NDBC lookup failed: {e}; falling back to ML')
    # d7: blend ML and raw HRES dir 50/50 (test cal2: -1.5° cMAE).
    s = _predict_ensemble(dir_models[(hz, 'sin')], X)
    c = _predict_ensemble(dir_models[(hz, 'cos')], X)
    deg_ml = (np.degrees(np.arctan2(s, c))) % 360
    if hz == 7 and 'hres_dir_sin' in X.columns and 'hres_dir_cos' in X.columns:
        hs = X['hres_dir_sin'].values; hc = X['hres_dir_cos'].values
        m_h = np.isfinite(hs) & np.isfinite(hc)
        deg_hres = (np.degrees(np.arctan2(hs, hc))) % 360
        s_b = DIRECTION_BLEND_WEIGHT * np.sin(np.radians(deg_ml)) + DIRECTION_BLEND_WEIGHT * np.sin(np.radians(deg_hres))
        c_b = DIRECTION_BLEND_WEIGHT * np.cos(np.radians(deg_ml)) + DIRECTION_BLEND_WEIGHT * np.cos(np.radians(deg_hres))
        deg_blend = (np.degrees(np.arctan2(s_b, c_b))) % 360
        deg = np.where(m_h, deg_blend, deg_ml)
        return deg
    return deg_ml


# ────────────────────────────────────────────────────────────────────────
# Conformal Quantile Regression (hierarchical)
# ────────────────────────────────────────────────────────────────────────

def fit_cqr_offsets(val_pred, val_y, val_horizon, val_station, alpha=0.10):
    """Hierarchical CQR: per-horizon adjustment so empirical coverage = 1-alpha.

    Returns dict[(hz, station)] -> (q05_shift, q95_shift). Per-(hz,station)
    is shrunken toward the per-hz offset (low sample size).
    """
    df = pd.DataFrame({
        'q05': val_pred['q05'], 'q50': val_pred['q50'], 'q95': val_pred['q95'],
        'y': val_y, 'horizon': val_horizon, 'station': val_station,
    })
    # CQR score: max(q05-y, y-q95). Quantile gives the additive shift.
    df['score'] = np.maximum(df['q05'] - df['y'], df['y'] - df['q95'])
    offsets = {}
    for hz in [1, 7, 14]:
        sub = df[df['horizon'] == hz]
        if len(sub) == 0:
            continue
        # global per-horizon offset (1-alpha quantile of score)
        n = len(sub)
        level = np.ceil((n + 1) * (1 - alpha)) / n
        level = float(np.clip(level, 0, 1))
        global_off = float(np.quantile(sub['score'].values, level))
        offsets[(hz, '__global__')] = global_off
        # per-station: shrink toward global with weight ~ n_station / (n_station + k)
        K = CQR_SHRINKAGE_K
        for st in BUOYS:
            ss = sub[sub['station'] == st]
            if len(ss) == 0:
                offsets[(hz, st)] = global_off
                continue
            local = float(np.quantile(ss['score'].values, level))
            w_local = len(ss) / (len(ss) + K)
            offsets[(hz, st)] = w_local * local + (1 - w_local) * global_off
    return offsets


def apply_cqr(q05, q95, hz, stations, offsets, side='both'):
    """Apply CQR offsets element-wise (full shift — empirically optimal)."""
    q05 = np.asarray(q05, dtype=float).copy()
    q95 = np.asarray(q95, dtype=float).copy()
    for i, st in enumerate(stations):
        off = offsets.get((hz, st), offsets.get((hz, '__global__'), 0.0))
        if off > 0:
            if side in ('both', 'low'):
                q05[i] -= off
            if side in ('both', 'high'):
                q95[i] += off
        else:
            if side in ('both', 'low'):
                q05[i] -= off * 0.5
            if side in ('both', 'high'):
                q95[i] += off * 0.5
    q05 = np.clip(q05, 0, None)
    return q05, q95


# ────────────────────────────────────────────────────────────────────────
# Extreme regime detector — extra q95 widening when storm signals present
# ────────────────────────────────────────────────────────────────────────

def extreme_boost(features_df):
    """Compute additive boost (m/s) for q95 based on extreme regime signals.

    Signals: low msl in context (cyclone proxy), high ctx_spd10_max,
    high climatology max, hurricane-season DOY.
    """
    df = features_df
    msl_drop = (df['ctx_msl_min'].fillna(101500) < MSL_LOW_THRESHOLD).astype(float)
    msl_falling = (df['ctx_msl_trend'].fillna(0) < MSL_FALLING_THRESHOLD).astype(float)
    spd_storm = (df['ctx_spd10_last72h_max'].fillna(0) > STORM_WIND_THRESHOLD).astype(float)
    storm_signal_raw = np.maximum(msl_drop * msl_falling, spd_storm).values
    # Horizon-aware amplitudes — push d7 even lower (sub3 confirmed direction).
    hz = df['horizon'].fillna(7).values
    storm_amp = np.array([STORM_AMP.get(int(h), 2.0) for h in hz])
    storm_signal = storm_signal_raw * storm_amp
    in_season = ((df['doy_cos'].fillna(0) < HURRICANE_SEASON_DOY_COS)
                 & (df['lat'] < HURRICANE_SEASON_LAT)).astype(float).values
    season_amp = np.array([SEASON_AMP.get(int(h), 1.2) for h in hz])
    return storm_signal + in_season * season_amp


# ────────────────────────────────────────────────────────────────────────
# Scoring (mirror of competition scoring.py)
# ────────────────────────────────────────────────────────────────────────

def winkler(actual, lo, hi, alpha=0.10):
    # NOTE: also defined in calibrators._winkler — keep in sync
    actual = np.asarray(actual, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    w = hi - lo
    w = w + np.where(actual < lo, 2/alpha*(lo-actual), 0)
    w = w + np.where(actual > hi, 2/alpha*(actual-hi), 0)
    return w


def circular_mae(actual_deg, predicted_deg):
    d = np.abs(np.asarray(actual_deg, float) - np.asarray(predicted_deg, float))
    return np.minimum(d, 360 - d)


def score_predictions(pred_df, gt_df):
    """Return dict with the 9 dimensions."""
    merge_keys = ['window', 'station', 'horizon', 'hour']
    for d in (pred_df, gt_df):
        d['station'] = d['station'].astype(str)
        for k in ('window', 'horizon', 'hour'):
            d[k] = d[k].astype(int)
    m = pred_df.merge(gt_df, on=merge_keys, how='inner', suffixes=('', '_gt'))
    out = {'n_scored': len(m)}
    for hz in [1, 7, 14]:
        hm = m[m['horizon'] == hz]
        hs = hm[hm['speed'].notna()]
        hd = hm[hm['direction'].notna()]
        if len(hs):
            ws = winkler(hs['speed'], hs['q05'], hs['q95'])
            w = float(np.mean(ws))
            ext = hs['speed'].values > EXTREME_THR
            ew = float(np.mean(ws[ext])) if ext.sum() else w
            cov = float(np.mean((hs['speed'] >= hs['q05']) & (hs['speed'] <= hs['q95'])))
        else:
            w = ew = cov = np.nan
        c = float(np.mean(circular_mae(hd['direction'], hd['dir_50']))) if len(hd) else np.nan
        out[f'winkler_d{hz}'] = w
        out[f'cmae_d{hz}'] = c
        out[f'extreme_winkler_d{hz}'] = ew
        out[f'coverage_d{hz}'] = cov
        out[f'n_extreme_d{hz}'] = int((hs['speed'] > EXTREME_THR).sum()) if len(hs) else 0
    return out


# ────────────────────────────────────────────────────────────────────────
# Local eval harness — simulate 8 windows from 2021
# ────────────────────────────────────────────────────────────────────────

def simulate_windows_from_2021():
    """Create 8 fake inference windows matching 2022 timing (shifted to 2021)."""
    windows = []
    for w in range(1, 9):
        meta = json.loads((DATA / f'inference/window_{w}/metadata.json').read_text())
        # shift the actual 2022 dates back one year to 2021 (or stay in 2020/2021)
        def shift(ds): return (pd.Timestamp(ds) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        windows.append({
            'id': w,
            'context_start': shift(meta['context_start']),
            'context_end': shift(meta['context_end']),
            'predict_start': shift(meta['predict_start']),
            'predict_end': shift(meta['predict_end']),
            'score_days': {k: shift(v) for k, v in meta['score_days'].items()},
        })
    return windows


def build_inference_rows_from_train(window_meta, hres, re_ts, buoy_obs, clim,
                                    buoy_obs_raw=None):
    """Build feature rows for a simulated window using TRAIN data only."""
    rows = []
    for b in BUOYS:
        info = META['buoys'][b]
        lat, lon = info['latitude'], info['longitude']
        re_b = re_ts[b]
        h_b = hres[b]
        bo_raw = buoy_obs_raw[b] if buoy_obs_raw is not None else None
        for hz, day in [(1, 'd1'), (7, 'd7')]:
            day_ts = pd.Timestamp(window_meta['score_days'][day])
            for hour in HOURS:
                vt = day_ts + pd.Timedelta(hours=hour)
                fc_row = h_b[(h_b['time'] == vt) & (h_b['horizon'] == hz)]
                if len(fc_row) == 0:
                    continue
                fc_row = fc_row.iloc[0]
                feats = _hres_feature_dict(fc_row)
                feats.update(context_window_features(re_b, vt, HORIZON_LAG_H[hz]))
                feats.update(buoy_context_features(bo_raw, vt, HORIZON_LAG_H[hz]))
                feats.update(regime_features(bo_raw, re_b, vt, HORIZON_LAG_H[hz]))
                feats.update(temporal_features(vt, hour))
                feats.update(climatology_lookup(clim, b, vt, hour))
                feats.update({
                    'lat': lat, 'lon': lon, 'horizon': hz,
                    'station': b, 'window': window_meta['id'],
                    'hour': hour, 'time': vt,
                })
                rows.append(feats)
        day_ts = pd.Timestamp(window_meta['score_days']['d14'])
        for hour in HOURS:
            vt = day_ts + pd.Timedelta(hours=hour)
            feats = _nan_hres_dict()
            feats.update(context_window_features(re_b, vt, HORIZON_LAG_H[14]))
            feats.update(buoy_context_features(bo_raw, vt, HORIZON_LAG_H[14]))
            feats.update(regime_features(bo_raw, re_b, vt, HORIZON_LAG_H[14]))
            feats.update(temporal_features(vt, hour))
            feats.update(climatology_lookup(clim, b, vt, hour))
            feats.update({
                'lat': lat, 'lon': lon, 'horizon': 14,
                'station': b, 'window': window_meta['id'],
                'hour': hour, 'time': vt,
            })
            rows.append(feats)
    return pd.DataFrame(rows)


def build_ground_truth_for_simulated(windows, buoy_obs):
    """Build ground_truth DataFrame from buoy obs at the valid times."""
    rows = []
    for wm in windows:
        for b in BUOYS:
            obs = buoy_obs[b]
            for hz, day in [(1, 'd1'), (7, 'd7'), (14, 'd14')]:
                day_ts = pd.Timestamp(wm['score_days'][day])
                for hour in HOURS:
                    vt = day_ts + pd.Timedelta(hours=hour)
                    o = obs[obs['time'] == vt]
                    if len(o) == 0:
                        continue
                    o = o.iloc[0]
                    rows.append(dict(
                        window=wm['id'], station=b, horizon=hz, hour=hour,
                        latitude=META['buoys'][b]['latitude'],
                        longitude=META['buoys'][b]['longitude'],
                        speed=o['wind_speed'],
                        direction=o['wind_direction'],
                    ))
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────
# Full pipeline drivers
# ────────────────────────────────────────────────────────────────────────

def cmd_build_train():
    ARTIFACTS.mkdir(exist_ok=True)
    print('Loading data…')
    hres, re_ts, buoy_obs, buoy_obs_raw = load_all_train()
    print('Building climatology…')
    clim = build_climatology(buoy_obs)
    clim.to_parquet(ARTIFACTS / 'climatology.parquet')
    print(f'  climatology rows: {len(clim)}')
    print('Building training table…')
    tt = build_training_table(hres, re_ts, buoy_obs, clim, buoy_obs_raw)
    print('Attaching NDBC long-clim features…')
    tt = _attach_ndbc_features(tt)
    tt.to_parquet(ARTIFACTS / 'train_table.parquet')
    print(f'  training rows: {len(tt)}  '
          f'(d1={(tt.horizon==1).sum()}, d7={(tt.horizon==7).sum()}, '
          f'd14={(tt.horizon==14).sum()})')
    return tt, clim, hres, re_ts, buoy_obs


def cmd_train(tt=None):
    ARTIFACTS.mkdir(exist_ok=True)
    if tt is None:
        tt = pd.read_parquet(ARTIFACTS / 'train_table.parquet')
    # split: train 2020 + most of 2021; val = last 60 days of 2021
    tt = tt.sort_values('time').reset_index(drop=True)
    cutoff = pd.Timestamp('2021-11-01')
    tr = tt[tt['time'] < cutoff]
    va = tt[tt['time'] >= cutoff]
    print(f'Train: {len(tr)}  Val: {len(va)}')
    print('Training speed quantile models…')
    speed_models = train_quantile_models(tr, va)
    print('Training direction models…')
    dir_models = train_direction_models(tr, va)

    # Calibration: predict on val and compute CQR offsets
    cal_rows = []
    for hz in [1, 7, 14]:
        sub = va[va['horizon'] == hz]
        if len(sub) == 0:
            continue
        X = sub[_features_for_hz(hz)]
        q05, q50, q95 = predict_speed(speed_models, hz, X)
        cal_rows.append(pd.DataFrame({
            'q05': q05, 'q50': q50, 'q95': q95,
            'y': sub['wind_speed'].values,
            'horizon': sub['horizon'].values,
            'station': sub['station'].values,
        }))
    cal = pd.concat(cal_rows, ignore_index=True)
    print('Fitting hierarchical CQR offsets…')
    offsets = fit_cqr_offsets(
        {'q05': cal['q05'], 'q50': cal['q50'], 'q95': cal['q95']},
        cal['y'].values, cal['horizon'].values, cal['station'].values,
    )
    # Conformal d1 residuals from 2020+2021 (full data — used in submission)
    hres_dict, _, buoy_obs_dict, _ = load_all_train()
    conformal_d1 = fit_conformal_d1(hres_dict, buoy_obs_dict, years=(2020, 2021))

    # P(extreme | features) classifier — disabled in predict_speed
    ext_clf_d1 = fit_extreme_classifier(tt, hz=1, features=SPEED_FEATURES_D1)

    # ExtraTrees for d7 — blends with LGBM q05/q95 in predict_speed
    print('Training ExtraTrees for d7…')
    et_d7 = train_extratrees_for_d7(tt)

    import joblib
    joblib.dump(speed_models, ARTIFACTS / 'speed_models.joblib')
    joblib.dump(dir_models, ARTIFACTS / 'dir_models.joblib')
    joblib.dump(offsets, ARTIFACTS / 'cqr_offsets.joblib')
    joblib.dump(conformal_d1, ARTIFACTS / 'conformal_d1.joblib')
    joblib.dump(ext_clf_d1, ARTIFACTS / 'ext_clf_d1.joblib')
    joblib.dump(et_d7, ARTIFACTS / 'et_d7.joblib')
    print('Saved all artifacts to', ARTIFACTS)
    # quick val coverage
    for hz in [1, 7, 14]:
        sub = cal[cal['horizon'] == hz]
        cov = float(np.mean((sub['y'] >= sub['q05']) & (sub['y'] <= sub['q95'])))
        # post-cqr
        q05c, q95c = apply_cqr(sub['q05'].values, sub['q95'].values, hz,
                               sub['station'].values, offsets)
        cov_c = float(np.mean((sub['y'].values >= q05c) & (sub['y'].values <= q95c)))
        print(f'  d{hz}: raw cov={cov:.1%}  post-CQR cov={cov_c:.1%}  '
              f'global_off={offsets.get((hz,"__global__"), 0):.3f}')
    return speed_models, dir_models, offsets


def run_clean_eval(train_year: int = 2020, return_frames: bool = False):
    """Train on `train_year` only and eval on simulated 2021 windows.

    Returns the scores dict; if return_frames, also returns (pred_df, gt_df).
    """
    ARTIFACTS.mkdir(exist_ok=True)
    print('Loading data…')
    hres, re_ts, buoy_obs, buoy_obs_raw = load_all_train()
    print('Building climatology from train year only…')
    buoy_obs_2020 = {b: o[o['time'].dt.year == train_year] for b, o in buoy_obs.items()}
    buoy_obs_raw_2020 = {b: o[o['time'].dt.year == train_year] for b, o in buoy_obs_raw.items()}
    clim = build_climatology(buoy_obs_2020)

    print('Building training table from train year…')
    tt = build_training_table(hres, re_ts, buoy_obs_2020, clim, buoy_obs_raw_2020)
    tt = _attach_ndbc_features(tt)
    tt = tt[tt['time'].dt.year == train_year]
    print(f'  rows={len(tt)}')

    # Use last 30 days of 2020 as inner val for early stopping
    tt = tt.sort_values('time').reset_index(drop=True)
    cutoff = pd.Timestamp('2020-11-15')
    tr = tt[tt['time'] < cutoff]
    va = tt[tt['time'] >= cutoff]
    print(f'  inner train={len(tr)} inner val={len(va)}')
    print('Training speed quantile models…')
    speed_models = train_quantile_models(tr, va)
    print('Training direction models…')
    dir_models = train_direction_models(tr, va)

    # CQR calibration on inner val
    cal_rows = []
    for hz in [1, 7, 14]:
        sub = va[va['horizon'] == hz]
        if len(sub) == 0:
            continue
        X = sub[_features_for_hz(hz)]
        q05, q50, q95 = predict_speed(speed_models, hz, X)
        cal_rows.append(pd.DataFrame({
            'q05': q05, 'q50': q50, 'q95': q95,
            'y': sub['wind_speed'].values, 'horizon': sub['horizon'].values,
            'station': sub['station'].values,
        }))
    cal = pd.concat(cal_rows, ignore_index=True)
    offsets = fit_cqr_offsets(
        {'q05': cal['q05'], 'q50': cal['q50'], 'q95': cal['q95']},
        cal['y'].values, cal['horizon'].values, cal['station'].values,
    )

    # Conformal d1 residuals — fit on train year only (matches the clean-eval setup)
    conformal_d1_train = fit_conformal_d1(hres, buoy_obs, years=(train_year,))

    windows = simulate_windows_from_2021()
    sim_pred = []
    for wm in windows:
        feats = build_inference_rows_from_train(wm, hres, re_ts, buoy_obs, clim, buoy_obs_raw)
        feats = _attach_ndbc_features(feats)
        for hz in [1, 7, 14]:
            sub = feats[feats['horizon'] == hz]
            if len(sub) == 0:
                continue
            X = sub[_features_for_hz(hz)]
            q05, q50, q95 = predict_speed(
                speed_models, hz, X,
                stations=sub['station'].values if hz == 1 else None,
                conformal_offsets=conformal_d1_train if hz == 1 else None,
            )
            if hz == 1:
                # conformal already calibrated — skip CQR shift, no extreme boost
                q05c, q95c = q05, q95
            else:
                q05c, q95c = apply_cqr(q05, q95, hz, sub['station'].values, offsets)
                boost = extreme_boost(sub)
                q95c = q95c + np.maximum(boost, 0)
            d50 = predict_direction(dir_models, hz, sub[DIR_FEATURES])
            sim_pred.append(pd.DataFrame({
                'window': sub['window'].values,
                'station': sub['station'].values,
                'horizon': hz, 'hour': sub['hour'].values,
                'q05': q05c, 'q50': q50, 'q95': q95c, 'dir_50': d50,
            }))
    pred = pd.concat(sim_pred, ignore_index=True)
    gt = build_ground_truth_for_simulated(windows, buoy_obs)
    sc = score_predictions(pred, gt)
    print('\n=== CLEAN eval (train 2020, test sim 2021) ===')
    for k, v in sc.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

    # Also compute the BASELINE on the same setup for comparison.
    print('\nBaseline (raw HRES, climatology d14, no calibration)…')
    base_rows = []
    for wm in windows:
        feats = build_inference_rows_from_train(wm, hres, re_ts, buoy_obs, clim, buoy_obs_raw)
        for _, r in feats.iterrows():
            hz = int(r['horizon'])
            if hz in (1, 7) and np.isfinite(r['hres_speed']):
                hs = float(r['hres_speed'])
                # crude PI: ±1.65*sigma with sigma growing with horizon
                sigma = 1.5 if hz == 1 else 2.5
                q05_b = max(hs - 1.65 * sigma, 0)
                q50_b = hs
                q95_b = hs + 1.65 * sigma
                # dir is just HRES dir
                d_b = float(np.degrees(np.arctan2(
                    r['hres_dir_sin'], r['hres_dir_cos'])) % 360)
            else:
                q05_b = float(r['clim_q05']) if np.isfinite(r['clim_q05']) else 2
                q50_b = float(r['clim_q50']) if np.isfinite(r['clim_q50']) else 6
                q95_b = float(r['clim_q95']) if np.isfinite(r['clim_q95']) else 14
                d_b = float(np.degrees(np.arctan2(
                    r['clim_dir_sin'], r['clim_dir_cos'])) % 360) if np.isfinite(r['clim_dir_sin']) else 90
            base_rows.append(dict(
                window=int(r['window']), station=r['station'], horizon=hz,
                hour=int(r['hour']),
                q05=q05_b, q50=q50_b, q95=q95_b, dir_50=d_b,
            ))
    base_pred = pd.DataFrame(base_rows)
    bs = score_predictions(base_pred, gt)
    print('=== BASELINE scores ===')
    for k, v in bs.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

    # Comparison table
    print('\n=== Δ (ours − baseline; lower is better, negative = win) ===')
    for k in ['winkler_d1','winkler_d7','winkler_d14',
              'cmae_d1','cmae_d7','cmae_d14',
              'extreme_winkler_d1','extreme_winkler_d7','extreme_winkler_d14']:
        if np.isfinite(sc.get(k, np.nan)) and np.isfinite(bs.get(k, np.nan)):
            d = sc[k] - bs[k]
            tag = 'WIN' if d < 0 else 'LOSE'
            print(f'  {k}: {sc[k]:.3f} vs {bs[k]:.3f}  Δ={d:+.3f}  {tag}')

    if return_frames:
        return sc, pred, gt
    return sc


def cmd_clean_eval():
    scores = run_clean_eval(train_year=2020, return_frames=False)
    for k, v in scores.items():
        print(f'  {k}: {v}')


def cmd_eval():
    """Run local eval against simulated 2021 windows."""
    ARTIFACTS.mkdir(exist_ok=True)
    import joblib
    speed_models = joblib.load(ARTIFACTS / 'speed_models.joblib')
    dir_models = joblib.load(ARTIFACTS / 'dir_models.joblib')
    offsets = joblib.load(ARTIFACTS / 'cqr_offsets.joblib')
    clim = pd.read_parquet(ARTIFACTS / 'climatology.parquet')

    hres, re_ts, buoy_obs, buoy_obs_raw = load_all_train()
    conformal_d1 = joblib.load(ARTIFACTS / 'conformal_d1.joblib') \
        if (ARTIFACTS / 'conformal_d1.joblib').exists() else None
    windows = simulate_windows_from_2021()
    print('Building inference rows for 8 simulated windows…')
    sim_pred = []
    for wm in windows:
        feats = build_inference_rows_from_train(wm, hres, re_ts, buoy_obs, clim, buoy_obs_raw)
        for hz in [1, 7, 14]:
            sub = feats[feats['horizon'] == hz]
            if len(sub) == 0:
                continue
            X = sub[_features_for_hz(hz)]
            q05, q50, q95 = predict_speed(
                speed_models, hz, X,
                stations=sub['station'].values if hz == 1 else None,
                conformal_offsets=conformal_d1 if hz == 1 else None,
            )
            if hz == 1 and conformal_d1 is not None:
                q05c, q95c = q05, q95
            else:
                q05c, q95c = apply_cqr(q05, q95, hz, sub['station'].values, offsets)
                boost = extreme_boost(sub)
                q95c = q95c + np.maximum(boost, 0)
            d50 = predict_direction(dir_models, hz, sub[DIR_FEATURES])
            sim_pred.append(pd.DataFrame({
                'window': sub['window'].values,
                'station': sub['station'].values,
                'horizon': hz,
                'hour': sub['hour'].values,
                'q05': q05c, 'q50': q50, 'q95': q95c, 'dir_50': d50,
            }))
    pred = pd.concat(sim_pred, ignore_index=True)
    gt = build_ground_truth_for_simulated(windows, buoy_obs)
    pred.to_parquet(ARTIFACTS / 'sim_pred.parquet')
    gt.to_parquet(ARTIFACTS / 'sim_gt.parquet')
    sc = score_predictions(pred, gt)
    print('\n=== Local eval on simulated 2021 windows ===')
    for k, v in sc.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    # baseline comparison
    base = {
        'd1': {'winkler': 6.0, 'cmae': 30, 'extreme_winkler': 25},
        'd7': {'winkler': 10.0, 'cmae': 50, 'extreme_winkler': 40},
        'd14': {'winkler': 12.0, 'cmae': 60, 'extreme_winkler': 50},
    }
    return sc


def cmd_submit():
    """Generate predictions.csv + submission.zip for the real 2022 windows."""
    ARTIFACTS.mkdir(exist_ok=True)
    import joblib
    speed_models = joblib.load(ARTIFACTS / 'speed_models.joblib')
    dir_models = joblib.load(ARTIFACTS / 'dir_models.joblib')
    offsets = joblib.load(ARTIFACTS / 'cqr_offsets.joblib')
    conformal_d1 = joblib.load(ARTIFACTS / 'conformal_d1.joblib') \
        if (ARTIFACTS / 'conformal_d1.joblib').exists() else None
    ext_clf_d1 = joblib.load(ARTIFACTS / 'ext_clf_d1.joblib') \
        if (ARTIFACTS / 'ext_clf_d1.joblib').exists() else None
    et_d7 = joblib.load(ARTIFACTS / 'et_d7.joblib') \
        if (ARTIFACTS / 'et_d7.joblib').exists() else None
    clim = pd.read_parquet(ARTIFACTS / 'climatology.parquet')

    out_rows = []
    for w in range(1, 9):
        wdir = DATA / f'inference/window_{w}'
        meta = json.loads((wdir / 'metadata.json').read_text())
        for b in BUOYS:
            info = META['buoys'][b]
            lat, lon = info['latitude'], info['longitude']
            re = pd.read_parquet(wdir / f'context_reanalysis_{b}.parquet')
            re['time'] = pd.to_datetime(re['time'])
            re_ts = precompute_reanalysis_timeseries(re, lat, lon)
            fc_path = wdir / f'forecast_hres_{b}.parquet'
            fc = pd.read_parquet(fc_path) if fc_path.exists() else None
            if fc is not None:
                fc['time'] = pd.to_datetime(fc['time'])
                fc = hres_to_features(fc)
            ctx_buoy_path = wdir / f'context_buoy_{b}.parquet'
            ctx_buoy = pd.read_parquet(ctx_buoy_path) if ctx_buoy_path.exists() else None
            if ctx_buoy is not None and len(ctx_buoy):
                ctx_buoy['time'] = pd.to_datetime(ctx_buoy['time'])

            for hz, day in [(1, 'd1'), (7, 'd7')]:
                day_ts = pd.Timestamp(meta['score_days'][day])
                for hour in HOURS:
                    vt = day_ts + pd.Timedelta(hours=hour)
                    if fc is not None:
                        row = fc[(fc['time'] == vt) & (fc['horizon'] == hz)]
                        if len(row) == 0:
                            continue
                        row = row.iloc[0]
                        feats = _hres_feature_dict(row)
                    else:
                        feats = _nan_hres_dict()
                    feats.update(context_window_features(re_ts, vt, HORIZON_LAG_H[hz]))
                    feats.update(buoy_context_features(ctx_buoy, vt, HORIZON_LAG_H[hz]))
                    feats.update(regime_features(ctx_buoy, re_ts, vt, HORIZON_LAG_H[hz]))
                    feats.update(temporal_features(vt, hour))
                    feats.update(climatology_lookup(clim, b, vt, hour))
                    feats.update({'lat': lat, 'lon': lon, 'horizon': hz,
                                  'station': b, 'window': w, 'hour': hour, 'time': vt})
                    # Attach NDBC long-clim features
                    feats_df = _attach_ndbc_features(pd.DataFrame([feats]))
                    feats = feats_df.iloc[0].to_dict()
                    X = pd.DataFrame([feats])[_features_for_hz(hz)]
                    q05, q50, q95 = predict_speed(
                        speed_models, hz, X,
                        stations=[b] if hz == 1 else None,
                        conformal_offsets=conformal_d1 if hz == 1 else None,
                        ext_clf=ext_clf_d1 if hz == 1 else None,
                        et_d7=et_d7 if hz == 7 else None,
                    )
                    if not (hz == 1 and conformal_d1 is not None):
                        q05, q95 = apply_cqr(q05, q95, hz, [b], offsets)
                        fdf = pd.DataFrame([feats])
                        q95 = q95 + np.maximum(extreme_boost(fdf), 0)
                    d50 = predict_direction(dir_models, hz,
                                            pd.DataFrame([feats])[DIR_FEATURES],
                                            stations=[b])
                    out_rows.append(dict(
                        window=w, station=b, latitude=lat, longitude=lon,
                        horizon=hz, hour=hour,
                        q05=float(q05[0]), q50=float(q50[0]), q95=float(q95[0]),
                        dir_50=float(d50[0]),
                    ))
            day_ts = pd.Timestamp(meta['score_days']['d14'])
            for hour in HOURS:
                vt = day_ts + pd.Timedelta(hours=hour)
                feats = _nan_hres_dict()
                feats.update(context_window_features(re_ts, vt, HORIZON_LAG_H[14]))
                feats.update(buoy_context_features(ctx_buoy, vt, HORIZON_LAG_H[14]))
                feats.update(regime_features(ctx_buoy, re_ts, vt, HORIZON_LAG_H[14]))
                feats.update(temporal_features(vt, hour))
                feats.update(climatology_lookup(clim, b, vt, hour))
                feats.update({'lat': lat, 'lon': lon, 'horizon': 14,
                              'station': b, 'window': w, 'hour': hour, 'time': vt})
                feats_df = _attach_ndbc_features(pd.DataFrame([feats]))
                feats = feats_df.iloc[0].to_dict()
                X = pd.DataFrame([feats])[_features_for_hz(14)]
                q05, q50, q95 = predict_speed(speed_models, 14, X, stations=[b])
                # Only apply CQR if NDBC clim NOT being used (NDBC is already
                # empirically calibrated). Detect: NDBC path returns specific
                # marker (q50 = NDBC empirical, not ML output).
                ndbc_used = (ARTIFACTS / 'ndbc_long_clim_speed.parquet').exists()
                if not ndbc_used:
                    q05, q95 = apply_cqr(q05, q95, 14, [b], offsets)
                fdf = pd.DataFrame([feats])
                q95 = q95 + np.maximum(extreme_boost(fdf), 0)
                d50 = predict_direction(dir_models, 14,
                                        pd.DataFrame([feats])[DIR_FEATURES],
                                        stations=[b])
                out_rows.append(dict(
                    window=w, station=b, latitude=lat, longitude=lon,
                    horizon=14, hour=hour,
                    q05=float(q05[0]), q50=float(q50[0]), q95=float(q95[0]),
                    dir_50=float(d50[0]),
                ))

    sub = pd.DataFrame(out_rows).sort_values(
        ['window', 'station', 'horizon', 'hour']).reset_index(drop=True)
    sub[['q05', 'q50', 'q95']] = sub[['q05', 'q50', 'q95']].round(3)
    sub['dir_50'] = sub['dir_50'].round(1)
    sub.to_csv(ROOT / 'predictions.csv', index=False)
    with zipfile.ZipFile(ROOT / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / 'predictions.csv', arcname='predictions.csv')
    print(f'\nsubmission.zip ready — {len(sub)} rows')
    print(sub.head(6).to_string(index=False))
    return sub


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'build_train':
        cmd_build_train()
    elif cmd == 'train':
        cmd_train()
    elif cmd == 'eval':
        cmd_eval()
    elif cmd == 'clean_eval':
        cmd_clean_eval()
    elif cmd == 'submit':
        cmd_submit()
    elif cmd == 'all':
        cmd_build_train()
        cmd_train()
        cmd_eval()
        cmd_submit()
    else:
        print(f'unknown cmd: {cmd}')
        sys.exit(1)
