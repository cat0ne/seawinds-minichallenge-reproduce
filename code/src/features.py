"""Feature engineering for buoy wind post-processing.

For each (station, valid_time, horizon) target row, we build features from:
- HRES forecast at valid_time (for d1/d7 only)
- Reanalysis patch context observable BEFORE valid_time
- Buoy obs context (may be missing in some inference windows)
- Temporal/seasonal/station features
- Climatology lookup

We carefully mirror the inference setup: at inference time, we only have
reanalysis up to context_end (which is valid_time - lead_h hours BEFORE
the d1 forecast). For d7 the freshest reanalysis is ~7 days old vs valid_time;
for d14 it's ~14 days old. Training must simulate this gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# Approximate gap from "context_end" to each horizon's valid time (in hours).
# context_end is the day before predict_start; we keep things conservative.
HORIZON_LAG_H = {1: 24, 7: 7 * 24, 14: 14 * 24}

ANEMO_HEIGHT_M = 5.0  # Buoy anemometer height (per metadata)
WIND_HEIGHT_M = 10.0  # ECMWF u10/v10 reference height
Z0 = 1.5e-4           # roughness length over open sea (Charnock approx)
# log-law scaling factor from 10 m to 5 m: ln(5/z0)/ln(10/z0)
H_SCALE = np.log(ANEMO_HEIGHT_M / Z0) / np.log(WIND_HEIGHT_M / Z0)


# ────────────────────────────────────────────────────────────────────────
# Reanalysis aggregation
# ────────────────────────────────────────────────────────────────────────

def _patch_spatial_stats(patch: pd.DataFrame) -> dict[str, float]:
    """Aggregate one 6-hourly snapshot of a 13x13 reanalysis patch."""
    spd10 = np.sqrt(patch['u10'].values ** 2 + patch['v10'].values ** 2)
    spd100 = np.sqrt(patch['u100'].values ** 2 + patch['v100'].values ** 2)
    return {
        'sp_spd10_mean': float(spd10.mean()),
        'sp_spd10_max': float(spd10.max()),
        'sp_spd10_std': float(spd10.std()),
        'sp_spd100_mean': float(spd100.mean()),
        'sp_spd100_max': float(spd100.max()),
        'sp_u10_mean': float(patch['u10'].mean()),
        'sp_v10_mean': float(patch['v10'].mean()),
        'sp_u100_mean': float(patch['u100'].mean()),
        'sp_v100_mean': float(patch['v100'].mean()),
        'sp_msl_min': float(patch['msl'].min()),
        'sp_msl_mean': float(patch['msl'].mean()),
        'sp_t2m_mean': float(patch['t2m'].mean()),
    }


def precompute_reanalysis_timeseries(
    re: pd.DataFrame,
    buoy_lat: float | None = None,
    buoy_lon: float | None = None,
) -> pd.DataFrame:
    """Collapse a station's reanalysis (time × grid) to (time × stats).

    Adds center-pixel features (nearest grid cell to the buoy) which are
    the single best proxy for the actual wind at the buoy location.
    """
    re = re.copy()
    re['spd10'] = np.sqrt(re['u10'] ** 2 + re['v10'] ** 2)
    re['spd100'] = np.sqrt(re['u100'] ** 2 + re['v100'] ** 2)

    # Find the center pixel — nearest grid cell to buoy
    if buoy_lat is None or buoy_lon is None:
        buoy_lat = re['latitude'].mean()
        buoy_lon = re['longitude'].mean()
    lats = np.sort(re['latitude'].unique())
    lons = np.sort(re['longitude'].unique())
    lat_c = lats[np.argmin(np.abs(lats - buoy_lat))]
    lon_c = lons[np.argmin(np.abs(lons - buoy_lon))]
    center_mask = (re['latitude'] == lat_c) & (re['longitude'] == lon_c)
    re_center = (re[center_mask]
                 .set_index('time')
                 [['u10', 'v10', 'u100', 'v100', 'spd10', 'spd100',
                   'msl', 't2m']]
                 .add_prefix('ctr_'))

    g = re.groupby('time')
    out = pd.DataFrame({
        'sp_spd10_mean': g['spd10'].mean(),
        'sp_spd10_max': g['spd10'].max(),
        'sp_spd10_std': g['spd10'].std(),
        'sp_spd100_mean': g['spd100'].mean(),
        'sp_spd100_max': g['spd100'].max(),
        'sp_u10_mean': g['u10'].mean(),
        'sp_v10_mean': g['v10'].mean(),
        'sp_u100_mean': g['u100'].mean(),
        'sp_v100_mean': g['v100'].mean(),
        'sp_msl_min': g['msl'].min(),
        'sp_msl_mean': g['msl'].mean(),
        'sp_t2m_mean': g['t2m'].mean(),
    })
    out = out.join(re_center).reset_index().sort_values('time').reset_index(drop=True)
    return out


def context_window_features(
    re_ts: pd.DataFrame,
    valid_time: pd.Timestamp,
    horizon_h: int,
    win_days: int = 14,
) -> dict[str, float]:
    """Build features from reanalysis available BEFORE valid_time.

    context_end = valid_time - horizon_h hours
    context_start = context_end - win_days * 24 hours
    """
    ctx_end = valid_time - pd.Timedelta(hours=horizon_h)
    ctx_start = ctx_end - pd.Timedelta(days=win_days)
    mask = (re_ts['time'] > ctx_start) & (re_ts['time'] <= ctx_end)
    sub = re_ts.loc[mask]
    keys = [
        'spd10_mean', 'spd10_max', 'spd10_std', 'spd100_mean', 'spd100_max',
        'u10_mean', 'v10_mean', 'msl_min', 'msl_mean', 'msl_trend',
        'spd10_last', 'spd10_last24h_mean', 'spd10_last72h_max',
        'ctr_u10_last', 'ctr_v10_last', 'ctr_spd10_last',
        'ctr_u100_last', 'ctr_v100_last', 'ctr_spd100_last',
        'ctr_msl_last', 'ctr_msl_24h_change',
        'ctr_spd10_mean', 'ctr_spd10_max', 'ctr_spd10_std',
    ]
    if len(sub) == 0:
        return {f'ctx_{k}': np.nan for k in keys}
    last = sub.iloc[-1]
    last24 = sub.tail(4)  # last 24h
    last72 = sub.tail(12)  # last 72h
    feats = {
        'ctx_spd10_mean': float(sub['sp_spd10_mean'].mean()),
        'ctx_spd10_max': float(sub['sp_spd10_max'].max()),
        'ctx_spd10_std': float(sub['sp_spd10_mean'].std()),
        'ctx_spd100_mean': float(sub['sp_spd100_mean'].mean()),
        'ctx_spd100_max': float(sub['sp_spd100_max'].max()),
        'ctx_u10_mean': float(sub['sp_u10_mean'].mean()),
        'ctx_v10_mean': float(sub['sp_v10_mean'].mean()),
        'ctx_msl_min': float(sub['sp_msl_min'].min()),
        'ctx_msl_mean': float(sub['sp_msl_mean'].mean()),
        'ctx_msl_trend': float(last['sp_msl_mean'] - sub['sp_msl_mean'].iloc[0]),
        'ctx_spd10_last': float(last['sp_spd10_mean']),
        'ctx_spd10_last24h_mean': float(last24['sp_spd10_mean'].mean()),
        'ctx_spd10_last72h_max': float(last72['sp_spd10_max'].max()),
        # Center-pixel features (best per-time-step proxy for the buoy)
        'ctx_ctr_u10_last': float(last.get('ctr_u10', np.nan))
            if 'ctr_u10' in sub.columns else np.nan,
        'ctx_ctr_v10_last': float(last.get('ctr_v10', np.nan))
            if 'ctr_v10' in sub.columns else np.nan,
        'ctx_ctr_spd10_last': float(last.get('ctr_spd10', np.nan))
            if 'ctr_spd10' in sub.columns else np.nan,
        'ctx_ctr_u100_last': float(last.get('ctr_u100', np.nan))
            if 'ctr_u100' in sub.columns else np.nan,
        'ctx_ctr_v100_last': float(last.get('ctr_v100', np.nan))
            if 'ctr_v100' in sub.columns else np.nan,
        'ctx_ctr_spd100_last': float(last.get('ctr_spd100', np.nan))
            if 'ctr_spd100' in sub.columns else np.nan,
        'ctx_ctr_msl_last': float(last.get('ctr_msl', np.nan))
            if 'ctr_msl' in sub.columns else np.nan,
        'ctx_ctr_msl_24h_change': float(
            last.get('ctr_msl', np.nan) - last24.iloc[0].get('ctr_msl', np.nan)
        ) if 'ctr_msl' in sub.columns and len(last24) > 1 else np.nan,
        'ctx_ctr_spd10_mean': float(sub['ctr_spd10'].mean())
            if 'ctr_spd10' in sub.columns else np.nan,
        'ctx_ctr_spd10_max': float(sub['ctr_spd10'].max())
            if 'ctr_spd10' in sub.columns else np.nan,
        'ctx_ctr_spd10_std': float(sub['ctr_spd10'].std())
            if 'ctr_spd10' in sub.columns else np.nan,
    }
    return feats


# ────────────────────────────────────────────────────────────────────────
# Buoy observation context features
# ────────────────────────────────────────────────────────────────────────

def buoy_context_features(
    buoy_obs_full: pd.DataFrame,  # full station obs (raw, not resampled)
    valid_time: pd.Timestamp,
    horizon_h: int,
    win_days: int = 14,
) -> dict[str, float]:
    """Features from buoy obs available BEFORE valid_time - horizon_h.

    Mirrors the inference context cutoff: at inference, context_buoy ends
    valid_time - horizon_h (approximately). We use the same convention here.
    Returns NaN-filled dict when no obs available (e.g., W1 42001 in 2022).
    """
    ctx_end = valid_time - pd.Timedelta(hours=horizon_h)
    ctx_start = ctx_end - pd.Timedelta(days=win_days)
    keys = [
        'buoy_ws_last', 'buoy_ws_last24h_mean', 'buoy_ws_last24h_max',
        'buoy_ws_ctx_mean', 'buoy_ws_ctx_max', 'buoy_ws_ctx_std',
        'buoy_dir_last_sin', 'buoy_dir_last_cos',
        'buoy_pres_last', 'buoy_pres_24h_change', 'buoy_pres_ctx_min',
        'buoy_gust_ctx_max', 'buoy_at_last', 'buoy_wt_last',
        'buoy_n_valid',
    ]
    if buoy_obs_full is None or len(buoy_obs_full) == 0:
        return {k: np.nan for k in keys}
    o = buoy_obs_full
    if 'time' not in o.columns:
        return {k: np.nan for k in keys}
    sub = o[(o['time'] > ctx_start) & (o['time'] <= ctx_end)]
    if len(sub) == 0:
        return {k: np.nan for k in keys}
    last24 = sub[sub['time'] > ctx_end - pd.Timedelta(hours=24)]
    last_row = sub.iloc[-1]
    feats = {
        'buoy_ws_last': float(last_row.get('wind_speed', np.nan)),
        'buoy_ws_last24h_mean': float(last24['wind_speed'].mean())
            if len(last24) else np.nan,
        'buoy_ws_last24h_max': float(last24['wind_speed'].max())
            if len(last24) else np.nan,
        'buoy_ws_ctx_mean': float(sub['wind_speed'].mean()),
        'buoy_ws_ctx_max': float(sub['wind_speed'].max()),
        'buoy_ws_ctx_std': float(sub['wind_speed'].std()),
        'buoy_dir_last_sin': float(np.sin(np.radians(last_row.get('wind_direction', np.nan))))
            if pd.notna(last_row.get('wind_direction', np.nan)) else np.nan,
        'buoy_dir_last_cos': float(np.cos(np.radians(last_row.get('wind_direction', np.nan))))
            if pd.notna(last_row.get('wind_direction', np.nan)) else np.nan,
        'buoy_pres_last': float(last_row.get('pressure_hPa', np.nan))
            if 'pressure_hPa' in sub.columns else np.nan,
        'buoy_pres_24h_change': float(
            last_row.get('pressure_hPa', np.nan) - last24.iloc[0].get('pressure_hPa', np.nan)
        ) if 'pressure_hPa' in sub.columns and len(last24) > 1 else np.nan,
        'buoy_pres_ctx_min': float(sub['pressure_hPa'].min())
            if 'pressure_hPa' in sub.columns else np.nan,
        'buoy_gust_ctx_max': float(sub['wind_gust'].max())
            if 'wind_gust' in sub.columns else np.nan,
        'buoy_at_last': float(last_row.get('air_temp_C', np.nan))
            if 'air_temp_C' in sub.columns else np.nan,
        'buoy_wt_last': float(last_row.get('water_temp_C', np.nan))
            if 'water_temp_C' in sub.columns else np.nan,
        'buoy_n_valid': float(sub['wind_speed'].notna().sum()),
    }
    return feats


# ────────────────────────────────────────────────────────────────────────
# Regime features — characterise the meteo state at predict time
# ────────────────────────────────────────────────────────────────────────

def regime_features(
    buoy_obs_full: pd.DataFrame | None,
    re_ts: pd.DataFrame,
    valid_time: pd.Timestamp,
    horizon_h: int,
    win_days: int = 14,
) -> dict[str, float]:
    """Regime characterisation from context window.

    Hypothesis: d7 calibration error depends on regime (trade-wind vs
    frontal/disturbed). Tagging the regime explicitly lets LGBM learn
    different residual structures per regime.

    Trade-wind signature in Caribbean/Gulf:
    - Easterly direction (60°-120°) dominant
    - Low buoy wind speed variance
    - Stable pressure (no large MSL excursions)
    - Reanalysis u10_mean negative (easterly = west-to-east vector negative)
    """
    keys = [
        'reg_east_frac', 'reg_ws_std_buoy', 'reg_ws_iqr_buoy',
        'reg_msl_range', 'reg_u10_mean', 'reg_v10_mean',
        'reg_trade_score', 'reg_disturbed_score',
    ]
    ctx_end = valid_time - pd.Timedelta(hours=horizon_h)
    ctx_start = ctx_end - pd.Timedelta(days=win_days)

    # Reanalysis-derived: always available
    re_mask = (re_ts['time'] > ctx_start) & (re_ts['time'] <= ctx_end)
    re_sub = re_ts.loc[re_mask]
    if len(re_sub) == 0:
        re_u10 = re_v10 = np.nan
        re_msl_range = np.nan
    else:
        re_u10 = float(re_sub['sp_u10_mean'].mean())
        re_v10 = float(re_sub['sp_v10_mean'].mean())
        re_msl_range = float(re_sub['sp_msl_mean'].max() - re_sub['sp_msl_mean'].min())

    # Buoy-derived: may be missing (e.g. W1 42001 in 2022)
    east_frac = np.nan
    ws_std = np.nan
    ws_iqr = np.nan
    if buoy_obs_full is not None and len(buoy_obs_full) and 'time' in buoy_obs_full.columns:
        sub = buoy_obs_full[(buoy_obs_full['time'] > ctx_start)
                            & (buoy_obs_full['time'] <= ctx_end)]
        if len(sub):
            dirs = sub['wind_direction'].dropna().values \
                if 'wind_direction' in sub.columns else np.array([])
            if len(dirs):
                east_frac = float(((dirs >= 60) & (dirs <= 120)).mean())
            ws = sub['wind_speed'].dropna().values
            if len(ws) >= 4:
                ws_std = float(ws.std())
                ws_iqr = float(np.percentile(ws, 75) - np.percentile(ws, 25))

    # Composite trade-wind score: high when easterly + low-variance + stable pressure
    # All three components in [0,1] roughly; combine multiplicatively.
    east_comp = east_frac if np.isfinite(east_frac) else 0.5
    var_comp = 1.0 / (1.0 + ws_std) if np.isfinite(ws_std) else 0.5  # higher var → lower score
    msl_comp = 1.0 / (1.0 + re_msl_range / 500.0) if np.isfinite(re_msl_range) else 0.5
    trade_score = float(east_comp * var_comp * msl_comp)

    # Disturbed score: complement signal (high variance + msl swing)
    disturbed = float((1 - east_comp) + (ws_std / 5.0 if np.isfinite(ws_std) else 0.5)
                      + (re_msl_range / 1500.0 if np.isfinite(re_msl_range) else 0.5))

    return {
        'reg_east_frac': east_frac,
        'reg_ws_std_buoy': ws_std,
        'reg_ws_iqr_buoy': ws_iqr,
        'reg_msl_range': re_msl_range,
        'reg_u10_mean': re_u10,
        'reg_v10_mean': re_v10,
        'reg_trade_score': trade_score,
        'reg_disturbed_score': disturbed,
    }


# ────────────────────────────────────────────────────────────────────────
# Temporal / station / climatology
# ────────────────────────────────────────────────────────────────────────

def temporal_features(valid_time: pd.Timestamp, hour: int) -> dict[str, float]:
    doy = valid_time.dayofyear
    return {
        'doy_sin': float(np.sin(2 * np.pi * doy / 365.25)),
        'doy_cos': float(np.cos(2 * np.pi * doy / 365.25)),
        'hour_sin': float(np.sin(2 * np.pi * hour / 24)),
        'hour_cos': float(np.cos(2 * np.pi * hour / 24)),
    }


def build_climatology(buoy_obs_6h: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-station per-DOY-bucket per-hour climatology of wind speed quantiles.

    Returns a long DataFrame indexed by (station, doy_bucket, hour) with
    q05/q50/q95 of wind_speed.
    """
    rows = []
    for buoy, obs in buoy_obs_6h.items():
        df = obs.copy()
        df['station'] = buoy
        df['doy'] = df['time'].dt.dayofyear
        df['hour'] = df['time'].dt.hour
        # 15-day moving DOY bucket — robust climatology for sparse tails
        for b_center in range(1, 367, 5):
            for hour in [0, 6, 12, 18]:
                # circular DOY distance ≤ 15 days, same hour
                d = np.abs(df['doy'].values - b_center)
                d = np.minimum(d, 365 - d)
                mask = (d <= 15) & (df['hour'].values == hour)
                ws = df.loc[mask, 'wind_speed'].dropna().values
                if len(ws) < 10:
                    continue
                rows.append({
                    'station': buoy, 'doy_bucket': b_center, 'hour': hour,
                    'clim_q05': float(np.percentile(ws, 5)),
                    'clim_q50': float(np.percentile(ws, 50)),
                    'clim_q95': float(np.percentile(ws, 95)),
                    'clim_mean': float(ws.mean()),
                    'clim_std': float(ws.std()),
                    'clim_max': float(ws.max()),
                    'clim_dir_sin': float(np.sin(np.radians(
                        df.loc[mask, 'wind_direction'].dropna().values
                    )).mean()),
                    'clim_dir_cos': float(np.cos(np.radians(
                        df.loc[mask, 'wind_direction'].dropna().values
                    )).mean()),
                })
    return pd.DataFrame(rows)


def climatology_lookup(clim: pd.DataFrame, station: str, valid_time: pd.Timestamp,
                       hour: int) -> dict[str, float]:
    doy = valid_time.dayofyear
    sub = clim[(clim['station'] == station) & (clim['hour'] == hour)]
    if len(sub) == 0:
        return {k: np.nan for k in ['clim_q05', 'clim_q50', 'clim_q95',
                                     'clim_mean', 'clim_std', 'clim_max',
                                     'clim_dir_sin', 'clim_dir_cos']}
    # nearest bucket
    d = np.abs(sub['doy_bucket'].values - doy)
    d = np.minimum(d, 365 - d)
    row = sub.iloc[np.argmin(d)]
    return {k: float(row[k]) for k in ['clim_q05', 'clim_q50', 'clim_q95',
                                        'clim_mean', 'clim_std', 'clim_max',
                                        'clim_dir_sin', 'clim_dir_cos']}


# ────────────────────────────────────────────────────────────────────────
# Buoy 6h resample helper
# ────────────────────────────────────────────────────────────────────────

def resample_buoy_obs(obs: pd.DataFrame, hours=(0, 6, 12, 18)) -> pd.DataFrame:
    """6-hourly mean of buoy obs, keep only scoring hours."""
    o = obs.copy()
    o['time'] = pd.to_datetime(o['time'])
    o = (o.set_index('time')[['wind_speed', 'wind_direction',
                               'pressure_hPa', 'air_temp_C', 'water_temp_C']]
          .resample('6h').mean().reset_index())
    return o[o['time'].dt.hour.isin(hours)].reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────
# HRES feature transforms
# ────────────────────────────────────────────────────────────────────────

def hres_to_features(hres: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a HRES forecast frame."""
    h = hres.copy()
    h['hres_dir_sin'] = np.sin(np.radians(h['hres_dir']))
    h['hres_dir_cos'] = np.cos(np.radians(h['hres_dir']))
    h['hres_speed_5m'] = h['hres_speed'] * H_SCALE  # downscale to buoy height
    return h
