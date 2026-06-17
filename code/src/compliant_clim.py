"""Provided-data-only d14 seasonal climatology (replaces the external NDBC long-record).
Per (station, doy-window, hour): empirical speed quantiles + circular-mean direction, built ONLY
from the provided 2020-2021 buoy observations. Schemas match the NDBC artifacts so the production
lookup shape is reused."""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
import numpy as np
import pandas as pd

from src.pipeline import ARTIFACTS, BUOYS, load_all_train

HOURS = [0, 6, 12, 18]


def _circ_mean_deg(deg):
    r = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.nanmean(np.sin(r)), np.nanmean(np.cos(r)))) % 360)


def build_provided_clim(buoy_obs, window_doy=15, min_n=20):
    """Returns (speed_df, dir_df) with NDBC-compatible schemas, from provided obs only."""
    sp_rows, dir_rows = [], []
    for b in BUOYS:
        o = buoy_obs[b].copy()
        o['time'] = pd.to_datetime(o['time'])
        o['hour'] = o['time'].dt.hour
        o['doy'] = o['time'].dt.dayofyear
        o = o[o['hour'].isin(HOURS)]
        for hr in HOURS:
            sub = o[o['hour'] == hr]
            doys = sub['doy'].values
            spd = sub['wind_speed'].values
            drc = sub['wind_direction'].values
            for doy in range(1, 367):
                dd = np.abs(doys - doy); dd = np.minimum(dd, 365 - dd)
                m = dd <= window_doy
                sp = spd[m]; sp = sp[np.isfinite(sp) & (sp >= 0) & (sp < 60)]
                if len(sp) >= min_n:
                    sp_rows.append({'station': str(b), 'doy': doy, 'hour': hr,
                                    'q05': float(np.quantile(sp, 0.05)),
                                    'q50': float(np.quantile(sp, 0.50)),
                                    'q95': float(np.quantile(sp, 0.95)),
                                    'mean': float(sp.mean()), 'std': float(sp.std()), 'n': int(len(sp))})
                dr = drc[m]; dr = dr[np.isfinite(dr) & (dr >= 0) & (dr < 360)]
                if len(dr) >= min_n:
                    dir_rows.append({'station': str(b), 'doy': doy, 'hour': hr,
                                     'dir_clim': _circ_mean_deg(dr), 'n': int(len(dr))})
    return pd.DataFrame(sp_rows), pd.DataFrame(dir_rows)


_CACHE = {'speed': None, 'dir': None}


def _doy_from_X(X):
    doy = (np.arctan2(X['doy_sin'].values, X['doy_cos'].values) * 365.25 / (2 * np.pi)) % 365.25
    doy = np.round(doy).astype(int); doy[doy < 1] = 1; doy[doy > 366] = 366
    hour = (np.arctan2(X['hour_sin'].values, X['hour_cos'].values) * 24 / (2 * np.pi)) % 24
    hour = (np.round(hour).astype(int)) % 24
    return doy, hour


def provided_speed_lookup(stations, X):
    """(q05,q50,q95) per row from provided_clim_speed.parquet; None if unavailable."""
    if _CACHE['speed'] is None:
        p = ARTIFACTS / 'provided_clim_speed.parquet'
        if not p.exists():
            return None
        _CACHE['speed'] = pd.read_parquet(p)
    if 'doy_sin' not in X.columns:
        return None
    df = _CACHE['speed']; lut = df.set_index(['station', 'doy', 'hour'])[['q05', 'q50', 'q95']].to_dict('index')
    doy, hour = _doy_from_X(X)
    q05 = np.empty(len(X)); q50 = np.empty(len(X)); q95 = np.empty(len(X))
    for i, (st, d, h) in enumerate(zip(map(str, stations), doy, hour)):
        v = lut.get((st, int(d), int(h)))
        if v is None:  # nearest available doy for that (station,hour)
            cand = df[(df.station == st) & (df.hour == int(h))]
            if not len(cand):
                return None
            j = (np.minimum(np.abs(cand.doy.values - d), 365 - np.abs(cand.doy.values - d))).argmin()
            v = cand.iloc[j][['q05', 'q50', 'q95']].to_dict()
        q05[i], q50[i], q95[i] = v['q05'], v['q50'], v['q95']
    return q05, q50, q95


def provided_dir_lookup(stations, X):
    """dir_clim (deg) per row from provided_clim_dir.parquet; falls back to 90.0."""
    if _CACHE['dir'] is None:
        p = ARTIFACTS / 'provided_clim_dir.parquet'
        if not p.exists():
            return None
        _CACHE['dir'] = pd.read_parquet(p)
    if 'doy_sin' not in X.columns:
        return None
    df = _CACHE['dir']; lut = df.set_index(['station', 'doy', 'hour'])['dir_clim'].to_dict()
    doy, hour = _doy_from_X(X)
    out = np.empty(len(X))
    for i, (st, d, h) in enumerate(zip(map(str, stations), doy, hour)):
        v = lut.get((st, int(d), int(h)))
        if v is None:  # nearest available doy for that (station,hour); else 90.0
            cand = df[(df.station == st) & (df.hour == int(h))]
            if len(cand):
                j = (np.minimum(np.abs(cand.doy.values - d), 365 - np.abs(cand.doy.values - d))).argmin()
                v = float(cand.iloc[j]['dir_clim'])
            else:
                v = 90.0
        out[i] = v
    return out


if __name__ == '__main__':
    import sys
    win = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    hres, re_ts, buoy_obs, buoy_obs_raw = load_all_train()
    sp, dr = build_provided_clim(buoy_obs, window_doy=win)
    sp.to_parquet(ARTIFACTS / 'provided_clim_speed.parquet')
    dr.to_parquet(ARTIFACTS / 'provided_clim_dir.parquet')
    # self-check: schemas match NDBC, all 6 stations present, quantiles monotone, dirs in range
    ndbc_cols = ['station', 'doy', 'hour', 'q05', 'q50', 'q95', 'mean', 'std', 'n']
    assert list(sp.columns) == ndbc_cols, list(sp.columns)
    assert set(sp.station) == set(map(str, BUOYS)), set(sp.station)
    assert (sp.q05 <= sp.q50).all() and (sp.q50 <= sp.q95).all()
    assert dr.dir_clim.between(0, 360).all()
    print(f'window={win}d  speed_rows={len(sp)} dir_rows={len(dr)}  '
          f'speed cells/station={len(sp)//6}  median n={int(sp.n.median())}')
    print('self-check OK')
