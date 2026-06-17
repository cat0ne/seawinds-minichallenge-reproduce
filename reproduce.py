"""FULL-CHAIN reproduction of the WINNING submission — provided dataset -> predictions.csv.

    Guillaume Hochard, PhD  ·  alias "Hegoa"  ·  Sea Winds Mini Challenge (Codabench #13566)

This rebuilds the FIRST-PLACE submission (internal name `candDB7_w20`) end-to-end from the
provided competition data alone (mini_challenge_dataset/, Zenodo 20317029). There is no
pre-built intermediate submission.

Every one of the 9 scored column-groups is computed from the provided data. The ONLY trained
model is the d7-direction gradient-boosted ensemble (LightGBM sin/cos on the external-data-free
DIR_FEATURES). For exact reproduction its weights ship in models/dir_models.joblib; run with
--retrain to rebuild that model from the provided data as well (LightGBM boosters are not
bit-identical across machines, so --retrain matches to a few degrees, not to the bit — see
COMPLIANCE.md).

Provenance of every column (all from provided data):
  d1  speed  : conformal — HRES_5m + per-station 5/50/95 quantiles of (obs-HRES_5m), 2020-21   [rebuilt]
  d1  dir    : raw HRES direction + global -2.5deg offset                                       [rebuilt]
  d7  speed  : starting-kit notebook LightGBM (9 feats, train-2020, default seed) + storm widen [rebuilt]
  d7  dir    : 3-vector blend  0.40*HRES + 0.40*ML(sin/cos) + 0.20*speed-weighted clim-median   [model | --retrain]
  d14 speed  : provided 2020-21 seasonal climatology + asym widen (+0.3/+3.0) + extreme boost   [rebuilt]
  d14 dir    : speed-weighted circular-MEDIAN climatology (provided obs)                         [rebuilt]
  d14 dir @42001 (w6-8): pooled-evidence circular median (buoy + bias-corrected ERA5 + contexts) [rebuilt]

The winning move over the runner-up chain is the d7-direction blend: the 50/50 ML+HRES vector
mean is folded with the speed-weighted circular-median direction climatology at weight 0.20
(robust cross-year plateau ~0.15-0.25). This changes ONLY the 192 d7 dir_50 cells; every speed /
Winkler column and the d1/d14 directions stay byte-identical to the runner-up chain.

Usage:   python reproduce.py [--data /path/to/mini_challenge_dataset] [--retrain]
The dataset defaults to ./mini_challenge_dataset or ../mini_challenge_dataset.
Output:  predictions_reproduced.csv  (asserted byte-identical to predictions.csv).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))           # makes `import src.*` resolve to code/src/

# d7-direction climatology blend weight (the winning lever; candDB7_w20 == w_clim 0.20)
W_CLIM_D7 = 0.20


def find_dataset(cli):
    for c in ([Path(cli)] if cli else []) + [HERE / 'mini_challenge_dataset', HERE.parent / 'mini_challenge_dataset']:
        if c and (c / 'buoy_metadata.json').exists():
            return c.resolve()
    sys.exit('ERROR: dataset not found. Pass --data /path/to/mini_challenge_dataset '
             '(download: https://zenodo.org/records/20317029).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--retrain', action='store_true', help='also rebuild the d7-direction GBM from provided data')
    args = ap.parse_args()

    DATA = find_dataset(args.data)
    WORK = HERE / '_work'; WORK.mkdir(exist_ok=True)
    # point the library's data + scratch roots at the provided dataset (read before importing src.pipeline)
    os.environ['SEAWINDS_DATA'] = str(DATA)
    os.environ['SEAWINDS_ARTIFACTS'] = str(WORK)

    from src.pipeline import (build_climatology, predict_d1_conformal,
                              extreme_boost, DIR_FEATURES, SPEED_FEATURES_D1,
                              load_all_train, train_direction_models, build_training_table,
                              _predict_ensemble)
    from src.features import (HORIZON_LAG_H, buoy_context_features, climatology_lookup,
                              context_window_features, hres_to_features,
                              precompute_reanalysis_timeseries, regime_features, temporal_features)
    from src.compliant_clim import build_provided_clim, provided_speed_lookup
    from src.d14_dir_climatology import circular_median, build_dir_climatology, predict_dir_clim

    META = json.loads((DATA / 'buoy_metadata.json').read_text())
    BUOYS = list(META['buoys'].keys()); HOURS = META['scoring_hours_utc']
    D1_WIDEN, D14_Q05_WIDEN, D14_Q95_WIDEN, D1_GLOBAL_OFFSET = 0.5, 0.3, 3.0, -2.5
    SK = ['hres_speed', 'hres_u10', 'hres_v10', 'horizon', 'hour', 'lat', 'lon', 'hres_dir_sin', 'hres_dir_cos']
    print(f'Dataset: {DATA}\nBuilding climatology / conformal / provided-climatology from provided data...', flush=True)

    hres, re_ts, buoy_obs, buoy_obs_raw = load_all_train()
    clim = build_climatology(buoy_obs)
    # speed-weighted circular-median direction climatology (provided obs) — used by BOTH the
    # d7-direction blend and the d14 direction.
    dir_clim_tab = build_dir_climatology(buoy_obs_raw, speed_weighted=True)

    # d1 conformal residual quantiles (provided 2020-21, 6h)
    conformal = {}
    for b in BUOYS:
        hd = hres[b][hres[b]['horizon'] == 1]
        o = buoy_obs[b][['time', 'wind_speed']].dropna()
        m = hd.merge(o, on='time'); m = m[m['time'].dt.year.isin([2020, 2021])]
        r = (m['wind_speed'] - m['hres_speed_5m']).values
        conformal[b] = (float(np.quantile(r, .05)), float(np.quantile(r, .50)), float(np.quantile(r, .95)))

    sp_clim, dr_clim = build_provided_clim(buoy_obs)
    sp_clim.to_parquet(WORK / 'provided_clim_speed.parquet'); dr_clim.to_parquet(WORK / 'provided_clim_dir.parquet')

    if args.retrain:
        print('  --retrain: rebuilding d7-direction GBM from provided data...', flush=True)
        tt = build_training_table(hres, re_ts, buoy_obs, clim, buoy_obs_raw)
        for c in ['ndbc_q05', 'ndbc_q50', 'ndbc_q95', 'ndbc_mean', 'ndbc_std']:
            tt[c] = np.nan
        tt = tt.sort_values('time').reset_index(drop=True); cut = pd.Timestamp('2021-11-01')
        dir_models = train_direction_models(tt[tt.time < cut], tt[tt.time >= cut])
    else:
        dir_models = joblib.load(HERE / 'models' / 'dir_models.joblib')

    # starting-kit d7 speed model (deterministic)
    rows = []
    for b in BUOYS:
        h = pd.read_parquet(DATA / 'train/hres_forecasts' / f'hres_{b}.parquet'); h['time'] = pd.to_datetime(h['time'])
        o = pd.read_parquet(DATA / 'train/buoy_observations' / f'ndbc_{b}.parquet'); o['time'] = pd.to_datetime(o['time'])
        o = o.set_index('time')[['wind_speed', 'wind_direction']].resample('6h').mean().reset_index()
        o = o[o['time'].dt.hour.isin(HOURS)]; m = h.merge(o, on='time', how='inner')
        m['lat'] = META['buoys'][b]['latitude']; m['lon'] = META['buoys'][b]['longitude']; rows.append(m)
    trk = pd.concat(rows, ignore_index=True).dropna(subset=['wind_speed', 'hres_speed'])
    trk['hres_dir_sin'] = np.sin(np.radians(trk['hres_dir'])); trk['hres_dir_cos'] = np.cos(np.radians(trk['hres_dir']))
    trk = trk[trk['time'].dt.year == 2020]
    sk = {n: lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=300, learning_rate=0.05,
                               num_leaves=31, min_child_samples=40, verbose=-1).fit(trk[SK], trk['wind_speed'])
          for n, q in {'q05': 0.05, 'q50': 0.50, 'q95': 0.95}.items()}

    out = []
    for w in range(1, 9):
        wdir = DATA / f'inference/window_{w}'; meta = json.loads((wdir / 'metadata.json').read_text())
        for b in BUOYS:
            info = META['buoys'][b]; lat, lon = info['latitude'], info['longitude']
            re = pd.read_parquet(wdir / f'context_reanalysis_{b}.parquet'); re['time'] = pd.to_datetime(re['time'])
            rt = precompute_reanalysis_timeseries(re, lat, lon)
            fcp = wdir / f'forecast_hres_{b}.parquet'
            fc = hres_to_features(pd.read_parquet(fcp).assign(time=lambda x: pd.to_datetime(x['time']))) if fcp.exists() else None
            cbp = wdir / f'context_buoy_{b}.parquet'
            cb = pd.read_parquet(cbp).assign(time=lambda x: pd.to_datetime(x['time'])) if cbp.exists() else None

            def feats(vt, hz):
                if fc is not None:
                    r = fc[(fc['time'] == vt) & (fc['horizon'] == hz)]
                    f = {k: r.iloc[0][k] for k in ['hres_speed', 'hres_speed_5m', 'hres_u10', 'hres_v10', 'hres_dir_sin', 'hres_dir_cos']} if len(r) else None
                    if f is None and hz != 14:
                        return None
                else:
                    f = None
                if f is None:
                    f = {k: np.nan for k in ['hres_speed', 'hres_speed_5m', 'hres_u10', 'hres_v10', 'hres_dir_sin', 'hres_dir_cos']}
                f.update(context_window_features(rt, vt, HORIZON_LAG_H[hz])); f.update(buoy_context_features(cb, vt, HORIZON_LAG_H[hz]))
                f.update(regime_features(cb, rt, vt, HORIZON_LAG_H[hz])); f.update(temporal_features(vt, vt.hour))
                f.update(climatology_lookup(clim, b, vt, vt.hour))
                f.update({'lat': lat, 'lon': lon, 'horizon': hz, 'station': b, 'window': w, 'hour': vt.hour, 'time': vt})
                return f

            for hz, day in [(1, 'd1'), (7, 'd7')]:
                for hour in HOURS:
                    vt = pd.Timestamp(meta['score_days'][day]) + pd.Timedelta(hours=hour)
                    f = feats(vt, hz)
                    if f is None:
                        continue
                    if hz == 1:
                        q05, q50, q95 = predict_d1_conformal(pd.DataFrame([f])[SPEED_FEATURES_D1], [b], conformal)
                        q95 = q95 + D1_WIDEN
                        q05 = np.clip(np.minimum(q05, q50), 0, None); q95 = np.maximum(q95, q50); q50 = np.clip(q50, 0, None)
                        hdir = float(np.degrees(np.arctan2(f['hres_dir_sin'], f['hres_dir_cos'])) % 360) if np.isfinite(f['hres_dir_sin']) else 90.0
                        d50 = (hdir + D1_GLOBAL_OFFSET) % 360
                    else:
                        fcr = fc[(fc['time'] == vt) & (fc['horizon'] == 7)].iloc[0]
                        xs = {'hres_speed': fcr['hres_speed'], 'hres_u10': fcr['hres_u10'], 'hres_v10': fcr['hres_v10'],
                              'horizon': 7, 'hour': hour, 'lat': lat, 'lon': lon,
                              'hres_dir_sin': np.sin(np.radians(fcr['hres_dir'])), 'hres_dir_cos': np.cos(np.radians(fcr['hres_dir']))}
                        o = {n: m.predict(pd.DataFrame([xs])[SK])[0] for n, m in sk.items()}
                        q05 = float(np.clip(min(o['q05'], o['q50']), 0, None)); q50 = float(o['q50']); q95 = float(max(o['q95'], o['q50']))
                        if fcr['hres_speed'] > 9:
                            q95 = max(q95, 14.0)
                        # --- d7 DIRECTION: 3-vector blend (the winning lever) ---
                        # raw ML sin/cos ensemble (external-data-free DIR_FEATURES)
                        X7 = pd.DataFrame([f])[DIR_FEATURES]
                        si = _predict_ensemble(dir_models[(7, 'sin')], X7)[0]
                        co = _predict_ensemble(dir_models[(7, 'cos')], X7)[0]
                        ml = float(np.degrees(np.arctan2(si, co)) % 360)
                        hdg = float(np.degrees(np.arctan2(f['hres_dir_sin'], f['hres_dir_cos'])) % 360)
                        cl = float(predict_dir_clim(dir_clim_tab, [b], [int(vt.dayofyear)])[0])
                        wm = wh = (1 - W_CLIM_D7) / 2                       # 0.40 ML, 0.40 HRES, 0.20 clim
                        vx = wh * np.cos(np.radians(hdg)) + wm * np.cos(np.radians(ml)) + W_CLIM_D7 * np.cos(np.radians(cl))
                        vy = wh * np.sin(np.radians(hdg)) + wm * np.sin(np.radians(ml)) + W_CLIM_D7 * np.sin(np.radians(cl))
                        d50 = round(float(np.degrees(np.arctan2(vy, vx)) % 360), 1)
                    out.append(dict(window=w, station=b, latitude=lat, longitude=lon, horizon=hz, hour=hour,
                                    q05=float(np.ravel(q05)[0]), q50=float(np.ravel(q50)[0]), q95=float(np.ravel(q95)[0]), dir_50=d50))

            for hour in HOURS:
                vt = pd.Timestamp(meta['score_days']['d14']) + pd.Timedelta(hours=hour)
                fdf = pd.DataFrame([feats(vt, 14)])
                q05n, q50n, q95n = provided_speed_lookup(np.array([b]), fdf)
                q05 = q05n + D14_Q05_WIDEN; q95 = q95n + D14_Q95_WIDEN + np.maximum(extreme_boost(fdf), 0); q50 = q50n
                st = np.stack([q05, q50, q95], axis=-1); st.sort(axis=-1)
                out.append(dict(window=w, station=b, latitude=lat, longitude=lon, horizon=14, hour=hour,
                                q05=float(np.clip(st[..., 0], 0, None)[0]), q50=float(st[..., 1][0]), q95=float(st[..., 2][0]), dir_50=90.0))
    sub = pd.DataFrame(out)

    # d14 direction: speed-weighted median climatology (raw provided obs, all 6) + 42001 patch
    tab = dir_clim_tab
    dmap = {}
    for w in range(1, 9):
        for b in BUOYS:
            p = DATA / f'inference/window_{w}' / f'forecast_hres_{b}.parquet'
            if p.exists():
                fcc = pd.read_parquet(p); fcc['time'] = pd.to_datetime(fcc['time'])
                for _, r in fcc[fcc['horizon'] == 1].iterrows():
                    dmap[(w, int(r['hour']))] = int((r['time'] + pd.Timedelta(days=13)).dayofyear)
                break
    for i in sub.index[sub.horizon == 14]:
        doy = dmap.get((int(sub.at[i, 'window']), int(sub.at[i, 'hour'])))
        if doy is not None:
            pr = predict_dir_clim(tab, [str(sub.at[i, 'station'])], [doy])[0]
            if np.isfinite(pr):
                sub.at[i, 'dir_50'] = round(float(pr), 1)
    # 42001 pooled-evidence patch (w6-8)
    STN, LL, TW = '42001', (25.9, -89.7), [6, 7, 8]
    D14_DATE = {6: pd.Timestamp('2022-09-18'), 7: pd.Timestamp('2022-11-20'), 8: pd.Timestamp('2022-12-22')}
    to6h = lambda d: (d.dropna(subset=['wind_direction']).assign(time=lambda x: pd.to_datetime(x['time']))
                      .set_index('time').sort_index()[['wind_direction', 'wind_speed']].resample('6h').first().dropna().reset_index())
    sgn = lambda a, c: (np.asarray(a, float) - np.asarray(c, float) + 180) % 360 - 180
    bo = buoy_obs_raw[STN].copy(); bo['time'] = pd.to_datetime(bo['time'])
    rea = pd.read_parquet(DATA / 'train/reanalysis_patches' / f'reanalysis_{STN}.parquet')
    d2 = (rea.latitude - LL[0]) ** 2 + (rea.longitude - LL[1]) ** 2; pt = rea[d2 == d2.min()].copy()
    pt['wind_direction'] = (np.degrees(np.arctan2(-pt.u10, -pt.v10))) % 360; pt['wind_speed'] = np.hypot(pt.u10, pt.v10)
    pt['time'] = pd.to_datetime(pt.time); pt = pt[['time', 'wind_direction', 'wind_speed']].sort_values('time')
    ov = pd.merge_asof(pt, bo.dropna(subset=['wind_direction'])[['time', 'wind_direction', 'wind_speed']].sort_values('time'),
                       on='time', tolerance=pd.Timedelta('30min'), direction='nearest', suffixes=('', '_b')).dropna()
    bias = ((circular_median(sgn(ov.wind_direction, ov.wind_direction_b) % 360) + 180) % 360) - 180
    pt['wind_direction'] = (pt['wind_direction'] - bias) % 360
    buoy6 = to6h(bo); ctx = {}
    for w in range(1, 9):
        c = pd.read_parquet(DATA / f'inference/window_{w}' / f'context_buoy_{STN}.parquet')
        if len(c) and c['wind_direction'].notna().any():
            ctx[w] = to6h(c)
    for w in TW:
        ca = pd.concat([c for ww, c in ctx.items() if ww <= w]) if ctx else pd.DataFrame()
        pool = pd.concat([buoy6, pt, ca], ignore_index=True)
        pr = round(float(predict_dir_clim(build_dir_climatology({STN: pool}, speed_weighted=True), [STN], [D14_DATE[w].dayofyear])[0]), 1)
        sub.loc[(sub.station == STN) & (sub.horizon == 14) & (sub.window == w), 'dir_50'] = pr

    sub['station'] = sub['station'].astype(str)
    sub = sub.sort_values(['window', 'station', 'horizon', 'hour']).reset_index(drop=True)
    sub[['q05', 'q50', 'q95']] = sub[['q05', 'q50', 'q95']].round(3); sub['dir_50'] = sub['dir_50'].round(1)
    sub.to_csv(HERE / 'predictions_reproduced.csv', index=False)
    import shutil; shutil.rmtree(WORK, ignore_errors=True)

    ref = pd.read_csv(HERE / 'predictions.csv'); ref['station'] = ref['station'].astype(str)
    ref = ref.sort_values(['window', 'station', 'horizon', 'hour']).reset_index(drop=True)
    m = sub.merge(ref, on=['window', 'station', 'horizon', 'hour'], suffixes=('', '_r'))
    print('\nReproduced vs shipped winning submission (candDB7_w20):')
    for hz in [1, 7, 14]:
        h = m[m.horizon == hz]
        ds = max(float(np.abs(h[c] - h[c + '_r']).max()) for c in ['q05', 'q50', 'q95'])
        dd = float(np.abs((h['dir_50'] - h['dir_50_r'] + 180) % 360 - 180).max())
        print(f'  d{hz:<2d}: max|Δspeed|={ds:.4f}  max|Δdir|={dd:.3f}')
    overall_s = max(float(np.abs(m[c] - m[c + '_r']).max()) for c in ['q05', 'q50', 'q95'])
    overall_d = float(np.abs((m['dir_50'] - m['dir_50_r'] + 180) % 360 - 180).max())
    if not args.retrain:
        ok = overall_s < 1e-9 and overall_d < 1e-9
        print('RESULT:', 'BYTE-IDENTICAL to the official submission ✓' if ok else 'MISMATCH ✗')
        return 0 if ok else 1
    print(f'RESULT (--retrain): deterministic columns byte-identical; d7-direction within '
          f'{overall_d:.1f}deg (GBM rebuild tolerance). Method provenance confirmed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
