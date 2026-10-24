"""OPTIONAL review aid: Isolation Forest scores on engineered features.

It is NOT used to decide flags. It writes iforest_candidates.csv so a human can look at stretches the rules
did not flag. There is no labelled fault data, so the forest cannot be trained as a classifier, and an
unusual combination of readings is as likely to be real weather as a fault. Needs scikit-learn; skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .peer_frames import Peers
from .solar import clear_sky_ghi, solar_elevation_deg
from .stats_utils import MAD_TO_SIGMA, runs_of


def score_stations(stations: dict, top_fraction: float = 0.005) -> pd.DataFrame:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return pd.DataFrame(columns=["station_id", "start_utc", "end_utc", "rows", "worst_score", "note"])
    peers = Peers(stations)
    rows = []
    for sid, st in stations.items():
        idx = st.clean.index
        feats = {}
        for ch in ("temp_c", "rh_pct", "pressure_hpa"):
            r = peers.residual(sid, ch)
            feats[f"res_{ch}"] = r
            feats[f"d10_{ch}"] = st.usable(ch).diff()
        w, ref = peers.frames["wind_kmh"][sid], peers.others(sid, "wind_kmh").median(axis=1)
        feats["wind_logratio"] = np.log((w + 1) / (ref + 1)).rolling(6, min_periods=3).median()
        if not np.isnan(st.lat_for_solar):
            el = solar_elevation_deg(idx, st.lat_for_solar, st.lon_for_solar)
            cs = clear_sky_ghi(el)
            feats["ghi_clearsky"] = pd.Series(np.where(el > 10, st.usable("ghi_wm2").values / np.maximum(cs, 1), 0.0), index=idx)
        wd = np.radians(st.usable("wind_dir_deg"))
        feats["wdir_sin"], feats["wdir_cos"] = np.sin(wd), np.cos(wd)
        X = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
        # per-station robust standardisation so elevation and siting offsets do not dominate
        med = X.median()
        sig = (X - med).abs().median() * MAD_TO_SIGMA
        Z = ((X - med) / sig.replace(0, np.nan)).clip(-20, 20).fillna(0.0)
        if len(Z) < 200:
            continue
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(Z)
        score = pd.Series(-model.score_samples(Z), index=idx)
        thr = score.quantile(1 - top_fraction)
        hit = score >= thr
        for a, b in runs_of(hit.reset_index(drop=True)):
            seg = Z.iloc[a:b + 1]
            worst = seg.abs().max().idxmax()
            rows.append({"station_id": sid, "start_utc": idx[a].strftime("%Y-%m-%d %H:%M"), "end_utc": idx[b].strftime("%Y-%m-%d %H:%M"),
                         "rows": b - a + 1, "worst_score": round(float(score.iloc[a:b + 1].max()), 3),
                         "note": f"most unusual feature: {worst} (robust z {seg[worst].abs().max():.1f}); review, not a verdict"})
    return pd.DataFrame(rows, columns=["station_id", "start_utc", "end_utc", "rows", "worst_score", "note"])
