"""Physical-validity checks: values that cannot be true whatever the weather was."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gps_checks import std_atm_pressure
from .models import Candidate
from .solar import clear_sky_ghi, etr_horizontal, solar_elevation_deg
from .station import Station
from .stats_utils import merge_runs, runs_of

# ---- physical limits and why --------------------------------------------------------------
# Air temperature: world records are -89.2 and +56.7 C; anything outside -90..60 is not weather.
TEMP_LIMITS = (-90.0, 60.0)
# Relative humidity cannot exceed 100%. Capacitive sensors may read ~0.5 above 100 from rounding.
RH_MAX = 100.5
# 10-minute mean wind above 250 km/h would be a violent tornado over a whole interval.
WIND_MAX_KMH = 250.0
WIND_DIR_RANGE = (0.0, 360.0)   # 360 is accepted: "from the north" is written as either 0 or 360
# Tipping bucket: 40 mm in 10 minutes is above any recorded 10-minute rainfall; scaled with the interval.
RAIN_MAX_PER_10MIN = 40.0
# Station pressure must lie within +-75 hPa of the standard atmosphere at the station's elevation
# (strongest storms move pressure by ~40-60 hPa; the margin keeps real storms out of the flags).
PRESSURE_BAND_HPA = 75.0
# Night: below -6 deg (end of civil twilight) healthy stations in this pack never exceed 11 W/m2, so 30 is 3x margin.
NIGHT_EL_DEG, NIGHT_GHI_MAX = -6.0, 30.0
# Daylight GHI above the extra-terrestrial ceiling (+100 for instrument tolerance) for a full hour is impossible
# (single readings near sunrise/sunset can overshoot because hourly reanalysis-style values are smeared across the hour).
ETR_MARGIN = 100.0
# In full daylight (sun >15 deg) a working pyranometer never reads exactly 0 for 30 minutes, even under thick cloud.
DAY_ZERO_EL_DEG, DAY_ZERO_MIN_SAMPLES = 15.0, 3
MERGE_GAP = 6  # samples; impossible readings less than an hour apart are one fault


def _runs_to_candidates(st: Station, ch: str, mask: pd.Series, reason: str, issue: str, text_fn, family="stuck",
                        merge_gap: int = 0, min_len: int = 1):
    out = []
    mask = mask & st.clean[ch].notna()
    runs = merge_runs(runs_of(mask.reset_index(drop=True)), merge_gap) if merge_gap else runs_of(mask.reset_index(drop=True))
    for a, b in runs:
        if b - a + 1 < min_len:
            continue
        idx = st.clean.index
        sub = st.clean[ch].iloc[a:b + 1]
        out.append(Candidate(st.sid, ch, idx[a], idx[b], reason, family, issue, "exclude",
                             text_fn(sub, idx[a], idx[b], int(mask.iloc[a:b + 1].sum())),
                             evidence={"rows": int(mask.iloc[a:b + 1].sum()), "vmax": float(sub.max()), "vmin": float(sub.min())}))
    return out


def check_physical(st: Station) -> list[Candidate]:
    c = st.clean
    out: list[Candidate] = []

    # temperature
    lo, hi = TEMP_LIMITS
    out += _runs_to_candidates(st, "temp_c", (c.temp_c < lo) | (c.temp_c > hi), "impossible", "Temperature impossible",
                               lambda s, a, b, n: f"{n} reading(s) between {s.min():.1f} and {s.max():.1f} C lie outside the physical range {lo:.0f}..{hi:.0f} C.",
                               merge_gap=MERGE_GAP)
    # humidity
    out += _runs_to_candidates(st, "rh_pct", (c.rh_pct > RH_MAX) | (c.rh_pct < 0), "impossible", "Humidity impossible",
                               lambda s, a, b, n: f"{n} reading(s) up to {s.max():.0f}% - relative humidity cannot exceed 100%.",
                               merge_gap=MERGE_GAP)
    # pressure: absolute band around the standard atmosphere for this elevation
    if not np.isnan(st.elev):
        p_std = float(std_atm_pressure(st.elev))
        out += _runs_to_candidates(st, "pressure_hpa", (c.pressure_hpa - p_std).abs() > PRESSURE_BAND_HPA, "impossible",
                                   "Pressure implausible for elevation",
                                   lambda s, a, b, n: f"{n} reading(s) ({s.min():.0f}..{s.max():.0f} hPa) are more than {PRESSURE_BAND_HPA:.0f} hPa from the {p_std:.0f} hPa a station at {st.elev:.0f} m normally reads.",
                                   merge_gap=MERGE_GAP)
    else:
        out += _runs_to_candidates(st, "pressure_hpa", (c.pressure_hpa < 300) | (c.pressure_hpa > 1100), "impossible", "Pressure impossible",
                                   lambda s, a, b, n: f"{n} reading(s) outside 300..1100 hPa.", merge_gap=MERGE_GAP)
    # wind
    out += _runs_to_candidates(st, "wind_kmh", (c.wind_kmh < 0) | (c.wind_kmh > WIND_MAX_KMH), "impossible", "Wind speed impossible",
                               lambda s, a, b, n: f"{n} reading(s) ({s.min():.0f}..{s.max():.0f} km/h) outside 0..{WIND_MAX_KMH:.0f} km/h.",
                               merge_gap=MERGE_GAP)
    out += _runs_to_candidates(st, "wind_dir_deg", (c.wind_dir_deg < WIND_DIR_RANGE[0]) | (c.wind_dir_deg > WIND_DIR_RANGE[1]), "impossible",
                               "Wind direction impossible",
                               lambda s, a, b, n: f"{n} reading(s) ({s.min():.0f}..{s.max():.0f} deg) outside 0..360.", merge_gap=MERGE_GAP)
    # rain
    cap = RAIN_MAX_PER_10MIN * st.interval / 10.0
    out += _runs_to_candidates(st, "rain_mm", (c.rain_mm < 0) | (c.rain_mm > cap), "impossible", "Rainfall impossible",
                               lambda s, a, b, n: f"{n} reading(s) ({s.min():.1f}..{s.max():.1f} mm per interval) outside 0..{cap:.0f} mm.",
                               merge_gap=MERGE_GAP)
    # sunlight
    lat, lon = st.lat_for_solar, st.lon_for_solar
    if not (np.isnan(lat) or np.isnan(lon)):
        el = pd.Series(solar_elevation_deg(c.index, lat, lon), index=c.index)
        g = c.ghi_wm2
        out += _runs_to_candidates(st, "ghi_wm2", g < 0, "impossible", "Sunlight negative",
                                   lambda s, a, b, n: f"{n} negative irradiance reading(s).", merge_gap=MERGE_GAP)
        out += _runs_to_candidates(st, "ghi_wm2", (el < NIGHT_EL_DEG) & (g > NIGHT_GHI_MAX), "night", "Sunlight at night",
                                   lambda s, a, b, n: f"{n} reading(s) up to {s.max():.0f} W/m2 while the sun is more than 6 deg below the horizon.",
                                   merge_gap=MERGE_GAP, min_len=2)
        etr = pd.Series(etr_horizontal(el.values), index=c.index)
        out += _runs_to_candidates(st, "ghi_wm2", (el > 15) & (g > etr + ETR_MARGIN), "impossible", "Sunlight above physical ceiling",
                                   lambda s, a, b, n: f"{n} reading(s) up to {s.max():.0f} W/m2 exceed the extra-terrestrial irradiance for the sun's height.",
                                   merge_gap=MERGE_GAP, min_len=6)
        out += _runs_to_candidates(st, "ghi_wm2", (el > DAY_ZERO_EL_DEG) & (g == 0), "dead", "Sunlight reads zero in daylight",
                                   lambda s, a, b, n: f"{n} reading(s) of exactly 0 W/m2 with the sun above 15 deg; a working pyranometer reads tens of W/m2 even under thick cloud.",
                                   merge_gap=MERGE_GAP, min_len=DAY_ZERO_MIN_SAMPLES)
    return out


def check_missing(st: Station) -> list[Candidate]:
    """Rows that exist but have an empty or non-numeric cell in a channel."""
    out: list[Candidate] = []
    if st.present is None:
        return out
    for ch in st.clean.columns:
        if st.raw[ch].notna().sum() == 0:
            continue          # channel absent from the file: reported once as 'channel not in file'
        miss = st.present & st.clean[ch].isna()
        for a, b in runs_of(miss.reset_index(drop=True)):
            idx = st.clean.index
            n = b - a + 1
            out.append(Candidate(st.sid, ch, idx[a], idx[b], "missing", "other", f"Missing values in {ch}", "exclude",
                                 f"{n} logger row(s) have an empty or non-numeric {ch} cell; the other channels of those rows are kept.",
                                 evidence={"rows": n}))
    return out
