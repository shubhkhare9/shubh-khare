"""Small solar-geometry helpers (NOAA low-precision formulae, good to ~0.1 deg / ~1 min).

Used to ask "is this sunlight reading plausible for this place and this clock time?".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1361.0  # W/m2


def _gamma(ts: pd.DatetimeIndex) -> np.ndarray:
    doy = ts.dayofyear.values + (ts.hour.values + ts.minute.values / 60.0) / 24.0
    return 2 * np.pi / 365.0 * (doy - 1)


def equation_of_time_min(ts) -> np.ndarray:
    g = _gamma(pd.DatetimeIndex(ts))
    return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                     - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))


def declination_rad(ts) -> np.ndarray:
    g = _gamma(pd.DatetimeIndex(ts))
    return (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))


def solar_elevation_deg(ts, lat: float, lon: float) -> np.ndarray:
    """Solar elevation above the horizon (degrees) for UTC timestamps at lat/lon (east positive)."""
    ts = pd.DatetimeIndex(ts)
    minutes = ts.hour.values * 60 + ts.minute.values + ts.second.values / 60.0
    true_solar_min = minutes + equation_of_time_min(ts) + 4.0 * lon
    ha = np.radians(true_solar_min / 4.0 - 180.0)
    la, dec = np.radians(lat), declination_rad(ts)
    sin_el = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))


def etr_horizontal(el_deg: np.ndarray) -> np.ndarray:
    """Extra-terrestrial irradiance on a horizontal plane: an absolute ceiling for GHI."""
    return SOLAR_CONSTANT * np.clip(np.sin(np.radians(el_deg)), 0, None)


def clear_sky_ghi(el_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky GHI: a simple, widely used reference curve."""
    el = np.asarray(el_deg, dtype=float)
    cosz = np.sin(np.radians(np.clip(el, 0.5, 90)))
    out = 1098.0 * cosz * np.exp(-0.057 / cosz)
    return np.where(el > 0.5, out, 0.0)


def solar_noon_utc_hours(date: pd.Timestamp, lon: float) -> float:
    """UTC hour of the sun's highest point on a date (equation of time evaluated at noon)."""
    noon = pd.DatetimeIndex([pd.Timestamp(date).normalize() + pd.Timedelta(hours=12)])
    eot = float(equation_of_time_min(noon)[0])
    return 12.0 - lon / 15.0 - eot / 60.0


def phase_offset_hours(ts: pd.DatetimeIndex, ghi: pd.Series, lat: float, lon: float, min_day_samples: int = 24):
    """Median over days of (observed - modelled) centroid time of the daylight curve, in hours.

    ~0 for a correct clock; about -6 if the logger clock runs six hours behind UTC. Cloud makes a
    single day's centroid wander by tens of minutes, so callers should use the median over days.
    Returns (median_offset_hours, n_days).
    """
    ts = pd.DatetimeIndex(ts)
    el = solar_elevation_deg(ts, lat, lon)
    model = clear_sky_ghi(el)
    df = pd.DataFrame({"obs": np.asarray(ghi, dtype=float), "mod": model,
                       "h": ts.hour + ts.minute / 60.0, "d": ts.normalize()}, index=ts)
    # A logger running *behind* shifts the curve to the previous/next UTC day, so use a
    # continuous time axis (hours since the first midnight) rather than hour-of-day.
    t0 = ts.min().normalize()
    df["t"] = (ts - t0).total_seconds() / 3600.0
    out = []
    # Group by "model solar day": cut the axis at the modelled night midpoint (solar midnight).
    solar_midnight_shift = (12.0 - lon / 15.0)  # UTC hour of solar noon (approx.); midnight is +-12 h
    cut = (df["t"] - (solar_midnight_shift - 12.0)) // 24.0
    for _, g in df.groupby(cut):
        g = g.dropna(subset=["obs"])
        if len(g) < min_day_samples or g["mod"].sum() <= 0:
            continue
        w_obs = np.clip(g["obs"].values, 0, None)
        if w_obs.sum() < 1e-6:
            continue
        # observed daylight centroid relative to modelled centroid, restricted to the model's daylight window
        c_obs = float((g["t"] * w_obs).sum() / w_obs.sum())
        c_mod = float((g["t"] * g["mod"]).sum() / g["mod"].sum())
        out.append(c_obs - c_mod)
    if not out:
        return np.nan, 0
    return float(np.median(out)), len(out)
