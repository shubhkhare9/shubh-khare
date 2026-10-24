"""Synthetic data-pack generator for tests (the real pack must not be committed).

Builds a pack in exactly the same format as the supplied one: shared weather + per-station offsets,
NMEA files, stations.csv and a maintenance log. ``inject_*`` helpers then plant known faults so tests
can check that the checker finds them on data it was never tuned on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import nmea_checksum
from src.gps_checks import std_atm_pressure
from src.solar import clear_sky_ghi, solar_elevation_deg

CHANNELS = ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "wind_dir_deg", "rain_mm", "ghi_wm2"]


def _ar1(rng, n, phi, sigma):
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def make_stations(n=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(dict(station_id=f"T{i + 1:02d}", name=f"Test {i + 1}", lat=round(39.5 + 0.12 * i + rng.uniform(-0.03, 0.03), 4),
                         lon=round(-105.4 + 0.1 * i + rng.uniform(-0.03, 0.03), 4), elevation_m=int(rng.uniform(1400, 2400)),
                         installed="2023-01-01", logger_interval_min=10))
    return pd.DataFrame(rows)


def make_weather(stations: pd.DataFrame, start="2025-05-01", days=21, seed=0):
    rng = np.random.default_rng(seed + 1)
    idx = pd.date_range(start, periods=days * 144, freq="10min")
    n = len(idx)
    syn_p = _ar1(rng, n, 0.9995, 0.12)               # slow shared pressure anomaly (hPa)
    syn_t = _ar1(rng, n, 0.999, 0.12)                # shared temperature anomaly (C)
    cloud = np.clip(0.75 + _ar1(rng, n, 0.995, 0.06), 0.15, 1.0)
    storm = np.clip(_ar1(rng, n, 0.998, 0.35), 0, None)
    wind_shared = np.clip(6 + 6 * storm + _ar1(rng, n, 0.98, 0.5), 0.2, None)
    rain_rate = np.clip(storm - 0.9, 0, None)
    humid = np.clip(45 + 10 * storm - 3 * syn_t + _ar1(rng, n, 0.99, 1.0), 5, 105)
    # three shared rain events (so a blocked gauge has something to miss): ~8 mm each over 6 hours
    forced = np.zeros(n)
    for d in (3, 9, 14):
        i0 = d * 144 + 36
        forced[i0:i0 + 36] = rng.choice([0.0, 0.2, 0.4, 0.6], size=36, p=[0.3, 0.3, 0.3, 0.1])
    out = {}
    for _, s in stations.iterrows():
        el = solar_elevation_deg(idx, s.lat, s.lon)
        local_h = np.asarray((idx.hour + idx.minute / 60 + s.lon / 15) % 24)
        temp = (6 - 6.5 * (s.elevation_m - 1500) / 1000 + 7 * np.sin((local_h - 9) / 24 * 2 * np.pi) * (cloud ** 0.5)
                + 2 * syn_t + _ar1(rng, n, 0.9, 0.08))
        pres = float(std_atm_pressure(s.elevation_m)) + syn_p * float(std_atm_pressure(s.elevation_m)) / 1013 * 2 - 0.3 + rng.normal(0, 0.08, n)
        rh = np.clip(humid + rng.normal(0, 1.5, n) + 0.4 * (temp.mean() - temp), 3, 100)
        wind = np.clip(wind_shared * rng.uniform(0.8, 1.2) + rng.normal(0, 1.0, n), 0, None)
        wdir = (180 + np.cumsum(rng.normal(0, 6, n))) % 360
        rain = np.round(np.clip(forced + rng.choice([0.0, 0.2], size=n, p=[0.8, 0.2]) * (forced > 0), 0, None) / 0.2) * 0.2
        ghi = clear_sky_ghi(el) * cloud * (1 + rng.normal(0, 0.03, n))
        df = pd.DataFrame({"timestamp_utc": idx.strftime("%Y-%m-%d %H:%M"), "temp_c": np.round(temp, 1), "rh_pct": np.round(rh),
                           "pressure_hpa": np.round(pres, 1), "wind_kmh": np.round(wind, 1), "wind_dir_deg": np.round(wdir).astype(int),
                           "rain_mm": np.round(rain, 1), "ghi_wm2": np.round(np.clip(ghi, 0, None)).astype(int)})
        out[s.station_id] = df
    return out


def make_gps(stations: pd.DataFrame, idx_start, days, seed=0):
    rng = np.random.default_rng(seed + 2)
    out = {}
    hours = pd.date_range(idx_start, periods=days * 24, freq="1h")
    for _, s in stations.iterrows():
        lines = []
        for t in hours:
            lat = s.lat + rng.normal(0, 0.00005)
            lon = s.lon + rng.normal(0, 0.00005)
            la = f"{int(abs(lat)):02d}{(abs(lat) % 1) * 60:07.4f}"
            lo = f"{int(abs(lon)):03d}{(abs(lon) % 1) * 60:07.4f}"
            body = f"GPRMC,{t:%H%M%S}.00,A,{la},{'N' if lat >= 0 else 'S'},{lo},{'E' if lon >= 0 else 'W'},0.0,0.0,{t:%d%m%y},,,A"
            lines.append(f"{t:%Y-%m-%d %H:%M:%S},${body}*{nmea_checksum(body)}")
            if t.hour % 6 == 0:
                gb = f"GPGGA,{t:%H%M%S}.00,{la},{'N' if lat >= 0 else 'S'},{lo},{'E' if lon >= 0 else 'W'},1,9,1.4,{s.elevation_m + rng.normal(0, 3):.1f},M,-17.0,M,,"
                lines.append(f"{t:%Y-%m-%d %H:%M:%S},${gb}*{nmea_checksum(gb)}")
        out[s.station_id] = lines
    return out


def write_pack(root: Path, stations, weather, gps, log_lines=()):
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "gps").mkdir(exist_ok=True)
    stations.to_csv(root / "stations.csv", index=False)
    for sid, df in weather.items():
        df.to_csv(root / "data" / f"{sid}.csv", index=False)
        (root / "gps" / f"{sid}.nmea").write_text("\n".join(gps[sid]) + "\n")
    (root / "maintenance_log.md").write_text("# log\n\n" + "\n".join(log_lines) + "\n")
    return root


# ------------------------------------------------------------------ fault injectors (return description)
def at(df, ts):
    return int(np.where(df.timestamp_utc.values == pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"))[0][0])


def inject_sentinel(df, ts, value=85.0):
    df.loc[at(df, ts), "temp_c"] = value


def inject_flatline(df, ts, n, channel="temp_c"):
    i = at(df, ts)
    df.loc[i:i + n - 1, channel] = df.loc[i, channel]


def inject_pressure_step(df, ts, offset):
    i = at(df, ts)
    df.loc[i:, "pressure_hpa"] = np.round(df.loc[i:, "pressure_hpa"] + offset, 1)


def inject_rh_stuck(df, ts, n):
    i = at(df, ts)
    df.loc[i:i + n - 1, "rh_pct"] = 101 + np.arange(n) // 20


def inject_wind_gain(df, ts, factor):
    i = at(df, ts)
    df.loc[i:, "wind_kmh"] = np.round(df.loc[i:, "wind_kmh"] * factor, 1)


def inject_rain_blocked(df, ts, n):
    i = at(df, ts)
    df.loc[i:i + n - 1, "rain_mm"] = 0.0


def inject_duplicate_block(df, ts, n):
    i = at(df, ts)
    return pd.concat([df, df.iloc[i:i + n]], ignore_index=True)


def inject_gap(df, ts, n):
    i = at(df, ts)
    return df.drop(index=range(i, i + n)).reset_index(drop=True)


def inject_clock_shift(df, gps_lines, ts, hours):
    """Rows from ``ts`` on are stamped ``hours`` earlier (a logger that restarted with the wrong clock)."""
    i = at(df, ts)
    t = pd.to_datetime(df.timestamp_utc)
    t.iloc[i:] = t.iloc[i:] - pd.Timedelta(hours=hours)
    df = df.copy()
    df["timestamp_utc"] = t.dt.strftime("%Y-%m-%d %H:%M")
    cut = pd.Timestamp(ts)
    new = []
    for L in gps_lines:
        lts, rest = L.split(",", 1)
        lt = pd.Timestamp(lts)
        new.append(f"{(lt - pd.Timedelta(hours=hours)):%Y-%m-%d %H:%M:%S},{rest}" if lt >= cut else L)
    return df, new


def inject_gps_position(gps_lines, dlat):
    out = []
    for L in gps_lines:
        lts, sent = L.split(",", 1)
        body, cs = sent[1:].split("*")
        f = body.split(",")
        if f[0] == "GPRMC":
            deg = int(float(f[3]) // 100); mins = float(f[3]) - deg * 100
            lat = deg + mins / 60 + dlat
            f[3] = f"{int(lat):02d}{(lat % 1) * 60:07.4f}"
        elif f[0] == "GPGGA":
            deg = int(float(f[2]) // 100); mins = float(f[2]) - deg * 100
            lat = deg + mins / 60 + dlat
            f[2] = f"{int(lat):02d}{(lat % 1) * 60:07.4f}"
        body = ",".join(f)
        out.append(f"{lts},${body}*{nmea_checksum(body)}")
    return out
