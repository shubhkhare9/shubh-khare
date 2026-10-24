"""Checks that use the raw GPS sentences: position vs install sheet, altitude, receiver health,
and any clock offset that the timeline stage did not already explain."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Flag
from .station import Station
from .stats_utils import runs_of

# Install-sheet coordinates carry 4 decimals (~11 m) and single GPS fixes here scatter by ~15 m, so a
# median distance above 250 m means the sheet (or the station's location) is wrong, not noise.
POSITION_TOLERANCE_M = 250
# A tight GPS cluster: 90% of valid fixes within this radius of their own median.
CLUSTER_RADIUS_M = 100
# GGA altitude is noisier (several tens of metres); only a >100 m disagreement is treated as real.
ALTITUDE_TOLERANCE_M = 100
# GPS receivers lose fix for an hour or two now and then (a few % of sentences here). Twelve hours
# without a single valid fix is an antenna/receiver problem.
GPS_BLACKOUT_H = 12


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _transposed(a: float, b: float) -> bool:
    sa, sb = f"{abs(a):.2f}".replace(".", ""), f"{abs(b):.2f}".replace(".", "")
    return sa != sb and sorted(sa) == sorted(sb)


def std_atm_pressure(h_m):
    return 1013.25 * (1 - 2.25577e-5 * np.asarray(h_m, dtype=float)) ** 5.25588


def std_atm_height(p_hpa):
    return (1 - (np.asarray(p_hpa, dtype=float) / 1013.25) ** (1 / 5.25588)) / 2.25577e-5


def check_position(st: Station):
    g = st.gps
    if g is None or g.empty:
        return
    fx = g[g.kind.isin(["RMC", "GGA"]) & g.valid & g.cks_ok & g.lat.notna() & g.lon.notna()]
    if len(fx) < 10:
        return
    lat_g, lon_g = float(fx.lat.median()), float(fx.lon.median())
    st.notes["gps_lat"], st.notes["gps_lon"] = lat_g, lon_g
    if np.isnan(st.lat) or np.isnan(st.lon):
        st.solar_lat, st.solar_lon = lat_g, lon_g
        st.add(Flag(st.sid, "metadata", fx.lts.min(), fx.lts.max(), "Station missing from stations.csv", "keep",
                    f"No install-sheet row; GPS median position {lat_g:.4f}, {lon_g:.4f} is used for sun-position checks.",
                    kind="no_sheet"))
        return
    d = haversine_m(st.lat, st.lon, fx.lat.values, fx.lon.values)
    d_med = float(np.median(d))
    if d_med <= POSITION_TOLERANCE_M:
        return
    own = haversine_m(lat_g, lon_g, fx.lat.values, fx.lon.values)
    tight = float((own <= CLUSTER_RADIUS_M).mean())
    alts = g[(g.kind == "GGA") & g.valid & g.cks_ok & g.alt.notna()].alt
    alt_med = float(alts.median()) if len(alts) >= 5 else np.nan
    alt_ok = (not np.isnan(alt_med)) and not np.isnan(st.elev) and abs(alt_med - st.elev) <= ALTITUDE_TOLERANCE_M
    dlat_m = (lat_g - st.lat) * 111_195.0
    dlon_m = (lon_g - st.lon) * 111_195.0 * np.cos(np.radians(lat_g))
    which = []
    if abs(dlat_m) > POSITION_TOLERANCE_M:
        which.append(f"latitude ({st.lat:.4f} on the sheet vs {lat_g:.4f} from GPS, {abs(dlat_m) / 1000:.1f} km {'north' if dlat_m > 0 else 'south'})")
    if abs(dlon_m) > POSITION_TOLERANCE_M:
        which.append(f"longitude ({st.lon:.4f} vs {lon_g:.4f}, {abs(dlon_m) / 1000:.1f} km {'east' if dlon_m > 0 else 'west'})")
    other = "longitude" if abs(dlat_m) > abs(dlon_m) else "latitude"
    hint = ""
    if abs(dlat_m) > POSITION_TOLERANCE_M and _transposed(st.lat, lat_g):
        hint = " The two latitudes use the same digits in a different order, which looks like a typing slip on the sheet."
    if abs(dlon_m) > POSITION_TOLERANCE_M and _transposed(st.lon, lon_g):
        hint = " The two longitudes use the same digits in a different order, which looks like a typing slip on the sheet."
    base = (f"{len(fx)} valid GPS fixes sit {d_med / 1000:.1f} km from the install-sheet position ({' and '.join(which)}); "
            f"{tight * 100:.0f}% of fixes are within {CLUSTER_RADIUS_M} m of their own median, "
            + (f"and GPS altitude {alt_med:.0f} m matches the sheet's {st.elev:.0f} m." if alt_ok else
               ("but GPS altitude does not match the sheet elevation." if not np.isnan(alt_med) else "")))
    if tight >= 0.9 and alt_ok and len(fx) >= 20:
        st.solar_lat, st.solar_lon = lat_g, lon_g
        st.add(Flag(st.sid, "metadata", fx.lts.min(), fx.lts.max(), "Install-sheet position disagrees with GPS", "correct",
                    f"{base}{hint} Correct the sheet to {lat_g:.4f}, {lon_g:.4f}; weather readings are unaffected.",
                    kind="gps_position", evidence={"dist_km": d_med / 1000, "lat_gps": lat_g, "lon_gps": lon_g}))
    else:
        st.add(Flag(st.sid, "metadata", fx.lts.min(), fx.lts.max(), "Install-sheet position disagrees with GPS", "exclude",
                    f"{base} Neither position can be trusted; do not use this station in spatial comparisons until checked.",
                    kind="gps_position", evidence={"dist_km": d_med / 1000}))


def check_elevation(st: Station):
    g = st.gps
    if g is None or g.empty or np.isnan(st.elev):
        return
    alts = g[(g.kind == "GGA") & g.valid & g.cks_ok & g.alt.notna()].alt
    if len(alts) < 10:
        return
    alt = float(alts.median())
    if abs(alt - st.elev) <= ALTITUDE_TOLERANCE_M:
        return
    pmed = st.clean["pressure_hpa"].median() if st.clean is not None else np.nan
    p_h = float(std_atm_height(pmed)) if not np.isnan(pmed) else np.nan
    st.add(Flag(st.sid, "metadata", g.lts.min(), g.lts.max(), "Elevation on sheet disagrees with GPS", "exclude",
                f"GPS altitude is {alt:.0f} m but the sheet says {st.elev:.0f} m"
                + (f"; the station's median pressure implies about {p_h:.0f} m." if not np.isnan(p_h) else ".")
                + " Elevation is used to judge pressure, so it needs checking.", kind="gps_elevation"))


def check_gps_health(st: Station):
    """Long stretches with no valid fix at all."""
    g = st.gps
    if g is None or g.empty:
        return
    ok = g[(g.kind == "RMC") & g.valid & g.cks_ok].lts.sort_values()
    if len(ok) < 5:
        return
    gaps = ok.diff().dt.total_seconds() / 3600.0
    logger_gaps = [(f.start, f.end) for f in st.flags if f.kind == "gap"]
    for i in np.where(gaps >= GPS_BLACKOUT_H)[0]:
        a, b = ok.iloc[i - 1], ok.iloc[i]
        if any(gs <= b and ge >= a for gs, ge in logger_gaps):
            continue        # the logger itself was off; the gap flag already says so
        st.add(Flag(st.sid, "metadata", a, b, "GPS receiver silent or invalid", "keep",
                    f"No valid GPS fix for {gaps.iloc[i]:.0f} h; weather values are not affected, but the clock and position cannot be verified for this stretch.",
                    kind="gps_blackout"))


def check_clock_offsets(st: Station, gpsoff: pd.DataFrame):
    """Clock offsets in GPS that no earlier stage explained (no backward step in the file)."""
    if gpsoff is None or gpsoff.empty:
        return
    explained = []
    for f in st.flags:
        if f.kind in ("clock_unknown", "clock_collision", "clock_constant"):
            explained.append((f.start, f.end))
        elif f.kind == "clock_shift":       # flag is in corrected time; GPS fixes carry the logger's stamps
            sh = pd.Timedelta(seconds=f.evidence["shift_s"])
            explained.append((f.start - sh, f.end - sh))
    off = gpsoff.copy()
    off["bad"] = off.off_s.abs() > 30
    for s, e in explained:
        off.loc[(off.lts >= s - pd.Timedelta(hours=1)) & (off.lts <= e + pd.Timedelta(hours=1)), "bad"] = False
    for a, b in runs_of(off["bad"].reset_index(drop=True)):
        if b - a + 1 < 3:
            continue
        seg = off.iloc[a:b + 1]
        prev_t = off.lts.iloc[a - 1] if a > 0 else seg.lts.iloc[0]
        next_t = off.lts.iloc[b + 1] if b + 1 < len(off) else seg.lts.iloc[-1]
        med = float(seg.off_s.median())
        st.add(Flag(st.sid, "all", prev_t, next_t, "Logger clock disagrees with GPS", "exclude",
                    f"{len(seg)} consecutive GPS fixes read {med / 60:+.1f} min from the logger clock (healthy fixes agree within 2 s), "
                    "so timestamps here cannot be trusted; rows are excluded rather than re-timed because the file has no step to pin down the change.",
                    kind="clock_offset", evidence={"offset_s": med}))
        st.exclude("all", st.window_mask(prev_t, next_t))
