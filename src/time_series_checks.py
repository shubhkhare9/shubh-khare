"""Single-station time-series checks: isolated spikes and frozen readings.

Every candidate is cross-checked with the other stations before it is called a fault. A change that
neighbours also see is weather (keep); one that only this sensor sees is the sensor (exclude).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Candidate
from .peer_frames import Peers
from .solar import solar_elevation_deg
from .station import Station
from .stats_utils import hampel_deviation, merge_runs, runs_of

# ---- spike rule: |value - median of its 7-sample neighbourhood| > max(abs_floor, K * local robust scale)
# abs_floor is the smallest jump that natural 10-minute weather essentially never makes at that channel
# (in this pack the biggest genuine 10-minute temperature change is ~3 C, pressure ~1 hPa, humidity ~25 %,
# wind ~28 km/h at the 2900 m ridge); K * scale adapts to noisy sites.
SPIKE_RULES = {
    "temp_c": dict(abs_floor=5.0, min_scale=0.3),
    "rh_pct": dict(abs_floor=30.0, min_scale=2.0),
    "pressure_hpa": dict(abs_floor=3.0, min_scale=0.15),
    "wind_kmh": dict(abs_floor=35.0, min_scale=2.0),
}
SPIKE_K = 8.0
# Values that data loggers/sensors emit when they fail rather than measure. Sensor power-on default for the
# common DS18B20 is 85 C; -40 C is the "no data" marker of several humidity/temperature chips.
SENTINELS = {
    "temp_c": {85.0, -40.0, -127.0, 127.0, -999.0, 999.0, -9999.0, 9999.0, 125.0, 255.0},
    "rh_pct": {-999.0, 999.0, -9999.0, 9999.0, 255.0},
    "pressure_hpa": {-999.0, 999.0, -9999.0, 9999.0, 0.0},
    "wind_kmh": {-999.0, 999.0, -9999.0, 9999.0, 255.0},
}

# ---- frozen readings: this many identical consecutive values (at the channel's resolution) is a stuck sensor.
# Longest natural run of identical values in the supplied healthy data: temperature 6, humidity 8 (unsaturated),
# pressure 9, wind direction 4. Thresholds are about twice that.
FLAT_MIN = {"temp_c": 12, "rh_pct": 12, "pressure_hpa": 24, "wind_dir_deg": 12, "wind_kmh": 12, "ghi_wm2": 12}
# A frozen reading is only a fault if the neighbours kept moving over the same period.
PEER_MOVE_MIN = {"temp_c": 2.0, "rh_pct": 8.0, "pressure_hpa": 1.5, "wind_dir_deg": 30.0, "wind_kmh": 4.0, "ghi_wm2": 100.0}
# Calm (exact 0 km/h) for this many consecutive samples while neighbours blow is a seized anemometer.
CALM_STUCK_MIN, CALM_PEER_KMH = 18, 10.0
# RH plateau at saturation counts as real if neighbours are also this humid for most of the plateau.
SATURATION_PEER_RH, SATURATION_PEER_FRAC = 95.0, 0.7


def _label(ts):
    return pd.Timestamp(ts).strftime("%d %b %H:%M")


def find_spikes(st: Station, peers: Peers | None) -> list[Candidate]:
    out: list[Candidate] = []
    for ch, rule in SPIKE_RULES.items():
        x = st.clean[ch]
        if x.notna().sum() < 20:
            continue
        dev, scale = hampel_deviation(x, 3, rule["min_scale"])
        thr = np.maximum(rule["abs_floor"], SPIKE_K * scale)
        spike = (dev.abs() > thr) & x.notna()
        for a, b in merge_runs(runs_of(spike.reset_index(drop=True)), 0):
            idx = st.clean.index
            seg = x.iloc[a:b + 1]
            d = float(dev.iloc[a:b + 1].iloc[np.argmax(np.abs(dev.iloc[a:b + 1].values))])
            around = x.iloc[max(0, a - 3):b + 4].drop(index=seg.index, errors="ignore").median()
            sentinel = all(v in SENTINELS.get(ch, set()) for v in seg.values)
            # --- do neighbours see the same jump at the same time?
            support, n_peers = 0, 0
            if peers is not None and peers.enough:
                for pid in peers.stations:
                    if pid == st.sid:
                        continue
                    px = peers.frames[ch][pid]
                    if px.iloc[max(0, a - 3):b + 4].notna().sum() < 4:
                        continue
                    n_peers += 1
                    pdv, _ = hampel_deviation(px.iloc[max(0, a - 6):b + 7], 3, rule["min_scale"])
                    near = pdv.reindex(idx[max(0, a - 1):b + 2])
                    if (np.sign(near) == np.sign(d)).any() and near.abs().max() >= 0.5 * abs(d):
                        support += 1
            real = n_peers >= 3 and support >= max(2, int(np.ceil(0.3 * n_peers)))
            if real:
                out.append(Candidate(st.sid, ch, idx[a], idx[b], "spike", "spike", f"Sharp {ch} change shared by neighbours", "keep",
                                     f"{ch} jumps {d:+.1f} from about {around:.1f} but {support} of {n_peers} other stations show the same move at the same time, so it is weather, not the sensor.",
                                     evidence={"dev": d, "support": support, "peers": n_peers}))
            elif sentinel:
                v = seg.iloc[0]
                out.append(Candidate(st.sid, ch, idx[a], idx[b], "sentinel", "spike", f"Sensor error code ({v:g}) in {ch}", "exclude",
                                     f"Single {ch} reading of {v:g} between neighbours near {around:.1f} (jump {d:+.1f}); {v:g} is a known sensor/logger fault value and no other station moved ({support} of {n_peers}).",
                                     evidence={"dev": d, "support": support, "peers": n_peers, "value": float(v)}))
            else:
                out.append(Candidate(st.sid, ch, idx[a], idx[b], "spike", "spike", f"Isolated spike in {ch}", "exclude",
                                     f"{ch} jumps {d:+.1f} (to {seg.iloc[np.argmax(np.abs(seg.values - around))]:.1f}) from about {around:.1f} and straight back; only {support} of {n_peers} other stations move, so it is not weather.",
                                     evidence={"dev": d, "support": support, "peers": n_peers}))
    return out


def find_flatlines(st: Station, peers: Peers | None) -> list[Candidate]:
    out: list[Candidate] = []
    idx = st.clean.index
    lat, lon = st.lat_for_solar, st.lon_for_solar
    el = pd.Series(solar_elevation_deg(idx, lat, lon), index=idx) if not (np.isnan(lat) or np.isnan(lon)) else None
    for ch, nmin in FLAT_MIN.items():
        x = st.clean[ch]
        same = (x == x.shift(1)) & x.notna()
        # runs of equal consecutive values: the run covers the sample before the first repeat as well
        runs = runs_of(same.reset_index(drop=True))
        for a, b in runs:
            first = a - 1                       # the first sample of the frozen value is a genuine reading
            length = b - first + 1
            if length < nmin:
                continue
            v = float(x.iloc[b])
            if ch == "wind_kmh" and v == 0:
                continue                         # calm handled below
            if ch == "ghi_wm2" and (el is None or v == 0 or not (el.iloc[first:b + 1] > 10).all()):
                continue
            if ch == "wind_dir_deg":
                wspd = st.clean["wind_kmh"].iloc[first:b + 1]
                if (wspd < 2).mean() > 0.5:
                    continue                     # calm air: a vane legitimately hangs in one position
            s_pos = a                              # flag from the first *repeated* reading
            span = slice(first, b + 1)
            ts0, ts1 = idx[s_pos], idx[b]
            moved, status = np.nan, ""
            if peers is not None and peers.enough:
                ref = peers.peer_median_raw(st.sid, ch).iloc[span].dropna()
                if len(ref) >= 3:
                    if ch == "wind_dir_deg":
                        ang = np.radians(ref.values)
                        moved = float(np.degrees(np.arctan2(np.sin(ang).std(), np.cos(ang).mean()))) if len(ref) > 1 else 0.0
                        moved = float(ref.max() - ref.min()) if (ref.max() - ref.min()) < 180 else 180.0
                    else:
                        moved = float(ref.max() - ref.min())
            peers_move = (not np.isnan(moved)) and moved >= PEER_MOVE_MIN[ch]
            # saturated humidity: neighbours also at ~100% means it really was foggy / raining
            if ch == "rh_pct" and v >= 99.5 and peers is not None and peers.enough:
                prm = peers.peer_median_raw(st.sid, ch).iloc[span]
                if (prm >= SATURATION_PEER_RH).mean() >= SATURATION_PEER_FRAC:
                    out.append(Candidate(st.sid, ch, ts0, ts1, "saturation", "stuck", "Humidity pinned at 100% but neighbours agree", "keep",
                                         f"rh_pct sits at {v:.0f}% for {length} readings ({length * st.interval / 60:.1f} h), and the median of the other stations is at or above {SATURATION_PEER_RH:.0f}% for {(prm >= SATURATION_PEER_RH).mean() * 100:.0f}% of that time, so this is real fog/rain saturation.",
                                         evidence={"rows": length, "value": v}))
                    continue
            if ch == "rh_pct" and v > RH_CAP:
                # impossible values are described by the physical check; the frozen run adds the evidence
                pass
            if peers is not None and peers.enough and not peers_move and not np.isnan(moved):
                continue                         # nothing else moved either: not evidence of a fault
            mv = f"; the other stations' median moved {moved:.1f} over the same time" if not np.isnan(moved) else ""
            out.append(Candidate(st.sid, ch, ts0, ts1, "flatline", "stuck", f"{ch} frozen at {v:g}", "exclude",
                                 f"{ch} repeats exactly {v:g} for {length} consecutive readings ({length * st.interval / 60:.1f} h){mv}; that is far beyond the {nmin}-reading limit for this channel.",
                                 evidence={"rows": length, "value": v, "peer_range": moved}))

    # exact-zero wind while neighbours blow
    if peers is not None and peers.enough:
        w = st.clean["wind_kmh"]
        zero = (w == 0) & w.notna()
        for a, b in runs_of(zero.reset_index(drop=True)):
            if b - a + 1 < CALM_STUCK_MIN:
                continue
            ref = peers.peer_median_raw(st.sid, "wind_kmh").iloc[a:b + 1]
            if (ref >= CALM_PEER_KMH).mean() >= 0.6:
                out.append(Candidate(st.sid, "wind_kmh", idx[a], idx[b], "flatline", "stuck", "Wind speed stuck at 0", "exclude",
                                     f"wind_kmh reads exactly 0 for {b - a + 1} readings while the other stations' median is at least {CALM_PEER_KMH:.0f} km/h for {(ref >= CALM_PEER_KMH).mean() * 100:.0f}% of that time.",
                                     evidence={"rows": b - a + 1}))
    return out


RH_CAP = 100.5


# ---- odd-looking but genuine: the strongest winds, when the network agrees
# 45 km/h (a strong breeze to near-gale 10-minute mean) is unusual enough to catch the eye of anyone reading the data.
EXTREME_WIND_KMH = 45.0
EXTREME_PEER_PCT = 90          # neighbours must be in their own top 10% of wind speed...
EXTREME_WINDOW_H = 3           # ...within this many hours


def find_extreme_wind(st: Station, peers: Peers | None) -> list[Candidate]:
    """One 'keep' flag per station for its strongest wind episode if it is large and real."""
    w = st.clean["wind_kmh"]
    if w.notna().sum() < 100 or peers is None or not peers.enough:
        return []
    top = w.max()
    if not top >= EXTREME_WIND_KMH or top < w.quantile(0.999):
        return []
    t = w.idxmax()
    win = slice(t - pd.Timedelta(hours=EXTREME_WINDOW_H), t + pd.Timedelta(hours=EXTREME_WINDOW_H))
    agree, n = 0, 0
    for pid in peers.stations:
        if pid == st.sid:
            continue
        px = peers.frames["wind_kmh"][pid]
        if px.notna().sum() < 100:
            continue
        n += 1
        if px.loc[win].max() >= np.nanpercentile(px.values, EXTREME_PEER_PCT):
            agree += 1
    if n < 3:
        return []
    # an episode = the samples within 80% of the peak around it
    near = w.loc[win]
    ep = near[near >= 0.8 * top]
    if agree >= max(2, int(np.ceil(0.4 * n))):
        txt = (f"Wind peaks at {top:.0f} km/h; {agree} of {n} other stations are also in their windiest 10% within {EXTREME_WINDOW_H} h of it, "
               "so this is a real storm, not a sensor fault.")
    else:
        txt = (f"Wind peaks at {top:.0f} km/h while only {agree} of {n} other stations are in their windiest 10% at that time; a local gust is physically "
               "plausible at an exposed site and nothing in the record points to the sensor, so it is kept (low confidence).")
    return [Candidate(st.sid, "wind_kmh", ep.index[0], ep.index[-1], "extreme", "other", f"Strong wind ({top:.0f} km/h)", "keep", txt,
                      evidence={"peak": float(top), "agree": agree, "peers": n})]
