"""Checks that compare a station with its neighbours: level steps, daytime drift, gain errors, blocked rain gauge.

Only the sensor's *difference from the network* is used, so real weather (which all stations share)
cancels out. Thresholds are chosen so that nothing in the healthy part of the supplied pack trips them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gps_checks import std_atm_pressure
from .models import Candidate
from .peer_frames import MIN_PEERS, Peers
from .solar import solar_elevation_deg
from .station import Station
from .stats_utils import best_split, hinge_fit, mad_sigma, segment_levels

# ---- abrupt level shifts in (station - neighbours), hourly medians
#   min_delta: smallest shift worth reporting; noise_mult: shift must also exceed this many robust sigmas
LEVEL_RULES = {
    "pressure_hpa": dict(min_delta=3.0, noise_mult=4.0, unit="hPa", correctable=True),
    "rh_pct": dict(min_delta=15.0, noise_mult=4.0, unit="%", correctable=False),
    "temp_c": dict(min_delta=4.0, noise_mult=4.0, unit="C", correctable=False),
}
MIN_SEGMENT_HOURS = 12            # a level must last half a day to count as a shift (not weather)
ABRUPT_FRACTION = 0.6             # >=60% of the shift must happen within one logging interval
LOCAL_STEP_TOLERANCE = 2.0        # hPa: local step and whole-segment shift must agree this well to correct

# ---- daytime-only temperature drift (radiation-shield / siting problems act when the sun is up)
MIDDAY_EL, NIGHT_EL = 30.0, -6.0  # sun elevation (deg) defining midday and night samples
DRIFT_MIN_RISE_C = 3.0            # total ramp (deg C) worth flagging
DRIFT_MIN_GAIN = 0.4              # share of variance the ramp must explain (healthy stations stay below ~0.2)
DRIFT_MIN_DAYS = 14

# ---- gain (multiplicative) errors: wind and sunlight compared with neighbours' daily means
GAIN_RULES = {
    "wind_kmh": dict(eps=1.0, min_ratio_change=2.0, label="wind speed"),
    "ghi_wm2": dict(eps=20.0, min_ratio_change=1.5, label="sunlight"),
}
GAIN_MIN_DAYS_SEG = 2

# ---- blocked rain gauge
BLOCK_HOURS = 12
BLIND_PEER_MEDIAN_MM = 2.0        # neighbours' median rain in a block that the station must have caught
BLIND_PEER_FRACTION = 0.6         # share of neighbours that saw at least 1 mm in the block
BLIND_TOTAL_MM = 10.0             # neighbours must have caught this much in total during the zero run
BURST_MM_PER_INTERVAL = 5.0       # >30 mm/h from a gauge that was dry: stored water, not rain


def _local_day(idx: pd.DatetimeIndex, lon: float) -> pd.DatetimeIndex:
    off = pd.Timedelta(hours=(0.0 if np.isnan(lon) else lon / 15.0))
    return (idx + off).normalize()


# =====================================================================================
def level_shifts(st: Station, peers: Peers) -> list[Candidate]:
    out: list[Candidate] = []
    idx = st.clean.index
    for ch, rule in LEVEL_RULES.items():
        res10 = peers.residual(st.sid, ch)
        res = res10.resample("1h").median()
        if res.notna().sum() < 4 * MIN_SEGMENT_HOURS:
            continue
        segs = segment_levels(res, MIN_SEGMENT_HOURS, rule["min_delta"], rule["noise_mult"])
        if len(segs) < 2:
            continue
        # which segment is "normal"? pressure has an absolute reference (standard atmosphere at the station's height)
        base = max(segs, key=lambda s: s["n"])
        if ch == "pressure_hpa" and not np.isnan(st.elev):
            ref = _network_pressure_deviation(peers, st.sid)
            if not np.isnan(ref):
                p_std = float(std_atm_pressure(st.elev))
                raw_p = peers.frames[ch][st.sid]
                for s in segs:
                    seg_p = raw_p.loc[res.index[s["start_pos"]]:res.index[s["end_pos"]] + pd.Timedelta(minutes=59)]
                    s["abs_dev"] = float(seg_p.median() - p_std - ref) if seg_p.notna().any() else np.inf
                base = min(segs, key=lambda s: abs(s["abs_dev"]))
        for i, s in enumerate(segs):
            if s is base or abs(s["level"] - base["level"]) < rule["min_delta"]:
                continue
            h0, h1 = res.index[s["start_pos"]], res.index[s["end_pos"]] + pd.Timedelta(minutes=50)
            delta = s["level"] - base["level"]
            # refine the start/end boundaries to the logging interval using the largest peer-adjusted jump
            start = _refine_boundary(st, peers, ch, h0) if i > 0 else idx[0]
            end = _refine_boundary(st, peers, ch, h1 + pd.Timedelta(minutes=10), end_side=True) if i < len(segs) - 1 else idx[-1]
            start, end = max(start, idx[0]), min(end, idx[-1])
            local = _local_step(st, peers, ch, start) if i > 0 else np.nan
            abrupt = (not np.isnan(local)) and abs(local) >= ABRUPT_FRACTION * abs(delta)
            ev = {"delta": delta, "local_step": local, "abrupt": abrupt, "base_level": base["level"]}
            where = f"{start:%d %b %H:%M} onwards" if end == idx[-1] else f"{start:%d %b %H:%M} to {end:%d %b %H:%M}"
            if ch == "pressure_hpa" and rule["correctable"] and abrupt and abs(local - delta) <= max(LOCAL_STEP_TOLERANCE, 0.35 * abs(delta)):
                ref_txt = ""
                if "abs_dev" in s and np.isfinite(s["abs_dev"]):
                    ref_txt = (f" The earlier level is much closer to the standard-atmosphere pressure for {st.elev:.0f} m "
                               f"({base['abs_dev']:+.1f} hPa from the network's typical offset) than this block ({s['abs_dev']:+.1f} hPa), so the new level is the wrong one.")
                ev["correction"] = -local
                out.append(Candidate(st.sid, ch, start, end, "step", "other", f"{ch} jumps by {local:+.1f} {rule['unit']} and stays there", "correct",
                                     f"{ch} steps {local:+.1f} {rule['unit']} within one reading (neighbours' change in the same hour removed) and stays {delta:+.1f} {rule['unit']} from its neighbours "
                                     f"{where}; the offset is constant, so it is corrected by subtracting {abs(local):.1f} {rule['unit']}.{ref_txt}",
                                     evidence=ev))
            else:
                how = "an abrupt" if abrupt else "a gradual"
                out.append(Candidate(st.sid, ch, start, end, "step", "other", f"{ch} level shifted away from neighbours", "exclude",
                                     f"{ch} runs {delta:+.1f} {rule['unit']} from its neighbours {where} after {how} change; the size of the error cannot be pinned down well enough to correct it.",
                                     evidence=ev))
    return out


def _network_pressure_deviation(peers: Peers, sid: str) -> float:
    devs = []
    for other, s in peers.stations.items():
        if other == sid or np.isnan(s.elev):
            continue
        m = s.clean["pressure_hpa"].median()
        if not np.isnan(m):
            devs.append(m - float(std_atm_pressure(s.elev)))
    return float(np.median(devs)) if len(devs) >= 3 else np.nan


def _refine_boundary(st, peers, ch, near, end_side=False):
    """Timestamp of the biggest peer-adjusted jump within +-6 h of ``near`` (end_side: last sample before it)."""
    x = peers.frames[ch][st.sid]
    ref = peers.others(st.sid, ch).median(axis=1)
    jump = (x.diff() - ref.diff()).abs()
    w = jump.loc[near - pd.Timedelta(hours=6): near + pd.Timedelta(hours=6)]
    if w.dropna().empty:
        return near
    t = w.idxmax()
    return t - pd.Timedelta(minutes=st.interval) if end_side else t


def _local_step(st, peers, ch, t0) -> float:
    """Change in the station across t0 minus the neighbours' change, using 1 h medians each side."""
    x = peers.frames[ch][st.sid]
    ref = peers.others(st.sid, ch).median(axis=1)
    h = pd.Timedelta(hours=1)
    iv = pd.Timedelta(minutes=st.interval)
    def jump(s):
        b, a = s.loc[t0 - h: t0 - iv], s.loc[t0: t0 + h - iv]
        return a.median() - b.median() if len(a.dropna()) >= 3 and len(b.dropna()) >= 3 else np.nan
    return float(jump(x) - np.nan_to_num(jump(ref), nan=0.0))


# =====================================================================================
def daytime_drift(st: Station, peers: Peers) -> list[Candidate]:
    """Warm/cold bias that appears only with the sun up (shield, siting, reflection, paint).

    For each day: (midday residual vs neighbours) - (night residual). A healthy station wanders around a
    constant; a fouled shield or a reflecting surface adds an error that grows with sunshine, so the
    daily excess ramps. We fit "flat, then ramp" and accept it only if the ramp explains most of the
    variance and is large compared with the day-to-day scatter.
    """
    lat, lon = st.lat_for_solar, st.lon_for_solar
    if np.isnan(lat) or np.isnan(lon):
        return []
    idx = st.clean.index
    el = pd.Series(solar_elevation_deg(idx, lat, lon), index=idx)
    r = peers.residual(st.sid, "temp_c")
    e = pd.DataFrame({"r": r, "d": _local_day(idx, lon)})
    mid = e[el > MIDDAY_EL].groupby("d").r.median()
    ngt = e[el < NIGHT_EL].groupby("d").r.median()
    ex = (mid - ngt).dropna()
    if len(ex) < DRIFT_MIN_DAYS:
        return []
    fit = hinge_fit(ex)
    if not fit or abs(fit["rise"]) < DRIFT_MIN_RISE_C or fit["gain"] < DRIFT_MIN_GAIN or abs(fit["rise"]) < 3 * fit["sigma"]:
        return []
    off = pd.Timedelta(hours=lon / 15.0)
    start = max(fit["onset"] - off, idx[0])
    end = idx[-1]
    daymask = (el > 5) & st.window_mask(start, end)
    night = r[(el < NIGHT_EL)].loc[start:end].median()
    last3 = float(ex.iloc[-3:].mean() - fit["base"])
    sign = "warmer" if fit["rise"] > 0 else "colder"
    return [Candidate(st.sid, "temp_c", start, end, "drift", "other", f"Daytime temperature drifts {sign} than neighbours", "exclude",
                      f"Midday readings pull {abs(last3):.1f} C {sign} than neighbours by {ex.index[-1]:%d %b} (ramp starting about {fit['onset']:%d %b}, "
                      f"explains {fit['gain'] * 100:.0f}% of the day-to-day variation; no other station's fit exceeds 20%) while night-time readings stay within {abs(night):.1f} C, "
                      "so the error appears only with the sun up and keeps growing; daytime values (sun above 5 deg) are excluded because it is not constant enough to correct.",
                      evidence={"mask": daymask, "onset": fit["onset"], "rise": fit["rise"], "gain": fit["gain"], "last3": last3})]


# =====================================================================================
def gain_shifts(st: Station, peers: Peers) -> list[Candidate]:
    out: list[Candidate] = []
    lat, lon = st.lat_for_solar, st.lon_for_solar
    idx = st.clean.index
    for ch, rule in GAIN_RULES.items():
        x = peers.frames[ch][st.sid]
        ref = peers.others(st.sid, ch).median(axis=1).where(peers.others(st.sid, ch).notna().sum(axis=1) >= MIN_PEERS)
        sel = pd.Series(True, index=idx)
        if ch == "ghi_wm2":
            if np.isnan(lat):
                continue
            sel = pd.Series(solar_elevation_deg(idx, lat, lon) > 10, index=idx)
        day = _local_day(idx, lon)
        d = pd.DataFrame({"x": x.where(sel), "r": ref.where(sel), "d": day}).dropna()
        daily = d.groupby("d").agg(x=("x", "mean"), r=("r", "mean"), n=("x", "size"))
        daily = daily[daily.n >= 36 if ch == "wind_kmh" else daily.n >= 18]
        if len(daily) < 8:
            continue
        lr = np.log((daily.x + rule["eps"]) / (daily.r + rule["eps"]))
        full = lr.reindex(pd.date_range(lr.index.min(), lr.index.max(), freq="D"))
        thr = np.log(rule["min_ratio_change"])
        segs = segment_levels(full, GAIN_MIN_DAYS_SEG, thr, 3.0)
        if len(segs) < 2:
            continue
        base = max(segs, key=lambda s: s["n"])
        for i, s in enumerate(segs):
            if s is base or abs(s["level"] - base["level"]) < thr:
                continue
            d0 = full.index[s["start_pos"]]
            d1 = full.index[s["end_pos"]]
            dseg = daily.loc[d0:d1]
            dbase = daily.loc[full.index[base["start_pos"]]: full.index[base["end_pos"]]]
            ratio = float((dseg.x.sum() / max(dseg.r.sum(), 1e-9)) / (dbase.x.sum() / max(dbase.r.sum(), 1e-9)))
            lo = d0 - pd.Timedelta(hours=(0 if np.isnan(lon) else lon / 15.0))
            hi = d1 + pd.Timedelta(hours=24) - pd.Timedelta(hours=(0 if np.isnan(lon) else lon / 15.0)) - pd.Timedelta(minutes=st.interval)
            lo, hi = max(lo, idx[0]), min(hi, idx[-1])
            if i > 0:                                  # refine the onset on the 10-minute series
                f = np.log((x + rule["eps"]) / (ref + rule["eps"])).where(sel)
                w = f.loc[lo - pd.Timedelta(hours=30): lo + pd.Timedelta(hours=30)].dropna()
                if len(w) > 60:
                    sp = best_split(w.values, 12)
                    if sp:
                        lo = w.index[sp[0]]
            hint = ""
            if ch == "wind_kmh" and 0.85 <= ratio * 3.6 <= 1.15:
                hint = (f" The ratio ({ratio:.2f}) is within 15% of 1/3.6, which is what a unit change from m/s to km/h would produce - "
                        "one fact to confirm on site before correcting.")
            elif ch == "wind_kmh" and ratio > 0 and 0.85 <= ratio / 3.6 <= 1.15:
                hint = f" The ratio ({ratio:.2f}) is close to 3.6, the m/s to km/h factor."
            mask = (sel & st.window_mask(lo, hi)) if ch == "ghi_wm2" else st.window_mask(lo, hi)
            out.append(Candidate(st.sid, ch, lo, hi, "gain", "other", f"{rule['label'].capitalize()} reads {ratio:.2f}x what it used to", "exclude",
                                 f"From {lo:%d %b %H:%M} the station's {rule['label']} runs at {ratio:.2f} times its earlier share of the neighbours' "
                                 f"(daily means over {len(daily)} days compared with the neighbours' median); "
                                 f"the gain is not known well enough to correct.{hint}",
                                 evidence={"ratio": ratio, "mask": mask}))
    return out


# =====================================================================================
def rain_blocked(st: Station, peers: Peers) -> list[Candidate]:
    out: list[Candidate] = []
    idx = st.clean.index
    x = peers.frames["rain_mm"][st.sid]
    others = peers.others(st.sid, "rain_mm")
    if others.shape[1] < MIN_PEERS or x.notna().sum() < 100:
        return out
    blk = ((idx - idx[0]) // pd.Timedelta(hours=BLOCK_HOURS)).values
    S = x.groupby(blk).sum(min_count=1)
    cover = x.notna().groupby(blk).mean()
    P = others.groupby(blk).sum(min_count=1)
    pmed = P.median(axis=1)
    pfrac = (P >= 1.0).sum(axis=1) / P.notna().sum(axis=1).clip(lower=1)
    blind = (S == 0) & (cover >= 0.9) & (pmed >= BLIND_PEER_MEDIAN_MM) & (pfrac >= BLIND_PEER_FRACTION)
    if not blind.any():
        return out
    zero = (S == 0) & (cover >= 0.5)
    block_ids = list(S.index)
    i = 0
    while i < len(block_ids):
        if not zero.iloc[i]:
            i += 1; continue
        j = i
        while j + 1 < len(block_ids) and zero.iloc[j + 1]:
            j += 1
        run = block_ids[i:j + 1]
        bl = [b for b in run if blind.loc[b]]
        if bl and float(pmed.loc[bl].sum()) >= BLIND_TOTAL_MM:
            b0 = bl[0]
            pos0 = np.where(blk == b0)[0]
            # first time at least 30% of neighbours were raining inside the first missed block
            rain_now = (others.iloc[pos0] > 0).sum(axis=1) / others.iloc[pos0].notna().sum(axis=1).clip(lower=1)
            hit = rain_now[rain_now >= 0.3]
            start = idx[pos0[0]] if hit.empty else hit.index[0]
            last_pos = np.where(blk == run[-1])[0][-1]
            end = idx[last_pos]
            # the last zero sample before the next non-zero reading
            after = x.iloc[last_pos + 1:]
            nz = after[after > 0]
            if len(nz):
                end = nz.index[0] - pd.Timedelta(minutes=st.interval)
            span_p = others.loc[start:end].sum().median()
            out.append(Candidate(st.sid, "rain_mm", start, end, "blocked", "other", "Rain gauge recorded nothing while neighbours had rain", "exclude",
                                 f"rain_mm is exactly 0 from {start:%d %b %H:%M} while the other stations' median caught {span_p:.1f} mm in the same period "
                                 f"({int(blind.loc[run].sum())} half-days with neighbours at {pmed.loc[bl].min():.1f}+ mm); the funnel was blocked, so a dry reading here is not evidence of dry weather.",
                                 evidence={"neighbour_mm": float(span_p)}))
            # release burst right after the blocked period
            if len(nz):
                t0 = nz.index[0]
                win = x.loc[t0: t0 + pd.Timedelta(hours=1)]
                big = win[win >= BURST_MM_PER_INTERVAL]
                if len(big):
                    t1 = big.index[-1]
                    tot = float(x.loc[t0:t1].sum())
                    out.append(Candidate(st.sid, "rain_mm", t0, t1, "burst", "other", "Stored water released in one burst", "exclude",
                                         f"{tot:.1f} mm arrive within {(t1 - t0).total_seconds() / 60 + st.interval:.0f} minutes right after the dry run, "
                                         f"with neighbours at {float(others.loc[t0:t1].sum().median()):.1f} mm for that hour; this is water that collected while the funnel was blocked "
                                         f"(neighbours caught {span_p:.1f} mm over the whole blocked period), so it is not rainfall at that time.",
                                         evidence={"total_mm": tot, "neighbour_mm": float(span_p)}))
        i = j + 1
    return out
