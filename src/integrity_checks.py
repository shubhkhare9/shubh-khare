"""Timeline integrity: bad timestamps, replayed blocks, wrong clocks, duplicates, gaps.

This is the first stage because every later check assumes that "row N is the 10 minutes ending at
timestamp T on the UTC clock". If that is false the weather values are fine but their *time* is
wrong, and the right action is to correct (or exclude) the time, not to blame the sensors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import CHANNELS, Flag
from .solar import phase_offset_hours
from .station import Station
from .stats_utils import runs_of, mad_sigma

# --- thresholds (and why) ------------------------------------------------------------------
# GPS time stamps are integer seconds and the pack's loggers agree with GPS to <=2 s when healthy,
# so anything above 30 s that persists is a real clock problem rather than jitter.
GPS_JITTER_S = 30
# Fewest valid fixes needed before a GPS clock offset is believed (fixes are ~hourly).
MIN_GPS_FIXES = 3
# Shifts smaller than half an interval cannot be re-aligned to the grid, so they are only reported.
MIN_FIX_SHIFT_S = 300


def infer_interval_min(ts: pd.Series, declared) -> int:
    d = ts.sort_values().drop_duplicates().diff().dropna().dt.total_seconds() / 60.0
    d = d[d > 0]
    inferred = int(round(d.median())) if len(d) else 10
    inferred = max(inferred, 1)
    return inferred


def _range_text(a, b) -> str:
    return f"{pd.Timestamp(a):%Y-%m-%d %H:%M} to {pd.Timestamp(b):%Y-%m-%d %H:%M}"


def _frac_identical(a: pd.DataFrame, b: pd.DataFrame) -> float:
    eq = (a.values == b.values) | (np.isnan(a.values) & np.isnan(b.values))
    return float(eq.all(axis=1).mean()) if len(eq) else 0.0


def gps_clock_offsets(gps: pd.DataFrame) -> pd.DataFrame:
    """Valid RMC fixes with (gps_time - logger_time) in seconds. Positive = logger behind GPS."""
    if gps is None or gps.empty or "gps_time" not in gps:
        return pd.DataFrame(columns=["lts", "off_s"])
    g = gps[(gps.kind == "RMC") & gps.valid & gps.cks_ok & gps.gps_time.notna() & gps.lts.notna()].copy()
    g["off_s"] = (g.gps_time - g.lts).dt.total_seconds()
    return g[["lts", "off_s"]].reset_index(drop=True)


def run_integrity(st: Station) -> pd.DataFrame:
    """Build st.clean (gridded, corrected clock) and add timeline flags. Returns the GPS offsets used."""
    raw = st.raw.copy()
    interval = st.interval
    flags: list[Flag] = []
    gpsoff = gps_clock_offsets(st.gps)
    step = pd.Timedelta(minutes=interval)

    # ---- 1. unparseable timestamps ---------------------------------------------------------
    bad = raw["ts"].isna()
    if bad.any():
        good_ts = raw["ts"].ffill().bfill()
        for a, b in runs_of(bad.reset_index(drop=True)):
            prev = raw["ts"].iloc[:a].dropna()
            nxt = raw["ts"].iloc[b + 1:].dropna()
            s = (prev.iloc[-1] + step) if len(prev) else good_ts.iloc[a]
            e = (nxt.iloc[0] - step) if len(nxt) else good_ts.iloc[b]
            flags.append(Flag(st.sid, "all", s, max(e, s), "Unreadable timestamps", "exclude",
                              f"{b - a + 1} row(s) have timestamps that cannot be parsed (e.g. '{raw['ts_text'].iloc[a]}'), "
                              "so they cannot be placed on the UTC clock.", kind="bad_timestamp"))
        raw = raw[~bad].copy()
    raw = raw.reset_index(drop=True)
    if raw.empty:
        st.clean = pd.DataFrame(columns=CHANNELS, index=pd.DatetimeIndex([], name="ts"))
        st.flags += flags
        return gpsoff

    # ---- 1b. a clock that is wrong by a constant amount for the whole record -----------------
    if len(gpsoff) >= 10:
        med = float(gpsoff.off_s.median())
        if abs(med) >= MIN_FIX_SHIFT_S and (gpsoff.off_s.sub(med).abs() <= 60).mean() >= 0.8:
            shift_all = round(med / (interval * 60)) * (interval * 60)
            raw["ts"] = raw["ts"] + pd.Timedelta(seconds=shift_all)
            flags.append(Flag(st.sid, "all", raw["ts"].min() - pd.Timedelta(seconds=shift_all), raw["ts"].max() - pd.Timedelta(seconds=shift_all),
                              "Logger clock wrong for whole record", "correct",
                              f"{len(gpsoff)} GPS fixes read {med / 3600:+.2f} h from the logger clock throughout; all timestamps shifted by "
                              f"{shift_all / 3600:+.2f} h.", kind="clock_constant", evidence={"shift_s": shift_all}))
            gpsoff = gpsoff.assign(off_s=gpsoff.off_s - shift_all)

    # ---- 2. split the file at every backward step in time ---------------------------------
    backward = raw["ts"].diff().dt.total_seconds() < 0
    raw["seg"] = backward.cumsum()
    raw["ts_fix"] = raw["ts"]
    raw["drop"] = False          # rows removed because they are replayed copies
    raw["bad_clock"] = False     # rows whose clock could not be corrected (-> excluded)
    last_fixed_end = None

    for k in sorted(raw["seg"].unique()):
        seg = raw[raw["seg"] == k]
        if k == 0:
            last_fixed_end = seg["ts_fix"].max()
            continue
        earlier = raw[(raw["seg"] < k) & ~raw["drop"]].drop_duplicates("ts_fix", keep="first").set_index("ts_fix")
        idx_overlap = seg["ts"].isin(earlier.index)
        n_over = int(idx_overlap.sum())
        replay = False
        if n_over >= 0.9 * len(seg):
            frac = _frac_identical(seg.loc[idx_overlap, CHANNELS], earlier.loc[seg.loc[idx_overlap, "ts"], CHANNELS])
            replay = frac >= 0.95
        s0, s1 = seg["ts"].iloc[0], seg["ts"].iloc[-1]
        jump_h = (seg["ts"].iloc[0] - raw.loc[seg.index[0] - 1, "ts"]).total_seconds() / 3600.0

        if replay:
            raw.loc[seg.index, "drop"] = True
            flags.append(Flag(st.sid, "all", s0, s1, "Block of old records written again", "exclude",
                              f"{len(seg)} rows from {_range_text(s0, s1)} are exact copies of rows already earlier in the file "
                              f"(every channel identical), appended after time stepped back {abs(jump_h):.1f} h; the first copy is kept.",
                              kind="replayed_block", evidence={"rows": len(seg)}))
            continue

        # -- the block is new data stamped on a wrong clock: estimate the shift two independent ways
        gsel = gpsoff[(gpsoff.lts >= s0 - step) & (gpsoff.lts <= s1 + step)]
        gps_shift = float(gsel.off_s.median()) if len(gsel) >= MIN_GPS_FIXES else np.nan
        n_gps_all = len(gsel)
        if not np.isnan(gps_shift):
            # fixes written just before the restart share this logger-time span but carry the old (correct) clock
            gsel = gsel[(gsel.off_s - gps_shift).abs() <= 300]
        gps_spread = float(gsel.off_s.max() - gsel.off_s.min()) if len(gsel) >= MIN_GPS_FIXES else np.nan
        cont_shift = (last_fixed_end + step - s0).total_seconds() if last_fixed_end is not None else np.nan
        shift_s = np.nan
        if not np.isnan(gps_shift) and abs(gps_shift) >= MIN_FIX_SHIFT_S:
            shift_s = round(gps_shift / (interval * 60)) * (interval * 60)
        elif not np.isnan(cont_shift):
            shift_s = round(cont_shift / (interval * 60)) * (interval * 60)
        agree = (not np.isnan(gps_shift)) and (not np.isnan(cont_shift)) and abs(gps_shift - cont_shift) <= interval * 60
        solar_off, n_days = phase_offset_hours(pd.DatetimeIndex(seg["ts"]), seg["ghi_wm2"], st.lat_for_solar, st.lon_for_solar)
        solar_txt = (f" The daylight curve in this block peaks {abs(solar_off):.1f} h {'early' if solar_off < 0 else 'late'}"
                     f" compared with the sun's position ({n_days} days)." if not np.isnan(solar_off) and abs(solar_off) >= 0.5 else "")

        if np.isnan(shift_s) or (not agree and np.isnan(gps_shift)):
            # shift known only from continuity (or not at all): not enough to rewrite time
            raw.loc[seg.index, "bad_clock"] = True
            flags.append(Flag(st.sid, "all", s0, s1, "Logger clock stepped back; true time unknown", "exclude",
                              f"Time jumps back {abs(jump_h):.1f} h at the start of this block and no GPS fix confirms a shift, so the "
                              f"{len(seg)} rows cannot be placed safely.{solar_txt}", kind="clock_unknown"))
            continue

        new_ts = seg["ts"] + pd.Timedelta(seconds=shift_s)
        collide = new_ts.isin(earlier.index).sum()
        if collide:
            raw.loc[seg.index, "bad_clock"] = True
            flags.append(Flag(st.sid, "all", s0, s1, "Logger clock wrong; corrected times would overlap", "exclude",
                              f"Block is stamped {shift_s / 3600:.1f} h off but {collide} corrected timestamps would collide with existing rows, "
                              "so it cannot be re-timed unambiguously.", kind="clock_collision"))
            continue
        raw.loc[seg.index, "ts_fix"] = new_ts
        last_fixed_end = new_ts.max()
        s_true0, s_true1 = new_ts.iloc[0], new_ts.iloc[-1]
        nover = int(n_over)
        sign = "behind" if shift_s > 0 else "ahead of"
        gtxt = (f"{len(gsel)} valid GPS fixes in the block read {gps_shift / 3600:+.2f} h from the logger clock (spread {gps_spread:.0f} s)"
                if len(gsel) >= MIN_GPS_FIXES else "no usable GPS fix in the block")
        ctxt = (f" Re-timed, the block continues the previous rows without a gap (continuity shift {cont_shift / 3600:+.2f} h)."
                if agree else "")
        otxt = (f" Before correction {nover} of its timestamps duplicated earlier ones." if nover else "")
        flags.append(Flag(st.sid, "all", s_true0, s_true1, "Logger clock wrong after restart", "correct",
                          f"Timestamps stepped back {abs(jump_h):.1f} h at {raw.loc[seg.index[0] - 1, 'ts']:%d %b %H:%M}; {gtxt}, so the clock ran {abs(shift_s) / 3600:.1f} h {sign} UTC "
                          f"(the file labels this block {_range_text(s0, s1)}). Corrected by adding {shift_s / 3600:+.1f} h.{ctxt}{otxt}{solar_txt}",
                          kind="clock_shift", evidence={"shift_s": shift_s, "gps_shift": gps_shift, "true_start": s_true0,
                                                         "true_end": s_true1, "rows": len(seg), "collisions_before": nover,
                                                         "solar_offset_h": solar_off}))

    kept = raw[~raw["drop"]].copy()

    # ---- 3. duplicates that remain on the corrected clock ----------------------------------
    kept = kept.sort_values(["ts_fix", "row"], kind="stable")
    dup_all = kept.duplicated("ts_fix", keep=False)
    if dup_all.any():
        grp = kept[dup_all].groupby("ts_fix")
        identical, conflicting = [], []
        for t, g in grp:
            same = _frac_identical(g[CHANNELS].iloc[1:], g[CHANNELS].iloc[[0] * (len(g) - 1)]) == 1.0
            (identical if same else conflicting).append(t)
        if identical:
            ser = pd.Series(True, index=pd.DatetimeIndex(identical)).sort_index()
            full = pd.Series(False, index=pd.date_range(ser.index.min(), ser.index.max(), freq=step))
            full.loc[ser.index] = True
            for a, b in runs_of(full):
                s, e = full.index[a], full.index[b]
                n = int(full.iloc[a:b + 1].sum())
                flags.append(Flag(st.sid, "all", s, e, "Repeated identical records", "exclude",
                                  f"{n} timestamp(s) appear twice with identical values; the second copy is dropped and the first kept.",
                                  kind="duplicate_identical"))
        if conflicting:
            ser = pd.Series(True, index=pd.DatetimeIndex(conflicting)).sort_index()
            full = pd.Series(False, index=pd.date_range(ser.index.min(), ser.index.max(), freq=step))
            full.loc[ser.index] = True
            for a, b in runs_of(full):
                s, e = full.index[a], full.index[b]
                n = int(full.iloc[a:b + 1].sum())
                flags.append(Flag(st.sid, "all", s, e, "Same time, different values", "exclude",
                                  f"{n} timestamp(s) occur more than once with different readings, so neither can be trusted.",
                                  kind="duplicate_conflict"))
            kept.loc[kept["ts_fix"].isin(conflicting), CHANNELS] = np.nan
        kept = kept.drop_duplicates("ts_fix", keep="first")

    # ---- 4. block rows that could not be re-timed, off-grid stamps -----------------------------
    unfixable = kept[kept["bad_clock"]]
    kept = kept[~kept["bad_clock"]]
    minutes = (kept["ts_fix"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)   # unit-safe (pandas 3 uses mixed resolutions)
    offgrid = (minutes % interval != 0) | (kept["ts_fix"].dt.second != 0)
    if offgrid.any():
        og = kept[offgrid]
        flags.append(Flag(st.sid, "all", og["ts_fix"].min(), og["ts_fix"].max(), "Timestamps off the logging grid", "exclude",
                          f"{len(og)} row(s) are not on a {interval}-minute boundary (e.g. {og['ts_fix'].iloc[0]:%H:%M:%S}); dropped rather than rounded.",
                          kind="off_grid"))
        kept = kept[~offgrid]

    # ---- 5. grid + gaps --------------------------------------------------------------------
    grid = pd.date_range(kept["ts_fix"].min(), kept["ts_fix"].max(), freq=step, name="ts")
    clean = kept.set_index("ts_fix")[CHANNELS].reindex(grid)
    present = pd.Series(grid.isin(kept["ts_fix"]), index=grid)
    for a, b in runs_of(~present):
        s, e = grid[a], grid[b]
        n = b - a + 1
        hrs = n * interval / 60.0
        # evidence: did the GPS receiver also go quiet (logger down) or only the data (link down)?
        gsub = st.gps[(st.gps.lts >= s) & (st.gps.lts <= e)] if st.gps is not None and not st.gps.empty else None
        gtxt = ""
        if gsub is not None and len(gsub) == 0 and hrs >= 3:
            gtxt = " The GPS file is also silent for this period, so the logger itself was not recording (not just a website outage)."
        elif gsub is not None and len(gsub) > 0 and hrs >= 3:
            gtxt = f" The GPS file still has {len(gsub)} lines in this period, so the logger kept running and the rows were lost."
        flags.append(Flag(st.sid, "all", s, e, "Missing records (gap)", "exclude",
                          f"{n} consecutive {interval}-minute record(s) are absent ({hrs:.1f} h); nothing is estimated for this period.{gtxt}",
                          kind="gap", evidence={"rows": n, "hours": hrs}))
    st.clean = clean
    st.present = present
    st.flags += flags
    st.notes["rows_in_file"] = len(st.raw)
    st.notes["rows_on_grid"] = int(present.sum())
    return gpsoff
