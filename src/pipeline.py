"""Run every check, in the order that lets later checks trust earlier ones."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import gps_checks, integrity_checks, peer_checks, time_series_checks, weather_checks
from .data_loader import load_pack
from .flag_builder import apply_and_flag, finalize
from .maintenance_checks import note_text, related_notes, review_log
from .models import Flag
from .peer_frames import Peers
from .station import Station


def _log(msg: str, quiet: bool):
    if not quiet:
        print(msg, file=sys.stderr)


def build_stations(pack) -> dict[str, Station]:
    stations: dict[str, Station] = {}
    for sid, raw in pack.data.items():
        meta = pack.stations.loc[sid] if sid in pack.stations.index else None
        declared = float(meta["logger_interval_min"]) if meta is not None and not pd.isna(meta.get("logger_interval_min")) else np.nan
        inferred = integrity_checks.infer_interval_min(raw["ts"].dropna(), declared) if raw["ts"].notna().any() else 10
        st = Station(sid=sid,
                     lat=float(meta["lat"]) if meta is not None else np.nan,
                     lon=float(meta["lon"]) if meta is not None else np.nan,
                     elev=float(meta["elevation_m"]) if meta is not None else np.nan,
                     interval=int(declared) if not np.isnan(declared) else inferred,
                     name=str(meta["name"]) if meta is not None and "name" in meta and not pd.isna(meta["name"]) else sid,
                     raw=raw, gps=pack.gps.get(sid))
        stations[sid] = st
    return stations


def run(pack_dir, quiet: bool = False, use_iforest: bool = False) -> dict:
    pack = load_pack(pack_dir)
    if not pack.data:
        raise SystemExit("error: no station CSV files found in data/")
    stations = build_stations(pack)
    _log(f"loaded {len(stations)} stations from {pack.root}", quiet)

    # ---- stage 1: timeline (clock, replays, duplicates, gaps) and GPS
    gpsoff = {}
    for sid, st in list(stations.items()):
        if st.raw["ts"].notna().sum() == 0:
            _log(f"{sid}: no readable timestamps, skipped", quiet)
            del stations[sid]
            continue
        gpsoff[sid] = integrity_checks.run_integrity(st)
        st.init_masks()
        gps_checks.check_position(st)
        gps_checks.check_elevation(st)
        gps_checks.check_gps_health(st)
        gps_checks.check_clock_offsets(st, gpsoff[sid])

    # sheet / file mismatches
    extra_flags: list[Flag] = []
    for sid in pack.stations.index:
        if sid not in pack.data:
            extra_flags.append(Flag(sid, "metadata", pd.Timestamp("1970-01-01"), pd.Timestamp("1970-01-01"), "Listed station has no data file", "keep",
                                    "stations.csv has a row for this station but data/ has no CSV for it.", kind="missing_file"))
    for note in pack.load_notes:
        if note["kind"] == "unreadable":
            extra_flags.append(Flag(note["station"], "all", pd.Timestamp("1970-01-01"), pd.Timestamp("1970-01-01"), "Data file unreadable", "exclude",
                                    f"The CSV could not be read: {note['detail'][:120]}", kind="unreadable"))
        elif note["kind"] == "csv_info":
            d = note["detail"]
            st = stations.get(note["station"])
            if st is None or st.clean is None or st.clean.empty:
                continue
            t0, t1 = st.clean.index[0], st.clean.index[-1]
            for ch in d["missing_channels"]:
                extra_flags.append(Flag(st.sid, ch, t0, t1, f"Channel {ch} not in file", "exclude",
                                        f"The station file has no {ch} column.", kind="missing_channel"))

    if not stations:
        raise SystemExit("error: no usable stations")

    # ---- stage 2: physical limits, spikes, frozen sensors (neighbour-checked)
    peers = Peers(stations)
    for sid, st in stations.items():
        cands = weather_checks.check_physical(st)
        cands += weather_checks.check_missing(st)
        cands += time_series_checks.find_spikes(st, peers)
        cands += time_series_checks.find_flatlines(st, peers)
        cands += time_series_checks.find_extreme_wind(st, peers)
        st.flags += apply_and_flag(st, cands, pack.log)
    peers.refresh()

    # ---- stage 3: comparisons with neighbours (needs >= 4 stations)
    if peers.enough:
        for sid, st in stations.items():
            cands = peer_checks.level_shifts(st, peers)
            cands += peer_checks.daytime_drift(st, peers)
            cands += peer_checks.gain_shifts(st, peers)
            cands += peer_checks.rain_blocked(st, peers)
            st.flags += apply_and_flag(st, cands, pack.log)
    else:
        _log("fewer than 4 stations: neighbour comparisons skipped", quiet)

    # ---- stage 4: attach log evidence to flags that did not get it yet, collect
    all_flags: list[Flag] = list(extra_flags)
    for st in stations.values():
        for f in st.flags:
            if "Volunteer log" not in f.note and f.channel in ("all", "metadata"):
                f.note += note_text(related_notes(pack.log, st.sid, f.channel, f.start, f.end))
        all_flags += st.flags

    df = finalize(all_flags)
    iforest_df = None
    if use_iforest:
        from .iforest import score_stations
        iforest_df = score_stations(stations)
    return {"flags": df, "flag_objects": all_flags, "stations": stations, "pack": pack,
            "log_review": review_log(pack.log, all_flags), "iforest": iforest_df}
