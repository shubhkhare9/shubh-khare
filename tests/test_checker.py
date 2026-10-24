"""Tests for the checker. They run on synthetic packs generated in tests/synth.py, never on the real pack.

    python -m pytest -q
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import synth  # noqa: E402
from src.data_loader import nmea_checksum, read_gps  # noqa: E402
from src.pipeline import run  # noqa: E402
from src.solar import solar_noon_utc_hours  # noqa: E402

DAYS = 21
START = "2025-05-01"


def build(tmp_path, faults=True, n=6, seed=0):
    st = synth.make_stations(n, seed)
    w = synth.make_weather(st, START, DAYS, seed)
    g = synth.make_gps(st, START, DAYS, seed)
    ids = list(st.station_id)
    log = []
    if faults:
        t1, t2, t3, t4, t5, t6 = ids[:6]
        synth.inject_sentinel(w[t1], "2025-05-06 03:20", 85.0)
        synth.inject_rain_blocked(w[t1], "2025-05-08 00:00", 8 * 144)
        synth.inject_flatline(w[t2], "2025-05-09 06:00", 40)
        g[t2] = synth.inject_gps_position(g[t2], 0.2)
        synth.inject_pressure_step(w[t3], "2025-05-11 12:00", 5.0)
        w[t3] = synth.inject_duplicate_block(w[t3], "2025-05-04 00:00", 144)
        synth.inject_rh_stuck(w[t4], "2025-05-07 08:00", 60)
        w[t4] = synth.inject_gap(w[t4], "2025-05-13 00:00", 30)
        synth.inject_wind_gain(w[t5], "2025-05-12 00:00", 0.3)
        w[t6], g[t6] = synth.inject_clock_shift(w[t6], g[t6], "2025-05-15 12:00", 3)
        log = ["2025-05-11  T03  Swapped the pressure board, no calibration.", "2025-05-15  T06  Battery swapped, logger rebooted."]
    root = synth.write_pack(tmp_path / "pack", st, w, g, log)
    return root, ids


def flags_for(df, station, channel=None, text=None):
    m = df.station_id == station
    if channel:
        m &= df.channel == channel
    if text:
        m &= df.issue.str.contains(text, case=False) | df.note.str.contains(text, case=False)
    return df[m]


# ------------------------------------------------------------------------------------------------
def test_clean_pack_raises_no_faults(tmp_path):
    root, _ = build(tmp_path, faults=False)
    df = run(root, quiet=True)["flags"]
    bad = df[df.action != "keep"]
    assert bad.empty, bad[["station_id", "channel", "start_utc", "issue", "action"]].to_string()


@pytest.fixture(scope="module")
def faulty(tmp_path_factory):
    root, ids = build(tmp_path_factory.mktemp("faulty"), faults=True)
    return run(root, quiet=True)["flags"], ids


def test_sentinel_spike_is_excluded(faulty):
    df, ids = faulty
    f = flags_for(df, ids[0], "temp_c", "error code")
    assert len(f) == 1 and f.iloc[0].action == "exclude" and f.iloc[0].start_utc == "2025-05-06 03:20"


def test_frozen_sensor_is_excluded(faulty):
    df, ids = faulty
    f = flags_for(df, ids[1], "temp_c", "frozen")
    assert len(f) == 1 and f.iloc[0].action == "exclude"
    assert abs((pd.Timestamp(f.iloc[0].end_utc) - pd.Timestamp("2025-05-09 06:00")).total_seconds() / 600 - 39) <= 1


def test_pressure_step_is_corrected(faulty):
    df, ids = faulty
    f = flags_for(df, ids[2], "pressure_hpa")
    assert len(f) == 1 and f.iloc[0].action == "correct"
    assert abs(pd.Timestamp(f.iloc[0].start_utc) - pd.Timestamp("2025-05-11 12:00")) <= pd.Timedelta(hours=1)
    assert re.search(r"subtracting 5\.\d", f.iloc[0].note) or re.search(r"subtracting 4\.\d", f.iloc[0].note)


def test_replayed_block_is_excluded(faulty):
    df, ids = faulty
    f = flags_for(df, ids[2], "all", "written again")
    assert len(f) == 1 and f.iloc[0].action == "exclude" and f.iloc[0].start_utc == "2025-05-04 00:00"


def test_impossible_humidity_run_is_one_flag(faulty):
    df, ids = faulty
    f = flags_for(df, ids[3], "rh_pct")
    f = f[f.action == "exclude"]
    assert len(f) == 1 and f.iloc[0].start_utc == "2025-05-07 08:00"


def test_gap_is_reported(faulty):
    df, ids = faulty
    f = flags_for(df, ids[3], "all", "gap")
    assert len(f) == 1 and f.iloc[0].start_utc == "2025-05-13 00:00" and "30 consecutive" in f.iloc[0].note


def test_wind_gain_error_is_excluded_not_corrected(faulty):
    df, ids = faulty
    f = flags_for(df, ids[4], "wind_kmh")
    f = f[f.action != "keep"]
    assert len(f) == 1 and f.iloc[0].action == "exclude"
    assert abs(pd.Timestamp(f.iloc[0].start_utc) - pd.Timestamp("2025-05-12 00:00")) <= pd.Timedelta(hours=14)


def test_clock_shift_is_corrected_with_gps_evidence(faulty):
    df, ids = faulty
    f = flags_for(df, ids[5], "all", "clock")
    assert len(f) == 1 and f.iloc[0].action == "correct"
    assert f.iloc[0].start_utc == "2025-05-15 12:00" and "3.0 h" in f.iloc[0].note and "GPS" in f.iloc[0].note


def test_gps_position_mismatch_flagged_as_metadata(faulty):
    df, ids = faulty
    f = flags_for(df, ids[1], "metadata")
    assert len(f) == 1 and "disagrees with GPS" in f.iloc[0].issue


def test_blocked_rain_gauge(faulty):
    df, ids = faulty
    f = flags_for(df, ids[0], "rain_mm")
    assert len(f) >= 1 and (f.action == "exclude").all()


def test_output_format(faulty):
    df, _ = faulty
    assert list(df.columns) == ["station_id", "channel", "start_utc", "end_utc", "issue", "action", "note"]
    assert df.action.isin(["exclude", "correct", "keep"]).all()
    for col in ("start_utc", "end_utc"):
        assert df[col].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}").all()
    assert (pd.to_datetime(df.start_utc) <= pd.to_datetime(df.end_utc)).all()
    assert df.note.str.len().gt(10).all()


def test_one_command_from_any_folder(tmp_path):
    root, _ = build(tmp_path / "x", faults=True)
    work = tmp_path / "elsewhere"
    work.mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "checker.py"), str(root)], cwd=work, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (work / "flags.csv").exists()
    assert len(pd.read_csv(work / "flags.csv")) > 5


# ------------------------------------------------------------------------------------------------
def test_survives_messy_input(tmp_path):
    root, ids = build(tmp_path, faults=False, n=5)
    p = root / "data" / f"{ids[0]}.csv"
    df = pd.read_csv(p, dtype=str)
    df = df[["rh_pct", "timestamp_utc", "temp_c", "pressure_hpa", "wind_kmh", "wind_dir_deg", "rain_mm", "ghi_wm2"]]   # reordered, ghi kept
    df.loc[10, "temp_c"] = "N/A"
    df.loc[20, "timestamp_utc"] = "not a date"
    df.loc[30, "pressure_hpa"] = ""
    df.to_csv(p, index=False)
    (root / "maintenance_log.md").unlink()
    for g in (root / "gps").glob("*.nmea"):
        g.unlink()
    out = run(root, quiet=True)["flags"]
    assert (out.issue.str.contains("Unreadable timestamps")).any()
    assert (out.issue.str.contains("Missing values in temp_c")).any()


def test_missing_channel_and_few_stations(tmp_path):
    root, ids = build(tmp_path, faults=False, n=3)
    p = root / "data" / f"{ids[1]}.csv"
    pd.read_csv(p).drop(columns=["ghi_wm2"]).to_csv(p, index=False)
    out = run(root, quiet=True)["flags"]     # fewer than 4 stations: neighbour checks are skipped, no crash
    assert out.issue.str.contains("not in file").any()


def test_station_not_in_sheet(tmp_path):
    root, ids = build(tmp_path, faults=False, n=5)
    sheet = pd.read_csv(root / "stations.csv")
    sheet[sheet.station_id != ids[0]].to_csv(root / "stations.csv", index=False)
    out = run(root, quiet=True)["flags"]
    assert flags_for(out, ids[0], "metadata", "missing from stations").shape[0] == 1


# ------------------------------------------------------------------------------------------------
def test_nmea_checksum_and_parse(tmp_path):
    body = "GPRMC,010000.00,A,3953.6735,N,10533.8914,W,0.0,0.0,010325,,,A"
    good = f"2025-03-01 01:00:00,${body}*{nmea_checksum(body)}"
    bad = f"2025-03-01 02:00:00,${body}*00"
    p = tmp_path / "g.nmea"
    p.write_text(good + "\n" + bad + "\n")
    g = read_gps(p)
    assert list(g.cks_ok) == [True, False]
    assert g.lat.iloc[0] == pytest.approx(39.8945583, abs=1e-6) and g.lon.iloc[0] == pytest.approx(-105.564857, abs=1e-5)
    assert g.gps_time.iloc[0] == pd.Timestamp("2025-03-01 01:00:00")


def test_solar_noon_s04_20_march():
    """Hand calculation in DECISIONS.md: about 19:08 UTC at lon -105.146 on 20 March 2025."""
    h = solar_noon_utc_hours(pd.Timestamp("2025-03-20"), -105.146)
    assert 19.10 <= h <= 19.17
