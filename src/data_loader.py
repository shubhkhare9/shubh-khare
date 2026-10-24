"""Load a data pack: station CSVs, NMEA files, stations.csv, maintenance_log.md.

Nothing here hard-codes station ids, dates or row counts. Stations are discovered
from ``data/*.csv``; ``stations.csv`` supplies metadata when it has a row for them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models import CHANNELS

STATION_COLS = ["station_id", "name", "lat", "lon", "elevation_m", "installed", "logger_interval_min"]


@dataclass
class Pack:
    root: Path
    stations: pd.DataFrame                      # index station_id (from stations.csv, may be partial)
    data: dict[str, pd.DataFrame] = field(default_factory=dict)   # raw logger rows per station
    gps: dict[str, pd.DataFrame] = field(default_factory=dict)
    log: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "station", "text"]))
    load_notes: list[dict] = field(default_factory=list)   # problems found while loading


# ---------------------------------------------------------------- stations.csv
def read_stations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STATION_COLS[1:], index=pd.Index([], name="station_id"))
    df = pd.read_csv(path, dtype=str, skipinitialspace=True, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if "station_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "station_id"})
    df["station_id"] = df["station_id"].str.strip()
    for c in ("lat", "lon", "elevation_m", "logger_interval_min"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.set_index("station_id")


# ---------------------------------------------------------------- station CSV
def read_station_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return rows in FILE ORDER with parsed timestamps (NaT where unparseable).

    ``row`` keeps the original order: the order matters because a logger that restarts with a
    wrong clock writes timestamps that jump backwards, and sorting would hide that.
    """
    raw = pd.read_csv(path, dtype=str, skipinitialspace=True, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    raw.columns = [c.strip().lower() for c in raw.columns]
    info = {"non_numeric": {}, "missing_channels": [], "extra_columns": []}
    tcol = next((c for c in raw.columns if c.startswith("timestamp")), raw.columns[0])
    df = pd.DataFrame({"row": np.arange(len(raw))})
    df["ts_text"] = raw[tcol].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["ts_text"], errors="coerce", format="mixed")
    if getattr(df["ts"].dt, "tz", None) is not None:          # tz-aware strings -> naive UTC
        df["ts"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    for c in CHANNELS:
        if c in raw.columns:
            num = pd.to_numeric(raw[c], errors="coerce")
            bad = int((num.isna() & raw[c].notna() & (raw[c].str.strip() != "")).sum())
            if bad:
                info["non_numeric"][c] = bad
            df[c] = num.astype(float)
        else:
            df[c] = np.nan
            info["missing_channels"].append(c)
    info["extra_columns"] = [c for c in raw.columns if c not in CHANNELS and c != tcol]
    return df, info


# ---------------------------------------------------------------- NMEA
_NMEA = re.compile(r"^\$(?P<body>[A-Z]{2}(?P<kind>RMC|GGA),[^*]*)\*(?P<cs>[0-9A-Fa-f]{2})\s*$")


def nmea_checksum(body: str) -> str:
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"{c:02X}"


def _dm_to_deg(value: str, hemi: str) -> float:
    if not value or not hemi:
        return np.nan
    try:
        v = float(value)
    except ValueError:
        return np.nan
    deg = int(v // 100)
    out = deg + (v - deg * 100) / 60.0
    return -out if hemi.upper() in ("S", "W") else out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_gps(path: Path) -> pd.DataFrame:
    """Parse ``logger_timestamp,NMEA`` lines. Bad checksums are kept but marked."""
    rows = []
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        lts_txt, sent = line.split(",", 1)
        m = _NMEA.match(sent.strip())
        lts = pd.to_datetime(lts_txt.strip(), errors="coerce")
        if not m or pd.isna(lts):
            rows.append({"lts": lts, "kind": "BAD", "cks_ok": False, "valid": False})
            continue
        body, kind = m.group("body"), m.group("kind")
        cks_ok = nmea_checksum(body) == m.group("cs").upper()
        f = body.split(",")
        rec = {"lts": lts, "kind": kind, "cks_ok": cks_ok, "valid": False,
               "gps_time": pd.NaT, "lat": np.nan, "lon": np.nan, "sats": np.nan, "hdop": np.nan, "alt": np.nan}
        try:
            if kind == "RMC":
                rec["valid"] = len(f) > 2 and f[2] == "A"
                rec["lat"], rec["lon"] = _dm_to_deg(f[3], f[4]), _dm_to_deg(f[5], f[6])
                if f[1] and len(f) > 9 and f[9]:
                    rec["gps_time"] = pd.to_datetime(f[9] + f[1][:6], format="%d%m%y%H%M%S", errors="coerce")
            else:  # GGA
                q = _f(f[6])
                rec["valid"] = bool(q >= 1)
                rec["lat"], rec["lon"] = _dm_to_deg(f[2], f[3]), _dm_to_deg(f[4], f[5])
                rec["sats"], rec["hdop"], rec["alt"] = _f(f[7]), _f(f[8]), _f(f[9])
        except IndexError:
            rec["valid"] = False
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- maintenance log
_LOG_LINE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+([A-Za-z]\w*)\s+(.*\S)\s*$")


def read_log(path: Path) -> pd.DataFrame:
    rows = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            m = _LOG_LINE.match(line)
            if m:
                rows.append({"date": pd.Timestamp(m.group(1)), "station": m.group(2).upper(), "text": m.group(3)})
    return pd.DataFrame(rows, columns=["date", "station", "text"])


# ---------------------------------------------------------------- the pack
def load_pack(root: str | Path) -> Pack:
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"error: pack folder not found: {root}")
    # tolerate a pack zipped with one wrapper folder
    if not (root / "data").is_dir():
        inner = [p for p in root.iterdir() if p.is_dir() and (p / "data").is_dir()]
        if len(inner) == 1:
            root = inner[0]
    if not (root / "data").is_dir():
        raise SystemExit(f"error: no data/ folder inside {root}")

    pack = Pack(root=root, stations=read_stations(root / "stations.csv"))
    for csv in sorted((root / "data").glob("*.csv")):
        sid = csv.stem.strip().upper()
        try:
            df, info = read_station_csv(csv)
        except Exception as e:  # unreadable file: report, carry on with the other stations
            pack.load_notes.append({"station": sid, "kind": "unreadable", "detail": str(e)})
            continue
        pack.data[sid] = df
        pack.load_notes.append({"station": sid, "kind": "csv_info", "detail": info})
        pack.gps[sid] = read_gps(root / "gps" / f"{csv.stem}.nmea")
    pack.log = read_log(root / "maintenance_log.md")
    return pack
