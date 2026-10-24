"""Use the volunteers' notebook as *evidence*, never as proof.

A note only gets attached to a flag when it concerns the same station, mentions the affected
part, and falls close in time. The data decides what is wrong; the note explains why.
"""
from __future__ import annotations

import re

import pandas as pd

# keyword -> channels the note could plausibly affect ("all" = timestamps / whole logger)
KEYWORDS = [
    (r"pressure|barometer", ["pressure_hpa"]),
    (r"humid|\brh\b|hygro", ["rh_pct"]),
    (r"anemometer|wind|cups|vane", ["wind_kmh", "wind_dir_deg"]),
    (r"rain|funnel|gauge|bucket", ["rain_mm"]),
    (r"shield|paint|post|thermometer|temperature|fence", ["temp_c"]),
    (r"pyranometer|sunlight|irradiance", ["ghi_wm2"]),
    (r"battery|reboot|rebooted|clock|logger|time", ["all"]),
    (r"website|sim\b|data plan|data stopped|came back|same day|duplicate|upload", ["all", "metadata"]),
]
# a note that mentions a repair/swap matters most for faults that *start* near it
LEAD_DAYS, LAG_DAYS = 1, 2


def note_channels(text: str) -> set[str]:
    t = text.lower()
    chans: set[str] = set()
    for pat, ch in KEYWORDS:
        if re.search(pat, t):
            chans.update(ch)
    return chans


def related_notes(log: pd.DataFrame, station: str, channel: str, start, end) -> list[pd.Series]:
    if log is None or log.empty:
        return []
    out = []
    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
    for _, r in log[log.station.isin([station, "ALL"])].iterrows():
        chans = note_channels(r.text)
        if not chans:
            continue
        if not (channel in chans or "all" in chans or channel == "all"):
            continue
        d0, d1 = r.date - pd.Timedelta(days=LEAD_DAYS), r.date + pd.Timedelta(days=LAG_DAYS)
        if s0 <= d1 and s1 >= d0:
            out.append(r)
    return out


def note_text(rows: list[pd.Series]) -> str:
    if not rows:
        return ""
    bits = []
    for r in rows[:2]:
        txt = r.text.strip()
        txt = txt if len(txt) <= 110 else txt[:107].rstrip() + "..."
        bits.append(f"{r.date:%d %b}: \"{txt}\"")
    return " Volunteer log " + "; ".join(bits) + "."


def review_log(log: pd.DataFrame, flags) -> list[dict]:
    """For each log entry, say whether the data show a matching problem (used in the report/README)."""
    rows = []
    if log is None or log.empty:
        return rows
    for _, r in log.iterrows():
        chans = note_channels(r.text)
        hit = [f for f in flags if f.station_id == r.station and f.action != "keep" and (
            f.channel in chans or f.channel == "all" or "all" in chans)
            and f.start <= r.date + pd.Timedelta(days=LAG_DAYS + 1) and f.end >= r.date - pd.Timedelta(days=LEAD_DAYS)]
        rows.append({"date": r.date, "station": r.station, "text": r.text, "matched_flags": len(hit),
                     "channels": sorted(chans)})
    return rows
