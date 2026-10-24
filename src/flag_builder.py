"""Turn candidates into flags: merge related stretches, apply the action to the working data,
attach volunteer-log evidence, and write flags.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .maintenance_checks import note_text, related_notes
from .models import FLAG_COLUMNS, Candidate, Flag
from .station import Station

MERGE_GAP_SAMPLES = 18     # stuck-family stretches closer than three hours are one fault
CHANNEL_LABEL = {"temp_c": "Temperature", "rh_pct": "Humidity", "pressure_hpa": "Pressure", "wind_kmh": "Wind speed",
                 "wind_dir_deg": "Wind direction", "rain_mm": "Rainfall", "ghi_wm2": "Sunlight"}


def _overlaps(a: Candidate, b: Candidate, slack: pd.Timedelta) -> bool:
    return a.channel == b.channel and a.start <= b.end + slack and b.start <= a.end + slack


def merge_candidates(cands: list[Candidate], st: Station) -> list[Candidate]:
    interval_min = st.interval
    slack = pd.Timedelta(minutes=interval_min)
    # 1. a spike/sentinel that is also outside the physical range: keep the richer description once
    spikes = [c for c in cands if c.family == "spike"]
    keep: list[Candidate] = []
    for c in cands:
        if c.reason == "impossible" and any(_overlaps(c, s, slack) and s.action != "keep" for s in spikes):
            for s in spikes:
                if _overlaps(c, s, slack) and "physical" not in s.text:
                    s.text += " It is also outside the physically possible range."
            continue
        keep.append(c)
    # 2. merge neighbouring stuck-family stretches with the same action on the same channel
    out: list[Candidate] = []
    for c in sorted(keep, key=lambda c: (c.channel, c.family, c.action, c.start)):
        copy = Candidate(**{**c.__dict__, "evidence": dict(c.evidence)})
        copy.evidence["parts"] = [(c.reason, c.text, dict(c.evidence), c.start, c.end)]
        if (out and c.family == "stuck" and out[-1].family == "stuck" and out[-1].channel == c.channel and out[-1].action == c.action
                and c.start <= out[-1].end + slack * (MERGE_GAP_SAMPLES + 1)):
            m = out[-1]
            m.end = max(m.end, c.end)
            if c.reason not in m.reason.split("+"):
                m.reason += "+" + c.reason
            m.evidence["parts"] += copy.evidence["parts"]
        else:
            out.append(copy)
    for m in out:
        parts = m.evidence["parts"]
        if len(parts) > 1 and m.action == "exclude":
            _compose_stuck(m, st)
    return sorted(out, key=lambda c: (c.start, c.channel))


def _compose_stuck(m: Candidate, st: Station):
    """One readable sentence for a fault that showed up as several symptoms (impossible values + frozen runs)."""
    parts = m.evidence["parts"]
    lab = CHANNEL_LABEL.get(m.channel, m.channel)
    reasons = {p[0] for p in parts}
    imp = [p for p in parts if p[0] == "impossible"]
    flat = [p for p in parts if p[0] == "flatline"]
    if imp and flat:
        n_imp = sum(p[2].get("rows", 0) for p in imp)
        vmax = max(p[2].get("vmax", np.nan) for p in imp)
        longest = max(p[2].get("rows", 0) for p in flat)
        moved = [p[2].get("peer_range") for p in flat if p[2].get("peer_range") is not None and not np.isnan(p[2].get("peer_range"))]
        hours = (m.end - m.start).total_seconds() / 3600 + st.interval / 60
        mv = f", while the median of the other stations moved by up to {max(moved):.0f} in those stretches" if moved else ""
        nxt_t = m.end + pd.Timedelta(minutes=st.interval)
        rec = ""
        if nxt_t in st.clean.index and not np.isnan(st.clean[m.channel].loc[nxt_t]):
            rec = f" The very next reading ({nxt_t:%d %b %H:%M}) drops back to {st.clean[m.channel].loc[nxt_t]:g}, so the sensor recovered by itself."
        m.issue = f"{lab} stuck at impossible values"
        m.text = (f"From {m.start:%d %b %H:%M} to {m.end:%d %b %H:%M} ({hours:.0f} h) {m.channel} has {n_imp} readings above its physical limit (up to {vmax:g}) "
                  f"and {len(flat)} stretches where the identical value repeats for up to {longest} readings{mv}.{rec}")
        m.evidence["rows"] = n_imp
    else:
        seen, texts = set(), []
        for p in parts:
            if p[1] not in seen:
                seen.add(p[1]); texts.append(p[1])
        m.text = " ".join(texts)
        if "flatline" in reasons:
            m.issue = f"{lab} frozen"
        elif "impossible" in reasons:
            m.issue = f"{lab} impossible"


def apply_and_flag(st: Station, cands: list[Candidate], log: pd.DataFrame) -> list[Flag]:
    """Apply each candidate's action to the station's working data and return the flags."""
    flags: list[Flag] = []
    for c in merge_candidates(cands, st):
        win = st.window_mask(c.start, c.end)
        n_rows = 0
        if c.action == "exclude":
            mask = c.evidence.get("mask", win)
            has = st.clean[c.channel].notna() if c.channel in st.clean else pd.Series(True, index=st.clean.index)
            n_rows = int((mask & has).sum())
            st.exclude(c.channel, mask)
        elif c.action == "correct" and "correction" in c.evidence:
            st.adjust[c.channel] = st.adjust[c.channel] + win.astype(float) * c.evidence["correction"]
            n_rows = int((win & st.clean[c.channel].notna()).sum())
        note = c.text
        if c.reason not in ("sentinel", "spike"):      # a one-sample glitch is not explained by a visit days away
            note += note_text(related_notes(log, st.sid, c.channel, c.start, c.end))
        f = Flag(st.sid, c.channel, c.start, c.end, c.issue, c.action, note, kind=c.reason,
                 evidence={**{k: v for k, v in c.evidence.items() if k not in ("mask", "parts")}, "n_rows": n_rows})
        flags.append(f)
    return flags


def finalize(all_flags: list[Flag]) -> pd.DataFrame:
    rows = [f.as_row() for f in sorted(all_flags, key=lambda f: (f.station_id, pd.Timestamp(f.start), f.channel))]
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)
