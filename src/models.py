"""Shared constants and the Flag record."""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

# The seven logger channels in the pack format. Anything else in a CSV is ignored
# (and reported once as a metadata flag) rather than guessed at.
CHANNELS = ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "wind_dir_deg", "rain_mm", "ghi_wm2"]

FLAG_COLUMNS = ["station_id", "channel", "start_utc", "end_utc", "issue", "action", "note"]
TS_FORMAT = "%Y-%m-%d %H:%M"
ACTIONS = ("exclude", "correct", "keep")


@dataclass
class Flag:
    station_id: str
    channel: str                 # a column name, "all" or "metadata"
    start: pd.Timestamp
    end: pd.Timestamp
    issue: str
    action: str                  # exclude | correct | keep
    note: str
    # internal only (never written to flags.csv)
    kind: str = ""               # machine-readable check name, used for de-duplication/tests
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.action in ACTIONS, self.action

    def as_row(self) -> dict:
        return {
            "station_id": self.station_id,
            "channel": self.channel,
            "start_utc": pd.Timestamp(self.start).strftime(TS_FORMAT),
            "end_utc": pd.Timestamp(self.end).strftime(TS_FORMAT),
            "issue": self.issue,
            "action": self.action,
            "note": " ".join(str(self.note).split()),
        }


@dataclass
class Candidate:
    """A suspicious stretch found by one check, before it is merged into a flag.

    ``family`` decides what may be merged: runs of the "stuck" family (impossible values,
    frozen or saturated readings) next to each other are one fault; single spikes stay separate.
    """
    station_id: str
    channel: str
    start: pd.Timestamp
    end: pd.Timestamp
    reason: str            # impossible | sentinel | spike | flatline | saturation | dead | step | drift | gain | blocked | burst | night
    family: str            # stuck | spike | other
    issue: str
    action: str            # proposed action
    text: str              # one sentence of evidence
    evidence: dict = field(default_factory=dict)
