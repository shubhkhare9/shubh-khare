"""Cross-station comparison helpers.

Weather is shared across a mountain front, so a healthy station should track its neighbours once its
fixed local offset (elevation, siting) is removed. We therefore compare each station with the median of
the *other* stations after centring every station on its own median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import CHANNELS
from .station import Station

MIN_PEERS = 3  # need at least this many other stations reporting at a given time


class Peers:
    def __init__(self, stations: dict[str, Station]):
        self.stations = stations
        self.frames: dict[str, pd.DataFrame] = {}
        self.refresh()

    def refresh(self):
        """(Re)build the wide per-channel frames from the stations' currently usable data."""
        idx = None
        for s in self.stations.values():
            idx = s.clean.index if idx is None else idx.union(s.clean.index)
        for c in CHANNELS:
            self.frames[c] = pd.concat({sid: s.usable(c) for sid, s in self.stations.items()}, axis=1).reindex(idx)

    @property
    def enough(self) -> bool:
        return len(self.stations) >= MIN_PEERS + 1

    def others(self, sid: str, channel: str) -> pd.DataFrame:
        return self.frames[channel].drop(columns=[sid])

    def centred(self, channel: str) -> pd.DataFrame:
        f = self.frames[channel]
        return f - f.median()

    def reference(self, sid: str, channel: str) -> pd.Series:
        """Median of the other stations' centred series; NaN where fewer than MIN_PEERS report."""
        c = self.centred(channel).drop(columns=[sid])
        ref = c.median(axis=1)
        return ref.where(c.notna().sum(axis=1) >= MIN_PEERS)

    def residual(self, sid: str, channel: str) -> pd.Series:
        """Station minus its neighbours (both centred on their own median), indexed on the common grid."""
        c = self.centred(channel)
        return (c[sid] - self.reference(sid, channel)).rename(sid)

    def peer_median_raw(self, sid: str, channel: str) -> pd.Series:
        o = self.others(sid, channel)
        return o.median(axis=1).where(o.notna().sum(axis=1) >= MIN_PEERS)

    def peer_count_where(self, sid: str, channel: str, cond) -> pd.Series:
        """Number of other stations for which ``cond(series)`` is True at each time."""
        o = self.others(sid, channel)
        return cond(o).sum(axis=1)
