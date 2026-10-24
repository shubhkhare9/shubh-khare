"""Per-station working state shared by all checks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import CHANNELS, Flag


@dataclass
class Station:
    sid: str
    lat: float                      # install-sheet latitude (may be NaN if the sheet has no row)
    lon: float
    elev: float
    interval: int                   # minutes
    name: str = ""
    raw: pd.DataFrame | None = None            # rows as logged (file order)
    gps: pd.DataFrame | None = None
    clean: pd.DataFrame | None = None          # gridded on the corrected UTC clock; one column per channel
    flags: list[Flag] = field(default_factory=list)
    excluded: dict[str, pd.Series] = field(default_factory=dict)   # channel -> bool mask on clean.index
    adjust: dict[str, pd.Series] = field(default_factory=dict)     # channel -> additive correction
    present: pd.Series | None = None           # True where a logger row exists on the grid
    solar_lat: float = np.nan       # latitude used for solar geometry (sheet, else GPS)
    solar_lon: float = np.nan
    notes: dict = field(default_factory=dict)  # free-form evidence for the report

    # -------------------------------------------------------------- helpers
    def init_masks(self):
        for c in CHANNELS:
            self.excluded[c] = pd.Series(False, index=self.clean.index)
            self.adjust[c] = pd.Series(0.0, index=self.clean.index)

    def add(self, flag: Flag):
        self.flags.append(flag)

    def exclude(self, channel: str, mask: pd.Series | np.ndarray):
        chans = CHANNELS if channel == "all" else [channel]
        for c in chans:
            self.excluded[c] = self.excluded[c] | pd.Series(np.asarray(mask, dtype=bool), index=self.clean.index)

    def window_mask(self, start, end) -> pd.Series:
        return pd.Series((self.clean.index >= start) & (self.clean.index <= end), index=self.clean.index)

    def usable(self, channel: str) -> pd.Series:
        """Channel values with already-excluded samples removed and corrections applied."""
        x = self.clean[channel] + self.adjust[channel]
        return x.where(~self.excluded[channel])

    @property
    def lat_for_solar(self) -> float:
        return self.solar_lat if not np.isnan(self.solar_lat) else self.lat

    @property
    def lon_for_solar(self) -> float:
        return self.solar_lon if not np.isnan(self.solar_lon) else self.lon
