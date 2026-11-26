"""E-bike battery model (issue #99, spec §7.2).

* Riding consumes ``distance × consumption_km_per_km`` of range (never below zero).
* A user takes an e-bike only if its charge covers the trip plus ``reserve_km``, unless
  ``dead_ebike_rideable`` (then an empty e-bike is ridden like a classic one).
* Mechanical bikes are always usable and have ``battery_km = 0``.
* Battery **bands** (fractions of the full range, e.g. ``(0.2, 0.5)`` -> bands ``<20 %``, ``20–50 %``,
  ``>= 50 %``) are what partitions report.
* No charging except battery swaps by operators (M4).
"""

from __future__ import annotations

import bisect

import numpy as np

from bss_sim.config import EnergyConfig
from bss_sim.state import ELECTRIC


class Battery:
    def __init__(self, cfg: EnergyConfig, range_km: float):
        self.cfg = cfg
        self.range_km = float(range_km)
        self.cons = float(cfg.consumption_km_per_km)
        self.reserve = float(cfg.reserve_km)
        self.rideable_empty = bool(cfg.dead_ebike_rideable)
        self._edges_km = [e * self.range_km for e in cfg.battery_bands]
        self.n_bands = len(cfg.battery_bands) + 1

    def need_km(self, trip_km: float) -> float:
        """Charge a user requires before taking an e-bike for a trip of ``trip_km``."""
        return trip_km * self.cons + self.reserve

    def usable(self, kind: int, battery_km: float, need_km: float) -> bool:
        return kind != ELECTRIC or self.rideable_empty or battery_km >= need_km

    def after_ride(self, battery_km: float, trip_km: float) -> float:
        return max(0.0, battery_km - trip_km * self.cons)

    def band(self, battery_km: float) -> int:
        return bisect.bisect_right(self._edges_km, battery_km)

    def bands(self, battery_km: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self._edges_km), battery_km, side="right")

    def draw(self, rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
        """Initial charges, uniform between ``low`` and ``high`` fractions of the range."""
        return rng.uniform(low, high, n) * self.range_km
