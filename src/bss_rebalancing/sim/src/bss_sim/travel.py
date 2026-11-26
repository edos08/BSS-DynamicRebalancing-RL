"""Truck travel times on the drive network (issue #121, spec §9 "Travel time").

A :class:`TravelModel` gives the driving time and distance between two **drive nodes** for a
departure at clock time ``t``. Paths are the free-flow fastest paths precomputed in the bundle
(``drive_time.npy``, ``drive_dist.npy``); traffic changes the time spent on them, not the route.

* :class:`FreeFlow`: ``scale x drive_time`` (``scale`` calibrates speed-limit free flow against a
  routing service's no-traffic times; 1 by default).
* :class:`CongestionFactors` (default once fitted, #57): the path's free-flow time on each road
  class times a slowdown factor for the departure's traffic day type and hour,
  ``sum_c f[day, hour, c] x t_c[a, b]`` (``drive_time_by_class.npy``). The factors come from the
  bundle's ``traffic/factors.json`` or from the configuration.

Optional log-normal noise (mean 1, ``noise_sigma``) is drawn from the run's ``travel`` stream.
Traffic day types: weekday, Saturday, Sunday; holidays count as Sundays.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from bss_sim.calendar import holidays

TRAFFIC_DAY_TYPES = ("weekday", "saturday", "sunday")


def traffic_day_type(date: dt.date) -> int:
    if date in holidays(date.year) or date.weekday() == 6:
        return 2
    return 1 if date.weekday() == 5 else 0


class FreeFlow:
    name = "freeflow"

    def __init__(self, drive_time: np.ndarray, drive_dist: np.ndarray, scale: float = 1.0,
                 noise_sigma: float = 0.0):
        self._time = np.asarray(drive_time)
        self._dist = np.asarray(drive_dist)
        self.scale = float(scale)
        self.sigma = float(noise_sigma)
        self.rng: np.random.Generator | None = None
        self.day0 = dt.date(2024, 1, 1)

    def bind(self, rng: np.random.Generator, day0: dt.date) -> None:
        """Called at every reset: the run's ``travel`` stream and the date of clock 0."""
        self.rng = rng
        self.day0 = day0

    def distance(self, a: int, b: int) -> float:
        return float(self._dist[a, b])

    def base_time(self, a: int, b: int, t: float) -> float:
        return self.scale * float(self._time[a, b])

    def time(self, a: int, b: int, t: float) -> float:
        """Driving time (s) from drive node ``a`` to ``b`` leaving at clock time ``t``."""
        base = self.base_time(a, b, t)
        if self.sigma > 0.0 and base > 0.0:
            s = self.sigma
            base *= math.exp(self.rng.normal(-0.5 * s * s, s))
        return base


class CongestionFactors(FreeFlow):
    name = "congestion"

    def __init__(self, drive_time: np.ndarray, drive_dist: np.ndarray, time_by_class: np.ndarray,
                 factors: np.ndarray, scale: float = 1.0, noise_sigma: float = 0.0):
        super().__init__(drive_time, drive_dist, scale, noise_sigma)
        f = np.asarray(factors, dtype=np.float64)
        C = time_by_class.shape[0]
        if f.shape != (len(TRAFFIC_DAY_TYPES), 24, C):
            raise ValueError(f"congestion factors must have shape (3, 24, {C}), got {f.shape}")
        if not (f > 0).all():
            raise ValueError("congestion factors must be > 0")
        self.factors = f
        self._by_class = time_by_class
        self._day_cache: dict[int, int] = {}

    def _day_type(self, day: int) -> int:
        d = self._day_cache.get(day)
        if d is None:
            d = self._day_cache[day] = traffic_day_type(self.day0 + dt.timedelta(days=day))
        return d

    def base_time(self, a: int, b: int, t: float) -> float:
        day, sec = divmod(t, 86400.0)
        f = self.factors[self._day_type(int(day)), int(sec // 3600)]
        return self.scale * float(np.dot(f, self._by_class[:, a, b]))


def load_factors(path: str | Path) -> np.ndarray:
    """``traffic/factors.json``: ``{"day_types": [...], "road_classes": [...], "factors": [3][24][C]}``."""
    data = json.loads(Path(path).read_text())
    if tuple(data.get("day_types", ())) != TRAFFIC_DAY_TYPES:
        raise ValueError(f"{path}: day_types must be {list(TRAFFIC_DAY_TYPES)}")
    return np.asarray(data["factors"], dtype=np.float64)


def make_travel(scenario, cfg) -> FreeFlow:
    """The travel model for a ``TravelConfig``: ``auto`` uses fitted factors when the bundle has them."""
    factors = None
    if cfg.factors is not None:
        factors = np.asarray(cfg.factors, dtype=np.float64)
    elif cfg.model in ("auto", "congestion") and scenario.traffic_factors_path() is not None:
        factors = load_factors(scenario.traffic_factors_path())
    if cfg.model == "congestion" and factors is None:
        raise ValueError("travel.model 'congestion' needs travel.factors or traffic/factors.json in the bundle (#57)")
    t, d = np.asarray(scenario.drive_time), np.asarray(scenario.drive_dist)
    if factors is None or cfg.model == "freeflow":
        return FreeFlow(t, d, cfg.scale, cfg.noise_sigma)
    return CongestionFactors(t, d, np.asarray(scenario.drive_time_by_class()), factors, cfg.scale, cfg.noise_sigma)
