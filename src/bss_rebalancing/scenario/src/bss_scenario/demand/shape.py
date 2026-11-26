"""Hourly OD demand shape with calendar exposure and empirical-Bayes smoothing (issue #79, spec §6.1).

``rates[g, h, o, d]`` = expected trips from place ``o`` to place ``d`` starting in hour ``h`` of a
day of group ``g`` whose volume factor is 1. The generator multiplies by the day's volume
factor.

Estimation over the fit-period days:

* **Counts** ``N[g, h, o, d]`` of valid trips.
* **Exposure** ``E[g]`` = sum over the group's calendar days of the month volume factor. Days with
  no trips still count (this fixes the legacy bug of dividing by "days with at least one trip").
* **Origin totals per hour** are smoothed towards the group's hourly profile ``π[g, h]`` with a
  Dirichlet prior of strength ``beta`` (pseudo-trips), keeping each origin's daily total:
  ``Ñ_o[g,h,o] = N_o[g,o] · (N_o[g,h,o] + beta·π[g,h]) / (N_o[g,o] + beta)``.
* **Destinations** are smoothed towards ``p0(d | o, time band)``, pooled over all groups and the
  hours of a coarse band (night, morning, midday, evening, late), with strength ``alpha``:
  ``p(d | o,g,h) = (N[g,h,o,d] + alpha·p0[b(h),o,d]) / (N_o[g,h,o] + alpha)``.
  ``p0`` is itself smoothed towards the band's overall destination distribution with
  ``gamma`` = 1 pseudo-trip.
* ``rates = Ñ_o / E[g] · p``. ``EXTERNAL → EXTERNAL`` is always 0.

Large counts are left almost untouched; pairs seen once or twice are pulled towards the pooled
pattern.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from bss_scenario.demand.calendar import day_group, group_labels
from bss_scenario.demand.places import Places
from bss_scenario.demand.volume import VolumeModel

#: Hour -> coarse time band used for the destination prior.
TIME_BANDS = np.array([0] * 6 + [1] * 4 + [2] * 6 + [3] * 4 + [4] * 4)  # 0-6, 6-10, 10-16, 16-20, 20-24
GAMMA = 1.0


@dataclass(frozen=True)
class ShapeConfig:
    scheme: str = "day_types"
    alpha: float = 1024.0  # chosen on Cambridge Apr-Nov 2024 held-out days (#80)
    beta: float = 512.0
    holidays_as_sunday: bool = True

    def __post_init__(self) -> None:
        group_labels(self.scheme)
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be >= 0")


@dataclass(frozen=True)
class ShapeModel:
    rates: np.ndarray            # float32 [G, 24, P, P]
    groups: tuple[str, ...]
    places: Places
    exposure: np.ndarray         # float64 [G]
    config: ShapeConfig
    counts_total: int
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, rates=self.rates, exposure=self.exposure,
                            groups=np.array(self.groups), station_ids=np.array(self.places.station_ids),
                            meta=np.array(json.dumps({"scheme": self.config.scheme, "alpha": self.config.alpha,
                                                      "beta": self.config.beta,
                                                      "holidays_as_sunday": self.config.holidays_as_sunday,
                                                      "counts_total": self.counts_total, **self.meta})))

    @classmethod
    def load(cls, path: str | Path) -> ShapeModel:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            config = ShapeConfig(meta.pop("scheme"), meta.pop("alpha"), meta.pop("beta"), meta.pop("holidays_as_sunday"))
            return cls(z["rates"], tuple(z["groups"].tolist()), Places(tuple(z["station_ids"].tolist())),
                       z["exposure"], config, meta.pop("counts_total"), meta)


def count_tensor(trips: pl.DataFrame, places: Places, scheme: str, holidays_as_sunday: bool = True) -> np.ndarray:
    """``N[g, h, o, d]`` from trips with ``o_place``, ``d_place`` and tz-aware local ``start_time``."""
    labels = group_labels(scheme)
    dates = trips["start_time"].dt.date()
    unique = dates.unique().to_list()
    group_of = {d: day_group(d, scheme, holidays_as_sunday) for d in unique}
    g = dates.replace_strict(group_of, return_dtype=pl.Int64).to_numpy()
    h = trips["start_time"].dt.hour().to_numpy().astype(np.int64)
    o = trips["o_place"].to_numpy().astype(np.int64)
    d = trips["d_place"].to_numpy().astype(np.int64)
    P = places.n
    flat = ((g * 24 + h) * P + o) * P + d
    return np.bincount(flat, minlength=len(labels) * 24 * P * P).reshape(len(labels), 24, P, P)


def exposure(days: list[dt.date], volume: VolumeModel | None, scheme: str, holidays_as_sunday: bool = True) -> np.ndarray:
    """Sum of month volume factors over each group's calendar days (1 per day without a model)."""
    out = np.zeros(len(group_labels(scheme)))
    for day in days:
        out[day_group(day, scheme, holidays_as_sunday)] += volume.factor(day) if volume else 1.0
    return out


def smooth_rates(N: np.ndarray, E: np.ndarray, alpha: float, beta: float, external: int) -> np.ndarray:
    """Apply the origin and destination smoothing described in the module docstring."""
    N = N.astype(np.float64)
    G, H, P, _ = N.shape
    No_h = N.sum(axis=3)                                   # [G, H, P]
    No = No_h.sum(axis=1, keepdims=True)                   # [G, 1, P]
    total_gh = No_h.sum(axis=2)                            # [G, H]
    pi = total_gh / np.maximum(total_gh.sum(axis=1, keepdims=True), 1e-300)  # [G, H]
    No_s = No * (No_h + beta * pi[:, :, None]) / np.maximum(No + beta, 1e-300)  # [G, H, P]

    # destination prior per band, pooled over groups and the band's hours
    C = np.zeros((TIME_BANDS.max() + 1, P, P))
    np.add.at(C, TIME_BANDS, N.sum(axis=0))                # [B, P, P]
    mask = np.ones((P, P))
    mask[external, external] = 0.0
    dest = C.sum(axis=1)                                   # [B, P] destination totals per band
    q = dest[:, None, :] * mask[None]                      # [B, P, P], no EXTERNAL -> EXTERNAL
    q = q / np.maximum(q.sum(axis=2, keepdims=True), 1e-300)
    p0 = (C + GAMMA * q) / np.maximum(C.sum(axis=2, keepdims=True) + GAMMA, 1e-300)

    prior = p0[TIME_BANDS][None]                           # [1, H, P, P]
    p = (N + alpha * prior) / np.maximum(No_h[..., None] + alpha, 1e-300)
    p = p * mask
    p = p / np.maximum(p.sum(axis=3, keepdims=True), 1e-300)
    rates = (No_s / np.maximum(E, 1e-300)[:, None, None])[..., None] * p
    return rates.astype(np.float32)


def fit_shape(trips: pl.DataFrame, places: Places, days: list[dt.date], volume: VolumeModel | None,
              config: ShapeConfig = ShapeConfig()) -> ShapeModel:
    """Fit the shape on ``trips`` (already restricted to ``days``) with exposure over ``days``."""
    N = count_tensor(trips, places, config.scheme, config.holidays_as_sunday)
    E = exposure(days, volume, config.scheme, config.holidays_as_sunday)
    rates = smooth_rates(N, E, config.alpha, config.beta, places.external)
    return ShapeModel(rates, group_labels(config.scheme), places, E, config, int(N.sum()))
