"""Ride-time model fitted from recorded trips (issue #81, spec §6.4 and §8.1).

Replaces the legacy fixed truncated Gaussian speed (15 ± 5 km/h).

* For trips between **different** in-city places, the **speed** is the bike-network distance
  between the two stations' nodes divided by the recorded duration. Its distribution is stored as
  a quantile table per (bike type, distance band). Short trips are slower (starts, stops, detours),
  and e-bikes are faster.
* For **round trips** (same place), distance says nothing, so the **duration** itself is stored as
  a quantile table per bike type.

Sampling uses the inverse CDF. ``cdf(t, dist, kind)`` = P(ride time ≤ t) = 1 − F_speed(dist / t) is
what the in-flight arrival forecast needs (#29). The file ``demand/ride_time.npz`` is read by the
simulator, which has its own small reader with the same semantics, so ``bss_sim`` never imports
``bss_scenario``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

BIKE_TYPES = ("classic", "electric")
DEFAULT_BANDS_M = (0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0, np.inf)
LEVELS = np.linspace(0.0, 1.0, 101)
MIN_TRIPS_PER_CELL = 200


@dataclass(frozen=True)
class RideTimeModel:
    bands_m: np.ndarray        # [B + 1] band edges (metres)
    levels: np.ndarray         # [Q] quantile levels in [0, 1]
    speed_q: np.ndarray        # [2, B, Q] speed quantiles (m/s), increasing along Q
    round_q: np.ndarray        # [2, Q] round-trip duration quantiles (s)
    counts: np.ndarray         # [2, B] trips used per cell (after pooling)
    round_counts: np.ndarray   # [2]
    max_duration_s: float = 10800.0  # no ride is longer (the cleaning limit; the fitted trips respect it)

    def band(self, dist_m: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.bands_m, dist_m, side="right") - 1, 0, len(self.bands_m) - 2)

    def sample(self, dist_m, kind, rng: np.random.Generator) -> np.ndarray:
        """Ride times (s). ``kind`` 0 = classic, 1 = electric; ``dist_m`` 0 means a round trip."""
        dist_m = np.asarray(dist_m, dtype=float)
        kind = np.asarray(kind, dtype=np.int64)
        u = rng.random(dist_m.shape)
        out = np.empty(dist_m.shape)
        rt = dist_m <= 0
        for k in (0, 1):
            m = rt & (kind == k)
            out[m] = np.interp(u[m], self.levels, self.round_q[k])
        b = self.band(dist_m)
        for k in (0, 1):
            for j in range(self.speed_q.shape[1]):
                m = ~rt & (kind == k) & (b == j)
                if m.any():
                    with np.errstate(divide="ignore"):
                        out[m] = dist_m[m] / np.interp(u[m], self.levels, self.speed_q[k, j])
        # the lowest speed quantile can be 0 m/s: cap at the longest allowed ride (#138)
        return np.minimum(out, self.max_duration_s)

    def cdf(self, t_s, dist_m, kind) -> np.ndarray:
        """P(ride time <= t) for each (t, distance, kind)."""
        t_s, dist_m = np.broadcast_arrays(np.asarray(t_s, float), np.asarray(dist_m, float))
        kind = np.broadcast_to(np.asarray(kind, np.int64), t_s.shape)
        out = np.zeros(t_s.shape)
        rt = dist_m <= 0
        b = self.band(dist_m)
        for k in (0, 1):
            m = rt & (kind == k)
            out[m] = np.interp(t_s[m], self.round_q[k], self.levels, left=0.0, right=1.0)
            for j in range(self.speed_q.shape[1]):
                m = ~rt & (kind == k) & (b == j) & (t_s > 0)
                if m.any():
                    v = dist_m[m] / t_s[m]  # needed speed: ride is over if actual speed >= v
                    out[m] = 1.0 - np.interp(v, self.speed_q[k, j], self.levels, left=0.0, right=1.0)
        out[t_s >= self.max_duration_s] = 1.0
        return out

    def save(self, path: str | Path) -> None:
        np.savez(path, bands_m=self.bands_m, levels=self.levels, speed_q=self.speed_q, round_q=self.round_q,
                 counts=self.counts, round_counts=self.round_counts, bike_types=np.array(BIKE_TYPES),
                 max_duration_s=np.float64(self.max_duration_s))

    @classmethod
    def load(cls, path: str | Path) -> RideTimeModel:
        with np.load(path) as z:
            cap = float(z["max_duration_s"]) if "max_duration_s" in z.files else 10800.0
            return cls(z["bands_m"], z["levels"], z["speed_q"], z["round_q"], z["counts"], z["round_counts"], cap)


def fit_ride_time(trips: pl.DataFrame, place_node: np.ndarray, bike_dist: np.ndarray, external: int,
                  bands_m=DEFAULT_BANDS_M, min_trips: int = MIN_TRIPS_PER_CELL,
                  max_duration_s: float = 10800.0) -> RideTimeModel:
    """Fit from valid trips with ``o_place``/``d_place``, ``bike_type`` and ``duration_s``.

    ``place_node[p]`` is the bike node of in-city place ``p``. Cells with fewer than ``min_trips``
    trips borrow from the neighbouring band of the same bike type (then from the other type).
    """
    t = trips.filter((pl.col("o_place") != external) & (pl.col("d_place") != external) & (pl.col("duration_s") > 0))
    o = t["o_place"].to_numpy()
    d = t["d_place"].to_numpy()
    dur = t["duration_s"].to_numpy().astype(float)
    kind = (t["bike_type"] == "electric").to_numpy().astype(np.int64)
    dist = bike_dist[place_node[o], place_node[d]].astype(float)
    rt = o == d
    bands = np.asarray(bands_m, dtype=float)
    B = len(bands) - 1
    band = np.clip(np.searchsorted(bands, dist, side="right") - 1, 0, B - 1)
    speed = np.where(rt, np.nan, dist / dur)

    samples = [[speed[~rt & (kind == k) & (band == j)] for j in range(B)] for k in (0, 1)]
    counts = np.array([[len(s) for s in row] for row in samples])
    speed_q = np.empty((2, B, len(LEVELS)))
    for k in (0, 1):
        for j in range(B):
            pool = samples[k][j]
            radius = 1
            while len(pool) < min_trips and radius < B:
                near = [samples[k][i] for i in (j - radius, j + radius) if 0 <= i < B]
                pool = np.concatenate([pool, *near])
                radius += 1
            if len(pool) < min_trips:
                pool = np.concatenate([pool, *samples[1 - k]])
            if len(pool) == 0:
                raise ValueError("no trips to fit ride speeds")
            speed_q[k, j] = np.quantile(pool, LEVELS)
    round_q = np.empty((2, len(LEVELS)))
    round_counts = np.array([int((rt & (kind == k)).sum()) for k in (0, 1)])
    for k in (0, 1):
        pool = dur[rt & (kind == k)]
        if len(pool) < min_trips:
            pool = dur[rt]
        round_q[k] = np.quantile(pool, LEVELS) if len(pool) else np.full(len(LEVELS), 600.0)
    return RideTimeModel(bands, LEVELS.copy(), speed_q, round_q, counts, round_counts, float(max_duration_s))


def summary(model: RideTimeModel) -> list[dict]:
    """Median and 10-90 % speeds (km/h) per bike type and band."""
    rows = []
    i10, i50, i90 = (int(np.argmin(np.abs(model.levels - q))) for q in (0.1, 0.5, 0.9))
    for k, name in enumerate(BIKE_TYPES):
        for j in range(model.speed_q.shape[1]):
            lo, hi = model.bands_m[j], model.bands_m[j + 1]
            rows.append({"bike_type": name, "band_m": f"{lo:.0f}-{hi:.0f}", "trips": int(model.counts[k, j]),
                         "p10_kmh": round(model.speed_q[k, j, i10] * 3.6, 1),
                         "median_kmh": round(model.speed_q[k, j, i50] * 3.6, 1),
                         "p90_kmh": round(model.speed_q[k, j, i90] * 3.6, 1)})
        rows.append({"bike_type": name, "band_m": "round trip", "trips": int(model.round_counts[k]),
                     "median_min": round(model.round_q[k, i50] / 60, 1)})
    return rows
