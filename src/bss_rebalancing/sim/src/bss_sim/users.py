"""User model: which bike a user takes, if any (issue #100, spec §7.1).

:class:`WalkingUsers` (default). On a request at node ``o``:

1. Look through the nodes within ``walk_radius_m`` in order of walking distance (the bundle's CSR,
   self first) for the **nearest usable bike**: a classic bike, or an e-bike whose charge covers the
   trip plus the reserve (see :mod:`bss_sim.energy`).
2. If the user prefers the other bike type, they walk up to ``kind_tolerance_m`` further for it
   (at the same node the preferred type is always taken).
3. The user accepts the walk ``w`` with probability ``exp(-w / walk_beta_m)``.

Outcomes: a bike and the walk in metres, or a failure reason:
``NO_BIKE_IN_RANGE`` (no bike at all within reach), ``LOW_BATTERY_ONLY`` (only e-bikes without
enough charge), ``DECLINED_WALK`` (a bike, but too far for this user).

Any object with the same ``choose`` method can replace it (``Simulator(user_model=...)``).
"""

from __future__ import annotations

import math

import numpy as np

from bss_sim.state import ELECTRIC

SERVED, NO_BIKE_IN_RANGE, LOW_BATTERY_ONLY, DECLINED_WALK, EXTERNAL_EMPTY, NO_DOCK = range(6)
REASONS = ("served", "no_bike_in_range", "low_battery_only", "declined_walk", "external_empty", "no_dock")
REQUEST_FAILURES = (NO_BIKE_IN_RANGE, LOW_BATTERY_ONLY, DECLINED_WALK, EXTERNAL_EMPTY)


class WalkingUsers:
    def __init__(self, walk_indptr: np.ndarray, walk_indices: np.ndarray, walk_dist: np.ndarray,
                 radius_m: float, beta_m: float, kind_tolerance_m: float = 0.0, rideable_empty: bool = False):
        # keep only the neighbours within the radius (the CSR is sorted by distance)
        n = len(walk_indptr) - 1
        keep = np.empty(n, dtype=np.int64)
        for v in range(n):
            a, b = walk_indptr[v], walk_indptr[v + 1]
            keep[v] = np.searchsorted(walk_dist[a:b], radius_m, side="right")
        sel = np.concatenate([np.arange(walk_indptr[v], walk_indptr[v] + keep[v]) for v in range(n)])
        indptr = np.concatenate([[0], np.cumsum(keep)])
        self.start: list[int] = indptr[:-1].tolist()
        self.end: list[int] = indptr[1:].tolist()
        self.indices_np = np.asarray(walk_indices)[sel].astype(np.int64)
        self.indices: list[int] = self.indices_np.tolist()  # Python lists: scalar reads on the hot path
        self.dist: list[float] = np.asarray(walk_dist, dtype=np.float64)[sel].tolist()
        self.beta = float(beta_m)
        self.tol = float(kind_tolerance_m)
        self.rideable_empty = bool(rideable_empty)
        self.fleet = None

    def bind(self, fleet) -> None:
        """Called by the simulator at every reset with the current fleet."""
        self.fleet = fleet

    def choose(self, o: int, preferred: int, need_km: float, u: float) -> tuple[int, float, int]:
        """``(bike, walk_m, SERVED)`` or ``(-1, walk_m, reason)``; ``u`` is the user's uniform draw."""
        f = self.fleet
        nxt, kind, batt = f.nxt, f.kind_l, f._batt
        ok_empty = self.rideable_empty
        first, first_w = -1, 0.0
        if self.tol == 0.0:  # fast path: a usable bike at the origin node itself (walk 0)
            bk = f.head[o]
            while bk >= 0:
                k = kind[bk]
                if k != ELECTRIC or ok_empty or batt[bk] >= need_km:
                    if k == preferred:
                        return bk, 0.0, SERVED
                    if first < 0:
                        first = bk
                bk = nxt[bk]
            if first >= 0:
                return first, 0.0, SERVED
        a = self.start[o]
        nb = self.indices_np[a:self.end[o]]
        hits = f.n_bikes[nb].nonzero()[0]
        if len(hits) == 0:
            return -1, 0.0, NO_BIKE_IN_RANGE
        head, dist, nb_l = f.head, self.dist, self.indices
        for j in hits.tolist():
            w = dist[a + j]
            if first >= 0 and w > first_w + self.tol:
                break
            bk = head[nb_l[a + j]]
            while bk >= 0:
                k = kind[bk]
                if k != ELECTRIC or ok_empty or batt[bk] >= need_km:
                    if k == preferred:
                        return self._accept(bk, w, u)
                    if first < 0:
                        first, first_w = bk, w
                bk = nxt[bk]
        if first < 0:
            return -1, 0.0, LOW_BATTERY_ONLY
        return self._accept(first, first_w, u)

    def _accept(self, bike: int, w: float, u: float) -> tuple[int, float, int]:
        if w > 0.0 and u >= math.exp(-w / self.beta):
            return -1, w, DECLINED_WALK
        return bike, w, SERVED
