"""Targets per zone and capacitated pick-up and delivery routes (issue #151).

**Targets** (:func:`zone_targets`). For the next ``horizon_s`` the forecast gives each zone its
expected departures ``D`` and arrivals ``A``. A zone needs bikes for its demand and for its net
outflow, so the bikes the operator can place are shared in proportion to
``w = D + max(D - A, 0)`` (largest remainder), with at least ``min_per_zone`` where there is any
demand, and never above a zone's docks in docked mode.

**Route** (:func:`plan_route`). One truck with ``capacity`` and ``load`` starts at a stop. Zones
with a surplus are pick-ups, zones with a deficit drops. Greedy: at every step take the feasible
visit that moves the most bikes per second of driving plus service (pick up only with room on
board, drop only with bikes on board). Then 2-opt on the visiting order, keeping only reversals
that shorten the route and keep the load within ``[0, capacity]`` everywhere. The route stops when
its time would exceed ``time_budget_s``. Quantities are planned; the truck executes them with real
times and reports what it really did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def allocate(total: int, weights: np.ndarray, floor: np.ndarray | None = None, cap: np.ndarray | None = None) -> np.ndarray:
    """Integers summing to ``min(total, sum(cap))``, proportional to ``weights`` (largest remainder),
    each at least ``floor`` (heaviest first if the total cannot cover every floor) and at most ``cap``."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    n = len(w)
    cap = np.full(n, 1 << 40, dtype=np.int64) if cap is None else np.asarray(cap, dtype=np.int64)
    floor = np.zeros(n, dtype=np.int64) if floor is None else np.minimum(np.asarray(floor, dtype=np.int64), cap)
    total = int(min(total, cap.sum()))
    out = np.zeros(n, dtype=np.int64)
    if total <= 0:
        return out
    for i in np.argsort(-w, kind="stable"):          # floors, heaviest zones first
        out[i] = min(floor[i], total - out.sum())
    remaining = total - int(out.sum())
    while remaining > 0:                              # every pass places at least one bike
        free = out < cap
        ww = np.where(free, w, 0.0)
        if ww.sum() <= 0:
            ww = free.astype(np.float64)
        share = ww / ww.sum() * remaining
        add = np.minimum(np.floor(share).astype(np.int64), cap - out)
        if add.sum() == 0:                            # largest remainders, one bike each
            for i in np.argsort(-share, kind="stable"):
                if free[i] and remaining > 0:
                    out[i] += 1
                    remaining -= 1
            continue
        out += add
        remaining -= int(add.sum())
    return out


def zone_targets(departures: np.ndarray, arrivals: np.ndarray, total: int, min_per_zone: int = 0,
                 capacity: np.ndarray | None = None) -> np.ndarray:
    D = np.asarray(departures, dtype=np.float64)
    A = np.asarray(arrivals, dtype=np.float64)
    w = D + np.maximum(D - A, 0.0)
    floor = np.where(D > 0, min_per_zone, 0)
    return allocate(total, w, floor, capacity)


@dataclass(frozen=True)
class Visit:
    zone: int
    qty: int          # > 0 pick up, < 0 drop


def route_time(order: list[Visit], start: int, T: np.ndarray, service, overhead: float) -> float:
    """Driving plus service time of a route (``T`` indexed by zone ids; ``start`` is a zone id or -1
    for the truck's own position, row ``-1`` of ``T``)."""
    t, here = 0.0, start
    for v in order:
        t += T[here, v.zone] + overhead + abs(v.qty) * service(v.qty)
        here = v.zone
    return t


def _feasible(order: list[Visit], load: int, capacity: int) -> bool:
    for v in order:
        load += v.qty
        if load < 0 or load > capacity:
            return False
    return True


def plan_route(surplus: np.ndarray, T: np.ndarray, capacity: int, load: int, pick_s: float = 30.0,
               drop_s: float = 30.0, overhead_s: float = 60.0, time_budget_s: float = np.inf,
               two_opt: bool = True) -> list[Visit]:
    """Visits for one truck. ``surplus[z]`` > 0: bikes to take from zone ``z``; < 0: bikes to bring.
    ``T`` is ``[Z + 1, Z]``: rows are zones, the last row is the truck's current position."""
    left = np.asarray(surplus, dtype=np.int64).copy()
    here = -1
    elapsed = 0.0
    route: list[Visit] = []
    while True:
        best, best_score = None, 0.0
        room = capacity - load
        for z in np.flatnonzero(left).tolist():
            q = min(left[z], room) if left[z] > 0 else -min(-left[z], load)
            if q == 0:
                continue
            dt = T[here, z] + overhead_s + abs(q) * (pick_s if q > 0 else drop_s)
            score = abs(q) / max(dt, 1.0)
            if score > best_score:
                best, best_score = (z, q, dt), score
        if best is None:
            break
        z, q, dt = best
        if elapsed + dt > time_budget_s:
            break
        route.append(Visit(z, int(q)))
        left[z] -= q
        load += q
        here = z
        elapsed += dt
    if two_opt and len(route) > 2:
        route = _two_opt(route, T, capacity, load - sum(v.qty for v in route), pick_s, drop_s, overhead_s)
    return route


def _two_opt(route: list[Visit], T, capacity, load0, pick_s, drop_s, overhead_s) -> list[Visit]:
    def service(q):
        return pick_s if q > 0 else drop_s

    best = route_time(route, -1, T, service, overhead_s)
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route) + 1):
                cand = route[:i] + route[i:j][::-1] + route[j:]
                if not _feasible(cand, load0, capacity):
                    continue
                t = route_time(cand, -1, T, service, overhead_s)
                if t < best - 1e-9:
                    route, best, improved = cand, t, True
    return route
