"""Bike state as a structure of arrays, with per-node lists (issue #98, spec §5.3, §6.5).

Every bike has a ``node``, a ``status``, a ``kind`` and a ``battery_km`` (NumPy arrays). Bikes
parked at a node are kept in an intrusive doubly linked list stored in arrays (``head[node]``,
``nxt[bike]``, ``prv[bike]``), so adding and removing a bike is O(1) and listing a node's bikes is
O(bikes there).

Storage: ``node``, ``status``, ``battery_km`` and ``n_bikes`` are NumPy arrays that are **views of
``array.array`` buffers** (``_node``, ``_status``, ``_batt``, ``_nb``). The hot path reads and
writes single elements through the buffers (several times faster than NumPy scalar access);
vectorised code (the user model's neighbourhood scan, checks, partitions) uses the NumPy views of
the same memory. The link arrays are plain Python lists.

Statuses:

* ``PARKED``: at a node, available to users and trucks;
* ``RESERVED``: chosen by a user who is walking to it (not available to others);
* ``RIDING``: in use (in-city trip, or arriving from outside);
* ``ON_TRUCK`` / ``DEPOT``: operators (M4);
* ``EXTERNAL``: outside the city, in the external reservoir.

The **external reservoir** (spec §6.5) holds the bikes outside the city: a trip to ``EXTERNAL`` puts
its bike there, a trip from ``EXTERNAL`` takes one out (of the preferred kind if possible).
Fleet conservation: the status counts always add up to the number of bikes; ``check`` verifies
every structure from scratch.
"""

from __future__ import annotations

from array import array

import numpy as np

PARKED, RESERVED, RIDING, ON_TRUCK, DEPOT, EXTERNAL = range(6)
STATUS_NAMES = ("parked", "reserved", "riding", "on_truck", "depot", "external")
MECHANICAL, ELECTRIC = 0, 1


class Fleet:
    """All bikes: ``n_city`` start in the city, ``n_external`` in the reservoir."""

    def __init__(self, n_nodes: int, kind: np.ndarray, battery_km: np.ndarray, range_km: float):
        n = len(kind)
        self.n_nodes = n_nodes
        self.n = n
        self.range_km = float(range_km)
        self.kind = np.asarray(kind, dtype=np.uint8).copy()
        self.kind_l: list[int] = self.kind.tolist()  # immutable, read on the hot path
        batt = np.asarray(battery_km, dtype=np.float64).copy()
        batt[self.kind == MECHANICAL] = 0.0
        self._batt = array("d", batt.tobytes())
        self._node = array("i", [-1]) * n
        self._status = array("B", [EXTERNAL]) * n
        self._nb = array("i", [0]) * n_nodes
        self.battery_km = np.frombuffer(self._batt, dtype=np.float64) if n else np.zeros(0)
        self.node = np.frombuffer(self._node, dtype=np.int32) if n else np.zeros(0, np.int32)
        self.status = np.frombuffer(self._status, dtype=np.uint8) if n else np.zeros(0, np.uint8)
        self.n_bikes = np.frombuffer(self._nb, dtype=np.int32)
        self.head: list[int] = [-1] * n_nodes
        self.nxt: list[int] = [-1] * n
        self.prv: list[int] = [-1] * n
        self.status_count = [0] * 6
        self.status_count[EXTERNAL] = n
        self.reservoir: tuple[list[int], list[int]] = ([], [])  # stacks per kind
        for b in range(n):
            self.reservoir[self.kind_l[b]].append(b)
        self.listeners: list = []  # partitions: f(bike, node, +1 parked / -1 left)

    # --- per-node lists -------------------------------------------------------------------
    def park(self, bike: int, node: int) -> None:
        """Put ``bike`` (not parked) at ``node``; status becomes PARKED."""
        st = self._status
        self.status_count[st[bike]] -= 1
        self.status_count[PARKED] += 1
        st[bike] = PARKED
        self._node[bike] = node
        h = self.head[node]
        self.nxt[bike] = h
        self.prv[bike] = -1
        if h >= 0:
            self.prv[h] = bike
        self.head[node] = bike
        self._nb[node] += 1
        for f in self.listeners:
            f(bike, node, 1)

    def unpark(self, bike: int, status: int) -> int:
        """Remove a parked ``bike`` from its node, give it ``status``; returns the node."""
        node = self._node[bike]
        p, q = self.prv[bike], self.nxt[bike]
        if p >= 0:
            self.nxt[p] = q
        else:
            self.head[node] = q
        if q >= 0:
            self.prv[q] = p
        self._nb[node] -= 1
        self.status_count[PARKED] -= 1
        self.status_count[status] += 1
        self._status[bike] = status
        for f in self.listeners:
            f(bike, node, -1)
        return node

    def set_status(self, bike: int, status: int) -> None:
        """Change the status of a bike that is not parked (use park/unpark for that)."""
        st = self._status
        self.status_count[st[bike]] -= 1
        self.status_count[status] += 1
        st[bike] = status

    def set_battery(self, bike: int, km: float) -> None:
        """Change a bike's charge; listeners see a parked bike leave and come back (band change)."""
        if self._status[bike] == PARKED:
            node = self._node[bike]
            for f in self.listeners:
                f(bike, node, -1)
            self._batt[bike] = km
            for f in self.listeners:
                f(bike, node, 1)
        else:
            self._batt[bike] = km

    def bikes_at(self, node: int) -> list[int]:
        out, b = [], self.head[node]
        while b >= 0:
            out.append(b)
            b = self.nxt[b]
        return out

    # --- external reservoir ---------------------------------------------------------------
    def take_external(self, preferred_kind: int) -> int:
        """A bike out of the reservoir (preferred kind first); -1 if empty. Status becomes RIDING."""
        stack = self.reservoir[preferred_kind]
        if not stack:
            stack = self.reservoir[1 - preferred_kind]
            if not stack:
                return -1
        bike = stack.pop()
        self.set_status(bike, RIDING)
        return bike

    def to_external(self, bike: int) -> None:
        self.set_status(bike, EXTERNAL)
        self._node[bike] = -1
        self.reservoir[self.kind_l[bike]].append(bike)

    # --- checks ---------------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        return dict(zip(STATUS_NAMES, self.status_count))

    def check(self) -> None:
        """Recompute every structure from scratch; raise AssertionError on any inconsistency."""
        st = np.bincount(self.status, minlength=6)
        assert st.tolist() == self.status_count, f"status counts {self.status_count} != {st.tolist()}"
        assert sum(self.status_count) == self.n, "fleet not conserved"
        parked = np.flatnonzero(self.status == PARKED)
        per_node = np.bincount(self.node[parked], minlength=self.n_nodes) if len(parked) else np.zeros(self.n_nodes, int)
        assert np.array_equal(per_node, self.n_bikes), "n_bikes does not match parked bikes"
        assert (self.n_bikes >= 0).all()
        seen = 0
        for node in np.flatnonzero(self.n_bikes):
            lst = self.bikes_at(int(node))
            assert len(lst) == self.n_bikes[node], f"list length at node {node}"
            assert all(self.node[b] == node and self.status[b] == PARKED for b in lst)
            seen += len(lst)
        assert seen == len(parked)
        res = sorted(self.reservoir[0] + self.reservoir[1])
        assert res == np.flatnonzero(self.status == EXTERNAL).tolist(), "reservoir does not match EXTERNAL bikes"
        assert all(self.kind[b] == k for k in (0, 1) for b in self.reservoir[k])
        e = self.kind == ELECTRIC
        assert (self.battery_km[e] >= -1e-9).all() and (self.battery_km[e] <= self.range_km + 1e-9).all()


def initial_nodes(rng: np.random.Generator, n: int, weights: np.ndarray, capacity: np.ndarray | None) -> np.ndarray:
    """``n`` nodes drawn with probability proportional to ``weights``, never above ``capacity``."""
    w = np.asarray(weights, dtype=np.float64).copy()
    if w.sum() <= 0:
        raise ValueError("no node with positive weight for the initial distribution")
    if capacity is None:
        return rng.choice(len(w), size=n, p=w / w.sum()).astype(np.int32)
    cap = np.asarray(capacity, dtype=np.int64)
    w[cap <= 0] = 0.0
    if cap[w > 0].sum() < n:
        raise ValueError(f"not enough docks for {n} bikes")
    out = np.empty(n, dtype=np.int32)
    used = np.zeros(len(w), dtype=np.int64)
    done = 0
    while done < n:  # draw, keep what fits, redraw the rest among nodes with free docks
        draw = rng.choice(len(w), size=n - done, p=w / w.sum())
        for node in draw:
            if used[node] < cap[node]:
                used[node] += 1
                out[done] = node
                done += 1
        w[used >= cap] = 0.0
    return out
