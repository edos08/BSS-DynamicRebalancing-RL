"""Partitions: per-group counters kept up to date on every event (issue #102, spec §5.4).

A controller registers a partition of the bike nodes, ``node -> group`` (e.g. a 300 m grid; ``-1``
for nodes in no group). The simulator then updates, for every group:

* ``bikes``: bikes parked now; ``ebikes``: parked e-bikes per battery band (``[G, bands]``);
* ``requests``: requests starting there; ``failures``: failed requests per reason (``[G, reasons]``);
* ``departures`` / ``arrivals``: trips that started / ended there (cumulative since ``reset``).

Updates are O(1) per event and per partition; reading a partition is O(groups). Counters are
Python lists internally (scalar updates are cheaper than on NumPy arrays); ``counts()`` returns
NumPy copies. ``recompute`` rebuilds the state counters from the fleet for tests and checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bss_sim.state import ELECTRIC, PARKED
from bss_sim.users import REASONS


@dataclass(frozen=True)
class PartitionCounts:
    bikes: np.ndarray        # int64 [G]
    ebikes: np.ndarray       # int64 [G, bands]
    requests: np.ndarray     # int64 [G]
    failures: np.ndarray     # int64 [G, reasons] (columns: bss_sim.users.REASONS)
    departures: np.ndarray   # int64 [G]
    arrivals: np.ndarray     # int64 [G]


class Partition:
    def __init__(self, name: str, node_to_group: np.ndarray, n_nodes: int, battery):
        g = np.asarray(node_to_group)
        if g.shape != (n_nodes,):
            raise ValueError(f"partition {name!r}: node_to_group needs one entry per bike node ({n_nodes})")
        if not np.issubdtype(g.dtype, np.integer) or (g < -1).any():
            raise ValueError(f"partition {name!r}: groups must be integers >= 0 (-1 for no group)")
        self.name = name
        self.n_groups = int(g.max()) + 1 if (g >= 0).any() else 0
        G = self.n_groups
        self.node_to_group = g.astype(np.int64)
        self._g: list[int] = np.where(g >= 0, g, G).tolist()  # -1 -> spill slot G, dropped on read
        self.battery = battery
        self.n_bands = battery.n_bands
        self.fleet = None
        self.reset()

    def bind(self, fleet) -> None:
        """Attach to a (new) fleet: state counters filled from it, event counters zeroed."""
        self.reset()
        self.fleet = fleet
        self.fill_from(fleet)
        fleet.listeners.append(self.on_park)

    def reset(self) -> None:
        G1, B = self.n_groups + 1, self.n_bands
        self.bikes = [0] * G1
        self.ebikes = [[0] * B for _ in range(G1)]
        self.requests = [0] * G1
        self.failures = [[0] * len(REASONS) for _ in range(G1)]
        self.departures = [0] * G1
        self.arrivals = [0] * G1

    # --- hooks ------------------------------------------------------------------------
    def on_park(self, bike: int, node: int, sign: int) -> None:
        fleet = self.fleet
        g = self._g[node]
        self.bikes[g] += sign
        if fleet.kind_l[bike] == ELECTRIC:
            self.ebikes[g][self.battery.band(fleet.battery_km[bike])] += sign

    def fill_from(self, fleet) -> None:
        """Set the state counters (bikes, e-bike bands) from the fleet as it is now."""
        G1 = self.n_groups + 1
        self.bikes = [0] * G1
        self.ebikes = [[0] * self.n_bands for _ in range(G1)]
        g, kind, batt = self._g, fleet.kind_l, fleet.battery_km
        for b in np.flatnonzero(fleet.status == PARKED).tolist():
            k = g[int(fleet.node[b])]
            self.bikes[k] += 1
            if kind[b] == ELECTRIC:
                self.ebikes[k][self.battery.band(batt[b])] += 1

    # --- reading ----------------------------------------------------------------------
    def counts(self) -> PartitionCounts:
        G = self.n_groups

        def arr(x):
            return np.array(x[:G], dtype=np.int64)
        return PartitionCounts(arr(self.bikes), arr(self.ebikes).reshape(G, self.n_bands), arr(self.requests),
                               arr(self.failures).reshape(G, len(REASONS)), arr(self.departures),
                               arr(self.arrivals))

    def recompute_state(self, fleet) -> tuple[np.ndarray, np.ndarray]:
        """Bikes and e-bikes per band per group, from scratch (for checks)."""
        G = self.n_groups
        parked = np.flatnonzero(fleet.status == PARKED)
        g = self.node_to_group[fleet.node[parked]]
        keep = g >= 0
        bikes = np.bincount(g[keep], minlength=G)
        e = keep & (fleet.kind[parked] == ELECTRIC)
        bands = self.battery.bands(fleet.battery_km[parked[e]])
        ebikes = np.zeros((G, self.n_bands), dtype=np.int64)
        np.add.at(ebikes, (g[e], bands), 1)
        return bikes, ebikes
