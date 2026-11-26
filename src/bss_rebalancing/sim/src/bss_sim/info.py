"""Information layer: what a controller may see (issue #127, spec §8).

Controllers never touch the event queue or the simulator's internals. ``Simulator.observe()``
returns an :class:`Observation`: copies and read-only views of the current state, plus the trips
in progress at the level of detail set by ``info.inflight_knowledge``:

==============  ======================================================================  ==========
level           controller sees                                                         realistic
==============  ======================================================================  ==========
``none``        nothing about trips in progress                                         pessimistic
``count``       how many bikes are in use                                               yes
``origin``      for each trip: where (node, station place) and when it started          yes (default)
``destination`` also where and when it will end                                         **no**: oracle
==============  ======================================================================  ==========

Trips that leave the city are not in progress for the operator: the bike is seen leaving the
service area (geofence) when the trip starts. Trips arriving **from** outside are in progress from
their start outside; their origin is ``EXTERNAL`` (node -1).

``destination`` is an **oracle** (the operator cannot know it) and needs ``info.allow_oracle``;
using it is recorded in the run metadata (``oracle_used``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from bss_sim.commands import Command
from bss_sim.partitions import PartitionCounts

LEVELS = ("none", "count", "origin", "destination")


@dataclass(frozen=True)
class InFlight:
    """Trips in progress (structure of arrays, sorted by bike id)."""

    count: int
    bike: np.ndarray | None = None        # int32  (origin, destination)
    kind: np.ndarray | None = None        # uint8  bike type
    o_place: np.ndarray | None = None     # int16  station place of the origin; EXTERNAL = n_places - 1
    o_node: np.ndarray | None = None      # int32  bike node where it started (-1 from outside)
    start_t: np.ndarray | None = None     # float64 clock time the trip started
    d_node: np.ndarray | None = None      # int32  destination (oracle only)
    end_t: np.ndarray | None = None       # float64 arrival time (oracle only)


@dataclass(frozen=True)
class TruckView:
    id: int
    stop: int                 # bike node of its current stop (where it left from, while driving)
    drive_node: int
    busy: bool
    current: Command | None
    eta: float                # when the current command ends (if busy)
    queued: int
    load: int                 # bikes on board
    ebikes_on_board: int
    stock: int                # charged batteries
    used: int                 # depleted batteries on board


@dataclass(frozen=True)
class Observation:
    t: float
    datetime: dt.datetime
    bikes: np.ndarray                     # int32 [nodes] parked bikes per node (copy)
    bike_node: np.ndarray                 # read-only views of every bike: node (-1 if not parked)...
    bike_status: np.ndarray               # ... status (bss_sim.state), kind and charge (km)
    bike_kind: np.ndarray
    bike_battery_km: np.ndarray
    partitions: dict[str, PartitionCounts]
    trucks: tuple[TruckView, ...]
    inflight: InFlight | None             # None when inflight_knowledge is "none"
    kpis: dict


def readonly(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.flags.writeable = False
    return v
