"""The simulator: discrete-event loop over demand, users and the fleet (spec §5).

```python
sim = Simulator.from_scenario("data/scenarios/cambridge-2024", SimConfig())
sim.reset(seed=7, start="2024-09-02T06:00")
notes = sim.run(until=TimeLimit(7 * 86400))
sim.kpis()
```

**Events.** ``SlotStart`` (every clock hour) asks the demand generator for that hour's requests;
the requests are then read in time order from that batch, without going through the heap.
A request from the city is handled by the user model (§7.1): a bike is reserved and
``TripStart`` follows after the walk; ``TripEnd`` parks the bike at the destination.
In **docked** mode (§7.1 item 4) bikes live at station nodes only; a user arriving at a full
station rides on to the nearest station with a free dock (``NO_DOCK`` is recorded when that is
farther than ``users.redirect_radius_m``; if every station is full the bike is left at the
original one and counted as ``dock_overflow``).
EXTERNAL ends use the reservoir (§6.5): a trip to outside makes the bike leave at ``TripStart``;
a request from outside takes a bike from the reservoir, which arrives at its destination after
the recorded or generated ride time (``external.ride_time_s`` when unknown).

**Reproducibility** (§10). ``reset`` is a pure function of (scenario, config, seed, start): every
random draw comes from :class:`~bss_sim.rng.Streams`, and per-slot draws (user decisions) use a
stream keyed by the slot, so they do not depend on what happened before.
"""

from __future__ import annotations

import datetime as dt
import heapq
import json
import math
from pathlib import Path

import numpy as np

from bss_sim.config import SimConfig
from bss_sim.demand import PoissonDemand, ReplayDemand
from bss_sim.energy import Battery
from bss_sim.info import InFlight, Observation, TruckView, readonly
from bss_sim.engine import (REDIRECT, SLOT_START, SNAPSHOT, TRIP_END, TRIP_REQUEST, TRIP_START, TRUCK, WAKEUP,
                            EventQueue, Notification, Until)
from bss_sim.partitions import Partition, PartitionCounts
from bss_sim.rng import Streams
from bss_sim.scenario import Scenario
from bss_sim import telemetry
from bss_sim.commands import Command, CommandResult, Drop, GoTo, PickUp, Restock, SwapBattery, ToDepot, Wait
from bss_sim.state import DEPOT, ELECTRIC, ON_TRUCK, RESERVED, RIDING, Fleet, initial_nodes
from bss_sim.travel import make_travel
from bss_sim.vehicle import CostModel, VanEnergy
from bss_sim.trucks import COMMAND_DONE, TRUCK_IDLE, TRUCK_RESTOCKING, TRUCK_RESUMED, Truck, validate
from bss_sim.users import EXTERNAL_EMPTY, NO_DOCK, REASONS, REQUEST_FAILURES, SERVED, WalkingUsers

USER_SLOT_STREAM = 0x5E5
REDIRECT_FALLBACK_SPEED_MPS = 3.5  # riding pace on to another dock when the trip's own pace is unknown


def _requested(cmd) -> int:
    return int(getattr(cmd, "n", 1))


def _parse_start(start) -> dt.datetime:
    if isinstance(start, dt.datetime):
        return start.replace(tzinfo=None)
    if isinstance(start, dt.date):
        return dt.datetime.combine(start, dt.time())
    return dt.datetime.fromisoformat(str(start)).replace(tzinfo=None)


class Kpis:
    """Counters always kept in memory (spec §11)."""

    def __init__(self, walk_radius_m: float = 400.0):
        self.walk_hist = telemetry.WalkHistogram(walk_radius_m)
        self.requests = 0                 # all requests, including from outside
        self.city_requests = 0            # requests starting in the city
        self.city_served = 0
        self.served = 0
        self.failures = [0] * len(REASONS)
        self.skipped_external_both = 0    # requests with both ends outside (none expected)
        self.walk_m_sum = 0.0
        self.walked = 0                   # served requests that needed a walk > 0
        self.trips_ended = 0
        self.ride_km = 0.0                # in-city km ridden
        self.ebike_km = 0.0               # of which on e-bikes (battery used / consumption)
        self.to_external = 0
        self.from_external = 0
        self.redirects = 0                # docked: rides on to another station because of a full one
        self.redirect_km = 0.0
        self.dock_overflow = 0            # docked: every station full, bike left over capacity

    def as_dict(self) -> dict:
        failed = {REASONS[r]: self.failures[r] for r in REQUEST_FAILURES}
        n_fail = sum(failed.values())
        return {
            "requests": self.requests,
            "served": self.served,
            "failed": n_fail,
            "failures": failed,
            "service_level": self.served / max(self.served + n_fail, 1),
            "city_requests": self.city_requests,
            "city_service_level": self.city_served / max(self.city_requests, 1),
            "walk_m_mean": self.walk_m_sum / max(self.city_served, 1),   # users served in the city
            "walked_share": self.walked / max(self.city_served, 1),
            "walk_m_p50": self.walk_hist.quantile(0.5),
            "walk_m_p90": self.walk_hist.quantile(0.9),
            "trips_ended": self.trips_ended,
            "ride_km": self.ride_km,
            "ebike_km": self.ebike_km,
            "to_external": self.to_external,
            "from_external": self.from_external,
            "skipped_external_both": self.skipped_external_both,
            "redirects": self.redirects,
            "no_dock": self.failures[NO_DOCK],
            "redirect_km": self.redirect_km,
            "dock_overflow": self.dock_overflow,
        }


class Simulator:
    def __init__(self, scenario: Scenario, config: SimConfig | None = None, user_model=None):
        config = config or SimConfig()
        if scenario.mode != config.mode:
            raise ValueError(f"scenario loaded for {scenario.mode!r} but config.mode is {config.mode!r}")
        if config.users.walk_radius_m > scenario.walk_max_m:
            raise ValueError(f"users.walk_radius_m {config.users.walk_radius_m} exceeds the bundle's walking "
                             f"neighbours ({scenario.walk_max_m} m)")
        self.scenario = scenario
        self.config = config
        u = config.users
        self.users = user_model or WalkingUsers(scenario.walk_indptr, scenario.walk_indices, scenario.walk_dist,
                                                u.walk_radius_m, u.walk_beta_m, u.kind_tolerance_m,
                                                config.energy.dead_ebike_rideable)
        self.battery = Battery(config.energy, config.fleet.battery_range_km)
        self._ext = scenario.demand.external
        self._dist = np.asarray(scenario.bike_dist)  # plain ndarray view: memmap.__getitem__ is slow in Python
        self._ebike_share = (config.fleet.ebike_share if config.fleet.ebike_share is not None
                             else scenario.demand.ebike_share)
        self.fleet: Fleet | None = None
        self._parts: dict[str, Partition] = {}
        self._docked = config.mode == "docked"
        if self._docked:
            self._init_docks()
        self.travel = make_travel(scenario, config.travel)
        tc = config.trucks
        self.depot_drive = scenario.nearest_drive_node(tc.depot_lat, tc.depot_lon)
        self.depot_stop = scenario.stop_at(self.depot_drive)
        self._b2d: list[int] = scenario.bike_to_drive.tolist()
        self.trucks: list[Truck] = []
        self._fc = None
        self.van = VanEnergy(config.vehicle)
        self.costs = CostModel(config.costs)

    def _init_docks(self) -> None:
        """Capacity per node and, for every station node, the other station nodes by riding distance."""
        cap = self.scenario.node_capacity()
        self._cap: list[int] = cap.tolist()
        st = np.unique(self.scenario.station_node)
        d = self._dist[np.ix_(st, st)].astype(np.float64)
        self._dock_order: dict[int, list[tuple[int, float]]] = {}
        for i, node in enumerate(st.tolist()):
            order = np.argsort(d[i], kind="stable")
            self._dock_order[node] = [(int(st[j]), float(d[i, j])) for j in order if j != i]

    @classmethod
    def from_scenario(cls, root: str | Path, config: SimConfig | None = None, **kw) -> Simulator:
        config = config or SimConfig()
        return cls(Scenario.load(root, mode=config.mode), config, **kw)

    # ------------------------------------------------------------------------------- reset
    def reset(self, seed: int, start, snapshot: np.ndarray | None = None) -> list[Notification]:
        """Start a new run at local time ``start``. ``snapshot``: initial node of each city bike
        (``fleet.initial_distribution = from_snapshot``)."""
        cfg, sc = self.config, self.scenario
        self.seed = int(seed)
        self.start = _parse_start(start)
        self.day0 = self.start.date()
        self.now = (self.start - dt.datetime.combine(self.day0, dt.time())).total_seconds()
        self.streams = Streams(self.seed)
        if cfg.demand.generator == "poisson":
            self.demand = PoissonDemand(sc.demand, seed=self.streams.int_seed("demand"))
        elif cfg.demand.generator == "replay":
            self.demand = ReplayDemand(sc.demand, sc.root, seed=self.streams.int_seed("demand"))
        else:
            self.demand = None
        self._user_seed = self.streams.int_seed("users")
        self._ext_rng = self.streams.generator("external")
        self.travel.bind(self.streams.generator("travel"), self.day0)
        self._fc_rng = self.streams.generator("forecast")
        self._t0 = self.now
        self._init_trucks()
        self._init_fleet(snapshot)
        self.users.bind(self.fleet)
        for part in self._parts.values():
            part.bind(self.fleet)

        self.queue = EventQueue()
        self.kpis_ = Kpis(cfg.users.walk_radius_m)
        self._snap_t: list[float] = []
        self._snap: list[np.ndarray] = []
        self.log = [] if cfg.telemetry.event_log else None
        self._notes: list[Notification] = []
        self._wake_tags: list = []
        n = self.fleet.n
        # per-bike current trip (Python lists: scalar access on the hot path)
        self._trip_dest: list[int] = [-1] * n
        self._trip_km: list[float] = [0.0] * n
        self._trip_ride: list[float] = [0.0] * n
        self._trip_req: list[int] = [-1] * n
        self._trip_oplace: list[int] = [-1] * n   # information layer: origin place and node,
        self._trip_onode: list[int] = [-1] * n    # start and arrival times of the current trip
        self._trip_t0: list[float] = [0.0] * n
        self._trip_eta: list[float] = [0.0] * n
        self.oracle_used = False
        # configuration scalars read on the hot path
        self._ext_km = cfg.energy.external_trip_km
        self._rt_speed = cfg.energy.round_trip_speed_mps
        self._ext_ride = cfg.external.ride_time_s
        self._walk_speed = cfg.users.walking_speed_mps
        self._bat_lo, self._bat_hi = cfg.fleet.initial_battery
        self._req_id = 0
        self._slot_requests: dict[int, int] = {}   # requests from the city per clock hour (seen in the app)
        self._slot_served: dict[int, int] = {}     # ... and how many of them got a bike
        self._ext_req: dict[int, int] = {}         # the same for requests from outside
        self._ext_served: dict[int, int] = {}
        P = self.scenario.demand.n_places
        self._place_req: list[int] = [0] * P       # requests and served requests per origin place,
        self._place_served: list[int] = [0] * P    # for the current slot (moved to the history hourly)
        self._place_hist: dict[int, tuple[list[int], list[int]]] = {}
        self._cur_slot = int(self.now // 3600)
        self._slot = int(self.now // 3600)
        self._rq: list[tuple] = []
        self._ri = 0
        self._rn = 0
        self.queue.push(self._slot * 3600.0, SLOT_START, self._slot)
        if cfg.telemetry.snapshot_every_s:
            self.queue.push(self.now, SNAPSHOT)
        return [Notification(self.now, TRUCK_IDLE, {"truck": tr.id, "reason": "start"}) for tr in self.trucks]

    def _init_trucks(self) -> None:
        tc = self.config.trucks
        self.trucks = []
        rng = self.streams.generator("truck_start")
        for i in range(tc.count):
            stop = self.depot_stop if tc.start == "depot" else int(rng.integers(self.scenario.n_nodes))
            self.trucks.append(Truck(i, stop, self._b2d[stop], stock=tc.battery_stock_initial))

    def _init_fleet(self, snapshot) -> None:
        cfg, sc = self.config, self.scenario
        rng = self.streams.generator("fleet")
        n_city, n_ext = cfg.fleet.size, cfg.external.reservoir
        n_ops = cfg.trucks.count * cfg.trucks.initial_load + cfg.trucks.depot_spare_bikes
        kind = (rng.random(n_city + n_ext) < self._ebike_share).astype(np.uint8)
        lo, hi = cfg.fleet.initial_battery
        battery = self.battery.draw(rng, n_city + n_ext, lo, hi)
        # operators' bikes: own stream (so trucks never change the street fleet), charged
        ops = self.streams.generator("trucks")
        kind = np.concatenate([kind, (ops.random(n_ops) < self._ebike_share).astype(np.uint8)])
        battery = np.concatenate([battery, np.full(n_ops, cfg.fleet.battery_range_km)])
        fleet = Fleet(sc.n_nodes, kind, battery, cfg.fleet.battery_range_km)
        capacity = sc.node_capacity() if cfg.mode == "docked" else None
        dist = cfg.fleet.initial_distribution
        if dist == "from_snapshot" or snapshot is not None:
            if snapshot is None:
                raise ValueError("initial_distribution 'from_snapshot' needs reset(..., snapshot=nodes)")
            nodes = np.asarray(snapshot, dtype=np.int32)
            if len(nodes) != n_city:
                raise ValueError(f"snapshot has {len(nodes)} bikes, fleet.size is {n_city}")
        elif dist == "proportional_to_demand":
            nodes = initial_nodes(rng, n_city, sc.departure_weights(), capacity)
        elif capacity is not None:  # uniform, docked: the same share of docks filled everywhere
            nodes = initial_nodes(rng, n_city, capacity.astype(np.float64), capacity)
        else:  # uniform, dockless: over the nodes where demand can start
            w = (sc.departure_weights() > 0).astype(np.float64)
            nodes = initial_nodes(rng, n_city, w, capacity)
        # ids: city bikes 0..n_city-1, then the reservoir, then bikes on trucks, then depot spares
        fleet.reservoir = tuple([b for b in range(n_city, n_city + n_ext) if fleet.kind_l[b] == k] for k in (0, 1))
        for b in range(n_city):
            fleet.park(b, int(nodes[b]))
        b = n_city + n_ext
        self.depot_bikes: list[int] = []
        for tr in self.trucks:
            for _ in range(cfg.trucks.initial_load):
                fleet.set_status(b, ON_TRUCK)
                tr.load.append(b)
                b += 1
        for _ in range(cfg.trucks.depot_spare_bikes):
            fleet.set_status(b, DEPOT)
            self.depot_bikes.append(b)
            b += 1
        self.fleet = fleet

    # -------------------------------------------------------------------------- controller API
    def wakeup(self, at: float | None = None, after: float | None = None, tag=None) -> None:
        """Ask for a ``"wakeup"`` notification at clock time ``at`` or ``after`` seconds from now."""
        if (at is None) == (after is None):
            raise ValueError("give exactly one of at= or after=")
        t = self.now + after if after is not None else at
        if t < self.now:
            raise ValueError("cannot schedule a wakeup in the past")
        self._wake_tags.append(tag)
        self.queue.push(float(t), WAKEUP, len(self._wake_tags) - 1)

    def inflight(self, level: str | None = None) -> InFlight | None:
        """Trips in progress at the configured knowledge level (or a lower one)."""
        level = level or self.config.info.inflight_knowledge
        allowed = ("none", "count", "origin", "destination")
        if allowed.index(level) > allowed.index(self.config.info.inflight_knowledge):
            raise PermissionError(f"inflight_knowledge is {self.config.info.inflight_knowledge!r}; "
                                  f"{level!r} is not available")
        return self._inflight(level)

    def _inflight(self, level: str) -> InFlight | None:
        """Trips in progress at ``level`` without the permission check (forecast internals)."""
        if level == "none":
            return None
        riding = np.flatnonzero(self.fleet.status == RIDING)
        if level == "count":
            return InFlight(len(riding))
        rl = riding.tolist()
        out = dict(bike=riding.astype(np.int32), kind=self.fleet.kind[riding].copy(),
                   o_place=np.array([self._trip_oplace[b] for b in rl], dtype=np.int16),
                   o_node=np.array([self._trip_onode[b] for b in rl], dtype=np.int32),
                   start_t=np.array([self._trip_t0[b] for b in rl], dtype=np.float64))
        if level == "destination":
            self.oracle_used = True
            out["d_node"] = np.array([self._trip_dest[b] for b in rl], dtype=np.int32)
            out["end_t"] = np.array([self._trip_eta[b] for b in rl], dtype=np.float64)
        return InFlight(len(riding), **out)

    def forecast(self, window_s: float, partition: str, provider: str = "mean", samples: int = 100):
        """Departures and arrivals per group of ``partition`` in ``(now, now + window_s]``
        (:mod:`bss_sim.forecast`). ``provider``: ``mean`` | ``sampled`` (``samples`` scenarios) |
        ``oracle`` (needs ``info.allow_oracle``)."""
        from bss_sim import forecast as fc

        if partition not in self._parts:
            raise KeyError(f"no partition {partition!r}; register it with add_partition")
        if not 0 < window_s <= fc.TMAX_S:
            raise ValueError(f"window_s must be in (0, {fc.TMAX_S:g}]")
        if self._fc is None:
            self._fc = fc.ForecastContext(self)
        level = self.config.info.inflight_knowledge
        if provider == "mean":
            return fc.mean_forecast(self._fc, window_s, partition, level)
        if provider == "sampled":
            return fc.sampled_forecast(self._fc, window_s, partition, level, samples, self._fc_rng)
        if provider == "oracle":
            if not self.config.info.allow_oracle:
                raise PermissionError("the oracle forecast needs info.allow_oracle: true (ablations only)")
            self.oracle_used = True
            return fc.oracle_forecast(self._fc, window_s, partition)
        raise ValueError(f"unknown forecast provider {provider!r}")

    def observe(self) -> Observation:
        """What a controller may see now (spec §8): copies and read-only views, never the queue."""
        f = self.fleet
        trucks = tuple(TruckView(tr.id, tr.stop, tr.drive_node, tr.busy, tr.current, tr.eta if tr.busy else self.now,
                                 len(tr.queue), len(tr.load), sum(f.kind_l[b] == ELECTRIC for b in tr.load),
                                 tr.stock, tr.used) for tr in self.trucks)
        return Observation(self.now, self.datetime, f.n_bikes.copy(), readonly(f.node), readonly(f.status),
                           readonly(f.kind), readonly(f.battery_km),
                           {name: part.counts() for name, part in self._parts.items()}, trucks,
                           self.inflight(), self.kpis())

    def add_partition(self, name: str, node_to_group: np.ndarray) -> None:
        """Register a partition of the bike nodes (``-1`` = no group); see :mod:`bss_sim.partitions`.

        Can be called before or during a run; event counters start at the call (and at every reset)."""
        if name in self._parts:
            raise ValueError(f"partition {name!r} already registered")
        part = Partition(name, node_to_group, self.scenario.n_nodes, self.battery)
        self._parts[name] = part
        if self.fleet is not None:
            part.bind(self.fleet)

    def partition(self, name: str) -> PartitionCounts:
        return self._parts[name].counts()

    def submit(self, truck: int, commands: list[Command], replace: bool = False) -> None:
        """Append ``commands`` to a truck's queue (``replace``: drop the queued ones first; the
        command in progress always finishes). An idle truck starts at once."""
        tr = self.trucks[truck]
        if replace:
            while tr.queue:
                cmd = tr.queue.popleft()
                self._result(tr, CommandResult(tr.id, cmd, 0, 0, self.now, self.now, "cancelled"))
        tr.queue.extend(commands)
        if not tr.busy:
            self._next_command(tr)

    @property
    def datetime(self) -> dt.datetime:
        return dt.datetime.combine(self.day0, dt.time()) + dt.timedelta(seconds=self.now)

    def kpis(self) -> dict:
        out = self.kpis_.as_dict()
        trucks = [self.truck_kpis(i) for i in range(len(self.trucks))]
        out["trucks"] = trucks
        out["truck_km"] = sum(t["km"] for t in trucks)
        out["truck_kwh"] = sum(t["kwh"] for t in trucks)
        out["bikes_moved"] = sum(t["dropped"] for t in trucks)
        out["batteries_swapped"] = sum(t["batteries_swapped"] for t in trucks)
        out["operating_cost_usd"] = sum(t["cost_usd"]["total"] for t in trucks)
        out["kgco2e"] = sum(t["kgco2e"] for t in trucks)
        out["energy_flags"] = sum(t["energy_flags"] for t in trucks)
        return out

    def check_invariants(self) -> None:
        self.fleet.check()
        on_trucks = sorted(b for tr in self.trucks for b in tr.load)
        assert on_trucks == np.flatnonzero(self.fleet.status == ON_TRUCK).tolist(), "bikes on trucks"
        assert sorted(self.depot_bikes) == np.flatnonzero(self.fleet.status == DEPOT).tolist(), "depot bikes"
        assert all(len(tr.load) <= self.config.trucks.capacity for tr in self.trucks), "truck over capacity"
        for part in self._parts.values():
            bikes, ebikes = part.recompute_state(self.fleet)
            c = part.counts()
            assert np.array_equal(bikes, c.bikes) and np.array_equal(ebikes, c.ebikes), f"partition {part.name}"
        if self._docked:
            nb = self.fleet.n_bikes
            cap = np.asarray(self._cap)
            assert nb[cap == 0].sum() == 0, "docked mode: bikes outside stations"
            assert np.maximum(nb - cap, 0).sum() <= self.kpis_.dock_overflow, "docked mode: station over capacity"

    def event_log(self) -> dict[str, np.ndarray]:
        """The event log as arrays (``telemetry.event_log`` must be on); columns in :mod:`bss_sim.telemetry`."""
        if self.log is None:
            raise RuntimeError("event log disabled (telemetry.event_log: false)")
        return telemetry.to_arrays(self.log)

    def snapshots(self) -> tuple[np.ndarray, np.ndarray]:
        """Times [K] and parked bikes per node [K, nodes] (``telemetry.snapshot_every_s``)."""
        n = self.scenario.n_nodes
        return (np.array(self._snap_t, dtype=np.float64),
                np.array(self._snap, dtype=np.int16).reshape(len(self._snap), n))

    def metadata(self) -> dict:
        """What identifies this run: bundle, config (and hash), seed, start, versions."""
        import bss_sim
        return {"bundle_id": self.scenario.bundle_id, "bundle": str(self.scenario.root),
                "config": self.config.to_dict(), "config_hash": self.config.config_hash(),
                "seed": self.seed, "start": self.start.isoformat(), "clock_s": self.now,
                "oracle_allowed": self.config.info.allow_oracle, "oracle_used": self.oracle_used,
                "bss_sim": bss_sim.__version__, "numpy": np.__version__}

    def save_log(self, path: str | Path) -> Path:
        """Write the event log (``.npz`` or ``.parquet``) and, if any, the snapshots next to it
        (``<name>.snapshots.npz``); the metadata is embedded in both."""
        path = telemetry.save_events(path, self.event_log(), self.metadata())
        if self._snap:
            t, bikes = self.snapshots()
            np.savez_compressed(path.with_name(path.stem + ".snapshots.npz"), t=t, bikes=bikes,
                                metadata=np.array(json.dumps(self.metadata(), sort_keys=True)))
        return path

    # ---------------------------------------------------------------------------------- loop
    def _load_slot(self, k: int) -> None:
        if self._slot_requests.get(self._cur_slot):
            self._place_hist[self._cur_slot] = (self._place_req, self._place_served)
            P = len(self._place_req)
            self._place_req, self._place_served = [0] * P, [0] * P
        self._slot_requests[k] = 0
        self._slot_served[k] = 0
        self._ext_req[k] = 0
        self._ext_served[k] = 0
        self._cur_slot = k
        if self.demand is None:
            self._rq, self._ri, self._rn = [], 0, 0
            return
        date = self.day0 + dt.timedelta(days=k // 24)
        req = self.demand.hour(date, k % 24)
        t = req.t_s + (k // 24) * 86400.0
        keep = t >= self.now
        n = int(keep.sum())
        rng = np.random.default_rng([self._user_seed, USER_SLOT_STREAM, k])
        u = rng.random(len(t))[keep]
        o, d = req.o_node[keep], req.d_node[keep]
        inside = (o >= 0) & (d >= 0)
        dist_m = np.zeros(n)
        dist_m[inside] = self._dist[o[inside], d[inside]]
        self._rq = list(zip(t[keep].tolist(), req.o_place[keep].tolist(), req.d_place[keep].tolist(),
                            o.tolist(), d.tolist(), req.ride_time_s[keep].tolist(),
                            req.kind[keep].tolist(), u.tolist(), dist_m.tolist()))
        self._ri, self._rn = 0, n

    def run(self, until: Until) -> list[Notification]:
        """Advance until a condition of ``until`` is met; returns the notifications emitted."""
        deadline = until.deadline(self.now)
        kinds = until.kinds()
        if deadline == math.inf and not kinds:
            raise ValueError("run(until=...) needs a time limit or a notification to stop on")
        heap = self.queue._heap
        pop = heapq.heappop
        debug = self.config.debug
        request, trip_start, trip_end = self._request, self._trip_start, self._trip_end
        rq, ri, rn = self._rq, self._ri, self._rn
        stop = False
        while not stop:
            if ri < rn:  # next request of the current slot, unless a queued event comes first
                r = rq[ri]
                tr = r[0]
                if not heap or tr < heap[0][0] or (tr == heap[0][0] and heap[0][1] > TRIP_REQUEST):
                    if tr > deadline:
                        break
                    self.now = tr
                    ri += 1
                    request(r)
                    if debug:
                        self.check_invariants()
                    continue
            if not heap or heap[0][0] > deadline:
                break
            t, _, _, kind, a, b = pop(heap)
            self.now = t
            if kind == TRIP_END:
                trip_end(a)
            elif kind == TRIP_START:
                trip_start(a)
            elif kind == SLOT_START:  # the previous slot's requests are all before this time
                self._load_slot(a)
                rq, ri, rn = self._rq, 0, self._rn
                self.queue.push((a + 1) * 3600.0, SLOT_START, a + 1)
            elif kind == SNAPSHOT:
                self._snap_t.append(t)
                self._snap.append(self.fleet.n_bikes.astype(np.int16))
                self.queue.push(t + self.config.telemetry.snapshot_every_s, SNAPSHOT)
            elif kind == TRUCK:
                tr = self.trucks[a]
                if b == tr.serial:  # not a stale event
                    n0 = len(self._notes)
                    self._truck_done(tr)
                    if kinds and any(nt.kind in kinds for nt in self._notes[n0:]):
                        stop = True
            elif kind == WAKEUP:
                note = Notification(t, "wakeup", {"tag": self._wake_tags[a]})
                self._notes.append(note)
                if "wakeup" in kinds:
                    stop = True
            if debug:
                self.check_invariants()
        self._ri = ri
        if not stop and deadline != math.inf:
            self.now = max(self.now, deadline)
        notes, self._notes = self._notes, []
        return notes

    # -------------------------------------------------------------------------------- events
    def _fail(self, rid: int, node: int, reason: int, walk: float = 0.0) -> None:
        self._last_walk = walk
        self.kpis_.failures[reason] += 1
        for p in self._parts.values():
            p.failures[p._g[node]][reason] += 1
        if self.log is not None:
            self.log.append((self.now, TRIP_REQUEST, rid, -1, node, reason, self._last_walk))

    def _request(self, r: tuple) -> None:
        t, op, dp, on, dn, ride, pref, u, dist_m = r
        rid = self._req_id
        self._req_id += 1
        k = self.kpis_
        k.requests += 1
        ext = self._ext
        f = self.fleet
        if op == ext:
            if dp == ext:
                k.skipped_external_both += 1
                return
            self._ext_req[self._cur_slot] += 1
            bike = f.take_external(pref)
            if bike < 0:
                self._fail(rid, dn, EXTERNAL_EMPTY)
                return
            if f.kind_l[bike] == ELECTRIC:  # charged outside the city: fresh draw
                f._batt[bike] = self._ext_rng.uniform(self._bat_lo, self._bat_hi) * self.battery.range_km
            k.served += 1
            k.from_external += 1
            self._ext_served[self._cur_slot] += 1
            self._trip_dest[bike] = dn
            self._trip_km[bike] = self._ext_km
            self._trip_req[bike] = rid
            if ride != ride:  # nan -> default delay from outside
                ride = self._ext_ride
            self._trip_ride[bike] = ride
            if self.log is not None:
                self.log.append((t, TRIP_REQUEST, rid, bike, dn, SERVED, 0.0))
            self._trip_oplace[bike], self._trip_onode[bike] = op, -1
            self._trip_t0[bike] = t
            self._trip_eta[bike] = t + ride
            self.queue.push(t + ride, TRIP_END, bike)
            return
        to_ext = dp == ext
        if ride != ride:  # in-city trip without a ride time (no distances loaded): typical pace
            ride = 0.0 if to_ext else dist_m / self._rt_speed
        trip_km = self._ext_km if to_ext else dist_m / 1000.0
        k.city_requests += 1
        self._slot_requests[self._cur_slot] += 1
        self._place_req[op] += 1
        for p in self._parts.values():
            p.requests[p._g[on]] += 1
        bike, walk, reason = self.users.choose(on, pref, self.battery.need_km(trip_km), u)
        if bike < 0:
            self._fail(rid, on, reason, walk)
            return
        k.served += 1
        k.city_served += 1
        self._slot_served[self._cur_slot] += 1
        self._place_served[op] += 1
        k.walk_hist.add(walk)
        node = f.unpark(bike, RESERVED)
        if not to_ext:
            if walk > 0.0:  # the bike is not at the origin node: distance from where it is
                trip_km = float(self._dist[node, dn]) / 1000.0
            if op == dp:  # round trip by place: distance from the ride time at a typical pace
                trip_km = max(trip_km, ride * self._rt_speed / 1000.0)
        self._trip_dest[bike] = -1 if to_ext else dn
        self._trip_km[bike] = trip_km
        self._trip_ride[bike] = ride
        self._trip_req[bike] = rid
        self._trip_oplace[bike], self._trip_onode[bike] = op, node
        if self.log is not None:
            self.log.append((t, TRIP_REQUEST, rid, bike, node, SERVED, walk))
        if walk > 0.0:
            k.walk_m_sum += walk
            k.walked += 1
            self.queue.push(t + walk / self._walk_speed, TRIP_START, bike)
        else:  # no walk: the trip starts now (same order as a TripStart event at time t)
            self._trip_start(bike)

    def _trip_start(self, bike: int) -> None:
        f = self.fleet
        if f.kind_l[bike] == ELECTRIC:
            f._batt[bike] = self.battery.after_ride(f._batt[bike], self._trip_km[bike])
            self.kpis_.ebike_km += self._trip_km[bike]
        if self._parts or self.log is not None:
            node = f._node[bike]  # where the bike was taken (kept after unpark)
            for p in self._parts.values():
                p.departures[p._g[node]] += 1
            if self.log is not None:
                self.log.append((self.now, TRIP_START, self._trip_req[bike], bike, node, SERVED,
                                 self._trip_ride[bike] if self._trip_dest[bike] >= 0 else 0.0))
        if self._trip_dest[bike] < 0:  # to outside: the bike leaves the city now
            self.kpis_.to_external += 1
            f.to_external(bike)
            return
        self.kpis_.ride_km += self._trip_km[bike]
        f.set_status(bike, RIDING)
        self._trip_t0[bike] = self.now
        self._trip_eta[bike] = self.now + self._trip_ride[bike]
        self.queue.push(self._trip_eta[bike], TRIP_END, bike)

    def _trip_end(self, bike: int) -> None:
        node = self._trip_dest[bike]
        if self._docked and self.fleet._nb[node] >= self._cap[node] and self._redirect(bike, node):
            return
        self.fleet.park(bike, node)
        self.kpis_.trips_ended += 1
        for p in self._parts.values():
            p.arrivals[p._g[node]] += 1
        if self.log is not None:
            self.log.append((self.now, TRIP_END, self._trip_req[bike], bike, node, SERVED, 0.0))

    def _redirect(self, bike: int, node: int) -> bool:
        """Docked mode, ``node`` full: ride on to the nearest station with a free dock.

        Returns False when every station is full (the bike is then left at ``node``)."""
        nb, cap = self.fleet._nb, self._cap
        for alt, extra_m in self._dock_order[node]:
            if nb[alt] < cap[alt]:
                break
        else:
            self.kpis_.dock_overflow += 1
            return False
        k = self.kpis_
        k.redirects += 1
        if extra_m > self.config.users.redirect_radius_m:
            k.failures[NO_DOCK] += 1
        extra_km = extra_m / 1000.0
        k.redirect_km += extra_km
        k.ride_km += extra_km
        f = self.fleet
        if f.kind_l[bike] == ELECTRIC:
            f._batt[bike] = self.battery.after_ride(f._batt[bike], extra_km)
            k.ebike_km += extra_km
        trip_m, ride = self._trip_km[bike] * 1000.0, self._trip_ride[bike]
        speed = trip_m / ride if trip_m > 0 and ride > 0 else REDIRECT_FALLBACK_SPEED_MPS
        self._trip_dest[bike] = alt
        self._trip_km[bike] = extra_km
        self._trip_ride[bike] = extra_m / speed
        if self.log is not None:
            self.log.append((self.now, REDIRECT, self._trip_req[bike], bike, alt,
                             NO_DOCK if extra_m > self.config.users.redirect_radius_m else SERVED, extra_m))
        self._trip_eta[bike] = self.now + extra_m / speed
        self.queue.push(self._trip_eta[bike], TRIP_END, bike)
        return True

    # ------------------------------------------------------------------------------- trucks
    def _result(self, tr: Truck, res: CommandResult) -> None:
        tr.commands_done += 1
        self._notes.append(Notification(self.now, COMMAND_DONE, {"truck": tr.id, "result": res}))

    def _next_command(self, tr: Truck) -> None:
        """Start the next queued command; emit ``truck_idle`` when there is none."""
        while tr.queue:
            cmd = tr.queue.popleft()
            reason = validate(cmd, self.scenario.n_nodes) or self._check(tr, cmd)
            if reason:
                self._result(tr, CommandResult(tr.id, cmd, _requested(cmd), 0, self.now, self.now, reason))
                continue
            if self._needs_restock(tr, cmd):
                if self.config.trucks.on_empty == "resume":
                    self._insert_restock(tr, cmd)
                    continue
                self._result(tr, CommandResult(tr.id, cmd, _requested(cmd), 0, self.now, self.now, "stock_empty"))
                self._cancel_queue(tr)
                tr.current = None
                self._notes.append(Notification(self.now, TRUCK_IDLE, {"truck": tr.id, "reason": "stock_empty"}))
                return
            tr.current, tr.t_start = cmd, self.now
            tr.serial += 1
            if self._start(tr, cmd):
                return
        tr.current = None
        if tr.idle_reported:   # already reported with its reason (stock empty)
            tr.idle_reported = False
            return
        self._notes.append(Notification(self.now, TRUCK_IDLE, {"truck": tr.id, "reason": "queue_empty"}))

    def _cancel_queue(self, tr: Truck) -> None:
        while tr.queue:
            cmd = tr.queue.popleft()
            self._result(tr, CommandResult(tr.id, cmd, _requested(cmd), 0, self.now, self.now, "cancelled"))

    def _needs_restock(self, tr: Truck, cmd: Command) -> bool:
        """Stock empty and this command (or the swap right after this move) needs batteries."""
        if tr.stock > 0:
            return False
        if isinstance(cmd, SwapBattery):
            return cmd.n > 0 and bool(self._swappable(tr.stop, cmd.radius_m))
        return isinstance(cmd, GoTo) and bool(tr.queue) and isinstance(tr.queue[0], SwapBattery) and tr.queue[0].n > 0

    def _insert_restock(self, tr: Truck, cmd: Command) -> None:
        """Spec §9.1 ``on_empty: resume``: depot first, then the interrupted command where it was
        heading (a swap at the current stop goes back there first)."""
        back = [cmd] if isinstance(cmd, GoTo) else [GoTo(tr.stop), cmd]
        for c in reversed([ToDepot(), Restock(), *back]):
            tr.queue.appendleft(c)
        tr.auto_restock = True
        self._notes.append(Notification(self.now, TRUCK_RESTOCKING, {"truck": tr.id, "reason": "stock_empty"}))

    def _check(self, tr: Truck, cmd: Command) -> str:
        """State-dependent validity, checked when the command starts."""
        wmax = self.scenario.walk_max_m
        if isinstance(cmd, (PickUp, SwapBattery)) and cmd.radius_m > wmax:
            return f"invalid: radius_m above the bundle's walking neighbours ({wmax} m)"
        if isinstance(cmd, Drop) and cmd.at is not None:
            reach = self._within(tr.stop, self.config.trucks.max_spread_m)
            if any(v not in reach for v in cmd.at):
                return "invalid: Drop(at=...) node out of reach of the stop"
        if isinstance(cmd, Restock) and tr.stop != self.depot_stop:
            return "not_at_depot"
        return ""

    def _energy(self, tr: Truck, kwh: float) -> None:
        """Energy is a KPI (D20); flag a van that uses more than its battery between depot visits."""
        tr.kwh += kwh
        tr.kwh_since_depot += kwh
        if tr.kwh_since_depot > self.config.vehicle.usable_kwh and not tr.over_battery:
            tr.over_battery = True
            tr.energy_flags += 1

    def truck_kpis(self, truck: int) -> dict:
        """Per-truck KPIs since reset: km, time split, bikes moved, swaps, kWh, cost (USD), CO2e."""
        tr = self.trucks[truck]
        shift_s = self.now - self._t0
        busy = tr.drive_s + tr.service_s + tr.wait_s
        if tr.busy:   # do not count the part of the current command that is still to come
            busy -= max(0.0, tr.eta - self.now)
        hours = shift_s / 3600.0
        return {"km": tr.km, "drive_s": tr.drive_s, "service_s": tr.service_s, "wait_s": tr.wait_s,
                "idle_s": max(0.0, shift_s - busy), "commands": tr.commands_done, "picked_up": tr.picked,
                "dropped": tr.dropped, "batteries_swapped": tr.swapped, "kwh": tr.kwh,
                "kwh_since_depot": tr.kwh_since_depot, "energy_flags": tr.energy_flags,
                "cost_usd": self.costs.cost_usd(tr.kwh, hours, tr.km, tr.swapped),
                "kgco2e": self.costs.kgco2e(tr.kwh)}

    def _busy_until(self, tr: Truck, t: float) -> None:
        tr.eta = t
        self.queue.push(t, TRUCK, tr.id, tr.serial)

    def _service(self, tr: Truck, k: int, per_bike: float) -> None:
        tr.planned = k
        dur = self.config.trucks.stop_overhead_s + k * per_bike
        tr.service_s += dur
        self._energy(tr, self.van.service_kwh(dur))
        self._busy_until(tr, self.now + dur)

    def _finish(self, tr: Truck, requested: int, executed: int, reason: str = "") -> None:
        self._result(tr, CommandResult(tr.id, tr.current, requested, executed, tr.t_start, self.now, reason))
        tr.current = None

    def _start(self, tr: Truck, cmd: Command) -> bool:
        """Begin ``cmd``; False when it finished at once (nothing to do), True when it takes time."""
        tc = self.config.trucks
        if isinstance(cmd, (GoTo, ToDepot)):
            stop = cmd.node if isinstance(cmd, GoTo) else self.depot_stop
            dest = self._b2d[stop] if isinstance(cmd, GoTo) else self.depot_drive
            t = self.travel.time(tr.drive_node, dest, self.now)
            d = self.travel.distance(tr.drive_node, dest)
            tr.km += d / 1000.0
            tr.drive_s += t
            self._energy(tr, self.van.drive_kwh(d, t, len(tr.load)))
            tr.target = (stop, dest)
            self._busy_until(tr, self.now + t)
        elif isinstance(cmd, Wait):
            tr.wait_s += cmd.seconds
            self._energy(tr, self.van.service_kwh(cmd.seconds))
            self._busy_until(tr, self.now + cmd.seconds)
        elif isinstance(cmd, PickUp):
            room = tc.capacity - len(tr.load)
            avail = len(self._pickable(tr.stop, cmd))
            k = min(cmd.n, room, avail)
            tr.limit = "" if k == cmd.n else ("truck_full" if room < min(cmd.n, avail) else "not_enough_bikes")
            if k == 0:
                self._finish(tr, cmd.n, 0, tr.limit)
                return False
            self._service(tr, k, tc.pickup_per_bike_s)
        elif isinstance(cmd, Drop):
            docks = self._free_docks(self._drop_targets(tr, cmd))
            k = min(cmd.n, len(tr.load), docks)
            tr.limit = "" if k == cmd.n else ("no_docks" if docks < min(cmd.n, len(tr.load)) else "not_enough_bikes_on_truck")
            if k == 0:
                self._finish(tr, cmd.n, 0, tr.limit)
                return False
            self._service(tr, k, tc.drop_per_bike_s)
        elif isinstance(cmd, SwapBattery):
            avail = len(self._swappable(tr.stop, cmd.radius_m))
            k = min(cmd.n, avail, tr.stock)
            tr.limit = "" if k == cmd.n else ("stock_empty" if tr.stock < min(cmd.n, avail) else "not_enough_ebikes")
            if k == 0:
                self._finish(tr, cmd.n, 0, tr.limit)
                return False
            self._service(tr, k, tc.swap_per_bike_s)
        elif isinstance(cmd, Restock):
            tr.service_s += tc.depot_restock_s
            self._energy(tr, self.van.service_kwh(tc.depot_restock_s))
            self._busy_until(tr, self.now + tc.depot_restock_s)
        return True

    def _truck_done(self, tr: Truck) -> None:
        cmd = tr.current
        if isinstance(cmd, (GoTo, ToDepot)):
            tr.stop, tr.drive_node = tr.target
            tr.target = None
            if tr.stop == self.depot_stop:   # a depot visit: a low van would be swapped here (D20)
                tr.kwh_since_depot = 0.0
                tr.over_battery = False
            self._finish(tr, 1, 1)
        elif isinstance(cmd, Wait):
            self._finish(tr, 1, 1)
        elif isinstance(cmd, PickUp):
            self._do_pickup(tr, cmd)
        elif isinstance(cmd, Drop):
            self._do_drop(tr, cmd)
        elif isinstance(cmd, SwapBattery):
            self._do_swap(tr, cmd)
        elif isinstance(cmd, Restock):
            self._do_restock(tr, cmd)
            if tr.auto_restock:
                tr.auto_restock = False
                self._notes.append(Notification(self.now, TRUCK_RESUMED, {"truck": tr.id}))
        self._next_command(tr)

    # --- services -------------------------------------------------------------------------
    def _within(self, node: int, radius_m: float) -> list[int]:
        """Bike nodes within ``radius_m`` walking distance of ``node`` (nearest first, itself first)."""
        sc = self.scenario
        a, b = int(sc.walk_indptr[node]), int(sc.walk_indptr[node + 1])
        k = int(np.searchsorted(sc.walk_dist[a:b], radius_m, side="right"))
        return sc.walk_indices[a:a + k].tolist()

    def _pickable(self, stop: int, cmd: PickUp) -> list[int]:
        """Parked bikes in reach, in the order the policy takes them."""
        f = self.fleet
        near = [(i, b) for i, v in enumerate(self._within(stop, cmd.radius_m)) for b in f.bikes_at(v)]
        if cmd.policy == "lowest_battery":
            e = sorted((f._batt[b], i, b) for i, b in near if f.kind_l[b] == ELECTRIC)
            return [b for _, _, b in e] + [b for i, b in near if f.kind_l[b] != ELECTRIC]
        return [b for _, b in near]

    def _swappable(self, stop: int, radius_m: float) -> list[int]:
        """Parked e-bikes in reach that are not full, lowest charge first."""
        f, full = self.fleet, self.battery.range_km
        near = [(f._batt[b], i, b) for i, v in enumerate(self._within(stop, radius_m)) for b in f.bikes_at(v)
                if f.kind_l[b] == ELECTRIC and f._batt[b] < full]
        return [b for _, _, b in sorted(near)]

    def _do_swap(self, tr: Truck, cmd: SwapBattery) -> None:
        f, full = self.fleet, self.battery.range_km
        todo = self._swappable(tr.stop, cmd.radius_m)[:min(tr.planned, tr.stock)]
        for b in todo:
            f.set_battery(b, full)
        tr.stock -= len(todo)
        tr.used += len(todo)
        tr.swapped += len(todo)
        reason = "taken_meanwhile" if len(todo) < tr.planned else tr.limit
        self._finish(tr, cmd.n, len(todo), reason)
        left = cmd.n - len(todo)
        if tr.stock == 0 and left > 0 and self._swappable(tr.stop, cmd.radius_m):
            if self.config.trucks.on_empty == "resume":   # come back for the rest after the depot
                self._insert_restock(tr, SwapBattery(left, cmd.radius_m))
            else:
                self._cancel_queue(tr)
                self._notes.append(Notification(self.now, TRUCK_IDLE, {"truck": tr.id, "reason": "stock_empty"}))
                tr.idle_reported = True

    def _drop_targets(self, tr: Truck, cmd: Drop) -> list[int]:
        return list(cmd.at) if cmd.at else [tr.stop]

    def _free_docks(self, nodes: list[int]) -> int:
        if not self._docked:
            return 1 << 30
        nb = self.fleet._nb
        return sum(max(0, self._cap[v] - nb[v]) for v in set(nodes))

    def _do_pickup(self, tr: Truck, cmd: PickUp) -> None:
        room = self.config.trucks.capacity - len(tr.load)
        take = self._pickable(tr.stop, cmd)[:min(tr.planned, room)]
        for b in take:
            self.fleet.unpark(b, ON_TRUCK)
            tr.load.append(b)
        tr.picked += len(take)
        reason = "taken_meanwhile" if len(take) < tr.planned else tr.limit
        self._finish(tr, cmd.n, len(take), reason)

    def _do_drop(self, tr: Truck, cmd: Drop) -> None:
        f = self.fleet
        # e-bikes with the most charge first, then classic bikes
        order = sorted(tr.load, key=lambda b: (f.kind_l[b] != ELECTRIC, -f._batt[b]))
        targets = self._drop_targets(tr, cmd)
        placed, i = 0, 0
        for b in order[:tr.planned]:
            for _ in range(len(targets)):  # round-robin over targets with room
                v = targets[i % len(targets)]
                i += 1
                if not self._docked or f._nb[v] < self._cap[v]:
                    tr.load.remove(b)
                    f.park(b, v)
                    placed += 1
                    break
            else:
                break
        tr.dropped += placed
        reason = "docks_taken_meanwhile" if placed < tr.planned else tr.limit
        self._finish(tr, cmd.n, placed, reason)

    def _do_restock(self, tr: Truck, cmd: Restock) -> None:
        tc, f = self.config.trucks, self.fleet
        full = self.battery.range_km
        tr.stock = tc.battery_stock_max if cmd.batteries is None else min(cmd.batteries, tc.battery_stock_max)
        tr.used = 0
        reason, moved, before = "", 0, len(tr.load)
        if cmd.load is not None:
            target = min(max(cmd.load, 0), tc.capacity)
            while len(tr.load) > target:                 # unload to the depot
                b = tr.load.pop()
                f.set_status(b, DEPOT)
                self.depot_bikes.append(b)
                moved += 1
            while len(tr.load) < target and self.depot_bikes:
                b = self.depot_bikes.pop()
                f.set_status(b, ON_TRUCK)
                tr.load.append(b)
                moved += 1
            if len(tr.load) < target:
                reason = "depot_empty"
        for b in tr.load:                                # charged at the depot
            if f.kind_l[b] == ELECTRIC:
                f._batt[b] = full
        requested = 0 if cmd.load is None else abs(min(max(cmd.load, 0), tc.capacity) - before)
        self._finish(tr, requested, moved, reason)
