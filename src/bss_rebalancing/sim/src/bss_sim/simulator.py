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
import math
from pathlib import Path

import numpy as np

from bss_sim.config import SimConfig
from bss_sim.demand import PoissonDemand, ReplayDemand
from bss_sim.energy import Battery
from bss_sim.engine import (REDIRECT, SLOT_START, TRIP_END, TRIP_REQUEST, TRIP_START, WAKEUP, EventQueue,
                            Notification, Until)
from bss_sim.partitions import Partition, PartitionCounts
from bss_sim.rng import Streams
from bss_sim.scenario import Scenario
from bss_sim.state import ELECTRIC, RESERVED, RIDING, Fleet, initial_nodes
from bss_sim.users import EXTERNAL_EMPTY, NO_DOCK, REASONS, REQUEST_FAILURES, SERVED, WalkingUsers

USER_SLOT_STREAM = 0x5E5
REDIRECT_FALLBACK_SPEED_MPS = 3.5  # riding pace on to another dock when the trip's own pace is unknown


def _parse_start(start) -> dt.datetime:
    if isinstance(start, dt.datetime):
        return start.replace(tzinfo=None)
    if isinstance(start, dt.date):
        return dt.datetime.combine(start, dt.time())
    return dt.datetime.fromisoformat(str(start)).replace(tzinfo=None)


class Kpis:
    """Counters always kept in memory (spec §11)."""

    def __init__(self):
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
            "walk_m_mean": self.walk_m_sum / max(self.served, 1),
            "walked_share": self.walked / max(self.served, 1),
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
        else:
            self.demand = ReplayDemand(sc.demand, sc.root, seed=self.streams.int_seed("demand"))
        self._user_seed = self.streams.int_seed("users")
        self._ext_rng = self.streams.generator("external")
        self._init_fleet(snapshot)
        self.users.bind(self.fleet)
        for part in self._parts.values():
            part.bind(self.fleet)

        self.queue = EventQueue()
        self.kpis_ = Kpis()
        self.log = [] if cfg.telemetry.event_log else None
        self._notes: list[Notification] = []
        self._wake_tags: list = []
        n = cfg.fleet.size + cfg.external.reservoir
        self._trip_dest = np.full(n, -1, dtype=np.int64)
        self._trip_km = np.zeros(n)
        self._trip_ride = np.zeros(n)
        self._trip_req = np.full(n, -1, dtype=np.int64)
        self._req_id = 0
        self._slot = int(self.now // 3600)
        self._rq: list[tuple] = []
        self._ri = 0
        self._rn = 0
        self.queue.push(self._slot * 3600.0, SLOT_START, self._slot)
        return []

    def _init_fleet(self, snapshot) -> None:
        cfg, sc = self.config, self.scenario
        rng = self.streams.generator("fleet")
        n_city, n_ext = cfg.fleet.size, cfg.external.reservoir
        n = n_city + n_ext
        kind = (rng.random(n) < self._ebike_share).astype(np.uint8)
        lo, hi = cfg.fleet.initial_battery
        battery = self.battery.draw(rng, n, lo, hi)
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
        # city bikes are 0..n_city-1; the reservoir keeps the others
        fleet.reservoir = tuple([b for b in range(n_city, n) if fleet.kind_l[b] == k] for k in (0, 1))
        for b in range(n_city):
            fleet.park(b, int(nodes[b]))
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

    @property
    def datetime(self) -> dt.datetime:
        return dt.datetime.combine(self.day0, dt.time()) + dt.timedelta(seconds=self.now)

    def kpis(self) -> dict:
        return self.kpis_.as_dict()

    def check_invariants(self) -> None:
        self.fleet.check()
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
        """The event log as arrays (``telemetry.event_log`` must be on)."""
        if self.log is None:
            raise RuntimeError("event log disabled (telemetry.event_log: false)")
        cols = list(zip(*self.log)) if self.log else [()] * 6
        return {"t": np.array(cols[0], dtype=np.float64), "event": np.array(cols[1], dtype=np.uint8),
                "request": np.array(cols[2], dtype=np.int64), "bike": np.array(cols[3], dtype=np.int32),
                "node": np.array(cols[4], dtype=np.int32), "outcome": np.array(cols[5], dtype=np.int8)}

    # ---------------------------------------------------------------------------------- loop
    def _load_slot(self, k: int) -> None:
        date = self.day0 + dt.timedelta(days=k // 24)
        req = self.demand.hour(date, k % 24)
        t = req.t_s + (k // 24) * 86400.0
        keep = t >= self.now
        n = int(keep.sum())
        rng = np.random.default_rng([self._user_seed, USER_SLOT_STREAM, k])
        u = rng.random(len(t))[keep]
        self._rq = list(zip(t[keep].tolist(), req.o_place[keep].tolist(), req.d_place[keep].tolist(),
                            req.o_node[keep].tolist(), req.d_node[keep].tolist(), req.ride_time_s[keep].tolist(),
                            req.kind[keep].tolist(), u.tolist()))
        self._ri, self._rn = 0, n

    def run(self, until: Until) -> list[Notification]:
        """Advance until a condition of ``until`` is met; returns the notifications emitted."""
        deadline = until.deadline(self.now)
        kinds = until.kinds()
        if deadline == math.inf and not kinds:
            raise ValueError("run(until=...) needs a time limit or a notification to stop on")
        heap = self.queue._heap
        debug = self.config.debug
        stop = False
        while not stop:
            th = heap[0][0] if heap else math.inf
            if self._ri < self._rn:
                tr = self._rq[self._ri][0]
                if tr < th or (tr == th and heap[0][1] > TRIP_REQUEST):
                    if tr > deadline:
                        break
                    self.now = tr
                    self._request(self._ri)
                    self._ri += 1
                    if debug:
                        self.check_invariants()
                    continue
            if th > deadline:
                break
            t, _, _, kind, a, b = self.queue.pop()
            self.now = t
            if kind == TRIP_END:
                self._trip_end(a)
            elif kind == TRIP_START:
                self._trip_start(a)
            elif kind == SLOT_START:
                self._load_slot(a)
                self.queue.push((a + 1) * 3600.0, SLOT_START, a + 1)
            elif kind == WAKEUP:
                note = Notification(t, "wakeup", {"tag": self._wake_tags[a]})
                self._notes.append(note)
                if "wakeup" in kinds:
                    stop = True
            if debug:
                self.check_invariants()
        if not stop and deadline != math.inf:
            self.now = max(self.now, deadline)
        notes, self._notes = self._notes, []
        return notes

    # -------------------------------------------------------------------------------- events
    def _fail(self, rid: int, node: int, reason: int) -> None:
        self.kpis_.failures[reason] += 1
        for p in self._parts.values():
            p.failures[p._g[node]][reason] += 1
        if self.log is not None:
            self.log.append((self.now, TRIP_REQUEST, rid, -1, node, reason))

    def _request(self, i: int) -> None:
        t, op, dp, on, dn, ride, pref, u = self._rq[i]
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
            bike = f.take_external(pref)
            if bike < 0:
                self._fail(rid, dn, EXTERNAL_EMPTY)
                return
            if f.kind_l[bike] == ELECTRIC:  # charged outside the city: fresh draw
                lo, hi = self.config.fleet.initial_battery
                f.battery_km[bike] = self._ext_rng.uniform(lo, hi) * self.battery.range_km
            k.served += 1
            k.from_external += 1
            self._trip_dest[bike] = dn
            self._trip_km[bike] = self.config.energy.external_trip_km
            self._trip_req[bike] = rid
            ride = ride if ride == ride else self.config.external.ride_time_s  # nan -> default
            self._trip_ride[bike] = ride
            if self.log is not None:
                self.log.append((t, TRIP_REQUEST, rid, bike, dn, SERVED))
            self.queue.push(t + ride, TRIP_END, bike)
            return
        if ride != ride:  # in-city trip without a ride time (no distances loaded): typical pace
            ride = float(self._dist[on, dn]) / self.config.energy.round_trip_speed_mps if dp != ext else 0.0
        if dp == ext:
            trip_km = self.config.energy.external_trip_km
        else:
            trip_km = float(self._dist[on, dn]) / 1000.0
        k.city_requests += 1
        for p in self._parts.values():
            p.requests[p._g[on]] += 1
        bike, walk, reason = self.users.choose(on, pref, self.battery.need_km(trip_km), u)
        if bike < 0:
            self._fail(rid, on, reason)
            return
        k.served += 1
        k.city_served += 1
        if walk > 0.0:
            k.walk_m_sum += walk
            k.walked += 1
        node = f.unpark(bike, RESERVED)
        if dp != ext:
            trip_km = float(self._dist[node, dn]) / 1000.0
            if op == dp:  # round trip by place: distance from the ride time at a typical pace
                trip_km = max(trip_km, ride * self.config.energy.round_trip_speed_mps / 1000.0)
        self._trip_dest[bike] = dn if dp != ext else -1
        self._trip_km[bike] = trip_km
        self._trip_ride[bike] = ride
        self._trip_req[bike] = rid
        if self.log is not None:
            self.log.append((t, TRIP_REQUEST, rid, bike, node, SERVED))
        self.queue.push(t + walk / self.config.users.walking_speed_mps, TRIP_START, bike)

    def _trip_start(self, bike: int) -> None:
        f = self.fleet
        if f.kind_l[bike] == ELECTRIC:
            before = f.battery_km[bike]
            f.battery_km[bike] = self.battery.after_ride(before, self._trip_km[bike])
            self.kpis_.ebike_km += self._trip_km[bike]
        if self._parts or self.log is not None:
            node = int(f.node[bike])  # where the bike was taken (kept after unpark)
            for p in self._parts.values():
                p.departures[p._g[node]] += 1
            if self.log is not None:
                self.log.append((self.now, TRIP_START, int(self._trip_req[bike]), bike, node, SERVED))
        if self._trip_dest[bike] < 0:  # to outside: the bike leaves the city now
            self.kpis_.to_external += 1
            f.to_external(bike)
            return
        self.kpis_.ride_km += self._trip_km[bike]
        f.set_status(bike, RIDING)
        self.queue.push(self.now + self._trip_ride[bike], TRIP_END, bike)

    def _trip_end(self, bike: int) -> None:
        node = int(self._trip_dest[bike])
        if self._docked and self.fleet.n_bikes[node] >= self._cap[node] and self._redirect(bike, node):
            return
        self.fleet.park(bike, node)
        self.kpis_.trips_ended += 1
        for p in self._parts.values():
            p.arrivals[p._g[node]] += 1
        if self.log is not None:
            self.log.append((self.now, TRIP_END, int(self._trip_req[bike]), bike, node, SERVED))

    def _redirect(self, bike: int, node: int) -> bool:
        """Docked mode, ``node`` full: ride on to the nearest station with a free dock.

        Returns False when every station is full (the bike is then left at ``node``)."""
        nb, cap = self.fleet.n_bikes, self._cap
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
            f.battery_km[bike] = self.battery.after_ride(f.battery_km[bike], extra_km)
            k.ebike_km += extra_km
        trip_m, ride = self._trip_km[bike] * 1000.0, self._trip_ride[bike]
        speed = trip_m / ride if trip_m > 0 and ride > 0 else REDIRECT_FALLBACK_SPEED_MPS
        self._trip_dest[bike] = alt
        self._trip_km[bike] = extra_km
        self._trip_ride[bike] = extra_m / speed
        if self.log is not None:
            self.log.append((self.now, REDIRECT, int(self._trip_req[bike]), bike, alt,
                             NO_DOCK if extra_m > self.config.users.redirect_radius_m else SERVED))
        self.queue.push(self.now + extra_m / speed, TRIP_END, bike)
        return True
