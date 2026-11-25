"""Demand generators: trip requests one hour at a time (issues #83, #84; spec §6.3).

Both generators return :class:`TripRequests`, a structure of arrays sorted by time. A request is a
user who **wants** to ride from ``origin`` to ``destination``; whether it is served is decided by
the simulator from its state (bikes nearby, battery, walking).

* :class:`PoissonDemand` samples from the fitted model:
  ``N ~ Poisson(day_factor × month_factor × Σ rates[g, h])``, OD pairs by one categorical draw,
  uniform start times within the hour, nodes through the spreader, ride times from the fitted
  model and the bike type preference from the observed e-bike share.
* :class:`ReplayDemand` (#84) replays recorded trips as requests.

**Reproducibility.** Every (date, hour) uses its own random stream derived from
``(seed, date, hour)``, and the day factor from ``(seed, date)``. A given hour is therefore the
same whatever was generated before it (spec §10).

**Places and nodes.** Place ``P-1`` is ``EXTERNAL``: a request from outside makes a bike
**appear** at its destination; a request to outside makes it **leave** the city. EXTERNAL ends
have node ``-1`` and no ride time (``nan``).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bss_sim.calendar import day_group

DAY_STREAM = 0xDA7
HOUR_STREAM = 0x40E


@dataclass(frozen=True)
class TripRequests:
    """Structure of arrays, sorted by ``t_s`` (seconds after local midnight of the date)."""

    t_s: np.ndarray            # float64
    o_place: np.ndarray        # int16
    d_place: np.ndarray        # int16
    o_node: np.ndarray         # int32, -1 for EXTERNAL
    d_node: np.ndarray         # int32, -1 for EXTERNAL
    ride_time_s: np.ndarray    # float32, nan when an end is EXTERNAL
    kind: np.ndarray           # int8 preferred bike type: 0 classic, 1 electric

    def __len__(self) -> int:
        return len(self.t_s)

    @staticmethod
    def concat(parts: list[TripRequests]) -> TripRequests:
        if not parts:
            return TripRequests(*(np.zeros(0, dtype=t) for t in
                                  (np.float64, np.int16, np.int16, np.int32, np.int32, np.float32, np.int8)))
        return TripRequests(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                              ("t_s", "o_place", "d_place", "o_node", "d_node", "ride_time_s", "kind")))


class RideTimes:
    """Reader of ``demand/ride_time.npz`` with the same semantics as ``bss_scenario``'s model."""

    def __init__(self, path: str | Path):
        with np.load(path) as z:
            self.bands_m = z["bands_m"]
            self.levels = z["levels"]
            self.speed_q = z["speed_q"]
            self.round_q = z["round_q"]

    def band(self, dist_m: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.bands_m, dist_m, side="right") - 1, 0, len(self.bands_m) - 2)

    def sample(self, dist_m: np.ndarray, kind: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Ride times (s) for uniforms ``u``; ``dist_m <= 0`` means a round trip."""
        out = np.empty(len(dist_m))
        rt = dist_m <= 0
        b = self.band(dist_m)
        for k in (0, 1):
            m = rt & (kind == k)
            out[m] = np.interp(u[m], self.levels, self.round_q[k])
            for j in range(self.speed_q.shape[1]):
                m = ~rt & (kind == k) & (b == j)
                if m.any():
                    out[m] = dist_m[m] / np.interp(u[m], self.levels, self.speed_q[k, j])
        return out

    def cdf(self, t_s, dist_m, kind) -> np.ndarray:
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
                    out[m] = 1.0 - np.interp(dist_m[m] / t_s[m], self.speed_q[k, j], self.levels, left=0.0, right=1.0)
        return out


class Spreader:
    """Place -> node sampling from ``demand/spread.npz`` (kernel) or the station node (docked)."""

    def __init__(self, path: str | Path, mode: str = "kernel"):
        with np.load(path) as z:
            indptr, nodes, probs = z["indptr"], z["nodes"], z["probs"].astype(np.float64)
            self.station_nodes = z["station_nodes"].astype(np.int32)
        if mode not in ("kernel", "station"):
            raise ValueError("mode must be 'kernel' or 'station'")
        self.mode = mode
        self.n = len(indptr) - 1
        if mode == "station":
            indptr = np.arange(self.n + 1)
            nodes = self.station_nodes
            probs = np.ones(self.n)
        self.indptr, self.nodes = indptr, nodes
        # cumulative probability inside each place, offset by the place index: place s spans (s, s+1]
        cum = np.empty_like(probs)
        for s in range(self.n):
            a, b = indptr[s], indptr[s + 1]
            c = np.cumsum(probs[a:b])
            cum[a:b] = s + c / c[-1]
        self._cum = cum

    def sample(self, place: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Node for each in-city place (place < n); callers handle EXTERNAL themselves."""
        idx = np.searchsorted(self._cum, place + u, side="right")
        idx = np.minimum(idx, self.indptr[place + 1] - 1)
        return self.nodes[idx].astype(np.int32)


@dataclass
class DemandModel:
    rates: np.ndarray           # float32 [G, 24, P, P]
    scheme: str
    holidays_as_sunday: bool
    month_factor: dict[str, float]
    sigma: float
    ebike_share: float
    spreader: Spreader
    ride_times: RideTimes
    bike_dist: np.ndarray | None  # float32 [N, N] or memmap; None -> ride times not sampled

    @property
    def n_places(self) -> int:
        return self.rates.shape[2]

    @property
    def external(self) -> int:
        return self.n_places - 1

    @classmethod
    def load(cls, bundle: str | Path, spread_mode: str = "kernel", with_distances: bool = True) -> DemandModel:
        root = Path(bundle)
        with np.load(root / "demand" / "shape.npz") as z:
            rates = z["rates"]
            meta = json.loads(str(z["meta"]))
        volume = json.loads((root / "demand" / "volume.json").read_text())
        dist = np.load(root / "network" / "bike_dist.npy", mmap_mode="r") if with_distances else None
        return cls(rates, meta["scheme"], bool(volume.get("holidays_as_sunday", True)), volume["month_factor"],
                   float(volume["sigma"]), float(volume.get("ebike_share", 0.0)),
                   Spreader(root / "demand" / "spread.npz", spread_mode),
                   RideTimes(root / "demand" / "ride_time.npz"), dist)

    def group(self, date: dt.date) -> int:
        return day_group(date, self.scheme, self.holidays_as_sunday)

    def month(self, date: dt.date) -> float:
        key = date.strftime("%Y-%m")
        if key in self.month_factor:
            return self.month_factor[key]
        same = [v for k, v in self.month_factor.items() if k[5:] == key[5:]]  # same month, other year
        if not same:
            raise KeyError(f"no volume factor for {key}")
        return float(np.mean(same))

    def nodes_and_times(self, o: np.ndarray, d: np.ndarray, kind: np.ndarray,
                        u_o: np.ndarray, u_d: np.ndarray, u_t: np.ndarray):
        ext = self.external
        o_node = np.full(len(o), -1, np.int32)
        d_node = np.full(len(d), -1, np.int32)
        m = o != ext
        o_node[m] = self.spreader.sample(o[m], u_o[m])
        m = d != ext
        d_node[m] = self.spreader.sample(d[m], u_d[m])
        ride = np.full(len(o), np.nan, np.float32)
        inside = (o != ext) & (d != ext)
        if self.bike_dist is not None and inside.any():
            dist = np.asarray(self.bike_dist[o_node[inside], d_node[inside]], dtype=float)
            dist[o[inside] == d[inside]] = 0.0  # round trip by place
            ride[inside] = self.ride_times.sample(dist, kind[inside], u_t[inside])
        return o_node, d_node, ride


class PoissonDemand:
    """Probabilistic generator (spec §6.3)."""

    def __init__(self, model: DemandModel, seed: int):
        self.model = model
        self.seed = int(seed)
        G, H, P, _ = model.rates.shape
        flat = model.rates.reshape(G, H, P * P).astype(np.float64)
        self._totals = flat.sum(axis=2)
        self._cum = np.cumsum(flat, axis=2) / np.maximum(self._totals[..., None], 1e-300)

    def day_factor(self, date: dt.date) -> float:
        """Month factor × log-normal day noise (mean 1), fixed for a given (seed, date)."""
        rng = np.random.default_rng([self.seed, DAY_STREAM, date.toordinal()])
        s = self.model.sigma
        return self.model.month(date) * float(np.exp(rng.normal(-0.5 * s * s, s)))

    def expected(self, date: dt.date, hour: int) -> float:
        return self.day_factor(date) * self._totals[self.model.group(date), hour]

    def hour(self, date: dt.date, hour: int) -> TripRequests:
        g = self.model.group(date)
        rng = np.random.default_rng([self.seed, HOUR_STREAM, date.toordinal(), hour])
        n = int(rng.poisson(self.day_factor(date) * self._totals[g, hour]))
        P = self.model.n_places
        cat = np.minimum(np.searchsorted(self._cum[g, hour], rng.random(n), side="right"), P * P - 1)
        o = (cat // P).astype(np.int16)
        d = (cat % P).astype(np.int16)
        t = hour * 3600.0 + rng.random(n) * 3600.0
        kind = (rng.random(n) < self.model.ebike_share).astype(np.int8)
        u = rng.random((3, n))
        o_node, d_node, ride = self.model.nodes_and_times(o, d, kind, u[0], u[1], u[2])
        order = np.argsort(t, kind="stable")
        return TripRequests(t[order], o[order], d[order], o_node[order], d_node[order], ride[order], kind[order])

    def day(self, date: dt.date) -> TripRequests:
        return TripRequests.concat([self.hour(date, h) for h in range(24)])
