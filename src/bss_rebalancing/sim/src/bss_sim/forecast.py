"""Forecasts of departures and arrivals per group (issues #128, #129; spec §8, §8.1).

``Simulator.forecast(window_s, partition, provider=...)`` predicts, for the window
``(now, now + window_s]`` and each group of a registered partition:

* ``departures``: requests that will start there (the demand: served or not);
* ``arrivals``: bikes that will arrive there = ``arrivals_new`` (from requests made in the window)
  + ``arrivals_inflight`` (trips already in progress, at the configured in-flight knowledge).

Providers (same output type, :class:`Forecast`):

* ``"mean"`` (:func:`mean_forecast`): expectations from the demand model, analytic;
* ``"sampled"`` (:func:`sampled_forecast`): ``K`` scenarios drawn with the run's independent
  ``forecast`` stream (arrays ``[K, G]``);
* ``"oracle"``: the simulator's own future requests and true destinations; ablation only, needs
  ``info.allow_oracle`` and is recorded in the run metadata.

**What the forecast knows.** The demand model (hourly OD rates, monthly level; the day's own noise
is unknown, so its mean 1 is used), the ride-time model, the spreader, the observed service level
of requests starting in the city and from outside (only served requests become arrivals), and the
trips in progress
at the configured level. It never reads the simulator's queue or its future requests (except the
oracle).

**Ride times between places** (:class:`RideTimeTable`): the generator draws a ride time from the
distance between the two *nodes* a request is spread to, so the distribution between two places is
a mixture over node pairs. It is tabulated once by sampling the generator's own code
(``DemandModel.nodes_and_times``), per bike type, origin and destination place, on a 60 s grid up
to 6 h.

**Trips in progress** (spec §8.1). A trip from place ``o`` started at ``s``, running for
``a = now - s``, arrives in group ``g`` within the window with probability

    sum_d p(d | o, s) [F_od(a + h) - F_od(a)] W[d, g]  /  sum_d p(d | o, s) [1 - F_od(a)]

with ``p(d | o, s)`` the demand shape at the start hour (destinations inside the city: a trip that
leaves the city is seen leaving), ``F`` the ride-time table for the bike's type and ``W[d, g]`` the
share of place ``d``'s nodes in group ``g``. Trips from outside arrive after the configured delay.
With ``count``, the composition of the trips in progress is the model's: trips started ``a``
seconds ago at rate ``lambda(now - a)`` and still running with probability ``1 - F(a)``; the
expected arrivals are scaled to the observed number of bikes in use. With ``none``, the same
expectation is used without the scaling.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

STEP_S = 60.0          # ride-time table grid
TMAX_S = 6 * 3600.0    # longest ride tabulated (F = 1 beyond)
QUAD_S = 120.0         # quadrature step over start times


@dataclass(frozen=True)
class Forecast:
    t: float
    window_s: float
    partition: str
    provider: str
    inflight_knowledge: str
    departures: np.ndarray          # [G] expected, or [K, G] scenarios
    arrivals: np.ndarray            # arrivals_new + arrivals_inflight
    arrivals_new: np.ndarray
    arrivals_inflight: np.ndarray


class RideTimeTable:
    """CDF of ride times between places, ``F[kind, o, d, t / STEP_S]``, for in-city places."""

    def __init__(self, model, samples: int = 128, seed: int = 0x7AB1E):
        P = model.n_places - 1
        rng = np.random.default_rng(seed)
        o, d = np.divmod(np.arange(P * P), P)
        n_grid = int(TMAX_S / STEP_S) + 1
        F = np.empty((2, P, P, n_grid), dtype=np.float32)
        for k in (0, 1):
            oo = np.repeat(o, samples).astype(np.int16)
            dd = np.repeat(d, samples).astype(np.int16)
            kk = np.full(len(oo), k, dtype=np.int8)
            u = rng.random((3, len(oo)))
            _, _, ride = model.nodes_and_times(oo, dd, kk, u[0], u[1], u[2])
            bins = np.minimum(np.ceil(np.asarray(ride, dtype=np.float64) / STEP_S), n_grid).astype(np.int64)
            counts = np.zeros((P * P, n_grid + 1))
            np.add.at(counts, (np.repeat(np.arange(P * P), samples), bins), 1.0)
            F[k] = (np.cumsum(counts, axis=1)[:, :n_grid] / samples).reshape(P, P, n_grid)
        self.F = F
        self.n_grid = n_grid

    def at(self, t: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grid index and interpolation weight for times ``t`` (clamped to the table)."""
        x = np.clip(np.asarray(t, dtype=np.float64) / STEP_S, 0.0, self.n_grid - 1.000001)
        i = x.astype(np.int64)
        return i, i + 1, x - i

    def cdf(self, kind, o, d, t) -> np.ndarray:
        i, j, w = self.at(t)
        F = self.F
        return F[kind, o, d, i] * (1 - w) + F[kind, o, d, j] * w

    def cdf_all(self, t: float, ebike_share: float) -> np.ndarray:
        """[P, P] mixture over bike types at a single time ``t``."""
        i, j, w = self.at(t)
        Fk = self.F[:, :, :, int(i)] * (1 - float(w)) + self.F[:, :, :, int(j)] * float(w)
        return (1 - ebike_share) * Fk[0] + ebike_share * Fk[1]

    def inverse(self, kind: int, o: int, d: int, u: float) -> float:
        """Ride time with CDF value ``u`` (linear inverse on the grid); ``inf`` beyond the table
        (a ride longer than ``TMAX_S``: it does not end within any forecast window)."""
        row = self.F[kind, o, d]
        j = int(np.searchsorted(row, u, side="left"))
        if j <= 0:
            return 0.0
        if j >= self.n_grid:
            return float("inf")
        lo, hi = float(row[j - 1]), float(row[j])
        frac = (u - lo) / (hi - lo) if hi > lo else 0.0
        return (j - 1 + frac) * STEP_S


class ForecastContext:
    """Per-simulator precomputation shared by the providers."""

    def __init__(self, sim):
        self.sim = sim
        m = sim.scenario.demand
        self.model = m
        self.P = m.n_places - 1          # in-city places
        self.ext = m.external
        self._table: RideTimeTable | None = None
        self._W: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def table(self) -> RideTimeTable:
        if self._table is None:
            self._table = RideTimeTable(self.model)
        return self._table

    def W(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(``W[place, group]`` share of each in-city place's spread nodes per group, node -> group)."""
        if name not in self._W:
            part = self.sim._parts[name]
            sp = self.model.spreader
            place = np.repeat(np.arange(sp.n), np.diff(sp.indptr))
            g = part.node_to_group[sp.nodes]
            keep = g >= 0
            W = np.zeros((sp.n, part.n_groups))
            np.add.at(W, (place[keep], g[keep]), sp.probs[keep])
            self._W[name] = (W, part.node_to_group)
        return self._W[name]

    def rates_at(self, t: float) -> np.ndarray:
        """Expected requests per second ``[P+1, P+1]`` at clock time ``t`` (day noise at its mean)."""
        sim = self.sim
        day, sec = divmod(t, 86400.0)
        date = sim.day0 + dt.timedelta(days=int(day))
        m = self.model
        return m.rates[m.group(date), int(sec // 3600)].astype(np.float64) * m.month(date) / 3600.0

    def served_fraction(self) -> float:
        """Observed share of requests starting in the city that got a bike (1 before any request)."""
        k = self.sim.kpis_
        return k.city_served / k.city_requests if k.city_requests else 1.0

    def served_fraction_external(self) -> float:
        """Observed share of requests from outside that got a bike (the reservoir can run empty)."""
        k = self.sim.kpis_
        n = k.requests - k.city_requests - k.skipped_external_both
        return k.from_external / n if n else 1.0

    def ext_delay(self) -> float:
        return self.sim.config.external.ride_time_s

    def ebike_share(self) -> float:
        return self.sim._ebike_share


def _starts(now: float, window: float) -> tuple[np.ndarray, float]:
    n = max(1, int(np.ceil(window / QUAD_S)))
    step = window / n
    return now + (np.arange(n) + 0.5) * step, step


def mean_forecast(ctx: ForecastContext, window: float, partition: str, level: str) -> Forecast:
    sim, P, ext = ctx.sim, ctx.P, ctx.ext
    W, _ = ctx.W(partition)
    G = W.shape[1]
    now, end = sim.now, sim.now + window
    sf, e, D = ctx.served_fraction(), ctx.ebike_share(), ctx.ext_delay()
    dep = np.zeros(G)
    new = np.zeros(G)
    starts, step = _starts(now, window)
    for s in starts:
        lam = ctx.rates_at(s) * step
        dep += lam[:P].sum(axis=1) @ W
        Fm = ctx.table.cdf_all(end - s, e)
        new += (sf * (lam[:P, :P] * Fm).sum(axis=0)) @ W
        if end - s >= D:
            new += ctx.served_fraction_external() * lam[ext, :P] @ W
    infl = inflight_mean(ctx, window, partition, level)
    return Forecast(now, window, partition, "mean", level, dep, new + infl, new, infl)


def _model_inflight(ctx: ForecastContext, window: float, W: np.ndarray) -> tuple[np.ndarray, float]:
    """Expected arrivals per group from trips in progress, and their expected number, under the model."""
    sim, P, ext = ctx.sim, ctx.P, ctx.ext
    now = sim.now
    sf, e, D = ctx.served_fraction(), ctx.ebike_share(), ctx.ext_delay()
    arr = np.zeros(W.shape[1])
    n_in = 0.0
    ages = (np.arange(int(TMAX_S / QUAD_S)) + 0.5) * QUAD_S
    for a in ages:
        if now - a < 0:          # before the start of the run nothing was riding
            break
        lam = ctx.rates_at(now - a) * QUAD_S
        Fa = ctx.table.cdf_all(a, e)
        Fah = ctx.table.cdf_all(a + window, e)
        city = sf * lam[:P, :P]
        n_in += float((city * (1 - Fa)).sum())
        arr += (city * (Fah - Fa)).sum(axis=0) @ W
        if a < D:                # from outside: in flight until D, arrives within the window if D - a <= h
            ext_lam = ctx.served_fraction_external() * lam[ext, :P]
            n_in += float(ext_lam.sum())
            if D - a <= window:
                arr += ext_lam @ W
    return arr, n_in


def inflight_mean(ctx: ForecastContext, window: float, partition: str, level: str) -> np.ndarray:
    """Expected arrivals per group, within the window, of the trips in progress now."""
    sim, P, ext = ctx.sim, ctx.P, ctx.ext
    W, node_group = ctx.W(partition)
    G = W.shape[1]
    if level in ("none", "count"):
        arr, n_in = _model_inflight(ctx, window, W)
        if level == "count":
            count = sim._inflight("count").count
            return arr * (count / n_in) if n_in > 0 else np.zeros(G)
        return arr
    fl = sim._inflight(level)
    out = np.zeros(G)
    if fl.count == 0:
        return out
    now, end = sim.now, sim.now + window
    if level == "destination":
        g = node_group[fl.d_node]
        hit = (fl.end_t <= end) & (g >= 0)
        np.add.at(out, g[hit], 1.0)
        return out
    D = ctx.ext_delay()
    for o, s, k in zip(fl.o_place.tolist(), fl.start_t.tolist(), fl.kind.tolist()):
        lam = ctx.rates_at(s)
        if o == ext:
            if now < s + D <= end:
                p = lam[ext, :P]
                if p.sum() > 0:
                    out += (p / p.sum()) @ W
            continue
        p = lam[o, :P]
        a = now - s
        Fa = ctx.table.cdf(k, o, np.arange(P), np.full(P, a))
        Fah = ctx.table.cdf(k, o, np.arange(P), np.full(P, a + window))
        alive = (p * (1 - Fa)).sum()
        if alive > 0:
            out += (p * (Fah - Fa) / alive) @ W
    return out


def sampled_forecast(ctx: ForecastContext, window: float, partition: str, level: str, K: int,
                     rng: np.random.Generator) -> Forecast:
    """``K`` scenarios of the window, drawn with the run's ``forecast`` stream."""
    from bss_sim.demand import PoissonDemand

    sim, ext = ctx.sim, ctx.ext
    W, node_group = ctx.W(partition)
    G = W.shape[1]
    now, end = sim.now, sim.now + window
    sf, sf_ext, D = ctx.served_fraction(), ctx.served_fraction_external(), ctx.ext_delay()
    dep = np.zeros((K, G))
    new = np.zeros((K, G))
    infl = np.zeros((K, G))
    first, last = int(now // 3600), int(np.ceil(end / 3600))
    for k in range(K):
        gen = PoissonDemand(ctx.model, seed=int(rng.integers(2**63)))   # own day level and requests
        for slot in range(first, last):
            date = sim.day0 + dt.timedelta(days=slot // 24)
            r = gen.hour(date, slot % 24)
            t = r.t_s + (slot // 24) * 86400.0
            win = (t > now) & (t <= end)
            city = win & (r.o_place != ext)
            g = node_group[r.o_node[city]]
            np.add.at(dep[k], g[g >= 0], 1.0)
            u = rng.random(len(t))
            served = city & (r.d_place != ext) & (u < sf)
            from_out = win & (r.o_place == ext) & (r.d_place != ext) & (u < sf_ext)
            arr_t = np.where(r.o_place == ext, t + np.where(np.isfinite(r.ride_time_s), r.ride_time_s, D),
                             t + np.nan_to_num(r.ride_time_s))
            hit = (served | from_out) & (arr_t <= end)
            g = node_group[r.d_node[hit]]
            np.add.at(new[k], g[g >= 0], 1.0)
        infl[k] = _inflight_sample(ctx, window, partition, level, rng)
    return Forecast(now, window, partition, "sampled", level, dep, new + infl, new, infl)


def _place_node(ctx: ForecastContext, place: int, rng: np.random.Generator) -> int:
    return int(ctx.model.spreader.sample(np.array([place]), rng.random(1))[0])


def _inflight_sample(ctx: ForecastContext, window: float, partition: str, level: str,
                     rng: np.random.Generator) -> np.ndarray:
    sim, P, ext = ctx.sim, ctx.P, ctx.ext
    W, node_group = ctx.W(partition)
    out = np.zeros(W.shape[1])
    now, end = sim.now, sim.now + window
    D = ctx.ext_delay()

    def land(place: int) -> None:
        g = node_group[_place_node(ctx, place, rng)]
        if g >= 0:
            out[g] += 1.0

    if level == "destination":
        fl = sim._inflight(level)
        g = node_group[fl.d_node]
        hit = (fl.end_t <= end) & (g >= 0)
        np.add.at(out, g[hit], 1.0)
        return out
    if level == "origin":
        fl = sim._inflight(level)
        for o, s, k in zip(fl.o_place.tolist(), fl.start_t.tolist(), fl.kind.tolist()):
            lam = ctx.rates_at(s)
            if o == ext:
                p = lam[ext, :P]
                if now < s + D <= end and p.sum() > 0:
                    land(int(rng.choice(P, p=p / p.sum())))
                continue
            a = now - s
            p = lam[o, :P]
            Fa = ctx.table.cdf(k, o, np.arange(P), np.full(P, a))
            w = p * (1 - Fa)
            if w.sum() <= 0:
                continue
            d = int(rng.choice(P, p=w / w.sum()))
            u = float(Fa[d]) + rng.random() * (1.0 - float(Fa[d]))   # ride time given still riding at a
            if ctx.table.inverse(k, o, d, u) - a <= window:
                land(d)
        return out
    # count / none: trips in progress drawn from the model's composition
    comp, p_arr = _composition(ctx, window)
    total = comp.sum()
    if total <= 0:
        return out
    n = sim._inflight("count").count if level == "count" else int(rng.poisson(total))
    idx = rng.choice(len(comp), size=n, p=comp / total)
    for i in idx.tolist():
        if rng.random() < p_arr[i]:
            land(i % (P + 1) if i % (P + 1) < P else 0)
    return out


def _composition(ctx: ForecastContext, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Expected trips in progress by destination-tagged cell, and each cell's arrival probability.

    Cells are ``age x origin x destination`` flattened with the destination last (so ``i % (P+1)``
    is the destination place); origin ``EXTERNAL`` included."""
    sim, P, ext = ctx.sim, ctx.P, ctx.ext
    sf, e, D = ctx.served_fraction(), ctx.ebike_share(), ctx.ext_delay()
    comp, parr = [], []
    ages = (np.arange(int(TMAX_S / QUAD_S)) + 0.5) * QUAD_S
    for a in ages:
        if sim.now - a < 0:
            break
        lam = ctx.rates_at(sim.now - a) * QUAD_S
        Fa = ctx.table.cdf_all(a, e)
        Fah = ctx.table.cdf_all(a + window, e)
        c = np.zeros((P + 1, P + 1))
        q = np.zeros((P + 1, P + 1))
        c[:P, :P] = sf * lam[:P, :P] * (1 - Fa)
        with np.errstate(invalid="ignore", divide="ignore"):
            q[:P, :P] = np.where(1 - Fa > 0, (Fah - Fa) / (1 - Fa), 0.0)
        if a < D:
            c[ext, :P] = ctx.served_fraction_external() * lam[ext, :P]
            q[ext, :P] = 1.0 if D - a <= window else 0.0
        comp.append(c.ravel())
        parr.append(q.ravel())
    if not comp:
        return np.zeros(1), np.zeros(1)
    return np.concatenate(comp), np.concatenate(parr)


def oracle_forecast(ctx: ForecastContext, window: float, partition: str) -> Forecast:
    """The simulator's own future requests (served or not) and the true trips in progress."""
    sim, ext = ctx.sim, ctx.ext
    W, node_group = ctx.W(partition)
    G = W.shape[1]
    now, end = sim.now, sim.now + window
    dep, new = np.zeros(G), np.zeros(G)
    if sim.demand is not None:
        for slot in range(int(now // 3600), int(np.ceil(end / 3600))):
            date = sim.day0 + dt.timedelta(days=slot // 24)
            r = sim.demand.hour(date, slot % 24)
            t = r.t_s + (slot // 24) * 86400.0
            win = (t > now) & (t <= end)
            city = win & (r.o_place != ext)
            g = node_group[r.o_node[city]]
            np.add.at(dep, g[g >= 0], 1.0)
            ride = np.where(np.isfinite(r.ride_time_s), r.ride_time_s, ctx.ext_delay())
            hit = win & (r.d_place != ext) & (t + ride <= end)
            g = node_group[r.d_node[hit]]
            np.add.at(new, g[g >= 0], 1.0)
    infl = _inflight_sample(ctx, window, partition, "destination", np.random.default_rng(0))
    return Forecast(now, window, partition, "oracle", "destination", dep, new + infl, new, infl)
