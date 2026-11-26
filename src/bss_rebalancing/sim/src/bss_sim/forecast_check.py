"""Forecast calibration check (issue #130, spec §8.1): forecasts versus what then happens.

Used by ``scripts/validate_forecast.py`` (report on real bundles) and by the CI tests (synthetic
bundle). Every ``every`` seconds between 06:00 and 22:00 of each simulated day, the forecasts of
the next ``window`` seconds are recorded for every station catchment; after the run they are
compared with the event log:

* arrivals of the trips in progress for the in-flight levels ``none``, ``count``, ``origin``
  (mean forecast) and ``destination`` (exact: the reference);
* arrivals from requests made in the window, and departures (requests; reference = the oracle).

A trip is in progress at ``t`` if it started riding by ``t`` (a user still walking to the bike is
not; trips from outside start at their request).
"""

from __future__ import annotations

import numpy as np

from bss_sim import forecast as fc
from bss_sim.config import SimConfig
from bss_sim.engine import TRIP_END, TRIP_REQUEST, TRIP_START, At
from bss_sim.simulator import Simulator


def catchments(sim: Simulator) -> np.ndarray:
    sp = sim.scenario.demand.spreader
    best = np.full(sim.scenario.n_nodes, -1.0)
    group = np.full(sim.scenario.n_nodes, -1, dtype=np.int64)
    place = np.repeat(np.arange(sp.n), np.diff(sp.indptr))
    for p, node, w in zip(place, sp.nodes, sp.probs):
        if w > best[node]:
            best[node], group[node] = w, p
    return group


def collect(bundle, generator: str, days: int, start: str, window: float, every: float, samples: int,
            seed: int, acc: dict, sim_config: dict | None = None) -> None:
    """One run of ``days`` days; appends (predicted, realised) pairs and PIT values to ``acc``."""
    base = {"demand": {"generator": generator}, "trucks": {"count": 0},
            "info": {"inflight_knowledge": "destination", "allow_oracle": True}, "telemetry": {"event_log": True}}
    for k, v in (sim_config or {}).items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    cfg = SimConfig.from_dict(base)
    sim = Simulator.from_scenario(bundle, cfg)
    groups = catchments(sim)
    sim.add_partition("catchments", groups)
    sim.reset(seed=seed, start=start)
    sim._fc = fc.ForecastContext(sim)
    ctx = sim._fc
    G = int(groups.max()) + 1
    t0 = sim.now
    times = [t0 + d * 86400 + h for d in range(days) for h in np.arange(6 * 3600, 22 * 3600, every)]
    rows = []
    rng = np.random.default_rng(seed + 1)
    for t in times:
        sim.run(At(t))
        rid0 = sim._req_id
        pred = {lv: fc.inflight_mean(ctx, window, "catchments", lv) for lv in ("none", "count", "origin", "destination")}
        mean = fc.mean_forecast(ctx, window, "catchments", "origin")
        samp = np.stack([fc._inflight_sample(ctx, window, "catchments", "origin", rng) for _ in range(samples)])
        orc = fc.oracle_forecast(ctx, window, "catchments")
        rows.append((t, rid0, pred, mean, samp, orc))
    sim.run(At(times[-1] + window + 1))
    log = sim.event_log()
    ends = log["event"] == TRIP_END
    # when each trip started riding (TripStart; trips from outside start at their request)
    started = {int(r): float(tt) for r, tt in zip(log["request"][log["event"] == TRIP_START],
                                                  log["t"][log["event"] == TRIP_START])}
    req_t = {int(r): float(tt) for r, tt in zip(log["request"][log["event"] == TRIP_REQUEST], log["t"][log["event"] == TRIP_REQUEST])}
    out = acc.setdefault("inflight", {"none": [], "count": [], "origin": [], "destination": []})
    pairs = {"new_arrivals": acc.setdefault("new_arrivals", []), "departures": acc.setdefault("departures", [])}
    for t, rid0, pred, mean, samp, orc in rows:
        m = ends & (log["t"] > t) & (log["t"] <= t + window)
        g = groups[log["node"][m]]
        # in progress at t = started riding by t (a user still walking to the bike is not)
        old = np.array([started.get(int(r), req_t.get(int(r), np.inf)) <= t for r in log["request"][m]], dtype=bool)
        real_in = np.bincount(g[old & (g >= 0)], minlength=G).astype(float)
        real_new = np.bincount(g[~old & (g >= 0)], minlength=G).astype(float)
        for lv in out:
            out[lv].append((pred[lv], real_in))
        pairs["new_arrivals"].append((mean.arrivals_new, real_new))
        pairs["departures"].append((mean.departures, orc.departures))
        # randomised PIT per group (counts are discrete): u = P(X < r) + V P(X = r)
        active = (samp.sum(axis=0) > 0) | (real_in > 0)
        u = (samp < real_in).mean(axis=0) + rng.random(G) * (samp == real_in).mean(axis=0)
        acc.setdefault("pit_cells", []).append(u[active])
        tot = samp.sum(axis=1)
        r = real_in.sum()
        acc.setdefault("pit_total", []).append(float((tot < r).mean() + rng.random() * (tot == r).mean()))
        acc.setdefault("block", []).append(seed * 1000 + int((t - t0) // 86400))   # (seed, day)
    acc["groups"] = G
    acc["windows"] = acc.get("windows", 0) + len(rows)


def summarise(acc: dict, meta: dict) -> dict:
    block = np.array(acc["block"])

    def metrics(ps):
        p = np.array([a.sum() for a, _ in ps])
        r = np.array([b.sum() for _, b in ps])
        err = p - r
        # windows of one day share its demand level: standard error over (seed, day) blocks
        means = np.array([err[block == b].mean() for b in np.unique(block)])
        mae = float(np.mean([np.abs(a - b).sum() for a, b in ps]))
        return {"mean_pred": float(p.mean()), "mean_real": float(r.mean()), "bias": float(err.mean()),
                "bias_se": float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float("nan"),
                "mae_groups": mae,
                "rel_bias": float(err.mean() / max(r.mean(), 1e-9))}

    cells = np.concatenate(acc["pit_cells"])
    return {**meta, "windows": acc["windows"], "groups": acc["groups"],
            "inflight": {lv: metrics(v) for lv, v in acc["inflight"].items()},
            "new_arrivals": metrics(acc["new_arrivals"]), "departures": metrics(acc["departures"]),
            "coverage": {"50": float(((cells >= 0.25) & (cells <= 0.75)).mean()),
                         "90": float(((cells >= 0.05) & (cells <= 0.95)).mean()), "cells": int(len(cells))},
            "pit_total_hist": np.histogram(acc["pit_total"], bins=10, range=(0, 1))[0].tolist()}


def report_md(results: list[dict]) -> str:
    """Markdown tables for ``summarise`` results (one column block per run)."""
    def pct(x):
        return f"{100 * x:+.1f} %"

    lines = ["| run | windows | level | mean realised | bias (± s.e.) | relative | MAE per window (sum over groups) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        name = r.get("name", r.get("generator", ""))
        for lv in ("none", "count", "origin", "destination"):
            m = r["inflight"][lv]
            lines.append(f"| {name} | {r['windows']} | `{lv}` | {m['mean_real']:.1f} | {m['bias']:+.2f} (± {m['bias_se']:.2f}) "
                         f"| {pct(m['rel_bias'])} | {m['mae_groups']:.1f} |")
    lines += ["", "| run | quantity | mean predicted | mean realised | bias (± s.e.) | relative |", "|---|---|---|---|---|---|"]
    for r in results:
        name = r.get("name", r.get("generator", ""))
        for q in ("departures", "new_arrivals"):
            m = r[q]
            lines.append(f"| {name} | {q.replace('_', ' ')} | {m['mean_pred']:.1f} | {m['mean_real']:.1f} "
                         f"| {m['bias']:+.2f} (± {m['bias_se']:.2f}) | {pct(m['rel_bias'])} |")
    lines += ["", "| run | inside 50 % interval | inside 90 % interval | group-windows | PIT of the total (10 bins) |",
              "|---|---|---|---|---|"]
    for r in results:
        c = r["coverage"]
        lines.append(f"| {r.get('name', r.get('generator', ''))} | {100 * c['50']:.1f} % | {100 * c['90']:.1f} % "
                     f"| {c['cells']:,} | {' '.join(str(x) for x in r['pit_total_hist'])} |")
    return "\n".join(lines)
