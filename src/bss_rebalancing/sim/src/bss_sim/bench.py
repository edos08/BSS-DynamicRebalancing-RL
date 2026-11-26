"""Performance benchmarks and budget (issue #105, spec §12).

```
python -m bss_sim.bench --make-synthetic /tmp/synth        # once: Cambridge-size synthetic bundle
python -m bss_sim.bench /tmp/synth --check --json bench.json
python -m bss_sim.bench data/scenarios/cambridge-2024      # real bundle (numbers for the report)
```

Measured, single core, default configuration (dockless, 800 bikes, logging off):

* ``load_s``: ``Simulator.from_scenario`` (budget 2 s);
* ``week_s``: one simulated week from Monday 00:00, best of ``--repeat`` runs (budget 0.5 s);
* ``peak_mb``: peak resident memory of this process after loading one simulator and running the
  weeks above (budget 600 MB; measured before the other benchmarks create more simulators);
* ``week_docked_s``, ``week_partition_s``: docked mode (800 bikes, or 44 % of the docks if fewer,
  as in Cambridge), and dockless with a 300 m-like grid partition (reported, no budget);
* ``threads_speedup``: ``--threads`` simulators in parallel threads vs one after the other
  (reported; > 1 only on free-threaded Python).

* ``week_truck_s``: one week with one truck; every 5 min a controller wakes up and gives it a random
  stop and a random service (pick up, drop or swap 2), replacing its queue (budget 1 s, #126).
``--check`` exits with status 1 when a budget is exceeded.

**Machine speed.** Shared CI runners differ by up to 2x in speed between runs of the same code.
Every run therefore also times a fixed pure-Python **reference workload** (``calib_s``: heap, list
and dict operations like the simulator's hot path) and the budget is checked on times scaled to
the reference machine: ``scaled = measured x REFERENCE_CALIB_S / calib_s``. Raw times are always
reported next to the scaled ones. ``REFERENCE_CALIB_S`` is the median calibration on GitHub's
ubuntu-latest runners with Python 3.11 (see docs/sim-v2/reports/performance.md).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import sysconfig
import threading
import time
from pathlib import Path

import numpy as np

from bss_sim.commands import Drop, GoTo, PickUp, SwapBattery
from bss_sim.config import SimConfig
from bss_sim.engine import At, OnNotification, TimeLimit
from bss_sim.simulator import Simulator

BUDGET = {"load_s": 2.0, "week_s": 0.5, "week_truck_s": 1.0, "peak_mb": 600.0}
SCALED = ("load_s", "week_s", "week_truck_s")  # time budgets are checked on machine-scaled times
REFERENCE_CALIB_S = 0.115           # GitHub ubuntu-latest, Python 3.11 (CI run 45809102526, #105)
START = "2024-09-16T00:00"
WEEK = 7 * 86400.0


def _peak_mb() -> float:
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 if sys.platform != "darwin" else rss / 1024.0 / 1024.0


def _week(sim: Simulator, seed: int) -> float:
    sim.reset(seed=seed, start=START)
    t = time.perf_counter()
    sim.run(TimeLimit(WEEK))
    return time.perf_counter() - t


def _week_truck(sim: Simulator, seed: int, every_s: float = 300.0) -> float:
    """One week with truck 0 commanded at random every ``every_s`` (controller time included)."""
    rng = np.random.default_rng(seed)
    sim.reset(seed=seed, start=START)
    n = sim.scenario.n_nodes
    end = sim.now + WEEK
    t = time.perf_counter()
    sim.wakeup(after=every_s)
    while sim.now < end:
        notes = sim.run(OnNotification("wakeup") | At(end))
        if any(nt.kind == "wakeup" for nt in notes):
            service = (PickUp(2), Drop(2), SwapBattery(2))[int(rng.integers(3))]
            sim.submit(0, [GoTo(int(rng.integers(n))), service], replace=True)
            sim.wakeup(after=every_s)
    return time.perf_counter() - t


def calibrate(repeat: int = 5, n: int = 300_000) -> float:
    """Best time of a fixed pure-Python workload (seconds): measures this machine and interpreter."""
    import heapq

    best = float("inf")
    for _ in range(repeat):
        heap: list = []
        lst = [0] * 4096
        d: dict = {}
        t = time.perf_counter()
        for i in range(n):
            k = (i * 2654435761) & 4095
            heapq.heappush(heap, (k * 0.5, i & 7, i))
            lst[k] += 1
            d[k] = lst[k]
            if len(heap) > 512:
                heapq.heappop(heap)
        best = min(best, time.perf_counter() - t)
    return best


def grid_partition(sim: Simulator, cells: int = 500) -> np.ndarray:
    """A partition with about ``cells`` groups of consecutive node ids (stands in for a 300 m grid)."""
    n = sim.scenario.n_nodes
    return (np.arange(n) * cells // n).astype(np.int64)


def run(bundle: str | Path, repeat: int = 3, threads: int = 4) -> dict:
    out: dict = {"bundle": str(bundle), "python": platform.python_version(),
                 "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
                 "numpy": np.__version__, "machine": platform.machine(), "repeat": repeat}
    out["calib_s"] = calibrate()
    out["speed_factor"] = (REFERENCE_CALIB_S / out["calib_s"]) if REFERENCE_CALIB_S else 1.0
    t = time.perf_counter()
    sim = Simulator.from_scenario(bundle)
    out["load_s"] = time.perf_counter() - t
    out["week_s"] = min(_week(sim, s) for s in range(repeat))
    k = sim.kpis()
    out["requests_per_week"] = k["requests"]
    out["us_per_request"] = out["week_s"] / max(k["requests"], 1) * 1e6
    out["peak_mb"] = _peak_mb()

    out["week_truck_s"] = min(_week_truck(sim, s) for s in range(repeat))
    tk = sim.truck_kpis(0)
    out["truck_week"] = {"commands": tk["commands"], "km": round(tk["km"], 1), "kwh": round(tk["kwh"], 2)}
    docks = int(sim.scenario.station_capacity.sum())
    size = min(SimConfig().fleet.size, int(0.44 * docks))  # Cambridge: 800 bikes for 1,824 docks
    docked = Simulator.from_scenario(bundle, SimConfig.from_dict({"mode": "docked", "fleet": {"size": size}}))
    out["week_docked_s"] = min(_week(docked, s) for s in range(repeat))
    sim.add_partition("grid", grid_partition(sim))
    out["week_partition_s"] = min(_week(sim, s) for s in range(repeat))

    if threads > 1:
        sims = [Simulator.from_scenario(bundle) for _ in range(threads)]
        t = time.perf_counter()
        for i, s in enumerate(sims):
            _week(s, i)
        serial = time.perf_counter() - t
        t = time.perf_counter()
        ts = [threading.Thread(target=_week, args=(s, i)) for i, s in enumerate(sims)]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        parallel = time.perf_counter() - t
        out["threads"] = threads
        out["threads_speedup"] = serial / parallel
    out["calib_s_after"] = calibrate()  # the machine may change speed during the run: report both
    for k in SCALED:
        out[f"{k}_scaled"] = out[k] * out["speed_factor"]
    out["budget"] = BUDGET
    out["over_budget"] = [k for k, v in BUDGET.items() if out.get(f"{k}_scaled", out[k]) > v]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", type=Path)
    ap.add_argument("--make-synthetic", type=Path, metavar="DIR", help="write a synthetic bundle and exit")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", type=Path, help="write the results here")
    ap.add_argument("--check", action="store_true", help="exit 1 if a budget is exceeded")
    args = ap.parse_args(argv)
    if args.make_synthetic:
        from bss_sim.synthetic import make_bundle
        t = time.perf_counter()
        make_bundle(args.make_synthetic)
        print(f"synthetic bundle written to {args.make_synthetic} in {time.perf_counter() - t:.1f} s")
        return 0
    if args.bundle is None:
        ap.error("bundle required (or --make-synthetic DIR)")
    res = run(args.bundle, args.repeat, args.threads)
    text = json.dumps(res, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    if args.check and res["over_budget"]:
        print(f"over budget: {', '.join(res['over_budget'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
