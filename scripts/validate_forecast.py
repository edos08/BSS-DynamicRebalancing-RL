"""Calibration of the forecasts (issue #130, spec §8.1).

Usage:
    python scripts/validate_forecast.py <bundle> [--generator poisson|replay] [--days 3] [--start 2024-09-16]
                                        [--window 1800] [--samples 50] [--out docs/sim-v2/reports]

Runs the simulator (no trucks) and, every ``--every`` seconds between 06:00 and 22:00, forecasts
the next ``--window`` seconds for every station catchment (each node in the group of the place
that spreads to it most). Then it compares with what actually happened (event log):

* **arrivals of trips in progress** (the part that depends on in-flight knowledge) for the levels
  ``none``, ``count``, ``origin`` (mean forecast) and ``destination`` (oracle, exact by
  construction: the reference for sharpness);
* **arrivals from new requests** and **departures** (requests) in the window.

Metrics: bias of the city-wide total (mean predicted - realised, with its standard error over
(seed, day) blocks, because the windows of a day share its demand level), mean
absolute error per group, and, for the sampled ``origin`` forecast, the share of realised values
inside the central 50 % and 90 % intervals (per group, pooled, from the randomised PIT because
counts are discrete) and the PIT histogram of the total.
Writes ``forecast-calibration[-<generator>].md`` and ``.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from bss_sim.forecast_check import collect, report_md, summarise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, nargs="?")
    ap.add_argument("--report", type=Path, nargs="+", metavar="JSON",
                    help="instead of running: print the Markdown tables of these result files")
    ap.add_argument("--generator", default="poisson", choices=["poisson", "replay"])
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--start", default="2024-09-16T00:00")
    ap.add_argument("--window", type=float, default=1800.0)
    ap.add_argument("--every", type=float, default=1200.0)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=1, help="independent runs pooled")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.report:
        print(report_md([{**json.loads(p.read_text()), "name": p.stem.replace("forecast-calibration-", "")}
                         for p in args.report]))
        return 0
    t = dt.datetime.now()
    acc: dict = {}
    for seed in range(args.seeds):
        collect(args.bundle, args.generator, args.days, args.start, args.window, args.every, args.samples, seed, acc)
    meta = {"bundle": str(args.bundle), "generator": args.generator, "days": args.days, "start": args.start,
            "seeds": args.seeds, "window_s": args.window, "every_s": args.every, "samples": args.samples}
    res = summarise(acc, meta)
    res["runtime_s"] = (dt.datetime.now() - t).total_seconds()
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"forecast-calibration-{args.generator}{'-' + args.tag if args.tag else ''}.json"
        (args.out / name).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
