"""Validate the demand generators on held-out days (issue #85).

Usage:
    python scripts/validate_demand.py data/scenarios/cambridge-2024 --out docs/sim-v2/reports

1. Split the fit-period days exactly as the model selection does (same seed), refit volume and
   shape on the **training** days only, with the chosen grouping and smoothing.
2. Generate the **held-out** days with ``bss_sim.PoissonDemand`` from that model.
3. Compare with the requests recorded on the held-out days (``bss_sim.ReplayDemand``).
4. Baseline ("empirical"): the recorded requests of the training days of the same group,
   compared with the same held-out days. The model should be at least as close as simply
   reusing past data.

Writes ``demand-validation.md`` and ``demand-validation.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from bss_scenario.demand.build import DemandConfig, split_days
from bss_scenario.demand.calendar import day_group, group_labels
from bss_scenario.demand.places import Places, assign_places
from bss_scenario.demand.shape import ShapeConfig, fit_shape
from bss_scenario.demand.volume import daily_counts, fit_volume
from bss_scenario.network.graph import Projection
from bss_scenario.trips.clean import read_trips
from bss_sim.checks import compare
from bss_sim.demand import DemandModel, PoissonDemand, ReplayDemand, TripRequests

METRICS = ("daily_mean_ratio", "hourly_tvd", "od_jsd", "origin_tvd", "destination_tvd", "ride_ks")


def train_only_view(bundle: Path, shape_model) -> Path:
    """A temporary bundle view sharing every file except a shape fitted on training days."""
    tmp = Path(tempfile.mkdtemp(prefix="bss-validate-"))
    (tmp / "network").mkdir()
    (tmp / "network" / "bike_dist.npy").symlink_to((bundle / "network" / "bike_dist.npy").resolve())
    shutil.copytree(bundle / "demand", tmp / "demand", symlinks=True,
                    ignore=shutil.ignore_patterns("shape.npz", "replay"))
    (tmp / "demand" / "replay").symlink_to((bundle / "demand" / "replay").resolve())
    shape_model.save(tmp / "demand" / "shape.npz")
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/sim-v2/reports"))
    ap.add_argument("--seeds", type=int, default=5, help="Poisson replications per held-out day")
    args = ap.parse_args()
    bundle = args.bundle

    report = json.loads((bundle / "demand" / "demand_report.json").read_text())
    chosen = report["chosen"]
    fp = tuple(dt.date.fromisoformat(d) for d in report["config"]["fit_period"])
    cfg = DemandConfig(fit_period=fp, seed=report["config"]["seed"])
    scheme = chosen["scheme"]

    import yaml
    scenario = yaml.safe_load((bundle / "scenario.yaml").read_text())
    stations = pl.read_parquet(bundle / "stations" / "stations.parquet")
    places = Places.from_stations(stations)
    trips = read_trips(sorted((bundle / "trips").glob("*.parquet")), valid_only=True)
    daily = daily_counts(trips, dt.date(fp[0].year, 1, 1), dt.date(fp[1].year, 12, 31))
    days = pl.date_range(fp[0], fp[1], "1d", eager=True).to_list()
    train, test = split_days(days, cfg.holdout_fraction, cfg.seed)
    fit_trips = assign_places(trips.filter(trips["start_time"].dt.date().is_in(train)), stations,
                              Projection(**scenario["projection"]))
    volume = fit_volume(daily, scheme, fp)
    shape = fit_shape(fit_trips, places, train, volume, ShapeConfig(scheme, chosen["alpha"], chosen["beta"]))

    view = train_only_view(bundle, shape)
    model = DemandModel.load(view)
    replay = ReplayDemand(model, bundle, seed=0)
    labels = group_labels(scheme)
    rows = []
    for g, label in enumerate(labels):
        test_g = [d for d in test if day_group(d, scheme) == g]
        train_g = [d for d in train if day_group(d, scheme) == g]
        if not test_g:
            continue
        recorded = TripRequests.concat([replay.day(d) for d in test_g])
        generated = TripRequests.concat([PoissonDemand(model, seed=s).day(d) for s in range(args.seeds) for d in test_g])
        past = TripRequests.concat([replay.day(d) for d in train_g])
        m = compare(generated, recorded, model.n_places, len(test_g) * args.seeds, len(test_g))
        e = compare(past, recorded, model.n_places, len(train_g), len(test_g))
        rows.append({"group": label, "heldout_days": len(test_g), "train_days": len(train_g),
                     **{f"model_{k}": m[k] for k in (*METRICS, "requests_per_day", "reference_per_day")},
                     **{f"empirical_{k}": e[k] for k in METRICS}})
    shutil.rmtree(view)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"bundle": scenario["id"], "scheme": scheme, "alpha": chosen["alpha"], "beta": chosen["beta"],
               "heldout_days": len(test), "train_days": len(train), "seeds": args.seeds, "groups": rows}
    (args.out / "demand-validation.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Demand generator validation",
        "",
        f"Bundle `{scenario['id']}`, model `{scheme}` (alpha {chosen['alpha']:g}, beta {chosen['beta']:g}) refitted on "
        f"{len(train)} training days; compared on {len(test)} held-out days of {fp[0]} – {fp[1]} "
        f"(seeded split, as in the model selection). Generated: `PoissonDemand`, {args.seeds} seeds per day. "
        "Reference: requests recorded on the held-out days. **Empirical** baseline: the recorded requests "
        "of the training days of the same group (reusing past data as-is).",
        "",
        "Lower is better for every metric except the daily ratio (1 = same volume). TVD = total variation "
        "distance; JSD in bits; KS = Kolmogorov–Smirnov statistic of in-city ride times.",
        "",
        "| group | held-out days | requests/day gen. / rec. | hourly TVD model (emp.) | OD JSD model (emp.) "
        "| origin TVD model (emp.) | dest. TVD model (emp.) | ride KS model (emp.) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['heldout_days']} | {r['model_requests_per_day']:,.0f} / {r['model_reference_per_day']:,.0f} "
            f"| {r['model_hourly_tvd']:.3f} ({r['empirical_hourly_tvd']:.3f}) | {r['model_od_jsd']:.3f} ({r['empirical_od_jsd']:.3f}) "
            f"| {r['model_origin_tvd']:.3f} ({r['empirical_origin_tvd']:.3f}) | {r['model_destination_tvd']:.3f} "
            f"({r['empirical_destination_tvd']:.3f}) | {r['model_ride_ks']:.3f} ({r['empirical_ride_ks']:.3f}) |")
    lines += ["", "Generated by `scripts/validate_demand.py`.", ""]
    (args.out / "demand-validation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
