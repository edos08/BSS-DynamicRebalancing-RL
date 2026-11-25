"""Select the demand-shape settings on held-out days and write ``demand/`` into a bundle (issue #80).

**Selection.** 20 % of the fit-period days (stratified by weekday, seeded) are held out. For each
candidate (day grouping × alpha × beta) the shape is fitted on the other days and scored by the
mean **conditional log-likelihood per held-out trip**: given that a held-out day of group ``g`` had
``n`` trips, each trip's (hour, origin, destination) is a draw from ``rates[g] / rates[g].sum()``.
This scores the shape alone (the day total is the volume model's job), so day groupings with
different numbers of groups are compared fairly. The best candidate is refitted on all fit-period
days.

**Outputs** (``<bundle>/demand/``):

=======================  ==================================================================
``places.parquet``       place index -> station id (``EXTERNAL`` last)
``volume.parquet``       every calendar day: group, holiday, count, month factor, day factor
``volume.json``          month factors, day-noise sigma, group levels, scheme
``shape.npz``            rates [G, 24, P, P] (trips/hour at volume factor 1), exposure, config
``ride_time.npz``        speed quantiles per (bike type, distance band), round-trip durations
``demand_report.json``   selection table, chosen settings, place-assignment report
=======================  ==================================================================
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from bss_scenario.demand.calendar import DAY_GROUPS, day_group
from bss_scenario.demand.places import Places, assign_places, assignment_report
from bss_scenario.demand.ride_time import fit_ride_time
from bss_scenario.demand.ride_time import summary as ride_summary
from bss_scenario.demand.shape import ShapeConfig, ShapeModel, fit_shape
from bss_scenario.demand.volume import VolumeModel, daily_counts, fit_volume
from bss_scenario.network.graph import Projection
from bss_scenario.trips.clean import read_trips


@dataclass(frozen=True)
class DemandConfig:
    fit_period: tuple[dt.date, dt.date] = (dt.date(2024, 4, 1), dt.date(2024, 11, 30))
    schemes: tuple[str, ...] = tuple(DAY_GROUPS)
    alphas: tuple[float, ...] = (0.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0)
    betas: tuple[float, ...] = (0.0, 8.0, 32.0, 128.0, 512.0, 2048.0)
    holdout_fraction: float = 0.2
    seed: int = 2024
    holidays_as_sunday: bool = True


def split_days(days: list[dt.date], fraction: float, seed: int) -> tuple[list[dt.date], list[dt.date]]:
    """Hold out ``fraction`` of the days of each weekday (seeded)."""
    rng = np.random.default_rng(seed)
    test: list[dt.date] = []
    for wd in range(7):
        same = [d for d in days if d.weekday() == wd]
        k = int(round(fraction * len(same)))
        test.extend(same[i] for i in sorted(rng.choice(len(same), size=k, replace=False)))
    test_set = set(test)
    return [d for d in days if d not in test_set], sorted(test)


def heldout_loglik(model: ShapeModel, trips: pl.DataFrame, floor: float = 1e-12) -> tuple[float, int]:
    """Mean conditional log-likelihood per trip, and the number of trips with zero predicted rate."""
    cfg = model.config
    dates = trips["start_time"].dt.date()
    group_of = {d: day_group(d, cfg.scheme, cfg.holidays_as_sunday) for d in dates.unique().to_list()}
    g = dates.replace_strict(group_of, return_dtype=pl.Int64).to_numpy()
    h = trips["start_time"].dt.hour().to_numpy()
    o = trips["o_place"].to_numpy()
    d = trips["d_place"].to_numpy()
    if trips.height == 0:
        return float("nan"), 0
    rates = model.rates.astype(np.float64)
    totals = rates.sum(axis=(1, 2, 3))
    p = rates[g, h, o, d] / totals[g]
    zero = int((p <= 0).sum())
    return float(np.log(np.maximum(p, floor)).mean()), zero


def select_shape(trips: pl.DataFrame, places: Places, days: list[dt.date], daily: pl.DataFrame,
                 config: DemandConfig) -> pl.DataFrame:
    train, test = split_days(days, config.holdout_fraction, config.seed)
    test_set = set(test)
    dates = trips["start_time"].dt.date()
    is_test = dates.is_in(list(test_set))
    train_trips, test_trips = trips.filter(~is_test), trips.filter(is_test)
    rows = []
    for scheme in config.schemes:
        volume = fit_volume(daily, scheme, config.fit_period, config.holidays_as_sunday)
        for alpha, beta in itertools.product(config.alphas, config.betas):
            shape_cfg = ShapeConfig(scheme, alpha, beta, config.holidays_as_sunday)
            model = fit_shape(train_trips, places, train, volume, shape_cfg)
            ll, zero = heldout_loglik(model, test_trips)
            rows.append({"scheme": scheme, "alpha": alpha, "beta": beta, "loglik_per_trip": ll,
                         "zero_prob_trips": zero})
    table = pl.DataFrame(rows).with_columns(pl.col("loglik_per_trip").fill_nan(None))
    return table.sort("loglik_per_trip", descending=True, nulls_last=True).with_columns(
        (pl.col("loglik_per_trip") - pl.col("loglik_per_trip").max()).alias("delta_vs_best"))


def build_demand(bundle_root: str | Path, config: DemandConfig = DemandConfig()) -> dict:
    """Fit volume and shape from the bundle's trips and stations and write ``demand/``."""
    root = Path(bundle_root)
    out = root / "demand"
    out.mkdir(exist_ok=True)
    scenario = yaml.safe_load((root / "scenario.yaml").read_text())
    stations = pl.read_parquet(root / "stations" / "stations.parquet")
    places = Places.from_stations(stations)
    trips = read_trips(sorted((root / "trips").glob("*.parquet")), valid_only=True)
    year_start = trips["start_time"].dt.date().min().replace(month=1, day=1)
    year_end = trips["start_time"].dt.date().max().replace(month=12, day=31)
    daily = daily_counts(trips, year_start, year_end)

    fp = config.fit_period
    fit_trips = assign_places(trips.filter(trips["start_time"].dt.date().is_between(*fp)), stations,
                              Projection(**scenario["projection"]))
    days = pl.date_range(fp[0], fp[1], "1d", eager=True).to_list()

    table = select_shape(fit_trips, places, days, daily, config)
    best = table.row(0, named=True)
    if best["loglik_per_trip"] is None:  # no held-out trips (tiny bundles): keep the defaults
        default = ShapeConfig()
        best = {"scheme": default.scheme, "alpha": default.alpha, "beta": default.beta, "loglik_per_trip": None}
    volume: VolumeModel = fit_volume(daily, best["scheme"], fp, config.holidays_as_sunday)
    model = fit_shape(fit_trips, places, days, volume,
                      ShapeConfig(best["scheme"], best["alpha"], best["beta"], config.holidays_as_sunday))

    places.table().write_parquet(out / "places.parquet")
    volume.days.write_parquet(out / "volume.parquet")
    (out / "volume.json").write_text(json.dumps({
        "scheme": volume.scheme, "month_factor": volume.month_factor, "group_level": volume.group_level,
        "sigma": volume.sigma, "sigma_all_days": volume.sigma_all_days,
        "fit_period": [d.isoformat() for d in fp]}, indent=2) + "\n")
    model.save(out / "shape.npz")

    in_city = stations.filter("in_city").sort("station_id")
    place_node = in_city["bike_node"].to_numpy()
    bike_dist = np.load(root / "network" / "bike_dist.npy", mmap_mode="r")
    place_dist = np.asarray(bike_dist[np.ix_(place_node, place_node)])  # [P-1, P-1] station-node distances
    ride = fit_ride_time(fit_trips, np.arange(len(place_node)), place_dist, places.external)
    ride.save(out / "ride_time.npz")
    report = {
        "config": {**asdict(config), "fit_period": [d.isoformat() for d in fp]},
        "chosen": {k: best[k] for k in ("scheme", "alpha", "beta", "loglik_per_trip")},
        "selection": table.to_dicts(),
        "places": places.n,
        "fit_trips": fit_trips.height,
        "expected_trips_per_day": dict(zip(model.groups, map(float, model.rates.sum(axis=(1, 2, 3))))),
        "assignment": assignment_report(fit_trips, places),
        "ride_time": ride_summary(ride),
    }
    (out / "demand_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report
