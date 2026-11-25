import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from bss_scenario import cli  # noqa: E402
from bss_scenario.demand import Places  # noqa: E402
from bss_scenario.demand.build import DemandConfig, heldout_loglik, select_shape, split_days  # noqa: E402
from bss_scenario.demand.shape import ShapeConfig, fit_shape  # noqa: E402
from bss_scenario.demand.volume import daily_counts  # noqa: E402

TZ = "America/New_York"
PLACES = Places(("A", "B", "C"))


def sample_trips(days, seed=0):
    """Workdays: A->B in the morning, B->A in the evening; weekends: C->C at noon."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in days:
        if day.weekday() < 5:
            plan = [(8, 0, 1, 40), (18, 1, 0, 40), (12, 2, 3, 5)]
        else:
            plan = [(12, 2, 2, 30), (15, 0, 3, 5)]
        for hour, o, d, lam in plan:
            for _ in range(rng.poisson(lam)):
                rows.append((dt.datetime(day.year, day.month, day.day, hour, int(rng.integers(60))), o, d))
    return pl.DataFrame({"start_time": pl.Series([r[0] for r in rows]).dt.replace_time_zone(TZ),
                         "o_place": [r[1] for r in rows], "d_place": [r[2] for r in rows]})


DAYS = pl.date_range(dt.date(2024, 6, 3), dt.date(2024, 7, 28), "1d", eager=True).to_list()


def test_split_days_is_stratified_and_seeded():
    train, test = split_days(DAYS, 0.25, seed=1)
    assert sorted(train + test) == DAYS and not set(train) & set(test)
    assert {d.weekday() for d in test} == set(range(7))
    assert split_days(DAYS, 0.25, seed=1) == (train, test) and split_days(DAYS, 0.25, seed=2) != (train, test)


def test_heldout_loglik_prefers_the_true_pattern():
    trips = sample_trips(DAYS)
    good = fit_shape(trips, PLACES, DAYS, None, ShapeConfig("workday_weekend", 1.0, 1.0))
    ll_good, zero = heldout_loglik(good, sample_trips(DAYS, seed=9))
    shuffled = trips.with_columns(pl.col("d_place").shuffle(seed=3))
    bad = fit_shape(shuffled, PLACES, DAYS, None, ShapeConfig("workday_weekend", 1.0, 1.0))
    assert heldout_loglik(bad, sample_trips(DAYS, seed=9))[0] < ll_good and zero == 0
    assert np.isnan(heldout_loglik(good, trips.head(0))[0])


def test_select_shape_ranks_candidates():
    trips = sample_trips(DAYS)
    daily = daily_counts(trips, dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    cfg = DemandConfig(fit_period=(DAYS[0], DAYS[-1]), schemes=("workday_weekend", "per_weekday"),
                       alphas=(0.0, 1.0), betas=(0.0, 1.0))
    table = select_shape(trips, PLACES, DAYS, daily, cfg)
    assert table.height == 8 and table["delta_vs_best"][0] == 0
    assert table["loglik_per_trip"].is_sorted(descending=True)
    # the generating process only distinguishes workdays from weekends: 7 groups cannot do better
    best_ww = table.filter(pl.col("scheme") == "workday_weekend")["loglik_per_trip"].max()
    assert best_ww >= table["loglik_per_trip"].max() - 0.01
