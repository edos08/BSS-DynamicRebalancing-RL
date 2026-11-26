import datetime as dt

import numpy as np
import polars as pl
import pytest

from bss_scenario.demand import (EXTERNAL, Places, assign_places, assignment_report, daily_counts, day_group,
                                 fit_volume, holidays)
from bss_scenario.network import Projection


def test_holidays_2024():
    h = holidays(2024)
    assert h[dt.date(2024, 11, 28)] == "Thanksgiving" and dt.date(2024, 11, 29) in h
    assert h[dt.date(2024, 4, 15)] == "Patriots' Day (MA)"
    assert h[dt.date(2024, 5, 27)] == "Memorial Day" and h[dt.date(2024, 9, 2)] == "Labor Day"
    assert h[dt.date(2024, 10, 14)] == "Indigenous Peoples' Day"
    assert dt.date(2024, 7, 4) in h and dt.date(2024, 6, 19) in h
    assert holidays(2021)[dt.date(2021, 7, 5)] == "Independence Day"  # Sunday -> observed Monday
    assert len(h) == 13


def test_day_groups():
    tue, fri, sun = dt.date(2024, 7, 9), dt.date(2024, 7, 12), dt.date(2024, 7, 14)
    assert [day_group(d, "per_weekday") for d in (tue, fri, sun)] == [1, 4, 6]
    assert [day_group(d, "day_types") for d in (tue, fri, sun)] == [0, 1, 3]
    assert [day_group(d, "workday_weekend") for d in (tue, fri, sun)] == [0, 0, 1]
    thanksgiving = dt.date(2024, 11, 28)
    assert day_group(thanksgiving, "day_types") == 3
    assert day_group(thanksgiving, "day_types", holidays_as_sunday=False) == 0
    with pytest.raises(ValueError, match="unknown day_groups"):
        day_group(tue, "hourly")


def synthetic_daily(seed=0, sigma=0.1):
    rng = np.random.default_rng(seed)
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2024, 12, 31), "1d", eager=True)
    month_level = {m: 1000 * (1 + m / 6) for m in range(1, 13)}
    counts = []
    for d in days.to_list():
        weekend = d.weekday() >= 5 or d in holidays(2024)
        counts.append(round(month_level[d.month] * (0.8 if weekend else 1.0) * np.exp(rng.normal(0, sigma))))
    return pl.DataFrame({"date": days, "count": counts})


def test_fit_volume_recovers_structure():
    vm = fit_volume(synthetic_daily(), "workday_weekend", (dt.date(2024, 4, 1), dt.date(2024, 11, 30)))
    assert vm.group_level["weekend"] == pytest.approx(0.8, rel=0.03)
    ratio = vm.month_factor["2024-12"] / vm.month_factor["2024-06"]
    assert ratio == pytest.approx((1 + 12 / 6) / (1 + 6 / 6), rel=0.05)
    fit_days = vm.days.filter("in_fit")
    assert fit_days["month_factor"].mean() == pytest.approx(1.0)
    assert vm.sigma == pytest.approx(0.1, rel=0.15)
    assert vm.factor(dt.date(2024, 6, 3)) == vm.month_factor["2024-06"]
    assert vm.days.filter(pl.col("date") == dt.date(2024, 11, 28))["holiday"].item()


def test_daily_counts_fills_missing_days():
    trips = pl.DataFrame({"start_time": [dt.datetime(2024, 7, 1, 8), dt.datetime(2024, 7, 1, 9),
                                         dt.datetime(2024, 7, 3, 9)]})
    d = daily_counts(trips, dt.date(2024, 7, 1), dt.date(2024, 7, 3))
    assert d["count"].to_list() == [2, 0, 1]
    vm = fit_volume(d, "day_types", (dt.date(2024, 7, 1), dt.date(2024, 7, 3)))  # empty days are fine
    assert vm.month_factor["2024-07"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no days in the fit period"):
        fit_volume(d, "day_types", (dt.date(2025, 1, 1), dt.date(2025, 1, 2)))


def test_assign_places():
    proj = Projection(42.37, -71.11)
    stations = pl.DataFrame({"station_id": ["B", "A", "Z"], "lat": [42.380, 42.370, 42.35],
                             "lon": [-71.10, -71.12, -71.05], "in_city": [True, True, False]})
    trips = pl.DataFrame({
        "start_lat": [42.3701, 42.3799, 42.35], "start_lon": [-71.1201, -71.1001, -71.05],
        "end_lat": [42.3799, 42.3700, 42.3702], "end_lon": [-71.1001, -71.1200, -71.1199],
        "start_in_city": [True, True, False], "end_in_city": [True, True, True],
        "start_station_id": ["A", None, "Z"], "end_station_id": ["B", "B", "A"],
    })
    out = assign_places(trips, stations, proj)
    places = Places.from_stations(stations)
    assert places.station_ids == ("A", "B") and places.external == 2
    assert places.table()["station_id"].to_list() == ["A", "B", EXTERNAL]
    assert out["o_place"].to_list() == [0, 1, 2] and out["d_place"].to_list() == [1, 0, 0]
    rep = assignment_report(out, places)
    assert rep["start"] == {"in_city": 2, "no_station_id": 1, "differs_from_recorded": 0}
    assert rep["end"]["differs_from_recorded"] == 1  # recorded B but nearest is A
    assert rep["external_share"] == pytest.approx(1 / 3)
