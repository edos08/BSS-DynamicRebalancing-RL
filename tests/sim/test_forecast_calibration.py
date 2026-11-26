"""Calibration of the forecasts on simulated data (issue #130, spec §8.1).

A forecast that is off gives the agent wrong information, so every in-flight level that claims to
be realistic must be unbiased and its sampled intervals must cover as often as they say.
Reference report on Cambridge: docs/sim-v2/reports/forecast-calibration.md.
"""

import pytest

from bss_sim.forecast_check import collect, summarise
from bss_sim.synthetic import make_bundle


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    bundle = make_bundle(tmp_path_factory.mktemp("cal"), n_side=24, n_stations=10, weekday_trips=3000, seed=7)
    acc: dict = {}
    for seed in range(3):
        collect(bundle, "poisson", days=2, start="2024-09-17T00:00", window=1800, every=1800, samples=40,
                seed=seed, acc=acc, sim_config={"fleet": {"size": 250}})
    return summarise(acc, {})


def test_reference_is_exact(result):
    d = result["inflight"]["destination"]
    assert d["bias"] == 0 and d["mae_groups"] == 0


@pytest.mark.parametrize("level", ["origin", "count"])
def test_inflight_arrivals_are_unbiased(result, level):
    m = result["inflight"][level]
    assert abs(m["bias"]) <= 3 * m["bias_se"] + 0.02 * m["mean_real"], m


def test_sampled_intervals_cover_as_stated(result):
    c = result["coverage"]
    assert c["cells"] > 300
    assert 0.40 <= c["50"] <= 0.60, c
    assert 0.82 <= c["90"] <= 0.96, c


def test_origin_is_sharper_than_not_knowing(result):
    inf = result["inflight"]
    assert inf["destination"]["mae_groups"] < inf["origin"]["mae_groups"] <= inf["none"]["mae_groups"] * 1.05
