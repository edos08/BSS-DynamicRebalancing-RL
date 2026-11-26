import datetime as dt
import json

import numpy as np
import pytest

from bss_sim.config import ConfigError, SimConfig, TravelConfig
from bss_sim.scenario import Scenario
from bss_sim.synthetic import make_bundle
from bss_sim.travel import CongestionFactors, FreeFlow, make_travel, traffic_day_type


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("syn"), n_side=20, n_stations=8, weekday_trips=500, seed=1)


def test_traffic_day_types():
    assert traffic_day_type(dt.date(2024, 9, 17)) == 0   # Tuesday
    assert traffic_day_type(dt.date(2024, 9, 21)) == 1   # Saturday
    assert traffic_day_type(dt.date(2024, 9, 22)) == 2   # Sunday
    assert traffic_day_type(dt.date(2024, 9, 2)) == 2    # Labor Day (Monday) counts as Sunday


def test_synthetic_drive_network_is_consistent(syn):
    sc = Scenario.load(syn)
    t, by = np.asarray(sc.drive_time), sc.drive_time_by_class()
    np.testing.assert_allclose(by.sum(axis=0), t, rtol=1e-5)
    assert sc.n_drive == 100 and sc.bike_to_drive.max() < 100
    assert sc.road_classes == ("major", "secondary", "local")
    assert (np.asarray(sc.drive_dist) >= t * 30 / 3.6 - 1e-3).all()


def test_freeflow_and_congestion(syn):
    sc = Scenario.load(syn)
    t, d, by = np.asarray(sc.drive_time), np.asarray(sc.drive_dist), np.asarray(sc.drive_time_by_class())
    ff = FreeFlow(t, d, scale=1.5)
    ff.bind(np.random.default_rng(0), dt.date(2024, 9, 16))
    assert ff.time(3, 77, 0.0) == pytest.approx(1.5 * t[3, 77]) and ff.distance(3, 77) == d[3, 77]
    ones = np.ones((3, 24, 3))
    cf = CongestionFactors(t, d, by, ones)
    cf.bind(np.random.default_rng(0), dt.date(2024, 9, 16))   # a Monday
    for a, b in ((0, 99), (12, 45), (60, 6)):
        assert cf.time(a, b, 8 * 3600.0) == pytest.approx(float(t[a, b]), rel=1e-5)
    f = ones.copy()
    f[0, 8, 0] = 2.0                                           # weekday 08:00, major roads twice as slow
    cf = CongestionFactors(t, d, by, f)
    cf.bind(np.random.default_rng(0), dt.date(2024, 9, 16))
    a, b = 0, 99
    assert cf.time(a, b, 8.5 * 3600) == pytest.approx(float(t[a, b] + by[0, a, b]), rel=1e-5)
    assert cf.time(a, b, 9.5 * 3600) == pytest.approx(float(t[a, b]), rel=1e-5)          # 09:00: factor 1
    assert cf.time(a, b, 5 * 86400 + 8.5 * 3600) == pytest.approx(float(t[a, b]), rel=1e-5)  # Saturday
    with pytest.raises(ValueError):
        CongestionFactors(t, d, by, np.ones((3, 24, 2)))
    with pytest.raises(ValueError):
        CongestionFactors(t, d, by, np.zeros((3, 24, 3)))


def test_noise_is_mean_one_and_seeded(syn):
    sc = Scenario.load(syn)
    t, d = np.asarray(sc.drive_time), np.asarray(sc.drive_dist)

    def draws(seed):
        m = FreeFlow(t, d, noise_sigma=0.3)
        m.bind(np.random.default_rng(seed), dt.date(2024, 9, 16))
        return np.array([m.time(0, 99, 0.0) for _ in range(20000)]) / t[0, 99]
    a = draws(1)
    assert a.mean() == pytest.approx(1.0, abs=0.01) and a.std() > 0.25
    np.testing.assert_array_equal(a, draws(1))


def test_make_travel_and_config(syn, tmp_path):
    sc = Scenario.load(syn)
    assert isinstance(make_travel(sc, TravelConfig()), FreeFlow)
    assert not isinstance(make_travel(sc, TravelConfig()), CongestionFactors)   # no fitted factors yet
    with pytest.raises(ValueError, match="factors"):
        make_travel(sc, TravelConfig(model="congestion"))
    f = np.full((3, 24, 3), 1.2).tolist()
    cfg = SimConfig.from_dict({"travel": {"model": "congestion", "factors": f}}).travel
    assert isinstance(make_travel(sc, cfg), CongestionFactors)
    (syn / "traffic").mkdir(exist_ok=True)
    (syn / "traffic" / "factors.json").write_text(json.dumps(
        {"day_types": ["weekday", "saturday", "sunday"], "road_classes": ["major", "secondary", "local"], "factors": f}))
    m = make_travel(Scenario.load(syn), TravelConfig())
    assert isinstance(m, CongestionFactors) and m.factors[0, 0, 0] == 1.2
    assert not isinstance(make_travel(Scenario.load(syn), TravelConfig(model="freeflow")), CongestionFactors)
    (syn / "traffic" / "factors.json").unlink()
    for bad in ({"model": "teleport"}, {"scale": 0}, {"noise_sigma": -1}):
        with pytest.raises(ConfigError):
            SimConfig.from_dict({"travel": bad})
