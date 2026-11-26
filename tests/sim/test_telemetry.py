import json

import numpy as np
import pytest

from bss_sim import telemetry
from bss_sim.config import SimConfig
from bss_sim.engine import TRIP_REQUEST, TRIP_START, TimeLimit
from bss_sim.simulator import Simulator

START = "2024-09-17T06:00"


@pytest.fixture(scope="module")
def sim(bundle):
    cfg = SimConfig.from_dict({"fleet": {"size": 20}, "external": {"reservoir": 30},
                               "telemetry": {"event_log": True, "snapshot_every_s": 900}})
    s = Simulator.from_scenario(bundle, cfg)
    s.reset(seed=1, start=START)
    s.run(TimeLimit(86400))
    return s


def test_log_values(sim):
    log = sim.event_log()
    assert set(log) == {name for name, _ in telemetry.COLUMNS}
    req = log["event"] == TRIP_REQUEST
    served = req & (log["outcome"] == 0) & (log["value"] > 0)
    assert (log["value"][served] <= sim.config.users.walk_radius_m).all()
    starts = log["event"] == TRIP_START
    assert (log["value"][starts] >= 0).all()
    k = sim.kpis()
    walks = log["value"][req & (log["outcome"] == 0)]
    city_served = k["served"] - k["from_external"]
    assert k["walk_m_mean"] == pytest.approx(walks.sum() / city_served, rel=1e-5)
    assert k["walked_share"] == pytest.approx((walks > 0).sum() / city_served)


@pytest.mark.parametrize("suffix", [".npz", ".parquet"])
def test_save_and_load(sim, tmp_path, suffix):
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
    path = sim.save_log(tmp_path / f"run{suffix}")
    arrays, meta = telemetry.load_events(path)
    for name, arr in sim.event_log().items():
        np.testing.assert_array_equal(arrays[name], arr)
    assert meta["seed"] == 1 and meta["config_hash"] == sim.config.config_hash()
    assert meta["config"]["telemetry"]["snapshot_every_s"] == 900
    with np.load(tmp_path / "run.snapshots.npz") as z:
        assert z["bikes"].shape == (len(z["t"]), sim.scenario.n_nodes)
        assert json.loads(str(z["metadata"]))["start"] == "2024-09-17T06:00:00"
    with pytest.raises(ValueError):
        sim.save_log(tmp_path / "run.csv")


def test_snapshots(sim):
    t, bikes = sim.snapshots()
    assert t[0] == 6 * 3600 and np.allclose(np.diff(t), 900)
    assert len(t) == 86400 // 900 + 1  # both ends included
    assert (bikes.sum(axis=1) <= 50).all()
    np.testing.assert_array_equal(bikes[-1], sim.fleet.n_bikes)


def test_logging_off(bundle):
    s = Simulator.from_scenario(bundle, SimConfig.from_dict({"fleet": {"size": 20}}))
    s.reset(seed=1, start=START)
    s.run(TimeLimit(3600))
    with pytest.raises(RuntimeError):
        s.event_log()
    t, bikes = s.snapshots()
    assert len(t) == 0 and bikes.shape == (0, s.scenario.n_nodes)


def test_walk_histogram():
    h = telemetry.WalkHistogram(400, 10)
    for w in [0, 0, 0, 5, 10, 10.5, 399, 400, 1000]:
        h.add(w)
    assert h.counts[0] == 3 and h.counts[1] == 2 and h.counts[2] == 1 and h.counts[-1] == 1
    assert h.quantile(0.3) == 0.0 and h.quantile(0.5) == 10.0 and h.quantile(1.0) == 410.0
    assert telemetry.WalkHistogram(400).quantile(0.5) == 0.0
