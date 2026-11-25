import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_network_paths import make_net  # noqa: E402

from bss_scenario.demand.spread import Spread, kernel_spread, spread_report, station_spread  # noqa: E402
from bss_scenario.network import walk_neighbours  # noqa: E402

# a line of 7 nodes, 100 m apart: 0-1-2-3-4-5-6 ; stations at nodes 1 and 5
LINE = [(i, i + 1, 100) for i in range(6)] + [(i + 1, i, 100) for i in range(6)]
NET = make_net(LINE, 7)
WALK = walk_neighbours(NET, 800)
STATIONS = np.array([1, 5])


def test_kernel_catchments_and_mass():
    sp = kernel_spread(STATIONS, WALK, radius_m=250, scale_m=100)
    n0, p0 = sp.of(0)
    n1, p1 = sp.of(1)
    assert n0.tolist() == [0, 1, 2, 3] and n1.tolist() == [4, 5, 6]   # node 3: tie -> station 0
    np.testing.assert_allclose([p0.sum(), p1.sum()], 1.0, rtol=1e-6)
    w = np.exp(-np.array([100, 0, 100, 200]) / 100)
    np.testing.assert_allclose(p0, w / w.sum(), rtol=1e-6)
    assert not set(n0) & set(n1)


def test_radius_limits_catchment():
    sp = kernel_spread(STATIONS, WALK, radius_m=100, scale_m=100)
    assert sp.of(0)[0].tolist() == [0, 1, 2] and sp.of(1)[0].tolist() == [4, 5, 6]
    rep = spread_report(sp, 7)
    assert rep["nodes_covered"] == 6 and rep["covered_share"] == pytest.approx(6 / 7)


def test_shared_node_keeps_both_stations():
    sp = kernel_spread(np.array([2, 2]), WALK, radius_m=150, scale_m=100)
    assert 2 in sp.of(0)[0] and sp.of(1)[0].tolist() == [2]
    np.testing.assert_allclose(sp.of(1)[1], [1.0])


def test_station_mode_and_errors(tmp_path):
    sp = station_spread(STATIONS)
    assert sp.of(1)[0].tolist() == [5] and sp.of(1)[1].tolist() == [1.0]
    sp.save(tmp_path / "s.npz")
    back = Spread.load(tmp_path / "s.npz")
    assert back.mode == "station" and np.array_equal(back.nodes, sp.nodes)
    with pytest.raises(ValueError, match="exceeds"):
        kernel_spread(STATIONS, walk_neighbours(NET, 100), radius_m=400)
