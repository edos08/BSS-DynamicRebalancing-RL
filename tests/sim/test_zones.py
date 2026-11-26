import numpy as np
import pytest

from bss_sim.config import SimConfig
from bss_sim.scenario import Scenario
from bss_sim.simulator import Simulator
from bss_sim.synthetic import make_bundle
from bss_sim.zones import grid_zones, station_zones


@pytest.fixture(scope="module")
def sc(tmp_path_factory):
    return Scenario.load(make_bundle(tmp_path_factory.mktemp("z"), n_side=20, n_stations=8, weekday_trips=500, seed=5))


def test_grid_zones_cover_every_node(sc):
    z = grid_zones(sc, cell_m=300)                      # 20 x 60 m = 1,140 m: 4 x 4 cells
    assert z.n == 16 and (z.node_to_zone >= 0).all() and z.name == "grid300"
    for k in range(z.n):
        nodes = z.nodes(k)
        assert z.node_to_zone[z.stop[k]] == k            # every stop lies in its own zone
        assert np.ptp(sc.bike_x[nodes]) < 300 and np.ptp(sc.bike_y[nodes]) < 300
    w = sc.departure_weights()
    for k in range(z.n):                                 # demand stop: the busiest node of the zone
        nodes = z.nodes(k)
        if w[nodes].max() > 0:
            assert w[z.stop[k]] == w[nodes].max()
    c = grid_zones(sc, cell_m=300, stop="center")
    k = 5
    d = np.hypot(sc.bike_x[c.nodes(k)] - c.center[k, 0], sc.bike_y[c.nodes(k)] - c.center[k, 1])
    assert c.stop[k] == c.nodes(k)[np.argmin(d)]
    with pytest.raises(ValueError):
        grid_zones(sc, cell_m=0)


def test_station_zones(sc):
    z = station_zones(sc)
    assert z.n == 8 and (z.stop == sc.station_node).all()
    assert (z.node_to_zone[sc.station_node] == np.arange(8)).all()
    covered = z.node_to_zone >= 0
    sp = sc.demand.spreader
    assert covered.sum() == len(np.unique(sp.nodes))      # exactly the nodes some station spreads to


def test_zones_as_partitions(sc, tmp_path):
    sim = Simulator(sc, SimConfig.from_dict({"fleet": {"size": 50}}))
    z = grid_zones(sc)
    sim.add_partition(z.name, z.node_to_zone)
    sim.reset(seed=1, start="2024-09-17T08:00")
    assert sim.partition(z.name).bikes.sum() == 50
