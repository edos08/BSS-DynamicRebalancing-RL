import numpy as np
import pytest

from bss_sim.state import (ELECTRIC, EXTERNAL, MECHANICAL, PARKED, RESERVED, RIDING, Fleet, initial_nodes)


def make(n=30, n_nodes=10, seed=0):
    rng = np.random.default_rng(seed)
    kind = (rng.random(n) < 0.4).astype(np.uint8)
    return Fleet(n_nodes, kind, rng.uniform(0, 60, n), 60.0)


def test_new_fleet_is_all_external():
    f = make()
    assert f.counts()["external"] == 30 and f.n_bikes.sum() == 0
    assert (f.battery_km[f.kind == MECHANICAL] == 0).all()
    f.check()


def test_random_operations_match_a_reference_model():
    f = make(n=40, n_nodes=6, seed=1)
    rng = np.random.default_rng(2)
    where = {}  # bike -> node for parked bikes (reference)
    for b in range(25):  # take 25 bikes out of the reservoir and park them
        bike = f.take_external(int(rng.integers(0, 2)))
        node = int(rng.integers(0, 6))
        f.park(bike, node)
        where[bike] = node
    for _ in range(2000):
        op = rng.integers(0, 3)
        if op == 0 and where:  # unpark a random parked bike
            bike = int(rng.choice(sorted(where)))
            assert f.unpark(bike, RESERVED) == where.pop(bike)
            f.set_status(bike, RIDING)
            f.park(bike, int(rng.integers(0, 6)))
            where[bike] = int(f.node[bike])
        elif op == 1 and where:  # a bike leaves the city, another comes in
            bike = int(rng.choice(sorted(where)))
            f.unpark(bike, RIDING)
            del where[bike]
            f.to_external(bike)
            new = f.take_external(int(rng.integers(0, 2)))
            assert new >= 0
            f.park(new, 3)
            where[new] = 3
        else:
            node = int(rng.integers(0, 6))
            assert sorted(f.bikes_at(node)) == sorted(b for b, v in where.items() if v == node)
    f.check()
    assert f.status_count[PARKED] == len(where) == 25
    assert sum(f.status_count) == 40


def test_reservoir_prefers_the_requested_kind():
    f = Fleet(3, np.array([MECHANICAL, ELECTRIC, MECHANICAL], np.uint8), np.full(3, 30.0), 60.0)
    assert f.kind[f.take_external(ELECTRIC)] == ELECTRIC
    assert f.kind[f.take_external(ELECTRIC)] == MECHANICAL  # none left of that kind: the other one
    assert f.kind[f.take_external(MECHANICAL)] == MECHANICAL
    assert f.take_external(MECHANICAL) == -1
    assert f.counts()["riding"] == 3
    f.check()


def test_check_detects_corruption():
    f = make()
    b = f.take_external(0)
    f.park(b, 2)
    f.n_bikes[2] += 1
    with pytest.raises(AssertionError):
        f.check()
    f.n_bikes[2] -= 1
    f.status[b] = EXTERNAL
    with pytest.raises(AssertionError):
        f.check()


def test_initial_nodes_proportional_and_capacity():
    rng = np.random.default_rng(0)
    w = np.array([0.0, 1.0, 3.0, 0.0, 6.0])
    nodes = initial_nodes(rng, 20000, w, None)
    share = np.bincount(nodes, minlength=5) / 20000
    np.testing.assert_allclose(share, w / w.sum(), atol=0.01)
    cap = np.array([5, 2, 3, 0, 4])
    nodes = initial_nodes(np.random.default_rng(1), 9, w, cap)
    counts = np.bincount(nodes, minlength=5)
    assert (counts <= cap).all() and counts.sum() == 9 and counts[0] == 0 and counts[3] == 0
    with pytest.raises(ValueError, match="docks"):
        initial_nodes(rng, 10, w, cap)
    with pytest.raises(ValueError):
        initial_nodes(rng, 1, np.zeros(3), None)


def test_numpy_views_share_the_hot_path_buffers():
    f = make(n=5, n_nodes=3)
    b = f.take_external(0)
    f.park(b, 2)
    assert f.n_bikes[2] == 1 and f.node[b] == 2 and f.status[b] == PARKED  # views see buffer writes
    f.battery_km[b] = 12.5                                                   # and the other way round
    assert f._batt[b] == 12.5
    assert f.n_bikes.base is not None and f.status.dtype == np.uint8
