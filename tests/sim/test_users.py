import math

import numpy as np
import pytest

from bss_sim.state import ELECTRIC, MECHANICAL, Fleet
from bss_sim.users import DECLINED_WALK, LOW_BATTERY_ONLY, NO_BIKE_IN_RANGE, SERVED, WalkingUsers


def line_csr(n=10, step=50.0, rmax=800.0):
    """Nodes on a line ``step`` m apart; neighbours within ``rmax``, sorted by distance, self first."""
    indptr, idx, dist = [0], [], []
    for v in range(n):
        nb = sorted(range(n), key=lambda u: (abs(u - v), u))
        nb = [u for u in nb if abs(u - v) * step <= rmax]
        idx += nb
        dist += [abs(u - v) * step for u in nb]
        indptr.append(len(idx))
    return np.array(indptr), np.array(idx, np.int32), np.array(dist, np.float32)


def setup(kinds, batteries, nodes, **kw):
    f = Fleet(10, np.array(kinds, np.uint8), np.array(batteries, float), 60.0)
    for b, node in enumerate(nodes):  # take every bike out of the reservoir and park it
        f.reservoir[kinds[b]].remove(b)
        f.set_status(b, 2)
        f.park(b, node)
    users = WalkingUsers(*line_csr(), radius_m=kw.pop("radius_m", 400.0), beta_m=kw.pop("beta_m", 150.0), **kw)
    users.bind(f)
    return f, users


def test_nearest_usable_bike_and_reasons():
    f, users = setup([MECHANICAL, ELECTRIC], [0.0, 2.0], [7, 2])
    bike, w, r = users.choose(0, MECHANICAL, 5.0, 0.0)          # e-bike at 100 m lacks charge; classic at 350 m
    assert (bike, w, r) == (0, 350.0, SERVED)
    assert users.choose(0, MECHANICAL, 1.0, 0.0)[:2] == (1, 100.0)  # short trip: the e-bike is enough
    assert users.choose(9, MECHANICAL, 1.0, 0.0)[2] == SERVED       # classic bike 100 m away
    f2, users2 = setup([ELECTRIC], [1.0], [1])
    assert users2.choose(0, ELECTRIC, 5.0, 0.0)[2] == LOW_BATTERY_ONLY
    assert users2.choose(9, ELECTRIC, 0.5, 0.0)[:2] == (0, 400.0)      # bike exactly at the radius: in reach
    assert setup([ELECTRIC], [30.0], [0])[1].choose(9, ELECTRIC, 0.5, 0.0)[2] == NO_BIKE_IN_RANGE  # 450 m
    f3, users3 = setup([ELECTRIC], [1.0], [1], rideable_empty=True)
    assert users3.choose(0, ELECTRIC, 5.0, 0.0)[0] == 0


def test_radius_is_inclusive_and_configurable():
    _, users = setup([MECHANICAL], [0.0], [8])
    assert users.choose(0, MECHANICAL, 0.0, 0.0)[:2] == (0, 400.0)      # 400 m: inclusive
    _, narrow = setup([MECHANICAL], [0.0], [8], radius_m=350.0)
    assert narrow.choose(0, MECHANICAL, 0.0, 0.0)[2] == NO_BIKE_IN_RANGE
    assert narrow.choose(1, MECHANICAL, 0.0, 0.0)[:2] == (0, 350.0)


def test_preferred_kind_at_same_node_and_with_tolerance():
    kinds, batt, nodes = [MECHANICAL, ELECTRIC, ELECTRIC], [0.0, 50.0, 50.0], [3, 3, 5]
    _, users = setup(kinds, batt, nodes)
    assert users.choose(3, ELECTRIC, 1.0, 0.0)[0] == 1            # same node: preferred type taken
    assert users.choose(3, MECHANICAL, 1.0, 0.0)[0] == 0
    _, users = setup([MECHANICAL, ELECTRIC], [0.0, 50.0], [1, 3])
    assert users.choose(0, ELECTRIC, 1.0, 0.0)[:2] == (0, 50.0)   # no tolerance: nearest
    _, tol = setup([MECHANICAL, ELECTRIC], [0.0, 50.0], [1, 3], kind_tolerance_m=100.0)
    assert tol.choose(0, ELECTRIC, 1.0, 0.0)[:2] == (1, 150.0)    # walks 100 m more for an e-bike
    _, tol = setup([MECHANICAL, ELECTRIC], [0.0, 50.0], [1, 5], kind_tolerance_m=100.0)
    assert tol.choose(0, ELECTRIC, 1.0, 0.0)[:2] == (0, 50.0)     # 250 m: beyond the tolerance


def test_acceptance_probability():
    _, users = setup([MECHANICAL], [0.0], [3])  # 150 m from node 0
    p = math.exp(-150.0 / 150.0)
    u = np.random.default_rng(0).random(20000)
    acc = np.mean([users.choose(0, MECHANICAL, 0.0, x)[2] == SERVED for x in u])
    assert acc == pytest.approx(p, abs=0.01)
    assert users.choose(0, MECHANICAL, 0.0, 0.99)[:3] == (-1, 150.0, DECLINED_WALK)
    assert users.choose(3, MECHANICAL, 0.0, 0.9999)[2] == SERVED  # no walk: always accepted
