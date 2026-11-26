import numpy as np

from bss_policies.planning import Visit, allocate, plan_route, route_time, zone_targets


def test_allocate_proportional_floors_and_caps():
    out = allocate(100, np.array([1.0, 1.0, 2.0]))
    assert out.tolist() == [25, 25, 50]
    out = allocate(10, np.array([1.0, 1.0, 1.0]))
    assert out.sum() == 10 and out.max() - out.min() <= 1
    out = allocate(20, np.array([0.0, 1.0, 9.0]), floor=np.array([3, 3, 3]))
    assert out.sum() == 20 and (out >= 3).all() and out[2] > out[1]
    out = allocate(20, np.array([1.0, 1.0, 1.0]), cap=np.array([2, 100, 100]))
    assert out.sum() == 20 and out[0] <= 2
    assert allocate(50, np.ones(3), cap=np.array([5, 5, 5])).tolist() == [5, 5, 5]   # capped below total
    assert allocate(4, np.array([1.0, 5.0, 3.0]), floor=np.array([2, 2, 2])).tolist() == [0, 2, 2]
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 20))
        w = rng.random(n) * (rng.random(n) > 0.3)
        cap = rng.integers(0, 30, n)
        floor = rng.integers(0, 4, n)
        total = int(rng.integers(0, 300))
        out = allocate(total, w, floor, cap)
        assert out.sum() == min(total, cap.sum()) and (out <= cap).all() and (out >= 0).all()


def test_zone_targets_follow_demand_and_net_outflow():
    D = np.array([10.0, 10.0, 0.0])
    A = np.array([10.0, 0.0, 5.0])           # zone 1 drains: twice the weight of zone 0
    t = zone_targets(D, A, total=30, min_per_zone=2)
    assert t.tolist() == [10, 20, 0]


def line(n, step=100.0):
    """Zones on a line, ``step`` s apart; the truck starts at zone 0's position (last row)."""
    pos = np.arange(n) * step
    T = np.abs(pos[:, None] - pos[None, :])
    return np.vstack([T, T[0]])


def check(route, surplus, capacity, load):
    left = np.asarray(surplus).copy()
    for v in route:
        load += v.qty
        assert 0 <= load <= capacity
        assert np.sign(v.qty) == np.sign(left[v.zone]) and abs(v.qty) <= abs(left[v.zone])
        left[v.zone] -= v.qty


def test_route_moves_bikes_from_surplus_to_deficit_within_capacity():
    surplus = np.array([12, 0, -5, 8, -10, -5])
    route = plan_route(surplus, line(6), capacity=10, load=0)
    check(route, surplus, 10, 0)
    moved = sum(-v.qty for v in route if v.qty < 0)
    assert moved == 20                         # all 20 surplus bikes reach the deficits
    route = plan_route(surplus, line(6), capacity=10, load=0, time_budget_s=500)
    assert route_time(route, -1, line(6), lambda q: 30.0, 60.0) <= 500


def test_route_starts_from_the_load_on_board_and_two_opt_helps():
    surplus = np.array([0, -4, -4, -4])
    route = plan_route(surplus, line(4), capacity=30, load=12)
    check(route, surplus, 30, 12)
    assert [v.zone for v in route] == [1, 2, 3]
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = 12
        xy = rng.random((n + 1, 2)) * 3000
        T = np.hypot(*(xy[:, None, :] - xy[None, :n, :]).transpose(2, 0, 1))
        T = np.vstack([T[:n], T[n]])
        s = rng.integers(-6, 7, n)
        load = int(rng.integers(0, 10))
        a = plan_route(s, T, 15, load, two_opt=False)
        b = plan_route(s, T, 15, load, two_opt=True)
        check(b, s, 15, load)

        def svc(q):
            return 30.0
        assert route_time(b, -1, T, svc, 60.0) <= route_time(a, -1, T, svc, 60.0) + 1e-9
        assert sorted((v.zone, v.qty) for v in a) == sorted((v.zone, v.qty) for v in b)


def test_nothing_to_do():
    assert plan_route(np.zeros(5, dtype=int), line(5), 10, 0) == []
    assert plan_route(np.array([-3, -3]), line(2), 10, 0) == []     # deficits but nothing on board
    assert plan_route(np.array([3, 3]), line(2), 10, 10) == []      # surplus but the truck is full
    assert Visit(1, 2).qty == 2
