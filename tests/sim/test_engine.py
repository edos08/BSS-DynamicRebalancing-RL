import math

import pytest

from bss_sim.engine import (TRIP_END, TRIP_REQUEST, TRIP_START, WAKEUP, At, EventQueue, OnNotification,
                            TimeLimit)


def test_order_is_time_then_priority_then_insertion():
    q = EventQueue()
    q.push(5.0, WAKEUP, 1)
    q.push(5.0, TRIP_REQUEST, 2)
    q.push(5.0, TRIP_END, 3)
    q.push(1.0, WAKEUP, 4)
    q.push(5.0, TRIP_REQUEST, 5)
    q.push(5.0, TRIP_START, 6)
    out = [q.pop()[4] for _ in range(len(q))]
    assert out == [4, 3, 6, 2, 5, 1]
    assert q.peek_time() == math.inf and q.peek() is None


def test_same_pushes_give_same_order():
    def run():
        q = EventQueue()
        for i in range(200):
            q.push(float(i % 7), [TRIP_END, TRIP_START, TRIP_REQUEST][i % 3], i)
        return [q.pop() for _ in range(len(q))]
    assert run() == run()


def test_conditions():
    u = TimeLimit(60) | OnNotification("wakeup")
    assert u.deadline(100.0) == 160.0 and u.kinds() == {"wakeup"}
    v = u | At(130.0) | OnNotification("truck_idle")
    assert v.deadline(100.0) == 130.0 and v.kinds() == {"wakeup", "truck_idle"}
    assert OnNotification("wakeup").deadline(5.0) == math.inf
    with pytest.raises(ValueError):
        TimeLimit(-1)
