"""Event queue, run-until conditions and notifications (issue #96, spec §5.1, §5.2).

The clock is in **seconds since local midnight of the run's start date** (float64). Days are
86,400 s: the local wall clock of the demand model is used as is, so DST days are not 23 or 25 h.

Events are ordered by ``(time, priority, sequence)``. The priority puts, at equal times, slot
starts first, then bikes arriving (so they can be taken), then departures, then requests, then
controller timers. The sequence number is a per-queue counter, so ties are broken by insertion
order and the order is deterministic.

Controllers never see the queue: ``Simulator.run(until=...)`` returns :class:`Notification` s.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

# event kinds (also the priority at equal times; lower first)
SLOT_START = 0
TRIP_END = 1
TRIP_START = 2
TRIP_REQUEST = 3
SNAPSHOT = 4
REDIRECT = 5  # log only: a docked trip continues to another station (the next TripEnd is queued)
TRUCK = 6     # a truck finishes its current drive, service or wait
WAKEUP = 9

EVENT_NAMES = {SLOT_START: "SlotStart", TRIP_END: "TripEnd", TRIP_START: "TripStart",
               TRIP_REQUEST: "TripRequest", SNAPSHOT: "Snapshot", REDIRECT: "Redirect", TRUCK: "Truck", WAKEUP: "Wakeup"}


class EventQueue:
    """Binary heap of ``(time, priority, sequence, kind, a, b)`` with integer payloads ``a``, ``b``."""

    __slots__ = ("_heap", "_seq")

    def __init__(self):
        self._heap: list[tuple] = []
        self._seq = 0

    def push(self, t: float, kind: int, a: int = 0, b: int = 0) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (t, kind, self._seq, kind, a, b))

    def pop(self) -> tuple:
        return heapq.heappop(self._heap)

    def peek_time(self) -> float:
        return self._heap[0][0] if self._heap else math.inf

    def peek(self) -> tuple | None:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._seq = 0


@dataclass(frozen=True)
class Notification:
    """Something a controller may want to react to. ``kind`` is e.g. ``"wakeup"`` (M4 adds trucks)."""

    t: float
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class Until:
    """Stop condition for ``Simulator.run``. Combine with ``|``: the run stops at the first one met."""

    def deadline(self, now: float) -> float:
        return math.inf

    def kinds(self) -> frozenset[str]:
        return frozenset()

    def __or__(self, other: Until) -> Until:
        return AnyOf((self, other))


@dataclass(frozen=True)
class TimeLimit(Until):
    """Stop after ``seconds`` of simulated time from the start of the ``run`` call."""

    seconds: float

    def __post_init__(self):
        if not self.seconds >= 0:
            raise ValueError("TimeLimit needs seconds >= 0")

    def deadline(self, now: float) -> float:
        return now + self.seconds


@dataclass(frozen=True)
class At(Until):
    """Stop at the absolute clock time ``t`` (seconds since midnight of the start date)."""

    t: float

    def deadline(self, now: float) -> float:
        return self.t


@dataclass(frozen=True)
class OnNotification(Until):
    """Stop right after the event that emits a notification of one of these kinds."""

    names: frozenset[str]

    def __init__(self, *names: str):
        object.__setattr__(self, "names", frozenset(names))

    def kinds(self) -> frozenset[str]:
        return self.names


@dataclass(frozen=True)
class AnyOf(Until):
    parts: tuple[Until, ...]

    def deadline(self, now: float) -> float:
        return min(p.deadline(now) for p in self.parts)

    def kinds(self) -> frozenset[str]:
        return frozenset().union(*(p.kinds() for p in self.parts))

    def __or__(self, other: Until) -> Until:
        return AnyOf((*self.parts, other))
