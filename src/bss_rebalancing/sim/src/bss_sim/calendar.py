"""Day groups and holidays (same rules as ``bss_scenario.demand.calendar``; checked by a test).

Duplicated on purpose: the simulator must not import the bundle builder (spec §3).
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

DAY_GROUPS: dict[str, tuple[int, ...]] = {
    "per_weekday": (0, 1, 2, 3, 4, 5, 6),
    "day_types": (0, 0, 0, 0, 1, 2, 3),
    "workday_weekend": (0, 0, 0, 0, 0, 1, 1),
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: dt.date) -> dt.date:
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[dt.date]:
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    return frozenset({
        _observed(dt.date(year, 1, 1)), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
        _nth_weekday(year, 4, 0, 3), _nth_weekday(year, 5, 0, -1), _observed(dt.date(year, 6, 19)),
        _observed(dt.date(year, 7, 4)), _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 10, 0, 2),
        _observed(dt.date(year, 11, 11)), thanksgiving, thanksgiving + dt.timedelta(days=1),
        _observed(dt.date(year, 12, 25)),
    })


def day_group(day: dt.date, scheme: str, holidays_as_sunday: bool = True) -> int:
    if scheme not in DAY_GROUPS:
        raise ValueError(f"unknown day_groups {scheme!r}")
    weekday = 6 if holidays_as_sunday and day in holidays(day.year) else day.weekday()
    return DAY_GROUPS[scheme][weekday]
