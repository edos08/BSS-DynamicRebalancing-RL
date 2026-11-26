"""Calendar: day groups and holidays (issue #78, spec §6.1).

A **day group** is a set of days that share one hourly OD shape. Holidays behave like Sundays
(Cambridge 2024: Thanksgiving had 16 % of a normal Thursday's trips, Indigenous Peoples' Day 55 %),
so by default they are put in the Sunday group.

Holidays: US federal holidays plus Massachusetts Patriots' Day, and the day after Thanksgiving,
computed by rule (no external package).
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

#: day group schemes: name -> (group labels, weekday (Mon=0) -> group index)
DAY_GROUPS: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "per_weekday": (("mon", "tue", "wed", "thu", "fri", "sat", "sun"), (0, 1, 2, 3, 4, 5, 6)),
    "day_types": (("mon-thu", "fri", "sat", "sun"), (0, 0, 0, 0, 1, 2, 3)),
    "workday_weekend": (("workday", "weekend"), (0, 0, 0, 0, 0, 1, 1)),
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th (1-based; -1 = last) given weekday (Mon=0) of a month."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: dt.date) -> dt.date:
    """Federal rule: Saturday holidays are observed on Friday, Sunday holidays on Monday."""
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def holidays(year: int) -> dict[dt.date, str]:
    """US federal holidays (observed dates), Patriots' Day (MA) and the day after Thanksgiving."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    days = {
        _observed(dt.date(year, 1, 1)): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Presidents' Day",
        _nth_weekday(year, 4, 0, 3): "Patriots' Day (MA)",
        _nth_weekday(year, 5, 0, -1): "Memorial Day",
        _observed(dt.date(year, 6, 19)): "Juneteenth",
        _observed(dt.date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 10, 0, 2): "Indigenous Peoples' Day",
        _observed(dt.date(year, 11, 11)): "Veterans Day",
        thanksgiving: "Thanksgiving",
        thanksgiving + dt.timedelta(days=1): "Day after Thanksgiving",
        _observed(dt.date(year, 12, 25)): "Christmas Day",
    }
    return dict(sorted(days.items()))


def group_labels(scheme: str) -> tuple[str, ...]:
    if scheme not in DAY_GROUPS:
        raise ValueError(f"unknown day_groups {scheme!r}; choose from {sorted(DAY_GROUPS)}")
    return DAY_GROUPS[scheme][0]


def day_group(day: dt.date, scheme: str, holidays_as_sunday: bool = True) -> int:
    """Group index of a date under ``scheme``."""
    group_labels(scheme)  # validates the scheme name
    mapping = DAY_GROUPS[scheme][1]
    weekday = 6 if holidays_as_sunday and day in holidays(day.year) else day.weekday()
    return mapping[weekday]
