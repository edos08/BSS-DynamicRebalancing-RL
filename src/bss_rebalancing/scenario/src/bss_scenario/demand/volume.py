"""Daily volume model (issue #78, spec §6.1).

Daily totals are modelled as::

    count(date) = base × month_factor(month) × group_level(group(date)) × day_noise(date)

fitted by least squares on ``log(1 + count)`` over every calendar day of the data year (month and group
dummies). ``day_noise`` is log-normal with mean 1: ``log ε ~ N(-σ²/2, σ²)``, σ from the residuals of
the fit-period days.

Only ``month_factor`` and ``σ`` are used by the generator: the group level and the base are part
of the hourly OD shape (whose rates are trips per hour on a day with factor 1). Month factors are
normalised so that their mean over the **fit-period** days is 1, matching the exposure used for
the shape.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from bss_scenario.demand.calendar import day_group, group_labels, holidays


@dataclass(frozen=True)
class VolumeModel:
    scheme: str
    month_factor: dict[str, float]       # "YYYY-MM" -> factor (mean 1 over fit-period days)
    group_level: dict[str, float]        # group label -> relative level (for reports)
    sigma: float                         # log-sd of the day noise over the fit period
    sigma_all_days: float                # same over every day (reported)
    days: pl.DataFrame                   # date, month, group, holiday, count, month_factor, day_factor

    def factor(self, date: dt.date) -> float:
        return self.month_factor[date.strftime("%Y-%m")]


def daily_counts(trips: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    """Valid trips per local start date for every calendar day in [start, end] (missing days = 0)."""
    days = pl.DataFrame({"date": pl.date_range(start, end, "1d", eager=True)})
    counts = trips.group_by(pl.col("start_time").dt.date().alias("date")).len("count")
    return days.join(counts, on="date", how="left").with_columns(pl.col("count").fill_null(0).cast(pl.Int64))


def fit_volume(daily: pl.DataFrame, scheme: str, fit_period: tuple[dt.date, dt.date],
               holidays_as_sunday: bool = True) -> VolumeModel:
    labels = group_labels(scheme)
    dates = daily["date"].to_list()
    groups = np.array([day_group(d, scheme, holidays_as_sunday) for d in dates])
    months = [d.strftime("%Y-%m") for d in dates]
    month_labels = sorted(set(months))
    count = daily["count"].to_numpy().astype(float)
    # log(1 + count): identical to log(count) at thousands of trips per day, and defined for
    # empty days (data gaps, tiny test bundles)
    y = np.log1p(count)

    m_idx = np.array([month_labels.index(m) for m in months])
    X = np.zeros((len(dates), len(month_labels) + len(labels) - 1))
    X[np.arange(len(dates)), m_idx] = 1.0                       # month dummies (absorb intercept)
    for g in range(1, len(labels)):                             # group dummies, first group = reference
        X[groups == g, len(month_labels) + g - 1] = 1.0
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    in_fit = np.array([fit_period[0] <= d <= fit_period[1] for d in dates])
    if not in_fit.any():
        raise ValueError(f"no days in the fit period {fit_period}")
    # day noise measured on the fit period (winter storms would inflate it for typical days)
    sigma = float(resid[in_fit].std(ddof=1)) if in_fit.sum() > 1 else 0.0
    sigma_all = float(resid.std(ddof=X.shape[1]))
    raw_month = np.exp(coef[:len(month_labels)])
    norm = raw_month[m_idx[in_fit]].mean()
    month_factor = {m: float(raw_month[i] / norm) for i, m in enumerate(month_labels)}
    group_level = {labels[0]: 1.0, **{labels[g]: float(np.exp(coef[len(month_labels) + g - 1]))
                                      for g in range(1, len(labels))}}
    hol = {d for y in {d.year for d in dates} for d in holidays(y)}
    days = daily.with_columns(
        pl.Series("month", months),
        pl.Series("group", [labels[g] for g in groups]),
        pl.Series("holiday", [d in hol for d in dates]),
        pl.Series("month_factor", raw_month[m_idx] / norm),
        pl.Series("day_factor", np.exp(resid)),
        pl.Series("in_fit", in_fit),
    )
    return VolumeModel(scheme, month_factor, group_level, sigma, sigma_all, days)
