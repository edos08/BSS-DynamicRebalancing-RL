"""Statistical comparison of request streams (issue #85, spec §11).

Pure NumPy metrics comparing two sets of :class:`~bss_sim.demand.TripRequests`, e.g. generated
vs recorded requests on held-out days:

* ``hourly_tvd``: total variation distance between hourly profiles (share of requests per hour).
* ``od_jsd``: Jensen–Shannon divergence (bits) between OD distributions over places.
* ``origin_tvd`` / ``destination_tvd``: TVD between per-place origin / destination shares.
* ``ride_ks``: two-sample Kolmogorov–Smirnov statistic between in-city ride times.
* ``daily_mean_ratio``: mean requests per day, generated / reference.

All are 0 for identical distributions.
"""

from __future__ import annotations

import numpy as np

from bss_sim.demand import TripRequests


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(_shares(a) - _shares(b)).sum())


def jsd_bits(a: np.ndarray, b: np.ndarray) -> float:
    p, q = _shares(a.ravel().astype(float)), _shares(b.ravel().astype(float))
    m = 0.5 * (p + q)

    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ks(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                        - np.searchsorted(b, grid, side="right") / len(b)).max())


def compare(generated: TripRequests, reference: TripRequests, n_places: int,
            days_generated: int = 1, days_reference: int = 1) -> dict:
    """All metrics between two request sets over the same kind of days."""
    def hourly(r):
        return np.bincount(np.minimum((r.t_s // 3600).astype(int), 23), minlength=24)

    def od(r):
        return np.bincount(r.o_place.astype(int) * n_places + r.d_place, minlength=n_places * n_places)

    def places(x):
        return np.bincount(x.astype(int), minlength=n_places)

    ext = n_places - 1
    inside_g = (generated.o_place != ext) & (generated.d_place != ext)
    inside_r = (reference.o_place != ext) & (reference.d_place != ext)
    return {
        "requests_per_day": len(generated) / days_generated,
        "reference_per_day": len(reference) / days_reference,
        "daily_mean_ratio": (len(generated) / days_generated) / max(len(reference) / days_reference, 1e-12),
        "hourly_tvd": tvd(hourly(generated), hourly(reference)),
        "od_jsd": jsd_bits(od(generated), od(reference)),
        "origin_tvd": tvd(places(generated.o_place), places(reference.o_place)),
        "destination_tvd": tvd(places(generated.d_place), places(reference.d_place)),
        "ride_ks": ks(generated.ride_time_s[inside_g].astype(float), reference.ride_time_s[inside_r].astype(float)),
    }
