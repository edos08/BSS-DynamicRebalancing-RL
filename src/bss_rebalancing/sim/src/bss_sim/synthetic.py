"""Synthetic scenario bundle of Cambridge size, for tests and benchmarks (issue #104).

CI has no real data (bundles are built from git-ignored downloads), so the performance budget
(spec §12) is enforced on a synthetic bundle with the same orders of magnitude as Cambridge 2024:

* ``n_side`` x ``n_side`` street grid (default 70 x 70 = 4,900 nodes, 60 m blocks: ~17 km²);
  network distance = Manhattan distance (stored as the float32 ``bike_dist`` matrix, ~96 MB);
* walking neighbours up to ``walk_max_m`` (CSR, sorted, self first);
* ``n_stations`` stations (default 91) on random nodes, ~19 docks each, plus ``EXTERNAL``;
* demand: four day groups, a two-peak hourly profile, gravity-like OD with ~54 % of trips
  having one end outside, ~9,500 requests on a weekday; kernel spreading over nodes within 400 m;
* ride times: speed quantiles by distance band (classic ~12 km/h, e-bike ~16 km/h).

Only the files the simulator reads are written (npz/json/npy), with ``schema_version`` 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HOURLY = np.array([0.6, 0.3, 0.2, 0.1, 0.2, 0.6, 1.8, 4.0, 7.0, 5.0, 4.0, 4.5, 5.2, 5.0, 5.0, 5.6, 7.0, 9.0,
                   7.5, 5.5, 4.0, 3.0, 2.0, 1.2])


def make_bundle(path: str | Path, n_side: int = 70, block_m: float = 60.0, n_stations: int = 91,
                weekday_trips: float = 9500.0, external_share: float = 0.54, walk_max_m: float = 600.0,
                seed: int = 0) -> Path:
    root = Path(path)
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    gx, gy = np.divmod(np.arange(n), n_side)
    x, y = gx * block_m, gy * block_m

    (root / "network").mkdir(parents=True, exist_ok=True)
    dist = (np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])).astype(np.float32)
    np.save(root / "network" / "bike_dist.npy", dist)

    indptr, idx, dd = [0], [], []
    r = int(walk_max_m // block_m)
    for v in range(n):
        lo_x, hi_x = max(gx[v] - r, 0), min(gx[v] + r, n_side - 1)
        lo_y, hi_y = max(gy[v] - r, 0), min(gy[v] + r, n_side - 1)
        cx, cy = np.meshgrid(np.arange(lo_x, hi_x + 1), np.arange(lo_y, hi_y + 1), indexing="ij")
        cand = (cx * n_side + cy).ravel()
        d = dist[v, cand]
        keep = d <= walk_max_m
        cand, d = cand[keep], d[keep]
        order = np.lexsort((cand, d))
        idx.append(cand[order])
        dd.append(d[order])
        indptr.append(indptr[-1] + len(order))
    np.savez(root / "network" / "bike_walk_nbrs.npz", indptr=np.array(indptr, np.int64),
             indices=np.concatenate(idx).astype(np.int32), dist=np.concatenate(dd).astype(np.float32),
             max_radius_m=np.float64(walk_max_m))

    S = n_stations
    st_node = np.sort(rng.choice(n, size=S, replace=False)).astype(np.int32)
    capacity = rng.integers(13, 27, size=S).astype(np.int32)
    (root / "stations").mkdir(exist_ok=True)
    np.savez(root / "stations" / "stations.npz", station_id=np.array([f"S{i:03d}" for i in range(S)]),
             bike_node=st_node, capacity=capacity, lat=42.36 + gy[st_node] * 1e-5, lon=-71.1 + gx[st_node] * 1e-5)

    # demand: rates[g, h, P, P] = expected requests in the hour (before the day factor)
    P = S + 1
    pop = rng.gamma(1.5, 1.0, S)
    sd = dist[np.ix_(st_node, st_node)].astype(np.float64)
    od = np.zeros((P, P))
    od[:S, :S] = pop[:, None] * pop[None, :] * np.exp(-sd / 1500.0)
    inner = od[:S, :S].sum()
    ext_total = inner * external_share / (1 - external_share)
    od[S, :S] = 0.5 * ext_total * pop / pop.sum()
    od[:S, S] = 0.5 * ext_total * pop / pop.sum()
    od /= od.sum()
    hourly = HOURLY / HOURLY.sum()
    level = np.array([1.0, 1.0, 0.83, 0.72])  # mon-thu, fri, sat, sun
    rates = weekday_trips * level[:, None, None, None] * hourly[None, :, None, None] * od[None, None]
    (root / "demand").mkdir(exist_ok=True)
    np.savez(root / "demand" / "shape.npz", rates=rates.astype(np.float32),
             meta=np.array(json.dumps({"scheme": "day_types", "synthetic": True})))
    months = {f"{yr}-{m:02d}": 1.0 for yr in (2023, 2024, 2025) for m in range(1, 13)}
    (root / "demand" / "volume.json").write_text(json.dumps({
        "scheme": "day_types", "month_factor": months, "sigma": 0.22, "ebike_share": 0.2,
        "holidays_as_sunday": True}))

    sp_ptr, sp_nodes, sp_probs = [0], [], []
    for s in range(S):
        d = dist[st_node[s]]
        near = np.flatnonzero(d <= 400.0)
        w = np.exp(-d[near] / 150.0)
        sp_nodes.append(near)
        sp_probs.append(w / w.sum())
        sp_ptr.append(sp_ptr[-1] + len(near))
    np.savez(root / "demand" / "spread.npz", mode=np.array("kernel"), indptr=np.array(sp_ptr, np.int64),
             nodes=np.concatenate(sp_nodes).astype(np.int32), probs=np.concatenate(sp_probs).astype(np.float32),
             radius_m=np.float64(400.0), scale_m=np.float64(150.0), station_nodes=st_node)

    levels = np.linspace(0.0, 1.0, 101)
    bands = np.array([0, 500, 1000, 2000, 3000, 5000, 8000, 1e9], dtype=np.float64)
    nb = len(bands) - 1
    speed_q = np.empty((2, nb, len(levels)))
    for k, med in ((0, 3.3), (1, 4.4)):  # m/s
        q = med * np.exp(0.35 * np.sqrt(2) * _erfinv(2 * np.clip(levels, 0.005, 0.995) - 1))
        speed_q[k] = q[None, :] * np.linspace(0.8, 1.1, nb)[:, None]
    round_q = np.stack([600 + 3000 * levels ** 1.5, 500 + 2500 * levels ** 1.5])
    np.savez(root / "demand" / "ride_time.npz", bands_m=bands, levels=levels, speed_q=speed_q, round_q=round_q,
             counts=np.ones((2, nb), np.int64), round_counts=np.ones(2, np.int64),
             bike_types=np.array(["classic", "electric"]))
    (root / "demand" / "replay").mkdir(exist_ok=True)
    (root / "demand" / "replay" / "index.json").write_text("[]")
    (root / "manifest.json").write_text(json.dumps({"bundle_id": f"synthetic-{n_side}x{n_side}", "schema_version": 1,
                                                    "synthetic": True, "seed": seed}, indent=2))
    return root


def _erfinv(y: np.ndarray) -> np.ndarray:
    """Inverse error function (Giles' single-precision approximation; NumPy has no erfinv)."""
    w = -np.log((1.0 - y) * (1.0 + y))
    small = w < 5.0
    out = np.empty_like(y)
    ws = w[small] - 2.5
    p = 2.81022636e-08
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164,
              0.246640727, 1.50140941):
        p = c + p * ws
    out[small] = p * y[small]
    wl = np.sqrt(w[~small]) - 3.0
    p = -0.000200214257
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047,
              1.00167406, 2.83297682):
        p = c + p * wl
    out[~small] = p * y[~small]
    return out
