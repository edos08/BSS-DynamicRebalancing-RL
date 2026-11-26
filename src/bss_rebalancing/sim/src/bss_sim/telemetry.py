"""Telemetry: event log, snapshots and run metadata (issue #103, spec §11).

**Event log** (``telemetry.event_log: true``): one row per event outcome, columns

=========  =======  ==============================================================================
column     dtype    meaning
=========  =======  ==============================================================================
t          float64  clock (s since local midnight of the start date)
event      uint8    ``TripRequest`` (3), ``TripStart`` (2), ``TripEnd`` (1), ``Redirect`` (5)
request    int64    request id (order of arrival in the run)
bike       int32    bike id (-1 when the request failed)
node       int32    request: origin (or destination if from outside); start: where the bike was
                    taken; end / redirect: where it is (going to be) docked or parked
outcome    int8     :data:`bss_sim.users.REASONS` index (0 = served)
value      float32  request: walk (m); start: ride time (s); redirect: extra distance (m)
=========  =======  ==============================================================================

Logging off costs one ``is None`` test per event. ``save`` writes ``.npz`` (NumPy only) or
``.parquet`` (needs ``pyarrow``), with the run metadata (bundle, config and its hash, seed, start,
versions) embedded, so a log is self-describing.

**Snapshots** (``telemetry.snapshot_every_s``): parked bikes per node at regular times, saved as
``t`` [K] and ``bikes`` [K, nodes] (int16) in an ``.npz``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

COLUMNS = (("t", np.float64), ("event", np.uint8), ("request", np.int64), ("bike", np.int32),
           ("node", np.int32), ("outcome", np.int8), ("value", np.float32))


def to_arrays(rows: list[tuple]) -> dict[str, np.ndarray]:
    cols = list(zip(*rows)) if rows else [()] * len(COLUMNS)
    return {name: np.array(c, dtype=dt) for (name, dt), c in zip(COLUMNS, cols)}


def save_events(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(metadata, sort_keys=True)
    if path.suffix == ".npz":
        np.savez_compressed(path, metadata=np.array(meta), **arrays)
    elif path.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # optional dependency
            raise ImportError("saving Parquet logs needs pyarrow (pip install pyarrow), or use .npz") from exc
        table = pa.table(arrays).replace_schema_metadata({"bss_sim": meta})
        pq.write_table(table, path)
    else:
        raise ValueError(f"unsupported log format {path.suffix!r}: use .npz or .parquet")
    return path


def load_events(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            return {name: z[name] for name, _ in COLUMNS}, json.loads(str(z["metadata"]))
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[b"bss_sim"])
    return {name: table[name].to_numpy() for name, _ in COLUMNS}, meta


class WalkHistogram:
    """Walking distances of served users: bin 0 = no walk, then ``(0, bin], (bin, 2 bin], ...``."""

    def __init__(self, max_m: float, bin_m: float = 10.0):
        self.bin = float(bin_m)
        self.counts = [0] * (math.ceil(max_m / bin_m) + 2)

    def add(self, w: float) -> None:
        i = 0 if w <= 0.0 else math.ceil(w / self.bin)
        self.counts[min(i, len(self.counts) - 1)] += 1

    def quantile(self, q: float) -> float:
        """Upper edge of the bin holding the ``q`` quantile (0 when it is a user who did not walk)."""
        total = sum(self.counts)
        if total == 0:
            return 0.0
        acc = 0
        for i, c in enumerate(self.counts):
            acc += c
            if acc >= q * total:
                return i * self.bin
        return (len(self.counts) - 1) * self.bin
