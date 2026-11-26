"""Named, independent random streams for one run (issue #95, spec §10).

Every stream is derived from the run seed and a fixed key computed from its **name**
(not from the order in which streams are created), so adding a component or a stream never
changes the numbers drawn by the others.
"""

from __future__ import annotations

import zlib

import numpy as np

STREAMS = ("demand", "users", "fleet", "external", "forecast", "travel")


class Streams:
    def __init__(self, seed: int):
        if int(seed) < 0:
            raise ValueError("seed must be >= 0")
        self.seed = int(seed)

    @staticmethod
    def key(name: str) -> int:
        return zlib.crc32(name.encode())

    def seq(self, name: str) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.seed, spawn_key=(self.key(name),))

    def generator(self, name: str) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.seq(name)))

    def int_seed(self, name: str) -> int:
        """A 64-bit integer seed for components that derive their own sub-streams (e.g. demand)."""
        lo, hi = self.seq(name).generate_state(2, np.uint32)
        return int(lo) | (int(hi) << 32)
