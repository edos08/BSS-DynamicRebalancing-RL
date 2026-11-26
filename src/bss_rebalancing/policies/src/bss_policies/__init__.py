"""Baseline controllers for the BSS simulator v2 (docs/sim-v2/SPEC.md §14).

Controllers drive ``bss_sim`` only through its public API (observe, forecast, submit, wakeup,
run); ``bss_sim`` never imports this package.
"""

from bss_policies.base import Controller, EpisodeResult, run_episode
from bss_policies.donothing import DoNothing
from bss_policies.greedy import GreedyDeficit
from bss_policies.static_tsp import StaticTSP

__version__ = "0.1.0"

__all__ = ["Controller", "DoNothing", "EpisodeResult", "GreedyDeficit", "StaticTSP", "run_episode", "__version__"]
