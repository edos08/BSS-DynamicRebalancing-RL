"""Gymnasium environments on the BSS simulator v2 (docs/sim-v2/SPEC.md §14)."""

import gymnasium as gym

from bss_gym.grid_env import ACTION_NAMES, GridZoneEnv, RewardConfig

__version__ = "0.1.0"

gym.register(id="bss_gym/GridZone-v0", entry_point="bss_gym.grid_env:GridZoneEnv")

__all__ = ["ACTION_NAMES", "GridZoneEnv", "RewardConfig", "__version__"]
