"""
BSS RL Training

Reinforcement Learning training and validation module for the BSS Dynamic Rebalancing project.
"""

__version__ = "1.0.0"

import warnings

# torch_geometric 2.8 calls torch.jit.script at import, deprecated in torch 2.15: harmless, not ours.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
__author__ = "Edoardo Scarpel"

from rl_training.agents import DQNAgent
from rl_training.memory import ReplayBuffer, PairData
from rl_training.networks import DQN
from rl_training.results import EpisodeResults, ResultsManager
from rl_training.utils import (
    set_seed,
    setup_device,
    convert_seconds_to_hours_minutes,
    build_cell_graph_from_cells,
    update_cell_graph_features,
    convert_graph_to_data,
)
from rl_training.logging_config import init_logging, LoggingConfig, get_logger

__all__ = [
    "DQNAgent",
    "ReplayBuffer",
    "PairData",
    "DQN",
    "EpisodeResults",
    "ResultsManager",
    "convert_graph_to_data",
    "convert_seconds_to_hours_minutes",
    "build_cell_graph_from_cells",
    "update_cell_graph_features",
    "set_seed",
    "setup_device",
    "init_logging",
    "LoggingConfig",
    "get_logger"
]
