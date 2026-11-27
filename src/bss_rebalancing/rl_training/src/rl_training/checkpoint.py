"""Training checkpoints: save and resume the whole training state (issue #170).

A checkpoint holds everything needed to continue a run exactly where it stopped:

* the train and target networks and the optimizer;
* epsilon and its schedule position (``steps_done``);
* the random-number states of Python, NumPy, torch and CUDA (action selection, batch sampling);
* the replay buffer, in a separate file (it can be several hundred MB);
* ``extra``: whatever the training loop needs, such as the next episode, the best validation
  score, slot counters and the W&B run id.

Files are written to a temporary name and then renamed, so an interruption never leaves a broken
checkpoint. ``keep`` older checkpoints are kept besides ``last.pt``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

LAST = "last.pt"


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    out = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG states are CPU ByteTensors; torch.load(map_location="cuda") moves them to the GPU (#180)
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        cuda = [s.cpu() for s in state["cuda"]][:torch.cuda.device_count()]   # visible devices may differ
        for i, s in enumerate(cuda):
            torch.cuda.set_rng_state(s, i)


def save_checkpoint(folder: str | Path, agent, extra: dict, episode: int, buffer: bool = True,
                    keep: int = 2) -> Path:
    """Write ``<folder>/ckpt_episode_NNN.pt`` (+ ``.buffer.pt``) and point ``last.pt`` to it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "train_model": agent.train_model.state_dict(),
        "target_model": agent.target_model.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "steps_done": agent.steps_done,
        "rng": rng_state(),
        "extra": dict(extra),
        "episode": int(episode),
        "has_buffer": bool(buffer and agent.replay_buffer is not None),
    }
    path = folder / f"ckpt_episode_{episode:03d}.pt"
    if state["has_buffer"]:
        _atomic_save(agent.replay_buffer.buffer, path.with_suffix(".buffer.pt"))
    _atomic_save(state, path)
    _atomic_save({"path": path.name}, folder / LAST)
    old = sorted(folder.glob("ckpt_episode_*.pt"))
    old = [p for p in old if not p.name.endswith(".buffer.pt")]
    for p in old[:-max(keep, 1)]:
        p.unlink(missing_ok=True)
        p.with_suffix(".buffer.pt").unlink(missing_ok=True)
    return path


def latest_checkpoint(folder: str | Path) -> Path | None:
    folder = Path(folder)
    if not (folder / LAST).exists():
        return None
    name = torch.load(folder / LAST, weights_only=True)["path"]
    path = folder / name
    return path if path.exists() else None


def load_checkpoint(path: str | Path, agent, map_location=None) -> dict:
    """Restore ``agent`` (networks, optimizer, epsilon, replay buffer) and the RNG states.
    Returns the checkpoint (``["episode"]``, ``["extra"]``)."""
    path = Path(path)
    state = torch.load(path, map_location=map_location or agent.device, weights_only=False)
    agent.train_model.load_state_dict(state["train_model"])
    agent.target_model.load_state_dict(state["target_model"])
    agent.optimizer.load_state_dict(state["optimizer"])
    agent.epsilon = state["epsilon"]
    agent.steps_done = state["steps_done"]
    if state.get("has_buffer") and agent.replay_buffer is not None:
        agent.replay_buffer.buffer = torch.load(path.with_suffix(".buffer.pt"), weights_only=False)
    set_rng_state(state["rng"])
    return state
