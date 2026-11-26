"""Controller interface and episode runner (issue #150).

A controller reacts to the simulator's notifications: ``reset(sim)`` once per episode, then
``act(sim, notes)`` whenever the simulator stops on a notification a controller cares about (a
truck became idle, a timer it set fired, a truck restocked on its own). It acts by submitting truck
commands and setting timers (``sim.wakeup``); it reads the state with ``sim.observe()`` and
``sim.forecast(...)``. It never touches the simulator's internals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from bss_sim.engine import At, Notification, OnNotification

STOP_ON = ("truck_idle", "wakeup", "truck_resumed")


class Controller:
    """Base class: does nothing. Subclasses override :meth:`reset` and :meth:`act`."""

    name = "controller"

    def reset(self, sim) -> None:
        """Called after ``sim.reset``; register partitions, set the first timers."""

    def act(self, sim, notes: list[Notification]) -> None:
        """React to notifications (``truck_idle``, ``wakeup``, ``truck_resumed``, ``command_done``...)."""

    def describe(self) -> dict:
        return {"name": self.name}


@dataclass
class EpisodeResult:
    controller: dict
    seed: int
    start: str
    duration_s: float
    kpis: dict
    decisions: int
    wall_s: float
    metadata: dict = field(default_factory=dict)


def run_episode(sim, controller: Controller, seed: int, start, duration_s: float,
                stop_on: tuple[str, ...] = STOP_ON, max_stalls: int = 10_000) -> EpisodeResult:
    """Run one episode of ``duration_s`` seconds from ``start`` with ``controller`` in charge.

    The simulator runs until one of the ``stop_on`` notifications (or the end); the controller
    then acts on everything emitted since its last turn. A controller that keeps the clock from
    moving (e.g. resubmitting invalid commands forever) is stopped after ``max_stalls`` turns.
    """
    t0 = time.perf_counter()
    notes = sim.reset(seed=seed, start=start)
    end = sim.now + duration_s
    controller.reset(sim)
    decisions = 0
    controller.act(sim, notes)
    stalls, last = 0, sim.now
    until = OnNotification(*stop_on) | At(end)
    while sim.now < end:
        notes = sim.run(until)
        if not notes:
            continue
        decisions += 1
        controller.act(sim, notes)
        stalls = stalls + 1 if sim.now == last else 0
        last = sim.now
        if stalls > max_stalls:
            raise RuntimeError(f"{controller.name}: the clock did not move for {max_stalls} turns at t={sim.now}")
    return EpisodeResult(controller.describe(), seed, str(start), duration_s, sim.kpis(), decisions,
                         time.perf_counter() - t0, sim.metadata())
