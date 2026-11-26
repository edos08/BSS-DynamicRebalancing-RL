"""Trucks: externally commanded operators (issues #123–#125, spec §9, D1–D3).

Each truck has a **command queue**. It is idle until it gets commands, executes them in order and,
when the queue runs empty, emits a ``"truck_idle"`` notification; every finished command emits a
``"command_done"`` notification carrying its :class:`~bss_sim.commands.CommandResult`.

A truck is always at a **stop**: a bike node and the drive node nearest to it. ``GoTo`` drives on
the drive network with the run's :mod:`travel model <bss_sim.travel>`; while it drives, its stop
is still the one it left (``eta`` tells when it arrives).

Execution is honest about time: a command occupies the truck for its driving or service time and
the state changes when it **completes**.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from bss_sim.commands import Command, GoTo, ToDepot, Wait

TRUCK_IDLE = "truck_idle"
COMMAND_DONE = "command_done"


@dataclass
class Truck:
    id: int
    stop: int                      # bike node of the current stop
    drive_node: int
    load: list[int] = field(default_factory=list)   # bike ids on board
    stock: int = 0                 # charged e-bike batteries
    used: int = 0                  # depleted batteries collected
    queue: deque = field(default_factory=deque)
    current: Command | None = None
    t_start: float = 0.0
    eta: float = 0.0               # when the current command ends
    serial: int = 0                # increments per command; stale events are ignored
    target: tuple[int, int] | None = None  # (bike node, drive node) of a drive in progress
    planned: int = 0               # quantity planned at the start of a service
    # accounting (KPIs, #126)
    km: float = 0.0
    drive_s: float = 0.0
    service_s: float = 0.0
    wait_s: float = 0.0
    commands_done: int = 0

    @property
    def busy(self) -> bool:
        return self.current is not None


def validate(cmd: Command, n_nodes: int) -> str:
    """Why a command is invalid ("" if it is fine)."""
    if isinstance(cmd, GoTo):
        if not (isinstance(cmd.node, int) and 0 <= cmd.node < n_nodes):
            return f"invalid: node {cmd.node!r} not a bike node"
    elif isinstance(cmd, Wait):
        if not (cmd.seconds >= 0):
            return "invalid: negative wait"
    elif not isinstance(cmd, ToDepot):
        return f"unsupported: {type(cmd).__name__}"
    return ""
