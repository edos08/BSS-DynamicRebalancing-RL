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

from bss_sim.commands import Command, Drop, GoTo, PickUp, Restock, ToDepot, Wait

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
    limit: str = ""                # why the plan is below the request ("" if it is not)
    # accounting (KPIs, #126)
    km: float = 0.0
    drive_s: float = 0.0
    service_s: float = 0.0
    wait_s: float = 0.0
    commands_done: int = 0
    picked: int = 0
    dropped: int = 0

    @property
    def busy(self) -> bool:
        return self.current is not None


def validate(cmd: Command, n_nodes: int) -> str:
    """Why a command is invalid regardless of the state ("" if it is fine)."""
    def node_ok(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < n_nodes

    def count_ok(n) -> bool:
        return isinstance(n, int) and not isinstance(n, bool) and n >= 0

    if isinstance(cmd, GoTo):
        if not node_ok(cmd.node):
            return f"invalid: node {cmd.node!r} not a bike node"
    elif isinstance(cmd, Wait):
        if not (cmd.seconds >= 0):
            return "invalid: negative wait"
    elif isinstance(cmd, PickUp):
        if not count_ok(cmd.n) or cmd.policy not in ("lowest_battery", "nearest") or not cmd.radius_m >= 0:
            return "invalid: PickUp needs n >= 0, policy lowest_battery|nearest, radius_m >= 0"
    elif isinstance(cmd, Drop):
        if not count_ok(cmd.n) or (cmd.at is not None and (not cmd.at or not all(node_ok(v) for v in cmd.at))):
            return "invalid: Drop needs n >= 0 and at = None or bike nodes"
    elif isinstance(cmd, Restock):
        if (cmd.load is not None and not count_ok(cmd.load)) or (cmd.batteries is not None and not count_ok(cmd.batteries)):
            return "invalid: Restock load / batteries must be >= 0"
    elif not isinstance(cmd, ToDepot):
        return f"unsupported: {type(cmd).__name__}"
    return ""
