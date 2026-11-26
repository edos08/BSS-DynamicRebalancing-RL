"""Truck commands and their results (issues #123–#125, spec §9).

Commands are plain frozen dataclasses. A controller submits them to a truck's queue with
``Simulator.submit(truck, [...])``; the truck executes them in order, taking the time each one
really takes, and reports a :class:`CommandResult` for each in a ``"command_done"`` notification.
Locations are **bike-network nodes** (where bikes are); the truck drives to the nearest drive node.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoTo:
    """Drive to the drive node nearest to bike node ``node``; that node becomes the truck's stop."""

    node: int


@dataclass(frozen=True)
class Wait:
    """Stay where it is for ``seconds`` (busy: the next command starts afterwards)."""

    seconds: float


@dataclass(frozen=True)
class PickUp:
    """Load up to ``n`` parked bikes within ``radius_m`` (walking distance) of the stop.

    ``policy``: ``"lowest_battery"`` (e-bikes with the least charge first, then classic bikes, by
    distance) or ``"nearest"`` (by distance)."""

    n: int
    policy: str = "lowest_battery"
    radius_m: float = 100.0


@dataclass(frozen=True)
class Drop:
    """Unload up to ``n`` bikes at the stop, or spread round-robin over the bike nodes ``at``
    (each within ``max_spread_m`` walking distance of the stop). E-bikes with the most charge first."""

    n: int
    at: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SwapBattery:
    """Replace the batteries of up to ``n`` parked e-bikes within ``radius_m``, lowest charge first,
    with charged ones from the truck's stock."""

    n: int
    radius_m: float = 100.0


@dataclass(frozen=True)
class ToDepot:
    """Drive to the depot."""


@dataclass(frozen=True)
class Restock:
    """At the depot: set the bikes on board to ``load`` (unload to / load from the depot's pool;
    ``None`` keeps them), refill the battery stock to ``batteries`` (``None`` = full), charge the
    e-bikes on board."""

    load: int | None = None
    batteries: int | None = None


Command = GoTo | Wait | PickUp | Drop | SwapBattery | ToDepot | Restock


@dataclass(frozen=True)
class CommandResult:
    truck: int
    command: Command
    requested: int            # bikes / batteries asked for (1 for moves and waits)
    executed: int             # what was actually done
    t_start: float
    t_end: float
    reason: str = ""          # "" = done as asked; otherwise why it was partial, skipped or rejected

    @property
    def ok(self) -> bool:
        return self.reason == ""
