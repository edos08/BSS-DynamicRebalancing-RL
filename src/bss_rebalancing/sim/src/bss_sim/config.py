"""Typed simulator configuration (issue #95, spec §7 and §10).

Frozen dataclasses with the defaults agreed in the spec. ``SimConfig.from_dict`` / ``from_yaml``
reject unknown keys and out-of-range values with a message naming the field;
``to_dict`` is the inverse and ``config_hash`` identifies a configuration in run metadata.

YAML is optional (``pyyaml``); the simulator itself only needs NumPy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Invalid configuration (unknown key, wrong type or out-of-range value)."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class DemandConfig:
    generator: str = "poisson"          # poisson | replay
    replay_seed_jitter: bool = True     # replay: seeded draw of nodes inside each place's catchment

    def __post_init__(self):
        _check(self.generator in ("poisson", "replay"), f"demand.generator must be 'poisson' or 'replay', got {self.generator!r}")


@dataclass(frozen=True)
class FleetConfig:
    size: int = 800                                   # bikes in the city at the start (spec §7.2)
    ebike_share: float | None = None                  # None -> share observed in the bundle's trips
    battery_range_km: float = 60.0                    # e-bike full range
    initial_battery: tuple[float, float] = (0.3, 1.0)  # uniform fraction of full range
    initial_distribution: str = "proportional_to_demand"  # | uniform | from_snapshot

    def __post_init__(self):
        _check(self.size >= 0, "fleet.size must be >= 0")
        _check(self.ebike_share is None or 0.0 <= self.ebike_share <= 1.0, "fleet.ebike_share must be in [0, 1]")
        _check(self.battery_range_km > 0, "fleet.battery_range_km must be > 0")
        lo, hi = self.initial_battery
        _check(0.0 <= lo <= hi <= 1.0, "fleet.initial_battery must be (low, high) with 0 <= low <= high <= 1")
        _check(self.initial_distribution in ("proportional_to_demand", "uniform", "from_snapshot"),
               f"fleet.initial_distribution: unknown {self.initial_distribution!r}")


@dataclass(frozen=True)
class EnergyConfig:
    consumption_km_per_km: float = 1.0   # battery km used per km ridden
    reserve_km: float = 1.0              # a user takes an e-bike only if it covers the trip + this reserve
    external_trip_km: float = 3.0        # assumed in-city part of a trip to/from EXTERNAL (battery check, use)
    dead_ebike_rideable: bool = False    # True: an e-bike without charge can be ridden like a classic bike
    round_trip_speed_mps: float = 2.5    # round trips by place: distance = ride time x this pace
    battery_bands: tuple[float, ...] = (0.2, 0.5)  # band edges (fraction of range) for aggregates

    def __post_init__(self):
        _check(self.consumption_km_per_km >= 0, "energy.consumption_km_per_km must be >= 0")
        _check(self.reserve_km >= 0, "energy.reserve_km must be >= 0")
        _check(self.external_trip_km >= 0, "energy.external_trip_km must be >= 0")
        _check(self.round_trip_speed_mps > 0, "energy.round_trip_speed_mps must be > 0")
        b = self.battery_bands
        _check(all(0 < x < 1 for x in b) and list(b) == sorted(set(b)),
               "energy.battery_bands must be increasing fractions in (0, 1)")


@dataclass(frozen=True)
class UserConfig:
    walk_beta_m: float = 150.0           # acceptance exp(-w / beta) (spec D17)
    walk_radius_m: float = 400.0
    walking_speed_mps: float = 1.4
    kind_tolerance_m: float = 0.0        # extra walk accepted to get the preferred bike type
    redirect_radius_m: float = 500.0     # docked: farther free dock than this counts as NO_DOCK

    def __post_init__(self):
        _check(self.walk_beta_m > 0, "users.walk_beta_m must be > 0")
        _check(self.walk_radius_m >= 0, "users.walk_radius_m must be >= 0")
        _check(self.walking_speed_mps > 0, "users.walking_speed_mps must be > 0")
        _check(self.kind_tolerance_m >= 0, "users.kind_tolerance_m must be >= 0")
        _check(self.redirect_radius_m >= 0, "users.redirect_radius_m must be >= 0")


@dataclass(frozen=True)
class ExternalConfig:
    reservoir: int = 500                 # bikes outside the city at the start (spec §6.5)
    ride_time_s: float = 900.0           # time from leaving the outside origin to arriving in the city

    def __post_init__(self):
        _check(self.reservoir >= 0, "external.reservoir must be >= 0")
        _check(self.ride_time_s >= 0, "external.ride_time_s must be >= 0")


@dataclass(frozen=True)
class TelemetryConfig:
    event_log: bool = False              # columnar log of every event outcome
    snapshot_every_s: float | None = None  # periodic per-node bike counts in the log

    def __post_init__(self):
        _check(self.snapshot_every_s is None or self.snapshot_every_s > 0, "telemetry.snapshot_every_s must be > 0")


@dataclass(frozen=True)
class SimConfig:
    mode: str = "dockless"               # dockless | docked
    demand: DemandConfig = field(default_factory=DemandConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    users: UserConfig = field(default_factory=UserConfig)
    external: ExternalConfig = field(default_factory=ExternalConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    debug: bool = False                  # check invariants after every event (slow)

    def __post_init__(self):
        _check(self.mode in ("dockless", "docked"), f"mode must be 'dockless' or 'docked', got {self.mode!r}")

    @classmethod
    def from_dict(cls, data: dict | None) -> SimConfig:
        return _from_dict(cls, data or {}, "")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimConfig:
        import yaml  # optional dependency
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        return _to_plain(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def replace(self, **changes) -> SimConfig:
        """Copy with top-level or dotted changes, e.g. ``replace(**{"users.walk_beta_m": 200})``."""
        data = self.to_dict()
        for key, value in changes.items():
            node = data
            *path, last = key.split(".")
            for p in path:
                node = node[p]
            node[last] = value
        return SimConfig.from_dict(data)


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return [_to_plain(x) for x in obj]
    return obj


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        _check(isinstance(value, dict), f"{where} must be a mapping")
        return _from_dict(tp, value, where + ".")
    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(tp)
        if value is None:
            _check(type(None) in args, f"{where} must not be null")
            return None
        return _coerce(next(a for a in args if a is not type(None)), value, where)
    if origin is tuple:
        _check(isinstance(value, (list, tuple)), f"{where} must be a list")
        args = typing.get_args(tp)
        inner = args[0]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(inner, v, where) for v in value)
        _check(len(value) == len(args), f"{where} must have {len(args)} items")
        return tuple(_coerce(a, v, where) for a, v in zip(args, value))
    if tp is bool:
        _check(isinstance(value, bool), f"{where} must be true or false")
        return value
    if tp is int:
        _check(isinstance(value, int) and not isinstance(value, bool), f"{where} must be an integer")
        return value
    if tp is float:
        _check(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where} must be a number")
        return float(value)
    if tp is str:
        _check(isinstance(value, str), f"{where} must be a string")
        return value
    raise ConfigError(f"{where}: unsupported type {tp}")


def _from_dict(cls: type, data: dict, prefix: str):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    _check(not unknown, f"unknown config key(s): {', '.join(prefix + k for k in sorted(unknown))}")
    return cls(**{k: _coerce(hints[k], v, prefix + k) for k, v in data.items()})
