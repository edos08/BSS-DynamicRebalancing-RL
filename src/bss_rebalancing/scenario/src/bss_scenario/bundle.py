"""Scenario bundle: build everything from one config, load and validate it (issue #16, spec §4).

A bundle is a directory the simulator only reads::

    <bundle>/
    ├── scenario.yaml        id, schema_version, timezone, slot length, fit period, components
    ├── manifest.json        inputs (with SHA-256), package versions, command line, outputs (SHA-256)
    ├── trips/               YYYY-MM.parquet + cleaning_report.json         (bss-scenario trips)
    ├── network/             nodes, edges, distance/time matrices, ...      (bss-scenario networks)
    ├── stations/            stations.parquet + stations_report.json        (bss-scenario stations)
    └── demand/              places, volume, OD shape (+ ride time, spreading)  (bss-scenario demand)

Traffic congestion (#57) is added as a further component with the same rules.

Rules: no pickles (Parquet, NumPy ``.npy``/``.npz``, JSON, YAML only); ``schema_version`` is
checked on load; every output file's SHA-256 is recorded and can be verified.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import yaml

from bss_scenario import __version__

SCHEMA_VERSION = 1
ALLOWED_SUFFIXES = {".parquet", ".npy", ".npz", ".json", ".yaml"}

#: Files each component must contain (relative to the bundle).
COMPONENT_FILES: dict[str, tuple[str, ...]] = {
    "trips": ("trips/cleaning_report.json",),
    "network": ("network/bike_nodes.parquet", "network/bike_edges.parquet", "network/bike_dist.npy",
                "network/bike_walk_nbrs.npz", "network/drive_nodes.parquet", "network/drive_edges.parquet",
                "network/drive_time.npy", "network/bike_to_drive.npz", "network/network_report.json"),
    "stations": ("stations/stations.parquet", "stations/stations_report.json"),
    "demand": ("demand/places.parquet", "demand/volume.parquet", "demand/volume.json", "demand/shape.npz",
               "demand/ride_time.npz", "demand/spread.npz", "demand/replay/index.json",
               "demand/demand_report.json"),
}


@dataclass(frozen=True)
class BuildConfig:
    """What to build. Loaded from YAML; relative paths are relative to the YAML file."""

    id: str
    trips_raw: Path
    boundary: Path
    bike_graphml: Path
    drive_graphml: Path
    gbfs_station_information: Path
    timezone: str = "America/New_York"
    slot_minutes: int = 60
    fit_period: tuple[str, str] = ("2024-04-01", "2024-11-30")
    min_duration_s: float = 60.0
    max_duration_s: float = 10800.0
    walk_radius_max_m: float = 800.0
    demand_seed: int = 2024

    @classmethod
    def from_yaml(cls, path: str | Path) -> BuildConfig:
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        base = path.parent
        inputs = data.get("inputs", {})
        required = ("trips_raw", "boundary", "bike_graphml", "drive_graphml", "gbfs_station_information")
        missing = [k for k in required if k not in inputs]
        if "id" not in data or missing:
            raise ValueError(f"{path}: needs 'id' and inputs {list(required)}; missing {missing or ['id']}")
        opts = data.get("options", {})
        return cls(
            id=str(data["id"]),
            **{k: (base / inputs[k]).resolve() for k in required},
            timezone=data.get("timezone", cls.timezone),
            slot_minutes=int(data.get("slot_minutes", cls.slot_minutes)),
            fit_period=tuple(data.get("fit_period", cls.fit_period)),
            min_duration_s=float(opts.get("min_duration_s", cls.min_duration_s)),
            max_duration_s=float(opts.get("max_duration_s", cls.max_duration_s)),
            walk_radius_max_m=float(opts.get("walk_radius_max_m", cls.walk_radius_max_m)),
            demand_seed=int(opts.get("demand_seed", cls.demand_seed)),
        )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    out = {"python": platform.python_version(), "bss-scenario": __version__}
    for pkg in ("numpy", "polars", "scipy", "networkx", "shapely", "pyyaml"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _input_files(config: BuildConfig) -> list[Path]:
    trips = sorted(config.trips_raw.glob("*-tripdata.csv")) if config.trips_raw.is_dir() else [config.trips_raw]
    return [*trips, config.boundary, config.bike_graphml, config.drive_graphml, config.gbfs_station_information]


def build_bundle(config: BuildConfig, out: str | Path, argv: list[str] | None = None) -> dict:
    """Build trips, networks and stations into ``out`` and write ``scenario.yaml`` and ``manifest.json``."""
    from bss_scenario.demand.build import DemandConfig, build_demand
    from bss_scenario.network import Projection, build_networks
    from bss_scenario.stations import build_stations, stations_report
    from bss_scenario.trips import CleaningConfig, clean_trips

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    inputs = _input_files(config)
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    trip_files = [p for p in inputs if p.suffix == ".csv"]
    clean_trips(trip_files, config.boundary, out / "trips",
                CleaningConfig(config.min_duration_s, config.max_duration_s, "touching_city"))
    net_report = build_networks(config.bike_graphml, config.drive_graphml, out / "network", config.walk_radius_max_m)
    stations = build_stations(sorted((out / "trips").glob("*.parquet")), config.boundary,
                              config.gbfs_station_information, out / "network" / "bike_nodes.parquet",
                              Projection(**net_report["projection"]))
    (out / "stations").mkdir(exist_ok=True)
    stations.write_parquet(out / "stations" / "stations.parquet")
    (out / "stations" / "stations_report.json").write_text(json.dumps(stations_report(stations), indent=2) + "\n")

    scenario = _scenario(config, net_report["projection"], ["network", "stations", "trips"])
    (out / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False))
    build_demand(out, DemandConfig(fit_period=tuple(dt.date.fromisoformat(d) for d in config.fit_period),
                                   seed=config.demand_seed))

    scenario = _scenario(config, net_report["projection"], sorted(COMPONENT_FILES))
    (out / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False))
    outputs = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
               if p.is_file() and p.name != "manifest.json"}
    manifest = {
        "bundle_id": config.id,
        "schema_version": SCHEMA_VERSION,
        "built_at": started.isoformat(timespec="seconds"),
        "command": argv if argv is not None else sys.argv,
        "versions": _versions(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in config.__dict__.items()},
        "inputs": {str(p): sha256(p) for p in inputs},
        "outputs": outputs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _scenario(config: BuildConfig, projection: dict, components: list[str]) -> dict:
    return {
        "id": config.id,
        "schema_version": SCHEMA_VERSION,
        "timezone": config.timezone,
        "slot_minutes": config.slot_minutes,
        "fit_period": list(config.fit_period),
        "components": components,
        "projection": projection,
    }


def rebuild_component(root: str | Path, component: str, argv: list[str] | None = None) -> dict:
    """Rebuild one component of an existing bundle (currently ``demand``) and refresh the metadata."""
    from bss_scenario.demand.build import DemandConfig, build_demand

    root = Path(root)
    bundle = load_bundle(root, skip=frozenset({component}))
    if component != "demand":
        raise ValueError(f"cannot rebuild {component!r} alone; run bss-scenario build")
    seed = bundle.manifest.get("config", {}).get("demand_seed", BuildConfig.demand_seed)
    build_demand(root, DemandConfig(fit_period=tuple(dt.date.fromisoformat(d) for d in bundle.scenario["fit_period"]),
                                    seed=seed))
    scenario = dict(bundle.scenario)
    scenario["components"] = sorted(set(scenario["components"]) | {component})
    (root / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False))
    manifest = dict(bundle.manifest)
    manifest.setdefault("rebuilt", []).append({
        "component": component, "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": argv if argv is not None else sys.argv, "versions": _versions()})
    manifest["outputs"] = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
                           if p.is_file() and p.name != "manifest.json"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class Bundle:
    root: Path
    scenario: dict
    manifest: dict

    def path(self, rel: str) -> Path:
        return self.root / rel


def load_bundle(root: str | Path, verify_hashes: bool = False, skip: frozenset[str] = frozenset()) -> Bundle:
    """Open a bundle: check ``schema_version``, required component files, file types and (optionally) hashes.

    Components in ``skip`` are not checked for completeness (used when rebuilding them).
    """
    root = Path(root)
    try:
        scenario = yaml.safe_load((root / "scenario.yaml").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
    except FileNotFoundError as exc:
        raise BundleError(f"{root} is not a scenario bundle: {exc.filename} missing") from exc
    version = scenario.get("schema_version")
    if version != SCHEMA_VERSION:
        raise BundleError(f"{root}: schema_version {version}, this bss-scenario reads {SCHEMA_VERSION}; "
                          "rebuild the bundle or use a matching version")
    for component in scenario.get("components", []):
        if component in skip:
            continue
        if component not in COMPONENT_FILES:
            raise BundleError(f"{root}: unknown component {component!r}")
        missing = [f for f in COMPONENT_FILES[component] if not (root / f).exists()]
        if missing:
            raise BundleError(f"{root}: component {component!r} incomplete, missing {missing}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix not in ALLOWED_SUFFIXES:
            raise BundleError(f"{root}: unexpected file type {p.relative_to(root)} (no pickles or other formats)")
    if verify_hashes:
        bad = [rel for rel, digest in manifest.get("outputs", {}).items()
               if not (root / rel).exists() or sha256(root / rel) != digest]
        if bad:
            raise BundleError(f"{root}: files changed or missing since build: {bad}")
    return Bundle(root=root, scenario=scenario, manifest=manifest)
