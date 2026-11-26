"""What the simulator reads from a scenario bundle (npz/json/npy only; spec §3, §4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bss_sim.demand import DemandModel

SCHEMA_VERSION = 1


class ScenarioError(ValueError):
    pass


@dataclass
class Scenario:
    root: Path
    bundle_id: str
    mode: str                     # dockless | docked (which spreader the demand uses)
    n_nodes: int
    walk_indptr: np.ndarray       # CSR walking neighbours, sorted by distance, self first
    walk_indices: np.ndarray
    walk_dist: np.ndarray         # metres
    walk_max_m: float
    bike_dist: np.ndarray         # float32 [N, N] metres (memory-mapped)
    station_node: np.ndarray      # int32 [S] in-city stations (place order)
    station_capacity: np.ndarray  # int32 [S]
    demand: DemandModel
    drive_time: np.ndarray        # float32 [M, M] free-flow seconds (memory-mapped)
    drive_dist: np.ndarray        # float32 [M, M] metres
    drive_lat: np.ndarray         # float64 [M]
    drive_lon: np.ndarray
    bike_to_drive: np.ndarray     # int32 [N] nearest drive node of each bike node
    bike_to_drive_m: np.ndarray   # float32 [N] distance to it
    road_classes: tuple[str, ...]

    @property
    def n_drive(self) -> int:
        return len(self.drive_lat)

    def drive_time_by_class(self) -> np.ndarray:
        """float32 [C, M, M], memory-mapped (only loaded when a congestion model is used)."""
        return np.load(self.root / "network" / "drive_time_by_class.npy", mmap_mode="r")

    def traffic_factors_path(self) -> Path | None:
        p = self.root / "traffic" / "factors.json"
        return p if p.exists() else None

    def stop_at(self, drive: int) -> int:
        """The bike node served by drive node ``drive``: among bike nodes mapped to it, the closest;
        if none, the bike node whose drive node is fastest to reach from it."""
        mine = np.flatnonzero(self.bike_to_drive == drive)
        if len(mine):
            return int(mine[np.argmin(self.bike_to_drive_m[mine])])
        return int(np.argmin(np.asarray(self.drive_time)[drive, self.bike_to_drive]))

    def nearest_drive_node(self, lat: float, lon: float) -> int:
        """Drive node closest to a coordinate (equirectangular distance; fine at city scale)."""
        k = np.cos(np.radians(lat))
        d2 = (self.drive_lat - lat) ** 2 + ((self.drive_lon - lon) * k) ** 2
        return int(np.argmin(d2))

    @classmethod
    def load(cls, root: str | Path, mode: str = "dockless") -> Scenario:
        root = Path(root)
        try:
            manifest = json.loads((root / "manifest.json").read_text())
        except FileNotFoundError as exc:
            raise ScenarioError(f"{root} is not a scenario bundle (no manifest.json)") from exc
        version = manifest.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ScenarioError(f"{root}: schema_version {version}, bss_sim reads {SCHEMA_VERSION}; rebuild the bundle")
        for rel in ("network/bike_walk_nbrs.npz", "network/bike_dist.npy", "stations/stations.npz",
                    "network/drive_time.npy", "network/drive_dist.npy", "network/drive_nodes.npz",
                    "network/drive_time_by_class.npy", "network/bike_to_drive.npz",
                    "demand/shape.npz", "demand/spread.npz", "demand/volume.json", "demand/ride_time.npz"):
            if not (root / rel).exists():
                raise ScenarioError(f"{root}: missing {rel}; rebuild the bundle with bss-scenario build")
        demand = DemandModel.load(root, spread_mode="kernel" if mode == "dockless" else "station")
        with np.load(root / "network" / "bike_walk_nbrs.npz") as z:
            indptr, indices, dist, rmax = z["indptr"], z["indices"], z["dist"].astype(np.float64), float(z["max_radius_m"])
        with np.load(root / "stations" / "stations.npz") as z:
            st_node, st_cap = z["bike_node"].astype(np.int32), z["capacity"].astype(np.int32)
        if len(st_node) != demand.n_places - 1:
            raise ScenarioError(f"{root}: {len(st_node)} stations but {demand.n_places - 1} in-city places")
        with np.load(root / "network" / "drive_nodes.npz") as z:
            d_lat, d_lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
            classes = tuple(str(c) for c in z["road_classes"])
        with np.load(root / "network" / "bike_to_drive.npz") as z:
            b2d, b2d_m = z["node"].astype(np.int32), z["dist_m"].astype(np.float32)
        return cls(root, manifest.get("bundle_id", root.name), mode, len(indptr) - 1, indptr, indices, dist, rmax,
                   demand.bike_dist, st_node, st_cap, demand,
                   np.load(root / "network" / "drive_time.npy", mmap_mode="r"),
                   np.load(root / "network" / "drive_dist.npy", mmap_mode="r"), d_lat, d_lon, b2d, b2d_m, classes)

    def departure_weights(self) -> np.ndarray:
        """Expected departures per node over a typical week (initial distribution)."""
        per_place = self.demand.rates.sum(axis=(0, 1, 3), dtype=np.float64)[:-1]
        return self.demand.spreader.node_weights(per_place, self.n_nodes)

    def node_capacity(self) -> np.ndarray:
        """Docks per bike node (0 where there is no station); stations sharing a node add up."""
        cap = np.zeros(self.n_nodes, dtype=np.int64)
        np.add.at(cap, self.station_node, self.station_capacity)
        return cap
