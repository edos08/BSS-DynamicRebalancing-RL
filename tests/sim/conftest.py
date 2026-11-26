"""Shared fixtures for the simulator tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scenario"))


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """Small bundle built with bss_scenario from synthetic trips."""
    import json

    import yaml
    from raw_csv import trip, write
    from test_network_build import street_grid

    from bss_scenario.bundle import BuildConfig, build_bundle

    root = tmp_path_factory.mktemp("b")
    raw = root / "raw"
    (raw / "trips").mkdir(parents=True)
    rng = np.random.default_rng(0)
    A, B, C = (42.3705, -71.1195), (42.3745, -71.1155), (42.3725, -71.1175)
    OUT = (42.3626, -71.05)
    rows, k = [], 0
    for day in range(1, 31):
        for _ in range(60):
            h = int(rng.choice([8, 9, 17, 18]))
            s, e, sid, eid = [(A, B, "A1", "B1"), (B, A, "B1", "A1"), (C, OUT, "C1", "X9"), (OUT, C, "X9", "C1")][k % 4]
            k += 1
            m = int(rng.integers(0, 50))
            rows.append(trip(f"t{k}", f"2024-09-{day:02d} {h:02d}:{m:02d}:00", f"2024-09-{day:02d} {h:02d}:{m + 8:02d}:00",
                             s_id=sid, e_id=eid, s=s, e=e, kind="electric_bike" if k % 5 == 0 else "classic_bike"))
    write(raw / "trips" / "202409-bluebikes-tripdata.csv", rows)
    ring = [[-71.13, 42.36], [-71.10, 42.36], [-71.10, 42.39], [-71.13, 42.39], [-71.13, 42.36]]
    (raw / "boundary.geojson").write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}))
    street_grid(raw / "bike.graphml", n=8, step=0.001)
    street_grid(raw / "drive.graphml", n=4, step=0.002)
    gb = raw / "gbfs" / "2026-11-25"
    gb.mkdir(parents=True)
    (gb / "station_information.json").write_text(json.dumps({"data": {"stations": [
        {"short_name": s, "capacity": 15, "lat": 0, "lon": 0} for s in ("A1", "B1", "C1")]}}))
    cfg = root / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"id": "t", "fit_period": ["2024-09-01", "2024-09-30"], "inputs": {
        "trips_raw": "raw/trips", "boundary": "raw/boundary.geojson", "bike_graphml": "raw/bike.graphml",
        "drive_graphml": "raw/drive.graphml", "gbfs_station_information": "raw/gbfs/2026-11-25/station_information.json"},
        "options": {"walk_radius_max_m": 400}}))
    out = root / "bundle"
    build_bundle(BuildConfig.from_yaml(cfg), out, argv=["test"])
    return out


@pytest.fixture(scope="session")
def model(bundle):
    from bss_sim.demand import DemandModel
    return DemandModel.load(bundle)
