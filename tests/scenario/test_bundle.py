import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from raw_csv import trip, write  # noqa: E402
from test_network_build import street_grid  # noqa: E402

from bss_scenario import cli  # noqa: E402
from bss_scenario.bundle import SCHEMA_VERSION, BuildConfig, BundleError, build_bundle, load_bundle  # noqa: E402

IN_A, IN_B, OUT = (42.371, -71.119), (42.375, -71.115), (42.3626, -71.05)


@pytest.fixture
def config_path(tmp_path):
    raw = tmp_path / "raw"
    (raw / "trips").mkdir(parents=True)
    write(raw / "trips" / "202407-bluebikes-tripdata.csv", [
        trip("t1", "2024-07-01 08:00:00", "2024-07-01 08:15:00", s_id="A1", e_id="B1", s=IN_A, e=IN_B),
        trip("t2", "2024-07-01 09:00:00", "2024-07-01 09:15:00", s_id="B1", e_id="X9", s=IN_B, e=OUT),
    ])
    ring = [[-71.13, 42.36], [-71.10, 42.36], [-71.10, 42.39], [-71.13, 42.39], [-71.13, 42.36]]
    (raw / "boundary.geojson").write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}))
    street_grid(raw / "bike.graphml", n=6, step=0.001)
    street_grid(raw / "drive.graphml", n=4, step=0.002)
    gbfs = raw / "gbfs" / "2026-11-25"
    gbfs.mkdir(parents=True)
    (gbfs / "station_information.json").write_text(json.dumps({"data": {"stations": [
        {"short_name": "A1", "capacity": 15, "lat": IN_A[0], "lon": IN_A[1]}]}}))
    cfg = tmp_path / "configs" / "test.yaml"
    cfg.parent.mkdir()
    cfg.write_text(yaml.safe_dump({
        "id": "test-2024", "fit_period": ["2024-07-01", "2024-07-31"],
        "inputs": {"trips_raw": "../raw/trips", "boundary": "../raw/boundary.geojson",
                   "bike_graphml": "../raw/bike.graphml", "drive_graphml": "../raw/drive.graphml",
                   "gbfs_station_information": "../raw/gbfs/2026-11-25/station_information.json"},
        "options": {"walk_radius_max_m": 300}}))
    return cfg


def test_config_paths_relative_to_yaml(config_path):
    cfg = BuildConfig.from_yaml(config_path)
    assert cfg.boundary == (config_path.parent.parent / "raw" / "boundary.geojson").resolve()
    assert cfg.walk_radius_max_m == 300 and cfg.fit_period == ("2024-07-01", "2024-07-31")


def test_config_requires_inputs(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("id: x\ninputs: {boundary: b.geojson}\n")
    with pytest.raises(ValueError, match="missing"):
        BuildConfig.from_yaml(p)


def test_build_and_load(config_path, tmp_path):
    out = tmp_path / "bundle"
    manifest = build_bundle(BuildConfig.from_yaml(config_path), out, argv=["test"])
    b = load_bundle(out, verify_hashes=True)
    assert b.scenario["id"] == "test-2024" and b.scenario["schema_version"] == SCHEMA_VERSION
    assert b.scenario["components"] == ["network", "stations", "trips"]
    assert "trips/2024-07.parquet" in manifest["outputs"] and "network/bike_dist.npy" in manifest["outputs"]
    assert len(manifest["inputs"]) == 5 and manifest["command"] == ["test"]
    assert manifest["versions"]["bss-scenario"]
    assert np.load(b.path("network/drive_time.npy")).shape == (16, 16)


def test_load_rejects_bad_bundles(config_path, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(BuildConfig.from_yaml(config_path), out, argv=["test"])

    (out / "stations" / "stations.parquet").write_bytes(b"tampered")
    with pytest.raises(BundleError, match="changed or missing"):
        load_bundle(out, verify_hashes=True)
    load_bundle(out)  # hashes not checked

    (out / "network" / "model.pkl").write_bytes(b"")
    with pytest.raises(BundleError, match="unexpected file type"):
        load_bundle(out)
    (out / "network" / "model.pkl").unlink()

    (out / "network" / "drive_time.npy").unlink()
    with pytest.raises(BundleError, match="incomplete"):
        load_bundle(out)

    scen = yaml.safe_load((out / "scenario.yaml").read_text())
    scen["schema_version"] = 99
    (out / "scenario.yaml").write_text(yaml.safe_dump(scen))
    with pytest.raises(BundleError, match="schema_version 99"):
        load_bundle(out)
    with pytest.raises(BundleError, match="not a scenario bundle"):
        load_bundle(tmp_path)


def test_cli_build_and_check(config_path, tmp_path, capsys):
    out = tmp_path / "b"
    assert cli.main(["build", str(config_path), "--out", str(out)]) == 0
    assert "bundle 'test-2024' written" in capsys.readouterr().out
    assert cli.main(["check", str(out)]) == 0
    (out / "scenario.yaml").write_text("schema_version: 0\n")
    assert cli.main(["check", str(out)]) == 1
