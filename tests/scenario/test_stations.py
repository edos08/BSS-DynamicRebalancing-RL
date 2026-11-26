import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from raw_csv import trip, write  # noqa: E402

from bss_scenario import cli  # noqa: E402
from bss_scenario.network import Projection  # noqa: E402
from bss_scenario.stations import STATIONS_SCHEMA, build_stations, stations_from_trips  # noqa: E402
from bss_scenario.trips import clean_trips, read_trips  # noqa: E402

A = (42.3736, -71.1190)     # in city
B = (42.3800, -71.1100)     # in city
OUT = (42.3626, -71.0500)   # outside
PROJ = Projection(42.375, -71.115)


@pytest.fixture
def setup(tmp_path):
    ring = [[-71.13, 42.36], [-71.10, 42.36], [-71.10, 42.39], [-71.13, 42.39], [-71.13, 42.36]]
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}))
    rows = [
        trip("t1", "2024-07-01 08:00:00", "2024-07-01 08:15:00", s_id="A1", e_id="B1", s=A, e=B),
        trip("t2", "2024-07-02 08:00:00", "2024-07-02 08:15:00", s_id="A1", e_id="X9", s=A, e=OUT),
        trip("t3", "2024-07-03 08:00:00", "2024-07-03 08:15:00", s_id="B1", e_id="A1", s=B, e=A),
        # e-bike GPS fix far from the station position: must not win the modal position
        trip("t4", "2024-07-04 08:00:00", "2024-07-04 08:15:00", s_id="A1", e_id="B1", s=(42.3740, -71.1195), e=B,
             kind="electric_bike"),
        trip("t5", "2024-07-05 08:00:00", "2024-07-05 08:00:20", s_id="Z0", e_id="Z0", s=A, e=A),  # short: invalid
    ]
    trips_dir = tmp_path / "trips"
    clean_trips([write(tmp_path / "202407-bluebikes-tripdata.csv", rows)], boundary, trips_dir)
    gbfs_dir = tmp_path / "gbfs" / "2026-11-25"
    gbfs_dir.mkdir(parents=True)
    gbfs = gbfs_dir / "station_information.json"
    gbfs.write_text(json.dumps({"data": {"stations": [
        {"short_name": "A1", "capacity": 19, "lat": A[0], "lon": A[1] + 0.0001},
        {"short_name": "X9", "capacity": 11, "lat": OUT[0], "lon": OUT[1]},
    ]}}))
    net = tmp_path / "network"
    net.mkdir()
    lat = np.array([42.3736, 42.3801, 42.3650]); lon = np.array([-71.1191, -71.1100, -71.1200])
    x, y = PROJ.to_xy(lat, lon)
    pl.DataFrame({"node_idx": np.arange(3, dtype=np.int32), "osm_id": [1, 2, 3], "lat": lat, "lon": lon,
                  "x": x, "y": y}).write_parquet(net / "bike_nodes.parquet")
    (net / "network_report.json").write_text(json.dumps({"projection": {"lat0": PROJ.lat0, "lon0": PROJ.lon0}}))
    return tmp_path, trips_dir, boundary, gbfs, net


def test_stations_from_trips(setup):
    _, trips_dir, *_ = setup
    st = stations_from_trips(read_trips(trips_dir.glob("*.parquet")))
    assert st["station_id"].to_list() == ["A1", "B1", "X9"]  # Z0 only has an invalid trip
    a = st.row(0, named=True)
    assert (a["lat"], a["lon"]) == A and a["departures"] == 3 and a["arrivals"] == 1
    assert a["name"] == "Station A1"


def test_build_stations(setup):
    _, trips_dir, boundary, gbfs, net = setup
    st = build_stations(sorted(trips_dir.glob("*.parquet")), boundary, gbfs, net / "bike_nodes.parquet", PROJ)
    assert dict(st.schema) == STATIONS_SCHEMA
    rows = {r["station_id"]: r for r in st.iter_rows(named=True)}
    assert rows["A1"]["capacity"] == 19 and rows["A1"]["capacity_source"] == "gbfs:2026-11-25"
    assert rows["A1"]["gbfs_offset_m"] == pytest.approx(8.2, abs=0.5)
    assert rows["B1"]["capacity"] == 19 and rows["B1"]["capacity_source"] == "median_in_city"
    assert rows["A1"]["bike_node"] == 0 and rows["B1"]["bike_node"] == 1
    assert rows["X9"]["in_city"] is False and rows["X9"]["bike_node"] is None and rows["X9"]["capacity"] == 11


def test_cli(setup, capsys):
    tmp, trips_dir, boundary, gbfs, net = setup
    out = tmp / "bundle" / "stations" / "stations.parquet"
    assert cli.main(["stations", "--trips", str(trips_dir), "--boundary", str(boundary), "--gbfs", str(gbfs),
                     "--network", str(net), "--out", str(out)]) == 0
    assert pl.read_parquet(out).height == 3
    report = json.loads((out.parent / "stations_report.json").read_text())
    assert report["in_city"] == 2 and report["not_in_gbfs_in_city"] == ["B1"]
    with np.load(out.parent / "stations.npz") as z:  # in-city stations only, sorted, plain arrays
        assert z["station_id"].tolist() == ["A1", "B1"]
        assert z["bike_node"].tolist() == [0, 1] and z["bike_node"].dtype == np.int32
        assert z["capacity"].tolist() == [19, 19] and z["capacity"].dtype == np.int32
        assert z["lat"].dtype == np.float64 and not z["station_id"].dtype.hasobject
    with pytest.raises(SystemExit):
        cli.main(["stations", "--trips", str(trips_dir), "--boundary", str(boundary), "--gbfs", str(gbfs),
                  "--network", str(net), "--out", str(tmp / "x.parquet")])
