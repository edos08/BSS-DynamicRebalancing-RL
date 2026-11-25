import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from raw_csv import trip, write  # noqa: E402

from bss_scenario import cli  # noqa: E402
from bss_scenario.trips import (  # noqa: E402
    TRIPS_SCHEMA,
    CleaningConfig,
    clean_trips,
    flag_trips,
    load_boundary,
    read_raw_trips,
    read_trips,
)

IN_A = (42.3736, -71.1190)    # inside the test city
IN_B = (42.3800, -71.1100)    # inside
OUT = (42.3626, -71.0500)     # outside (east)
OUT2 = (42.3500, -71.0600)    # outside


@pytest.fixture
def boundary_path(tmp_path):
    ring = [[-71.13, 42.36], [-71.10, 42.36], [-71.10, 42.39], [-71.13, 42.39], [-71.13, 42.36]]
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    p = tmp_path / "boundary.geojson"
    p.write_text(json.dumps(fc))
    return p


ROWS = [
    trip("ok", "2024-07-01 08:00:00", "2024-07-01 08:15:00", s=IN_A, e=IN_B),
    trip("short", "2024-07-01 09:00:00", "2024-07-01 09:00:30", s=IN_A, e=IN_A),
    trip("long", "2024-07-01 10:00:00", "2024-07-01 14:00:00", s=IN_A, e=IN_B),
    trip("neg", "2024-07-01 11:00:05", "2024-07-01 11:00:00", s=IN_A, e=IN_B),
    trip("noend", "2024-07-01 12:00:00", "2024-07-01 13:00:00", s=IN_A, e=None, e_id=None),
    trip("leaves", "2024-07-01 13:00:00", "2024-07-01 13:20:00", s=IN_A, e=OUT),
    trip("arrives", "2024-07-01 14:00:00", "2024-07-01 14:20:00", s=OUT, e=IN_B),
    trip("outside", "2024-07-01 15:00:00", "2024-07-01 15:20:00", s=OUT, e=OUT2),
    trip("aug", "2024-08-01 08:00:00", "2024-08-01 08:10:00", s=IN_A, e=IN_B, kind="electric_bike"),
]


def flagged(tmp_path, boundary_path, config=CleaningConfig()):
    df, _ = read_raw_trips([write(tmp_path / "202407-bluebikes-tripdata.csv", ROWS)])
    return flag_trips(df, load_boundary(boundary_path), config)


def by_id(df):
    return {r["trip_id"]: r for r in df.iter_rows(named=True)}


def test_flags(tmp_path, boundary_path):
    rows = by_id(flagged(tmp_path, boundary_path))
    assert rows["ok"]["valid"] and rows["leaves"]["valid"] and rows["arrives"]["valid"]
    expected = {"short": "flag_short", "long": "flag_long", "neg": "flag_negative", "noend": "flag_no_end"}
    for tid, flag in expected.items():
        r = rows[tid]
        assert r[flag] and not r["valid"], tid
        assert sum(r[f] for f in expected.values()) == 1, f"{tid} has more than one flag"
    assert rows["ok"]["duration_s"] == 900.0
    assert rows["neg"]["duration_s"] == -5.0


def test_city_boundary(tmp_path, boundary_path):
    rows = by_id(flagged(tmp_path, boundary_path))
    assert (rows["ok"]["start_in_city"], rows["ok"]["end_in_city"]) == (True, True)
    assert (rows["leaves"]["start_in_city"], rows["leaves"]["end_in_city"]) == (True, False)
    assert (rows["arrives"]["start_in_city"], rows["arrives"]["end_in_city"]) == (False, True)
    assert (rows["outside"]["start_in_city"], rows["outside"]["end_in_city"]) == (False, False)
    assert rows["noend"]["end_in_city"] is None


def test_thresholds_are_configurable(tmp_path, boundary_path):
    rows = by_id(flagged(tmp_path, boundary_path, CleaningConfig(min_duration_s=10, max_duration_s=5 * 3600)))
    assert rows["short"]["valid"] and rows["long"]["valid"]


@pytest.mark.parametrize("kwargs", [dict(min_duration_s=-1), dict(min_duration_s=100, max_duration_s=50),
                                    dict(keep="nothing")])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        CleaningConfig(**kwargs)


def test_clean_trips_writes_months_and_report(tmp_path, boundary_path):
    raw = write(tmp_path / "202407-bluebikes-tripdata.csv", ROWS)
    out = tmp_path / "bundle" / "trips"
    report = clean_trips([raw], boundary_path, out)

    assert sorted(p.name for p in out.iterdir()) == ["2024-07.parquet", "2024-08.parquet", "cleaning_report.json"]
    df = read_trips(out.glob("*.parquet"))
    assert dict(df.schema) == TRIPS_SCHEMA
    assert "outside" not in df["trip_id"].to_list()          # touching_city drops it
    assert df.height == len(ROWS) - 1
    assert read_trips(out.glob("*.parquet"), valid_only=True)["trip_id"].sort().to_list() == \
        ["arrives", "aug", "leaves", "ok"]

    july = report["kept_by_month"][0]
    assert july["month"] == "2024-07" and july["trips"] == 7 and july["valid"] == 3
    assert (july["both_in_city"], july["start_only_in_city"], july["end_only_in_city"], july["end_unknown"]) == (4, 1, 1, 1)
    assert report["all_trips"] == {"rows": 9, "touching_city": 8}
    assert report["outputs"]["2024-08.parquet"]["rows"] == 1
    assert json.loads((out / "cleaning_report.json").read_text()) == report


def test_keep_all(tmp_path, boundary_path):
    raw = write(tmp_path / "202407-bluebikes-tripdata.csv", ROWS)
    report = clean_trips([raw], boundary_path, tmp_path / "out", CleaningConfig(keep="all"))
    assert sum(m["trips"] for m in report["kept_by_month"]) == len(ROWS)


def test_output_is_deterministic(tmp_path, boundary_path):
    raw = write(tmp_path / "202407-bluebikes-tripdata.csv", ROWS)
    a = clean_trips([raw], boundary_path, tmp_path / "a")
    b = clean_trips([raw], boundary_path, tmp_path / "b")
    assert a["outputs"] == b["outputs"]  # same rows and sha256


def test_read_trips_rejects_other_schema(tmp_path):
    p = tmp_path / "x.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(p)
    with pytest.raises(ValueError, match="trips schema"):
        read_trips([p])


def test_cli(tmp_path, boundary_path, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write(raw_dir / "202407-bluebikes-tripdata.csv", ROWS)
    out = tmp_path / "trips"
    assert cli.main(["trips", "--raw", str(raw_dir), "--boundary", str(boundary_path), "--out", str(out)]) == 0
    assert "9 trips read, 8 kept, 4 valid" in capsys.readouterr().out
    assert (out / "2024-07.parquet").exists()
    assert cli.main(["trips", "--raw", str(tmp_path / "empty_dir_missing"), "--boundary", str(boundary_path),
                     "--out", str(out)]) == 2
