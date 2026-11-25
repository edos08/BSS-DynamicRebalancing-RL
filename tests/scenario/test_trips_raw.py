import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from raw_csv import trip, write  # noqa: E402

from bss_scenario.trips import NORMALISED_SCHEMA, read_raw_trips  # noqa: E402

UTC = dt.timezone.utc


def utc(*args):
    return dt.datetime(*args, tzinfo=UTC)


def read_one(tmp_path, rows, name="202407-bluebikes-tripdata.csv"):
    return read_raw_trips([write(tmp_path / name, rows)])


def test_schema_and_order(tmp_path):
    df, rep = read_one(tmp_path, [
        trip("b", "2024-07-11 09:00:00", "2024-07-11 09:20:00"),
        trip("a", "2024-07-11 08:00:00.250", "2024-07-11 08:10:00.500", kind="electric_bike", user="casual"),
    ])
    assert dict(df.schema) == NORMALISED_SCHEMA
    assert df["trip_id"].to_list() == ["a", "b"]  # sorted by start time
    assert df["bike_type"].to_list() == ["electric", "classic"]
    assert df["user_type"].to_list() == ["casual", "member"]
    assert rep.rows_read == rep.rows_out == 2 and rep.duplicates_dropped == 0


def test_mixed_precision_and_summer_offset(tmp_path):
    df, _ = read_one(tmp_path, [trip("a", "2024-07-11 08:00:00.250", "2024-07-11 08:10:00")])
    # EDT = UTC-4; milliseconds kept
    assert df["start_time"].dt.convert_time_zone("UTC").item() == utc(2024, 7, 11, 12, 0, 0, 250_000)
    assert df["end_time"].dt.convert_time_zone("UTC").item() == utc(2024, 7, 11, 12, 10)


def test_winter_offset(tmp_path):
    df, _ = read_one(tmp_path, [trip("a", "2024-01-15 08:00:00", "2024-01-15 08:10:00")],
                     name="202401-bluebikes-tripdata.csv")
    assert df["start_time"].dt.convert_time_zone("UTC").item() == utc(2024, 1, 15, 13, 0)  # EST = UTC-5


def test_dst_fall_back_end_in_repeated_hour(tmp_path):
    # 01:50 (first, EDT) -> 01:05 (second, EST): a 15 minute trip, not -45 minutes
    df, rep = read_one(tmp_path, [trip("a", "2024-11-03 01:50:00", "2024-11-03 01:05:00")],
                       name="202411-bluebikes-tripdata.csv")
    duration = (df["end_time"] - df["start_time"]).dt.total_seconds().item()
    assert duration == 15 * 60
    assert rep.dst_end_shifted == 1 and rep.dst_ambiguous_unresolved == 0


def test_dst_fall_back_unresolvable_is_counted(tmp_path):
    df, rep = read_one(tmp_path, [trip("a", "2024-11-03 01:10:00", "2024-11-03 01:20:00")],
                       name="202411-bluebikes-tripdata.csv")
    assert (df["end_time"] - df["start_time"]).dt.total_seconds().item() == 600
    assert rep.dst_end_shifted == 0 and rep.dst_ambiguous_unresolved == 1


def test_spring_forward_gap_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="spring-forward"):
        read_one(tmp_path, [trip("a", "2024-03-10 02:30:00", "2024-03-10 03:40:00")],
                 name="202403-bluebikes-tripdata.csv")


def test_month_boundary_duplicate_keeps_precise_copy_and_start_month(tmp_path):
    may = write(tmp_path / "202405-bluebikes-tripdata.csv",
                [trip("x", "2024-05-31 23:57:48", "2024-06-01 00:10:19")])
    june = write(tmp_path / "202406-bluebikes-tripdata.csv", [
        trip("x", "2024-05-31 23:57:48.942", "2024-06-01 00:10:19.183"),
        trip("y", "2024-06-02 10:00:00", "2024-06-02 10:05:00"),
    ])
    df, rep = read_raw_trips([june, may])
    assert df.height == 2 and rep.duplicates_dropped == 1
    x = df.filter(pl.col("trip_id") == "x")
    assert x["month"].item() == "2024-05"
    assert x["start_time"].dt.millisecond().item() == 942
    # the precise copy came from the June file, so its start month differs from its file
    assert rep.month_differs_from_file == 1


def test_missing_values_become_null(tmp_path):
    df, _ = read_one(tmp_path, [trip("a", "2024-07-11 08:00:00", "2024-07-12 09:00:00", e_id=None, e=None)])
    row = df.row(0, named=True)
    assert row["end_station_id"] is None and row["end_station_name"] is None
    assert row["end_lat"] is None and row["end_lon"] is None


def test_unknown_rideable_type_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="rideable_type"):
        read_one(tmp_path, [trip("a", "2024-07-11 08:00:00", "2024-07-11 08:10:00", kind="docked_bike")])


def test_wrong_schema_is_an_error(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text("tripduration,starttime\n1,2\n")
    with pytest.raises(ValueError, match="2024 BlueBikes schema"):
        read_raw_trips([p])
