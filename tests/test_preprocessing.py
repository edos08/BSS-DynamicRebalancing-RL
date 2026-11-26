import json

import polars as pl
import pytest

from preprocessing.core.converter import (
    TripDataConverter, encode_station_id_alnum, encode_station_id_numeric,
)
from preprocessing.core.sources import get_source, load_sources
from preprocessing.config import DEFAULT_SOURCES_JSON


def test_bundled_sources():
    sources = load_sources(DEFAULT_SOURCES_JSON)
    assert {"bluebikes", "citibike"} <= set(sources)
    src = get_source(DEFAULT_SOURCES_JSON, "bluebikes")
    assert src.url_for(2022, 9).endswith("202209-bluebikes-tripdata.zip")
    assert src.month_str == "09-10"
    with pytest.raises(KeyError):
        get_source(DEFAULT_SOURCES_JSON, "nope")


def test_station_id_encoding():
    assert encode_station_id_numeric("74.0") == 7400
    assert encode_station_id_alnum("HB101") == int("2" + "0802" + "101" + "00")
    assert encode_station_id_alnum("6140.05") == int("1" + "6140" + "05")
    with pytest.raises(ValueError):
        encode_station_id_alnum("???")


def test_converter_citibike_like(tmp_path):
    src = get_source(DEFAULT_SOURCES_JSON, "citibike")
    raw = tmp_path / "raw.csv"
    pl.DataFrame({
        "ride_id": ["a", "b", "c"], "rideable_type": ["x"] * 3, "member_casual": ["m"] * 3,
        "started_at": ["2024-09-01 08:00:00.123", "2024-09-01 09:00:00", "2024-09-02 10:00:00"],
        "ended_at": ["2024-09-01 08:10:00.000", "2024-09-01 09:20:00", "2024-09-02 10:05:00"],
        "start_station_name": ["A", "B", "C"], "start_station_id": ["6140.05", "HB101", "6140.05"],
        "end_station_name": ["B", "C", "A"], "end_station_id": ["HB101", "6140.05", None],
        "start_lat": [40.75] * 3, "start_lng": [-73.98] * 3, "end_lat": [40.75] * 3, "end_lng": [-73.98] * 3,
    }).write_csv(raw)
    out = tmp_path / "trips.csv"
    TripDataConverter(src).convert(str(tmp_path), str(out), cleanup=True)

    df = pl.read_csv(out)
    assert df.height == 2                       # row with null end station dropped
    assert set(df.columns) >= {"starttime", "stoptime", "start station id", "end station id"}
    assert df["start station id"].dtype == pl.Int64
    assert not raw.exists()                     # raw CSV cleaned up, converted CSV kept


def test_cli_cell_size_override(monkeypatch, tmp_path):
    from preprocessing import cli
    seen = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda cfg, src, steps, verbose=False: seen.update(cell=cfg.cell_size))
    monkeypatch.setattr("sys.argv", ["bss-preprocess", "--source", "bluebikes", "--data-path", str(tmp_path)])
    cli.main()
    assert seen["cell"] == 300
    monkeypatch.setattr("sys.argv", ["bss-preprocess", "--source", "bluebikes", "--data-path", str(tmp_path),
                                     "--cell-size", "500"])
    cli.main()
    assert seen["cell"] == 500
