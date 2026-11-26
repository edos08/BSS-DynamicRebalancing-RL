import polars as pl
import pytest

from bss_scenario.traffic.freeflow import MPH, edge_speeds, parse_maxspeed_kmh, speed_report


@pytest.mark.parametrize("value,expected", [
    ("25 mph", 25 * MPH), ("40 km/h", 40.0), ("40 kph", 40.0), ("25", 25 * MPH),
    ("25 mph;20 mph", 22.5 * MPH), (" 30 MPH ", 30 * MPH), ("signals", None), ("none", None), (None, None),
    ("0", None), ("20 mph;walk", 20 * MPH),
])
def test_parse_maxspeed(value, expected):
    got = parse_maxspeed_kmh(value)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_bare_unit_kmh():
    assert parse_maxspeed_kmh("50", bare_unit="km/h") == 50.0


def test_edge_speeds_and_report():
    edges = pl.DataFrame({"length_m": [100.0, 200.0, 50.0], "highway": ["residential", "trunk", "mystery"],
                          "maxspeed": ["20 mph", None, None]})
    out = edge_speeds(edges)
    assert out["speed_source"].to_list() == ["maxspeed", "default", "default"]
    assert out["speed_kmh"].to_list() == pytest.approx([20 * MPH, 35 * MPH, 20 * MPH])
    assert out["freeflow_s"][0] == pytest.approx(100 / (20 * MPH / 3.6))
    rep = speed_report(out)
    assert rep["explicit_share_edges"] == pytest.approx(1 / 3)
    assert rep["explicit_share_length"] == pytest.approx(100 / 350)
