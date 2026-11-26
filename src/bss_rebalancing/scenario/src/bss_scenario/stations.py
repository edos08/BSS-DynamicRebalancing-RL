"""Stations table of the scenario bundle (issue #15, spec §4 and §7.1).

Built from the cleaned trips (what stations existed in the data period and how much they were
used), completed with dock capacities from a GBFS ``station_information`` snapshot, and mapped to
the nearest bike-network node.

* ``lat``/``lon``: the most frequent coordinate reported for the station in the trips (classic
  bikes report the station position; e-bikes report GPS fixes, which are scattered).
* ``capacity``: docks from GBFS, matched on the station code (GBFS ``short_name``). Stations
  missing from the snapshot (closed or renamed since) get the median capacity of matched
  in-city stations; ``capacity_source`` tells which.
* ``bike_node``: nearest bike-network node, only for stations inside the city (others are part
  of the ``EXTERNAL`` reservoir in the simulator).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl
import shapely
from scipy.spatial import cKDTree

from bss_scenario.network.graph import Projection
from bss_scenario.traffic.sample import haversine_m
from bss_scenario.trips.clean import load_boundary, read_trips

STATIONS_SCHEMA: dict[str, pl.DataType] = {
    "station_id": pl.String(),
    "name": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "in_city": pl.Boolean(),
    "departures": pl.Int64(),
    "arrivals": pl.Int64(),
    "first_seen": pl.Date(),
    "last_seen": pl.Date(),
    "capacity": pl.Int32(),
    "capacity_source": pl.String(),
    "gbfs_offset_m": pl.Float64(),
    "bike_node": pl.Int32(),
    "bike_node_dist_m": pl.Float64(),
}


def _ends(trips: pl.DataFrame) -> pl.DataFrame:
    cols = ["station_id", "name", "lat", "lon", "time", "kind"]
    start = trips.select(pl.col("start_station_id").alias("station_id"), pl.col("start_station_name").alias("name"),
                         pl.col("start_lat").alias("lat"), pl.col("start_lon").alias("lon"),
                         pl.col("start_time").alias("time"), pl.lit("dep").alias("kind"))
    end = trips.select(pl.col("end_station_id").alias("station_id"), pl.col("end_station_name").alias("name"),
                       pl.col("end_lat").alias("lat"), pl.col("end_lon").alias("lon"),
                       pl.col("end_time").alias("time"), pl.lit("arr").alias("kind"))
    return pl.concat([start, end]).select(cols).drop_nulls(["station_id", "lat", "lon"])


def stations_from_trips(trips: pl.DataFrame) -> pl.DataFrame:
    """One row per station seen in ``trips`` (valid trips only), with modal name and position."""
    ends = _ends(trips.filter("valid")).with_columns(pl.col("lat").round(6), pl.col("lon").round(6))
    position = (ends.group_by("station_id", "lat", "lon").len()
                .sort(["station_id", "len", "lat", "lon"], descending=[False, True, False, False])
                .unique("station_id", keep="first").drop("len"))
    name = (ends.drop_nulls("name").group_by("station_id", "name").len()
            .sort(["station_id", "len", "name"], descending=[False, True, False])
            .unique("station_id", keep="first").drop("len"))
    usage = ends.group_by("station_id").agg(
        (pl.col("kind") == "dep").sum().cast(pl.Int64).alias("departures"),
        (pl.col("kind") == "arr").sum().cast(pl.Int64).alias("arrivals"),
        pl.col("time").min().dt.date().alias("first_seen"),
        pl.col("time").max().dt.date().alias("last_seen"),
    )
    return position.join(name, on="station_id", how="left").join(usage, on="station_id").sort("station_id")


def read_gbfs_capacity(path: str | Path) -> pl.DataFrame:
    """GBFS ``station_information.json`` → short_name, capacity, gbfs_lat, gbfs_lon."""
    data = json.loads(Path(path).read_text())
    rows = [(s.get("short_name"), s.get("capacity"), s.get("lat"), s.get("lon")) for s in data["data"]["stations"]]
    return (pl.DataFrame(rows, schema={"station_id": pl.String, "capacity": pl.Int32, "gbfs_lat": pl.Float64,
                                       "gbfs_lon": pl.Float64}, orient="row")
            .drop_nulls("station_id").unique("station_id", keep="first"))


def build_stations(
    trip_files: Iterable[str | Path],
    boundary: str | Path,
    gbfs_station_information: str | Path,
    bike_nodes: str | Path,
    projection: Projection,
) -> pl.DataFrame:
    stations = stations_from_trips(read_trips(trip_files))
    geom = load_boundary(boundary)
    stations = stations.with_columns(
        pl.Series("in_city", shapely.contains_xy(geom, stations["lon"].to_numpy(), stations["lat"].to_numpy())))

    gbfs = read_gbfs_capacity(gbfs_station_information)
    snapshot = Path(gbfs_station_information).parent.name
    stations = stations.join(gbfs, on="station_id", how="left")
    matched_in_city = stations.filter(pl.col("in_city") & pl.col("capacity").is_not_null())["capacity"]
    fallback = int(matched_in_city.median()) if matched_in_city.len() else None
    stations = stations.with_columns(
        pl.when(pl.col("capacity").is_null()).then(pl.lit("median_in_city"))
        .otherwise(pl.lit(f"gbfs:{snapshot}")).alias("capacity_source"),
        pl.col("capacity").fill_null(fallback).cast(pl.Int32),
        pl.Series("gbfs_offset_m", haversine_m(stations["lat"].to_numpy(), stations["lon"].to_numpy(),
                                               stations["gbfs_lat"].fill_null(np.nan).to_numpy(),
                                               stations["gbfs_lon"].fill_null(np.nan).to_numpy())).fill_nan(None),
    ).drop("gbfs_lat", "gbfs_lon")

    nodes = pl.read_parquet(bike_nodes)
    tree = cKDTree(nodes.select("x", "y").to_numpy())
    x, y = projection.to_xy(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    dist, idx = tree.query(np.column_stack([x, y]), k=1)
    in_city = stations["in_city"].to_numpy()
    stations = stations.with_columns(
        pl.Series("bike_node", np.where(in_city, nodes["node_idx"].to_numpy()[idx], -1), dtype=pl.Int32)
        .replace(-1, None),
        pl.Series("bike_node_dist_m", np.where(in_city, dist, np.nan)).fill_nan(None),
    )
    return stations.select(list(STATIONS_SCHEMA)).cast(STATIONS_SCHEMA)


def write_stations(stations: pl.DataFrame, out_dir: str | Path) -> None:
    """Write ``stations.parquet`` and ``stations.npz``.

    The npz holds the **in-city** stations as plain arrays for the simulator, which never reads
    Parquet (spec §3). They are sorted by ``station_id``, so index ``i`` is demand place ``i``.
    Arrays: ``station_id`` (str), ``bike_node`` (int32), ``capacity`` (int32), ``lat``, ``lon`` (float64).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stations.write_parquet(out_dir / "stations.parquet")
    city = stations.filter("in_city").sort("station_id")
    np.savez(out_dir / "stations.npz",
             station_id=np.array(city["station_id"].to_list(), dtype=str),
             bike_node=city["bike_node"].to_numpy().astype(np.int32),
             capacity=city["capacity"].to_numpy().astype(np.int32),
             lat=city["lat"].to_numpy().astype(np.float64),
             lon=city["lon"].to_numpy().astype(np.float64))


def stations_report(stations: pl.DataFrame) -> dict:
    city = stations.filter("in_city")
    return {
        "stations": stations.height,
        "in_city": city.height,
        "in_city_capacity_from_gbfs": int((city["capacity_source"] != "median_in_city").sum()),
        "in_city_docks": int(city["capacity"].sum()),
        "in_city_gbfs_offset_m_p90": float(city["gbfs_offset_m"].drop_nulls().quantile(0.9) or 0.0),
        "bike_node_dist_m": {"median": float(city["bike_node_dist_m"].median()),
                             "max": float(city["bike_node_dist_m"].max())},
        "not_in_gbfs_in_city": city.filter(pl.col("capacity_source") == "median_in_city")["station_id"].to_list(),
    }
