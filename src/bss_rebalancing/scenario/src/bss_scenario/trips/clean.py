"""Clean normalised trips into the bundle's ``trips/YYYY-MM.parquet`` tables.

Cleaning **flags** problems instead of deleting rows, so every recorded trip stays traceable and
each consumer decides what to use. ``valid`` (no flag set) is what demand estimation uses.

Flags (profiled on 2024, see issue #13):

``flag_negative``  ends before it starts after DST handling (clock glitches of a few seconds)
``flag_short``     shorter than ``min_duration_s`` (false starts, re-docking)
``flag_long``      longer than ``max_duration_s`` (lost bikes, very long rentals)
``flag_no_end``    no end coordinates (bike not returned; closed by the operator after ~25 h)

City boundary: ``start_in_city`` / ``end_in_city`` tell whether each end lies inside the city
polygon (``end_in_city`` is null without end coordinates). A trip with one end outside is an
``EXTERNAL`` trip for the simulator (spec §6.5). By default only trips touching the city are kept.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import shapely

from bss_scenario import __version__
from bss_scenario.trips.raw import NORMALISED_SCHEMA, read_raw_trips

#: Version of the trips table layout; bump on any change to :data:`TRIPS_SCHEMA`.
TRIPS_SCHEMA_VERSION = 1

FLAGS = ("flag_negative", "flag_short", "flag_long", "flag_no_end")

#: Columns and dtypes of ``trips/YYYY-MM.parquet``.
TRIPS_SCHEMA: dict[str, pl.DataType] = {
    **NORMALISED_SCHEMA,
    "duration_s": pl.Float64(),
    "start_in_city": pl.Boolean(),
    "end_in_city": pl.Boolean(),
    **{f: pl.Boolean() for f in FLAGS},
    "valid": pl.Boolean(),
}

REPORT_NAME = "cleaning_report.json"


@dataclass(frozen=True)
class CleaningConfig:
    min_duration_s: float = 60.0
    max_duration_s: float = 3 * 3600.0
    keep: Literal["touching_city", "all"] = "touching_city"

    def __post_init__(self) -> None:
        if not 0 <= self.min_duration_s < self.max_duration_s:
            raise ValueError("need 0 <= min_duration_s < max_duration_s")
        if self.keep not in ("touching_city", "all"):
            raise ValueError(f"keep must be 'touching_city' or 'all', got {self.keep!r}")


def load_boundary(path: str | Path) -> shapely.Geometry:
    """Load a (Multi)Polygon from a GeoJSON file (FeatureCollection, Feature or Geometry), WGS84."""
    data = json.loads(Path(path).read_text())
    if data.get("type") == "FeatureCollection":
        geoms = [shapely.geometry.shape(f["geometry"]) for f in data["features"]]
    elif data.get("type") == "Feature":
        geoms = [shapely.geometry.shape(data["geometry"])]
    else:
        geoms = [shapely.geometry.shape(data)]
    geom = shapely.union_all(geoms)
    if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty:
        raise ValueError(f"{path}: boundary must be a non-empty (Multi)Polygon, got {geom.geom_type}")
    shapely.prepare(geom)
    return geom


def _inside(boundary: shapely.Geometry, lat: pl.Series, lon: pl.Series) -> pl.Series:
    lat_np = lat.to_numpy(allow_copy=True).astype(float)
    lon_np = lon.to_numpy(allow_copy=True).astype(float)
    missing = np.isnan(lat_np) | np.isnan(lon_np)
    inside = shapely.contains_xy(boundary, np.nan_to_num(lon_np), np.nan_to_num(lat_np))
    return (
        pl.DataFrame({"inside": inside, "missing": missing})
        .select(pl.when(pl.col("missing")).then(None).otherwise(pl.col("inside")))
        .to_series()
    )


def flag_trips(df: pl.DataFrame, boundary: shapely.Geometry, config: CleaningConfig) -> pl.DataFrame:
    """Add duration, city-boundary and quality-flag columns to a normalised trips table."""
    df = df.with_columns(
        (pl.col("end_time") - pl.col("start_time")).dt.total_milliseconds().cast(pl.Float64).truediv(1000)
        .alias("duration_s"),
        _inside(boundary, df["start_lat"], df["start_lon"]).fill_null(False).alias("start_in_city"),
        _inside(boundary, df["end_lat"], df["end_lon"]).alias("end_in_city"),
    )
    no_end = pl.col("end_lat").is_null() | pl.col("end_lon").is_null()
    df = df.with_columns(
        (pl.col("duration_s") < 0).alias("flag_negative"),
        pl.col("duration_s").is_between(0, config.min_duration_s, closed="left").alias("flag_short"),
        (pl.col("duration_s") > config.max_duration_s).alias("flag_long"),
        no_end.alias("flag_no_end"),
    )
    df = df.with_columns((~pl.any_horizontal(FLAGS)).alias("valid"))
    return df.select(list(TRIPS_SCHEMA))


def _touching() -> pl.Expr:
    return pl.col("start_in_city") | pl.col("end_in_city").fill_null(False)


def _month_stats(df: pl.DataFrame) -> list[dict]:
    start_in = pl.col("start_in_city")
    end_in = pl.col("end_in_city").fill_null(False)
    end_out = ~pl.col("end_in_city").fill_null(True)
    return (
        df.group_by("month")
        .agg(
            pl.len().alias("trips"),
            pl.col("valid").sum().alias("valid"),
            *[pl.col(f).sum().alias(f) for f in FLAGS],
            (start_in & end_in).sum().alias("both_in_city"),
            (start_in & end_out).sum().alias("start_only_in_city"),
            (~start_in & end_in).sum().alias("end_only_in_city"),
            pl.col("end_in_city").is_null().sum().alias("end_unknown"),
            (pl.col("bike_type") == "electric").sum().alias("electric"),
        )
        .sort("month")
        .to_dicts()
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_trips(
    raw_paths: Iterable[str | Path],
    boundary_path: str | Path,
    out_dir: str | Path,
    config: CleaningConfig = CleaningConfig(),
) -> dict:
    """Read raw files, flag, keep trips per ``config.keep``, write one Parquet per month and a report.

    Returns the report (also written to ``out_dir/cleaning_report.json``).
    """
    raw_paths = sorted(Path(p) for p in raw_paths)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trips, read_report = read_raw_trips(raw_paths)
    flagged = flag_trips(trips, load_boundary(boundary_path), config)
    kept = flagged.filter(_touching()) if config.keep == "touching_city" else flagged

    outputs = {}
    for (month,), part in kept.partition_by("month", as_dict=True, maintain_order=True).items():
        path = out_dir / f"{month}.parquet"
        part.write_parquet(path, compression="zstd", statistics=True)
        outputs[path.name] = {"rows": part.height, "sha256": _sha256(path)}

    report = {
        "schema_version": TRIPS_SCHEMA_VERSION,
        "bss_scenario_version": __version__,
        "config": asdict(config),
        "inputs": {
            "raw": [{"file": p.name, "sha256": _sha256(p)} for p in raw_paths],
            "boundary": {"file": Path(boundary_path).name, "sha256": _sha256(Path(boundary_path))},
        },
        "read": {k: v for k, v in read_report.as_dict().items() if k != "files"},
        "all_trips": {"rows": flagged.height, "touching_city": int(flagged.select(_touching().sum()).item())},
        "kept_by_month": _month_stats(kept),
        "outputs": outputs,
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    return report


def read_trips(paths: Iterable[str | Path], *, valid_only: bool = False) -> pl.DataFrame:
    """Read cleaned monthly trip tables, checking their columns against :data:`TRIPS_SCHEMA`."""
    frames = []
    for p in sorted(Path(x) for x in paths):
        df = pl.read_parquet(p)
        if dict(df.schema) != TRIPS_SCHEMA:
            raise ValueError(f"{p}: columns do not match trips schema v{TRIPS_SCHEMA_VERSION}")
        frames.append(df)
    df = pl.concat(frames) if frames else pl.DataFrame(schema=TRIPS_SCHEMA)
    return df.filter("valid") if valid_only else df
