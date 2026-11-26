"""Read raw BlueBikes trip files (2024 schema) into a normalised table.

The raw monthly CSVs published by BlueBikes (2023 onwards) have the columns::

    ride_id, rideable_type, started_at, ended_at,
    start_station_name, start_station_id, end_station_name, end_station_id,
    start_lat, start_lng, end_lat, end_lng, member_casual

Quirks handled here (profiled on 2024, see issue #13):

* Timestamps are **naive local time** (America/New_York), with either second or millisecond
  precision. They are localised; a time that does not exist (spring-forward gap) is an error.
* Times in the repeated hour of the autumn DST change are ambiguous. Both ends are read as the
  first occurrence, except when that would make the trip end before it starts: then the end is
  read as the second occurrence (+1 h). Other ambiguous cases cannot be resolved from the data
  and are counted in the report.
* Trips crossing a month boundary appear in both monthly files (one copy with millisecond
  precision). One copy is kept, preferring the more precise one.
* The month a trip belongs to is its local start month, not the file it came from.

Quality flags and the city boundary are applied later (``bss_scenario.trips.clean``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

TIMEZONE = "America/New_York"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%.f"

RAW_COLUMNS = (
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
)

BIKE_TYPES = {"classic_bike": "classic", "electric_bike": "electric"}
USER_TYPES = {"member": "member", "casual": "casual"}
BikeType = pl.Enum(["classic", "electric"])
UserType = pl.Enum(["member", "casual"])

#: Columns and dtypes of the table returned by :func:`read_raw_trips`.
NORMALISED_SCHEMA: dict[str, pl.DataType] = {
    "trip_id": pl.String(),
    "month": pl.String(),
    "bike_type": BikeType,
    "user_type": UserType,
    "start_time": pl.Datetime("ms", TIMEZONE),
    "end_time": pl.Datetime("ms", TIMEZONE),
    "start_station_id": pl.String(),
    "start_station_name": pl.String(),
    "end_station_id": pl.String(),
    "end_station_name": pl.String(),
    "start_lat": pl.Float64(),
    "start_lon": pl.Float64(),
    "end_lat": pl.Float64(),
    "end_lon": pl.Float64(),
}


@dataclass
class ReadReport:
    """What the reader changed or could not resolve. Written into the cleaning report."""

    files: list[str] = field(default_factory=list)
    rows_read: int = 0
    duplicates_dropped: int = 0
    dst_end_shifted: int = 0
    dst_ambiguous_unresolved: int = 0
    month_differs_from_file: int = 0
    rows_out: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _read_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema=False)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: not the 2024 BlueBikes schema, missing columns {missing}")
    # Empty strings (quoted "") and nulls both mean "missing".
    return df.select(
        [pl.col(c).str.strip_chars().replace("", None).alias(c) for c in RAW_COLUMNS]
    ).with_columns(pl.lit(path.name).alias("_file"))


def _check_categories(df: pl.DataFrame, column: str, allowed: dict[str, str]) -> None:
    unknown = set(df[column].drop_nulls().unique().to_list()) - set(allowed)
    if unknown or df[column].null_count():
        raise ValueError(f"unexpected values in {column}: {sorted(unknown) or 'null'}")


def _localise(col: str, ambiguous: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_datetime(TIMESTAMP_FORMAT, time_unit="ms")
        .dt.replace_time_zone(TIMEZONE, ambiguous=ambiguous, non_existent="raise")
    )


def read_raw_trips(paths: Iterable[str | Path]) -> tuple[pl.DataFrame, ReadReport]:
    """Read and normalise raw monthly trip CSVs.

    Returns the trips sorted by ``start_time`` (ties by ``trip_id``) with the columns of
    :data:`NORMALISED_SCHEMA`, and a :class:`ReadReport`.
    """
    files = sorted(Path(p) for p in paths)
    if not files:
        raise ValueError("no input files")
    report = ReadReport(files=[f.name for f in files])
    df = pl.concat([_read_csv(f) for f in files], how="vertical")
    report.rows_read = df.height

    _check_categories(df, "rideable_type", BIKE_TYPES)
    _check_categories(df, "member_casual", USER_TYPES)

    # Duplicates: keep the most precise copy (millisecond timestamps are longer strings).
    df = (
        df.with_columns(pl.col("started_at").str.len_chars().alias("_precision"))
        .sort(["ride_id", "_precision", "_file"], descending=[False, True, False])
        .unique(subset="ride_id", keep="first", maintain_order=True)
    )
    report.duplicates_dropped = report.rows_read - df.height

    try:
        df = df.with_columns(
            _localise("started_at", "earliest").alias("start_time"),
            _localise("ended_at", "earliest").alias("_end_early"),
            _localise("ended_at", "latest").alias("_end_late"),
            _localise("started_at", "latest").alias("_start_late"),
        )
    except pl.exceptions.ComputeError as exc:  # non-existent local time
        raise ValueError(f"timestamp in the spring-forward gap: {exc}") from exc

    end_ambiguous = pl.col("_end_early") != pl.col("_end_late")
    start_ambiguous = pl.col("start_time") != pl.col("_start_late")
    shift_end = end_ambiguous & (pl.col("_end_early") < pl.col("start_time"))
    df = df.with_columns(
        pl.when(shift_end).then(pl.col("_end_late")).otherwise(pl.col("_end_early")).alias("end_time"),
        shift_end.alias("_shifted"),
        ((start_ambiguous | end_ambiguous) & ~shift_end).alias("_unresolved"),
    )
    report.dst_end_shifted = int(df["_shifted"].sum())
    report.dst_ambiguous_unresolved = int(df["_unresolved"].sum())

    df = df.with_columns(pl.col("start_time").dt.strftime("%Y-%m").alias("month"))
    # Files are named YYYYMM-bluebikes-tripdata.csv; other names are not compared.
    file_month = pl.col("_file").str.extract(r"^(\d{4})(\d{2})", 1) + "-" + pl.col("_file").str.extract(r"^\d{4}(\d{2})", 1)
    report.month_differs_from_file = int(df.select((pl.col("month") != file_month).fill_null(False).sum()).item())

    out = df.select(
        pl.col("ride_id").alias("trip_id"),
        "month",
        pl.col("rideable_type").replace_strict(BIKE_TYPES).cast(BikeType).alias("bike_type"),
        pl.col("member_casual").replace_strict(USER_TYPES).cast(UserType).alias("user_type"),
        "start_time",
        "end_time",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
        pl.col("start_lat").cast(pl.Float64),
        pl.col("start_lng").cast(pl.Float64).alias("start_lon"),
        pl.col("end_lat").cast(pl.Float64),
        pl.col("end_lng").cast(pl.Float64).alias("end_lon"),
    ).sort(["start_time", "trip_id"])
    report.rows_out = out.height
    return out, report
