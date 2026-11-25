"""Places: the spatial unit of demand (issue #78, spec §6.1 and §6.5).

Places are the stations **inside the city** (sorted by station id) plus one ``EXTERNAL`` place,
always the last index. Every trip end inside the city is assigned to the **nearest in-city
station** by straight-line distance. This covers dockless ends (e-bikes left outside a station)
and keeps one consistent rule. Every end outside the city is ``EXTERNAL``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from bss_scenario.network.graph import Projection

EXTERNAL = "EXTERNAL"


@dataclass(frozen=True)
class Places:
    station_ids: tuple[str, ...]  # in-city stations; place index = position; EXTERNAL = len(station_ids)

    @property
    def n(self) -> int:
        return len(self.station_ids) + 1

    @property
    def external(self) -> int:
        return len(self.station_ids)

    def table(self) -> pl.DataFrame:
        return pl.DataFrame({"place": np.arange(self.n, dtype=np.int16),
                             "station_id": [*self.station_ids, EXTERNAL]})

    @classmethod
    def from_stations(cls, stations: pl.DataFrame) -> Places:
        ids = stations.filter("in_city")["station_id"].sort().to_list()
        if not ids:
            raise ValueError("no in-city stations")
        return cls(tuple(ids))


def assign_places(trips: pl.DataFrame, stations: pl.DataFrame, projection: Projection) -> pl.DataFrame:
    """Add ``o_place`` / ``d_place`` (Int16) to trips with known ends and city flags."""
    places = Places.from_stations(stations)
    st = stations.filter("in_city").sort("station_id")
    sx, sy = projection.to_xy(st["lat"].to_numpy(), st["lon"].to_numpy())
    tree = cKDTree(np.column_stack([sx, sy]))

    def nearest(lat: pl.Series, lon: pl.Series, inside: pl.Series) -> np.ndarray:
        x, y = projection.to_xy(lat.fill_null(projection.lat0).to_numpy(), lon.fill_null(projection.lon0).to_numpy())
        _, idx = tree.query(np.column_stack([x, y]), k=1)
        return np.where(inside.fill_null(False).to_numpy(), idx, places.external).astype(np.int16)

    return trips.with_columns(
        pl.Series("o_place", nearest(trips["start_lat"], trips["start_lon"], trips["start_in_city"])),
        pl.Series("d_place", nearest(trips["end_lat"], trips["end_lon"], trips["end_in_city"])),
    )


def assignment_report(trips: pl.DataFrame, places: Places) -> dict:
    """How often the nearest-station rule differs from the recorded station id (in-city ends)."""
    ids = pl.Series(places.table()["station_id"])
    out = {}
    for end, col in (("start", "o_place"), ("end", "d_place")):
        inside = trips.filter(pl.col(col) != places.external)
        recorded = inside[f"{end}_station_id"]
        assigned = ids.gather(inside[col].cast(pl.UInt32))
        out[end] = {
            "in_city": inside.height,
            "no_station_id": int(recorded.null_count()),
            "differs_from_recorded": int(((recorded != assigned) & recorded.is_not_null()).sum()),
        }
    out["external_share"] = float(((trips["o_place"] == places.external) | (trips["d_place"] == places.external)).mean())
    return out
