"""Free-flow speed and travel time of drive edges (issue #54, spec §9).

Speed comes from the OSM ``maxspeed`` tag when present, otherwise from a default per road class.
This is a **speed-limit** free flow: the overall level is later calibrated against the routing
service's no-traffic times (#57), so only the relative speeds between road classes matter much.

Parsing rules for ``maxspeed``:

* ``"25 mph"`` → 25 mph; ``"40 km/h"`` / ``"40 kph"`` → 40 km/h.
* A bare number has no unit. OSM says km/h, but in US data it is almost always a forgotten
  ``mph`` (e.g. ``"25"`` on a Cambridge street), so ``bare_unit`` defaults to mph.
* Several values (``"25 mph;20 mph"``, from merged OSM ways) → their mean.
* Anything else (``"signals"``, ``"none"``, ``"walk"``) → missing → class default.

Defaults (mph) reflect Cambridge: most city streets are posted 20 mph (default for city-owned
streets) and arterials 25 mph; parkways (OSM ``trunk``) 35 mph.
"""

from __future__ import annotations

import re

import polars as pl

MPH = 1.609344  # km/h per mph

DEFAULT_SPEED_MPH: dict[str, float] = {
    "motorway": 55, "trunk": 35, "primary": 25, "secondary": 25, "tertiary": 25,
    "motorway_link": 35, "trunk_link": 25, "primary_link": 20, "secondary_link": 20, "tertiary_link": 20,
    "unclassified": 20, "residential": 20, "living_street": 10, "service": 10,
}
FALLBACK_SPEED_MPH = 20.0

#: Road classes a van may not use.
EXCLUDED_FOR_VANS = frozenset({"busway"})

_VALUE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kph|kmh)?\s*$", re.IGNORECASE)


def parse_maxspeed_kmh(value: str | None, bare_unit: str = "mph") -> float | None:
    """Parse an OSM ``maxspeed`` value (possibly ``;``-separated) into km/h, or ``None``."""
    if value is None:
        return None
    speeds = []
    for part in str(value).split(";"):
        m = _VALUE.match(part)
        if not m:
            continue
        number = float(m.group(1))
        unit = (m.group(2) or bare_unit).lower()
        speeds.append(number * MPH if unit == "mph" else number)
    speeds = [s for s in speeds if s > 0]
    return sum(speeds) / len(speeds) if speeds else None


def edge_speeds(
    edges: pl.DataFrame,
    defaults_mph: dict[str, float] | None = None,
    bare_unit: str = "mph",
) -> pl.DataFrame:
    """Add ``speed_kmh``, ``speed_source`` (``maxspeed`` | ``default``) and ``freeflow_s`` to an edge table."""
    defaults_mph = DEFAULT_SPEED_MPH if defaults_mph is None else defaults_mph
    parsed = edges["maxspeed"].map_elements(lambda v: parse_maxspeed_kmh(v, bare_unit), return_dtype=pl.Float64)
    default_kmh = edges["highway"].map_elements(
        lambda h: defaults_mph.get(h, FALLBACK_SPEED_MPH) * MPH, return_dtype=pl.Float64
    )
    return edges.with_columns(
        parsed.fill_null(default_kmh).alias("speed_kmh"),
        pl.when(parsed.is_null()).then(pl.lit("default")).otherwise(pl.lit("maxspeed")).alias("speed_source"),
    ).with_columns((pl.col("length_m") / (pl.col("speed_kmh") / 3.6)).alias("freeflow_s"))


def speed_report(edges: pl.DataFrame) -> dict:
    """Share of edges (and of length) whose speed comes from an explicit ``maxspeed``."""
    explicit = edges["speed_source"] == "maxspeed"
    return {
        "edges": edges.height,
        "explicit_share_edges": float(explicit.mean()),
        "explicit_share_length": float(edges.filter(explicit)["length_m"].sum() / edges["length_m"].sum()),
        "default_by_class": edges.filter(~explicit).group_by("highway").len().sort("len", descending=True).to_dicts(),
    }
