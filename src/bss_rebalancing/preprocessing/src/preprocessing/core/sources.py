"""
Dynamic data-source registry for the preprocessing pipeline.

Every bike-share feed (BlueBikes, CitiBike NYC, whatever you bolt on next
year) is a JSON entry now, not a hardcoded string buried in a script nobody
will remember to update. Add a source, don't touch code.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Source:
    id: str
    base_url: str
    url_prefix: str
    url_suffix: str
    graph_place: str
    network_type: str
    bbox: Tuple[float, float, float, float]
    years: List[int]
    months: List[int]
    station_id_type: str          # "numeric" | "alphanumeric"
    multi_csv_per_zip: bool       # citibike zips explode into many CSVs, bluebikes doesn't
    column_map: Dict[str, str]    # source_col -> canonical_col ("" or missing = drop)
    vital_columns: List[str]
    cell_size: int = 300

    def zip_filename(self, year: int, month: int) -> str:
        return f"{self.url_prefix}{year}{str(month).zfill(2)}{self.url_suffix}"

    def url_for(self, year: int, month: int) -> str:
        return f"{self.base_url}{self.zip_filename(year, month)}"

    @property
    def month_str(self) -> str:
        return f"{str(self.months[0]).zfill(2)}-{str(self.months[-1]).zfill(2)}"


def load_sources(json_path: str) -> Dict[str, Source]:
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"sources.json not found at {json_path}")

    raw = json.loads(path.read_text())
    sources = {}
    for source_id, entry in raw.get("sources", {}).items():
        try:
            sources[source_id] = Source(
                id=source_id,
                base_url=entry["base_url"],
                url_prefix=entry.get("url_prefix", "") or "",
                url_suffix=entry["url_suffix"],
                graph_place=entry["graph_place"],
                network_type=entry.get("network_type", "bike"),
                bbox=tuple(entry["bbox"]),
                years=entry.get("years", []),
                months=entry.get("months", []),
                station_id_type=entry.get("station_id_type", "alphanumeric"),
                multi_csv_per_zip=entry.get("multi_csv_per_zip", False),
                column_map=entry.get("column_map", {}),
                vital_columns=entry.get("vital_columns", []),
                cell_size=entry.get("cell_size", 300),
            )
        except KeyError as e:
            raise KeyError(f"Source '{source_id}' missing required field {e}")

    if not sources:
        raise ValueError(f"No sources defined in {json_path}")
    return sources


def get_source(json_path: str, source_id: str) -> Source:
    sources = load_sources(json_path)
    if source_id not in sources:
        raise KeyError(f"Unknown source '{source_id}'. Available: {list(sources.keys())}")
    return sources[source_id]