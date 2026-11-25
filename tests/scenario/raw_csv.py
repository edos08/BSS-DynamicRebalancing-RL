"""Helpers to write small raw BlueBikes CSVs (2024 schema) for tests."""

import csv

HEADER = ["ride_id", "rideable_type", "started_at", "ended_at", "start_station_name", "start_station_id",
          "end_station_name", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"]


def trip(ride_id, start, end, *, kind="classic_bike", user="member", s_id="A1", e_id="B1",
         s=(42.3736, -71.1190), e=(42.3626, -71.0843)):
    return {
        "ride_id": ride_id, "rideable_type": kind, "started_at": start, "ended_at": end,
        "start_station_name": f"Station {s_id}" if s_id else "", "start_station_id": s_id or "",
        "end_station_name": f"Station {e_id}" if e_id else "", "end_station_id": e_id or "",
        "start_lat": s[0], "start_lng": s[1],
        "end_lat": "" if e is None else e[0], "end_lng": "" if e is None else e[1],
        "member_casual": user,
    }


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)
    return path
