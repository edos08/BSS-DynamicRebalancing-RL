"""Clip the OSM graphs and the boundary to a box (issue #166).

Usage:
    python scripts/clip_osm.py --bbox 42.36889381 42.35248869 -71.07231001 -71.11736849 \
        --out data/raw/osm/cambridge_se

``--bbox north south east west`` uses the same order as the legacy ``sources.json``. The default
is the zone of the legacy environment (south-east Cambridge: MIT, Cambridgeport, Kendall). Writes
``<out>_bike.graphml``, ``<out>_drive.graphml`` (nodes inside the box, plus the edges between them;
``bss-scenario`` then keeps the largest strongly connected part) and ``<out>_boundary.geojson``
(the box intersected with the city boundary).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import shapely
from shapely.geometry import box, mapping

from bss_scenario.trips.clean import load_boundary

LEGACY_BBOX = (42.36889381, 42.35248869, -71.07231001, -71.11736849)


def clip_graph(src: Path, dst: Path, n: float, s: float, e: float, w: float) -> tuple[int, int]:
    g = nx.read_graphml(src)
    keep = [v for v, d in g.nodes(data=True) if s <= float(d["y"]) <= n and w <= float(d["x"]) <= e]
    sub = g.subgraph(keep).copy()
    nx.write_graphml(sub, dst)
    return sub.number_of_nodes(), sub.number_of_edges()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, default=LEGACY_BBOX, metavar=("N", "S", "E", "W"))
    ap.add_argument("--src", type=Path, default=Path("data/raw/osm/cambridge"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/osm/cambridge_se"))
    args = ap.parse_args()
    n, s, e, w = args.bbox
    for kind in ("bike", "drive"):
        nodes, edges = clip_graph(Path(f"{args.src}_{kind}.graphml"), Path(f"{args.out}_{kind}.graphml"), n, s, e, w)
        print(f"{kind}: {nodes} nodes, {edges} edges")
    city = load_boundary(Path(f"{args.src}_boundary.geojson"))
    zone = shapely.intersection(city, box(w, s, e, n))
    Path(f"{args.out}_boundary.geojson").write_text(json.dumps(mapping(zone)))
    print(f"boundary: {zone.area * 111_000 * 82_000 / 1e6:.2f} km2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
