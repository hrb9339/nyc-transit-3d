"""Real proposed subway/rail lines, and each hexagon's distance to them.

These are used later to check the model: if its top gaps fall along corridors
that planners have already chosen, that's good evidence it works.

Alignments are approximate and taken from OpenStreetMap:
  - Interborough Express (IBX): planned light rail on the existing Bay Ridge Branch
    and Fremont freight tracks, from Sunset Park (Brooklyn) to Jackson Heights (Queens).
    Freight sidings and yards are dropped; the line is cut at its planned ends.
  - Second Avenue Subway Phase 2: ways tagged as under construction, up Second Avenue
    and west along 125th Street (including tail tracks). The separate "125th Street
    Subway" idea (extending the line west to Broadway) is left out.
  - Utica Avenue line: a long-proposed extension of the 3/4 trains south from
    Crown Heights-Utica Av, drawn along Utica Avenue itself (the street the tunnel would follow).

Writes:  data/proposals.geojson                  the three lines, for the map
         data/features/proposals.csv             distance from each hexagon center to each line
Usage:   python scripts/features_proposals.py
"""

import json
import re

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString, mapping
from shapely.ops import linemerge, unary_union

from features_common import DATA, RAW, UTM, hex_centers, load_hexes, write_features

PBF = RAW / "NewYork.osm.pbf"
PBF_URL = "https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf"
WALK_M = 800  # "near a line" threshold, same walking radius used for station areas

# Planned ends (lon/lat boxes) used to trim the freight tracks to the IBX route.
IBX_LON_MIN = -74.02      # west end near 4th Av-Sunset Park; drops the waterfront float-bridge spur
IBX_LAT_MAX = 40.7475     # north end at Roosevelt Avenue, Jackson Heights
UTICA_LAT_MAX = 40.6690   # Crown Heights-Utica Av station, where the extension would start

LINES = {
    "ibx": ("Interborough Express (planned light rail)", "https://ibx.mta.info/"),
    "sas2": ("Second Avenue Subway Phase 2 (under construction)", "https://www.mta.info/project/second-avenue-subway-phase-2"),
    "utica": ("Utica Avenue subway extension (proposed)",
              "https://www.openstreetmap.org/search?query=Utica%20Avenue%2C%20Brooklyn"),
}


def read_ways():
    """Way geometries for the three corridors, as {key: [LineString, ...]}."""
    out = {k: [] for k in LINES}
    fp = osmium.FileProcessor(str(PBF), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for w in fp:
        if not w.is_way():
            continue
        t = dict(w.tags)
        name = t.get("name", "")
        key = None
        if "railway" in t and re.fullmatch(r"Bay Ridge Branch|Fremont Industrial Track", name) \
                and t.get("railway") == "rail" and "service" not in t:
            key = "ibx"
        elif name == "Second Avenue Subway Phase 2" and t.get("railway") == "construction":
            key = "sas2"
        elif name == "Utica Avenue" and t.get("highway") in ("primary", "secondary", "tertiary"):
            key = "utica"
        if key is None:
            continue
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            continue
        if len(coords) >= 2:
            out[key].append(LineString(coords))
    return out


def merge(lines):
    """Join touching segments; returns a LineString or MultiLineString."""
    g = unary_union(lines)
    return linemerge(g) if g.geom_type == "MultiLineString" else g


def trim(key, lines):
    merged = merge(lines)
    parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    if key == "ibx":
        parts = [LineString([c for c in p.coords if c[0] >= IBX_LON_MIN and c[1] <= IBX_LAT_MAX]) for p in parts]
    elif key == "utica":
        parts = [LineString([c for c in p.coords if c[1] <= UTICA_LAT_MAX]) for p in parts if len(p.coords) >= 2]
    parts = [p for p in parts if len(p.coords) >= 2]
    return merge(parts)


def main():
    if not PBF.exists():
        raise SystemExit(f"download the NYC OpenStreetMap extract first:\n  curl -o {PBF} {PBF_URL}")
    ways = read_ways()
    lines = {}
    for key, parts in ways.items():
        lines[key] = trim(key, parts)
        km = gpd.GeoSeries([lines[key]], crs="EPSG:4326").to_crs(UTM).length.iloc[0] / 1000
        print(f"{key}: {len(parts)} ways -> {km:.1f} km of track")

    (DATA / "proposals.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"id": k, "name": LINES[k][0], "source": LINES[k][1]},
                      "geometry": mapping(lines[k].simplify(0.00002))} for k in LINES],
    }, separators=(",", ":")))
    print("wrote data/proposals.geojson")

    centers = hex_centers(load_hexes()).to_crs(UTM)
    out = pd.DataFrame({"hex_id": centers.hex_id})
    for key in LINES:
        line = gpd.GeoSeries([lines[key]], crs="EPSG:4326").to_crs(UTM).iloc[0]
        out[f"dist_{key}_m"] = centers.distance(line).round(0).astype(int).values
        out[f"near_{key}"] = (out[f"dist_{key}_m"] <= WALK_M).astype(int)
    out["near_any_proposal"] = out[[f"near_{k}" for k in LINES]].max(axis=1)
    print("hexagons within 800 m: " + ", ".join(f"{k} {out[f'near_{k}'].sum()}" for k in LINES))

    columns = {"hex_id": "hexagon id (matches data/hexgrid.geojson)"}
    for key, (name, _) in LINES.items():
        columns[f"dist_{key}_m"] = f"straight-line distance from the hexagon center to the {name} alignment, m"
        columns[f"near_{key}"] = f"1 if within {WALK_M} m of the {name.split(' (')[0]} alignment, else 0"
    columns["near_any_proposal"] = f"1 if within {WALK_M} m of any of the three lines"
    write_features(
        out, "proposals",
        "Distance from each hexagon to real proposed transit lines",
        sources=[
            ("OpenStreetMap contributors, New York extract from BBBike (ODbL); alignments of the Bay Ridge Branch, "
             "Fremont Industrial Track, Second Avenue Subway Phase 2 and Utica Avenue", PBF_URL),
            *[(f"{name}, project information" if k != "utica" else
              f"{name}: no project page; evaluated in the MTA 20-Year Needs Assessment (2023). Link is the street it follows",
              url) for k, (name, url) in LINES.items()],
        ],
        columns=columns,
        notes=[
            "Alignments are approximate: the IBX follows the freight tracks it will reuse; Utica follows the street.",
            "For validation only. Don't use these as model inputs, or the model will learn the planners' answer.",
            "Line geometry is in data/proposals.geojson.",
        ],
    )


if __name__ == "__main__":
    main()
