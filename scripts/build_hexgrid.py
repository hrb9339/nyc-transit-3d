"""Hexagonal analysis grid over New York City.

Pointy-top hexagons with a 500 m circumradius (center to corner), laid out in
UTM zone 18N so every cell is the same true size (~0.65 km²). Each cell is a
candidate "station area" for the transit-gap analysis. Cells are cut to the
shoreline and mostly-water cells are dropped, so the grid traces the boroughs.

Inputs:  NYC borough boundaries clipped to the shoreline (NYC Open Data gthc-hcne)
         data/subway-stations.geojson (from build_data.py)
Writes:  data/hexgrid.geojson       land-clipped hexagons with attributes, for display
         data/hexgrid_land.geojson  same shapes with ids only, for Earth Engine statistics

Usage:   python scripts/build_hexgrid.py
"""

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import Point, Polygon, mapping, shape
from shapely.ops import unary_union

DATA = Path(__file__).resolve().parent.parent / "data"
BOROUGHS_URL = "https://data.cityofnewyork.us/resource/gthc-hcne.geojson"
RADIUS_M = 500
MIN_LAND_FRACTION = 0.1  # drop cells that are at least 90% water
SIMPLIFY_M = 5           # shoreline detail far finer than this isn't visible or needed

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
_to_wgs = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
to_utm = _to_utm.transform


def utm(geom):
    return shapely.transform(geom, lambda c: np.column_stack(_to_utm.transform(c[:, 0], c[:, 1])))


def wgs(geom, nd=6):
    return shapely.transform(geom, lambda c: np.round(np.column_stack(_to_wgs.transform(c[:, 0], c[:, 1])), nd))


def hexagon(cx, cy, r):
    return Polygon([
        (cx + r * math.cos(math.radians(60 * k + 30)), cy + r * math.sin(math.radians(60 * k + 30)))
        for k in range(6)
    ])


def main():
    print("Downloading borough boundaries...")
    with urllib.request.urlopen(BOROUGHS_URL, timeout=60) as r:
        boroughs = json.loads(r.read())
    boro_shapes = [
        (f["properties"].get("boroname") or f["properties"].get("boro_name"), utm(shape(f["geometry"])).buffer(0))
        for f in boroughs["features"]
    ]
    land = unary_union([g for _, g in boro_shapes])
    shapely.prepare(land)

    stations = json.loads((DATA / "subway-stations.geojson").read_text())
    station_pts = [
        (s["properties"]["name"], Point(to_utm(*s["geometry"]["coordinates"])))
        for s in stations["features"]
    ]

    # Lay out the honeycomb over the city's bounding box.
    minx, miny, maxx, maxy = land.bounds
    dx, dy = math.sqrt(3) * RADIUS_M, 1.5 * RADIUS_M
    hex_area = 1.5 * math.sqrt(3) * RADIUS_M ** 2

    features, ids_only, centers = [], [], []
    row = 0
    y = miny - RADIUS_M
    while y < maxy + RADIUS_M:
        col = 0
        x = minx - RADIUS_M + (dx / 2 if row % 2 else 0)
        while x < maxx + RADIUS_M:
            cell = hexagon(x, y, RADIUS_M)
            land_part = cell.intersection(land) if land.intersects(cell) else None
            if land_part is not None and land_part.area / hex_area >= MIN_LAND_FRACTION:
                nearest = min(Point(x, y).distance(p) for _, p in station_pts)
                boro = max(boro_shapes, key=lambda b: land_part.intersection(b[1]).area)[0]
                lon, lat = _to_wgs.transform(x, y)
                shape_wgs = mapping(wgs(land_part.simplify(SIMPLIFY_M)))
                props = {
                    "hex_id": f"r{row:03d}c{col:03d}",
                    "borough": boro,
                    "center": [round(lon, 6), round(lat, 6)],
                    "land_km2": round(land_part.area / 1e6, 3),
                    "land_fraction": round(land_part.area / hex_area, 3),
                    "stations": 0,
                    "station_names": [],
                    "nearest_station_m": round(nearest),
                }
                centers.append((x, y, props))
                features.append({"type": "Feature", "properties": props, "geometry": shape_wgs})
                ids_only.append({"type": "Feature", "properties": {"hex_id": props["hex_id"]}, "geometry": shape_wgs})
            x += dx
            col += 1
        y += dy
        row += 1

    # On a hex lattice a point lies in the cell with the nearest center, so this
    # assigns every station to exactly one cell (no double counts on edges).
    for name, p in station_pts:
        _, _, props = min(centers, key=lambda c: (c[0] - p.x) ** 2 + (c[1] - p.y) ** 2)
        props["stations"] += 1
        props["station_names"] = sorted(set(props["station_names"]) | {name})

    for name, feats in (("hexgrid.geojson", features), ("hexgrid_land.geojson", ids_only)):
        (DATA / name).write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))

    with_station = sum(1 for f in features if f["properties"]["stations"])
    print(f"{len(features)} hexagons (radius {RADIUS_M} m, {hex_area / 1e6:.2f} km² each)")
    print(f"  {with_station} contain at least one subway station, {len(features) - with_station} contain none")
    by_boro = {}
    for f in features:
        by_boro[f["properties"]["borough"]] = by_boro.get(f["properties"]["borough"], 0) + 1
    print("  by borough:", by_boro)


if __name__ == "__main__":
    main()
