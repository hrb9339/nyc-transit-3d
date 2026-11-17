"""Bus service per hexagon: stops, routes and how many buses come on a weekday.

Some places far from the subway have strong bus service (Select Bus Service,
express buses to Manhattan), and buses feed riders into subway stations. This
counts, for every hexagon, the bus stops inside it and the buses that stop
there on one ordinary weekday (a Wednesday), from the MTA's six bus GTFS feeds
(five NYC Transit borough feeds + MTA Bus Company).

Writes:  data/features/bus.csv
Usage:   python scripts/features_bus.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from features_common import UTM, hex_centers, load_hexes, write_features
from gtfs_common import departures, gtfs_zip, parse_day, read

DAY = parse_day("2026-11-18")   # a Wednesday inside every feed's service period
PEAK = (7, 10)
BASE = "https://rrgtfsfeeds.s3.amazonaws.com/"
FEEDS = {
    "gtfs_bx": "Bronx", "gtfs_b": "Brooklyn", "gtfs_m": "Manhattan",
    "gtfs_q": "Queens", "gtfs_si": "Staten Island", "gtfs_busco": "MTA Bus Company",
}


def route_kind(route_id):
    """SBS = Select Bus Service (route ids end in '+'); express = BM, BxM, QM, SIM, X routes."""
    r = route_id.upper()
    if r.endswith("+") or r.endswith("-SBS"):
        return "sbs"
    if r.startswith(("BM", "BXM", "QM", "SIM", "X")):
        return "express"
    return "local"


def main():
    stops, deps = [], []
    for name, label in FEEDS.items():
        zf = gtfs_zip(name, BASE + name + ".zip")
        s = read(zf, "stops", usecols=["stop_id", "stop_lat", "stop_lon"])
        stops.append(s)
        d = departures(zf, DAY)
        print(f"  {label}: {len(s):,} stops, {d.trip_id.nunique():,} weekday trips, {len(d):,} stop events")
        deps.append(d)
    stops = pd.concat(stops).drop_duplicates("stop_id")
    deps = pd.concat(deps).drop_duplicates(["stop_id", "trip_id"])
    deps["kind"] = deps.route_id.map(route_kind)

    # Only stops with service that day.
    stops = stops[stops.stop_id.isin(deps.stop_id)]
    pts = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops.stop_lon.astype(float),
                                                              stops.stop_lat.astype(float)), crs="EPSG:4326")
    hexes = load_hexes()
    in_hex = gpd.sjoin(pts, hexes[["hex_id", "geometry"]], predicate="within")[["stop_id", "hex_id"]]
    d = deps.merge(in_hex, on="stop_id")
    peak = d[(d.hour >= PEAK[0]) & (d.hour < PEAK[1])]

    out = pd.DataFrame({"hex_id": hexes.hex_id}).set_index("hex_id")
    out["bus_stops"] = in_hex.groupby("hex_id").size()
    out["bus_routes"] = d.groupby("hex_id").route_id.nunique()
    out["sbs_routes"] = d[d.kind == "sbs"].groupby("hex_id").route_id.nunique()
    out["express_routes"] = d[d.kind == "express"].groupby("hex_id").route_id.nunique()
    out["bus_stop_events_weekday"] = d.groupby("hex_id").size()
    out["bus_trips_weekday"] = d.groupby("hex_id").trip_id.nunique()
    out["bus_trips_peak_per_hour"] = peak.groupby("hex_id").trip_id.nunique() / (PEAK[1] - PEAK[0])
    out["express_trips_peak_per_hour"] = peak[peak.kind == "express"].groupby("hex_id").trip_id.nunique() / (PEAK[1] - PEAK[0])
    out = out.fillna(0)
    for c in out.columns:
        out[c] = out[c].round(1) if "per_hour" in c else out[c].astype(int)
    out["bus_trips_per_km2"] = (out.bus_trips_weekday / hexes.set_index("hex_id").land_km2).round(0)

    # Distance from the center to the nearest served stop (straight line).
    c = hex_centers(hexes).to_crs(UTM)
    sp = pts.to_crs(UTM)
    dist = cKDTree(np.column_stack([sp.geometry.x, sp.geometry.y])).query(
        np.column_stack([c.geometry.x, c.geometry.y]))[0]
    out["nearest_bus_stop_m"] = np.round(dist).astype(int)
    out = out.reset_index()

    print(f"{len(stops):,} served stops, {len(in_hex):,} inside the grid; "
          f"median {out.bus_trips_weekday.median():.0f} weekday trips per hexagon; "
          f"{(out.bus_stops == 0).sum()} hexagons with no bus stop")

    write_features(
        out, "bus",
        "MTA bus service per hexagon (weekday schedule)",
        sources=[(f"MTA bus GTFS: {label}", BASE + name + ".zip") for name, label in FEEDS.items()]
        + [("MTA developer resources (feed index)", "https://new.mta.info/developers")],
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "bus_stops": "bus stops inside the hexagon with service on the sample weekday",
            "bus_routes": "distinct bus routes stopping in the hexagon",
            "sbs_routes": "Select Bus Service routes stopping in the hexagon",
            "express_routes": "express routes (BM, BxM, QM, SIM, X) stopping in the hexagon",
            "bus_stop_events_weekday": "bus stops made in the hexagon on the weekday (a trip stopping at 3 stops "
                                       "here counts 3)",
            "bus_trips_weekday": "distinct bus trips stopping in the hexagon at least once on the weekday",
            "bus_trips_peak_per_hour": "distinct bus trips per hour stopping in the hexagon, 7-10 am",
            "express_trips_peak_per_hour": "express bus trips per hour stopping in the hexagon, 7-10 am "
                                           "(a sign of long commutes to Manhattan)",
            "bus_trips_per_km2": "bus_trips_weekday per km² of hexagon land",
            "nearest_bus_stop_m": "straight-line distance from the hexagon center to the nearest served bus stop, m",
        },
        notes=[
            f"Sample weekday: Wed {DAY.isoformat()} (all six feeds were valid then).",
            "Both directions count; a route running every 10 minutes each way gives ~12 trips per hour.",
            "Stops are counted in the hexagon they fall in, so a busy route along a hexagon edge may be split "
            "between neighbours.",
            "Express buses mostly run toward Manhattan in the morning, so their peak counts are one-directional.",
        ],
    )


if __name__ == "__main__":
    main()
