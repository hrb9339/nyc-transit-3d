"""Walking access to rail that is not the subway: LIRR, Metro-North and PATH.

The subway model predicts subway riders, but a hexagon next to an LIRR or PATH
station is not a transit gap even if it is far from the subway. This table lets
the model (and the gap ranking) see that. The Staten Island Railway is already
in the subway station data, so it is not repeated here.

Distances are walks along the same OpenStreetMap pedestrian network as walk.csv
(reuses data/raw/walk_graph.npz, so run features_walk.py first), measured to
the station's GTFS location (no entrance data for these railroads).

Train counts are from each railroad's GTFS schedule on one ordinary weekday
(a Wednesday): all departures that day, and departures per hour in the
7-10 am peak.

Writes:  data/features/rail.csv
Usage:   python scripts/features_rail.py
"""

import heapq
from datetime import date

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from features_common import RAW, load_hexes, write_features
from gtfs_common import departures, gtfs_zip, read

WALK_M = 800
WALK_CAP_M = 5_000    # same cap as walk.csv
NEAR_GRID_M = 4_000   # ignore stations farther than this from every hexagon center
PEAK = (7, 10)

# name, label, GTFS url, info page, weekday to count trains on
FEEDS = [
    ("gtfslirr", "LIRR", "https://rrgtfsfeeds.s3.amazonaws.com/gtfslirr.zip",
     "https://new.mta.info/developers", date(2026, 11, 18)),
    ("gtfsmnr", "Metro-North", "https://rrgtfsfeeds.s3.amazonaws.com/gtfsmnr.zip",
     "https://new.mta.info/developers", date(2026, 11, 18)),
    # PATH's published feed (via Trillium) was last valid on 2026-06-01; station
    # locations have not changed, and its schedule is used as a typical weekday.
    ("path2", "PATH", "http://data.trilliumtransit.com/gtfs/path-nj-us/path-nj-us.zip",
     "https://www.panynj.gov/path/en/schedules-maps.html", date(2026, 5, 13)),
]

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform


def local_dijkstra(graph, start, limit):
    """Distances from `start` to every node within `limit` meters: {node: meters}."""
    indptr, indices, data = graph.indptr, graph.indices, graph.data
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, np.inf):
            continue
        for k in range(indptr[u], indptr[u + 1]):
            v, nd = indices[k], d + data[k]
            if nd <= limit and nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def load_stations():
    """One row per station: system, name, lon, lat, weekday_trains, peak_trains_per_hour."""
    rows = []
    for name, label, url, _, day in FEEDS:
        zf = gtfs_zip(name, url)
        stops = read(zf, "stops")
        if "parent_station" not in stops:
            stops["parent_station"] = np.nan
        if "location_type" not in stops:
            stops["location_type"] = "0"
        stops["location_type"] = stops.location_type.fillna("0")
        # Station = parent station if there is one, else the stop itself.
        stops["station_id"] = stops.parent_station.where(stops.parent_station.notna(), stops.stop_id)
        stations = stops[stops.stop_id == stops.station_id].copy()

        dep = departures(zf, day).merge(stops[["stop_id", "station_id"]], on="stop_id")
        dep = dep.drop_duplicates(["station_id", "trip_id"])  # one stop per trip per station
        daily = dep.groupby("station_id").size()
        peak = dep[(dep.hour >= PEAK[0]) & (dep.hour < PEAK[1])].groupby("station_id").size() / (PEAK[1] - PEAK[0])

        stations["system"] = label
        stations["weekday_trains"] = stations.station_id.map(daily).fillna(0).astype(int)
        stations["peak_trains_per_hour"] = stations.station_id.map(peak).fillna(0).round(1)
        stations = stations[stations.weekday_trains > 0]   # drop stops with no service
        rows.append(stations[["system", "stop_name", "stop_lon", "stop_lat", "weekday_trains",
                              "peak_trains_per_hour"]])
    st = pd.concat(rows, ignore_index=True)
    st["lon"] = st.stop_lon.astype(float)
    st["lat"] = st.stop_lat.astype(float)
    return st.rename(columns={"stop_name": "name"}).drop(columns=["stop_lon", "stop_lat"])


def main():
    g = np.load(RAW / "walk_graph.npz")
    x, y, keep = g["x"], g["y"], g["in_main"]
    graph = csr_matrix((g["data"], g["indices"], g["indptr"]), shape=(len(x), len(x)))
    keep_ids = np.flatnonzero(keep)
    tree = cKDTree(np.column_stack([x[keep], y[keep]]))

    def snap(lon, lat):
        px, py = to_utm(np.asarray(lon), np.asarray(lat))
        d, i = tree.query(np.column_stack([px, py]))
        return keep_ids[i], d, px, py

    hexes = load_hexes()
    h_node, h_snap, hx, hy = snap(hexes.center_lon, hexes.center_lat)

    st = load_stations()
    s_node, s_snap, sx, sy = snap(st.lon, st.lat)
    near = cKDTree(np.column_stack([hx, hy])).query(np.column_stack([sx, sy]))[0] <= NEAR_GRID_M
    st, s_node, s_snap, sx, sy = st[near].reset_index(drop=True), s_node[near], s_snap[near], sx[near], sy[near]
    print(f"{len(st)} stations near the city: {st.system.value_counts().to_dict()}")

    def walk_to(mask):
        """Capped walk from every hexagon center to the nearest station in `mask`, and its index."""
        idx = np.flatnonzero(mask)
        d, _, src = dijkstra(graph, directed=False, indices=s_node[idx], min_only=True, return_predecessors=True)
        # src = graph node of the nearest source; map back to a station (ties: first)
        node_to_station = pd.Series(idx, index=s_node[idx]).groupby(level=0).first()
        raw = d[h_node] + h_snap + s_snap[node_to_station.reindex(src[h_node]).fillna(idx[0]).astype(int).values]
        nearest = node_to_station.reindex(src[h_node]).values
        return raw, nearest

    out = pd.DataFrame({"hex_id": hexes.hex_id})
    raw_all, nearest = walk_to(np.ones(len(st), bool))
    nearest = np.where(np.isfinite(raw_all), nearest, np.nan)
    out["walk_to_other_rail_m"] = np.round(np.minimum(raw_all, WALK_CAP_M)).astype(int)
    out["other_rail_capped"] = (raw_all > WALK_CAP_M).astype(int)
    out["straight_to_other_rail_m"] = np.round(cKDTree(np.column_stack([sx, sy])).query(np.column_stack([hx, hy]))[0]).astype(int)
    ok = ~np.isnan(nearest) & (raw_all <= WALK_CAP_M)
    ni = np.nan_to_num(nearest).astype(int)
    out["nearest_other_rail"] = np.where(ok, st.name.values[ni], "")
    out["nearest_other_rail_system"] = np.where(ok, st.system.values[ni], "")
    out["nearest_other_rail_weekday_trains"] = np.where(ok, st.weekday_trains.values[ni], 0)
    out["nearest_other_rail_peak_per_hour"] = np.where(ok, st.peak_trains_per_hour.values[ni], 0)
    for label, col in [("LIRR", "walk_to_lirr_m"), ("Metro-North", "walk_to_metro_north_m"), ("PATH", "walk_to_path_m")]:
        raw, _ = walk_to(st.system.eq(label).values)
        out[col] = np.round(np.minimum(raw, WALK_CAP_M)).astype(int)
    out["other_rail_within_800"] = (raw_all <= WALK_M).astype(int)

    # Stations and trains within an 800 m walk of each center (one short search per station).
    count = np.zeros(len(hexes), int)
    trains = np.zeros(len(hexes), int)
    hex_by_node = pd.Series(np.arange(len(hexes)), index=h_node).groupby(level=0).agg(list)
    for i in range(len(st)):
        reach = local_dijkstra(graph, int(s_node[i]), WALK_M)
        for node, d in reach.items():
            for h in hex_by_node.get(node, []):
                if d + h_snap[h] + s_snap[i] <= WALK_M:
                    count[h] += 1
                    trains[h] += st.weekday_trains.iat[i]
    out["other_rail_stations_800m"] = count
    out["other_rail_trains_800m"] = trains

    # Any rail at all: the subway entrance walk from walk_graph.npz vs. other rail.
    subway_raw = g["to_entrance"][h_node] + h_snap
    any_raw = np.minimum(subway_raw, raw_all)
    out["walk_to_any_rail_m"] = np.round(np.minimum(any_raw, WALK_CAP_M)).astype(int)
    out["any_rail_within_800"] = (any_raw <= WALK_M).astype(int)

    gain = ((subway_raw > WALK_M) & (raw_all <= WALK_M)).sum()
    print(f"{out.other_rail_within_800.sum()} hexagon centers within an 800 m walk of LIRR/Metro-North/PATH; "
          f"{gain} of them are not within 800 m of the subway")

    st[["system", "name", "lon", "lat", "weekday_trains", "peak_trains_per_hour"]].to_csv(
        RAW / "other_rail_stations.csv", index=False)

    write_features(
        out, "rail",
        "Walking access to LIRR, Metro-North and PATH per hexagon",
        sources=[
            ("MTA Long Island Rail Road GTFS (stations and weekday schedule)", FEEDS[0][2]),
            ("MTA Metro-North Railroad GTFS (stations and weekday schedule)", FEEDS[1][2]),
            ("MTA developer resources (feed index)", "https://new.mta.info/developers"),
            ("PATH GTFS published by the Port Authority via Trillium Transit", FEEDS[2][2]),
            ("PATH schedules and maps", FEEDS[2][3]),
            ("OpenStreetMap pedestrian network (from features_walk.py, data/raw/walk_graph.npz)",
             "https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf"),
        ],
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "walk_to_other_rail_m": f"walking distance from the hexagon center to the nearest LIRR, Metro-North or "
                                    f"PATH station, m (capped at {WALK_CAP_M:,})",
            "other_rail_capped": f"1 if that walk is over {WALK_CAP_M:,} m (or impossible) and was capped",
            "straight_to_other_rail_m": "straight-line distance to the nearest such station, m (ignores rivers)",
            "nearest_other_rail": "name of the nearest station by walk (blank if capped)",
            "nearest_other_rail_system": "LIRR, Metro-North or PATH (blank if capped)",
            "nearest_other_rail_weekday_trains": "trains stopping at that station on a weekday (all directions)",
            "nearest_other_rail_peak_per_hour": "trains per hour stopping there, 7-10 am weekday",
            "walk_to_lirr_m": f"walking distance to the nearest LIRR station, m (capped at {WALK_CAP_M:,})",
            "walk_to_metro_north_m": f"walking distance to the nearest Metro-North station, m (capped at {WALK_CAP_M:,})",
            "walk_to_path_m": f"walking distance to the nearest PATH station, m (capped at {WALK_CAP_M:,})",
            "other_rail_within_800": "1 if any LIRR, Metro-North or PATH station is within an 800 m walk",
            "other_rail_stations_800m": "number of LIRR, Metro-North and PATH stations within an 800 m walk",
            "other_rail_trains_800m": "weekday trains summed over those stations",
            "walk_to_any_rail_m": f"walking distance to the nearest subway entrance or other rail station, m "
                                  f"(capped at {WALK_CAP_M:,})",
            "any_rail_within_800": "1 if a subway entrance or other rail station is within an 800 m walk",
        },
        notes=[
            "Staten Island Railway is not here: it is part of the subway station data.",
            f"Stations more than {NEAR_GRID_M // 1000} km from every hexagon center are left out (most of the "
            "suburban network). Stations just over the Nassau and Westchester lines are kept.",
            "Distances go to the station's GTFS point, not its entrances, so they can be off by a platform length.",
            "Weekday counts: MTA railroads on Wed 2026-11-18; PATH on Wed 2026-05-13 (its last published schedule).",
            "Penn Station, Grand Central and World Trade Center are also subway stations; the model should "
            "read these columns together with the subway ones.",
            "Station list with coordinates and train counts: data/raw/other_rail_stations.csv (git-ignored).",
        ],
    )


if __name__ == "__main__":
    main()
