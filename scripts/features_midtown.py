"""Travel time to Midtown and Lower Manhattan, door to door by subway.

Most NYC jobs are in Midtown and Lower Manhattan, and a station with a quick
ride there tends to have more riders. For every hexagon this estimates:

    walk to a subway station + 5 min wait + scheduled ride (with transfers)

to Midtown (Times Sq-42 St or Grand Central-42 St) and to Lower Manhattan
(Fulton St), taking the best station, not just the nearest one. It also gives
the straight-line distance to each, which works the same for hexagons with and
without a station.

Subway network: data/network.json (station-to-station scheduled travel times
from MTA GTFS, built by scripts/build_data.py). Platforms of different lines
within 300 m of each other are linked as transfers (walk + 3 min). The Staten
Island Railway reaches Manhattan by the Staten Island Ferry (25 min ride +
15 min average wait, St George to Whitehall St/South Ferry).

Walking: the OpenStreetMap network from features_walk.py (data/raw/walk_graph.npz),
75 m per minute.

Writes:  data/features/midtown.csv
         data/ridership/station_travel_time.csv   ride time per ridership station complex
Usage:   python scripts/features_midtown.py
"""

import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from features_common import DATA, RAW, load_hexes, write_features

WALK_M_PER_MIN = 75
WAIT_MIN = 5
TRANSFER_RADIUS_M = 300
TRANSFER_PENALTY_MIN = 3
FERRY_MIN = 25 + 15
CAP_MIN = 150
MIDTOWN = ["127", "725", "902", "R16", "631", "723", "901"]   # Times Sq-42 St, Grand Central-42 St platforms
DOWNTOWN = ["229", "418", "A38", "M22"]                        # Fulton St (Manhattan) platforms
MIDTOWN_POINT = (-73.9817, 40.7540)    # between Times Sq and Grand Central
DOWNTOWN_POINT = (-74.0080, 40.7100)   # Fulton St

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform


def subway_graph():
    """Parent stations (platform id without N/S) and a directed graph of minutes."""
    net = json.loads((DATA / "network.json").read_text())
    stops = {}
    for sid, (lon, lat, name) in net["stops"].items():
        stops.setdefault(sid[:-1] if sid[-1] in "NS" else sid, (lon, lat, name))
    ids = list(stops)
    idx = {s: i for i, s in enumerate(ids)}
    lon = np.array([stops[s][0] for s in ids])
    lat = np.array([stops[s][1] for s in ids])
    x, y = to_utm(lon, lat)

    src, dst, w = [], [], []
    for key, (seconds, _) in net["segments"].items():
        a, b = (k[:-1] if k[-1] in "NS" else k for k in key.split("|"))
        if a != b and a in idx and b in idx:
            src.append(idx[a]); dst.append(idx[b]); w.append(seconds / 60)
    # Transfers between nearby platforms of different lines.
    tree = cKDTree(np.column_stack([x, y]))
    for i, j in tree.query_pairs(TRANSFER_RADIUS_M):
        t = np.hypot(x[i] - x[j], y[i] - y[j]) * 1.3 / WALK_M_PER_MIN + TRANSFER_PENALTY_MIN
        src += [i, j]; dst += [j, i]; w += [t, t]
    # Staten Island Ferry.
    for a, b in [("S31", "142"), ("S31", "R27")]:
        src += [idx[a], idx[b]]; dst += [idx[b], idx[a]]; w += [FERRY_MIN, FERRY_MIN]

    n = len(ids)
    best = pd.DataFrame({"s": src, "d": dst, "w": w}).groupby(["s", "d"]).w.min().reset_index()
    g = csr_matrix((best.w, (best.s, best.d)), shape=(n, n))
    names = [stops[s][2] for s in ids]
    return ids, idx, lon, lat, x, y, names, g


def ride_minutes(g, idx, targets):
    """Scheduled minutes from every station to the nearest target station."""
    t = [idx[s] for s in targets if s in idx]
    d = dijkstra(g.T, directed=True, indices=t, min_only=True)   # reverse graph: from targets back
    return d


def main():
    ids, idx, lon, lat, sx, sy, names, g = subway_graph()
    ride_mid = ride_minutes(g, idx, MIDTOWN)
    ride_down = ride_minutes(g, idx, DOWNTOWN)
    print(f"{len(ids)} stations; median ride to Midtown {np.nanmedian(ride_mid[np.isfinite(ride_mid)]):.0f} min, "
          f"to Fulton St {np.nanmedian(ride_down[np.isfinite(ride_down)]):.0f} min")

    # Walk network + one extra node per destination linked to every station's street node
    # with weight = (wait + ride) in walking meters. One Dijkstra then gives door-to-door.
    wg = np.load(RAW / "walk_graph.npz")
    x, y, keep = wg["x"], wg["y"], wg["in_main"]
    n = len(x)
    walk = csr_matrix((wg["data"], wg["indices"], wg["indptr"]), shape=(n, n))
    keep_ids = np.flatnonzero(keep)
    tree = cKDTree(np.column_stack([x[keep], y[keep]]))
    s_snap, s_i = tree.query(np.column_stack([sx, sy]))
    s_node = keep_ids[s_i]

    hexes = load_hexes()
    hx, hy = to_utm(hexes.center_lon.values, hexes.center_lat.values)
    h_snap, h_i = tree.query(np.column_stack([hx, hy]))
    h_node = keep_ids[h_i]

    def door_to_door(ride):
        ok = np.isfinite(ride)
        minutes = np.where(ride[ok] > 0, ride[ok] + WAIT_MIN, 0)
        cost_m = minutes * WALK_M_PER_MIN + s_snap[ok]
        extra = coo_matrix((cost_m, (np.full(ok.sum(), n), s_node[ok])), shape=(n + 1, n + 1))
        # walk edges are stored both ways; the extra edges point out of node n only
        wc = walk.tocoo()
        big = coo_matrix((wc.data, (wc.row, wc.col)), shape=(n + 1, n + 1)).tocsr() + extra.tocsr()
        d = dijkstra(big, directed=True, indices=n)
        total = (d[h_node] + h_snap) / WALK_M_PER_MIN
        return np.minimum(total, CAP_MIN), total > CAP_MIN

    mid, mid_cap = door_to_door(ride_mid)
    down, down_cap = door_to_door(ride_down)

    def straight(pt):
        px, py = to_utm(*pt)
        return np.round(np.hypot(hx - px, hy - py) / 1000, 2)

    out = pd.DataFrame({"hex_id": hexes.hex_id})
    out["minutes_to_midtown"] = np.round(mid, 1)
    out["minutes_to_downtown"] = np.round(down, 1)
    out["midtown_capped"] = (mid_cap | down_cap).astype(int)
    out["km_to_midtown"] = straight(MIDTOWN_POINT)
    out["km_to_downtown"] = straight(DOWNTOWN_POINT)
    out["minutes_to_nearest_cbd"] = np.minimum(out.minutes_to_midtown, out.minutes_to_downtown)
    print(f"hexagons: median {out.minutes_to_midtown.median():.0f} min to Midtown, "
          f"{out.minutes_to_downtown.median():.0f} min to Fulton St; {out.midtown_capped.sum()} capped at {CAP_MIN}")

    # Ride time for each ridership station complex (for the training table).
    sc_path = DATA / "ridership" / "station_complexes.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path, comment="#")
        cx, cy = to_utm(sc.longitude.values, sc.latitude.values)
        # A complex can span several platforms: take the best one within 250 m of its point.
        st_tree = cKDTree(np.column_stack([sx, sy]))
        rm, rd = [], []
        for px, py in zip(cx, cy):
            near = st_tree.query_ball_point([px, py], 250) or [st_tree.query([px, py])[1]]
            rm.append(np.min(ride_mid[near])); rd.append(np.min(ride_down[near]))
        travel = pd.DataFrame({
            "station_complex_id": sc.station_complex_id,
            "station_complex": sc.station_complex,
            "ride_minutes_to_midtown": np.round(rm, 1),
            "ride_minutes_to_downtown": np.round(rd, 1),
            "km_to_midtown": np.round(np.hypot(cx - to_utm(*MIDTOWN_POINT)[0], cy - to_utm(*MIDTOWN_POINT)[1]) / 1000, 2),
            "km_to_downtown": np.round(np.hypot(cx - to_utm(*DOWNTOWN_POINT)[0], cy - to_utm(*DOWNTOWN_POINT)[1]) / 1000, 2),
        })
        write_features(
            travel, "midtown",
            "Scheduled subway ride time from each ridership station complex to Midtown and Lower Manhattan",
            sources=SOURCES, columns=STATION_COLUMNS, notes=NOTES,
            path=DATA / "ridership" / "station_travel_time.csv",
        )

    write_features(
        out, "midtown",
        "Door-to-door subway travel time to Midtown and Lower Manhattan per hexagon",
        sources=SOURCES,
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "minutes_to_midtown": f"minutes from the hexagon center to Times Sq-42 St or Grand Central-42 St: walk to "
                                  f"the best station + {WAIT_MIN} min wait + scheduled ride with transfers "
                                  f"(capped at {CAP_MIN}; walking only if that is faster)",
            "minutes_to_downtown": "same, to Fulton St (Lower Manhattan)",
            "midtown_capped": f"1 if either time was over {CAP_MIN} min (or impossible) and was capped",
            "km_to_midtown": "straight-line distance from the center to Midtown (between Times Sq and Grand Central), km",
            "km_to_downtown": "straight-line distance from the center to Fulton St, km",
            "minutes_to_nearest_cbd": "the smaller of minutes_to_midtown and minutes_to_downtown",
        },
        notes=NOTES,
    )


SOURCES = [
    ("MTA subway GTFS (scheduled station-to-station times, via scripts/build_data.py -> data/network.json)",
     "https://new.mta.info/developers"),
    ("Staten Island Ferry schedule (25-minute crossing)", "https://www.siferry.com/"),
    ("OpenStreetMap pedestrian network (from features_walk.py)",
     "https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf"),
]
STATION_COLUMNS = {
    "station_complex_id": "MTA station complex id (matches data/ridership/station_complexes.csv)",
    "station_complex": "station complex name",
    "ride_minutes_to_midtown": "scheduled minutes from the complex to Times Sq or Grand Central, with transfers, "
                               "no initial wait (0 = the complex is in Midtown)",
    "ride_minutes_to_downtown": "same, to Fulton St",
    "km_to_midtown": "straight-line distance to Midtown, km",
    "km_to_downtown": "straight-line distance to Fulton St, km",
}
NOTES = [
    f"Walking {WALK_M_PER_MIN} m/min; {WAIT_MIN} min average wait at the first station; transfers between platforms "
    f"within {TRANSFER_RADIUS_M} m cost the walk (x1.3) + {TRANSFER_PENALTY_MIN} min. Waits at transfers are not "
    "modelled beyond that, so lines with rare service look a little faster than they are.",
    "Ride times are scheduled weekday-style times between consecutive stations (median over trips), not live data.",
    "Staten Island includes the ferry (25 min + 15 min average wait) between St George and Whitehall St/South Ferry.",
    "For a hexagon with no station, this is the time using existing stations; a new station would make it shorter.",
    "Access to the subway is on foot only (no bus to the station), so places far from the subway (eastern "
    "Queens, southern Staten Island) get long times; that is the gap this project looks for, but real trips "
    "there often use a bus. bus.csv has the bus service to read alongside.",
]


if __name__ == "__main__":
    main()
