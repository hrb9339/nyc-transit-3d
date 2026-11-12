"""Walking distance to the subway, along real streets and paths.

Straight-line distance makes places across a river, rail yard or expressway look
closer to a station than they are. This builds the pedestrian network from
OpenStreetMap and measures, for every hexagon center:
  - walking distance to the nearest subway entrance (not the station center:
    entrances are often a block or more away)
  - how much of the area within an 800 m walk is actually reachable
  - how many station complexes can be reached within an 800 m walk
  - street intersection density (a standard walkability measure)

Network: OpenStreetMap ways passing the same filter OSMnx uses for
network_type="walk" (no motorways, no private or foot=no ways), every way
walkable in both directions. Distances are along the network plus the
straight-line hop from the point to its nearest network node.

Writes:  data/features/walk.csv
         data/raw/walk_graph.npz   network + walking distance from every node to the
                                   nearest entrance (git-ignored; reused later for
                                   station catchments)
Usage:   python scripts/features_walk.py
"""

import heapq
import re
import urllib.parse

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from features_common import RAW, fetch, load_hexes, write_features

PBF = RAW / "NewYork.osm.pbf"
PBF_URL = "https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf"
ENTRANCES_CSV = "https://data.ny.gov/resource/i9wp-a4ja.csv"
ENTRANCES_PAGE = "https://data.ny.gov/Transportation/MTA-Subway-Entrances-and-Exits-2024/i9wp-a4ja"
WALK_M = 800
WALKSHED_CIRCLE_KM2 = np.pi * (WALK_M / 1000) ** 2
WALK_CAP_M = 5_000  # an hour+ on foot: not a realistic walk to the subway
MIN_COMPONENT_NODES = 5_000  # drop tiny disconnected bits (isolated paths, rooftops, piers)

# OSMnx network_type="walk" filter (osmnx/_overpass.py), applied to OSM tags.
EXCLUDED_HIGHWAY = re.compile(
    r"abandoned|bus_guideway|construction|corridor|cycleway|elevator|escalator|motor|motorway|motorway_link"
    r"|^no$|planned|platform|proposed|raceway|razed|rest_area|services")

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform


def walkable(tags):
    hw = tags.get("highway")
    if hw is None or tags.get("area") == "yes":
        return False
    if EXCLUDED_HIGHWAY.search(hw):
        return False
    if "private" in tags.get("access", "") or "no" == tags.get("foot") or "private" in tags.get("service", ""):
        return False
    return True


def build_network():
    """Pedestrian graph: node coords (UTM, m) and an undirected CSR matrix of edge lengths."""
    node_index, xs, ys, src, dst = {}, [], [], [], []
    fp = osmium.FileProcessor(str(PBF), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for w in fp:
        if not w.is_way() or not walkable(w.tags):
            continue
        prev = None
        for n in w.nodes:
            if not n.location.valid():
                prev = None
                continue
            i = node_index.get(n.ref)
            if i is None:
                i = node_index[n.ref] = len(xs)
                xs.append(n.location.lon)
                ys.append(n.location.lat)
            if prev is not None and prev != i:
                src.append(prev)
                dst.append(i)
            prev = i
    x, y = to_utm(np.array(xs), np.array(ys))
    src, dst = np.array(src), np.array(dst)
    length = np.hypot(x[src] - x[dst], y[src] - y[dst])
    n = len(x)
    graph = csr_matrix((np.r_[length, length], (np.r_[src, dst], np.r_[dst, src])), shape=(n, n))
    graph.sum_duplicates()
    return x, y, graph


def main_components(graph):
    """Nodes in sizable connected pieces. Staten Island is its own piece: it connects to the
    rest of the city only by expressway bridges, which pedestrians can't use."""
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return sizes[labels] >= MIN_COMPONENT_NODES, int((sizes >= MIN_COMPONENT_NODES).sum())


def load_entrances():
    query = urllib.parse.urlencode({"$limit": 10_000})
    fetch(f"{ENTRANCES_CSV}?{query}", RAW / "subway_entrances.csv")
    e = pd.read_csv(RAW / "subway_entrances.csv")
    return e[e.entry_allowed.str.upper().eq("YES")].dropna(subset=["entrance_latitude", "entrance_longitude"])


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


def main():
    if not PBF.exists():
        raise SystemExit(f"download the NYC OpenStreetMap extract first:\n  curl -o {PBF} {PBF_URL}")
    print("Building the walking network...")
    x, y, graph = build_network()
    keep, pieces = main_components(graph)
    print(f"  {graph.shape[0]:,} nodes, {graph.nnz // 2:,} segments, {graph.data.sum() / 2 / 1000:,.0f} km; "
          f"{pieces} connected pieces of {MIN_COMPONENT_NODES:,}+ nodes hold {keep.sum():,} nodes")
    tree = cKDTree(np.column_stack([x[keep], y[keep]]))
    keep_ids = np.flatnonzero(keep)

    def snap(lon, lat):
        px, py = to_utm(np.asarray(lon), np.asarray(lat))
        d, i = tree.query(np.column_stack([px, py]))
        return keep_ids[i], d, px, py

    # Walking distance from every node to the nearest entrance (one multi-source pass).
    ent = load_entrances()
    ent_node, ent_snap, ex, ey = snap(ent.entrance_longitude, ent.entrance_latitude)
    print(f"  {len(ent):,} entrances (entry allowed), median snap {np.median(ent_snap):.0f} m")
    to_entrance = dijkstra(graph, directed=False, indices=np.unique(ent_node), min_only=True)

    hexes = load_hexes()
    h_node, h_snap, hx, hy = snap(hexes.center_lon, hexes.center_lat)
    straight = cKDTree(np.column_stack([ex, ey])).query(np.column_stack([hx, hy]))[0]
    walk_raw = to_entrance[h_node] + h_snap
    capped = walk_raw > WALK_CAP_M
    walk_m = np.minimum(walk_raw, WALK_CAP_M)

    # Within an 800 m walk of each center: reachable area and station complexes.
    node_complexes = pd.Series(ent.complex_id.values, index=ent_node).groupby(level=0).agg(set)
    shed_km2, complexes, street_km = [], [], []
    for start, hop in zip(h_node, h_snap):
        reach = local_dijkstra(graph, start, WALK_M - hop) if hop < WALK_M else {}
        nodes = np.fromiter(reach, dtype=np.int64)
        if len(nodes) >= 3:
            pts = shapely.multipoints(np.column_stack([x[nodes], y[nodes]]))
            shed_km2.append(shapely.concave_hull(pts, ratio=0.3).buffer(25).area / 1e6)
        else:
            shed_km2.append(0.0)
        found = set().union(*(node_complexes.get(n, set()) for n in nodes)) if len(nodes) else set()
        complexes.append(len(found))
        sub = graph[nodes][:, nodes] if len(nodes) else None
        street_km.append(sub.data.sum() / 2 / 1000 if sub is not None else 0.0)

    # Intersection density: nodes where 3+ segments meet, per km² of hexagon land.
    degree = np.diff(graph.indptr)
    inter = keep & (degree >= 3)
    hex_utm = hexes.to_crs("EPSG:32618")
    ipts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x[inter], y[inter]), crs="EPSG:32618")
    counts = gpd.sjoin(ipts, hex_utm[["hex_id", "geometry"]], predicate="within").groupby("hex_id").size()

    out = pd.DataFrame({"hex_id": hexes.hex_id})
    out["walk_to_entrance_m"] = np.round(walk_m).astype(int)
    out["straight_to_entrance_m"] = np.round(straight).astype(int)
    out["walk_capped"] = capped.astype(int)
    out["detour_ratio"] = np.round(np.minimum(walk_raw / np.maximum(straight, 50), 10), 2)
    out["center_snap_m"] = np.round(h_snap).astype(int)
    out["walk_within_400"] = (walk_m <= 400).astype(int)
    out["walk_within_800"] = (walk_m <= 800).astype(int)
    out["complexes_800m_walk"] = complexes
    out["walkshed_800_km2"] = np.round(shed_km2, 3)
    out["walkshed_ratio"] = np.round(np.array(shed_km2) / WALKSHED_CIRCLE_KM2, 3)
    out["street_km_800m_walk"] = np.round(street_km, 2)
    out["intersections_per_km2"] = (counts.reindex(hexes.hex_id).fillna(0).values / hexes.land_km2.values).round(0)

    print(f"Median walk {np.median(walk_m):.0f} m vs straight-line {np.median(straight):.0f} m; "
          f"median detour {out.detour_ratio.median():.2f}; "
          f"{out.walk_within_800.sum()} hexagon centers within an 800 m walk of an entrance, "
          f"{capped.sum()} capped at {WALK_CAP_M:,} m")

    np.savez_compressed(RAW / "walk_graph.npz", x=x, y=y, indptr=graph.indptr, indices=graph.indices,
                        data=graph.data, to_entrance=to_entrance, in_main=keep)

    write_features(
        out, "walk",
        "Walking access to the subway per hexagon (street network)",
        sources=[
            ("OpenStreetMap contributors, New York extract from BBBike (ODbL), pedestrian network", PBF_URL),
            ("MTA Subway Entrances and Exits: 2024 (data.ny.gov i9wp-a4ja), entrances with entry allowed", ENTRANCES_PAGE),
            ("OSMnx documentation (walk network filter used here)", "https://osmnx.readthedocs.io/"),
        ],
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "walk_to_entrance_m": f"walking distance from the hexagon center to the nearest subway entrance, m "
                                  f"(capped at {WALK_CAP_M:,})",
            "straight_to_entrance_m": "straight-line distance from the center to the nearest entrance, m",
            "walk_capped": f"1 if the real walk is over {WALK_CAP_M:,} m (or impossible) and was capped",
            "detour_ratio": "walk / straight-line (1.0 = direct; high = rivers, yards or highways in the way); "
                            "uncapped walk, straight-line floored at 50 m, ratio capped at 10",
            "center_snap_m": "distance from the center to the nearest walkable path (large = center in a park, "
                             "airport or marsh)",
            "walk_within_400": "1 if the center is within a 400 m walk of an entrance (about 5 minutes)",
            "walk_within_800": "1 if the center is within an 800 m walk of an entrance (about 10 minutes)",
            "complexes_800m_walk": "number of station complexes with an entrance within an 800 m walk",
            "walkshed_800_km2": "area reachable within an 800 m walk (concave hull of reachable nodes + 25 m), km²",
            "walkshed_ratio": f"walkshed area / area of an 800 m circle ({WALKSHED_CIRCLE_KM2:.2f} km²); "
                              "low = poor street grid or water",
            "street_km_800m_walk": "km of walkable streets and paths within an 800 m walk",
            "intersections_per_km2": "street/path intersections (3+ segments) per km² of hexagon land",
        },
        notes=[
            "Walking speed is not assumed; divide by ~80 m/min for minutes.",
            f"Network pieces under {MIN_COMPONENT_NODES:,} nodes are dropped; centers are snapped to the nearest "
            "remaining node and the snap distance is added.",
            "OSM maps NYC sidewalks as separate paths in many places, so intersection density counts "
            "sidewalk crossings too; compare hexagons with each other, not with other cities.",
        ],
    )


if __name__ == "__main__":
    main()
