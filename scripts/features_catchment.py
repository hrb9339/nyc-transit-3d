"""What is within an 800 m walk of each station and each candidate site.

The first model described each station by the 500 m hexagon it sits in, so the
127 stations that share a hexagon look identical, and a station near a hexagon
edge is described by the wrong side of it. This describes every point by its
walking catchment instead: the area reachable within 800 m along streets and
paths (the walk network from scripts/features_walk.py), as a polygon.

Each per-hexagon feature is carried onto the catchment by area overlap:
  - counts (people, jobs, floor area, bus trips, destinations, ...) are summed in
    proportion to the share of each hexagon's land the catchment covers;
  - rates and shares (income, share renters, night light, distances, ...) are
    averaged, weighted by the overlap area;
  - densities are recomputed as catchment count / catchment land.
Students are counted exactly from the school and college points.

Points: subway station complexes (training rows, at the MTA complex location) and
hexagons without a station (candidates, at the hexagon center), so both are
described the same way.

Writes:  data/model/catchment_stations.csv    one row per station complex, ws_* columns
         data/model/catchment_candidates.csv  one row per candidate hexagon, ws_* columns
Usage:   python scripts/features_catchment.py   (after features_walk.py, features_students.py,
                                                 build_training_table.py)
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from features_common import DATA, RAW, UTM, load_hexes, write_features
from features_walk import local_dijkstra

WALK_M = 800
MODEL = DATA / "model"
to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform

# Per-hexagon features that add up over an area (summed by land share); everything else
# numeric is averaged by overlap area. Densities are recomputed from the sums.
EXTENSIVE = [
    "land_km2", "ntl_sum", "lit_pixels", "bright_pixels", "acs_population", "households", "workers",
    "transit_commuters", "no_vehicle_households", "bus_stops", "bus_stop_events_weekday", "bus_trips_weekday",
    "bus_trips_peak_per_hour", "express_trips_peak_per_hour", "population", "jobs", "hospitals", "clinics",
    "colleges", "museums", "libraries", "courthouses", "gov_offices", "ferry_landings", "shopping_centers",
    "shopping_center_floor_m2", "venues", "destinations", "lot_area_km2", "res_units", "buildings",
    "floor_area_m2", "res_floor_m2", "com_floor_m2", "office_floor_m2", "retail_floor_m2", "industrial_floor_m2",
]
DENSITIES = {   # recomputed: numerator / ws_land_km2
    "pop_density": "population", "job_density": "jobs", "bus_trips_per_km2": "bus_trips_weekday",
    "res_units_per_km2": "res_units", "ntl_density": "ntl_sum",
}
STUDENT_KINDS = {"students_k12_public": ["public", "charter"], "students_k12_private": ["private"],
                 "students_college": ["college"]}


def catchments(lon, lat, graph, x, y, tree, keep_ids):
    """Walking catchment polygon (UTM) for each point, or None if it is off the network."""
    px, py = to_utm(np.asarray(lon), np.asarray(lat))
    snap_d, snap_i = tree.query(np.column_stack([px, py]))
    polys = []
    for start, hop, cx, cy in zip(keep_ids[snap_i], snap_d, px, py):
        nodes = np.fromiter(local_dijkstra(graph, start, WALK_M - hop), np.int64) if hop < WALK_M else []
        if len(nodes) >= 3:
            pts = shapely.multipoints(np.column_stack([np.append(x[nodes], cx), np.append(y[nodes], cy)]))
            polys.append(shapely.concave_hull(pts, ratio=0.3).buffer(25))
        else:
            polys.append(shapely.Point(cx, cy).buffer(100))   # off-network: a small disc, flagged below
    return polys, snap_d


def describe(ids, polys, snap_d, hexes, feats, schools):
    ws = gpd.GeoDataFrame({"pid": ids}, geometry=polys, crs=UTM)
    ws["ws_area_km2"] = ws.area / 1e6
    hx = hexes.to_crs(UTM)
    hx["hex_m2"] = hx.area
    ov = gpd.overlay(ws[["pid", "geometry"]], hx[["hex_id", "hex_m2", "geometry"]], how="intersection",
                     keep_geom_type=True)
    ov["a"] = ov.area
    ov = ov.merge(feats, on="hex_id", how="left").copy()
    ov["share_of_hex"] = ov.a / ov.hex_m2                  # for counts
    ov["share_of_ws"] = ov.a / ov.groupby("pid").a.transform("sum")   # for rates

    numeric = [c for c in feats.columns if c != "hex_id"]
    ext = [c for c in EXTENSIVE if c in numeric]
    inten = [c for c in numeric if c not in ext and c not in DENSITIES]
    sums = ov[ext].mul(ov.share_of_hex, axis=0).groupby(ov.pid).sum()
    # Weighted means skip missing values: renormalize by the weight of the non-missing rows.
    vals = ov[inten]
    w = vals.notna().mul(ov.share_of_ws, axis=0)
    means = vals.fillna(0).mul(ov.share_of_ws, axis=0).groupby(ov.pid).sum() / w.groupby(ov.pid).sum()
    base = pd.concat([sums, means], axis=1).reindex(pd.Index(ids, name="pid"))
    land = base.land_km2.where(base.land_km2 > 0.01)
    extra = {d: base[num] / land for d, num in DENSITIES.items()}

    # Students: exact counts of school and college points inside the catchment.
    pts = gpd.GeoDataFrame(schools, geometry=gpd.points_from_xy(schools.longitude, schools.latitude),
                           crs="EPSG:4326").to_crs(UTM)
    hit = gpd.sjoin(pts[["kind", "students", "geometry"]], ws[["pid", "geometry"]], predicate="within")
    def per_point(rows, col=None):
        g = rows.groupby("pid")
        return (g[col].sum() if col else g.size()).reindex(base.index).fillna(0)

    for col, kinds in STUDENT_KINDS.items():
        extra[col] = per_point(hit[hit.kind.isin(kinds)], "students")
    extra["k12_schools"] = per_point(hit[hit.kind != "college"])
    extra["college_campuses"] = per_point(hit[hit.kind == "college"])
    extra["students"] = sum(extra[c] for c in STUDENT_KINDS)
    extra["student_density"] = extra["students"] / land
    extra["area_km2"] = ws.set_index("pid").ws_area_km2
    extra["snap_m"] = pd.Series(np.round(snap_d), index=base.index)
    out = pd.concat([base, pd.DataFrame(extra)], axis=1).add_prefix("ws_").copy()
    return out.round(4).reset_index()


def main():
    g = np.load(RAW / "walk_graph.npz")
    x, y, keep = g["x"], g["y"], g["in_main"]
    graph = csr_matrix((g["data"], g["indices"], g["indptr"]), shape=(len(x), len(x)))
    keep_ids = np.flatnonzero(keep)
    tree = cKDTree(np.column_stack([x[keep], y[keep]]))

    cols = pd.read_csv(MODEL / "feature_columns.csv")
    hex_feats = pd.read_csv(DATA / "features" / "hex_features.csv", comment="#")
    use = [c for c in cols[cols.role == "feature"].column if c in hex_feats.columns
           and c not in ("other_stations_1km", "nearest_other_station_m")]
    feats = hex_feats[["hex_id", *use]]
    hexes = load_hexes()
    schools = pd.read_csv(DATA / "students" / "schools.csv", comment="#")

    sc = pd.read_csv(DATA / "ridership" / "station_complexes.csv", comment="#", dtype={"station_complex_id": str})
    sc = sc[sc.transit_mode == "subway"]
    cand = pd.read_csv(MODEL / "candidates.csv", comment="#")[["hex_id"]].merge(
        hex_feats[["hex_id", "center_lon", "center_lat"]], on="hex_id")

    print(f"Walking catchments ({WALK_M} m): {len(sc)} stations, {len(cand)} candidate hexagons...")
    desc = dict(zip(cols.column, cols.meaning))
    out = {}
    for name, ids, lon, lat in [("stations", sc.station_complex_id, sc.longitude, sc.latitude),
                                ("candidates", cand.hex_id, cand.center_lon, cand.center_lat)]:
        polys, snap = catchments(lon.values, lat.values, graph, x, y, tree, keep_ids)
        out[name] = describe(ids.values, polys, snap, hexes, feats, schools)
        print(f"  {name}: median catchment {out[name].ws_area_km2.median():.2f} km², "
              f"{(out[name].ws_snap_m > 300).sum()} points more than 300 m from the network")

    columns = {"ws_area_km2": f"area reachable within an {WALK_M} m walk (concave hull of reached network "
                              "nodes, 25 m buffer), km²",
               "ws_snap_m": "distance from the point to the nearest walk-network node, m"}
    for c in out["stations"].columns:
        if c.startswith("ws_") and c not in columns:
            base = c[3:]
            how = ("sum over the catchment" if base in EXTENSIVE or base in STUDENT_KINDS or base in
                   ("students", "k12_schools", "college_campuses") else
                   "recomputed over the catchment" if base in DENSITIES or base == "student_density" else
                   "area-weighted mean over the catchment")
            columns[c] = f"{how}: {desc.get(base, base)}"
    sources = [("Walk network: data/raw/walk_graph.npz (scripts/features_walk.py, OpenStreetMap)",
                "https://download.bbbike.org/osm/bbbike/NewYork/"),
               ("Per-hexagon features: data/features/hex_features.csv", "data/features/hex_features.csv"),
               ("Schools and colleges: data/students/schools.csv", "data/students/schools.csv")]
    notes = ["Counts are carried from hexagons by land share, so they assume people/jobs/floor area are spread "
             "evenly within each 500 m hexagon; students are exact (school points).",
             "Rates and shares are overlap-weighted means of the hexagon values.",
             "Join on station_complex_id (stations) or hex_id (candidates)."]
    write_features(out["stations"].rename(columns={"pid": "station_complex_id"}), "catchment",
                   f"What is within an {WALK_M} m walk of each subway station complex",
                   sources, {"station_complex_id": "MTA station complex id", **columns}, notes=notes,
                   path=MODEL / "catchment_stations.csv")
    write_features(out["candidates"].rename(columns={"pid": "hex_id"}), "catchment",
                   f"What is within an {WALK_M} m walk of each candidate hexagon center",
                   sources, {"hex_id": "hexagon id", **columns}, notes=notes,
                   path=MODEL / "catchment_candidates.csv")


if __name__ == "__main__":
    main()
