"""Build the model tables from the per-hexagon features and station ridership.

Training rows are station complexes (what has ridership), not hexagons. Each
station gets the features of the hexagon it sits in, plus a few station-level
columns. Candidate rows are hexagons with no station: the model predicts the
riders a new station at the hexagon center would get.

A column is only a model input if it means the same thing for an existing
station and for an imagined new one. Columns measuring distance to the
existing subway (walk to the nearest entrance, door-to-door time, ...) are near
zero for every training row and large for every candidate, so the model would
learn nothing useful from them and extrapolate badly; they are dropped and
replaced by "competition" columns that work for both: how many other stations
are close by, and how far the nearest one is.

Writes:  data/model/training.csv        one row per subway station complex (target: weekday_riders)
         data/model/candidates.csv      one row per hexagon without a station
         data/model/feature_columns.csv every column: role (feature/target/id/group/info/excluded) and why
Usage:   python scripts/build_training_table.py   (after features_merge.py and features_midtown.py)
"""

import heapq

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from features_common import DATA, RAW, write_features

MODEL = DATA / "model"
WALK_M = 800
COMPETITION_M = 1000
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform

# Columns that are not model inputs, with the reason.
EXCLUDED = {
    "ids": (["hex_id", "center_lon", "center_lat", "ntaname", "cdtaname", "cdta2020", "nta_share", "nta_count"],
            "identifier or bookkeeping, not a property of the place"),
    "target": (["ridership_complexes", "weekday_riders", "saturday_riders", "sunday_riders", "weekday_riders_2019",
                "recovery_vs_2019"], "ridership: this is the answer"),
    "validation": (["dist_ibx_m", "near_ibx", "dist_sas2_m", "near_sas2", "dist_utica_m", "near_utica",
                    "near_any_proposal"], "proposed lines are kept for checking results, not for training"),
    "existing_subway": (["stations", "nearest_station_m", "walk_to_entrance_m", "straight_to_entrance_m",
                         "walk_capped", "detour_ratio", "walk_within_400", "walk_within_800", "complexes_800m_walk",
                         "walk_to_any_rail_m", "any_rail_within_800", "minutes_to_midtown", "minutes_to_downtown",
                         "midtown_capped", "minutes_to_nearest_cbd"],
                        "distance to the existing subway: ~0 for every station, large for every candidate; "
                        "replaced by other_stations_1km and nearest_other_station_m"),
    "leaky": (["share_commute_subway", "share_commute_transit"],
              "partly caused by having a subway station nearby (leaks the answer); share_no_vehicle is the safer signal"),
    "text": (["nearest_other_rail", "nearest_other_rail_system"], "station name / category text"),
}
GROUP_COLS = ["nta2020", "borough"]   # for neighborhood holdout cross-validation


def local_dijkstra(graph, start, limit):
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


def column_descriptions():
    """'name: meaning' lines from the merged file's header."""
    desc = {}
    for line in open(DATA / "features" / "hex_features.csv", encoding="utf-8"):
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        if ": " in body and body.split(": ")[0].isidentifier():
            name, meaning = body.split(": ", 1)
            desc.setdefault(name, meaning)
    return desc


def competition(px, py, sx, sy, exclude_self):
    """Other station complexes within 1 km, and distance to the nearest other one."""
    tree = cKDTree(np.column_stack([sx, sy]))
    k = 2 if exclude_self else 1
    d, _ = tree.query(np.column_stack([px, py]), k=k)
    nearest = d[:, 1] if exclude_self else d
    counts = np.array([len(c) for c in tree.query_ball_point(np.column_stack([px, py]), COMPETITION_M)])
    return counts - (1 if exclude_self else 0), np.round(nearest)


def main():
    hexes = pd.read_csv(DATA / "features" / "hex_features.csv", comment="#")
    sc = pd.read_csv(DATA / "ridership" / "station_complexes.csv", comment="#")
    tt = pd.read_csv(DATA / "ridership" / "station_travel_time.csv", comment="#")
    desc = column_descriptions()

    excluded = {c: reason for cols, reason in EXCLUDED.values() for c in cols}
    features = [c for c in hexes.columns if c not in excluded and c not in GROUP_COLS]
    features.append("borough_code")   # borough as a number-free one-hot below
    hexes = hexes.assign(borough_code=hexes.borough)

    # ---- training rows: subway station complexes (SIR left out: fares only at 2 stations)
    sc = sc[sc.transit_mode == "subway"].copy()
    sx, sy = to_utm(sc.longitude.values, sc.latitude.values)
    sc["other_stations_1km"], sc["nearest_other_station_m"] = competition(sx, sy, sx, sy, exclude_self=True)
    sc["complexes_in_hex"] = sc.groupby("hex_id").station_complex_id.transform("size")
    sc = sc.merge(tt[["station_complex_id", "ride_minutes_to_midtown", "ride_minutes_to_downtown"]],
                  on="station_complex_id", how="left")

    station_cols = ["other_stations_1km", "nearest_other_station_m"]
    train = sc[["station_complex_id", "station_complex", "hex_id", "weekday_riders", "weekday_riders_median",
                "saturday_riders", "sunday_riders", "weekday_riders_2019", "days_with_data", "complexes_in_hex",
                "ride_minutes_to_midtown", "ride_minutes_to_downtown", *station_cols]].merge(
        hexes[["hex_id", *GROUP_COLS, *features]], on="hex_id", how="left")
    train.insert(4, "log_weekday_riders", np.round(np.log1p(train.weekday_riders), 4))
    train = pd.get_dummies(train, columns=["borough_code"], prefix="boro", dtype=int)

    # ---- candidate rows: hexagons with no station complex
    has = set(sc.hex_id)
    cand = hexes[~hexes.hex_id.isin(has) & (hexes.stations == 0)].copy()
    cx, cy = to_utm(cand.center_lon.values, cand.center_lat.values)
    cand["other_stations_1km"], cand["nearest_other_station_m"] = competition(cx, cy, sx, sy, exclude_self=False)

    # Share of the 800 m walking area around the center not already within an 800 m walk of
    # the subway (or of LIRR/Metro-North/PATH): the "new coverage" a station here would add.
    g = np.load(RAW / "walk_graph.npz")
    x, y, keep = g["x"], g["y"], g["in_main"]
    graph = csr_matrix((g["data"], g["indices"], g["indptr"]), shape=(len(x), len(x)))
    keep_ids = np.flatnonzero(keep)
    tree = cKDTree(np.column_stack([x[keep], y[keep]]))
    rail = pd.read_csv(RAW / "other_rail_stations.csv")
    rx, ry = to_utm(rail.lon.values, rail.lat.values)
    r_snap, r_i = tree.query(np.column_stack([rx, ry]))
    to_rail = dijkstra(graph, directed=False, indices=np.unique(keep_ids[r_i][r_snap < 300]), min_only=True)
    covered_subway = g["to_entrance"] <= WALK_M
    covered_any = covered_subway | (to_rail <= WALK_M)
    snap_d, snap_i = tree.query(np.column_stack([cx, cy]))
    new_sub, new_any = [], []
    for start, hop in zip(keep_ids[snap_i], snap_d):
        nodes = np.fromiter(local_dijkstra(graph, start, WALK_M - hop), np.int64) if hop < WALK_M else np.array([], int)
        new_sub.append(1 - covered_subway[nodes].mean() if len(nodes) else np.nan)
        new_any.append(1 - covered_any[nodes].mean() if len(nodes) else np.nan)
    cand["uncovered_share_subway"] = np.round(new_sub, 3)
    cand["uncovered_share_any_rail"] = np.round(new_any, 3)

    cand_cols = ["hex_id", *GROUP_COLS, "ntaname", "uncovered_share_subway", "uncovered_share_any_rail",
                 *station_cols, *features,
                 *EXCLUDED["validation"][0], "walk_to_entrance_m", "walk_to_any_rail_m"]
    cand = cand[cand_cols]
    cand = pd.get_dummies(cand, columns=["borough_code"], prefix="boro", dtype=int)
    boro = [c for c in train.columns if c.startswith("boro_")]
    for c in boro:
        if c not in cand:
            cand[c] = 0

    model_features = [c for c in train.columns if c in set(features) | set(station_cols) or c.startswith("boro_")]
    print(f"training: {len(train)} stations; candidates: {len(cand)} hexagons; {len(model_features)} model features")
    print(f"  candidates with >50% of their walk area uncovered by any rail: {(cand.uncovered_share_any_rail > 0.5).sum()}")

    # ---- column guide
    own = {
        "station_complex_id": ("id", "MTA station complex id"),
        "station_complex": ("id", "station complex name"),
        "weekday_riders": ("target", "mean weekday entries, 2025 (the thing to predict)"),
        "log_weekday_riders": ("target", "log(1 + weekday_riders): train on this, ridership is very skewed"),
        "weekday_riders_median": ("info", "median weekday entries, 2025"),
        "saturday_riders": ("info", "mean Saturday entries, 2025"),
        "sunday_riders": ("info", "mean Sunday entries, 2025"),
        "weekday_riders_2019": ("info", "mean weekday entries, 2019 (pre-COVID check)"),
        "days_with_data": ("info", "days in 2025 with ridership (Rockaway stations were closed for months)"),
        "complexes_in_hex": ("info", "station complexes sharing this hexagon (they share its features)"),
        "ride_minutes_to_midtown": ("info", "scheduled ride to Times Sq/Grand Central; not known for a new "
                                            "station until its line is chosen, so not a default feature"),
        "ride_minutes_to_downtown": ("info", "scheduled ride to Fulton St; same caveat"),
        "other_stations_1km": ("feature", f"other station complexes within {COMPETITION_M} m (competition)"),
        "nearest_other_station_m": ("feature", "straight-line distance to the nearest other station complex, m"),
        "uncovered_share_subway": ("ranking", f"candidates: share of the {WALK_M} m walk area around the center "
                                              f"not within {WALK_M} m of a subway entrance"),
        "uncovered_share_any_rail": ("ranking", "same, counting LIRR/Metro-North/PATH as covered too: "
                                                "new riders ~ predicted riders x this"),
        "nta2020": ("group", "2020 neighborhood code: hold out whole neighborhoods in cross-validation"),
        "borough": ("group", "borough name"),
        "ntaname": ("info", "neighborhood name"),
    }
    guide = []
    for c in dict.fromkeys([*train.columns, *cand.columns]):
        if c in own:
            role, meaning = own[c]
        elif c.startswith("boro_"):
            role = "feature" if c in train.columns else "info"
            meaning = f"1 if the hexagon is in {c[5:]}" + ("" if c in train.columns else
                      " (no training stations there, so not a feature; see notes)")
        elif c in excluded:
            role, meaning = "excluded", f"{desc.get(c, '')} [not an input: {excluded[c]}]"
        else:
            role, meaning = "feature", desc.get(c, "")
        guide.append({"column": c, "role": role, "in_training": int(c in train.columns),
                      "in_candidates": int(c in cand.columns), "meaning": meaning})
    guide = pd.DataFrame(guide)
    extra = set(guide[guide.role == "feature"].column) ^ set(model_features)
    assert not extra, extra

    MODEL.mkdir(exist_ok=True)
    srcs = [("Per-hexagon features (see each table's header for its own sources)", "data/features/hex_features.csv"),
            ("Station ridership", "data/ridership/station_complexes.csv"),
            ("Station ride times", "data/ridership/station_travel_time.csv")]
    notes = ["Model inputs = rows with role 'feature' in data/model/feature_columns.csv "
             f"({len(model_features)} columns). Load that list rather than typing column names.",
             "Each station takes the features of the hexagon it is in; stations sharing a hexagon share features."]
    gd = dict(zip(guide.column, guide.meaning))
    write_features(train, "training", "Model training table: one row per subway station complex",
                   srcs, {c: gd[c] for c in train.columns},
                   notes=notes + ["Staten Island Railway stations are left out (fares are collected at only 2 of them)."],
                   path=MODEL / "training.csv")
    write_features(cand, "candidates", "Candidate locations: hexagons with no subway station",
                   srcs + [("Walking network and other-rail stations", "data/raw/walk_graph.npz, other_rail_stations.csv")],
                   {c: gd[c] for c in cand.columns},
                   notes=notes + ["Excluded columns (proposed lines, walk to the existing subway) are kept here for "
                                  "checking and mapping results; do not feed them to the model.",
                                  "Includes parks, airports and cemeteries: filter with nonresidential_nta / "
                                  "share_unbuilt_unlivable before ranking.",
                                  "Staten Island has no subway stations in the training data, so predictions there "
                                  "are extrapolation: report them separately or leave them out."],
                   path=MODEL / "candidates.csv")
    guide.to_csv(MODEL / "feature_columns.csv", index=False)
    print(f"wrote data/model/feature_columns.csv: {guide.role.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
