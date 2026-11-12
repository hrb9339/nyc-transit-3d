"""Major destinations per hexagon: places people travel to.

Counts of hospitals, colleges, museums, shopping centers, stadiums, arenas and other trip
attractors in each hexagon, plus straight-line distance from the hexagon center to
the nearest hospital and college. Residents and job counts don't fully capture
these (a hospital or a stadium draws far more visitors than it has workers).

Sources: NYC Facilities Database (most destinations) and PLUTO building classes
(shopping centers, and the tax lots of major stadiums and arenas). Airports are covered by data/features/landuse.csv.

Writes:  data/features/destinations.csv
Usage:   python scripts/features_destinations.py
"""

import urllib.parse

import geopandas as gpd
import pandas as pd

from features_common import RAW, UTM, fetch, hex_centers, load_hexes, write_features

FACDB_CSV = "https://data.cityofnewyork.us/resource/ji82-xba5.csv"
FACDB_PAGE = "https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5"
PLUTO_PAGE = "https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks"
SQFT_TO_M2 = 0.09290304

# column -> (description, filter on the Facilities Database)
FACILITIES = {
    "hospitals": ("hospitals (factype HOSPITAL or ACUTE CARE HOSPITAL)",
                  lambda f: f.factype.isin(["HOSPITAL", "ACUTE CARE HOSPITAL"])),
    "clinics": ("clinics and diagnostic/treatment centers (other HOSPITALS AND CLINICS, excluding mobile and school-based)",
                lambda f: f.facsubgrp.eq("HOSPITALS AND CLINICS")
                & ~f.factype.isin(["HOSPITAL", "ACUTE CARE HOSPITAL"])
                & ~f.factype.str.contains("MOBILE|SCHOOL BASED", na=False)),
    "colleges": ("colleges and universities (facsubgrp COLLEGES OR UNIVERSITIES)",
                 lambda f: f.facsubgrp.eq("COLLEGES OR UNIVERSITIES")),
    "museums": ("museums (facsubgrp MUSEUMS)", lambda f: f.facsubgrp.eq("MUSEUMS")),
    "libraries": ("public libraries (facsubgrp PUBLIC LIBRARIES)", lambda f: f.facsubgrp.eq("PUBLIC LIBRARIES")),
    "courthouses": ("courthouses (facsubgrp COURTHOUSES AND JUDICIAL)",
                    lambda f: f.facsubgrp.eq("COURTHOUSES AND JUDICIAL")),
    "gov_offices": ("city government offices (facsubgrp CITY GOVERNMENT OFFICES)",
                    lambda f: f.facsubgrp.eq("CITY GOVERNMENT OFFICES")),
    "ferry_landings": ("ferry landings and ports (facsubgrp PORTS AND FERRY LANDINGS)",
                       lambda f: f.facsubgrp.eq("PORTS AND FERRY LANDINGS")),
}
SHOPPING_CENTER = "K6"  # PLUTO building class
# Stadiums and arenas with roughly 5,000+ seats, located by their PLUTO tax lot (BBL).
# Building class Q6 alone is not usable: it also covers small park lots and school ballfields.
VENUES = {
    "2024930001": "Yankee Stadium",
    "4017870020": "Citi Field",
    "1007810001": "Madison Square Garden (condo lot has no coordinates; uses Penn Station lot on the same block)",
    "3011180001": "Barclays Center",
    "4115430002": "Aqueduct Racetrack",
    "1022440001": "Columbia Baker Athletics Complex",
    "4032190001": "Forest Hills Stadium",
    "5000020020": "Staten Island ballpark (St. George)",
    "3070730101": "Maimonides Park (Coney Island)",
}


def count_in_hexes(points, hexes, name):
    joined = gpd.sjoin(points, hexes[["hex_id", "geometry"]], how="inner", predicate="within")
    return joined.groupby("hex_id").size().rename(name)


def nearest_m(points, centers):
    """Straight-line distance from each hexagon center to the nearest point, meters."""
    joined = gpd.sjoin_nearest(centers, points[["geometry"]], how="left", distance_col="d")
    return joined.groupby("hex_id").d.min().round(0)


def main():
    hexes = load_hexes().to_crs(UTM)
    centers = hex_centers(load_hexes()).to_crs(UTM)

    query = urllib.parse.urlencode({
        "$select": "uid,facname,facgroup,facsubgrp,factype,latitude,longitude", "$limit": 100_000,
    })
    fetch(f"{FACDB_CSV}?{query}", RAW / "facilities.csv")
    fac = pd.read_csv(RAW / "facilities.csv").dropna(subset=["latitude", "longitude"])
    fac = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.longitude, fac.latitude), crs="EPSG:4326").to_crs(UTM)
    print(f"Facilities Database: {len(fac):,} facilities with a location")

    out = pd.DataFrame(index=hexes.hex_id)
    selected = {}
    for col, (_, rule) in FACILITIES.items():
        selected[col] = fac[rule(fac)]
        out[col] = count_in_hexes(selected[col], hexes, col)

    if not (RAW / "pluto.csv").exists():
        raise SystemExit("run scripts/features_landuse.py first (it downloads data/raw/pluto.csv)")
    lots = pd.read_csv(RAW / "pluto.csv", dtype={"bbl": str, "bldgclass": str},
                       usecols=["bbl", "bldgclass", "lotarea", "bldgarea", "latitude", "longitude"])
    lots["bbl"] = lots.bbl.str.split(".").str[0]
    lots = lots[lots.bldgclass.eq(SHOPPING_CENTER) | lots.bbl.isin(VENUES)].dropna(subset=["latitude", "longitude"])
    missing = set(VENUES) - set(lots.bbl)
    if missing:
        raise SystemExit(f"venue lots not found in PLUTO: {missing}")
    lots = gpd.GeoDataFrame(lots, geometry=gpd.points_from_xy(lots.longitude, lots.latitude), crs="EPSG:4326").to_crs(UTM)
    malls = lots[lots.bldgclass.eq(SHOPPING_CENTER)]
    out["shopping_centers"] = count_in_hexes(malls, hexes, "shopping_centers")
    mall_floor = gpd.sjoin(malls, hexes[["hex_id", "geometry"]], predicate="within").groupby("hex_id").bldgarea.sum()
    out["shopping_center_floor_m2"] = (mall_floor * SQFT_TO_M2).round(0)
    out["venues"] = count_in_hexes(lots[lots.bbl.isin(VENUES)], hexes, "venues")

    count_cols = [*FACILITIES, "shopping_centers", "venues"]
    out = out.fillna(0)
    out[count_cols + ["shopping_center_floor_m2"]] = out[count_cols + ["shopping_center_floor_m2"]].astype(int)
    out["destinations"] = out[["hospitals", "colleges", "museums", "courthouses", "shopping_centers", "venues"]].sum(axis=1)
    out["nearest_hospital_m"] = nearest_m(selected["hospitals"], centers).reindex(out.index).astype(int)
    out["nearest_college_m"] = nearest_m(selected["colleges"], centers).reindex(out.index).astype(int)
    out = out.reset_index()

    totals = ", ".join(f"{c} {out[c].sum()}" for c in count_cols)
    print(f"In hexagons: {totals}; {(out.destinations > 0).sum()} hexagons with a major destination")

    columns = {"hex_id": "hexagon id (matches data/hexgrid.geojson)"}
    columns |= {col: f"number of {desc}" for col, (desc, _) in FACILITIES.items()}
    columns |= {
        "shopping_centers": "number of shopping-center lots (PLUTO building class K6)",
        "shopping_center_floor_m2": "floor area of those shopping centers, m²",
        "venues": "number of major stadiums and arenas (~5,000+ seats): " + "; ".join(v.split(" (")[0] for v in VENUES.values()),
        "destinations": "major trip attractors: hospitals + colleges + museums + courthouses + shopping centers + venues",
        "nearest_hospital_m": "straight-line distance from the hexagon center to the nearest hospital, m",
        "nearest_college_m": "straight-line distance from the hexagon center to the nearest college, m",
    }
    write_features(
        out, "destinations",
        "Major destinations (trip attractors) per hexagon",
        sources=[
            ("NYC Department of City Planning, Facilities Database (NYC Open Data ji82-xba5)", FACDB_PAGE),
            ("NYC Department of City Planning, PLUTO 26v2, building class K6 and venue tax lots (NYC Open Data 64uk-42ks)", PLUTO_PAGE),
            ("PLUTO data dictionary, building class codes (linked from the PLUTO page)",
             "https://www.nyc.gov/content/planning/pages/resources/datasets/mappluto-pluto-change"),
        ],
        columns=columns,
        notes=[
            "Counted by point location (facility address or lot center) inside the hexagon.",
            "K-12 schools are not counted here; school enrollment is collected separately.",
            "Airports are in data/features/landuse.csv (share_airport).",
            "The USTA tennis center (US Open) is inside the Flushing Meadows park tax lot and is not listed separately.",
        ],
    )


if __name__ == "__main__":
    main()
