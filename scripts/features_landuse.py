"""Land use and buildings per hexagon, from the city's tax-lot database (PLUTO).

Shows how much of each hexagon is land nobody lives on (parks, cemeteries,
airports, rail yards) and how much floor space and housing it has.

Lots are assigned to hexagons in two ways:
  - Ordinary lots (under LARGE_LOT_SQFT) go to the hexagon containing their center point.
    They are tiny next to a 0.65 km² hexagon, so the error is negligible.
  - Large lots (airports, cemeteries, big parks, rail yards, housing campuses) are cut
    by their real outlines, and their area, floor space and units are split between
    hexagons by the share of the lot inside each.

Writes:  data/features/landuse.csv
Usage:   python scripts/features_landuse.py
"""

import json
import urllib.parse

import geopandas as gpd
import numpy as np
import pandas as pd

from features_common import RAW, UTM, fetch, load_hexes, write_features

SQFT_TO_M2 = 0.09290304
LARGE_LOT_SQFT = 100_000  # ~0.93 ha
PLUTO_CSV = "https://data.cityofnewyork.us/resource/64uk-42ks.csv"
PLUTO_PAGE = "https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks"
MAPPLUTO_QUERY = "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/MAPPLUTO/FeatureServer/0/query"
MAPPLUTO_PAGE = "https://www.nyc.gov/content/planning/pages/resources/datasets/mappluto-pluto-change"
FIELDS = ["bbl", "landuse", "bldgclass", "lotarea", "bldgarea", "resarea", "comarea", "officearea",
          "retailarea", "factryarea", "unitsres", "numbldgs", "yearbuilt", "latitude", "longitude", "version"]

# PLUTO land use codes
LANDUSE = {
    "residential": ["01", "02", "03"],   # 1-2 family, walk-up, elevator buildings
    "mixed": ["04"],                      # residential + commercial
    "commercial": ["05"],                 # commercial and office
    "industrial": ["06"],                 # industrial and manufacturing
    "transport_utility": ["07"],          # airports, rail yards, depots, utilities
    "institution": ["08"],                # schools, hospitals, government, religious
    "open_space": ["09"],                 # parks, beaches, outdoor recreation
    "parking": ["10"],
    "vacant": ["11"],
}
CEMETERY, AIRPORT = "Z8", "T1"           # building classes


def load_lots():
    query = urllib.parse.urlencode({"$select": ",".join(FIELDS), "$limit": 2_000_000, "$order": "bbl"})
    fetch(f"{PLUTO_CSV}?{query}", RAW / "pluto.csv")
    lots = pd.read_csv(RAW / "pluto.csv", dtype={"bbl": str, "landuse": str, "bldgclass": str})
    lots["bbl"] = lots.bbl.str.split(".").str[0]
    lots["landuse"] = lots.landuse.str.split(".").str[0].str.zfill(2)
    num = ["lotarea", "bldgarea", "resarea", "comarea", "officearea", "retailarea", "factryarea", "unitsres", "numbldgs"]
    lots[num] = lots[num].fillna(0)
    return lots


def load_large_outlines():
    """Outlines of large lots from the Department of City Planning's MapPLUTO service."""
    path = RAW / "mappluto_large_lots.geojson"
    if not path.exists():
        features, offset = [], 0
        while True:
            params = urllib.parse.urlencode({
                "where": f"LotArea >= {LARGE_LOT_SQFT}", "outFields": "BBL", "outSR": 4326,
                "resultOffset": offset, "resultRecordCount": 2000, "orderByFields": "OBJECTID", "f": "geojson",
            })
            page = json.loads(fetch(f"{MAPPLUTO_QUERY}?{params}"))
            features += page["features"]
            print(f"  large lot outlines: {len(features)}")
            if len(page["features"]) < 2000:
                break
            offset += 2000
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    outlines = gpd.read_file(path)
    outlines["bbl"] = outlines.BBL.astype("int64").astype(str)
    return outlines[["bbl", "geometry"]]


def main():
    hexes = load_hexes().to_crs(UTM)
    lots = load_lots()
    print(f"PLUTO {lots.version.iloc[0]}: {len(lots):,} lots")

    # Category flags per lot, then values to add up.
    for name, codes in LANDUSE.items():
        lots[f"is_{name}"] = lots.landuse.isin(codes)
    lots["is_cemetery"] = lots.bldgclass.eq(CEMETERY)
    lots["is_airport"] = lots.bldgclass.eq(AIRPORT)
    lots["is_unbuilt_unlivable"] = lots.is_open_space | lots.is_cemetery | lots.is_airport | \
        (lots.is_transport_utility & lots.unitsres.eq(0) & lots.bldgarea.lt(0.05 * lots.lotarea))
    lots["lot_m2"] = lots.lotarea * SQFT_TO_M2
    area_cats = [*LANDUSE, "cemetery", "airport", "unbuilt_unlivable"]
    for cat in area_cats:
        lots[f"{cat}_m2"] = lots.lot_m2 * lots[f"is_{cat}"]
    for col in ["bldgarea", "resarea", "comarea", "officearea", "retailarea", "factryarea"]:
        lots[f"{col}_m2"] = lots[col] * SQFT_TO_M2
    sum_cols = ["lot_m2", *[f"{c}_m2" for c in area_cats],
                "bldgarea_m2", "resarea_m2", "comarea_m2", "officearea_m2", "retailarea_m2", "factryarea_m2",
                "unitsres", "numbldgs"]

    # Large lots: split by outline overlap with hexagons.
    outlines = load_large_outlines().to_crs(UTM)
    big = lots[lots.lotarea >= LARGE_LOT_SQFT].merge(outlines, on="bbl", how="inner")
    big = gpd.GeoDataFrame(big, geometry="geometry", crs=UTM)
    big["outline_m2"] = big.area
    pieces = gpd.overlay(big[["bbl", "outline_m2", "geometry"]], hexes[["hex_id", "geometry"]],
                         how="intersection", keep_geom_type=True)
    pieces["frac"] = pieces.area / pieces.outline_m2
    pieces = pieces.merge(lots[["bbl", *sum_cols]], on="bbl")
    pieces[sum_cols] = pieces[sum_cols].mul(pieces.frac, axis=0)
    big_sums = pieces.groupby("hex_id")[sum_cols].sum()

    # Everything else (and large lots with no outline): by center point.
    small = lots[~lots.bbl.isin(big.bbl)].dropna(subset=["latitude", "longitude"])
    pts = gpd.GeoDataFrame(small, geometry=gpd.points_from_xy(small.longitude, small.latitude), crs="EPSG:4326").to_crs(UTM)
    joined = gpd.sjoin(pts, hexes[["hex_id", "geometry"]], how="inner", predicate="within")
    small_sums = joined.groupby("hex_id")[sum_cols].sum()
    years = joined[joined.yearbuilt > 1700].groupby("hex_id").yearbuilt.median()

    total_lot = lots.lot_m2.sum()
    kept = pieces.lot_m2.sum() + small_sums.lot_m2.sum()
    print(f"{len(big):,} large lots by outline, {len(joined):,} lots by point; "
          f"{kept / total_lot:.1%} of all lot area lands in a hexagon")

    s = big_sums.add(small_sums, fill_value=0).reindex(hexes.hex_id).fillna(0)
    land_m2 = hexes.set_index("hex_id").land_km2 * 1e6
    lot = s.lot_m2.replace(0, np.nan)

    out = pd.DataFrame(index=s.index)
    out["lot_area_km2"] = (s.lot_m2 / 1e6).round(4)
    out["lot_coverage"] = (s.lot_m2 / land_m2).clip(upper=1).round(3)
    for cat in area_cats:
        out[f"share_{cat}"] = (s[f"{cat}_m2"] / lot).round(3)
    out["res_units"] = s.unitsres.round(0).astype(int)
    out["res_units_per_km2"] = (s.unitsres / land_m2 * 1e6).round(0)
    out["buildings"] = s.numbldgs.round(0).astype(int)
    out["floor_area_m2"] = s.bldgarea_m2.round(0)
    out["res_floor_m2"] = s.resarea_m2.round(0)
    out["com_floor_m2"] = s.comarea_m2.round(0)
    out["office_floor_m2"] = s.officearea_m2.round(0)
    out["retail_floor_m2"] = s.retailarea_m2.round(0)
    out["industrial_floor_m2"] = s.factryarea_m2.round(0)
    out["built_far"] = (s.bldgarea_m2 / lot).round(2)
    out["median_year_built"] = years.reindex(s.index)
    out = out.reset_index()

    print(f"Median share unbuilt/unlivable {out.share_unbuilt_unlivable.median():.2f}; "
          f"{(out.share_unbuilt_unlivable > 0.5).sum()} hexagons over half; "
          f"{out.res_units.sum():,} housing units (PLUTO citywide {lots.unitsres.sum():,.0f})")

    shares = {
        "residential": "1-2 family, walk-up and elevator residential (land use 01-03)",
        "mixed": "mixed residential and commercial (04)",
        "commercial": "commercial and office (05)",
        "industrial": "industrial and manufacturing (06)",
        "transport_utility": "transportation and utility: airports, rail yards, bus depots, power plants (07)",
        "institution": "public facilities and institutions: schools, hospitals, government (08)",
        "open_space": "open space and outdoor recreation: parks, beaches (09)",
        "parking": "parking facilities (10)",
        "vacant": "vacant land (11)",
        "cemetery": "cemeteries (building class Z8)",
        "airport": "airports (building class T1)",
        "unbuilt_unlivable": "land nobody lives on: open space + cemeteries + airports + transport/utility lots "
                             "that have no housing and almost no buildings (rail yards, depots)",
    }
    columns = {
        "hex_id": "hexagon id (matches data/hexgrid.geojson)",
        "lot_area_km2": "total tax-lot area in the hexagon, km² (excludes streets and water)",
        "lot_coverage": "lot area / hexagon land area (the rest is mostly streets)",
        **{f"share_{k}": f"share of the hexagon's lot area that is {v}" for k, v in shares.items()},
        "res_units": "residential units (homes)",
        "res_units_per_km2": "residential units per km² of land",
        "buildings": "number of buildings",
        "floor_area_m2": "total building floor area, m²",
        "res_floor_m2": "residential floor area, m²",
        "com_floor_m2": "commercial floor area (office + retail + other commercial), m²",
        "office_floor_m2": "office floor area, m² (a proxy for jobs)",
        "retail_floor_m2": "retail floor area, m²",
        "industrial_floor_m2": "factory and industrial floor area, m²",
        "built_far": "floor area / lot area (built density; Midtown towers are 10+)",
        "median_year_built": "median year built of lots with a building (point-assigned lots only)",
    }
    write_features(
        out, "landuse",
        f"Land use and buildings per hexagon (PLUTO {lots.version.iloc[0]})",
        sources=[
            (f"NYC Department of City Planning, PLUTO {lots.version.iloc[0]} lot attributes (NYC Open Data 64uk-42ks)", PLUTO_PAGE),
            (f"NYC Department of City Planning, MapPLUTO {lots.version.iloc[0]} lot outlines (ArcGIS feature service), "
             f"lots of {LARGE_LOT_SQFT:,} sq ft or more", MAPPLUTO_PAGE),
            ("MapPLUTO feature service endpoint", MAPPLUTO_QUERY.rsplit("/query", 1)[0]),
            ("PLUTO data dictionary (land use codes, building classes), linked from this page", MAPPLUTO_PAGE),
        ],
        columns=columns,
        notes=[
            f"Lots under {LARGE_LOT_SQFT:,} sq ft are assigned by center point; larger lots are split by outline overlap.",
            "Shares are of lot area, not hexagon area, so streets don't count; they can overlap "
            "(cemetery and airport are subsets of other land uses). Empty when a hexagon has no lots.",
            "PLUTO areas are in square feet in the source and converted to m² here.",
        ],
    )


if __name__ == "__main__":
    main()
