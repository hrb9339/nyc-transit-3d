"""Who lives in each hexagon: income, cars, how people commute (ACS 5-year).

American Community Survey 2020-2024 5-year estimates, census tract level, from
the Census Bureau's table-based summary files (no API key needed).

Tracts are much bigger than hexagons in the outer boroughs, so tract numbers
are spread down to 2020 Census blocks and back up to hexagons:
  - person counts (people, workers, commuters) in proportion to block population
  - household counts (households, cars, renters) in proportion to block housing units
  - a block split across hexagons is divided by area, as in demographics.csv
Medians (income, rent) can't be added up, so a hexagon gets the average of its
tracts' medians weighted by the households it took from each tract.

Writes:  data/features/acs.csv
Usage:   python scripts/features_acs.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from features_common import RAW, UTM, fetch, load_hexes, write_features

YEAR = 2024   # 2020-2024 5-year
SF = f"https://www2.census.gov/programs-surveys/acs/summary_file/{YEAR}/table-based-SF/data/5YRData/acsdt5y{YEAR}-{{t}}.dat"
SF_PAGE = "https://www.census.gov/programs-surveys/acs/data/summary-file.html"
BLOCKS = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_36_tabblock20.zip"
BLOCKS_PAGE = "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Blocks"
COUNTIES = ["005", "047", "061", "081", "085"]
MIN_DENOM = 50   # shares are left blank when the hexagon has fewer people/households/workers than this

# Count variables: name -> (table, [estimate columns summed], weight: "pop" or "hu")
under18 = [3, 4, 5, 6, 27, 28, 29, 30]
over65 = [20, 21, 22, 23, 24, 25, 44, 45, 46, 47, 48, 49]
COUNTS = {
    "acs_population":    ("b01003", [1], "pop"),
    "age_under18":       ("b01001", under18, "pop"),
    "age_65plus":        ("b01001", over65, "pop"),
    "poverty_universe":  ("b17001", [1], "pop"),
    "below_poverty":     ("b17001", [2], "pop"),
    "labor_force":       ("b23025", [3], "pop"),
    "unemployed":        ("b23025", [5], "pop"),
    "workers":           ("b08301", [1], "pop"),
    "drove_alone":       ("b08301", [3], "pop"),
    "transit_commuters": ("b08301", [10], "pop"),
    "bus_commuters":     ("b08301", [11], "pop"),
    "subway_commuters":  ("b08301", [12], "pop"),
    "rail_commuters":    ("b08301", [13], "pop"),
    "walk_bike":         ("b08301", [18, 19], "pop"),
    "work_from_home":    ("b08301", [21], "pop"),
    "commuters_timed":   ("b08303", [1], "pop"),
    "commute_60plus":    ("b08303", [12, 13], "pop"),
    "households":        ("b08201", [1], "hu"),
    "no_vehicle":        ("b08201", [2], "hu"),
    "occupied_units":    ("b25003", [1], "hu"),
    "renter_units":      ("b25003", [3], "hu"),
}
MEDIANS = {"median_hh_income": ("b19013", 1), "median_gross_rent": ("b25064", 1)}


def load_table(t):
    """One ACS table for NYC tracts: DataFrame indexed by 11-digit tract GEOID."""
    path = RAW / "acs" / f"acsdt5y{YEAR}-{t}.dat"
    fetch(SF.format(t=t), path)
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("|")
        for line in f:
            if line.startswith("1400000US36") and line[11:14] in COUNTIES:   # tracts in the five counties
                rows.append(line.rstrip("\n").split("|"))
    df = pd.DataFrame(rows, columns=header)
    df["tract"] = df.GEO_ID.str[-11:]
    est = [c for c in df.columns if "_E" in c]
    df[est] = df[est].apply(pd.to_numeric, errors="coerce")
    df[est] = df[est].where(df[est] >= 0)   # -666666666 etc. = not available
    return df.set_index("tract")[est]


def tract_values():
    tables = {t: load_table(t) for t in {v[0] for v in [*COUNTS.values(), *MEDIANS.values()]}}
    out = pd.DataFrame(index=tables["b01003"].index)
    for name, (t, cols, _) in COUNTS.items():
        tab = tables[t]
        out[name] = sum(tab[f"{t.upper()}_E{c:03d}"] for c in cols)
    for name, (t, c) in MEDIANS.items():
        out[name] = tables[t][f"{t.upper()}_E{c:03d}"]
    return out


def main():
    tracts = tract_values()
    print(f"{len(tracts):,} NYC tracts; ACS population {tracts.acs_population.sum():,.0f}")

    fetch(BLOCKS, RAW / "tl_2020_36_tabblock20.zip")
    blocks = gpd.read_file(f"zip://{RAW / 'tl_2020_36_tabblock20.zip'}")
    blocks = blocks[blocks.COUNTYFP20.isin(COUNTIES)][["GEOID20", "POP20", "HOUSING20", "ALAND20", "geometry"]]
    blocks["tract"] = blocks.GEOID20.str[:11]

    # Share of each tract's people / housing units in each block. Tracts with no block
    # population (rare) fall back to land area.
    for col, w in [("POP20", "w_pop"), ("HOUSING20", "w_hu")]:
        tot = blocks.groupby("tract")[col].transform("sum")
        area = blocks.groupby("tract").ALAND20.transform("sum")
        blocks[w] = np.where(tot > 0, blocks[col] / tot.where(tot > 0, 1), blocks.ALAND20 / area.where(area > 0, 1))

    # Split blocks across hexagons by area.
    hexes = load_hexes()
    b = blocks.to_crs(UTM)
    b["block_m2"] = b.area
    pieces = gpd.overlay(b, hexes[["hex_id", "geometry"]].to_crs(UTM), how="intersection", keep_geom_type=True)
    pieces["share"] = pieces.area / pieces.block_m2.where(pieces.block_m2 > 0, 1)

    t = tracts.reindex(pieces.tract)
    out = pd.DataFrame({"hex_id": pieces.hex_id.values})
    for name, (_, _, w) in COUNTS.items():
        out[name] = t[name].values * pieces[f"w_{w}"].values * pieces.share.values
    hh_w = t["households"].values * pieces.w_hu.values * pieces.share.values
    for name in MEDIANS:
        m = t[name].values
        ok = ~np.isnan(m)
        out[name + "_x"] = np.where(ok, m * hh_w, 0)
        out[name + "_w"] = np.where(ok, hh_w, 0)
    g = out.groupby("hex_id").sum().reindex(hexes.hex_id).fillna(0)

    res = pd.DataFrame({"hex_id": hexes.hex_id.values})

    def share(num, den):
        return np.round(np.where(g[den].values >= MIN_DENOM, g[num].values / np.maximum(g[den].values, 1), np.nan), 3)

    def median(name):
        w = g[name + "_w"].values
        return np.round(np.where(w >= MIN_DENOM, g[name + "_x"].values / np.maximum(w, 1e-9), np.nan), -1)

    res["acs_population"] = g.acs_population.round(0).values
    res["households"] = g.households.round(0).values
    res["workers"] = g.workers.round(0).values
    res["median_hh_income"] = median("median_hh_income")
    res["median_gross_rent"] = median("median_gross_rent")
    res["share_below_poverty"] = share("below_poverty", "poverty_universe")
    res["unemployment_rate"] = share("unemployed", "labor_force")
    res["share_no_vehicle"] = share("no_vehicle", "households")
    res["share_renters"] = share("renter_units", "occupied_units")
    res["share_under18"] = share("age_under18", "acs_population")
    res["share_65plus"] = share("age_65plus", "acs_population")
    res["share_commute_transit"] = share("transit_commuters", "workers")
    res["share_commute_subway"] = share("subway_commuters", "workers")
    res["share_commute_bus"] = share("bus_commuters", "workers")
    res["share_commute_rail"] = share("rail_commuters", "workers")
    res["share_commute_drive_alone"] = share("drove_alone", "workers")
    res["share_commute_walk_bike"] = share("walk_bike", "workers")
    res["share_work_from_home"] = share("work_from_home", "workers")
    res["share_commute_60min_plus"] = share("commute_60plus", "commuters_timed")
    res["transit_commuters"] = g.transit_commuters.round(0).values
    res["no_vehicle_households"] = g.no_vehicle.round(0).values

    print(f"hexagon totals: population {res.acs_population.sum():,.0f}, households {res.households.sum():,.0f}, "
          f"workers {res.workers.sum():,.0f}; {res.median_hh_income.isna().sum()} hexagons without income")

    sf = lambda t: (f"ACS {YEAR - 4}-{YEAR} 5-year, table {t.upper()}", SF.format(t=t))
    write_features(
        res, "acs",
        f"Income, cars and commuting per hexagon (ACS {YEAR - 4}-{YEAR} 5-year, tracts spread via 2020 blocks)",
        sources=[("ACS summary file documentation", SF_PAGE)]
        + [sf(t) for t in ["b01003", "b01001", "b19013", "b25064", "b17001", "b23025", "b08201", "b25003",
                           "b08301", "b08303"]]
        + [("2020 Census blocks (TIGER/Line tabblock20, POP20 and HOUSING20 used as weights)", BLOCKS_PAGE)],
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "acs_population": "people (ACS estimate, B01003)",
            "households": "households (B08201)",
            "workers": "workers 16+ (B08301)",
            "median_hh_income": "median household income, 2024 dollars (household-weighted mean of tract medians, B19013)",
            "median_gross_rent": "median gross rent per month, dollars (household-weighted mean of tract medians, B25064)",
            "share_below_poverty": "share of people below the poverty line (B17001)",
            "unemployment_rate": "unemployed / civilian labor force (B23025)",
            "share_no_vehicle": "share of households with no car (B08201)",
            "share_renters": "share of occupied homes that are rented (B25003)",
            "share_under18": "share of people under 18 (B01001)",
            "share_65plus": "share of people 65 and over (B01001)",
            "share_commute_transit": "share of workers commuting by public transit, excluding taxi (B08301)",
            "share_commute_subway": "share of workers commuting by subway or elevated rail (B08301)",
            "share_commute_bus": "share of workers commuting by bus (B08301)",
            "share_commute_rail": "share of workers commuting by long-distance/commuter rail, e.g. LIRR (B08301)",
            "share_commute_drive_alone": "share of workers driving alone (B08301)",
            "share_commute_walk_bike": "share of workers walking or cycling (B08301)",
            "share_work_from_home": "share of workers working from home (B08301)",
            "share_commute_60min_plus": "share of commuters (not working from home) traveling 60+ minutes (B08303)",
            "transit_commuters": "workers commuting by public transit (count)",
            "no_vehicle_households": "households with no car (count)",
        },
        notes=[
            f"Shares and medians are blank where the hexagon has fewer than {MIN_DENOM} people, households or "
            "workers (parks, airports, industrial areas): too few for a meaningful value.",
            "ACS values are survey estimates with margins of error, and spreading tracts to blocks assumes "
            "people in a tract are alike. Treat small differences between neighbouring hexagons as noise.",
            "The 2020-2024 survey period includes COVID; work-from-home is higher than before 2020.",
            "share_commute_subway is partly an outcome of having a subway nearby. Fine as an input for "
            "describing demand, but it leaks the answer if used to predict ridership at existing stations; "
            "share_no_vehicle and income are safer inputs.",
        ],
    )


if __name__ == "__main__":
    main()
