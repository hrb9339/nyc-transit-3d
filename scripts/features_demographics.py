"""Population and jobs per hexagon, from Census blocks and LODES.

Population comes straight from the 2020 Census block shapefile (POP20), so no
separate Census API call is needed. Jobs come from LODES8 Workplace Area
Characteristics, total jobs per block (C000).

Blocks are small (often a single city block), so a block's population and
jobs are split across every hexagon it overlaps, in proportion to how much of
the block's area falls in each one (i.e. assuming people/jobs are spread
evenly within a block).

Writes:  data/features/demographics.csv
Usage:   python scripts/features_demographics.py [--lodes-year 2021]
"""

import argparse
import gzip
import io

import geopandas as gpd
import pandas as pd

from features_common import RAW, UTM, fetch, load_hexes, write_features

COUNTIES = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_36_tabblock20.zip"
TIGER_PAGE = "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Blocks"
LODES_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/ny/wac/ny_wac_S000_JT00_{year}.csv.gz"
LODES_PAGE = "https://lehd.ces.census.gov/data/#lodes"


def load_blocks():
    """2020 Census blocks in the 5 boroughs, with population, in UTM."""
    zip_path = RAW / "tl_2020_36_tabblock20.zip"
    fetch(TIGER_URL, zip_path)
    blocks = gpd.read_file(f"zip://{zip_path}")
    blocks = blocks[blocks.COUNTYFP20.isin(COUNTIES)][["GEOID20", "POP20", "geometry"]]
    blocks = blocks.rename(columns={"GEOID20": "geoid", "POP20": "population"})
    blocks["geometry"] = blocks.geometry.make_valid()
    return blocks.to_crs(UTM)


def load_jobs(year):
    """15-digit block geocode -> total jobs (LODES8 WAC, C000), for the 5 boroughs."""
    path = RAW / f"ny_wac_{year}.csv.gz"
    raw = fetch(LODES_URL.format(year=year), path)
    wac = pd.read_csv(io.BytesIO(raw), compression="gzip", dtype={"w_geocode": str})
    wac = wac[wac.w_geocode.str[2:5].isin(COUNTIES)]
    return wac[["w_geocode", "C000"]].rename(columns={"w_geocode": "geoid", "C000": "jobs"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lodes-year", type=int, default=2021)
    args = parser.parse_args()

    print("Loading hexagons...")
    hexes = load_hexes().to_crs(UTM)
    hexes["hex_area"] = hexes.area

    print("Loading Census blocks (population)...")
    blocks = load_blocks()
    print(f"  {len(blocks)} blocks, {blocks.population.sum():,} people")

    print(f"Loading jobs (LODES8 WAC {args.lodes_year})...")
    jobs = load_jobs(args.lodes_year)
    blocks = blocks.merge(jobs, on="geoid", how="left")
    blocks["jobs"] = blocks.jobs.fillna(0)
    print(f"  {blocks.jobs.gt(0).sum()} blocks with jobs, {blocks.jobs.sum():,.0f} jobs")

    print("Overlaying blocks onto hexagons...")
    blocks["block_area"] = blocks.area
    blocks = blocks[blocks.block_area > 0]
    pieces = gpd.overlay(
        hexes[["hex_id", "hex_area", "geometry"]],
        blocks[["geoid", "population", "jobs", "block_area", "geometry"]],
        how="intersection", keep_geom_type=True,
    )
    frac = pieces.area / pieces.block_area
    pieces["pop_piece"] = pieces.population * frac
    pieces["jobs_piece"] = pieces.jobs * frac
    totals = pieces.groupby("hex_id")[["pop_piece", "jobs_piece"]].sum()

    out = hexes[["hex_id", "land_km2"]].merge(totals, on="hex_id", how="left").fillna(0)
    out["population"] = out.pop_piece.round().astype(int)
    out["jobs"] = out.jobs_piece.round().astype(int)
    out["pop_density"] = (out.population / out.land_km2).round(1)
    out["job_density"] = (out.jobs / out.land_km2).round(1)
    out = out[["hex_id", "population", "jobs", "pop_density", "job_density"]]

    print(f"  allocated {out.population.sum():,} people, {out.jobs.sum():,} jobs "
          f"({out.population.sum() / blocks.population.sum():.1%} / {out.jobs.sum() / blocks.jobs.sum():.1%} of raw totals)")

    write_features(
        out, "demographics",
        "Population and jobs per hexagon (2020 Census blocks + LODES)",
        sources=[
            ("2020 Census block population (TIGER/Line tabblock20, POP20)", TIGER_PAGE),
            (f"LODES8 Workplace Area Characteristics, total jobs per block (C000), {args.lodes_year}", LODES_PAGE),
        ],
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "population": "2020 Census population allocated to the hexagon",
            "jobs": f"LODES8 WAC total jobs ({args.lodes_year}) allocated to the hexagon",
            "pop_density": "population / land_km2",
            "job_density": "jobs / land_km2",
        },
        notes=[
            "A block's population and jobs are split across every hexagon it overlaps, in proportion to the "
            "share of the block's area in each one (assumes people/jobs are spread evenly within a block).",
            "Blocks are small (often a single city block), so this is a close approximation, not exact.",
            "Areas computed in UTM 18N (EPSG:32618) on the shoreline-clipped hexagons.",
        ],
    )


if __name__ == "__main__":
    main()
