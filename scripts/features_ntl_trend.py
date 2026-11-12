"""Night-time light trend per hexagon, 2014-2025, using Google Earth Engine.

Mean VIIRS radiance on each hexagon's land for every year, then how much it
changed. Rising light is a sign of new development (and future riders).

The annual composites come in two versions: V2.1 covers 2013-2021 and V2.2
covers 2022 onward. Both use band 'average_masked' (fires, aurora, outliers and
dim background removed). The script prints the citywide mean per year so any
jump at the 2021/2022 switch is visible.

Writes:  data/features/ntl_trend.csv
Usage:   python scripts/features_ntl_trend.py --project ml-ntl-dataset
"""

import argparse
import json

import ee
import numpy as np
import pandas as pd

from features_common import DATA, write_features

FIRST_YEAR, LAST_YEAR = 2014, 2025
COLLECTIONS = {"NOAA/VIIRS/DNB/ANNUAL_V21": range(2013, 2022), "NOAA/VIIRS/DNB/ANNUAL_V22": range(2022, 2100)}
SCALE = 463.83  # native VIIRS pixel, meters


def year_image(year):
    name = next(c for c, years in COLLECTIONS.items() if year in years)
    img = ee.ImageCollection(name).filter(ee.Filter.calendarRange(year, year, "year")).first()
    return ee.Image(img).select("average_masked").unmask(0).max(0).rename(f"y{year}")


def hex_collection():
    gj = json.loads((DATA / "hexgrid_land.geojson").read_text())
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(f["geometry"]), {"hex_id": f["properties"]["hex_id"]}) for f in gj["features"]
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    args = parser.parse_args()
    ee.Initialize(project=args.project)

    years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    stack = ee.Image.cat([year_image(y) for y in years])
    print(f"Reducing {len(years)} years over the hexagons...")
    result = stack.reduceRegions(collection=hex_collection(), reducer=ee.Reducer.mean(),
                                 scale=SCALE, tileScale=2).getInfo()
    df = pd.DataFrame([f["properties"] for f in result["features"]])
    cols = [f"ntl_{y}" for y in years]
    df = df.rename(columns={f"y{y}": f"ntl_{y}" for y in years})[["hex_id", *cols]]

    print("Citywide mean radiance by year (check for a jump at 2021 -> 2022):")
    print("  " + "  ".join(f"{y}:{df[f'ntl_{y}'].mean():.1f}" for y in years))

    # Early and late levels as 3-year averages, so one noisy year doesn't drive the change.
    early = df[[f"ntl_{y}" for y in years[:3]]].mean(axis=1)
    late = df[[f"ntl_{y}" for y in years[-3:]]].mean(axis=1)
    df["ntl_early"] = early.round(3)
    df["ntl_late"] = late.round(3)
    df["ntl_change"] = (late - early).round(3)
    df["ntl_change_pct"] = ((late - early) / (early + 1) * 100).round(1)

    # Trend: OLS slope of log(radiance + 1) per year, as % growth per year. Fit on every year.
    t = np.array(years, dtype=float)
    logs = np.log1p(df[cols].to_numpy())
    slope = np.polyfit(t - t.mean(), logs.T, 1)[0]
    df["ntl_trend_pct_per_year"] = ((np.exp(slope) - 1) * 100).round(2)
    df[cols] = df[cols].round(3)

    order = [f["properties"]["hex_id"] for f in json.loads((DATA / "hexgrid.geojson").read_text())["features"]]
    df = df.set_index("hex_id").loc[order].reset_index()
    print(f"Median change {df.ntl_change_pct.median():.1f}%, "
          f"{(df.ntl_trend_pct_per_year > 2).sum()} hexagons growing > 2%/yr, "
          f"{(df.ntl_trend_pct_per_year < -2).sum()} falling > 2%/yr")

    columns = {"hex_id": "hexagon id (matches data/hexgrid.geojson)"}
    columns |= {f"ntl_{y}": f"mean radiance on the hexagon's land in {y}, nW/cm²/sr" for y in years}
    columns |= {
        "ntl_early": f"average of {years[0]}-{years[2]} mean radiance",
        "ntl_late": f"average of {years[-3]}-{years[-1]} mean radiance",
        "ntl_change": "ntl_late minus ntl_early, nW/cm²/sr",
        "ntl_change_pct": "(ntl_late - ntl_early) / (ntl_early + 1) x 100; the +1 keeps dark cells from blowing up",
        "ntl_trend_pct_per_year": f"growth per year from a straight-line fit of log(radiance + 1) over {years[0]}-{years[-1]}",
    }
    write_features(
        df, "ntl_trend",
        f"Night-time light per hexagon, {years[0]}-{years[-1]} (VIIRS annual composites)",
        sources=[
            ("Earth Observation Group (Colorado School of Mines), VIIRS Nighttime Lights annual V2.1, 2013-2021, "
             "Earth Engine NOAA/VIIRS/DNB/ANNUAL_V21",
             "https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_ANNUAL_V21"),
            ("Earth Observation Group, VIIRS Nighttime Lights annual V2.2, 2022 onward, "
             "Earth Engine NOAA/VIIRS/DNB/ANNUAL_V22",
             "https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_ANNUAL_V22"),
            ("Elvidge et al. (2021), Annual time series of global VIIRS nighttime lights, Remote Sensing 13(5):922",
             "https://doi.org/10.3390/rs13050922"),
        ],
        columns=columns,
        notes=[
            "Band average_masked (fires, aurora, outliers and dim background removed), negatives set to 0.",
            "Pixels are ~464 m, so each hexagon covers about 4 pixels and neighbors share values.",
            "2014-2021 come from V2.1 and 2022+ from V2.2; small level shifts at the switch are possible.",
            "Citywide light dips in 2016-2018 as NYC switched streetlights to LED (VIIRS is less sensitive to white LED light),",
            "and 2020 is lower (COVID-19); the 3-year averages and the trend fit smooth over single years.",
        ],
    )


if __name__ == "__main__":
    main()
