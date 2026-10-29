"""Night-time light (NTL) statistics per hexagon, using Google Earth Engine.

Satellite: VIIRS Day/Night Band (Suomi NPP / NOAA-20), ~464 m pixels.
  - Main:   NOAA/VIIRS/DNB/ANNUAL_V22, latest year, band 'average_masked'
            (annual composite with fires, aurora, outliers and dim background removed)
  - Recent: NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG, median of the latest 12 months, band 'avg_rad'
Units: radiance in nW/cm²/sr.

Per hexagon (land part only, from data/hexgrid_land.geojson):
  ntl_mean     mean radiance
  ntl_median   median radiance
  ntl_max      brightest pixel
  ntl_sum      "sum of lights" (SOL): radiance summed over pixels, weighted by how
               much of each pixel falls inside the hexagon (the usual NTL aggregate)
  pixels       number of pixels covering the land area (fractional)
  lit_pixels   pixels with radiance > LIT
  bright_pixels pixels with radiance > BRIGHT
  ntl_recent_mean / ntl_recent_sum   same, from the recent 12-month median

Writes:
  data/ntl/hex_ntl.csv                     table for analysis
  data/ntl/viirs_annual_<year>_nyc.tif     raster download (GeoTIFF) for reference
  data/ntl/viirs_recent12_nyc.tif          recent 12-month median raster
  data/ntl/ntl_web.png + ntl_web.json      colored image + bounds for the website
  data/hexgrid.geojson                     NTL columns added to each hexagon

Usage:
  earthengine authenticate          (once, opens a browser)
  python scripts/gee_ntl.py --project YOUR-CLOUD-PROJECT-ID
"""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import ee

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "ntl"
NYC_BBOX = [-74.27, 40.48, -73.68, 40.93]  # west, south, east, north
LIT = 1.0      # nW/cm²/sr: above typical dark background
BRIGHT = 50.0  # nW/cm²/sr: dense commercial / high-rise lighting


def annual_image():
    col = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V22").sort("system:time_start", False)
    img = ee.Image(col.first())
    year = ee.Date(img.get("system:time_start")).get("year").getInfo()
    return img.select("average_masked").unmask(0).rename("ntl"), year


def recent_image():
    col = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").sort("system:time_start", False).limit(12)
    dates = col.aggregate_array("system:time_start").getInfo()
    first = ee.Date(min(dates)).format("YYYY-MM").getInfo()
    last = ee.Date(max(dates)).format("YYYY-MM").getInfo()
    return col.select("avg_rad").median().max(0).rename("ntl"), f"{first} to {last}"


def hex_collection():
    gj = json.loads((DATA / "hexgrid_land.geojson").read_text())
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(f["geometry"]), {"hex_id": f["properties"]["hex_id"]}) for f in gj["features"]
    ])


def stats(image, hexes, scale):
    stack = image.addBands([
        image.gt(LIT).rename("lit"),
        image.gt(BRIGHT).rename("bright"),
        ee.Image.constant(1).rename("px"),
    ])
    reducer = ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True) \
        .combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.sum(), sharedInputs=True)
    result = stack.reduceRegions(collection=hexes, reducer=reducer, scale=scale, tileScale=2).getInfo()
    out = {}
    for f in result["features"]:
        p = f["properties"]
        out[p["hex_id"]] = {
            "ntl_mean": p.get("ntl_mean"),
            "ntl_median": p.get("ntl_median"),
            "ntl_max": p.get("ntl_max"),
            "ntl_sum": p.get("ntl_sum"),
            "pixels": p.get("px_sum"),
            "lit_pixels": p.get("lit_sum"),
            "bright_pixels": p.get("bright_sum"),
        }
    return out


def download(url, path):
    with urllib.request.urlopen(url, timeout=300) as r:
        path.write_bytes(r.read())
    print(f"  saved {path.relative_to(DATA.parent)} ({path.stat().st_size // 1024} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    args = parser.parse_args()
    ee.Initialize(project=args.project)
    OUT.mkdir(exist_ok=True)

    annual, year = annual_image()
    recent, recent_label = recent_image()
    scale = annual.projection().nominalScale().getInfo()
    region = ee.Geometry.Rectangle(NYC_BBOX, "EPSG:4326", False)
    print(f"VIIRS annual {year}, recent median {recent_label}, pixel {scale:.0f} m")

    print("Computing hexagon statistics...")
    hexes = hex_collection()
    main_stats = stats(annual, hexes, scale)
    recent_stats = stats(recent, hexes, scale)

    rows = []
    grid = json.loads((DATA / "hexgrid.geojson").read_text())
    for f in grid["features"]:
        p = f["properties"]
        s, r = main_stats.get(p["hex_id"], {}), recent_stats.get(p["hex_id"], {})
        ntl = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
        ntl["ntl_recent_mean"] = round(r["ntl_mean"], 3) if r.get("ntl_mean") is not None else None
        ntl["ntl_recent_sum"] = round(r["ntl_sum"], 3) if r.get("ntl_sum") is not None else None
        ntl["ntl_density"] = round(ntl["ntl_sum"] / p["land_km2"], 3) if ntl.get("ntl_sum") is not None else None
        p.update(ntl)
        rows.append({
            "hex_id": p["hex_id"], "borough": p["borough"], "center_lon": p["center"][0], "center_lat": p["center"][1],
            "land_km2": p["land_km2"], "stations": p["stations"], "nearest_station_m": p["nearest_station_m"], **ntl,
        })

    with open(OUT / "hex_ntl.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved data/ntl/hex_ntl.csv ({len(rows)} hexagons)")
    grid["metadata"] = {"ntl_source": f"VIIRS DNB annual V2.2 {year} (average_masked)",
                        "ntl_recent_source": f"VIIRS DNB monthly VCMSLCFG median {recent_label}",
                        "units": "nW/cm2/sr", "lit_threshold": LIT, "bright_threshold": BRIGHT}
    (DATA / "hexgrid.geojson").write_text(json.dumps(grid, separators=(",", ":")))

    print("Downloading rasters...")
    for img, name in ((annual, f"viirs_annual_{year}_nyc.tif"), (recent, "viirs_recent12_nyc.tif")):
        url = img.clip(region).getDownloadURL({"region": region, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"})
        download(url, OUT / name)

    # Web overlay: log-stretched, smoothed, in Web Mercator so it lines up with the map.
    vis = annual.resample("bicubic").add(1).log10().visualize(
        min=0, max=2.5, palette=["000000", "1a0b3d", "5c1a6b", "b8326b", "f0703a", "fbc54a", "fff7d6"]
    )
    url = vis.getThumbURL({"region": region, "dimensions": 2048, "crs": "EPSG:3857", "format": "png"})
    download(url, OUT / "ntl_web.png")
    (OUT / "ntl_web.json").write_text(json.dumps({
        "bounds": NYC_BBOX, "source": f"VIIRS DNB annual V2.2 {year}", "units": "nW/cm2/sr",
        "stretch": "log10(radiance + 1), 0 to 2.5",
    }))


if __name__ == "__main__":
    main()
