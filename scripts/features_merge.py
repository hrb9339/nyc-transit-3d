"""Join every per-hexagon feature table into one file.

Takes the base grid attributes (borough, land area, stations, 2025 night light)
and every data/features/<name>.csv, joins them on hex_id, and writes
data/features/hex_features.csv. The header repeats each table's sources and
column descriptions, so the merged file documents itself.

Run the individual scripts first (or after any of them changes):
    python scripts/features_neighborhoods.py
    python scripts/features_landuse.py
    python scripts/features_destinations.py
    python scripts/features_ntl_trend.py --project ml-ntl-dataset
    python scripts/features_proposals.py
    python scripts/features_walk.py
    python scripts/features_merge.py

Writes:  data/features/hex_features.csv
"""

import json
from datetime import date

import pandas as pd

from features_common import DATA, FEATURES, ROOT

OUT = FEATURES / "hex_features.csv"
BASE_COLUMNS = {
    "hex_id": "hexagon id (matches data/hexgrid.geojson)",
    "borough": "borough containing most of the hexagon",
    "center_lon": "hexagon center longitude (WGS84)",
    "center_lat": "hexagon center latitude (WGS84)",
    "land_km2": "land area of the hexagon after cutting to the shoreline, km²",
    "stations": "subway stations inside the hexagon (each station counted in one hexagon)",
    "nearest_station_m": "straight-line distance from the center to the nearest station center, m",
    "ntl_mean": "2025 mean night-time radiance on the hexagon's land, nW/cm²/sr (VIIRS annual V2.2)",
    "ntl_sum": "2025 sum of lights (radiance summed over pixels, weighted by share inside the hexagon)",
    "ntl_density": "ntl_sum / land_km2",
    "lit_pixels": "pixels with radiance > 1",
    "bright_pixels": "pixels with radiance > 50",
}
BASE_SOURCES = [
    "Hexagon grid: scripts/build_hexgrid.py (500 m radius, UTM 18N, cut to NYC shoreline, "
    "https://data.cityofnewyork.us/resource/gthc-hcne)",
    "Stations: MTA Subway Stations, data.ny.gov 39hk-dx4f, https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f",
    "2025 night light: scripts/gee_ntl.py, Earth Engine NOAA/VIIRS/DNB/ANNUAL_V22, "
    "https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_ANNUAL_V22",
]


def read_header(path):
    """Return (title, sources, columns) parsed from a feature file's '#' header."""
    title, sources, columns, section = None, [], {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        text = line[1:].strip()
        if title is None:
            title = text
        elif text in ("Sources:", "Columns:", "Notes:"):
            section = text
        elif section == "Sources:" and text.startswith("- "):
            sources.append(text[2:])
        elif section == "Columns:" and ": " in text:
            col, desc = text.split(": ", 1)
            columns[col] = desc
    return title, sources, columns


def main():
    grid = json.loads((DATA / "hexgrid.geojson").read_text())
    rows = []
    for f in grid["features"]:
        p = f["properties"]
        rows.append({k: p.get(k) for k in BASE_COLUMNS if k not in ("center_lon", "center_lat")}
                    | {"center_lon": p["center"][0], "center_lat": p["center"][1]})
    df = pd.DataFrame(rows)[list(BASE_COLUMNS)]

    columns = dict(BASE_COLUMNS)
    blocks = [("Base grid (data/hexgrid.geojson)", BASE_SOURCES)]
    tables = sorted(p for p in FEATURES.glob("*.csv") if p != OUT)
    for path in tables:
        title, sources, cols = read_header(path)
        table = pd.read_csv(path, comment="#")
        assert len(table) == len(df) and table.hex_id.is_unique, f"{path.name}: expected one row per hexagon"
        clash = [c for c in table.columns if c != "hex_id" and c in df.columns]
        if clash:
            raise SystemExit(f"{path.name}: columns already present from another table: {clash}")
        df = df.merge(table, on="hex_id", how="left", validate="one_to_one")
        for c in table.columns:
            if c != "hex_id":
                columns[c] = f"[{path.stem}] {cols.get(c, '')}"
        blocks.append((f"{title} (data/features/{path.name})", sources))

    lines = [
        "# All per-hexagon features for the NYC transit-gap project, one row per hexagon",
        f"# Built {date.today().isoformat()} by scripts/features_merge.py from the tables listed below",
        "# Read with: pandas.read_csv('data/features/hex_features.csv', comment='#')",
        "#",
        "# Sources:",
    ]
    for title, sources in blocks:
        lines.append(f"#   {title}")
        lines += [f"#     - {s}" for s in sources]
    lines += ["#", "# Columns ([table] = file the column comes from):"]
    lines += [f"#   {c}: {columns[c]}" for c in df.columns]
    lines += [
        "#",
        "# Notes:",
        "#   [proposals] columns are for validating results only; don't use them as model inputs.",
        "#   Per-table notes and caveats are in each data/features/<table>.csv header.",
        "#",
    ]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        df.to_csv(fh, index=False, lineterminator="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} rows, {len(df.columns)} columns from {len(tables)} tables")


if __name__ == "__main__":
    main()
