"""Shared helpers for the per-hexagon feature scripts (scripts/features_*.py).

Every feature table is written to data/features/<name>.csv, one row per hexagon,
keyed by hex_id. The top of each file is a block of '#' comment lines giving the
title, the data sources with links, and what every column means. Read with:

    pandas.read_csv("data/features/<name>.csv", comment="#")
"""

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATURES = DATA / "features"
RAW = DATA / "raw"          # large downloads, git-ignored
UTM = "EPSG:32618"          # UTM zone 18N, meters
USER_AGENT = "nyc-transit-gaps (NYU course project)"


def load_hexes():
    """Hexagon grid (land-clipped polygons) as a GeoDataFrame in WGS84."""
    hexes = gpd.read_file(DATA / "hexgrid.geojson")
    center = hexes.pop("center").map(lambda c: json.loads(c) if isinstance(c, str) else c)
    hexes["center_lon"] = center.str[0]
    hexes["center_lat"] = center.str[1]
    return hexes[["hex_id", "borough", "center_lon", "center_lat", "land_km2", "geometry"]]


def hex_centers(hexes):
    """Hexagon centers as points, in the same row order as `hexes`."""
    return gpd.GeoDataFrame(
        hexes[["hex_id"]],
        geometry=gpd.points_from_xy(hexes.center_lon, hexes.center_lat),
        crs="EPSG:4326",
    )


def fetch(url, path=None, data=None):
    """Download `url` (POST if `data` is given). Cache to `path` if given."""
    if path and Path(path).exists():
        return Path(path).read_bytes()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(body)
    return body


OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def overpass(query, cache):
    """Run an Overpass API query and return the parsed JSON (cached).

    Tries each public server in turn, since any one of them is often busy.
    """
    if (RAW / cache).exists():
        return json.loads((RAW / cache).read_bytes())
    data = ("data=" + urllib.parse.quote(query)).encode()
    for server in OVERPASS_SERVERS:
        try:
            req = urllib.request.Request(server, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            result = json.loads(body)
        except Exception as e:  # timeout, 429/504, or an HTML error page
            print(f"  overpass: {server} failed ({e}), trying next")
            continue
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / cache).write_bytes(body)
        return result
    raise RuntimeError("all Overpass servers failed")


def write_features(df, name, title, sources, columns, notes=()):
    """Write `df` to data/features/<name>.csv with a commented header.

    sources: list of (description, url)
    columns: dict of column name -> meaning (every column in df must be listed)
    notes:   extra lines (method, caveats)
    """
    missing = [c for c in df.columns if c not in columns]
    if missing:
        raise ValueError(f"columns without a description: {missing}")
    lines = [f"# {title}", f"# Built {date.today().isoformat()} by scripts/features_{name}.py", "#", "# Sources:"]
    lines += [f"#   - {desc}: {url}" for desc, url in sources]
    lines += ["#", "# Columns:"]
    lines += [f"#   {col}: {columns[col]}" for col in df.columns]
    if notes:
        lines += ["#", "# Notes:"] + [f"#   {n}" for n in notes]
    lines.append("#")

    FEATURES.mkdir(parents=True, exist_ok=True)
    path = FEATURES / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        df.to_csv(f, index=False, lineterminator="\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(df)} rows, {len(df.columns)} columns")
