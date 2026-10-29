"""Build subway line + station GeoJSON for the 3D map.

Sources:
  - MTA static GTFS (track shapes + official route colors)
    https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip
  - MTA Subway Stations, data.ny.gov (dataset 39hk-dx4f)

Usage:  python scripts/build_data.py
Writes: data/subway-lines.geojson, data/subway-stations.geojson,
        data/network.json (track segments used to place live trains)
"""

import csv
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

GTFS_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
STATIONS_URL = "https://data.ny.gov/resource/39hk-dx4f.json?$limit=5000"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def read_csv(zf, name):
    with zf.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")))


def build_lines(zf):
    routes = {r["route_id"]: r for r in read_csv(zf, "routes.txt")}

    # Which shapes each route actually uses. Keep one direction ("N") to avoid
    # drawing every line twice; fall back to all shapes if a route has none.
    route_shapes = defaultdict(set)
    for t in read_csv(zf, "trips.txt"):
        if t["shape_id"]:
            route_shapes[t["route_id"]].add(t["shape_id"])

    points = defaultdict(list)
    for p in read_csv(zf, "shapes.txt"):
        points[p["shape_id"]].append(
            (int(p["shape_pt_sequence"]), round(float(p["shape_pt_lon"]), 5), round(float(p["shape_pt_lat"]), 5))
        )

    features = []
    for route_id, shape_ids in route_shapes.items():
        route = routes.get(route_id)
        if not route:
            continue
        northbound = {s for s in shape_ids if "..N" in s} or shape_ids
        lines, seen = [], set()
        for sid in sorted(northbound):
            coords = [[lon, lat] for _, lon, lat in sorted(points.get(sid, []))]
            key = tuple(map(tuple, coords))
            if len(coords) > 1 and key not in seen:
                seen.add(key)
                lines.append(coords)
        if not lines:
            continue
        # GTFS says white text everywhere; MTA signage uses black on the yellow N/Q/R/W.
        text_color = "000000" if route["route_color"].upper() == "F6BC26" else (route["route_text_color"] or "FFFFFF")
        features.append({
            "type": "Feature",
            "properties": {
                "route_id": route_id,
                "name": route["route_short_name"],
                "long_name": route["route_long_name"],
                "color": "#" + (route["route_color"] or "808183"),
                "text_color": "#" + text_color,
                "sort_order": int(route["route_sort_order"] or 0),
            },
            "geometry": {"type": "MultiLineString", "coordinates": lines},
        })
    features.sort(key=lambda f: f["properties"]["sort_order"])
    return {"type": "FeatureCollection", "features": features}


def gtfs_seconds(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def nearest_index(coords, lon, lat, start):
    # Lat ~40.7°: scale longitude so distances are roughly isotropic.
    best, best_d = start, float("inf")
    for i in range(start, len(coords)):
        dx = (coords[i][0] - lon) * 0.758
        dy = coords[i][1] - lat
        d = dx * dx + dy * dy
        if d < best_d:
            best, best_d = i, d
    return best


def build_network(zf):
    """Station-to-station track segments for placing live trains.

    stops:    stop_id (with N/S suffix, as used by the realtime feed) -> [lon, lat, name]
    segments: "A|B" -> [scheduled travel seconds, [[lon, lat], ...] following the track]
    shapes:   shape_id -> ordered stop_ids (realtime trip_ids end in a shape_id)
    pred:     "route|B" -> most common previous stop before B on that route
    """
    stops = {
        s["stop_id"]: [round(float(s["stop_lon"]), 5), round(float(s["stop_lat"]), 5), s["stop_name"]]
        for s in read_csv(zf, "stops.txt")
    }

    shape_pts = defaultdict(list)
    for p in read_csv(zf, "shapes.txt"):
        shape_pts[p["shape_id"]].append(
            (int(p["shape_pt_sequence"]), round(float(p["shape_pt_lon"]), 5), round(float(p["shape_pt_lat"]), 5))
        )
    shape_coords = {k: [[lon, lat] for _, lon, lat in sorted(v)] for k, v in shape_pts.items()}

    trips = {t["trip_id"]: t for t in read_csv(zf, "trips.txt")}
    trip_stops = defaultdict(list)
    for st in read_csv(zf, "stop_times.txt"):
        trip_stops[st["trip_id"]].append(
            (int(st["stop_sequence"]), st["stop_id"], gtfs_seconds(st["arrival_time"]), gtfs_seconds(st["departure_time"]))
        )

    travel = defaultdict(list)
    pred_count = defaultdict(lambda: defaultdict(int))
    shape_stops = {}
    for trip_id, sts in trip_stops.items():
        sts.sort()
        trip = trips.get(trip_id)
        if not trip:
            continue
        for (_, a, _, dep_a), (_, b, arr_b, _) in zip(sts, sts[1:]):
            travel[(a, b)].append(arr_b - dep_a)
            pred_count[(trip["route_id"], b)][a] += 1
        sid = trip["shape_id"]
        if sid and len(sts) > len(shape_stops.get(sid, [])):
            shape_stops[sid] = [s[1] for s in sts]

    # Cut each shape at its stops to get track geometry between consecutive stations.
    geometry = {}
    for sid, seq in shape_stops.items():
        coords = shape_coords.get(sid)
        if not coords:
            continue
        idx, cursor = [], 0
        for stop_id in seq:
            lon, lat, _ = stops[stop_id]
            cursor = nearest_index(coords, lon, lat, cursor)
            idx.append(cursor)
        for (a, ia), (b, ib) in zip(zip(seq, idx), zip(seq[1:], idx[1:])):
            if (a, b) not in geometry:
                geometry[(a, b)] = [stops[a][:2]] + coords[ia + 1:ib] + [stops[b][:2]]

    segments = {}
    for (a, b), times in travel.items():
        times.sort()
        coords = geometry.get((a, b)) or [stops[a][:2], stops[b][:2]]
        segments[f"{a}|{b}"] = [max(times[len(times) // 2], 30), coords]

    pred = {f"{r}|{b}": max(c, key=c.get) for (r, b), c in pred_count.items()}
    used = {s for k in segments for s in k.split("|")}
    return {
        "stops": {k: v for k, v in stops.items() if k in used},
        "segments": segments,
        "shapes": shape_stops,
        "pred": pred,
    }


def build_stations():
    rows = json.loads(fetch(STATIONS_URL))
    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "properties": {
                "name": r["stop_name"],
                "line": r.get("line", ""),
                "routes": r.get("daytime_routes", ""),
                "borough": r.get("borough", ""),
                "structure": r.get("structure", ""),
                "ada": r.get("ada", "0") != "0",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["gtfs_longitude"]), float(r["gtfs_latitude"])],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading GTFS...")
    zf = zipfile.ZipFile(io.BytesIO(fetch(GTFS_URL)))
    lines = build_lines(zf)
    (OUT_DIR / "subway-lines.geojson").write_text(json.dumps(lines, separators=(",", ":")))
    print(f"  {len(lines['features'])} routes")

    network = build_network(zf)
    (OUT_DIR / "network.json").write_text(json.dumps(network, separators=(",", ":")))
    print(f"  {len(network['segments'])} track segments")

    print("Downloading stations...")
    stations = build_stations()
    (OUT_DIR / "subway-stations.geojson").write_text(json.dumps(stations, separators=(",", ":")))
    print(f"  {len(stations['features'])} stations")


if __name__ == "__main__":
    main()
