# NYC Transit 3D

A 3D map of New York City with every subway line drawn in its official MTA color, on top of extruded 3D buildings.

This is the visualization frontend for the transit-gap analysis project (population density → Random Forest classification → candidate new lines).

## Run it

The page loads local GeoJSON with `fetch`, so it has to be served over HTTP. Opening `index.html` directly from disk won't work.

```bash
cd nyc-transit-3d
python -m http.server 8000
```

Then open http://localhost:8000.

No API keys are needed.

## Stack

| Piece | What |
|---|---|
| Rendering | [MapLibre GL JS](https://maplibre.org/) 4.7.1 (WebGL, open source) |
| Basemap + 3D buildings | [OpenFreeMap](https://openfreemap.org/) "dark" style, OpenStreetMap building heights |
| Subway lines | MTA static GTFS `shapes.txt` + `routes.txt` (official colors) |
| Stations | MTA Subway Stations, data.ny.gov (`39hk-dx4f`) |

## Live trains

`realtime.js` pulls the 8 MTA GTFS-realtime feeds straight from the browser every 30 seconds (they're free, keyless and CORS-enabled) and decodes the protobuf with a small built-in reader.

The feeds don't include GPS. They give each train's upcoming stops and predicted arrival times. Each train is placed between its previous and next station by time, along the real track geometry in `data/network.json`, and animated smoothly between refreshes. Positions are estimates: a train held between stations will keep moving on the map until the next feed update corrects it.

## Hex grid and night-time light

Use **View** in the side panel to switch between **3D city**, **Night lights** (top-down satellite view) and **Hex grid**.

**Hex grid** (`scripts/build_hexgrid.py`): 1,402 pointy-top hexagons with a 500 m circumradius (0.65 km² each), laid out in UTM 18N so all cells are the same true size. Cells are cut to the shoreline (NYC's shoreline-clipped borough boundaries) and cells that are 90%+ water are dropped, so the grid traces the five boroughs. Each records its borough, land area, the subway stations inside it (each station counted in exactly one cell) and the distance from its center to the nearest station. 322 hexagons contain a station; 1,080 don't. With ~464 m satellite pixels, each cell covers about 4 pixels, so neighbouring cells' night-light values are partly shared.

**Night-time light** (`scripts/gee_ntl.py`, Google Earth Engine project `ml-ntl-dataset`):

- Satellite: VIIRS Day/Night Band, ~464 m pixels, radiance in nW/cm²/sr
- Main: annual composite V2.2, latest year (2025), band `average_masked`
- Cross-check: median of the latest 12 monthly composites (Sep 2025 – Aug 2026); correlates 0.97 with the annual values

Per hexagon (land part only): mean, median and max radiance; sum of lights (radiance summed over pixels, weighted by the share of each pixel inside the hexagon); pixel count; lit pixels (> 1); bright pixels (> 50).

Outputs in `data/ntl/`:

- `hex_ntl.csv`: one row per hexagon with all the values above
- `viirs_annual_2025_nyc.tif`, `viirs_recent12_nyc.tif`: GeoTIFF rasters for use in QGIS or Python
- `ntl_web.png` + `ntl_web.json`: colored image for the website

To rerun: `earthengine authenticate` once, then `python scripts/gee_ntl.py --project ml-ntl-dataset`.

## Files

- `index.html`, `style.css`, `app.js`: the frontend
- `realtime.js`: live train feed parsing, positioning and animation
- `views.js`: 3D city / night lights / hex grid views and legend
- `scripts/build_hexgrid.py`, `scripts/gee_ntl.py`: hexagon grid and Earth Engine night-light statistics
- `data/hexgrid.geojson` (with night-light values), `data/hexgrid_land.geojson` (land-clipped, for statistics)
- `data/network.json`: station-to-station track segments + scheduled travel times
- `data/subway-lines.geojson`: one MultiLineString per route
- `data/subway-stations.geojson`: 496 stations with routes, borough, ADA status
- `scripts/build_data.py`: regenerates both GeoJSON files from the MTA sources

## Refresh the data

```bash
python scripts/build_data.py
```

## Features

- Live trains: hover a dot for its route, destination and next stop
- Toggle individual lines with the colored route bullets
- Hover a line for its name; click a station for its routes and accessibility
- 3D buildings and station layers can be switched on and off
- Camera presets (Midtown, Lower Manhattan, Brooklyn, Queens/LIC, Bronx) and auto-rotate
- Right-click and drag (or Ctrl + drag) to tilt and rotate the view

## Adding analysis layers later

Add a GeoJSON source and a `fill-extrusion` or `fill` layer in `app.js` for things like census-tract density or the High/Medium/Low classification. Extruding polygons by population density works well in 3D.
