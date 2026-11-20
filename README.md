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

Use **View** in the side panel to switch between **3D city**, **Night lights** (top-down satellite view), **Hex grid**, **Station distance** (each hexagon colored by straight-line distance from its center to the nearest subway station), **Walk distance** (walking distance along streets to the nearest subway entrance, from `data/features/walk.csv`) **People** and **Ridership by hour**:

- **People**: hexagons colored by people per km² of land, with a switch between **Residents** (2020 Census blocks), **Workers** (LODES jobs), **Students** (`data/features/students.csv`; zoom in to see each school, colored by type) and **Daytime**, the weekday daytime population `Pop_day = W + S + a·R` from the proposal, with a slider for `a` (the share of residents assumed to stay near home, an assumption, default 0.5). All four share one color scale so they can be compared.
- **Ridership by hour**: a circle at each station complex whose area is its mean entries in the selected clock hour over the last 6 months (`data/ridership/hourly_web.json`). Switch weekday / Saturday / Sunday, drag the hour slider, click a bar in the citywide chart, or press **Play** to step through the day. Hover a station for its 24-hour profile. Ridership is drawn at stations rather than on hexagons because that is where it is counted (only 322 of the 1,402 hexagons have a station).

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

## Per-hexagon features

Each `scripts/features_*.py` writes one table to `data/features/`, one row per hexagon, keyed by `hex_id`. Every file starts with `#` comment lines listing its sources (with links), what each column means, and caveats. Read them with:

```python
pd.read_csv("data/features/hex_features.csv", comment="#")
```

| Table | Script | What it has | Sources |
|---|---|---|---|
| `neighborhoods.csv` | `features_neighborhoods.py` | 2020 NTA and CDTA of each hexagon, flag for non-residential NTAs (parks, airports, cemeteries) | NYC DCP NTAs |
| `landuse.csv` | `features_landuse.py` | Share of land by use, including land nobody lives on (parks, cemeteries, airports, rail yards); housing units; residential, office, retail and industrial floor area; built density | PLUTO 26v2 + MapPLUTO outlines for large lots |
| `destinations.csv` | `features_destinations.py` | Hospitals, clinics, colleges, museums, libraries, courthouses, government offices, ferry landings, shopping centers, major stadiums and arenas; distance to nearest hospital and college | NYC Facilities Database, PLUTO |
| `ntl_trend.csv` | `features_ntl_trend.py` | Mean night light for every year 2014–2025, change and growth per year | VIIRS annual V2.1 + V2.2 (Earth Engine) |
| `walk.csv` | `features_walk.py` | Walking distance along streets to the nearest subway entrance, detour ratio, area and stations reachable in an 800 m walk, intersection density | OpenStreetMap (BBBike extract), MTA subway entrances |
| `demographics.csv` | `features_demographics.py` | Population (2020 Census blocks) and jobs (LODES 2021), totals and per km² | Census 2020, LEHD LODES8 |
| `rail.csv` | `features_rail.py` | Walking distance to LIRR, Metro-North and PATH, nearest station and its weekday trains, stations within an 800 m walk, walk to any rail including the subway | MTA LIRR + Metro-North GTFS, PATH GTFS, OSM walking network |
| `bus.csv` | `features_bus.py` | Bus stops, routes (local, SBS, express) and weekday and 7-10 am buses per hexagon; distance to nearest stop | MTA bus GTFS (5 boroughs + MTA Bus) |
| `acs.csv` | `features_acs.py` | Income, rent, poverty, unemployment, car-free households, renters, age, how people commute and commute length (tracts spread to hexagons via 2020 block population and housing) | ACS 2020-2024 5-year summary files, 2020 Census blocks |
| `ridership.csv` | `features_ridership.py` | Weekday, Saturday and Sunday subway entries (2025) and the 2019 weekday baseline, summed per hexagon. **The model's target, not an input.** Per-station values are in `data/ridership/station_complexes.csv` | MTA Subway Hourly Ridership (data.ny.gov) |
| `students.csv` | `features_students.py` | Students enrolled in public/charter K-12 (CCD 2023-24), private K-12 (PSS 2021-22) and college (IPEDS fall 2023, excluding online-only) per hexagon, school and campus counts, students per km². Counted where the school is. Every school with its enrollment: `data/students/schools.csv` | NCES CCD, PSS, IPEDS |
| `midtown.csv` | `features_midtown.py` | Door-to-door subway time to Midtown (Times Sq / Grand Central) and Lower Manhattan (Fulton St): walk + wait + scheduled ride with transfers; straight-line distance to each. Ride time per station complex in `data/ridership/station_travel_time.csv` | MTA subway GTFS (`data/network.json`), Staten Island Ferry, OSM walking network |
| `proposals.csv` | `features_proposals.py` | Distance to the Interborough Express, Second Avenue Subway Phase 2 and Utica Avenue alignments. **For validation only, not a model input** | OpenStreetMap; line geometry in `data/proposals.geojson` |
| `hex_features.csv` | `features_merge.py` | All of the above plus the base grid columns (163 columns) | All of the above |

Rebuild everything (raw downloads go to `data/raw/`, which is git-ignored, about 350 MB):

```bash
pip install geopandas osmium scipy earthengine-api
curl -o data/raw/NewYork.osm.pbf https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf
python scripts/features_neighborhoods.py
python scripts/features_landuse.py        # before destinations (downloads PLUTO)
python scripts/features_destinations.py
python scripts/features_ntl_trend.py --project ml-ntl-dataset
python scripts/features_proposals.py
python scripts/features_walk.py           # ~2 minutes
python scripts/features_rail.py           # after walk (reuses its network)
python scripts/features_bus.py
python scripts/features_acs.py
python scripts/features_ridership.py      # ~15 minutes the first time (one request per day)
python scripts/features_ridership_hourly.py  # last 6 months by hour of day, ~2 minutes (not a feature table)
python scripts/features_students.py
python scripts/features_midtown.py        # after walk and ridership
python scripts/features_merge.py
```

Teammates adding a dataset: write `scripts/features_<name>.py` that calls `write_features()` from `scripts/features_common.py` (it enforces the header and a description for every column), then rerun `features_merge.py`.

## Model tables

`python scripts/build_training_table.py` (after the feature scripts) writes `data/model/`:

| File | Rows | What |
|---|---|---|
| `training.csv` | 424 subway station complexes | Target `weekday_riders` (train on `log_weekday_riders`), the features of the station's hexagon, and competition columns (other stations within 1 km, nearest other station) |
| `candidates.csv` | 1,080 hexagons with no station | The same features for a new station at the hexagon center, plus `uncovered_share_any_rail`: share of its 800 m walk area not already near the subway, LIRR, Metro-North or PATH |
| `feature_columns.csv` | one per column | Role of every column (`feature`, `target`, `group`, `info`, `ranking`, `excluded`) and why. Use the `feature` rows as model inputs |

Columns measuring distance to the existing subway are excluded: they are about zero for every station and large for every candidate, so they can't be learned from. Neighborhood (`nta2020`) is the grouping for holdout cross-validation. Staten Island has no training stations (the Railway collects fares at only 2 stations), so predictions there are extrapolation.

```python
import pandas as pd
cols = pd.read_csv("data/model/feature_columns.csv")
features = cols[cols.role == "feature"].column.tolist()
train = pd.read_csv("data/model/training.csv", comment="#")
X, y, groups = train[features], train.log_weekday_riders, train.nta2020
```

## First model

`python scripts/model_ridership.py` (needs `pip install scikit-learn tabulate`) trains on `data/model/training.csv` plus the student columns from `data/features/students.csv` (joined by `hex_id`, so the training table doesn't need rebuilding for them) and writes `data/model/results/`:

| File | What |
|---|---|
| `summary.md` | All results on one page |
| `cv_metrics.csv` | MAE / RMSE / R² (log scale), MAE in riders, median % error and rank correlation for each model, with whole neighborhoods held out (5-fold by NTA) and whole boroughs held out |
| `oof_predictions.csv` | Every station's held-out prediction from each model, sorted by Random Forest error |
| `importance.csv` | Permutation importance of each feature on held-out neighborhoods |
| `candidate_predictions.csv` | Every hexagon without a station: predicted riders (with 10th-90th percentile across trees), new riders = predicted × share of the 800 m walk area not already near rail, eligibility, rank |

Models, each adding one thing: the mean, a 1-feature fit on people in the hexagon, ridge regression on all features, and a Random Forest without and with the student counts. The script also trains without the three Second Avenue Subway stations (opened 2017) and predicts them.

## Files

- `index.html`, `style.css`, `app.js`: the frontend
- `realtime.js`: live train feed parsing, positioning and animation
- `views.js`: 3D city / night lights / hex grid / station distance views and legend
- `ridership.js`: the ridership-by-hour view and its controls
- `scripts/model_ridership.py`: baselines vs Random Forest, neighborhood holdout, candidate ranking (`data/model/results/`)
- `data/ridership/station_hourly.csv`: entries per station complex per hour (weekday / Saturday / Sunday), last 6 months; `hourly_web.json` is the same for the map
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
