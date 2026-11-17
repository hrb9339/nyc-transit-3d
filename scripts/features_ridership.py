"""Subway ridership per station complex and per hexagon: the model's target.

MTA Subway Hourly Ridership counts entries (OMNY taps and MetroCard swipes) per
station complex per hour. It replaced the old turnstile data. This adds up a
full year per complex per day, then averages over ordinary weekdays (federal
holidays dropped), Saturdays and Sundays. 2019 is pulled the same way as a
pre-COVID baseline.

Data is fetched one month at a time, aggregated on the server (one row per
complex per day), and cached in data/raw/ridership/.

Writes:  data/ridership/station_complexes.csv  one row per station complex (training target)
         data/features/ridership.csv           one row per hexagon (sum over complexes inside it)
Usage:   python scripts/features_ridership.py [--year 2025] [--baseline 2019]
"""

import argparse
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from features_common import DATA, RAW, USER_AGENT, load_hexes, write_features

DATASETS = {   # first year covered -> (id, page)
    2017: ("t69i-h2me", "https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2017-2019/t69i-h2me"),
    2020: ("wujg-7c2s", "https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s"),
    2025: ("5wq4-mkjj", "https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-2025/5wq4-mkjj"),
}
CACHE = RAW / "ridership"


def dataset(year):
    return DATASETS[max(k for k in DATASETS if k <= year)]


def soql(ds, params, cache):
    """Run a SoQL query; cache only real results (the server sometimes answers {"error": "Timeout"})."""
    path = CACHE / cache
    if path.exists():
        return pd.DataFrame(json.loads(path.read_bytes()))
    url = f"https://data.ny.gov/resource/{ds}.json?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
        except OSError as e:   # timeouts, dropped connections
            print(f"  {cache}: {e}; retry {attempt + 1}")
            continue
        rows = json.loads(body)
        if isinstance(rows, list):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            return pd.DataFrame(rows)
        print(f"  {cache}: server said {rows.get('message', rows)}; retry {attempt + 1}")
    raise RuntimeError(f"query failed 5 times: {cache}")


def month_where(year, month):
    end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
    return f"transit_timestamp >= '{year}-{month:02d}-01' AND transit_timestamp < '{end}' AND transit_mode != 'tram'"


def daily(year):
    """Riders and transfers per complex per day for a whole year."""
    ds = dataset(year)[0]

    def one_day(day):
        nxt = day + pd.Timedelta(days=1)
        rows = soql(ds, {
            "$select": "station_complex_id, sum(ridership) AS riders, sum(transfers) AS transfers",
            "$where": f"transit_timestamp >= '{day:%Y-%m-%d}' AND transit_timestamp < '{nxt:%Y-%m-%d}' "
                      "AND transit_mode != 'tram'",
            "$group": "station_complex_id",
            "$limit": 10_000,
        }, f"day_{day:%Y%m%d}.json")
        rows["day"] = day.strftime("%Y-%m-%dT00:00:00.000")
        return rows

    def one_month(month):
        # A whole month aggregated on the server times out, so ask one day at a time
        # and keep the month as one cache file.
        path = CACHE / f"daily_{year}_{month:02d}.json"
        if path.exists():
            return pd.DataFrame(json.loads(path.read_bytes()))
        days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
        with ThreadPoolExecutor(6) as pool:
            m = pd.concat(pool.map(one_day, days), ignore_index=True)
        path.write_text(m.to_json(orient="records"))
        for day in days:   # the per-day files are now in the month file
            (CACHE / f"day_{day:%Y%m%d}.json").unlink(missing_ok=True)
        print(f"  {year}-{month:02d}: {len(m):,} complex-days")
        return m

    d = pd.concat([one_month(m) for m in range(1, 13)], ignore_index=True)
    d["day"] = pd.to_datetime(d.day)
    d[["riders", "transfers"]] = d[["riders", "transfers"]].astype(float)
    return d


def complexes(year):
    """Name, borough, mode and location of each complex (from one ordinary weekday;
    a month-long query times out)."""
    ds = dataset(year)[0]
    c = soql(ds, {
        "$select": "station_complex_id, station_complex, borough, transit_mode, latitude, longitude",
        "$where": f"transit_timestamp >= '{year}-10-15' AND transit_timestamp < '{year}-10-16' "
                  "AND transit_mode != 'tram'",
        "$group": "station_complex_id, station_complex, borough, transit_mode, latitude, longitude",
        "$limit": 10_000,
    }, f"complexes_{year}.json")
    c[["latitude", "longitude"]] = c[["latitude", "longitude"]].astype(float)
    return c.drop_duplicates("station_complex_id")


def averages(d, year):
    """Mean riders per complex on ordinary weekdays, Saturdays, Sundays."""
    holidays = USFederalHolidayCalendar().holidays(f"{year}-01-01", f"{year}-12-31")
    d = d[~d.day.isin(holidays)]
    dow = d.day.dt.dayofweek
    wk = d[dow < 5].groupby("station_complex_id")
    return pd.DataFrame({
        "weekday_riders": wk.riders.mean(),
        "weekday_riders_median": wk.riders.median(),
        "weekday_transfers": wk.transfers.mean(),
        "saturday_riders": d[dow == 5].groupby("station_complex_id").riders.mean(),
        "sunday_riders": d[dow == 6].groupby("station_complex_id").riders.mean(),
        "days_with_data": d.groupby("station_complex_id").day.nunique(),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--baseline", type=int, default=2019)
    args = ap.parse_args()
    y, b = args.year, args.baseline

    print(f"Fetching {y} and {b} daily ridership (one request per day, cached by month)...")
    cur = averages(daily(y), y)
    base = averages(daily(b), b)
    info = complexes(y).set_index("station_complex_id")

    st = info.join(cur, how="inner")
    st[f"weekday_riders_{b}"] = base.weekday_riders.reindex(st.index)
    st[f"recovery_vs_{b}"] = (st.weekday_riders / st[f"weekday_riders_{b}"]).round(3)
    st = st.reset_index()

    hexes = load_hexes()
    # Hexagon each complex falls in; a few waterfront stations sit just outside the
    # shoreline-clipped cells, so they take the nearest cell (within 300 m).
    pts = gpd.GeoDataFrame(st[[]], geometry=gpd.points_from_xy(st.longitude, st.latitude), crs="EPSG:4326")
    joined = gpd.sjoin_nearest(pts.to_crs("EPSG:32618"), hexes[["hex_id", "geometry"]].to_crs("EPSG:32618"),
                               how="left", max_distance=300)
    st["hex_id"] = joined.groupby(level=0).hex_id.first().reindex(st.index).values

    num = ["weekday_riders", "weekday_riders_median", "weekday_transfers", "saturday_riders", "sunday_riders",
           f"weekday_riders_{b}"]
    st[num] = st[num].round(0)
    st = st.sort_values("weekday_riders", ascending=False)
    cols = ["station_complex_id", "station_complex", "borough", "transit_mode", "latitude", "longitude", "hex_id",
            *num, f"recovery_vs_{b}", "days_with_data"]
    st = st[cols]

    sources = [
        (f"MTA Subway Hourly Ridership: Beginning 2025 (data.ny.gov {DATASETS[2025][0]})", DATASETS[2025][1]),
        (f"MTA Subway Hourly Ridership: 2017-2019 (data.ny.gov {DATASETS[2017][0]})", DATASETS[2017][1]),
        (f"MTA Subway Hourly Ridership: 2020-2024 (data.ny.gov {DATASETS[2020][0]})", DATASETS[2020][1]),
    ]
    common_notes = [
        "Ridership = entries (OMNY taps + MetroCard swipes), not exits. Free transfers into the subway from "
        "buses or other lines are counted in 'transfers', which are part of ridership.",
        f"Averages over {y} (and {b} for the baseline); US federal holidays are dropped from the weekday average.",
        "Staten Island Railway fares are collected only at St George and Tompkinsville, so the other SIR stations "
        "show little or no ridership (transit_mode = staten_island_railway). Treat them separately in the model.",
        "Roosevelt Island Tramway rows are excluded.",
        "Averages count only days with data, so a station closed for part of the year (the Rockaway A/S "
        "stations were shut for months in 2025) gets its average over open days; see days_with_data.",
        "Complexes that opened, closed or were renumbered after the baseline year have no baseline value.",
    ]
    write_features(
        st, "station_complexes",
        f"Subway ridership per station complex, {y} (training target)",
        sources=sources,
        columns={
            "station_complex_id": "MTA station complex id",
            "station_complex": "station complex name (lines in brackets)",
            "borough": "borough",
            "transit_mode": "subway or staten_island_railway",
            "latitude": "complex location (MTA)", "longitude": "complex location (MTA)",
            "hex_id": "hexagon the complex falls in (matches data/hexgrid.geojson)",
            "weekday_riders": f"mean entries per ordinary weekday, {y}",
            "weekday_riders_median": f"median entries per ordinary weekday, {y} (robust to closures)",
            "weekday_transfers": f"mean free transfers counted per weekday, {y}",
            "saturday_riders": f"mean entries per Saturday, {y}",
            "sunday_riders": f"mean entries per Sunday, {y}",
            f"weekday_riders_{b}": f"mean entries per ordinary weekday, {b} (pre-COVID baseline)",
            f"recovery_vs_{b}": f"weekday_riders / weekday_riders_{b}",
            "days_with_data": f"days in {y} with any ridership recorded",
        },
        notes=common_notes,
        path=DATA / "ridership" / "station_complexes.csv",
    )

    by_hex = st.groupby("hex_id")
    out = pd.DataFrame({"hex_id": hexes.hex_id}).set_index("hex_id")
    out["ridership_complexes"] = by_hex.size()
    for c in ["weekday_riders", "saturday_riders", "sunday_riders", f"weekday_riders_{b}"]:
        out[c] = by_hex[c].sum(min_count=1)
    out = out.fillna({"ridership_complexes": 0}).astype({"ridership_complexes": int})
    out[f"recovery_vs_{b}"] = (out.weekday_riders / out[f"weekday_riders_{b}"]).round(3)
    out = out.reset_index()

    print(f"{len(st)} complexes; {st.weekday_riders.sum():,.0f} weekday entries in {y} "
          f"({st[f'weekday_riders_{b}'].sum():,.0f} in {b}); {out.ridership_complexes.gt(0).sum()} hexagons with ridership")
    print(st.head(5)[["station_complex", "weekday_riders", f"recovery_vs_{b}"]].to_string(index=False))

    write_features(
        out, "ridership",
        f"Subway ridership per hexagon, {y} (target, not a model input)",
        sources=sources,
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "ridership_complexes": "station complexes located in the hexagon",
            "weekday_riders": f"weekday entries summed over those complexes, {y} (blank = no station)",
            "saturday_riders": f"Saturday entries summed, {y}",
            "sunday_riders": f"Sunday entries summed, {y}",
            f"weekday_riders_{b}": f"weekday entries summed, {b}",
            f"recovery_vs_{b}": f"weekday_riders / weekday_riders_{b}",
        },
        notes=["This is what the model learns to predict. Do not use it (or anything computed from it) as an "
               "input feature.",
               "Per-complex values, which are the real training rows, are in data/ridership/station_complexes.csv.",
               *common_notes],
    )


if __name__ == "__main__":
    main()
