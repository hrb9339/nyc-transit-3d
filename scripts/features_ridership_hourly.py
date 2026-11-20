"""Subway ridership by hour of day per station complex, over the last 6 months.

Uses MTA Subway Hourly Ridership (data.ny.gov, Beginning 2025), which counts
entries (OMNY taps + MetroCard swipes) per station complex per hour. The most
recent 6 months in the dataset are pulled one day at a time, aggregated on the
server to one row per complex per hour, and cached in data/raw/ridership/hourly/.

For every complex this averages the entries in each clock hour (0-23) over
ordinary weekdays (federal holidays dropped), Saturdays and Sundays. A complex's
average counts only the days it had any ridership, so a station closed for a few
weeks is not dragged down by the closed days.

This is the time-of-day profile of the model's target. It is not a model input,
so it is written to data/ridership/, not data/features/.

Writes:  data/ridership/station_hourly.csv  one row per complex and day type, 24 hour columns
         data/ridership/hourly_web.json     the same, compact, for the map's Ridership view
Usage:   python scripts/features_ridership_hourly.py [--months 6]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from features_common import DATA, write_features
from features_ridership import CACHE, DATASETS, complexes, soql

DS, PAGE = DATASETS[2025]
HOURLY_CACHE = CACHE / "hourly"
DAY_TYPES = ["weekday", "saturday", "sunday"]
HOURS = [f"h{h:02d}" for h in range(24)]


def latest_day():
    """Last day in the dataset that has its 23:00 hour (i.e. is complete)."""
    mx = soql(DS, {"$select": "max(transit_timestamp) AS mx"}, "_latest_probe.json")
    (CACHE / "_latest_probe.json").unlink(missing_ok=True)   # never cache "latest"
    ts = pd.Timestamp(mx.mx.iloc[0])
    return ts.normalize() if ts.hour == 23 else ts.normalize() - pd.Timedelta(days=1)


def one_day(day):
    nxt = day + pd.Timedelta(days=1)
    rows = soql(DS, {
        "$select": "station_complex_id, date_extract_hh(transit_timestamp) AS hour, sum(ridership) AS riders",
        "$where": f"transit_timestamp >= '{day:%Y-%m-%d}' AND transit_timestamp < '{nxt:%Y-%m-%d}' "
                  "AND transit_mode != 'tram'",
        "$group": "station_complex_id, hour",
        "$limit": 50_000,
    }, f"hourly/{day:%Y%m%d}.json")
    rows["day"] = day
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    args = ap.parse_args()

    end = latest_day()
    start = end - pd.DateOffset(months=args.months) + pd.Timedelta(days=1)
    days = pd.date_range(start, end)
    print(f"Fetching hourly ridership {start:%Y-%m-%d} to {end:%Y-%m-%d} ({len(days)} days, cached per day)...")
    HOURLY_CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as pool:
        d = pd.concat(pool.map(one_day, days), ignore_index=True)
    d["hour"] = d.hour.astype(int)
    d["riders"] = d.riders.astype(float)

    holidays = USFederalHolidayCalendar().holidays(start, end)
    dow = d.day.dt.day_of_week
    d["day_type"] = "weekday"
    d.loc[dow == 5, "day_type"] = "saturday"
    d.loc[dow == 6, "day_type"] = "sunday"
    d = d[~(d.day.isin(holidays) & (d.day_type == "weekday"))]

    # Days each complex was open, per day type, then mean entries per hour over those days.
    open_days = d.groupby(["station_complex_id", "day_type"]).day.nunique().rename("days")
    tot = d.groupby(["station_complex_id", "day_type", "hour"]).riders.sum().unstack("hour", fill_value=0)
    tot = tot.reindex(columns=range(24), fill_value=0)
    avg = tot.div(open_days, axis=0).round(1)
    avg.columns = HOURS
    avg["daily_riders"] = avg[HOURS].sum(axis=1).round(0)
    avg["days"] = open_days
    avg = avg.reset_index()

    info = complexes(end.year)[["station_complex_id", "station_complex", "borough", "transit_mode",
                                "latitude", "longitude"]]
    st = info.merge(avg, on="station_complex_id", how="inner")
    missing = set(avg.station_complex_id) - set(info.station_complex_id)
    if missing:
        print(f"  {len(missing)} complexes with ridership but no location on the reference day were dropped: "
              f"{sorted(missing)[:10]}")
    st["day_type"] = pd.Categorical(st.day_type, DAY_TYPES, ordered=True)
    st = st.assign(_id=st.station_complex_id.astype(int)).sort_values(["_id", "day_type"])
    st["day_type"] = st.day_type.astype(str)
    st = st[["station_complex_id", "station_complex", "borough", "transit_mode", "latitude", "longitude",
             "day_type", "days", "daily_riders", *HOURS]]

    n_days = d.groupby("day_type").day.nunique().reindex(DAY_TYPES)
    period = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    write_features(
        st, "ridership_hourly",
        f"Subway entries by hour of day per station complex, {period}",
        sources=[(f"MTA Subway Hourly Ridership: Beginning 2025 (data.ny.gov {DS})", PAGE)],
        columns={
            "station_complex_id": "MTA station complex id (matches data/ridership/station_complexes.csv)",
            "station_complex": "station complex name (lines in brackets)",
            "borough": "borough",
            "transit_mode": "subway or staten_island_railway",
            "latitude": "complex location (MTA)", "longitude": "complex location (MTA)",
            "day_type": "weekday (federal holidays dropped), saturday or sunday",
            "days": "days of this type in the period on which the complex had any entries",
            "daily_riders": "mean entries per day of this type (sum of the 24 hour columns)",
            **{h: f"mean entries in the hour starting {int(h[1:]):02d}:00 (local time)" for h in HOURS},
        },
        notes=[
            f"Period: the last {args.months} months in the dataset, {period}: "
            f"{n_days.weekday} weekdays, {n_days.saturday} Saturdays, {n_days.sunday} Sundays.",
            "Entries only (OMNY taps + MetroCard swipes), not exits. Roosevelt Island Tramway excluded.",
            "Staten Island Railway collects fares only at St George and Tompkinsville, so other SIR stations show "
            "little or nothing.",
            "Time-of-day profile of the model's target: not a model input.",
        ],
        path=DATA / "ridership" / "station_hourly.csv",
        script="features_ridership_hourly.py",
    )

    # Compact version for the browser: one entry per complex, 24 values per day type.
    web = {
        "period": period,
        "days": {t: int(n_days[t]) for t in DAY_TYPES},
        "source": f"MTA Subway Hourly Ridership (data.ny.gov {DS})",
        "citywide": {t: [round(v) for v in st[st.day_type == t][HOURS].sum()] for t in DAY_TYPES},
        "complexes": [],
    }
    for cid, g in st.groupby("station_complex_id", sort=False):
        r = g.iloc[0]
        entry = {"id": cid, "name": r.station_complex, "lon": round(r.longitude, 5), "lat": round(r.latitude, 5)}
        for t in DAY_TYPES:
            row = g[g.day_type == t]
            entry[t] = [round(v) for v in row[HOURS].iloc[0]] if len(row) else [0] * 24
        web["complexes"].append(entry)
    out = DATA / "ridership" / "hourly_web.json"
    out.write_text(json.dumps(web, separators=(",", ":")))
    print(f"wrote {out.relative_to(DATA.parent)}: {len(web['complexes'])} complexes")

    wk = pd.Series(web["citywide"]["weekday"])
    print(f"Citywide weekday entries: {wk.sum():,.0f}; busiest hour {wk.idxmax():02d}:00 ({wk.max():,.0f})")


if __name__ == "__main__":
    main()
