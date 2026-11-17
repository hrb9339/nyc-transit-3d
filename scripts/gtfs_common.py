"""Small helpers for reading GTFS feeds (zip files in data/raw/gtfs/)."""

import zipfile
from datetime import date, datetime

import pandas as pd

from features_common import RAW, fetch

GTFS_DIR = RAW / "gtfs"


def gtfs_zip(name, url):
    """Download a GTFS zip once and open it."""
    path = GTFS_DIR / f"{name}.zip"
    fetch(url, path)
    return zipfile.ZipFile(path)


def read(zf, table, **kw):
    """Read one GTFS table; empty DataFrame if the file is missing."""
    if f"{table}.txt" not in zf.namelist():
        return pd.DataFrame()
    return pd.read_csv(zf.open(f"{table}.txt"), dtype=str, **kw)


def services_on(zf, day: date):
    """service_ids running on `day` (calendar.txt weekdays + calendar_dates.txt exceptions)."""
    ymd = day.strftime("%Y%m%d")
    active = set()
    cal = read(zf, "calendar")
    if len(cal):
        weekday = day.strftime("%A").lower()
        on = cal[(cal[weekday] == "1") & (cal.start_date <= ymd) & (cal.end_date >= ymd)]
        active |= set(on.service_id)
    cd = read(zf, "calendar_dates")
    if len(cd):
        today = cd[cd.date == ymd]
        active |= set(today[today.exception_type == "1"].service_id)
        active -= set(today[today.exception_type == "2"].service_id)
    return active


def departures(zf, day: date, hours=None):
    """Stop events on `day`: DataFrame of stop_id, trip_id, route_id, hour.

    `hours` = (start, end) keeps only departures in that clock-hour range.
    """
    trips = read(zf, "trips", usecols=["trip_id", "route_id", "service_id"])
    trips = trips[trips.service_id.isin(services_on(zf, day))]
    st = read(zf, "stop_times", usecols=["trip_id", "stop_id", "departure_time"])
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id")
    st["hour"] = st.departure_time.str.split(":").str[0].astype(float)
    if hours:
        st = st[(st.hour >= hours[0]) & (st.hour < hours[1])]
    return st[["stop_id", "trip_id", "route_id", "hour"]]


def parse_day(s):
    return datetime.strptime(s, "%Y-%m-%d").date()
