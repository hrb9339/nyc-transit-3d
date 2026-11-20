"""Students per hexagon: K-12 schools (public, charter, private) and colleges.

Students are the third people count next to residents (demographics.csv) and
workers (jobs in demographics.csv). Every school and college is a point with an
enrollment, so the counts are added up directly into the hexagon each point falls in.

  - Public and charter K-12: NCES Common Core of Data (CCD) school directory,
    2023-24, total enrollment (includes pre-K where the school has it).
  - Private K-12: NCES Private School Universe Survey (PSS), 2021-22, the latest release.
  - Colleges: IPEDS Fall 2023 enrollment (all students, EF2023A) minus students
    enrolled exclusively in distance education (EF2023A_DIST), located at the
    institution's address (HD2023).

Writes:  data/features/students.csv     one row per hexagon
         data/students/schools.csv      one row per school or college (the points)
         data/students/students_web.json the points, compact, for the map's Students view
Usage:   python scripts/features_students.py
"""

import io
import json
import zipfile

import geopandas as gpd
import pandas as pd

from features_common import DATA, RAW, UTM, fetch, load_hexes, write_features

NYC_COUNTIES = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan", "36081": "Queens",
                "36085": "Staten Island"}
CCD_URL = "https://educationdata.urban.org/api/v1/schools/ccd/directory/2023/?fips=36"
CCD_PAGE = "https://nces.ed.gov/ccd/files.asp"
PSS_URL = "https://nces.ed.gov/surveys/pss/zip/pss2122_pu_csv.zip"
PSS_PAGE = "https://nces.ed.gov/surveys/pss/pssdata.asp"
IPEDS = "https://nces.ed.gov/ipeds/datacenter/data/{}.zip"
IPEDS_PAGE = "https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx"
CACHE = RAW / "students"
OPEN_STATUSES = {1, 3, 4, 5, 8}   # CCD school_status: open, new, added, changed agency, reopened


def read_zip_csv(name, url):
    body = fetch(url, CACHE / f"{name}.zip")
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        member = next(n for n in z.namelist() if n.lower().endswith(".csv") and "_rv" not in n.lower())
        return pd.read_csv(z.open(member), encoding="latin-1", dtype=str, skipinitialspace=True)


def public_schools():
    data = json.loads(fetch(CCD_URL, CACHE / "ccd_2023_ny.json"))
    assert data.get("next") is None, "CCD answer was paginated; fetch every page"
    s = pd.DataFrame(data["results"])
    s = s[s.county_code.astype(str).isin(NYC_COUNTIES) & s.school_status.isin(OPEN_STATUSES)
          & (s.enrollment.fillna(0) > 0)]
    return pd.DataFrame({
        "school_id": "ccd:" + s.ncessch.astype(str),
        "name": s.school_name.str.title(),
        "kind": s.charter.map({1: "charter"}).fillna("public"),
        "latitude": s.latitude, "longitude": s.longitude,
        "students": s.enrollment.astype(int),
        "year": "2023-24",
    })


def private_schools():
    p = read_zip_csv("pss2122", PSS_URL)
    p = p[(p.PSTABB == "NY") & ("36" + p.PCNTY.str.zfill(3)).isin(NYC_COUNTIES)]
    p = p.assign(n=pd.to_numeric(p.NUMSTUDS, errors="coerce"))
    p = p[p.n > 0]
    return pd.DataFrame({
        "school_id": "pss:" + p.PPIN,
        "name": p.PINST.str.title(),
        "kind": "private",
        "latitude": p.LATITUDE22.astype(float), "longitude": p.LONGITUDE22.astype(float),
        "students": p.n.round().astype(int),
        "year": "2021-22",
    })


def colleges():
    hd = read_zip_csv("HD2023", IPEDS.format("HD2023"))
    hd.columns = hd.columns.str.replace("﻿", "").str.replace("ï»¿", "")
    ef = read_zip_csv("EF2023A", IPEDS.format("EF2023A"))
    de = read_zip_csv("EF2023A_DIST", IPEDS.format("EF2023A_DIST"))
    hd = hd[hd.COUNTYCD.isin(NYC_COUNTIES)]
    total = ef[ef.EFALEVEL.str.strip() == "1"].set_index("UNITID").EFTOTLT.astype(float)
    online = de[de.EFDELEV.str.strip() == "1"].set_index("UNITID").EFDEEXC.astype(float)
    c = hd.set_index("UNITID").join(total.rename("total"), how="inner").join(online.rename("online"))
    c["students"] = (c.total - c.online.fillna(0)).clip(lower=0)
    c = c[c.students > 0].reset_index()
    return pd.DataFrame({
        "school_id": "ipeds:" + c.UNITID,
        "name": c.INSTNM,
        "kind": "college",
        "latitude": c.LATITUDE.astype(float), "longitude": c.LONGITUD.astype(float),
        "students": c.students.round().astype(int),
        "year": "fall 2023",
    })


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    pts = pd.concat([public_schools(), private_schools(), colleges()], ignore_index=True)

    hexes = load_hexes()
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.longitude, pts.latitude), crs="EPSG:4326")
    # Each point goes to the hexagon it falls in; a few waterfront campuses sit just outside the
    # shoreline-clipped cells, so they take the nearest cell within 300 m.
    joined = gpd.sjoin_nearest(g.to_crs(UTM), hexes[["hex_id", "geometry"]].to_crs(UTM), how="left",
                               max_distance=300)
    pts["hex_id"] = joined.groupby(level=0).hex_id.first().reindex(pts.index).values
    outside = pts.hex_id.isna()
    if outside.any():
        print(f"  {outside.sum()} points ({pts.students[outside].sum():,} students) are outside the grid and dropped")
    pts = pts.sort_values(["kind", "students"], ascending=[True, False])

    write_features(
        pts, "students_points",
        "Schools and colleges in NYC with enrollment (one row per school or college)",
        sources=[
            ("NCES Common Core of Data school directory 2023-24, via Urban Institute Education Data API", CCD_PAGE),
            ("NCES Private School Universe Survey 2021-22 (pss2122_pu)", PSS_PAGE),
            ("IPEDS HD2023, EF2023A and EF2023A_DIST (Fall 2023 enrollment)", IPEDS_PAGE),
        ],
        columns={
            "school_id": "source and id (ccd: NCES school id, pss: PSS PPIN, ipeds: IPEDS UNITID)",
            "name": "school or institution name",
            "kind": "public, charter, private (K-12) or college",
            "latitude": "location (source)", "longitude": "location (source)",
            "students": "enrollment (colleges: excluding students enrolled only online)",
            "year": "school year of the enrollment count",
            "hex_id": "hexagon the point falls in (blank = outside the grid)",
        },
        path=DATA / "students" / "schools.csv",
        script="features_students.py",
    )

    out = pd.DataFrame({"hex_id": hexes.hex_id}).set_index("hex_id")
    k12 = pts[pts.kind != "college"]
    out["students_k12_public"] = pts[pts.kind.isin(["public", "charter"])].groupby("hex_id").students.sum()
    out["students_k12_private"] = pts[pts.kind == "private"].groupby("hex_id").students.sum()
    out["students_college"] = pts[pts.kind == "college"].groupby("hex_id").students.sum()
    out = out.fillna(0).astype(int)
    out["students"] = out.sum(axis=1)
    out["k12_schools"] = k12.groupby("hex_id").size().reindex(out.index, fill_value=0)
    out["college_campuses"] = pts[pts.kind == "college"].groupby("hex_id").size().reindex(out.index, fill_value=0)
    land = hexes.set_index("hex_id").land_km2
    out["student_density"] = (out.students / land).round(0)
    out = out.reset_index()

    write_features(
        out, "students",
        "Students per hexagon: K-12 (public, charter, private) and college enrollment",
        sources=[
            ("NCES Common Core of Data school directory 2023-24 (public and charter enrollment)", CCD_PAGE),
            ("NCES Private School Universe Survey 2021-22", PSS_PAGE),
            ("IPEDS Fall 2023 enrollment (EF2023A, EF2023A_DIST) and institution locations (HD2023)", IPEDS_PAGE),
        ],
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "students_k12_public": "students enrolled in public and charter schools in the hexagon (CCD 2023-24)",
            "students_k12_private": "students enrolled in private schools in the hexagon (PSS 2021-22)",
            "students_college": "college students in the hexagon (IPEDS fall 2023, excluding online-only)",
            "students": "students_k12_public + students_k12_private + students_college",
            "k12_schools": "K-12 schools (public, charter, private) located in the hexagon",
            "college_campuses": "IPEDS institutions located in the hexagon",
            "student_density": "students / land_km2",
        },
        notes=[
            "Each school counts where it is, not where its students live.",
            "Several schools sharing one building (common in NYC) are separate rows at the same point.",
            "IPEDS reports each institution at one address, so multi-campus colleges put all their students at "
            "the main campus (e.g. Fordham's Lincoln Center students are counted at Rose Hill; Touro and Pace "
            "students at other campuses, some outside NYC, are counted at the NYC address). CUNY colleges are "
            "separate institutions, so they are placed correctly.",
            "Points list with names: data/students/schools.csv.",
        ],
    )

    web = {
        "sources": "NCES CCD 2023-24, PSS 2021-22, IPEDS fall 2023",
        "totals": pts.groupby("kind").students.sum().to_dict(),
        "points": [[round(r.longitude, 5), round(r.latitude, 5), int(r.students), r.kind, r.name]
                   for r in pts.itertuples()],
    }
    (DATA / "students" / "students_web.json").write_text(json.dumps(web, separators=(",", ":")))
    print("students by kind:", {k: f"{v:,}" for k, v in web["totals"].items()},
          f"total {pts.students.sum():,}; {(out.students > 0).sum()} hexagons with students")


if __name__ == "__main__":
    main()
