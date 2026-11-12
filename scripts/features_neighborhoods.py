"""Neighborhood of each hexagon (NYC 2020 Neighborhood Tabulation Areas).

Each hexagon gets the NTA that covers the largest part of its land, plus the
Community District Tabulation Area (CDTA) it sits in and the share of its land
in that NTA. Used for reporting results by neighborhood and, later, for the
area-based train/test split (hold out whole neighborhoods or CDTAs).

Writes:  data/features/neighborhoods.csv
Usage:   python scripts/features_neighborhoods.py
"""

import geopandas as gpd

from features_common import RAW, UTM, fetch, load_hexes, write_features

NTA_URL = "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=1000"
NTA_PAGE = "https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd"


def main():
    fetch(NTA_URL, RAW / "nta2020.geojson")
    ntas = gpd.read_file(RAW / "nta2020.geojson").to_crs(UTM)
    hexes = load_hexes().to_crs(UTM)
    hexes["hex_area"] = hexes.area

    pieces = gpd.overlay(hexes[["hex_id", "hex_area", "geometry"]],
                         ntas[["nta2020", "ntaname", "ntatype", "cdta2020", "cdtaname", "geometry"]],
                         how="intersection", keep_geom_type=True)
    pieces["share"] = pieces.area / pieces.hex_area
    best = pieces.sort_values("share", ascending=False).drop_duplicates("hex_id")
    counts = pieces[pieces.share >= 0.05].groupby("hex_id").size().rename("nta_count")

    out = hexes[["hex_id"]].merge(best, on="hex_id", how="left").merge(counts, on="hex_id", how="left")
    out["nta_share"] = out.share.round(3)
    out["nta_count"] = out.nta_count.fillna(1).astype(int)
    out["nonresidential_nta"] = out.ntatype.astype(str).ne("0").astype(int)
    out = out[["hex_id", "nta2020", "ntaname", "cdta2020", "cdtaname", "nta_share", "nta_count", "nonresidential_nta"]]

    print(f"{out.nta2020.nunique()} NTAs, {out.cdta2020.nunique()} CDTAs; "
          f"{(out.nta_share < 0.5).sum()} hexagons split with no NTA covering half")
    write_features(
        out, "neighborhoods",
        "Neighborhood (2020 NTA) of each hexagon",
        sources=[("NYC Department of City Planning, 2020 Neighborhood Tabulation Areas (NYC Open Data 9nt8-h7nd)", NTA_PAGE)],
        columns={
            "hex_id": "hexagon id (matches data/hexgrid.geojson)",
            "nta2020": "2020 NTA code covering the largest share of the hexagon's land",
            "ntaname": "NTA name",
            "cdta2020": "Community District Tabulation Area code (groups of NTAs; 71 citywide incl. large parks and airports)",
            "cdtaname": "CDTA name",
            "nta_share": "share of the hexagon's land inside that NTA (0-1)",
            "nta_count": "number of NTAs covering at least 5% of the hexagon",
            "nonresidential_nta": "1 if the main NTA is a non-residential NTA (ntatype 5-9: large park, cemetery, airport, military base, Rikers Island), else 0",
        },
        notes=["Areas computed in UTM 18N (EPSG:32618) on the shoreline-clipped hexagons."],
    )


if __name__ == "__main__":
    main()
