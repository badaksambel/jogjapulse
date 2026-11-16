import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import numpy as np
import pandas as pd
import geopandas as gpd
from keplergl import KeplerGl

BOUNDARY_SHP = (
    "/Users/widyawan/Library/CloudStorage/GoogleDrive-widyawan@ugm.ac.id/"
    "My Drive/mypaper/2026 Scientific Data/data/batas wilayah DIY/diy_kalurahan_2023.shp"
)
CSV_PATH = "../release/output/home_maids_vs_population_kapanewon.csv"  # full-period home (22:00-05:59, >= 2 nights); legacy: ../release/data/validasi_kapanewon_des2021_apr2022_legacy_home.csv
OUTPUT_HTML = "choropleth_ratio_kapanewon.html"

# --- Load data -------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

gdf_kel = gpd.read_file(BOUNDARY_SHP)

# Dissolve kelurahan/kalurahan polygons up to kapanewon (kecamatan) level.
# KEC_KAPN names repeat across different KAB_KOTA (e.g. "JETIS" exists in both
# YOGYAKARTA and BANTUL), so dissolve on the combination of the two.
gdf_kapn = gdf_kel.dissolve(by=["KAB_KOTA", "KEC_KAPN"], as_index=False)
gdf_kapn = gdf_kapn[["KAB_KOTA", "KEC_KAPN", "geometry"]]

# --- Join ratio data ---------------------------------------------------------
gdf = gdf_kapn.merge(df, on=["KAB_KOTA", "KEC_KAPN"], how="left")

missing = gdf[gdf["ratio"].isna()]
if len(missing):
    print("Warning: no ratio data for:", missing[["KAB_KOTA", "KEC_KAPN"]].to_dict("records"))

gdf = gdf.dropna(subset=["ratio"]).reset_index(drop=True)

# --- Color mapping: log-transformed ratio (kepler's own "quantile" color
# scale, set below, then bins this field into equal-count classes) ---
gdf["log_ratio"] = np.log10(gdf["ratio"])

gdf_out = gdf[
    ["KAB_KOTA", "KEC_KAPN", "n_users", "pop_actual", "ratio",
     "log_ratio", "geometry"]
]

geojson = gdf_out.to_json()

# --- Build the keplergl map --------------------------------------------------
map_config = {
    "version": "v1",
    "config": {
        "visState": {
            "layers": [
                {
                    "id": "ratio_choropleth",
                    "type": "geojson",
                    "config": {
                        "dataId": "kapanewon_ratio",
                        "label": "User/Population Ratio",
                        "columns": {"geojson": "_geojson"},
                        "isVisible": True,
                        "visConfig": {
                            "opacity": 0.8,
                            "strokeOpacity": 0.8,
                            "thickness": 0.5,
                            "colorRange": {
                                "name": "Global Warming",
                                "type": "sequential",
                                "category": "Uber",
                                "colors": [
                                    "#5A1846", "#900C3F", "#C70039",
                                    "#E3611C", "#F1920E", "#FFC300",
                                ],
                            },
                            "filled": True,
                            "stroked": True,
                        },
                    },
                    "visualChannels": {
                        "colorField": {
                            "name": "log_ratio",
                            "type": "real",
                        },
                        "colorScale": "quantile",
                    },
                }
            ],
            "interactionConfig": {
                "tooltip": {
                    "fieldsToShow": {
                        "kapanewon_ratio": [
                            {"name": "KAB_KOTA", "format": None},
                            {"name": "KEC_KAPN", "format": None},
                            {"name": "n_users", "format": None},
                            {"name": "pop_actual", "format": None},
                            {"name": "ratio", "format": None},
                        ]
                    },
                    "enabled": True,
                }
            },
        }
    },
}

kepler_map = KeplerGl(height=700, data={"kapanewon_ratio": geojson}, config=map_config)
kepler_map.save_to_html(file_name=OUTPUT_HTML)

print(f"Saved map to {OUTPUT_HTML}")
print(gdf_out[["KAB_KOTA", "KEC_KAPN", "ratio", "log_ratio"]]
      .sort_values("ratio", ascending=False)
      .to_string(index=False))
