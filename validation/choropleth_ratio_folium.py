import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import json

import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib
import pandas as pd

BOUNDARY_SHP = (
    "/Users/widyawan/Library/CloudStorage/GoogleDrive-widyawan@ugm.ac.id/"
    "My Drive/mypaper/2026 Scientific Data/data/batas wilayah DIY/diy_kalurahan_2023.shp"
)
CSV_PATH = "../release/output/home_maids_vs_population_kapanewon.csv"  # full-period home (22:00-05:59, >= 2 nights); legacy: ../release/data/validasi_kapanewon_des2021_apr2022_legacy_home.csv
OUTPUT_HTML = "choropleth_ratio_kapanewon.html"
N_BINS = 6

# --- Load data ---------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

gdf_kel = gpd.read_file(BOUNDARY_SHP)

# Dissolve kelurahan/kalurahan polygons up to kapanewon (kecamatan) level.
# KEC_KAPN names repeat across different KAB_KOTA (e.g. "JETIS" exists in both
# YOGYAKARTA and BANTUL), so dissolve on the combination of the two.
gdf_kapn = gdf_kel.dissolve(by=["KAB_KOTA", "KEC_KAPN"], as_index=False)
gdf_kapn = gdf_kapn[["KAB_KOTA", "KEC_KAPN", "geometry"]]

# --- Join ratio data -----------------------------------------------------------
gdf = gdf_kapn.merge(df, on=["KAB_KOTA", "KEC_KAPN"], how="left")

missing = gdf[gdf["ratio"].isna()]
if len(missing):
    print("Warning: no ratio data for:", missing[["KAB_KOTA", "KEC_KAPN"]].to_dict("records"))

gdf = gdf.dropna(subset=["ratio"]).reset_index(drop=True)

# Unique join key since KEC_KAPN names repeat across KAB_KOTA
gdf["join_key"] = gdf["KAB_KOTA"] + " - " + gdf["KEC_KAPN"]

# --- Quantile bins (equal-count classes) instead of a linear color scale -----
_, bin_edges = pd.qcut(gdf["ratio"], q=N_BINS, retbins=True, duplicates="drop")
bin_edges = sorted(set(bin_edges.tolist()))
n_classes = len(bin_edges) - 1

# Sample a bluish colormap for each quantile class
class_colors = [
    matplotlib.colors.rgb2hex(c)
    for c in matplotlib.colormaps["Blues"](
        [i / (n_classes - 1) for i in range(n_classes)]
    )
]
colormap = cm.StepColormap(
    colors=class_colors,
    index=bin_edges,
    vmin=bin_edges[0],
    vmax=bin_edges[-1],
    caption="User / Population Ratio (quantile bins)",
)


def ratio_to_color(ratio_value):
    return colormap(ratio_value)


# --- Build the folium map (no basemap, zoomed to DIY extent only) ------------
m = folium.Map(tiles=None, zoom_control=True)

minx, miny, maxx, maxy = gdf.total_bounds
m.fit_bounds([[miny, minx], [maxy, maxx]])

geojson_data = json.loads(gdf.to_json())

folium.GeoJson(
    geojson_data,
    style_function=lambda feature: {
        "fillColor": ratio_to_color(feature["properties"]["ratio"]),
        "color": "white",
        "weight": 0.7,
        "fillOpacity": 0.85,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["KAB_KOTA", "KEC_KAPN", "n_users", "pop_actual", "ratio"],
        aliases=["Kabupaten/Kota", "Kapanewon", "Jumlah user", "Populasi", "Rasio"],
        localize=True,
    ),
).add_to(m)

colormap.add_to(m)

m.save(OUTPUT_HTML)

print(f"Saved map to {OUTPUT_HTML}")
print("Quantile bin edges:", [round(b, 4) for b in bin_edges])
print(gdf[["KAB_KOTA", "KEC_KAPN", "ratio"]].sort_values("ratio", ascending=False).to_string(index=False))
