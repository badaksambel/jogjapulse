import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import affinity

import plot_style as ps

ps.apply()  # Helvetica/Arial, 10/9/8 pt as the other paper figures, final printed size

BOUNDARY_SHP = (
    "/Users/widyawan/Library/CloudStorage/GoogleDrive-widyawan@ugm.ac.id/"
    "My Drive/mypaper/2026 Scientific Data/data/batas wilayah DIY/diy_kalurahan_2023.shp"
)
CSV_PATH = "../release/output/home_maids_vs_population_kapanewon.csv"  # full-period home (22:00-05:59, >= 2 nights); legacy: ../release/data/validasi_kapanewon_des2021_apr2022_legacy_home.csv
OUTPUT_STEM = "choropleth_ratio_kapanewon_static"
N_BINS = 6
BACKGROUND_COLOR = "#f0f0f0"

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

# --- Rotate the whole map so its longest axis becomes horizontal ------------
union_geom = gdf.geometry.union_all()
mrr_coords = list(union_geom.minimum_rotated_rectangle.exterior.coords)

edges = []
for i in range(4):
    (x1, y1), (x2, y2) = mrr_coords[i], mrr_coords[i + 1]
    length = np.hypot(x2 - x1, y2 - y1)
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    edges.append((length, angle))

_, longest_angle = max(edges, key=lambda e: e[0])
rotation_deg = -longest_angle
# Normalize into (-90, 90] so the shape isn't flipped upside down
while rotation_deg > 90:
    rotation_deg -= 180
while rotation_deg <= -90:
    rotation_deg += 180

origin = union_geom.centroid
gdf["geometry"] = gdf["geometry"].apply(
    lambda geom: affinity.rotate(geom, rotation_deg/3, origin=origin, use_radians=False)
)

print(f"Rotated map by {rotation_deg:.1f} degrees to make it horizontal.")

# --- Quantile bins (equal-count classes) + Blues colormap -------------------
# GONDOMANAN is an extreme outlier (more MAIDs than residents), so it's excluded when computing
# the quantile bins/legend max — otherwise it alone would dominate the top bin.
# It's still plotted with everyone else, just clipped to the darkest color below.
ratio_no_outlier = gdf.loc[gdf["KEC_KAPN"] != "GONDOMANAN", "ratio"]
_, bin_edges = pd.qcut(ratio_no_outlier, q=N_BINS, retbins=True, duplicates="drop")
bin_edges = sorted(set(bin_edges.tolist()))
n_classes = len(bin_edges) - 1

cmap = plt.get_cmap("Blues", n_classes)
norm = mcolors.BoundaryNorm(bin_edges, cmap.N, clip=True)

# --- Plot ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(ps.FULL_WIDTH, 4.2))
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

gdf.plot(
    column="ratio",
    cmap=cmap,
    norm=norm,
    edgecolor="white",
    linewidth=0.6,
    ax=ax,
    legend=True,
    legend_kwds={
        "label": "MAIDs / population ratio (quantile bins)",
        "ticks": bin_edges,
        "format": "%.3f",
        "shrink": 0.7,
    },
)

ax.set_axis_off()
ax.set_title("MAIDs / population ratio per kapanewon")

plt.tight_layout()
ps.save(fig, OUTPUT_STEM, facecolor=BACKGROUND_COLOR)
plt.show()
