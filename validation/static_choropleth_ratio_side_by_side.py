import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
from paths import fig_path, out_path  # noqa: E402  inputs under JOGJAPULSE_OUT, figures to JOGJAPULSE_FIG

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import geopandas as gpd
import pandas as pd

import plot_style as ps

ps.apply()  # Helvetica/Arial, 10/9/8 pt as the other paper figures, final printed size

BOUNDARY_SHP = "../release/data/diy_kalurahan_2023/diy_kalurahan_2023.shp"  # Geoportal DIY 2023 kalurahan
CSV_PATH = out_path("output", "home_maids_vs_population_kapanewon.csv")  # full-period home (22:00-05:59, >= 2 nights), default coordinates removed
OUTPUT_STEM = "choropleth_ratio_side_by_side"
CRS = 32749  # UTM 49S: metres, north up, so the scale bar and north arrow are exact
N_BINS = 6
BACKGROUND_COLOR = "#f0f0f0"
KAB_NAMES = {"YOGYAKARTA": "Kota Yogyakarta", "KULON PROGO": "Kulon Progo", "BANTUL": "Bantul",
             "GUNUNGKIDUL": "Gunungkidul", "SLEMAN": "Sleman"}
# Kota Yogyakarta is small: its label sits outside the polygon with a leader line (offset in metres)
LABEL_OFFSET = {"YOGYAKARTA": (-17000, -11000), "BANTUL": (-3000, -4000), "GUNUNGKIDUL": (9000, -6000)}
LEADER = {"YOGYAKARTA"}  # labels drawn away from their polygon, with a leader line
CLOSE_GAPS_M = 30  # kalurahan polygons do not touch exactly; close slivers when dissolving

# --- Data ----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
gdf_kel = gpd.read_file(BOUNDARY_SHP).to_crs(CRS)


def dissolve(by, cols):
    out = gdf_kel.dissolve(by=by, as_index=False)[cols + ["geometry"]]
    out["geometry"] = out.geometry.buffer(CLOSE_GAPS_M).buffer(-CLOSE_GAPS_M)
    return out


gdf_kapn = dissolve(["KAB_KOTA", "KEC_KAPN"], ["KAB_KOTA", "KEC_KAPN"])
gdf_kapn = gdf_kapn.merge(df, on=["KAB_KOTA", "KEC_KAPN"], how="inner")
assert len(gdf_kapn) == 78, len(gdf_kapn)

# Kabupaten/kota ratio from all its kapanewon: the same values as Figure 3b
df_kab = df.groupby("KAB_KOTA", as_index=False)[["n_users", "pop_actual"]].sum()
df_kab["ratio"] = df_kab["n_users"] / df_kab["pop_actual"]
gdf_kab = dissolve("KAB_KOTA", ["KAB_KOTA"])
gdf_kab = gdf_kab.merge(df_kab, on="KAB_KOTA", how="inner")
for _, r in df_kab.iterrows():
    print(f"{KAB_NAMES[r.KAB_KOTA]}: ratio {r.ratio:.3f}")

# One colour scale for both panels: quantile bins of the 78 kapanewon
_, edges = pd.qcut(gdf_kapn["ratio"], q=N_BINS, retbins=True)
edges = list(edges)
cmap = plt.get_cmap("Blues", N_BINS)
norm = mcolors.BoundaryNorm(edges, cmap.N, clip=True)
print("bin edges:", [round(e, 3) for e in edges])


def label_kabupaten(ax, gdf, fontsize):
    for _, row in gdf.iterrows():
        p = row.geometry.representative_point()
        dx, dy = LABEL_OFFSET.get(row["KAB_KOTA"], (0, 0))
        arrow = dict(arrowstyle="-", color="black", lw=0.6) if row["KAB_KOTA"] in LEADER else None
        t = ax.annotate(KAB_NAMES[row["KAB_KOTA"]], xy=(p.x, p.y), xytext=(p.x + dx, p.y + dy),
                        ha="center", va="center", fontsize=fontsize, fontweight="bold",
                        arrowprops=arrow, zorder=6)
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])


def scale_bar_and_north(ax, length_km=20):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    bx, by = x0 + 0.04 * w, y0 + 0.06 * h
    seg = length_km * 1000 / 2
    for i, color in enumerate(["black", "white"]):  # two alternating segments
        ax.add_patch(mpatches.Rectangle((bx + i * seg, by), seg, 0.012 * h, facecolor=color,
                                        edgecolor="black", linewidth=0.6, zorder=7))
    for x, txt in [(bx, "0"), (bx + seg, f"{length_km // 2}"), (bx + 2 * seg, f"{length_km} km")]:
        ax.text(x, by + 0.025 * h, txt, ha="center", va="bottom", fontsize=ps.SMALL_SIZE, zorder=7)
    nx, ny = x0 + 0.95 * w, y0 + 0.80 * h
    ax.annotate("N", xy=(nx, ny + 0.13 * h), xytext=(nx, ny), ha="center", va="top",
                fontsize=ps.LEGEND_SIZE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10), zorder=7)


# --- Plot ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, 3.5))
fig.patch.set_facecolor(BACKGROUND_COLOR)
xmin, ymin, xmax, ymax = gdf_kab.total_bounds
pad = 2500

# (a) kapanewon, with kabupaten outlines and labels
ax = axes[0]
gdf_kapn.plot(column="ratio", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.4, ax=ax)
gdf_kab.boundary.plot(ax=ax, color="0.25", linewidth=0.8)
label_kabupaten(ax, gdf_kab, ps.SMALL_SIZE)
ax.set_title("Kapanewon", y=1.0)

# (b) kabupaten/kota, same colour scale
ax = axes[1]
gdf_kab.plot(column="ratio", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.8, ax=ax)
label_kabupaten(ax, gdf_kab, ps.SMALL_SIZE)
ax.set_title("Kabupaten/kota", y=1.0)

for ax, letter in zip(axes, ["a", "b"]):
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ps.panel_letter(ax, letter)
    scale_bar_and_north(ax)

# Shared colour bar
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.24, wspace=0.05)
cax = fig.add_axes([0.26, 0.15, 0.48, 0.035])
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=edges)
cbar.ax.set_xticklabels([f"{e:.3f}" for e in edges], fontsize=ps.FONT_SIZE)
cbar.set_label("MAIDs / population ratio (quantile classes of kapanewon)", fontsize=ps.LABEL_SIZE)

ps.save(fig, OUTPUT_STEM, facecolor=BACKGROUND_COLOR)
