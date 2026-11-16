import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# Input from od_kabupaten.py; override with OD_CSV=... (e.g. for layout tests)
OD_CSV = os.environ.get("OD_CSV", "od_kabupaten_monthly.csv")
KAP_WKT = "../release/kapanewon_wkt.csv"
OUTPUT = os.environ.get("OUTPUT", "od_flow_map_kabupaten")

# Colors follow the validation/ figures
COLOR_FLOW = "royalblue"
COLOR_INTRA_FILL = "lightblue"
COLOR_AREA = "#f0f0f0"
COLOR_BORDER = "gray"
ARC_RAD = 0.2           # arc curvature
MAX_LINEWIDTH = 10      # arc width (pt) for the largest inter-kabupaten flow
MAX_CIRCLE_KM = 7       # circle radius for the largest intra-kabupaten flow

plt.rcParams["font.size"] = 12  # default 10; larger for readability in the paper

# Kabupaten/kota polygons (dissolved from kapanewon) in UTM 49S, km
kap = pd.read_csv(KAP_WKT, dtype={"KAB_CODE": str, "KEC_CODE": str})
kab = gpd.GeoDataFrame(kap, geometry=gpd.GeoSeries.from_wkt(kap["wkt"]), crs="EPSG:4326").to_crs("EPSG:32749")
kab = kab.dissolve(by=["KAB_CODE", "KAB_KOTA"], as_index=False)[["KAB_CODE", "KAB_KOTA", "geometry"]]
kab["geometry"] = kab.buffer(5).buffer(-5)  # close sliver gaps between dissolved kalurahan (metres)
kab["geometry"] = kab.geometry.scale(1 / 1000, 1 / 1000, origin=(0, 0))
pts = kab.set_index("KAB_CODE").geometry.representative_point()
xy = pd.DataFrame({"x": pts.x, "y": pts.y})
name = kab.set_index("KAB_CODE")["KAB_KOTA"].str.title().replace({"Yogyakarta": "Kota Yogyakarta"})

# Whole-period flows = sum of monthly distinct devices (device-months)
od = pd.read_csv(OD_CSV, dtype={"origin_kab_code": str, "dest_kab_code": str})
flows = od.groupby(["origin_kab_code", "dest_kab_code"])["n_maids"].sum().reset_index()
intra = flows[flows["origin_kab_code"] == flows["dest_kab_code"]].set_index("origin_kab_code")["n_maids"]
# One line per kabupaten pair: total flow in both directions (i->j + j->i)
inter = flows[flows["origin_kab_code"] != flows["dest_kab_code"]].copy()
inter["a"] = inter[["origin_kab_code", "dest_kab_code"]].min(axis=1)
inter["b"] = inter[["origin_kab_code", "dest_kab_code"]].max(axis=1)
inter = inter.groupby(["a", "b"])["n_maids"].sum().reset_index().sort_values("n_maids")
max_inter = inter["n_maids"].max()

fig, ax = plt.subplots(figsize=(9, 8.5))
kab.plot(ax=ax, color=COLOR_AREA, edgecolor=COLOR_BORDER, linewidth=1, zorder=0)

# Intra-kabupaten flows: circles (area proportional to flow)
for code, n in intra.items():
    r = MAX_CIRCLE_KM * np.sqrt(n / intra.max())
    ax.add_patch(plt.Circle((xy.loc[code, "x"], xy.loc[code, "y"]), r, facecolor=COLOR_INTRA_FILL,
                            edgecolor=COLOR_FLOW, linewidth=1, alpha=0.6, zorder=1))

# Inter-kabupaten flows: one arc per pair, width proportional to total flow (smallest drawn first)
for _, f in inter.iterrows():
    o, d = xy.loc[f["a"]], xy.loc[f["b"]]
    lw = 0.8 + MAX_LINEWIDTH * f["n_maids"] / max_inter
    ax.add_patch(FancyArrowPatch((o["x"], o["y"]), (d["x"], d["y"]), connectionstyle=f"arc3,rad={ARC_RAD}",
                                 arrowstyle="-", linewidth=lw, capstyle="round",
                                 color=COLOR_FLOW, alpha=0.7, shrinkA=0, shrinkB=0, zorder=2))

# Labels with white halo (as in the choropleth figures)
for code, row in xy.iterrows():
    ax.text(row["x"], row["y"], f"{name[code]}\n{intra.get(code, 0) / 1000:,.0f}k within",
            ha="center", va="center", fontsize=10, fontweight="bold", color="black", zorder=3,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")])

# Legend: arc widths for round flow values
def nice_values(vmax):
    exp = 10 ** np.floor(np.log10(vmax))
    steps = [v * exp for v in (0.1, 0.2, 0.5, 1, 2, 5) if v * exp <= vmax]
    return steps[-3:]

handles = [Line2D([], [], color=COLOR_FLOW, alpha=0.7, linewidth=0.8 + MAX_LINEWIDTH * v / max_inter,
                  label=f"{v / 1000:,.0f}k" if v >= 1000 else f"{v:,.0f}") for v in nice_values(max_inter)]
handles.append(Line2D([], [], marker="o", linestyle="", markersize=14, markerfacecolor=COLOR_INTRA_FILL,
                      markeredgecolor=COLOR_FLOW, alpha=0.6, label="within kabupaten"))
ax.legend(handles=handles, title="Flow, both directions\n(device-months)", loc="lower left", fontsize=10, title_fontsize=10,
          facecolor="white", edgecolor="gray", framealpha=0.9, borderpad=0.8, labelspacing=0.9)

# Scale bar (10 km) and north arrow
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
sx, sy = x0 + (x1 - x0) * 0.04, y1 - (y1 - y0) * 0.06  # top-left corner, clear of the map
ax.plot([sx, sx + 10], [sy, sy], color="black", linewidth=3, solid_capstyle="butt")
ax.text(sx + 5, sy - (y1 - y0) * 0.012, "10 km", ha="center", va="top", fontsize=10)
nx, ny = x1 - (x1 - x0) * 0.06, y1 - (y1 - y0) * 0.12
ax.annotate("N", xy=(nx, ny + (y1 - y0) * 0.07), xytext=(nx, ny), ha="center", va="top", fontsize=12,
            arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.5))

ax.set_aspect("equal")
ax.set_axis_off()

months = sorted(od["month"].unique())
print(f"Flows summed over {months[0]} .. {months[-1]} ({len(months)} months)")
print(flows.assign(origin=lambda t: name.loc[t["origin_kab_code"]].values,
                   dest=lambda t: name.loc[t["dest_kab_code"]].values)
      .pivot(index="origin", columns="dest", values="n_maids").to_string())

plt.tight_layout()
plt.savefig(f"{OUTPUT}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT}.pdf", bbox_inches="tight")
print(f"Tersimpan: {OUTPUT}.png, {OUTPUT}.pdf")
