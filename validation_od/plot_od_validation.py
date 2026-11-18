import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
from paths import fig_path, out_path  # noqa: E402  inputs under JOGJAPULSE_OUT, figures to JOGJAPULSE_FIG

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr

# Inputs: released (suppressed, k = 5) OD files, so the figures are reproducible from the open data
OD_CSV = out_path("output", "od_home_visits_kapanewon_monthly.csv")
ORIGIN_CSV = out_path("output", "od_origin_homes_monthly.csv")
KAP_WKT = "../release/data/kapanewon_wkt.csv"
POP_CSV = out_path("output", "home_maids_vs_population_kapanewon.csv")  # pop_actual (BPS)

OUTPUT_A = fig_path("od_validation_gravity_reciprocity")   # panels (a) gravity fit, (b) reciprocity
OUTPUT_B = fig_path("od_validation_distance_quality")      # panels (a) distance decay, (b) quality by month
STATS_CSV = fig_path("od_validation_stats.csv")

# Colors follow the validation/ figures
COLOR_GOOD = "royalblue"     # Dec 2021 - Feb 2022
COLOR_REDUCED = "indianred"  # Mar - Apr 2022 (reduced_quality)
GROUPS = [("Dec 2021 – Feb 2022", False, COLOR_GOOD), ("Mar – Apr 2022", True, COLOR_REDUCED)]
K = 5  # suppression threshold of the released OD files: only flows >= K devices are published
MONTH_LABELS = {"2021-12": "Dec", "2022-01": "Jan", "2022-02": "Feb", "2022-03": "Mar", "2022-04": "Apr"}

# Figures are drawn at their final printed size (7.2 in wide) so the font sizes below
# are the real sizes in the paper (same template as the validation/ figures).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

# ============================================================
# Data: published inter-zonal flows with origin homes, destination population, distance
# ============================================================
kap = pd.read_csv(KAP_WKT)
geo = gpd.GeoSeries.from_wkt(kap["wkt"], crs="EPSG:4326").to_crs("EPSG:32749").centroid
xy = pd.DataFrame({"cx": geo.x.values, "cy": geo.y.values}, index=kap["KEC_CODE"])
pop_tab = pd.read_csv(POP_CSV).merge(kap[["KAB_KOTA", "KEC_KAPN", "KEC_CODE"]], on=["KAB_KOTA", "KEC_KAPN"])
pop = pop_tab.set_index("KEC_CODE")["pop_actual"]

od = pd.read_csv(OD_CSV)
origin = pd.read_csv(ORIGIN_CSV)
homes = origin.set_index(["month", "origin_code"])["n_homed_maids"]

od = od[od["origin_code"] != od["dest_code"]].copy()
od["dist_km"] = np.hypot(xy.loc[od["origin_code"], "cx"].values - xy.loc[od["dest_code"], "cx"].values,
                         xy.loc[od["origin_code"], "cy"].values - xy.loc[od["dest_code"], "cy"].values) / 1000
od["H_i"] = homes.loc[list(zip(od["month"], od["origin_code"]))].values
od["P_j"] = pop.loc[od["dest_code"]].values

# ============================================================
# Per-month fits: gravity log T = a + b log H_i + c log P_j - gamma log d; reciprocity; homes vs BPS
# ============================================================
stats, recip = [], []
od["predicted"] = np.nan
od["adjusted_flow"] = np.nan  # log T - b log H_i - c log P_j (component-plus-residual for distance)
for month, g in od.groupby("month"):
    X = sm.add_constant(np.log(g[["H_i", "P_j", "dist_km"]]))
    fit = sm.OLS(np.log(g["n_maids"]), X).fit()
    od.loc[g.index, "predicted"] = np.exp(fit.fittedvalues)
    # Component-plus-residual for distance, centred per month (geometric mean = 1) so months are comparable
    adj = np.log(g["n_maids"]) - fit.params["H_i"] * np.log(g["H_i"]) - fit.params["P_j"] * np.log(g["P_j"])
    od.loc[g.index, "adjusted_flow"] = np.exp(adj - adj.mean())

    t = g.set_index(["origin_code", "dest_code"])["n_maids"]
    pairs = [(i, j) for i, j in t.index if i < j and (j, i) in t.index]
    ij, ji = np.array([t[p] for p in pairs]), np.array([t[(p[1], p[0])] for p in pairs])
    recip.append(pd.DataFrame({"month": month, "flow_ij": ij, "flow_ji": ji}))

    o = origin[origin["month"] == month].merge(pop_tab[["KEC_CODE", "KEC_KAPN"]],
                                               left_on="origin_code", right_on="KEC_CODE")
    stats.append({"month": month, "pairs": len(g), "gravity_R2": fit.rsquared,
                  "gamma_distance": -fit.params["dist_km"],
                  "reciprocity_r": pearsonr(np.log(ij), np.log(ji))[0], "reciprocity_pairs": len(pairs),
                  "homes_vs_bps_r": pearsonr(pop.loc[o["origin_code"]].values, o["n_homed_maids"].values)[0]})
stats = pd.DataFrame(stats)
recip = pd.concat(recip)
stats.round(3).to_csv(STATS_CSV, index=False)
print(stats.round(3).to_string(index=False))

reduced_months = set(od.loc[od["reduced_quality"], "month"])
recip["reduced_quality"] = recip["month"].isin(reduced_months)

def stat_range(col, reduced):
    s = stats.loc[stats["month"].isin(reduced_months) == reduced, col]
    return f"{s.min():.2f}–{s.max():.2f}"

def log_axes(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, zorder=1, label="1:1 line")
    fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_aspect("equal")

def caption(ax, text, letter):
    """Panel title plus bold lower-case panel letter at the top left."""
    ax.set_title(text)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")

def finish(ax):
    ax.grid(color="lightgray", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ============================================================
# Figure A: (a) gravity observed vs predicted, (b) reciprocity
# ============================================================
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(7.2, 3.6))

lo, hi = 3, max(od["n_maids"].max(), od["predicted"].max()) * 1.5
for label, reduced, color in GROUPS:
    g = od[od["reduced_quality"] == reduced]
    ax1.scatter(g["predicted"], g["n_maids"], s=4, alpha=0.35, color=color, linewidths=0,
                label=f"{label} (R² = {stat_range('gravity_R2', reduced)})", zorder=2)
log_axes(ax1, lo, hi)
ax1.axhline(K, color="0.35", linestyle=":", linewidth=1, zorder=1, label=f"k = {K} (flows < {K} not published)")
ax1.set_xlabel("Predicted flow (gravity model)")
ax1.set_ylabel("Observed flow (devices)")
caption(ax1, "Gravity model fit", "a")

lo3, hi3 = 3, max(recip["flow_ij"].max(), recip["flow_ji"].max()) * 1.5
for label, reduced, color in GROUPS:
    g = recip[recip["reduced_quality"] == reduced]
    ax3.scatter(g["flow_ij"], g["flow_ji"], s=4, alpha=0.35, color=color, linewidths=0,
                label=f"{label} (r = {stat_range('reciprocity_r', reduced)})", zorder=2)
log_axes(ax3, lo3, hi3)
ax3.axhline(K, color="0.35", linestyle=":", linewidth=1, zorder=1, label=f"k = {K} (flows < {K} not published)")
ax3.axvline(K, color="0.35", linestyle=":", linewidth=1, zorder=1)
ax3.set_xlabel(r"Flow $i \rightarrow j$ (devices)")
ax3.set_ylabel(r"Flow $j \rightarrow i$ (devices)")
caption(ax3, "Reciprocity", "b")

for ax in (ax1, ax3):
    finish(ax)
    handles, labels = ax.get_legend_handles_labels()
    pts = [(h, l) for h, l in zip(handles, labels) if l.startswith(("Dec", "Mar"))]  # period colours, per panel
    if ax is ax1:  # narrow two-line labels keep the legend clear of the largest flows
        pts = [(h, l.replace(" (", "\n(")) for h, l in pts]
    leg = ax.legend(*zip(*pts), loc="upper left", fontsize=8, facecolor="white", edgecolor="gray",
                    framealpha=0.9, markerscale=2.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
# Reference lines, shared by both panels, in one legend below the figure
handles, labels = ax3.get_legend_handles_labels()
lines = [(h, l) for h, l in zip(handles, labels) if not l.startswith(("Dec", "Mar"))]
fig.legend(*zip(*lines), loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0))

plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(f"{OUTPUT_A}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_A}.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure B: (a) distance decay, (b) quality by month
# ============================================================
fig, (ax2, ax4) = plt.subplots(1, 2, figsize=(7.2, 3.4))

d_line = np.logspace(np.log10(od["dist_km"].min()), np.log10(od["dist_km"].max()), 50)
for label, reduced, color in GROUPS:
    g = od[od["reduced_quality"] == reduced]
    ax2.scatter(g["dist_km"], g["adjusted_flow"], s=4, alpha=0.3, color=color, linewidths=0, zorder=2)
    # Line: mean monthly gamma of the group, intercept from the group's mean adjusted flow
    gamma = stats.loc[stats["month"].isin(reduced_months) == reduced, "gamma_distance"].mean()
    a = np.mean(np.log(g["adjusted_flow"]) + gamma * np.log(g["dist_km"]))
    ax2.plot(d_line, np.exp(a - gamma * np.log(d_line)), color=color, linewidth=1.5, zorder=3,
             label=f"{label} (γ = {stat_range('gamma_distance', reduced)})")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
ax2.set_xlabel("Distance between kapanewon centroids (km)")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
ax2.set_ylabel("Relative flow, adjusted for\norigin and destination size")
caption(ax2, "Distance decay", "a")
ax2.legend(loc="lower left", fontsize=8, facecolor="white", edgecolor="gray", framealpha=0.9)

x = np.arange(len(stats))
metrics = [("homes_vs_bps_r", "Homes vs BPS population (r)", "black", "o", "-"),
           ("reciprocity_r", "Reciprocity (r)", COLOR_GOOD, "s", "--"),
           ("gravity_R2", "Gravity model (R²)", "seagreen", "^", "-.")]
for col, label, color, marker, ls in metrics:
    ax4.plot(x, stats[col], color=color, marker=marker, linestyle=ls, linewidth=1.5, markersize=5, label=label)
red_idx = [i for i, m in enumerate(stats["month"]) if m in reduced_months]
ax4.axvspan(min(red_idx) - 0.5, max(red_idx) + 0.5, color=COLOR_REDUCED, alpha=0.12, linewidth=0, zorder=0)
ax4.text(np.mean(red_idx), 0.04, "reduced\nquality", ha="center", va="bottom", fontsize=8, color=COLOR_REDUCED)
ax4.set_xticks(x)
ax4.set_xticklabels([MONTH_LABELS[m] for m in stats["month"]])
ax4.set_xlim(-0.5, len(stats) - 0.5)
ax4.set_ylim(0, 1)
ax4.set_xlabel("Month (Dec 2021 – Apr 2022)")
ax4.set_ylabel("Validation statistic")
caption(ax4, "Quality by month", "b")
ax4.legend(loc="lower left", fontsize=8, facecolor="white", edgecolor="gray", framealpha=0.9)

for ax in (ax2, ax4):
    finish(ax)

plt.tight_layout()
plt.savefig(f"{OUTPUT_B}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_B}.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Tersimpan: {OUTPUT_A}.png/.pdf, {OUTPUT_B}.png/.pdf, {STATS_CSV}")
plt.show()