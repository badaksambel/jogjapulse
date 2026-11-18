import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import sys

sys.path.insert(0, "../release")
from paths import fig_path  # noqa: E402  inputs/outputs in JOGJAPULSE_FIG if set

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Log y-axes keep the post-February months readable; set False for the linear version
LOG_SCALE = True

CSV_PATH = fig_path("temporal_daily.csv")
SUFFIX = "_log" if LOG_SCALE else ""
OUTPUT_PNG = fig_path(f"daily_pings_users_coverage{SUFFIX}.png")
OUTPUT_PDF = fig_path(f"daily_pings_users_coverage{SUFFIX}.pdf")

# Colors follow the validation/ figures: black for the primary series, royalblue as accent
COLOR_PINGS = "black"
COLOR_USERS = "royalblue"
COLOR_BREAK = "gray"

# Figure is drawn at its final printed size (7.2 in wide) so the font sizes below
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

df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date")
data_end = df["date"].max() + pd.Timedelta(days=1)

# Abrupt drops in the number of active MAIDs (panel size), not behavioural changes:
#   1 Jan 2022: ~98k -> ~60k MAIDs/day, stays at the lower level through January
#   16 Feb 2022: ~71k -> ~29k MAIDs/day in one day while pings per MAID doubled
PANEL_BREAKS = ["2022-01-01", "2022-02-16"]

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax2 = ax.twinx()

if LOG_SCALE:
    # Same ratio (x50) on both axes so the two lines stay comparable
    ymin_pings, ymax_pings = 0.03, 1.5
    ymin_users, ymax_users = 5, 250
else:
    ymin_pings, ymax_pings = 0, df["n_pings"].max() / 1e6 * 1.15
    ymin_users, ymax_users = 0, df["n_unique_users"].max() / 1e3 * 1.15

for date in PANEL_BREAKS:
    x = pd.Timestamp(date)
    ax.axvline(x, color=COLOR_BREAK, linestyle="--", linewidth=1, zorder=1)
    ax.text(x + pd.Timedelta(days=1.5), ymax_pings * 0.97,
            f"Panel size change\n({x.strftime('%-d %b')})",
            ha="left", va="top", fontsize=8, color=COLOR_BREAK)

line_pings, = ax.plot(df["date"], df["n_pings"] / 1e6, color=COLOR_PINGS,
                      linestyle="-", linewidth=1.5, label="Pings", zorder=3)
line_users, = ax2.plot(df["date"], df["n_unique_users"] / 1e3, color=COLOR_USERS,
                       linestyle="--", linewidth=1.2, label="Unique MAIDs", zorder=3)

ax.set_ylabel(r"Pings per day ($\times10^6$)")
ax2.set_ylabel(r"Unique MAIDs per day ($\times10^3$)", color=COLOR_USERS)
if LOG_SCALE:
    for axis, ticks in ((ax, [0.05, 0.1, 0.2, 0.5, 1]), (ax2, [5, 10, 20, 50, 100, 200])):
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        axis.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(ymin_pings, ymax_pings)
ax2.set_ylim(ymin_users, ymax_users)
ax2.tick_params(axis="y", colors=COLOR_USERS)
ax2.spines["right"].set_color(COLOR_USERS)

ax.set_xlim(df["date"].min() - pd.Timedelta(days=1), data_end)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax.set_xlabel("Date")

ax.grid(axis="y", color="lightgray", linewidth=0.6, alpha=0.7)
ax.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax.legend(handles=[line_pings, line_users], loc="center right",
          facecolor="white", edgecolor="gray", framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.savefig(OUTPUT_PDF)
print(f"Tersimpan: {OUTPUT_PNG}, {OUTPUT_PDF}")
plt.show()
