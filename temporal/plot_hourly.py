import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "temporal_hourly.csv"
OUTPUT_PNG = "hourly_pings_users.png"
OUTPUT_PDF = "hourly_pings_users.pdf"

# Colors follow the validation/ figures: black for the primary series, royalblue as accent
COLOR_PINGS = "black"
COLOR_USERS = "royalblue"

plt.rcParams["font.size"] = 12  # default 10; larger for readability in the paper

df = pd.read_csv(CSV_PATH).sort_values("hour")

fig, ax = plt.subplots(figsize=(6, 5))
ax2 = ax.twinx()

# Left axis: pings (x10^6)
line_pings, = ax.plot(
    df["hour"], df["n_pings"] / 1e6,
    color=COLOR_PINGS, linestyle="-", linewidth=2, marker="o", markersize=5,
    label="Pings", zorder=3,
)
ax.set_xlabel("Hour of day")
ax.set_ylabel(r"Pings ($\times10^6$)")

# Right axis: unique users (x10^3)
line_users, = ax2.plot(
    df["hour"], df["n_unique_users"] / 1e3,
    color=COLOR_USERS, linestyle="--", linewidth=1.5, marker="s", markersize=5,
    markerfacecolor="lightblue", markeredgecolor=COLOR_USERS,
    label="Unique MAIDs", zorder=3,
)
ax2.set_ylabel(r"Unique MAIDs ($\times10^3$)", color=COLOR_USERS)
ax2.tick_params(axis="y", colors=COLOR_USERS)
ax2.spines["right"].set_color(COLOR_USERS)

ax.set_xticks(range(0, 24, 3))
ax.set_xlim(-0.7, 23.7)
ax.grid(axis="y", color="lightgray", linewidth=0.6, alpha=0.7)
ax.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax.legend(handles=[line_pings, line_users], loc="lower center",
          facecolor="white", edgecolor="gray", framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.savefig(OUTPUT_PDF)
print(f"Tersimpan: {OUTPUT_PNG}, {OUTPUT_PDF}")
plt.show()
