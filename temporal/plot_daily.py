import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "temporal_daily.csv"
OUTPUT_PNG = "daily_pings_users_events.png"
OUTPUT_PDF = "daily_pings_users_events.pdf"

# Colors follow the validation/ figures: black for the primary series, royalblue as accent
COLOR_PINGS = "black"
COLOR_USERS = "royalblue"
COLOR_PPKM = "indianred"
COLOR_RAMADAN = "seagreen"
COLOR_NATARU = "gray"

plt.rcParams["font.size"] = 12  # default 10; larger for readability in the paper

df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date")
data_end = df["date"].max() + pd.Timedelta(days=1)

# Event periods (end dates are inclusive, shading runs to the end of that day)
#   Nataru: Inmendagri 66/2021, 24 Dec 2021 - 2 Jan 2022
#   PPKM L3 DIY: from 8 Feb 2022 (Inmendagri 9/2022), extended to 7 Mar
#   PPKM L4 DIY: 8-21 Mar 2022 (Inmendagri 15/2022, extended by 16/2022)
#   PPKM L3 DIY again: from 22 Mar 2022 (level 4 removed Java-Bali-wide)
#   Omicron peak: national daily record 64,718 cases on 16 Feb 2022 (Satgas COVID-19)
#   Ramadan 1443 H: 3 Apr 2022 (Kemenag isbat), until after the end of the data
SPANS = [
    ("2021-12-24", "2022-01-02", COLOR_NATARU, 0.12, "Nataru\nrestrictions"),
    ("2022-02-08", "2022-03-07", COLOR_PPKM, 0.12, "PPKM L3"),
    ("2022-03-08", "2022-03-21", COLOR_PPKM, 0.28, "PPKM\nL4"),
    ("2022-03-22", "2022-04-02", COLOR_PPKM, 0.12, "PPKM\nL3"),
    ("2022-04-03", None, COLOR_RAMADAN, 0.12, "Ramadan"),
]
LINES = [
    ("2022-02-16", COLOR_PPKM, "16 Feb\nOmicron peak\n(national)"),
]

fig, ax = plt.subplots(figsize=(12, 5))
ax2 = ax.twinx()

ymax_pings = df["n_pings"].max() / 1e6 * 1.3  # headroom for the event labels
ymax_users = df["n_unique_users"].max() / 1e3 * 1.3

for start, end, color, alpha, label in SPANS:
    x0 = pd.Timestamp(start)
    x1 = pd.Timestamp(end) + pd.Timedelta(days=1) if end else data_end
    ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=0, zorder=0)
    ax.text(x0 + (x1 - x0) / 2, ymax_pings * 0.97, label, ha="center", va="top",
            fontsize=10, color=color)

for date, color, label in LINES:
    x = pd.Timestamp(date)
    ax.axvline(x, color=color, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(x + pd.Timedelta(days=1.5), ymax_pings * 0.72, label, ha="left", va="top",
            fontsize=10, color=color)

line_pings, = ax.plot(df["date"], df["n_pings"] / 1e6, color=COLOR_PINGS,
                      linestyle="-", linewidth=1.8, label="Pings", zorder=3)
line_users, = ax2.plot(df["date"], df["n_unique_users"] / 1e3, color=COLOR_USERS,
                       linestyle="--", linewidth=1.5, label="Unique MAIDs", zorder=3)

ax.set_ylabel(r"Pings per day ($\times10^6$)")
ax.set_ylim(0, ymax_pings)
ax2.set_ylabel(r"Unique MAIDs per day ($\times10^3$)", color=COLOR_USERS)
ax2.set_ylim(0, ymax_users)
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
