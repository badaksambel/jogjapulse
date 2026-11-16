import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "cohort_daily_mobility.csv"
OUTPUT_PNG = "cohort_daily_mobility_events.png"
OUTPUT_PDF = "cohort_daily_mobility_events.pdf"

# Colors follow the validation/ figures: black for the primary series, royalblue as accent
COLOR_AWAY = "black"
COLOR_RG = "royalblue"
COLOR_PPKM = "indianred"
COLOR_RAMADAN = "seagreen"
COLOR_NATARU = "gray"

plt.rcParams["font.size"] = 12  # default 10; larger for readability in the paper

df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date").set_index("date")
data_end = df.index.max() + pd.Timedelta(days=1)

# Same events as plot_daily.py (sources listed there)
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

PANELS = [
    ("share_away_2km", 100, "MAIDs > 2 km from home (%)", COLOR_AWAY),
    ("mean_rg_km", 1, "Mean radius of gyration (km)", COLOR_RG),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

for i, (ax, (col, scale, ylabel, color)) in enumerate(zip(axes, PANELS)):
    y = df[col] * scale
    y_roll = y.rolling(7, center=True, min_periods=4).mean()
    lo, hi = y.min(), y.max()
    ymin, ymax = lo - (hi - lo) * 0.1, hi + (hi - lo) * (0.55 if i == 0 else 0.1)

    for start, end, span_color, alpha, label in SPANS:
        x0 = pd.Timestamp(start)
        x1 = pd.Timestamp(end) + pd.Timedelta(days=1) if end else data_end
        ax.axvspan(x0, x1, color=span_color, alpha=alpha, linewidth=0, zorder=0)
        if i == 0:  # event labels only on the top panel
            ax.text(x0 + (x1 - x0) / 2, ymax - (ymax - ymin) * 0.03, label, ha="center", va="top",
                    fontsize=10, color=span_color)

    for date, line_color, label in LINES:
        x = pd.Timestamp(date)
        ax.axvline(x, color=line_color, linestyle="--", linewidth=1.2, zorder=1)
        if i == 0:
            ax.text(x + pd.Timedelta(days=1.5), ymax - (ymax - ymin) * 0.28, label, ha="left",
                    va="top", fontsize=10, color=line_color)

    daily, = ax.plot(y.index, y, color=color, linewidth=0.8, alpha=0.35, label="Daily", zorder=2)
    roll, = ax.plot(y_roll.index, y_roll, color=color, linewidth=2, label="7-day mean", zorder=3)

    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", color="lightgray", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[daily, roll], loc="lower left",
              facecolor="white", edgecolor="gray", framealpha=0.9)

axes[1].set_xlim(df.index.min() - pd.Timedelta(days=1), data_end)
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.savefig(OUTPUT_PDF)
print(f"Tersimpan: {OUTPUT_PNG}, {OUTPUT_PDF}")
plt.show()
