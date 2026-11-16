"""Figure 5: mean pings and mean distinct MAIDs per day, (a) by hour of day, (b) by day of week.

Means of daily values, so weekdays that occur 21 or 22 times in the period are comparable.
Inputs: temporal_date_hour.csv (temporal_date_hour.py) and temporal_daily.csv (temporal_validasi.py).
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder
sys.path.insert(0, "../validation")

import matplotlib.pyplot as plt
import pandas as pd

import plot_style as ps

ps.apply()

OUTPUT_STEM = "hourly_dow_pings_users"
COLOR_PINGS = "black"
COLOR_USERS = "royalblue"
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# (a) per hour: mean over the 151 days of pings and of distinct MAIDs in that date-hour
hourly = pd.read_csv("temporal_date_hour.csv").groupby("hour")[["n_pings", "n_unique_users"]].mean()

# (b) per weekday: mean over the days with that weekday of daily pings and daily distinct MAIDs
daily = pd.read_csv("temporal_daily.csv", parse_dates=["date"])
daily["dow"] = daily["date"].dt.dayofweek  # 0 = Monday
dow = daily.groupby("dow").agg(n_pings=("n_pings", "mean"), n_unique_users=("n_unique_users", "mean"),
                               n_days=("date", "size"))

for name, t in [("hour", hourly), ("weekday", dow)]:
    for c in ["n_pings", "n_unique_users"]:
        print(f"{name} {c}: min {t[c].min():,.0f}, max {t[c].max():,.0f}, max/min - 1 = {t[c].max() / t[c].min() - 1:.1%}")


def dual_axis_panel(ax, x, pings, users, pings_label, users_label):
    """Pings on the left y-axis, distinct MAIDs on the right; both axes start at zero."""
    ax2 = ax.twinx()
    line_p, = ax.plot(x, pings / 1e3, color=COLOR_PINGS, linewidth=1.5, marker="o", markersize=4,
                      label="Pings", zorder=3)
    line_u, = ax2.plot(x, users / 1e3, color=COLOR_USERS, linestyle="--", linewidth=1.2, marker="s",
                       markersize=4, markerfacecolor="lightblue", markeredgecolor=COLOR_USERS,
                       label="Distinct MAIDs", zorder=3)
    ax.set_ylabel(pings_label)
    ax2.set_ylabel(users_label, color=COLOR_USERS)
    ax2.tick_params(axis="y", colors=COLOR_USERS)
    ax2.spines["right"].set_color(COLOR_USERS)
    ax.set_ylim(0, pings.max() / 1e3 * 1.12)
    ax2.set_ylim(0, users.max() / 1e3 * 1.12)
    ax.grid(axis="y", color="lightgray", linewidth=0.6, alpha=0.7)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    return line_p, line_u


fig, (ax_h, ax_d) = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, 3.2))

lp, lu = dual_axis_panel(ax_h, hourly.index, hourly["n_pings"], hourly["n_unique_users"],
                         r"Pings per hour ($\times10^3$)", r"Distinct MAIDs per hour ($\times10^3$)")
ax_h.set_xlabel("Hour of day (WIB)")
ax_h.set_xticks(range(0, 24, 3))
ax_h.set_xlim(-0.7, 23.7)

dual_axis_panel(ax_d, dow.index, dow["n_pings"], dow["n_unique_users"],
                r"Pings per day ($\times10^3$)", r"Distinct MAIDs per day ($\times10^3$)")
ax_d.set_xlabel("Day of week (number of days)")
ax_d.set_xticks(list(dow.index))
ax_d.set_xticklabels([f"{DOW_LABELS[i]}\n({n})" for i, n in dow["n_days"].items()])
ax_d.set_xlim(-0.4, 6.4)

for ax, letter in zip((ax_h, ax_d), ("a", "b")):
    ax.legend(handles=[lp, lu], loc="lower center", facecolor="white", edgecolor="gray", framealpha=0.9)
    ps.panel_letter(ax, letter)

plt.tight_layout()
ps.save(fig, OUTPUT_STEM)
