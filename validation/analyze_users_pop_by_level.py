import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import plot_style as ps

ps.apply()  # Helvetica/Arial, 10/9/8 pt as the other paper figures, final printed size

df = pd.read_csv("../release/output/home_maids_vs_population_kapanewon.csv")  # full-period home (22:00-05:59, >= 2 nights); legacy: ../release/data/validasi_kapanewon_des2021_apr2022_legacy_home.csv
# Exclude Gondomanan (outlier)
df = df[df["KEC_KAPN"] != "GONDOMANAN"]

# Aggregate to KAB_KOTA level
df_kab = df.groupby("KAB_KOTA", as_index=False)[["n_users", "pop_actual"]].sum()


def scatter_with_fit(ax, x, y, xlabel, ylabel, title, labels=None):
    r, p_value = pearsonr(x, y)

    # Fit in log-log space so the line is straight on log axes
    log_x, log_y = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = 10 ** (slope * np.log10(x_line) + intercept)

    ax.scatter(x, y, **ps.POINT_KW)
    ax.plot(x_line, y_line, **ps.FIT_KW, label="Best fit line")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if labels is not None:
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=ps.SMALL_SIZE)
        ax.margins(0.12)

    ps.stats_box(ax, 0.03, 0.97, f"Pearson r = {r:.3f}\np = {p_value:.2e}\nn = {len(x)}")
    ax.legend(loc="lower right")

    print(f"{title}: Pearson r = {r:.4f}, p = {p_value:.4e}, n = {len(x)}")


fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, 3.6))

scatter_with_fit(
    axes[0],
    df["pop_actual"], df["n_users"],
    "Population", "Number of MAIDs (home location)",
    "Correlation per kapanewon",
)

scatter_with_fit(
    axes[1],
    df_kab["pop_actual"], df_kab["n_users"],
    "Population", "Number of MAIDs (home location)",
    "Correlation per kabupaten/kota",
    labels=df_kab["KAB_KOTA"].str.title(),
)

for ax, letter in zip(axes, ["a", "b"]):
    ps.panel_letter(ax, letter)

plt.tight_layout()
ps.save(fig, "scatter_users_vs_pop_by_level")
plt.show()
