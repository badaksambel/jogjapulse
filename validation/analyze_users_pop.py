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

x = df["pop_actual"]
y = df["n_users"]

r, p_value = pearsonr(x, y)
print(f"Pearson r = {r:.4f}")
print(f"p-value   = {p_value:.4e}")
print(f"n         = {len(df)}")

# Line of best fit (least squares, linear axes)
slope, intercept = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 3.2))
ax.scatter(x, y, **ps.POINT_KW)
ax.plot(x_line, y_line, **ps.FIT_KW, label="Best fit line")
ax.set_xlabel("Population")
ax.set_ylabel("Number of MAIDs (home location)")
ax.ticklabel_format(style="sci", scilimits=(3, 3), useMathText=True)  # x10^3 instead of long numbers

ps.stats_box(ax, 0.04, 0.96, f"Pearson r = {r:.3f}\np = {p_value:.2e}\nn = {len(df)}")
ax.legend(loc="lower right")

plt.tight_layout()
ps.save(fig, "scatter_users_vs_pop")
plt.show()
