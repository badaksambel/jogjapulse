"""Figure 3: home-located devices vs BPS population, (a) 78 kapanewon, (b) 5 kabupaten/kota.

Input: full-period homes (22:00-05:59, >= 2 nights) from source pings without default coordinates
(release/home_location_full_period.py). With the default points removed there is no outlier, so all
units are used. Statistics: Pearson r on linear and log10 scales with Fisher-z 95% CI, Spearman rho,
and the log-log OLS slope (the fit drawn).
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, norm, pearsonr, spearmanr

import plot_style as ps

ps.apply()  # Helvetica/Arial, 10/9/8 pt as the other paper figures, final printed size

df = pd.read_csv("../release/output/home_maids_vs_population_kapanewon.csv")
df_kab = df.groupby("KAB_KOTA", as_index=False)[["n_users", "pop_actual"]].sum()


def r_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh(z - norm.ppf(0.975) * se), np.tanh(z + norm.ppf(0.975) * se)


def describe(x, y, name):
    lx, ly = np.log10(x), np.log10(y)
    r_lin, p_lin = pearsonr(x, y)
    r_log, p_log = pearsonr(lx, ly)
    rho = spearmanr(x, y)[0]
    fit = linregress(lx, ly)
    n = len(x)
    lo, hi = r_ci(r_lin, n) if n > 3 else (np.nan, np.nan)
    lo2, hi2 = r_ci(r_log, n) if n > 3 else (np.nan, np.nan)
    print(f"FIG3 {name}: n = {n}; linear r = {r_lin:.3f} [{lo:.2f}-{hi:.2f}] p = {p_lin:.1e}; "
          f"log r = {r_log:.3f} [{lo2:.2f}-{hi2:.2f}] p = {p_log:.1e}; rho = {rho:.3f}; "
          f"log-log slope = {fit.slope:.2f} (SE {fit.stderr:.2f})")
    return r_log, rho, fit


def panel(ax, x, y, title, labels=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r_log, rho, fit = describe(x, y, title)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.scatter(x, y, **ps.POINT_KW)
    ax.plot(xl, 10 ** (fit.intercept + fit.slope * np.log10(xl)), **ps.FIT_KW, label="Best fit line (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Population")
    ax.set_ylabel("Number of MAIDs (home location)")
    ax.set_title(title)
    ps.stats_box(ax, 0.03, 0.97, f"n = {len(x)}\nlog-log r = {r_log:.3f}\nSpearman $\\rho$ = {rho:.3f}")
    if labels is not None:
        for xi, yi, lab in zip(x, y, labels):
            right = xi == x.max()  # rightmost label on the left of its point, inside the axes
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(-5, 5) if right else (5, 5),
                        ha="right" if right else "left", fontsize=ps.SMALL_SIZE)
        ax.margins(0.12)
    ax.legend(loc="lower right")


fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, 3.6))
panel(axes[0], df["pop_actual"], df["n_users"], "Correlation per kapanewon")
panel(axes[1], df_kab["pop_actual"], df_kab["n_users"], "Correlation per kabupaten/kota",
      labels=df_kab["KAB_KOTA"].str.title())
for ax, letter in zip(axes, ["a", "b"]):
    ps.panel_letter(ax, letter)

plt.tight_layout()
ps.save(fig, "scatter_users_vs_pop_kapanewon_dual_correlation")
