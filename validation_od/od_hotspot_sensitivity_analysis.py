"""OD validation statistics per hotspot variant (inputs from od_hotspot_sensitivity.py).

Each variant's unsuppressed monthly flows are suppressed at k = 5 (primary suppression only), then:
gravity OLS on published inter-zonal pairs, Tobit with unpublished pairs censored at 5,
reciprocity r on pairs published in both directions, and monthly homes vs BPS population (r, excl.
nothing, all 78 kapanewon). Writes od_hotspot_sensitivity_summary.csv (aggregates only).
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import itertools

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

K = 5
WORK = "../release/od_work"
VARIANTS = ["all", "nohot10000", "nohot1000", "nohot100"]
MONTHS = ["2021-12", "2022-01", "2022-02", "2022-03", "2022-04"]

kap = pd.read_csv("../release/kapanewon_wkt.csv")
geo = gpd.GeoSeries.from_wkt(kap["wkt"], crs="EPSG:4326").to_crs("EPSG:32749").centroid
xy = pd.DataFrame({"cx": geo.x.values, "cy": geo.y.values}, index=kap["KEC_CODE"])
pop = (pd.read_csv("../release/output/home_maids_vs_population_kapanewon.csv")
       .merge(kap[["KAB_KOTA", "KEC_KAPN", "KEC_CODE"]], on=["KAB_KOTA", "KEC_KAPN"]).set_index("KEC_CODE")["pop_actual"])
codes = list(pop.index)
pairs = pd.DataFrame(list(itertools.permutations(codes, 2)), columns=["home_kap", "dest_kap"])
pairs["dist_km"] = np.hypot(xy.loc[pairs.home_kap, "cx"].values - xy.loc[pairs.dest_kap, "cx"].values,
                            xy.loc[pairs.home_kap, "cy"].values - xy.loc[pairs.dest_kap, "cy"].values) / 1000
pairs["P_j"] = pop.loc[pairs.dest_kap].values


def tobit(y, X, cen, c):
    X = np.asarray(X)
    b0 = np.linalg.lstsq(X[~cen], y[~cen], rcond=None)[0]

    def nll(t):
        b, s = t[:-1], np.exp(t[-1])
        mu = X @ b
        return -(stats.norm.logpdf(y[~cen], mu[~cen], s).sum() + stats.norm.logcdf((c - mu[cen]) / s).sum())

    return optimize.minimize(nll, np.r_[b0, 0.0], method="BFGS").x[:-1]


rows = []
for v in VARIANTS:
    od = pd.read_csv(f"{WORK}/od_monthly_raw_{v}.csv")
    hc = pd.read_csv(f"{WORK}/home_monthly_{v}.csv").set_index(["month", "home_kap"])["n_homed_maids"]
    for m in MONTHS:
        d = pairs.merge(od[od.month == m][["home_kap", "dest_kap", "n_maids"]], on=["home_kap", "dest_kap"], how="left")
        d["n_maids"] = d["n_maids"].fillna(0)
        d["H_i"] = hc.reindex(list(zip([m] * len(d), d.home_kap))).values
        d = d.dropna(subset=["H_i"])
        X = sm.add_constant(np.log(d[["H_i", "P_j", "dist_km"]]))
        pub = (d["n_maids"] >= K).values
        f = sm.OLS(np.log(d.loc[pub, "n_maids"]), X[pub]).fit()
        bt = tobit(np.where(pub, np.log(d["n_maids"].clip(lower=1)), np.log(K)), X, ~pub, np.log(K))
        t = d.set_index(["home_kap", "dest_kap"])["n_maids"]
        ij = [(i, j) for i, j in t.index if i < j and t[(i, j)] >= K and t.get((j, i), 0) >= K]
        a, b = np.array([t[p] for p in ij]), np.array([t[(p[1], p[0])] for p in ij])
        h = hc.loc[m].reindex(codes).fillna(0)
        rows.append({"variant": v, "month": m, "homed": int(h.sum()), "pairs_pub": int(pub.sum()),
                     "R2": f.rsquared, "gamma_ols": -f.params["dist_km"], "gamma_tobit": -bt[3],
                     "recip_r": stats.pearsonr(np.log(a), np.log(b))[0], "recip_pairs": len(ij),
                     "homes_bps_r_log": stats.pearsonr(np.log(pop.loc[codes]), np.log(h.clip(lower=1)))[0],
                     "homes_bps_r_lin_excl_gondomanan": stats.pearsonr(pop.drop("34.71.10", errors="ignore").reindex(codes).dropna(),
                                                                       h.reindex(pop.drop("34.71.10", errors="ignore").reindex(codes).dropna().index))[0]})

out = pd.DataFrame(rows)
out.round(3).to_csv("od_hotspot_sensitivity_summary.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(3).to_string(index=False))
