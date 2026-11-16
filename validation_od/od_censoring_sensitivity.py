"""Sensitivity of the gravity and reciprocity statistics (Figure 6) to the k = 5 censoring of OD flows.

Per month, compares:
  published   OLS on log flows, published pairs only (n >= 5): the values in the paper and Figure 6
  tobit       Tobit on log flows, public data only: every ordered pair i != j that is not published
              is treated as left-censored at log(5); reproducible from the released files
  all_pos     OLS on log flows, all pairs with >= 1 device (internal, unsuppressed counts)
  ppml        Poisson pseudo-maximum likelihood on all pairs including zeros (internal), the usual
              gravity estimator when flows contain zeros (Santos Silva & Tenreyro 2006)
Reciprocity: Pearson r of log flows for pairs published in both directions vs all pairs with >= 1
device in both directions (internal), plus Spearman rho over all pairs including zeros (internal).

Internal counts come from ../release/od_work/od_monthly_raw.csv (od_feasibility.py). They are NOT
released; only the aggregate statistics in od_censoring_sensitivity.csv are written.
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
OD_CSV = "../release/output/od_home_visits_kapanewon_monthly.csv"
ORIGIN_CSV = "../release/output/od_origin_homes_monthly.csv"
KAP_WKT = "../release/kapanewon_wkt.csv"
POP_CSV = "../release/output/home_maids_vs_population_kapanewon.csv"
RAW_CSV = "../release/od_work/od_monthly_raw.csv"  # internal, unsuppressed
OUT_CSV = "od_censoring_sensitivity.csv"

kap = pd.read_csv(KAP_WKT)
geo = gpd.GeoSeries.from_wkt(kap["wkt"], crs="EPSG:4326").to_crs("EPSG:32749").centroid
xy = pd.DataFrame({"cx": geo.x.values, "cy": geo.y.values}, index=kap["KEC_CODE"])
pop = (pd.read_csv(POP_CSV).merge(kap[["KAB_KOTA", "KEC_KAPN", "KEC_CODE"]], on=["KAB_KOTA", "KEC_KAPN"])
       .set_index("KEC_CODE")["pop_actual"])
codes = list(pop.index)

pub = pd.read_csv(OD_CSV)
origin = pd.read_csv(ORIGIN_CSV)
homes = origin.set_index(["month", "origin_code"])["n_homed_maids"]
raw = pd.read_csv(RAW_CSV).rename(columns={"home_kap": "origin_code", "dest_kap": "dest_code"})


def all_pairs(month):
    """Every ordered pair i != j of kapanewon with covariates, published and internal flows (0 if absent)."""
    d = pd.DataFrame(list(itertools.permutations(codes, 2)), columns=["origin_code", "dest_code"])
    d["month"] = month
    d["dist_km"] = np.hypot(xy.loc[d.origin_code, "cx"].values - xy.loc[d.dest_code, "cx"].values,
                            xy.loc[d.origin_code, "cy"].values - xy.loc[d.dest_code, "cy"].values) / 1000
    d["H_i"] = homes.reindex(list(zip(d.month, d.origin_code))).values
    d["P_j"] = pop.loc[d.dest_code].values
    key = ["month", "origin_code", "dest_code"]
    d = d.merge(pub[key + ["n_maids"]].rename(columns={"n_maids": "pub"}), on=key, how="left")
    d = d.merge(raw[key + ["n_maids"]].rename(columns={"n_maids": "raw"}), on=key, how="left")
    d["raw"] = d["raw"].fillna(0)
    return d.dropna(subset=["H_i"])


def design(d):
    return sm.add_constant(np.log(d[["H_i", "P_j", "dist_km"]]))


def tobit(y, X, censored, c):
    """Left-censored (at c) normal regression by maximum likelihood; returns coefficients and sigma."""
    X = np.asarray(X)
    beta0 = np.linalg.lstsq(X[~censored], y[~censored], rcond=None)[0]

    def nll(theta):
        b, s = theta[:-1], np.exp(theta[-1])
        mu = X @ b
        ll_obs = stats.norm.logpdf(y[~censored], mu[~censored], s)
        ll_cen = stats.norm.logcdf((c - mu[censored]) / s)
        return -(ll_obs.sum() + ll_cen.sum())

    res = optimize.minimize(nll, np.r_[beta0, 0.0], method="BFGS")
    return res.x[:-1], np.exp(res.x[-1])


def reciprocity(d, col, min_flow):
    t = d.set_index(["origin_code", "dest_code"])[col]
    ij = [(i, j) for i, j in t.index if i < j]
    a, b = np.array([t[p] for p in ij]), np.array([t[(p[1], p[0])] for p in ij])
    keep = (a >= min_flow) & (b >= min_flow)
    return stats.pearsonr(np.log(a[keep]), np.log(b[keep]))[0], int(keep.sum()), stats.spearmanr(a, b)[0]


rows = []
for month in sorted(pub["month"].unique()):
    d = all_pairs(month)
    X = design(d)
    names = ["const", "b_origin", "c_dest", "gamma"]
    is_pub = d["pub"].notna().values

    f = sm.OLS(np.log(d.loc[is_pub, "pub"]), X[is_pub]).fit()
    r_pub = {"n": int(is_pub.sum()), "R2": f.rsquared, **dict(zip(names, f.params.values))}

    # Tobit: published values observed; everything else (true 0-4, or a complementary-suppressed cell) censored
    y = np.where(is_pub, np.log(d["pub"].fillna(1)), np.log(K))
    bt, s = tobit(y, X, ~is_pub, np.log(K))
    r_tob = {"n": len(d), **dict(zip(names, bt))}

    pos = d["raw"].values > 0
    f = sm.OLS(np.log(d.loc[pos, "raw"]), X[pos]).fit()
    r_all = {"n": int(pos.sum()), "R2": f.rsquared, **dict(zip(names, f.params.values))}

    f = sm.GLM(d["raw"], X, family=sm.families.Poisson()).fit(cov_type="HC1")
    r_ppml = {"n": len(d), "pseudo_R2": np.corrcoef(d["raw"], f.fittedvalues)[0, 1] ** 2,
              **dict(zip(names, f.params.values))}

    rec_pub, n_rec_pub, _ = reciprocity(d.fillna({"pub": 0}), "pub", K)
    rec_all, n_rec_all, rho_all = reciprocity(d, "raw", 1)

    for method, r in [("published", r_pub), ("tobit", r_tob), ("all_pos", r_all), ("ppml", r_ppml)]:
        rows.append({"month": month, "method": method, "n_pairs": r["n"],
                     "R2": r.get("R2", r.get("pseudo_R2", np.nan)),
                     "b_origin": r["b_origin"], "c_dest": r["c_dest"], "gamma": -r["gamma"]})
    rows.append({"month": month, "method": "reciprocity_published", "n_pairs": n_rec_pub, "r": rec_pub})
    rows.append({"month": month, "method": "reciprocity_all_pos", "n_pairs": n_rec_all, "r": rec_all,
                 "rho_incl_zeros": rho_all})

out = pd.DataFrame(rows)
out.round(3).to_csv(OUT_CSV, index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
