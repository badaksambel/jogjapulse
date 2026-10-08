import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

WORK = "od_work"  # internal, unsuppressed counts from od_feasibility.py
K = 5

# Kapanewon codes, names, centroids (UTM 49S) and BPS population
kap = gpd.GeoDataFrame(pd.read_csv("kapanewon_wkt.csv"),
                       geometry=gpd.GeoSeries.from_wkt(pd.read_csv("kapanewon_wkt.csv")["wkt"]), crs="EPSG:4326")
kap = kap.to_crs("EPSG:32749")
kap["cx"], kap["cy"] = kap.geometry.centroid.x, kap.geometry.centroid.y
val = pd.read_csv("../validation/validasi_kapanewon_des2021_apr2022.csv")
kap = kap.merge(val[["KAB_KOTA", "KEC_KAPN", "pop_actual"]], on=["KAB_KOTA", "KEC_KAPN"])
kap = pd.DataFrame(kap[["KEC_CODE", "KEC_KAPN", "cx", "cy", "pop_actual"]])
assert len(kap) == 78
name = kap.set_index("KEC_CODE")["KEC_KAPN"]
pop = kap.set_index("KEC_CODE")["pop_actual"]

# ============================================================
# 1. Monthly home counts vs BPS population, by minimum-nights threshold
# ============================================================
home = pd.read_csv(f"{WORK}/home_monthly_kapanewon.csv")
# Pings before 06:00 on 1 Dec belong to the night of 30 Nov; drop that partial "month"
home = home[home["month"] >= "2021-12"]
rows = []
for month, g in home.groupby("month"):
    g = g.set_index("home_kap").reindex(kap["KEC_CODE"]).fillna(0)
    keep = (name.reindex(g.index) != "GONDOMANAN").values
    for t in (1, 2, 3):
        x, y = pop.reindex(g.index).values[keep], g[f"n_home_min{t}"].values[keep]
        ok = y > 0
        rows.append({"month": month, "min_nights": t, "homed_maids": int(g[f"n_home_min{t}"].sum()),
                     "r_linear": round(pearsonr(x, y)[0], 3),
                     "r_loglog": round(pearsonr(np.log10(x[ok]), np.log10(y[ok]))[0], 3),
                     "kapanewon_with_homes": int(ok.sum())})
home_res = pd.DataFrame(rows)
print("== Homes per month vs BPS (excl. Gondomanan) ==\n", home_res.to_string(index=False))

# ============================================================
# 2. OD suppression at k: monthly and weekly (inter-zonal flows, i != j)
# ============================================================
def suppression(od, period_col):
    out = []
    for p, g in od.groupby(period_col):
        inter = g[g["home_kap"] != g["dest_kap"]]
        kept = inter[inter["n_maids"] >= K]
        out.append({period_col: p,
                    "pairs_nonzero": len(inter), "pairs_published": len(kept),
                    "pairs_published_pct_of_6006": round(len(kept) / (78 * 77) * 100, 1),
                    "flow_kept_pct": round(kept["n_maids"].sum() / inter["n_maids"].sum() * 100, 1),
                    "intra_share_pct": round(g.loc[g["home_kap"] == g["dest_kap"], "n_maids"].sum()
                                             / g["n_maids"].sum() * 100, 1)})
    return pd.DataFrame(out)

od_m = pd.read_csv(f"{WORK}/od_monthly_raw.csv")
od_w = pd.read_csv(f"{WORK}/od_weekly_raw.csv")
sup_m = suppression(od_m, "month")
sup_w = suppression(od_w, "week")
print("\n== Monthly OD, k = 5 ==\n", sup_m.to_string(index=False))
print("\n== Weekly OD, k = 5 (summary over weeks) ==\n",
      sup_w[["pairs_published", "flow_kept_pct"]].describe().round(1).loc[["mean", "min", "max"]].to_string())

# ============================================================
# 3. Gravity model on published monthly inter-zonal pairs:
#    log T_ij = a + b log H_i + c log P_j - gamma log d_ij
#    H_i = homed MAIDs in origin (panel size), P_j = BPS population of destination
# ============================================================
xy = kap.set_index("KEC_CODE")[["cx", "cy"]]
homes2 = home.set_index(["month", "home_kap"])["n_home_min2"]
grav, sym = [], []
for month, g in od_m.groupby("month"):
    g = g[(g["home_kap"] != g["dest_kap"]) & (g["n_maids"] >= K)].copy()
    o, d = xy.loc[g["home_kap"]].values, xy.loc[g["dest_kap"]].values
    g["dist_km"] = np.hypot(*(o - d).T) / 1000
    g["H_i"] = homes2.loc[list(zip([month] * len(g), g["home_kap"]))].values
    g["P_j"] = pop.loc[g["dest_kap"]].values
    X = sm.add_constant(np.log(g[["H_i", "P_j", "dist_km"]]))
    fit = sm.OLS(np.log(g["n_maids"]), X).fit()
    grav.append({"month": month, "pairs": len(g), "R2": round(fit.rsquared, 3),
                 "b_origin": round(fit.params["H_i"], 2), "c_dest_pop": round(fit.params["P_j"], 2),
                 "gamma_distance": round(-fit.params["dist_km"], 2)})
    # Symmetry of flows i->j vs j->i (pairs published in both directions)
    t = g.set_index(["home_kap", "dest_kap"])["n_maids"]
    both = [(i, j) for i, j in t.index if (j, i) in t.index and i < j]
    if both:
        a, b = np.log([t[p] for p in both]), np.log([t[(p[1], p[0])] for p in both])
        sym.append({"month": month, "pairs_both_directions": len(both), "r_log_ij_ji": round(pearsonr(a, b)[0], 3)})
grav_res, sym_res = pd.DataFrame(grav), pd.DataFrame(sym)
print("\n== Gravity model (published monthly inter-zonal pairs) ==\n", grav_res.to_string(index=False))
print("\n== Symmetry ==\n", sym_res.to_string(index=False))

# Top destinations (inflow of distinct MAIDs from other kapanewon), December
dec = od_m[(od_m["month"] == "2021-12") & (od_m["home_kap"] != od_m["dest_kap"])]
top = dec.groupby("dest_kap")["n_maids"].sum().sort_values(ascending=False).head(8)
print("\n== Top destinations, Dec 2021 ==\n", pd.DataFrame({"kapanewon": name.loc[top.index].values,
                                                           "inflow_maids": top.values}).to_string(index=False))

home_res.to_csv(f"{WORK}/analysis_home_vs_bps.csv", index=False)
sup_m.to_csv(f"{WORK}/analysis_od_suppression_monthly.csv", index=False)
sup_w.to_csv(f"{WORK}/analysis_od_suppression_weekly.csv", index=False)
grav_res.merge(sym_res, on="month", how="left").to_csv(f"{WORK}/analysis_gravity.csv", index=False)
print(f"\nTersimpan di {WORK}/: analysis_*.csv")
