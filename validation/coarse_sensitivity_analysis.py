"""Home-vs-population statistics per coarse/default-coordinate variant (output of coarse_sensitivity.py)."""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUTLIER = "GONDOMANAN"
kap = pd.read_csv("../release/data/kapanewon_wkt.csv")
kap["area_km2"] = gpd.GeoSeries.from_wkt(kap["wkt"], crs=4326).to_crs(32749).area.values / 1e6
pop = pd.read_csv("../release/output/home_maids_vs_population_kapanewon.csv")[["KAB_KOTA", "KEC_KAPN", "pop_actual"]]
base = kap[["KAB_KOTA", "KEC_KAPN", "KEC_CODE", "area_km2"]].merge(pop, on=["KAB_KOTA", "KEC_KAPN"])
base["density"] = base["pop_actual"] / base["area_km2"]

homes = pd.read_csv("coarse_sensitivity_homes.csv")
stats = pd.read_csv("coarse_sensitivity_stats.csv").set_index("variant")

rows = []
for v, h in homes.groupby("variant", sort=False):
    d = base.merge(h[["KEC_CODE", "n_users"]], on="KEC_CODE", how="left").fillna({"n_users": 0})
    d["ratio"] = d["n_users"] / d["pop_actual"]
    x = d[d["KEC_KAPN"] != OUTLIER]
    r = {"variant": v, "maids_with_home": int(stats.loc[v, "maids_with_home"]),
         "pings_kept_pct": 100 * stats.loc[v, "pings"] / stats.loc["all", "pings"],
         "gondomanan_ratio": d.loc[d["KEC_KAPN"] == OUTLIER, "ratio"].iloc[0]}
    for tag, t in [("77", x), ("78", d)]:
        r[f"r_lin_{tag}"] = pearsonr(t["pop_actual"], t["n_users"])[0]
        pos = t[t["n_users"] > 0]
        r[f"r_log_{tag}"] = pearsonr(np.log10(pos["pop_actual"]), np.log10(pos["n_users"]))[0]
        r[f"rho_{tag}"] = spearmanr(t["pop_actual"], t["n_users"])[0]
    r["ratio_min_77"], r["ratio_max_77"] = x["ratio"].min(), x["ratio"].max()
    r["ratio_fold_77"] = r["ratio_max_77"] / r["ratio_min_77"]
    r["rho_ratio_density_77"] = spearmanr(x["ratio"], x["density"])[0]
    top5 = d.nlargest(5, "n_users")
    r["top5"] = ", ".join(top5["KEC_KAPN"].str.title())
    r["top5_share_maids"] = top5["n_users"].sum() / d["n_users"].sum()
    r["top5_share_pop"] = top5["pop_actual"].sum() / d["pop_actual"].sum()
    rows.append(r)

out = pd.DataFrame(rows)
out.round(3).to_csv("coarse_sensitivity_summary.csv", index=False)
pd.set_option("display.width", 250)
print(out.drop(columns="top5").round(3).T.to_string())
print(out[["variant", "top5"]].to_string(index=False))
