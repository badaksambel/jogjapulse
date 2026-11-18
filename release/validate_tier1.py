import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from paths import out_path

K = int(os.environ.get("TIER1_K", 5))  # which Tier-1 build to validate (see tier1_presence.py)
OUT_DIR = "output" if K == 5 else f"output_k{K}"
HOURLY_CSV = out_path(OUT_DIR, "presence_kapanewon_hourly.csv")
VALIDATION_CSV = out_path("output", "home_maids_vs_population_kapanewon.csv")  # home_location_full_period.py
OUTPUT_CSV = out_path(OUT_DIR, "validation_night_presence_kapanewon.csv")

NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]  # 22:00-05:59 WIB, same night window as the home definition
PERIODS = {
    "Dec 2021 - Apr 2022": ("2021-12-01", "2022-04-30"),
    "Dec 2021 - Jan 2022": ("2021-12-01", "2022-01-31"),  # largest panel, before the Feb drop
}

b = pd.read_csv(HOURLY_CSV, parse_dates=["date"])
val = pd.read_csv(VALIDATION_CSV)  # KAB_KOTA, KEC_KAPN, n_users (home-based MAIDs), pop_actual (BPS)

night = b[b["hour"].isin(NIGHT_HOURS)]
n_slots_total = night["date"].nunique() * len(NIGHT_HOURS)

rows, per_kap = [], []
for label, (start, end) in PERIODS.items():
    sub = night[night["date"].between(start, end)]
    n_slots = sub["date"].nunique() * len(NIGHT_HOURS)
    # Mean hourly night presence over published (unsuppressed) rows, and share of night slots published
    kap = sub.groupby(["kab_kota", "kapanewon"]).agg(
        night_presence=("n_maids", "mean"), slots_published=("n_maids", "size")).reset_index()
    kap["slots_published_pct"] = kap["slots_published"] / n_slots * 100
    df = kap.merge(val, left_on=["kab_kota", "kapanewon"], right_on=["KAB_KOTA", "KEC_KAPN"], how="inner")
    # Kapanewon whose night hours are all below k have no published night rows and drop out here
    missing = sorted(set(val["KEC_KAPN"]) - set(df["kapanewon"]))
    print(f"{label}: {len(df)} of {len(val)} kapanewon with published night rows; excluded: {missing}")

    # All kapanewon: with default coordinates removed upstream there is no outlier
    for name, mask in (("all kapanewon", df["kapanewon"].notna()),):
        r_lin, _ = pearsonr(df.loc[mask, "pop_actual"], df.loc[mask, "night_presence"])
        r_pop, p_pop = pearsonr(np.log10(df.loc[mask, "pop_actual"]), np.log10(df.loc[mask, "night_presence"]))
        r_home, p_home = pearsonr(np.log10(df.loc[mask, "n_users"]), np.log10(df.loc[mask, "night_presence"]))
        rows.append({"period": label, "sample": name, "n": int(mask.sum()),
                     "r_night_vs_bps_pop_linear": round(r_lin, 3),
                     "r_night_vs_bps_pop": round(r_pop, 3), "p_bps": p_pop,
                     "r_night_vs_home_maids": round(r_home, 3), "p_home": p_home,
                     "min_slots_published_pct": round(df.loc[mask, "slots_published_pct"].min(), 1)})
    per_kap.append(df.assign(period=label)[["period", "kab_kota", "kapanewon", "night_presence",
                                            "slots_published_pct", "n_users", "pop_actual"]])

res = pd.DataFrame(rows)
print(res.to_string(index=False))
pd.concat(per_kap).to_csv(OUTPUT_CSV, index=False)
print(f"Tersimpan: {OUTPUT_CSV}")
