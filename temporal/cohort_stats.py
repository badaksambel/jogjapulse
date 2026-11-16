import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
import statsmodels.formula.api as smf

CSV_PATH = "cohort_daily_mobility.csv"
OUTPUT_CSV = "cohort_period_effects.csv"

# Periods (inclusive); baseline = pre-Nataru December
PERIODS = [
    ("baseline", "2021-12-01", "2021-12-23"),
    ("Nataru", "2021-12-24", "2022-01-02"),
    ("Jan-early Feb", "2022-01-03", "2022-02-07"),
    ("PPKM L3", "2022-02-08", "2022-03-07"),
    ("PPKM L4", "2022-03-08", "2022-03-21"),
    ("PPKM L3 (after L4)", "2022-03-22", "2022-04-02"),
    ("Ramadan", "2022-04-03", "2022-04-30"),
]
METRICS = {
    "share_away_2km": "Share of MAIDs > 2 km from home",
    "mean_rg_km": "Mean radius of gyration (km)",
}

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df["period"] = None
for name, start, end in PERIODS:
    df.loc[df["date"].between(start, end), "period"] = name
df["dow"] = df["date"].dt.day_name()

rows = []
for metric, label in METRICS.items():
    # OLS: metric ~ period + day-of-week; Newey-West SEs (7 lags) for serial correlation
    model = smf.ols(f"{metric} ~ C(period, Treatment('baseline')) + C(dow)", data=df).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7})
    base_mean = df.loc[df["period"] == "baseline", metric].mean()
    ci = model.conf_int()
    for name, _, _ in PERIODS[1:]:
        term = f"C(period, Treatment('baseline'))[T.{name}]"
        rows.append({
            "metric": metric,
            "period": name,
            "baseline_mean": base_mean,
            "effect": model.params[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
            "pct_of_baseline": model.params[term] / base_mean * 100,
            "p_value": model.pvalues[term],
            "n_days": int((df["period"] == name).sum()),
        })
    print(f"\n{label}: baseline mean = {base_mean:.3f}, R2 = {model.rsquared:.3f}")

res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
res.to_csv(OUTPUT_CSV, index=False)
print(f"Tersimpan: {OUTPUT_CSV} ({len(res)} baris)")
