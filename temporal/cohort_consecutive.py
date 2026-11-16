import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window

spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()

df_filtered = spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")

# Active days per MAID per month (datetime = WIB wall clock, session tz UTC)
maid_month = (
    df_filtered
    .withColumn("date", F.to_date("datetime"))
    .withColumn("month_idx", (F.year("datetime") - 2021) * 12 + F.month("datetime"))  # Dec'21 = 12
    .groupBy("maid", "month_idx")
    .agg(F.countDistinct("date").alias("active_days"))
    .cache()
)

MIN_DAYS = [1, 3, 5, 10]  # minimum active days required in EACH month of the run
rows = []
for min_days in MIN_DAYS:
    # Longest run of consecutive months in which the MAID has >= min_days active days
    m = maid_month.filter(F.col("active_days") >= min_days)
    w = Window.partitionBy("maid").orderBy("month_idx")
    runs = (
        m.withColumn("grp", F.col("month_idx") - F.row_number().over(w))
        .groupBy("maid", "grp").agg(F.count("*").alias("run_len"))
        .groupBy("maid").agg(F.max("run_len").alias("max_run"))
        .groupBy("max_run").agg(F.count("*").alias("n_maids"))
        .toPandas()
    )
    for _, r in runs.iterrows():
        rows.append({"min_active_days_per_month": min_days,
                     "max_consecutive_months": int(r["max_run"]),
                     "n_maids": int(r["n_maids"])})

import pandas as pd

res = pd.DataFrame(rows).sort_values(["min_active_days_per_month", "max_consecutive_months"])
# Cumulative: MAIDs with a run of AT LEAST k consecutive months
res["n_maids_at_least"] = (
    res.sort_values("max_consecutive_months", ascending=False)
    .groupby("min_active_days_per_month")["n_maids"].cumsum()
)
print(res.to_string(index=False))
res.to_csv("cohort_consecutive_months.csv", index=False)
print("Tersimpan: cohort_consecutive_months.csv")

# Which month pairs do the >=2-consecutive-month MAIDs (>= 1 day each) come from?
pairs = (
    maid_month.alias("a").join(maid_month.alias("b"),
                               (F.col("a.maid") == F.col("b.maid")) &
                               (F.col("b.month_idx") == F.col("a.month_idx") + 1))
    .groupBy(F.col("a.month_idx").alias("first_month"))
    .agg(F.count("*").alias("n_maids_in_pair"),
         F.expr("percentile_approx(a.active_days + b.active_days, 0.5)").alias("median_days_2m"))
    .orderBy("first_month").toPandas()
)
print(pairs.to_string(index=False))
maid_month.unpersist()
