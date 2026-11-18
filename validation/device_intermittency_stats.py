"""Observation intermittency of the panel, after removal of the default coordinates (Background section).

  1. active days per MAID (whole period)
  2. pings per active day (per MAID-day)
  3. gap between consecutive pings of the same MAID (overall, and within the same day)
  4. per month: active MAIDs, MAIDs with >= 1 and >= 2 distinct nights
     (night = 22:00-05:59 WIB, pings before 06:00 belong to the previous night)

Only aggregate statistics are printed (prefix INTERM) and written to intermittency_summary.csv and
intermittency_monthly.csv under JOGJAPULSE_OUT.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
import pandas as pd  # noqa: E402
from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from source_data import get_spark, load_pings, out_path  # noqa: E402

MONTHS = ["2021-12", "2022-01", "2022-02", "2022-03", "2022-04"]
Q = [0.1, 0.25, 0.5, 0.75, 0.9]
ACC = 10000  # percentile_approx accuracy

spark = get_spark("device_intermittency_stats")
p = (load_pings(spark)
     .select("maid", "datetime")
     .withColumn("date", F.to_date("datetime"))
     .withColumn("hour", F.hour("datetime"))
     .where(F.date_format("date", "yyyy-MM").isin(MONTHS))
     .cache())
print(f"INTERM pings={p.count()} maids={p.select('maid').distinct().count()}")


def summary(df, colname, label):
    """Mean and quantiles of one numeric column."""
    r = df.agg(F.count(colname).alias("n"), F.mean(colname).alias("mean"),
               F.expr(f"percentile_approx({colname}, array({','.join(map(str, Q))}), {ACC})").alias("q")).first()
    row = {"metric": label, "n": r["n"], "mean": round(r["mean"], 3)}
    row.update({f"p{int(q * 100)}": float(v) for q, v in zip(Q, r["q"])})
    print("INTERM", " ".join(f"{k}={v}" for k, v in row.items()))
    return row


rows = [summary(p.groupBy("maid").agg(F.countDistinct("date").alias("active_days")), "active_days", "active_days_per_maid"),
        summary(p.groupBy("maid", "date").agg(F.count("*").alias("pings")), "pings", "pings_per_active_day")]

w = Window.partitionBy("maid").orderBy("datetime")
gaps = (p.withColumn("prev", F.lag("datetime").over(w))
        .withColumn("prev_date", F.lag("date").over(w))
        .where(F.col("prev").isNotNull())
        .withColumn("gap_min", (F.col("datetime").cast("long") - F.col("prev").cast("long")) / 60.0))
rows.append(summary(gaps, "gap_min", "gap_minutes_all"))
rows.append(summary(gaps.where(F.col("date") == F.col("prev_date")), "gap_min", "gap_minutes_same_day"))
pd.DataFrame(rows).to_csv(out_path("intermittency_summary.csv"), index=False)

active_m = (p.withColumn("month", F.date_format("date", "yyyy-MM"))
            .groupBy("month").agg(F.countDistinct("maid").alias("active_maids")))
nights = (p.where((F.col("hour") >= 22) | (F.col("hour") < 6))
          .withColumn("night_date", F.when(F.col("hour") < 6, F.date_sub("date", 1)).otherwise(F.col("date")))
          .withColumn("month", F.date_format("night_date", "yyyy-MM"))
          .where(F.col("month").isin(MONTHS))
          .groupBy("maid", "month").agg(F.countDistinct("night_date").alias("n_nights")))
night_m = nights.groupBy("month").agg(F.count("*").alias("maids_ge1_night"),
                                      F.sum((F.col("n_nights") >= 2).cast("int")).alias("maids_ge2_nights"))
monthly = (active_m.join(night_m, "month", "left")
           .withColumn("pct_ge2_of_active", F.round(F.col("maids_ge2_nights") / F.col("active_maids") * 100, 1))
           .withColumn("pct_ge2_of_ge1", F.round(F.col("maids_ge2_nights") / F.col("maids_ge1_night") * 100, 1))
           .orderBy("month").toPandas())
print("INTERM monthly\n" + monthly.to_string(index=False))
monthly.to_csv(out_path("intermittency_monthly.csv"), index=False)
