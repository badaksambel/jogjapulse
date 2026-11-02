"""Per-device observation statistics for the JogjaPulse source pings (reviewer comment M10).

Characterises the app-usage-gated intermittency of the panel:
  1. active days per MAID (whole period)
  2. pings per active day (per MAID-day)
  3. gap between consecutive pings of the same MAID (overall, and within the same day)
  4. per month: active MAIDs, MAIDs with >= 1 and >= 2 distinct nights
     (night = 22:00-05:59 WIB, pings before 06:00 belong to the previous night;
      same definition as release/tier1_od.py)

Only aggregate statistics are printed and written; no device-level data leave the cluster.

Submit:
  ./ilum.sh submit ilum_jobs/device_intermittency_stats.py device_intermittency \
      -F executors=4 -F executorMemory=8g -F executorCores=2
  ./ilum.sh wait <jobId>
  ./ilum.sh logs <jobId> 200000 | grep INTERM
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

SRC = "s3a://ilum-jogjapulse/mobility/gps_limited_user_month"
OUT = "s3a://ilum-data/tmp/jogjapulse_intermittency"
MONTHS = ["2021-12", "2022-01", "2022-02", "2022-03", "2022-04"]
Q = [0.1, 0.25, 0.5, 0.75, 0.9]
ACC = 10000  # percentile_approx accuracy

spark = SparkSession.builder.appName("device_intermittency_stats").getOrCreate()

# Ilum batch jobs only get per-bucket S3A settings for its own buckets (ilum-data, ilum-files, ...),
# not for ilum-jogjapulse. Both live on the same MinIO, so reuse the ilum-data settings (values never printed).
hconf = spark.sparkContext._jsc.hadoopConfiguration()
src_prefix, dst_prefix = "spark.hadoop.fs.s3a.bucket.ilum-data.", "fs.s3a.bucket.ilum-jogjapulse."
copied = [k[len(src_prefix):] for k, _ in spark.sparkContext.getConf().getAll() if k.startswith(src_prefix)]
for k in copied:
    hconf.set(dst_prefix + k, spark.sparkContext.getConf().get(src_prefix + k))
print("INTERM s3a settings copied to ilum-jogjapulse:", sorted(copied))
# datetime holds WIB wall-clock time; the session time zone stays UTC (see AGENTS.md)
print("INTERM spark", spark.version, "| session tz", spark.conf.get("spark.sql.session.timeZone"))

p = (spark.read.parquet(SRC)
     .select("maid", "datetime")
     .withColumn("date", F.to_date("datetime"))
     .withColumn("hour", F.hour("datetime"))
     .where(F.date_format("date", "yyyy-MM").isin(MONTHS))
     .cache())

n_pings = p.count()
n_maids = p.select("maid").distinct().count()
print(f"INTERM pings={n_pings} maids={n_maids}")


def summary(df, colname, label):
    """Mean and quantiles of one numeric column; printed and returned as a one-row dict."""
    r = df.agg(
        F.count(colname).alias("n"),
        F.mean(colname).alias("mean"),
        F.expr(f"percentile_approx({colname}, array({','.join(map(str, Q))}), {ACC})").alias("q"),
    ).first()
    row = {"metric": label, "n": r["n"], "mean": round(r["mean"], 3)}
    row.update({f"p{int(q * 100)}": float(v) for q, v in zip(Q, r["q"])})
    print("INTERM", " ".join(f"{k}={v}" for k, v in row.items()))
    return row


rows = []

# 1. Active days per MAID
active_days = p.groupBy("maid").agg(F.countDistinct("date").alias("active_days"))
rows.append(summary(active_days, "active_days", "active_days_per_maid"))

# 2. Pings per active day (MAID-day)
per_day = p.groupBy("maid", "date").agg(F.count("*").alias("pings"))
rows.append(summary(per_day, "pings", "pings_per_active_day"))

# 3. Gap between consecutive pings of the same MAID (minutes)
w = Window.partitionBy("maid").orderBy("datetime")
gaps = (p.withColumn("prev", F.lag("datetime").over(w))
         .withColumn("prev_date", F.lag("date").over(w))
         .where(F.col("prev").isNotNull())
         .withColumn("gap_min", (F.col("datetime").cast("long") - F.col("prev").cast("long")) / 60.0))
rows.append(summary(gaps, "gap_min", "gap_minutes_all"))
rows.append(summary(gaps.where(F.col("date") == F.col("prev_date")), "gap_min", "gap_minutes_same_day"))
zero = gaps.agg(F.mean((F.col("gap_min") == 0).cast("double")).alias("share")).first()["share"]
print(f"INTERM share_of_gaps_equal_0s={zero:.4f}  (duplicate timestamps)")

pd_summary = spark.createDataFrame(rows)
pd_summary.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT}/summary")

# 4. Monthly: active MAIDs and MAIDs with >= 1 / >= 2 distinct nights
active_m = (p.withColumn("month", F.date_format("date", "yyyy-MM"))
             .groupBy("month").agg(F.countDistinct("maid").alias("active_maids")))
nights = (p.where((F.col("hour") >= 22) | (F.col("hour") < 6))
           .withColumn("night_date", F.when(F.col("hour") < 6, F.date_sub("date", 1)).otherwise(F.col("date")))
           .withColumn("month", F.date_format("night_date", "yyyy-MM"))
           .where(F.col("month").isin(MONTHS))
           .groupBy("maid", "month").agg(F.countDistinct("night_date").alias("n_nights")))
night_m = nights.groupBy("month").agg(
    F.count("*").alias("maids_ge1_night"),
    F.sum((F.col("n_nights") >= 2).cast("int")).alias("maids_ge2_nights"),
)
monthly = (active_m.join(night_m, "month", "left")
           .withColumn("pct_ge2_of_active", F.round(F.col("maids_ge2_nights") / F.col("active_maids") * 100, 1))
           .withColumn("pct_ge2_of_ge1", F.round(F.col("maids_ge2_nights") / F.col("maids_ge1_night") * 100, 1))
           .orderBy("month"))
for r in monthly.collect():
    print("INTERM month", " ".join(f"{k}={v}" for k, v in r.asDict().items()))
monthly.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT}/monthly")

print(f"INTERM written to {OUT}/summary and {OUT}/monthly")
spark.stop()
