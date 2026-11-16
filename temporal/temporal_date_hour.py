"""Pings and distinct MAIDs per date x hour (WIB), for per-day means in Figure 5.

The earlier temporal_hourly.csv / temporal_dow.csv count distinct MAIDs over all days that share
an hour or a weekday; this table lets the figure show means of daily values instead.
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()
# datetime is WIB wall-clock time stored without a zone; hour()/to_date() read it in the session time zone,
# so the session must be UTC (checked, never changed here)
assert spark.conf.get("spark.sql.session.timeZone") in ("UTC", "Etc/UTC", "GMT"), "Spark session time zone must be UTC"

# datetime is WIB wall-clock time; the session time zone stays UTC (do not change it)
df = spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")
out = (
    df.withColumn("date", F.to_date("datetime"))
    .withColumn("hour", F.hour("datetime"))
    .groupBy("date", "hour")
    .agg(F.count("*").alias("n_pings"), F.countDistinct("maid").alias("n_unique_users"))
    .orderBy("date", "hour")
    .toPandas()
)
out.to_csv("temporal_date_hour.csv", index=False)
print(f"Saved temporal_date_hour.csv: {len(out)} rows, {out.n_pings.sum():,} pings")
