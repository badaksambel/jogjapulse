"""Pings and distinct MAIDs per date x hour (WIB), for per-day means in Figure 5.

The earlier temporal_hourly.csv / temporal_dow.csv count distinct MAIDs over all days that share
an hour or a weekday; this table lets the figure show means of daily values instead.
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
from paths import fig_path  # noqa: E402
from source_data import get_spark, raw_pings  # noqa: E402
from pyspark.sql import functions as F

spark = get_spark("temporal_date_hour")  # Spark Connect or local; session time zone checked (release/source_data.py)

# datetime is WIB wall-clock time; the session time zone stays UTC (do not change it)
df = raw_pings(spark)
out = (
    df.withColumn("date", F.to_date("datetime"))
    .withColumn("hour", F.hour("datetime"))
    .groupBy("date", "hour")
    .agg(F.count("*").alias("n_pings"), F.countDistinct("maid").alias("n_unique_users"))
    .orderBy("date", "hour")
    .toPandas()
)
out.to_csv(fig_path("temporal_date_hour.csv"), index=False)
print(f"Saved temporal_date_hour.csv: {len(out)} rows, {out.n_pings.sum():,} pings")
