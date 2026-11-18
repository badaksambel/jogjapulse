"""List the default coordinates removed before aggregation (see source_data.py).

Lists latitude, longitude, n_devices, n_pings for every exact coordinate pair reported by
>= DEFAULT_MIN_DEVICES distinct devices, plus the kapanewon it falls in. On the source data this
produced data/default_points.csv (the 25 points removed in the article). This copy writes to
sample_run/default_points_found.csv so that list is kept (the synthetic sample has no default points):
    .venv-spark/bin/python release/find_default_points.py
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from pyspark.sql import functions as F

from source_data import DEFAULT_MIN_DEVICES, get_spark, out_path, raw_pings

spark = get_spark("find_default_points")

raw = raw_pings(spark)
kap = spark.createDataFrame(pd.read_csv("data/kapanewon_wkt.csv")).withColumn(
    "geom", F.expr("ST_GeomFromWKT(wkt)")).select("KAB_KOTA", "KEC_KAPN", "geom")
pts = (raw.groupBy("latitude", "longitude")
       .agg(F.countDistinct("maid").alias("n_devices"), F.count("*").alias("n_pings"))
       .filter(F.col("n_devices") >= DEFAULT_MIN_DEVICES)
       .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
       .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"), "left")
       .drop("pt", "geom").orderBy(F.desc("n_devices")).toPandas())
pts.to_csv(out_path("default_points_found.csv"), index=False)

n_raw = raw.count()
n_dev = raw.select("maid").distinct().count()
removed = pts["n_pings"].sum()
print(pts.to_string(index=False))
print(f"DEFAULT {len(pts)} points, {removed:,} of {n_raw:,} pings ({removed / n_raw:.1%}); raw devices {n_dev:,}")
