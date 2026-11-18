import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from source_data import LEGACY_HOME, get_spark, raw_pings
from pyspark.sql import functions as F

spark = get_spark("check_home_location")  # Spark Connect or local; session time zone checked (release/source_data.py)

# Checks the stored home locations (night mode of 4-decimal bins, 20:00-07:59) against the
# released dataset and the kapanewon validation table of dataset v1.0 (on the synthetic sample the
# counts do not match that table, which comes from the source data). Read-only.
home = spark.read.parquet(LEGACY_HOME)
gps_maids = raw_pings(spark).select("maid").distinct()

print("home rows:", home.count(),
      "| distinct maids in home:", home.select("maid").distinct().count(),
      "| maids in gps:", gps_maids.count(),
      "| overlap:", home.join(gps_maids, "maid").count())

# How many homes rest on very few night pings (n = pings in the chosen bin)
home.select(
    F.expr("percentile_approx(n, array(0.25, 0.5, 0.75, 0.9))").alias("n_quantiles"),
    F.avg((F.col("n") == 1).cast("double")).alias("share_n1"),
    F.avg((F.col("n") <= 2).cast("double")).alias("share_n_le2"),
).show(truncate=False)

# Reproduce the per-kapanewon counts and compare with the validation table
kap = spark.createDataFrame(pd.read_csv("data/kapanewon_wkt.csv")).withColumn("geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")
per_kap = (
    home.withColumn("pt", F.expr("ST_Point(home_lon, home_lat)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))
    .groupBy("KAB_KOTA", "KEC_KAPN").agg(F.count("*").alias("n_home"))
    .toPandas()
)
v = pd.read_csv("data/validasi_kapanewon_des2021_apr2022_legacy_home.csv")  # table built from this home
m = v.merge(per_kap, on=["KAB_KOTA", "KEC_KAPN"], how="left")
print("homes inside DIY:", int(per_kap["n_home"].sum()),
      "| sum n_users in validation table:", int(v["n_users"].sum()),
      "| kapanewon where n_home == n_users:", int((m["n_home"] == m["n_users"]).sum()), "of", len(m))
