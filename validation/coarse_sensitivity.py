"""Sensitivity of the full-period home table to coarse and default coordinates (reviewer M6).

A ping is flagged as
  coarse   both coordinates lie on a 0.01-degree grid (<= 2 decimals, about 1 km or coarser)
  hotspot  its exact coordinate pair is reported by >= HOT devices over the whole period
           (GPS fixes of many devices do not coincide exactly; such points are likely IP, cell or
           geocoded locations, e.g. the Yogyakarta city centroid in Gondomanan)
Homes are recomputed with the rules of release/home_location_full_period.py for several variants.
Output under JOGJAPULSE_OUT/sensitivity/: coarse_sensitivity_homes.csv (devices per home kapanewon per
variant; aggregates only) and coarse_sensitivity_stats.csv (ping and device counts per variant).
Analysis: coarse_sensitivity_analysis.py.
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
from source_data import get_spark, out_path, raw_pings  # noqa: E402
from pyspark.sql import Window, functions as F

spark = get_spark("coarse_sensitivity")  # Spark Connect or local; session time zone checked (release/source_data.py)

MIN_NIGHTS = 2
NIGHT_START, NIGHT_END = 22, 6
PERIOD = ("2021-12-01", "2022-04-30")


def on_grid(c, dec, tol):
    """Coordinate lies on a 10^-dec degree grid (longitude is stored with float32 precision: looser tolerance)."""
    x = F.col(c) * (10 ** dec)
    return F.abs(x - F.round(x)) < tol


src = raw_pings(spark)
pt_maids = src.groupBy("latitude", "longitude").agg(F.countDistinct("maid").alias("pt_maids"))
pings = (src.join(pt_maids, ["latitude", "longitude"])
         .withColumn("coarse", on_grid("latitude", 2, 1e-5) & on_grid("longitude", 2, 1e-3)))

VARIANTS = {  # name -> filter that keeps a ping
    "all": F.lit(True),
    "no_coarse": ~F.col("coarse"),
    "no_hot10000": F.col("pt_maids") < 10000,
    "no_hot1000": F.col("pt_maids") < 1000,
    "no_hot200": F.col("pt_maids") < 200,
    "no_coarse_hot1000": ~F.col("coarse") & (F.col("pt_maids") < 1000),
    "no_coarse_hot100": ~F.col("coarse") & (F.col("pt_maids") < 100),
}

kap_pd = pd.read_csv("../release/data/kapanewon_wkt.csv")
kap = spark.createDataFrame(kap_pd).withColumn("geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")

night = (
    pings.withColumn("hour", F.hour("datetime"))
    .filter((F.col("hour") >= NIGHT_START) | (F.col("hour") < NIGHT_END))
    .withColumn("date", F.to_date("datetime"))
    .withColumn("night_date", F.when(F.col("hour") < NIGHT_END, F.date_sub("date", 1)).otherwise(F.col("date")))
    .filter(F.col("night_date").between(*PERIOD))
    .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))
    .select("maid", F.col("KEC_CODE").alias("kap"), "night_date", "coarse", "pt_maids")
    .cache()
)

w_home = Window.partitionBy("maid").orderBy(F.desc("n_nights"), F.desc("n_night_pings"), "kap")
homes, stats = [], []
for name, keep in VARIANTS.items():
    n_all = pings.filter(keep).agg(F.count("*").alias("p"), F.countDistinct("maid").alias("m")).first()
    home = (night.filter(keep).groupBy("maid", "kap")
            .agg(F.countDistinct("night_date").alias("n_nights"), F.count("*").alias("n_night_pings"))
            .withColumn("rk", F.row_number().over(w_home)).filter("rk = 1"))
    s = home.agg(F.count("*").alias("night_maids"),
                 F.sum((F.col("n_nights") >= MIN_NIGHTS).cast("long")).alias("homed")).first()
    per = (home.filter(F.col("n_nights") >= MIN_NIGHTS).groupBy("kap").agg(F.count("*").alias("n_users"))
           .toPandas().rename(columns={"kap": "KEC_CODE"}))
    per["variant"] = name
    homes.append(per)
    stats.append({"variant": name, "pings": n_all.p, "maids": n_all.m,
                  "maids_with_night_pings": s.night_maids, "maids_with_home": s.homed})
    print("SENS", stats[-1])

night.unpersist()
pd.concat(homes).to_csv(out_path("sensitivity", "coarse_sensitivity_homes.csv"), index=False)
pd.DataFrame(stats).to_csv(out_path("sensitivity", "coarse_sensitivity_stats.csv"), index=False)
print("SENS saved coarse_sensitivity_homes.csv, coarse_sensitivity_stats.csv")
