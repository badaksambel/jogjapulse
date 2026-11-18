"""Monthly homes and OD flows with and without hotspot pings (reviewer M6), plus the hotspot distribution.

Hotspot: an exact coordinate pair reported by >= HOT distinct devices over the whole period.
Same rules as release/od_feasibility.py (home 22:00-05:59, >= 2 nights; destinations 08:00-17:59).
Outputs
  od_work/od_monthly_raw_<variant>.csv, home_monthly_<variant>.csv   UNSUPPRESSED, internal only
  sensitivity/hotspot_distribution.csv   points and pings by number of devices per exact point
(both under JOGJAPULSE_OUT)
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
from source_data import get_spark, out_dir, out_path, raw_pings  # noqa: E402
from pyspark.sql import Window, functions as F

spark = get_spark("od_hotspot_sensitivity")  # Spark Connect or local; session time zone checked (release/source_data.py)

WORK = out_dir("od_work")  # internal, unsuppressed: never release
NIGHT_START, NIGHT_END, DAY_START, DAY_END, MIN_NIGHTS = 22, 6, 8, 18, 2
VARIANTS = {"all": None, "nohot10000": 10000, "nohot1000": 1000, "nohot100": 100}
BINS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000]

src = raw_pings(spark)
pt = src.groupBy("latitude", "longitude").agg(F.countDistinct("maid").alias("pt_maids"), F.count("*").alias("pt_pings")).cache()

# Distribution of exact points by number of devices, with coordinate precision (decimals of latitude)
dec = F.length(F.regexp_extract(F.col("latitude").cast("string"), r"\.(\d+)$", 1))
bucket = F.lit(None)
for lo, hi in reversed(list(zip(BINS[:-1], BINS[1:]))):
    bucket = F.when((F.col("pt_maids") >= lo) & (F.col("pt_maids") < hi), F.lit(lo)).otherwise(bucket)
dist = (pt.withColumn("bin_from", bucket).groupBy("bin_from")
        .agg(F.count("*").alias("points"), F.sum("pt_pings").alias("pings"), F.sum("pt_maids").alias("device_points"),
             F.avg(dec).alias("mean_lat_decimals"), F.avg((dec <= 4).cast("int")).alias("share_le4_decimals"))
        .orderBy("bin_from").toPandas())
dist.to_csv(out_path("sensitivity", "hotspot_distribution.csv"), index=False)
print("HOTDIST\n" + dist.to_string(index=False))

kap = spark.createDataFrame(pd.read_csv("../release/data/kapanewon_wkt.csv")).withColumn(
    "geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")
pings = (
    src.join(pt.select("latitude", "longitude", "pt_maids"), ["latitude", "longitude"])
    .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))
    .select("maid", F.col("KEC_CODE").alias("kap"), F.to_date("datetime").alias("date"),
            F.hour("datetime").alias("hour"), "pt_maids")
    .cache()
)

w_home = Window.partitionBy("maid", "month").orderBy(F.desc("n_nights"), F.desc("n_night_pings"), "kap")
for name, hot in VARIANTS.items():
    p = pings if hot is None else pings.filter(F.col("pt_maids") < hot)
    night = (p.filter((F.col("hour") >= NIGHT_START) | (F.col("hour") < NIGHT_END))
             .withColumn("night_date", F.when(F.col("hour") < NIGHT_END, F.date_sub("date", 1)).otherwise(F.col("date")))
             .withColumn("month", F.date_format("night_date", "yyyy-MM")))
    home = (night.groupBy("maid", "month", "kap")
            .agg(F.countDistinct("night_date").alias("n_nights"), F.count("*").alias("n_night_pings"))
            .withColumn("rk", F.row_number().over(w_home)).filter("rk = 1")
            .filter(F.col("n_nights") >= MIN_NIGHTS).select("maid", "month", F.col("kap").alias("home_kap")))
    day = (p.filter((F.col("hour") >= DAY_START) & (F.col("hour") < DAY_END))
           .select("maid", F.date_format("date", "yyyy-MM").alias("month"), F.col("kap").alias("dest_kap")).distinct())
    od = (day.join(home, ["maid", "month"]).groupBy("month", "home_kap", "dest_kap")
          .agg(F.countDistinct("maid").alias("n_maids")).toPandas())
    hc = home.groupBy("month", "home_kap").agg(F.count("*").alias("n_homed_maids")).toPandas()
    od.to_csv(f"{WORK}/od_monthly_raw_{name}.csv", index=False)
    hc.to_csv(f"{WORK}/home_monthly_{name}.csv", index=False)
    print(f"HOTOD {name}: {len(od)} OD rows, homed device-months {hc.n_homed_maids.sum():,}")

pings.unpersist()
pt.unpersist()
