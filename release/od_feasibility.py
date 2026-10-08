import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window

# Tunnel address; override with SPARK_REMOTE=sc://localhost:15003 etc.
spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()

# Feasibility of a home-based origin-destination (OD) product at kapanewon level.
# Outputs are UNSUPPRESSED internal counts for analysis only -- do not release od_work/.
OUT_DIR = "od_work"
os.makedirs(OUT_DIR, exist_ok=True)

NIGHT_START, NIGHT_END = 22, 6   # home: 22:00-05:59 WIB; pings before 06:00 belong to the previous night
DAY_START, DAY_END = 8, 18       # destinations: 08:00-17:59 WIB
OD_MIN_NIGHTS = 2                # home must be observed on >= 2 distinct nights in the month

kap = spark.createDataFrame(pd.read_csv("kapanewon_wkt.csv")).withColumn("geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")

pings = (
    spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")
    .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))
    .select(
        "maid",
        F.col("KEC_CODE").alias("kap"),
        F.to_date("datetime").alias("date"),  # datetime = WIB wall clock, session tz UTC
        F.hour("datetime").alias("hour"),
    )
    .cache()
)
print(f"pings inside DIY: {pings.count():,}")

# ============================================================
# 1. MONTHLY HOME: kapanewon with most distinct nights (ties: more pings)
# ============================================================
night = (
    pings.filter((F.col("hour") >= NIGHT_START) | (F.col("hour") < NIGHT_END))
    .withColumn("night_date", F.when(F.col("hour") < NIGHT_END, F.date_sub("date", 1)).otherwise(F.col("date")))
    .withColumn("month", F.date_format("night_date", "yyyy-MM"))
)
w_home = Window.partitionBy("maid", "month").orderBy(F.desc("n_nights"), F.desc("n_night_pings"), "kap")
home = (
    night.groupBy("maid", "month", "kap")
    .agg(F.countDistinct("night_date").alias("n_nights"), F.count("*").alias("n_night_pings"))
    .withColumn("rk", F.row_number().over(w_home))
    .filter("rk = 1")
    .select("maid", "month", F.col("kap").alias("home_kap"), "n_nights")
    .cache()
)

# Homes per kapanewon per month, by minimum-nights threshold
home_counts = (
    home.select("month", "home_kap",
                *[(F.col("n_nights") >= t).cast("int").alias(f"n_home_min{t}") for t in (1, 2, 3)])
    .groupBy("month", "home_kap").sum()
    .toDF("month", "home_kap", "n_home_min1", "n_home_min2", "n_home_min3")
    .orderBy("month", "home_kap").toPandas()
)
home_counts.to_csv(f"{OUT_DIR}/home_monthly_kapanewon.csv", index=False)

# ============================================================
# 2. DAYTIME DESTINATIONS of MAIDs with a reliable home
# ============================================================
day = (
    pings.filter((F.col("hour") >= DAY_START) & (F.col("hour") < DAY_END))
    .withColumn("month", F.date_format("date", "yyyy-MM"))
    .withColumn("week", F.date_trunc("week", "date").cast("date"))
    .select("maid", "month", "week", "date", F.col("kap").alias("dest_kap"))
    .distinct()
)
homed = home.filter(F.col("n_nights") >= OD_MIN_NIGHTS).select("maid", "month", "home_kap")
visits = day.join(homed, ["maid", "month"]).cache()

# Coverage: homed MAIDs with any daytime observation / any daytime visit outside home
coverage = (
    visits.groupBy("maid", "month", "home_kap")
    .agg(F.max((F.col("dest_kap") != F.col("home_kap")).cast("int")).alias("goes_out"))
    .groupBy("month").agg(F.count("*").alias("homed_with_daytime"), F.sum("goes_out").alias("homed_going_out"))
    .toPandas()
)
homed_n = homed.groupBy("month").agg(F.count("*").alias("homed_maids")).toPandas()
coverage = homed_n.merge(coverage, on="month", how="left").sort_values("month")
coverage.to_csv(f"{OUT_DIR}/od_coverage_monthly.csv", index=False)
print(coverage.to_string(index=False))

# ============================================================
# 3. OD COUNTS (distinct MAIDs home i -> visited j), monthly and weekly
# ============================================================
od_month = (
    visits.groupBy("month", "home_kap", "dest_kap")
    .agg(F.countDistinct("maid").alias("n_maids"), F.countDistinct("maid", "date").alias("n_maid_days"))
    .orderBy("month", "home_kap", "dest_kap").toPandas()
)
od_month.to_csv(f"{OUT_DIR}/od_monthly_raw.csv", index=False)

od_week = (
    visits.groupBy("week", "home_kap", "dest_kap")
    .agg(F.countDistinct("maid").alias("n_maids"))
    .orderBy("week", "home_kap", "dest_kap").toPandas()
)
od_week.to_csv(f"{OUT_DIR}/od_weekly_raw.csv", index=False)

print(f"Tersimpan di {OUT_DIR}/: home_monthly_kapanewon.csv ({len(home_counts)}), od_coverage_monthly.csv, "
      f"od_monthly_raw.csv ({len(od_month)}), od_weekly_raw.csv ({len(od_week)})")

visits.unpersist()
home.unpersist()
pings.unpersist()
