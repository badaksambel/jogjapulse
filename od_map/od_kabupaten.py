import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window

# Tunnel address; override with SPARK_REMOTE=sc://localhost:15003 etc.
spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()

# Home-based visitation OD between kabupaten/kota, same definitions as release/tier1_od.py:
#   monthly home = kapanewon with most distinct nights 22:00-05:59 WIB (>= 2 nights), mapped to its kabupaten;
#   destination = kabupaten with a ping 08:00-17:59 WIB in the same month.
# Monthly flows are distinct devices; the period total is their sum (device-months).
# Distinct counts cannot be summed from the kapanewon-level product, hence this separate run.
K = 5
MIN_NIGHTS = 2
NIGHT_START, NIGHT_END = 22, 6
DAY_START, DAY_END = 8, 18
MONTHS = ["2021-12", "2022-01", "2022-02", "2022-03", "2022-04"]
OUTPUT_CSV = "od_kabupaten_monthly.csv"

kap_pd = pd.read_csv("../release/kapanewon_wkt.csv")
kap = spark.createDataFrame(kap_pd).withColumn("geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")

pings = (
    spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")
    .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))
    .select("maid", F.col("KEC_CODE").alias("kap"), F.col("KAB_CODE").alias("kab"),
            F.to_date("datetime").alias("date"),  # datetime = WIB wall clock, session tz UTC
            F.hour("datetime").alias("hour"))
    .cache()
)

night = (
    pings.filter((F.col("hour") >= NIGHT_START) | (F.col("hour") < NIGHT_END))
    .withColumn("night_date", F.when(F.col("hour") < NIGHT_END, F.date_sub("date", 1)).otherwise(F.col("date")))
    .withColumn("month", F.date_format("night_date", "yyyy-MM"))
    .filter(F.col("month").isin(MONTHS))
)
w_home = Window.partitionBy("maid", "month").orderBy(F.desc("n_nights"), F.desc("n_night_pings"), "kap")
home = (
    night.groupBy("maid", "month", "kap", "kab")
    .agg(F.countDistinct("night_date").alias("n_nights"), F.count("*").alias("n_night_pings"))
    .withColumn("rk", F.row_number().over(w_home))
    .filter((F.col("rk") == 1) & (F.col("n_nights") >= MIN_NIGHTS))
    .select("maid", "month", F.col("kab").alias("origin_kab_code"))
)

day = (
    pings.filter((F.col("hour") >= DAY_START) & (F.col("hour") < DAY_END))
    .withColumn("month", F.date_format("date", "yyyy-MM"))
    .select("maid", "month", F.col("kab").alias("dest_kab_code"))
    .distinct()
)

od = (
    day.join(home, ["maid", "month"])
    .groupBy("month", "origin_kab_code", "dest_kab_code")
    .agg(F.countDistinct("maid").alias("n_maids"))
    .orderBy("month", "origin_kab_code", "dest_kab_code")
    .toPandas()
)
pings.unpersist()

names = kap_pd[["KAB_CODE", "KAB_KOTA"]].drop_duplicates()
od = (od.merge(names.rename(columns={"KAB_CODE": "origin_kab_code", "KAB_KOTA": "origin_kab_kota"}), on="origin_kab_code")
        .merge(names.rename(columns={"KAB_CODE": "dest_kab_code", "KAB_KOTA": "dest_kab_kota"}), on="dest_kab_code"))
assert od["n_maids"].min() >= K, "a kabupaten pair is below k; suppress before publishing"
od.to_csv(OUTPUT_CSV, index=False)
print(od.groupby(["origin_kab_kota", "dest_kab_kota"])["n_maids"].sum().unstack().to_string())
print(f"Tersimpan: {OUTPUT_CSV} ({len(od)} baris), min n_maids = {od['n_maids'].min()}")
