import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from pyspark.sql import functions as F

from source_data import get_spark, load_pings, out_path
from pyspark.sql import Window

spark = get_spark("home_location_full_period")  # Spark Connect or local; session time zone checked (source_data.py)

# Full-period home per MAID, using the same rules as the monthly OD home (tier1_od.py):
#   the kapanewon with the most distinct nights 22:00-05:59 WIB (ties: more night pings, then
#   kapanewon code); pings before 06:00 belong to the previous night; >= MIN_NIGHTS distinct nights.
# Output: devices per home kapanewon vs BPS population (the spatial validation table).
# The v1.0 table (20:00-07:59 mode of 4-decimal bins, no minimum) is data/validasi_*_legacy_home.csv
# (home_location_legacy.py).
MIN_NIGHTS = 2
NIGHT_START, NIGHT_END = 22, 6
PERIOD = ("2021-12-01", "2022-04-30")  # night dates
POPULATION_CSV = "data/bps_population_kapanewon_2021.csv"  # input: BPS population per kapanewon
HOME_CSV = out_path("output", "home_maids_vs_population_kapanewon.csv")  # released; read by validation/ and validation_od/

kap_pd = pd.read_csv("data/kapanewon_wkt.csv")
kap = spark.createDataFrame(kap_pd).withColumn("geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")

night = (
    load_pings(spark)  # source pings without default coordinates (source_data.py)
    .withColumn("hour", F.hour("datetime"))  # datetime = WIB wall clock, session tz UTC
    .filter((F.col("hour") >= NIGHT_START) | (F.col("hour") < NIGHT_END))
    .withColumn("date", F.to_date("datetime"))
    .withColumn("night_date", F.when(F.col("hour") < NIGHT_END, F.date_sub("date", 1)).otherwise(F.col("date")))
    .filter(F.col("night_date").between(*PERIOD))
    .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))
    .select("maid", F.col("KEC_CODE").alias("kap"), "night_date")
)

w_home = Window.partitionBy("maid").orderBy(F.desc("n_nights"), F.desc("n_night_pings"), "kap")
home = (
    night.groupBy("maid", "kap")
    .agg(F.countDistinct("night_date").alias("n_nights"), F.count("*").alias("n_night_pings"))
    .withColumn("rk", F.row_number().over(w_home))
    .filter(F.col("rk") == 1)
)

# Devices with any night ping in DIY, and those meeting the minimum (for the methods text)
stats = home.agg(F.count("*").alias("maids_with_night_pings"),
                 F.sum((F.col("n_nights") >= MIN_NIGHTS).cast("long")).alias("maids_with_home"),
                 F.expr("percentile_approx(n_nights, array(0.25, 0.5, 0.75))").alias("n_nights_q")).toPandas()
print("HOMESTATS", stats.to_dict("records"))

per_kap = (
    home.filter(F.col("n_nights") >= MIN_NIGHTS)
    .groupBy("kap").agg(F.count("*").alias("n_users"))
    .toPandas()
    .rename(columns={"kap": "KEC_CODE"})
)

# ============================================================
# Home-located devices vs BPS population per kapanewon (released file)
# ============================================================
pop = pd.read_csv(POPULATION_CSV)
new = (kap_pd[["KAB_KOTA", "KEC_KAPN", "KEC_CODE"]]
       .merge(per_kap, on="KEC_CODE", how="left").fillna({"n_users": 0})
       .merge(pop, on=["KAB_KOTA", "KEC_KAPN"], how="inner"))
assert len(new) == len(pop) == 78, (len(new), len(pop))
new["n_users"] = new["n_users"].astype(int)
new["pop_actual"] = new["pop_actual"].astype(float)
new["ratio"] = new["n_users"] / new["pop_actual"]
new["weight"] = new["pop_actual"] / new["n_users"]  # post-stratification weight P_i / H_i (residents per homed device)
new = new[["KAB_KOTA", "KEC_KAPN", "n_users", "pop_actual", "ratio", "weight"]].sort_values("ratio", ascending=False)

new.to_csv(HOME_CSV, index=False)
print(f"Saved {HOME_CSV}: {new['n_users'].sum():,} home-located devices in {len(new)} kapanewon")
