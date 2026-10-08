import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window

# Tunnel address; override with SPARK_REMOTE=sc://localhost:15003 etc.
spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()

# Home-based visitation OD between kapanewon, per month (same definitions as od_feasibility.py).
#   Home:        per MAID and month, the kapanewon with the most distinct nights 22:00-05:59 WIB
#                (ties: more night pings); pings before 06:00 belong to the previous night;
#                a home requires >= MIN_NIGHTS distinct nights.
#   Destination: kapanewon where the MAID is observed 08:00-17:59 WIB in the same month.
#   Flow:        distinct MAIDs with home i observed in j during the month (i == j included).
# Only distinct MAID counts are published (not additive -> no differencing across rows).
K = 5
MIN_NIGHTS = 2
NIGHT_START, NIGHT_END = 22, 6
DAY_START, DAY_END = 8, 18
MONTHS = ["2021-12", "2022-01", "2022-02", "2022-03", "2022-04"]
REDUCED_QUALITY = {"2022-03", "2022-04"}  # small device panel, weak gravity fit (see README)
OUT_DIR = "output"

kap_pd = pd.read_csv("kapanewon_wkt.csv")
kap = spark.createDataFrame(kap_pd).withColumn("geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")

pings = (
    spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")
    .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))
    .select("maid", F.col("KEC_CODE").alias("kap"),
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
    night.groupBy("maid", "month", "kap")
    .agg(F.countDistinct("night_date").alias("n_nights"), F.count("*").alias("n_night_pings"))
    .withColumn("rk", F.row_number().over(w_home))
    .filter((F.col("rk") == 1) & (F.col("n_nights") >= MIN_NIGHTS))
    .select("maid", "month", F.col("kap").alias("origin_code"))
    .cache()
)

day = (
    pings.filter((F.col("hour") >= DAY_START) & (F.col("hour") < DAY_END))
    .withColumn("month", F.date_format("date", "yyyy-MM"))
    .select("maid", "month", F.col("kap").alias("dest_code"))
    .distinct()
)
visits = day.join(home, ["maid", "month"]).cache()

# OD flows
od = visits.groupBy("month", "origin_code", "dest_code").agg(F.countDistinct("maid").alias("n_maids")).toPandas()

# Origin denominators: homed MAIDs, and homed MAIDs observed at least once in daytime
homes = home.groupBy("month", "origin_code").agg(F.count("*").alias("n_homed_maids")).toPandas()
observed = visits.groupBy("month", "origin_code").agg(F.countDistinct("maid").alias("n_homed_maids_daytime")).toPandas()
origin = homes.merge(observed, on=["month", "origin_code"], how="left").fillna({"n_homed_maids_daytime": 0})

visits.unpersist()
home.unpersist()
pings.unpersist()

# ============================================================
# Labels, suppression, output
# ============================================================
labels = kap_pd[["KEC_CODE", "KEC_KAPN", "KAB_CODE", "KAB_KOTA"]]

def label(df, code_col, prefix):
    lab = labels.rename(columns={"KEC_CODE": code_col, "KEC_KAPN": f"{prefix}_kapanewon",
                                 "KAB_CODE": f"{prefix}_kab_code", "KAB_KOTA": f"{prefix}_kab_kota"})
    return df.merge(lab, on=code_col, how="left")

od_all = len(od)
od_pub = od[od["n_maids"] >= K]
od_pub = label(label(od_pub, "origin_code", "origin"), "dest_code", "dest")
od_pub["reduced_quality"] = od_pub["month"].isin(REDUCED_QUALITY)
od_pub = od_pub[["month", "origin_kab_code", "origin_kab_kota", "origin_code", "origin_kapanewon",
                 "dest_kab_code", "dest_kab_kota", "dest_code", "dest_kapanewon", "n_maids", "reduced_quality"]]
od_pub = od_pub.sort_values(["month", "origin_code", "dest_code"])

# Denominators: suppress rows where either count is below k (blank = fewer than k)
origin_pub = label(origin, "origin_code", "origin")
origin_pub = origin_pub[origin_pub["n_homed_maids"] >= K].copy()
origin_pub["n_homed_maids_daytime"] = origin_pub["n_homed_maids_daytime"].astype("Int64")  # nullable int: blank if < k
origin_pub.loc[origin_pub["n_homed_maids_daytime"] < K, "n_homed_maids_daytime"] = pd.NA
origin_pub["reduced_quality"] = origin_pub["month"].isin(REDUCED_QUALITY)
origin_pub = origin_pub[["month", "origin_kab_code", "origin_kab_kota", "origin_code", "origin_kapanewon",
                         "n_homed_maids", "n_homed_maids_daytime", "reduced_quality"]].sort_values(["month", "origin_code"])

od_pub.to_csv(f"{OUT_DIR}/od_home_visits_kapanewon_monthly.csv", index=False)
origin_pub.to_csv(f"{OUT_DIR}/od_origin_homes_monthly.csv", index=False)

inter = od[od["origin_code"] != od["dest_code"]]
summary = pd.DataFrame([{
    "k": K, "min_nights": MIN_NIGHTS,
    "od_rows_total": od_all, "od_rows_published": len(od_pub),
    "od_rows_suppressed_pct": round((1 - len(od_pub) / od_all) * 100, 1),
    "inter_zonal_flow_kept_pct": round(inter.loc[inter["n_maids"] >= K, "n_maids"].sum() / inter["n_maids"].sum() * 100, 1),
    "origin_rows_published": len(origin_pub),
}])
summary.to_csv(f"{OUT_DIR}/od_summary.csv", index=False)
print(summary.to_string(index=False))
print(f"Tersimpan: {OUT_DIR}/od_home_visits_kapanewon_monthly.csv ({len(od_pub):,} baris), "
      f"{OUT_DIR}/od_origin_homes_monthly.csv ({len(origin_pub)} baris), {OUT_DIR}/od_summary.csv")
