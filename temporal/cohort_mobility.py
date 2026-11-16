import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window

spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()

df_filtered = spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")

# datetime = jam lokal WIB (epoch + 7 jam), session timezone server = UTC -> to_date/hour sudah WIB
# Proyeksi sederhana lat/lon -> km (equirectangular, cukup akurat untuk skala DIY, lat ~ -7.8)
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * 0.9907  # cos(7.8 deg)
HOME_GRID_DEG = 0.005             # ~550 m grid untuk lokasi rumah
AWAY_KM = 2.0                     # "keluar rumah" = titik terjauh > 2 km dari rumah
MIN_PINGS_DAY = 3                 # radius of gyration hanya untuk MAID-hari dengan >= 3 ping

pts = (
    df_filtered
    .withColumn("date", F.to_date("datetime"))
    .withColumn("hour", F.hour("datetime"))
    .withColumn("ym", F.date_format("datetime", "yyyy-MM"))
    .withColumn("week", F.date_trunc("week", "datetime"))
    .withColumn("x_km", F.col("longitude") * KM_PER_DEG_LON)
    .withColumn("y_km", F.col("latitude") * KM_PER_DEG_LAT)
    .select("maid", "date", "hour", "ym", "week", "latitude", "longitude", "x_km", "y_km")
    .cache()
)
print(f"pings: {pts.count():,}")

# ============================================================
# 1. BERAPA BULAN / MINGGU TIAP MAID AKTIF
# ============================================================
maid_activity = pts.groupBy("maid").agg(
    F.countDistinct("ym").alias("n_months"),
    F.countDistinct("week").alias("n_weeks"),
    F.countDistinct("date").alias("n_days"),
).cache()

n_weeks_total = pts.select("week").distinct().count()
month_dist = maid_activity.groupBy("n_months").agg(
    F.count("*").alias("n_maids"),
    F.expr("percentile_approx(n_days, 0.5)").alias("median_active_days"),
).orderBy("n_months").toPandas()
print(month_dist)
print(f"total weeks in data: {n_weeks_total}")

# ============================================================
# 2. KOHORT STABIL = aktif di kelima bulan (Des 2021 - Apr 2022)
# ============================================================
cohort = maid_activity.filter(F.col("n_months") == 5).select("maid", "n_weeks", "n_days")
cohort_pd = cohort.toPandas()
n_cohort = len(cohort_pd)
print(f"stable cohort (active in all 5 months): {n_cohort:,} MAIDs")
print(cohort_pd[["n_weeks", "n_days"]].describe().round(1))

pts_c = pts.join(F.broadcast(cohort.select("maid")), "maid")

# ============================================================
# 3. LOKASI RUMAH = grid ~550 m paling sering pada malam hari (22:00-05:59)
# ============================================================
night = pts_c.filter((F.col("hour") >= 22) | (F.col("hour") < 6)).withColumn(
    "cell", F.concat_ws("_",
                        F.floor(F.col("latitude") / HOME_GRID_DEG).cast("long"),
                        F.floor(F.col("longitude") / HOME_GRID_DEG).cast("long")))
w_home = Window.partitionBy("maid").orderBy(F.desc("n"))
home = (
    night.groupBy("maid", "cell")
    .agg(F.count("*").alias("n"), F.avg("x_km").alias("home_x"), F.avg("y_km").alias("home_y"))
    .withColumn("rk", F.row_number().over(w_home))
    .filter("rk = 1")
    .select("maid", "home_x", "home_y")
)

# ============================================================
# 4. METRIK HARIAN PER MAID (kohort saja)
# ============================================================
maid_day = (
    pts_c.join(home, "maid", "left")
    .withColumn("d_home_km", F.sqrt((F.col("x_km") - F.col("home_x")) ** 2 +
                                    (F.col("y_km") - F.col("home_y")) ** 2))
    .groupBy("maid", "date")
    .agg(
        F.count("*").alias("n_pings"),
        F.sqrt(F.var_pop("x_km") + F.var_pop("y_km")).alias("rg_km"),
        F.max("d_home_km").alias("max_d_home_km"),
    )
)

daily = (
    maid_day.groupBy("date")
    .agg(
        F.count("*").alias("n_active_maids"),
        F.expr("percentile_approx(n_pings, 0.5)").alias("median_pings_per_maid"),
        F.expr("percentile_approx(case when n_pings >= %d then rg_km end, 0.5)" % MIN_PINGS_DAY)
         .alias("median_rg_km"),
        F.avg(F.when(F.col("n_pings") >= MIN_PINGS_DAY, F.col("rg_km"))).alias("mean_rg_km"),
        F.avg((F.col("max_d_home_km") > AWAY_KM).cast("double")).alias("share_away_2km"),
        F.sum(F.col("max_d_home_km").isNull().cast("int")).alias("n_no_home"),
    )
    .withColumn("share_cohort_active", F.col("n_active_maids") / F.lit(n_cohort))
    .orderBy("date")
)
daily_pd = daily.toPandas()
print(daily_pd.head(10).to_string())

# ============================================================
# Simpan hasil ke CSV (di folder temporal/)
# ============================================================
outputs = {
    "cohort_month_distribution.csv": month_dist,
    "cohort_daily_mobility.csv": daily_pd,
}
for filename, pdf in outputs.items():
    pdf.to_csv(filename, index=False)
    print(f"Tersimpan: {filename} ({len(pdf)} baris)")

pts.unpersist()
maid_activity.unpersist()
