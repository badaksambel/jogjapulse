import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from pyspark.sql import SparkSession, functions as F

# Tunnel address; override with SPARK_REMOTE=sc://localhost:15003 etc.
spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()

K = int(os.environ.get("TIER1_K", 5))  # minimum distinct MAIDs per published row (small-count suppression)
CELL_M = 500      # grid cell size in metres, UTM zone 49S (EPSG:32749)
OUT_DIR = "output" if K == 5 else f"output_k{K}"
os.makedirs(OUT_DIR, exist_ok=True)

# Disclosure control between the two products:
#   A (cell x day) publishes n_maids only; B (kapanewon x hour) publishes n_maids and n_pings.
#   n_pings is additive, so publishing it in both would let users recover suppressed cells by
#   subtracting visible cells from kapanewon totals. Distinct MAID counts are not additive across
#   cells or hours, so they are safe in both products.

# Kapanewon polygons (dissolved from Geoportal DIY 2023 kalurahan, see make_kapanewon_wkt.py)
kap = spark.createDataFrame(pd.read_csv("data/kapanewon_wkt.csv")).withColumn("geom", F.expr("ST_GeomFromWKT(wkt)")).drop("wkt")

raw = spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")
n_raw = raw.count()

pings = (
    raw
    .withColumn("pt", F.expr("ST_Point(longitude, latitude)"))
    .join(F.broadcast(kap), F.expr("ST_Contains(geom, pt)"))  # keep pings inside DIY only
    .withColumn("utm", F.expr("ST_Transform(pt, 'EPSG:4326', 'EPSG:32749')"))
    .select(
        "maid", "KAB_CODE", "KAB_KOTA", "KEC_CODE", "KEC_KAPN",
        F.to_date("datetime").alias("date"),  # datetime = WIB wall clock, session tz UTC
        F.hour("datetime").alias("hour"),
        F.floor(F.expr("ST_X(utm)") / CELL_M).cast("int").alias("cell_e"),
        F.floor(F.expr("ST_Y(utm)") / CELL_M).cast("int").alias("cell_n"),
    )
    .cache()
)
n_diy = pings.count()
print(f"pings: {n_raw:,} raw, {n_diy:,} inside DIY ({n_diy / n_raw * 100:.1f}%)")

# ============================================================
# A. Presence per 500 m cell per day
# ============================================================
cell_day = (
    pings.groupBy("cell_e", "cell_n", "date")
    .agg(F.countDistinct("maid").alias("n_maids"))
    .cache()
)
a_rows_total = cell_day.count()
a = cell_day.filter(F.col("n_maids") >= K)

# Cell geometry helpers: lower-left corner in UTM and centroid in WGS84
cells = (
    a.select("cell_e", "cell_n").distinct()
    .withColumn("x_min_utm", F.col("cell_e") * CELL_M)
    .withColumn("y_min_utm", F.col("cell_n") * CELL_M)
    .withColumn("c", F.expr(f"ST_Transform(ST_Point(x_min_utm + {CELL_M / 2}, y_min_utm + {CELL_M / 2}), "
                            "'EPSG:32749', 'EPSG:4326')"))
    .withColumn("centroid_lon", F.round(F.expr("ST_X(c)"), 6))
    .withColumn("centroid_lat", F.round(F.expr("ST_Y(c)"), 6))
    .drop("c")
)
a_pd = (
    a.join(cells, ["cell_e", "cell_n"])
    .withColumn("cell_id", F.concat(F.lit("E"), F.col("cell_e"), F.lit("_N"), F.col("cell_n")))
    .select("cell_id", "x_min_utm", "y_min_utm", "centroid_lon", "centroid_lat", "date", "n_maids")
    .orderBy("date", "cell_id")
    .toPandas()
)

# ============================================================
# B. Presence per kapanewon per hour
# ============================================================
kap_hour = (
    pings.groupBy("KAB_CODE", "KAB_KOTA", "KEC_CODE", "KEC_KAPN", "date", "hour")
    .agg(F.countDistinct("maid").alias("n_maids"), F.count("*").alias("n_pings"))
    .cache()
)
b_rows_total = kap_hour.count()
b_pd = (
    kap_hour.filter(F.col("n_maids") >= K)
    .withColumnRenamed("KAB_CODE", "kab_code").withColumnRenamed("KAB_KOTA", "kab_kota")
    .withColumnRenamed("KEC_CODE", "kapanewon_code").withColumnRenamed("KEC_KAPN", "kapanewon")
    .orderBy("date", "hour", "kapanewon_code")
    .toPandas()
)

# Share of presence lost to suppression (sum of distinct MAIDs over published vs all rows)
lost = {}
for name, df in (("A_cell500m_daily", cell_day), ("B_kapanewon_hourly", kap_hour)):
    s = df.agg(F.sum("n_maids").alias("all"),
               F.sum(F.when(F.col("n_maids") >= K, F.col("n_maids")).otherwise(0)).alias("kept")).toPandas().iloc[0]
    lost[name] = round((1 - s["kept"] / s["all"]) * 100, 1)

a_pd.to_csv(f"{OUT_DIR}/presence_cell500m_daily.csv", index=False)
b_pd.to_csv(f"{OUT_DIR}/presence_kapanewon_hourly.csv", index=False)

summary = pd.DataFrame([
    {"product": "A_cell500m_daily", "k": K, "rows_total": a_rows_total, "rows_published": len(a_pd),
     "rows_suppressed_pct": round((1 - len(a_pd) / a_rows_total) * 100, 1),
     "maid_presence_suppressed_pct": lost["A_cell500m_daily"],
     "n_cells_published": a_pd["cell_id"].nunique()},
    {"product": "B_kapanewon_hourly", "k": K, "rows_total": b_rows_total, "rows_published": len(b_pd),
     "rows_suppressed_pct": round((1 - len(b_pd) / b_rows_total) * 100, 1),
     "maid_presence_suppressed_pct": lost["B_kapanewon_hourly"],
     "n_cells_published": b_pd["kapanewon_code"].nunique()},
])
summary.insert(0, "pings_inside_diy_pct", round(n_diy / n_raw * 100, 1))
summary.to_csv(f"{OUT_DIR}/tier1_summary.csv", index=False)
print(summary.to_string(index=False))
print(f"Tersimpan: {OUT_DIR}/presence_cell500m_daily.csv ({len(a_pd):,} baris), "
      f"{OUT_DIR}/presence_kapanewon_hourly.csv ({len(b_pd):,} baris), {OUT_DIR}/tier1_summary.csv")

pings.unpersist()
cell_day.unpersist()
kap_hour.unpersist()
