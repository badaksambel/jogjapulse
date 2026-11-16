"""Checks of the device-level source data: schema, time zone, duplicates, default coordinates.

Prints aggregate counts only (prefix SRC). Run through Spark Connect:
    ./ilum.sh tunnel-bg && .venv-spark/bin/python validation/source_data_checks.py
"""
import os

from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()

# datetime is stored as WIB wall-clock time in a timestamp column. Spark functions such as hour() and
# to_date() read it in the session time zone, so the session must be UTC for them to return WIB.
tz = spark.conf.get("spark.sql.session.timeZone")
print(f"SRC session time zone: {tz}")
assert tz in ("UTC", "Etc/UTC", "GMT", "Z", "+00:00"), "session time zone must be UTC"

df = spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")
print("SRC columns:", ", ".join(f"{f.name}:{f.dataType.simpleString()}" for f in df.schema.fields))

# DIY bounding box (generous) for the default/outlier coordinate check
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -8.25, -7.50, 110.00, 110.85

d = df.withColumn("wib_from_epoch", F.from_unixtime(F.col("timestamp") + 7 * 3600).cast("timestamp"))
row = d.agg(
    F.count("*").alias("rows"),
    F.countDistinct("maid").alias("maids"),
    F.min("datetime").alias("dt_min"), F.max("datetime").alias("dt_max"),
    F.sum(F.when(F.col("datetime") != F.col("wib_from_epoch"), 1).otherwise(0)).alias("dt_ne_epoch_plus_7h"),
    F.sum(F.when(F.date_format("datetime", "yyyyMM").cast("int") != F.col("month"), 1).otherwise(0)).alias("month_mismatch"),
    F.sum(F.when(F.col("timestamp") != F.floor("timestamp"), 1).otherwise(0)).alias("fractional_epoch"),
    *[F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in ["maid", "latitude", "longitude", "timestamp", "datetime"]],
    F.sum(((F.col("latitude") == 0) | (F.col("longitude") == 0)).cast("int")).alias("zero_coord"),
    F.sum((~F.col("latitude").between(LAT_MIN, LAT_MAX) | ~F.col("longitude").between(LON_MIN, LON_MAX)).cast("int")).alias("outside_diy_bbox"),
    F.min("latitude").alias("lat_min"), F.max("latitude").alias("lat_max"),
    F.min("longitude").alias("lon_min"), F.max("longitude").alias("lon_max"),
).first().asDict()
for k, v in row.items():
    print(f"SRC {k}: {v}")

# Duplicates: identical (maid, timestamp, latitude, longitude), and identical (maid, timestamp)
dup_full = df.count() - df.dropDuplicates(["maid", "timestamp", "latitude", "longitude"]).count()
dup_time = df.count() - df.dropDuplicates(["maid", "timestamp"]).count()
print(f"SRC exact duplicate rows (maid, timestamp, lat, lon): {dup_full}")
print(f"SRC rows sharing (maid, timestamp) with another row: {dup_time}")

# Default coordinates: single points reported by unusually many devices (centroids, geocoder defaults)
top = (df.groupBy("latitude", "longitude")
       .agg(F.countDistinct("maid").alias("n_maids"), F.count("*").alias("n_pings"))
       .orderBy(F.desc("n_maids")).limit(10).collect())
for r in top:
    print(f"SRC top point lat {r.latitude:.6f} lon {r.longitude:.6f}: {r.n_maids} devices, {r.n_pings} pings")

# Coordinate precision: number of decimals of latitude as stored
prec = (df.select(F.length(F.regexp_extract(F.col("latitude").cast("string"), r"\.(\d+)$", 1)).alias("dec"))
        .groupBy("dec").count().orderBy("dec").collect())
print("SRC latitude decimals:", {r.dec: r["count"] for r in prec})
