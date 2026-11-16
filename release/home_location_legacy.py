import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window

# Tunnel address; override with SPARK_REMOTE=sc://localhost:15003 etc.
spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()
# datetime is WIB wall-clock time stored without a zone; hour()/to_date() read it in the session time zone,
# so the session must be UTC (checked, never changed here)
assert spark.conf.get("spark.sql.session.timeZone") in ("UTC", "Etc/UTC", "GMT"), "Spark session time zone must be UTC"

# LEGACY home definition, kept for the robustness check only (superseded by home_location_full_period.py).
# This is the code that produced s3a://ilum-jogjapulse/mobility/home_location (511,507 MAIDs) and the
# earlier validation table (data/validasi_kapanewon_des2021_apr2022_legacy_home.csv):
#   night pings 20:00-07:59 WIB over the whole period, coordinates rounded to 4 decimals (~11 m),
#   home = the most frequent bin. No minimum number of nights; ties are broken arbitrarily.
# Per-kapanewon counts: see check_home_location.py. Not re-run here (the stored output is reused).
df_limited = spark.read.parquet("s3a://ilum-jogjapulse/mobility/gps_limited_user_month")

df_night = df_limited.withColumn("hour", F.hour("datetime")).filter((F.col("hour") >= 20) | (F.col("hour") < 8))
df_night_binned = (df_night.withColumn("lat_bin", F.round("latitude", 4))
                   .withColumn("lon_bin", F.round("longitude", 4)))
loc_freq = df_night_binned.groupBy("maid", "lat_bin", "lon_bin").agg(F.count("*").alias("n"))
w = Window.partitionBy("maid").orderBy(F.desc("n"))
home_location = (loc_freq.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") == 1)
                 .select("maid", F.col("lat_bin").alias("home_lat"), F.col("lon_bin").alias("home_lon"), "n"))

if __name__ == "__main__" and os.environ.get("WRITE_LEGACY_HOME") == "1":
    home_location.write.mode("overwrite").parquet("s3a://ilum-jogjapulse/mobility/home_location")
