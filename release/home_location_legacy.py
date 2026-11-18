import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

from source_data import LEGACY_HOME, get_spark, raw_pings
from pyspark.sql import functions as F
from pyspark.sql import Window

spark = get_spark("home_location_legacy")  # Spark Connect or local; session time zone checked (release/source_data.py)

# LEGACY home definition, kept for the robustness check only (superseded by home_location_full_period.py).
# This is the code that produced LEGACY_HOME (source_data.py; 511,507 MAIDs) and the
# earlier validation table (data/validasi_kapanewon_des2021_apr2022_legacy_home.csv):
#   night pings 20:00-07:59 WIB over the whole period, coordinates rounded to 4 decimals (~11 m),
#   home = the most frequent bin. No minimum number of nights; ties are broken arbitrarily.
# Per-kapanewon counts: see check_home_location.py. The table is written only with WRITE_LEGACY_HOME=1.
df_limited = raw_pings(spark)

df_night = df_limited.withColumn("hour", F.hour("datetime")).filter((F.col("hour") >= 20) | (F.col("hour") < 8))
df_night_binned = (df_night.withColumn("lat_bin", F.round("latitude", 4))
                   .withColumn("lon_bin", F.round("longitude", 4)))
loc_freq = df_night_binned.groupBy("maid", "lat_bin", "lon_bin").agg(F.count("*").alias("n"))
w = Window.partitionBy("maid").orderBy(F.desc("n"))
home_location = (loc_freq.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") == 1)
                 .select("maid", F.col("lat_bin").alias("home_lat"), F.col("lon_bin").alias("home_lon"), "n"))

if __name__ == "__main__" and os.environ.get("WRITE_LEGACY_HOME") == "1":
    home_location.write.mode("overwrite").parquet(LEGACY_HOME)
