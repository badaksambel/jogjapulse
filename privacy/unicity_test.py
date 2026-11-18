import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
from source_data import get_spark, out_path, raw_pings  # noqa: E402
from pyspark.sql import functions as F
from pyspark.sql import Window

spark = get_spark("unicity_test")  # Spark Connect or local; session time zone checked (release/source_data.py)

OUTPUT_CSV = out_path("unicity_results.csv")

# Unicity test (de Montjoye et al. 2013, Sci Rep 3:1376):
# for a random sample of MAIDs, draw p random spatio-temporal points from each trace and count
# how many MAIDs in the WHOLE dataset contain all p points. Unique = only the MAID itself matches.
SETTINGS = [  # (label, grid size km, time bin minutes)
    ("~100 m / 30 min", 0.1, 30),
    ("500 m / 30 min", 0.5, 30),
    ("500 m / 1 h", 0.5, 60),
    ("1 km / 1 h", 1.0, 60),
    ("1 km / 4 h", 1.0, 240),
    ("2 km / 4 h", 2.0, 240),
]
POINTS = [2, 4]
N_SAMPLE = 5000
SEED = 42

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * 0.9907  # cos(7.8 deg), DIY

pings = (
    raw_pings(spark)
    .select(
        "maid",
        F.to_date("datetime").alias("date"),
        (F.hour("datetime") * 60 + F.minute("datetime")).alias("minute_of_day"),
        (F.col("longitude") * KM_PER_DEG_LON).alias("x_km"),
        (F.col("latitude") * KM_PER_DEG_LAT).alias("y_km"),
    )
    .cache()
)
print(f"pings: {pings.count():,}")

rows = []
for label, grid_km, bin_min in SETTINGS:
    # Distinct (MAID, cell, date, time bin) tuples at this resolution
    traces = pings.select(
        "maid",
        F.concat_ws(
            "_",
            F.floor(F.col("x_km") / grid_km).cast("long"),
            F.floor(F.col("y_km") / grid_km).cast("long"),
            F.col("date").cast("string"),
            F.floor(F.col("minute_of_day") / bin_min).cast("int"),
        ).alias("key"),
    ).distinct().cache()

    n_points = traces.groupBy("maid").agg(F.count("*").alias("n_pts"))

    for p in POINTS:
        # Sample MAIDs that have at least p distinct points at this resolution
        eligible = n_points.filter(F.col("n_pts") >= p)
        n_eligible = eligible.count()
        sample = eligible.orderBy(F.rand(SEED)).limit(N_SAMPLE).select("maid")

        # p random points per sampled MAID
        w = Window.partitionBy("maid").orderBy(F.rand(SEED + p))
        chosen = (
            traces.join(F.broadcast(sample), "maid")
            .withColumn("rk", F.row_number().over(w))
            .filter(F.col("rk") <= p)
            .select(F.col("maid").alias("target"), "key")
        )

        # All MAIDs whose trace contains every chosen point of the target
        matches = (
            traces.join(F.broadcast(chosen), "key")
            .groupBy("target", "maid").agg(F.count("*").alias("hits"))
            .filter(F.col("hits") == p)
            .groupBy("target").agg(F.count("*").alias("n_matching"))
        )
        stats = matches.agg(
            F.count("*").alias("n_tested"),
            F.avg((F.col("n_matching") == 1).cast("double")).alias("unicity"),
            F.avg((F.col("n_matching") < 5).cast("double")).alias("share_lt5"),
            F.expr("percentile_approx(n_matching, 0.5)").alias("median_matching"),
        ).toPandas().iloc[0]

        row = {
            "setting": label, "grid_km": grid_km, "time_bin_min": bin_min, "p_points": p,
            "n_eligible_maids": n_eligible, "n_tested": int(stats["n_tested"]),
            "unicity_pct": round(stats["unicity"] * 100, 1),
            "share_matched_by_lt5_pct": round(stats["share_lt5"] * 100, 1),
            "median_matching_maids": int(stats["median_matching"]),
        }
        rows.append(row)
        print(row)

    traces.unpersist()

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUTPUT_CSV, index=False)
print(f"Tersimpan: {OUTPUT_CSV} ({len(res)} baris)")
pings.unpersist()
