import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from source_data import load_pings

# Tunnel address; override with SPARK_REMOTE=sc://localhost:15003 etc.
spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()
# datetime is WIB wall-clock time stored without a zone; hour()/to_date() read it in the session time zone,
# so the session must be UTC (checked, never changed here)
assert spark.conf.get("spark.sql.session.timeZone") in ("UTC", "Etc/UTC", "GMT"), "Spark session time zone must be UTC"

OUTPUT_CSV = "presence_suppression_stats.csv"
OUTPUT_MONTH_CSV = "presence_suppression_by_month.csv"

# Tier-1 open data candidate: presence counts per grid cell and time bin, with small-count suppression.
# Grid in UTM zone 49S (EPSG:32749) so cells are true squares in metres.
CONFIGS = [  # (label, cell size m, time resolution)
    ("500 m x hour", 500, "hour"),
    ("500 m x day", 500, "day"),
    ("1 km x hour", 1000, "hour"),
]
K_VALUES = [5, 10]

pings = (
    load_pings(spark)  # source pings without default coordinates (source_data.py)
    .withColumn("utm", F.expr("ST_Transform(ST_Point(longitude, latitude), 'EPSG:4326', 'EPSG:32749')"))
    .select(
        "maid",
        F.to_date("datetime").alias("date"),  # datetime = WIB wall clock, session tz UTC
        F.hour("datetime").alias("hour"),
        F.expr("ST_X(utm)").alias("e"),
        F.expr("ST_Y(utm)").alias("n"),
    )
    .cache()
)
print(f"pings: {pings.count():,}")

rows, month_rows = [], []
for label, cell_m, time_res in CONFIGS:
    keys = ["cell_e", "cell_n", "date"] + (["hour"] if time_res == "hour" else [])
    agg = (
        pings
        .withColumn("cell_e", F.floor(F.col("e") / cell_m).cast("int"))
        .withColumn("cell_n", F.floor(F.col("n") / cell_m).cast("int"))
        .groupBy(*keys)
        .agg(F.countDistinct("maid").alias("n_maids"), F.count("*").alias("n_pings"))
        .withColumn("month", F.date_format("date", "yyyy-MM"))
        .cache()
    )

    for k in K_VALUES:
        kept = F.col("n_maids") >= k
        s = agg.agg(
            F.count("*").alias("rows"),
            F.sum(kept.cast("int")).alias("rows_kept"),
            F.sum("n_maids").alias("maid_presence"),
            F.sum(F.when(kept, F.col("n_maids")).otherwise(0)).alias("maid_presence_kept"),
            F.sum("n_pings").alias("pings"),
            F.sum(F.when(kept, F.col("n_pings")).otherwise(0)).alias("pings_kept"),
        ).toPandas().iloc[0]
        cells = agg.groupBy("cell_e", "cell_n").agg(F.max("n_maids").alias("max_maids"))
        c = cells.agg(F.count("*").alias("cells"),
                      F.sum((F.col("max_maids") >= k).cast("int")).alias("cells_ever_kept")).toPandas().iloc[0]
        row = {
            "config": label, "k": k,
            "rows_total": int(s["rows"]),
            "rows_suppressed_pct": round((1 - s["rows_kept"] / s["rows"]) * 100, 1),
            "maid_presence_suppressed_pct": round((1 - s["maid_presence_kept"] / s["maid_presence"]) * 100, 1),
            "pings_suppressed_pct": round((1 - s["pings_kept"] / s["pings"]) * 100, 1),
            "cells_total": int(c["cells"]),
            "cells_never_published_pct": round((1 - c["cells_ever_kept"] / c["cells"]) * 100, 1),
        }
        rows.append(row)
        print(row)

        by_month = (
            agg.groupBy("month")
            .agg(F.count("*").alias("rows"),
                 F.avg((~kept).cast("double")).alias("rows_suppressed"),
                 (1 - F.sum(F.when(kept, F.col("n_maids")).otherwise(0)) / F.sum("n_maids"))
                 .alias("maid_presence_suppressed"))
            .orderBy("month").toPandas()
        )
        for _, m in by_month.iterrows():
            month_rows.append({
                "config": label, "k": k, "month": m["month"], "rows": int(m["rows"]),
                "rows_suppressed_pct": round(m["rows_suppressed"] * 100, 1),
                "maid_presence_suppressed_pct": round(m["maid_presence_suppressed"] * 100, 1),
            })

    agg.unpersist()

res = pd.DataFrame(rows)
res_month = pd.DataFrame(month_rows)
print(res.to_string(index=False))
print(res_month.to_string(index=False))
res.to_csv(OUTPUT_CSV, index=False)
res_month.to_csv(OUTPUT_MONTH_CSV, index=False)
print(f"Tersimpan: {OUTPUT_CSV}, {OUTPUT_MONTH_CSV}")
pings.unpersist()
