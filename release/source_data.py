"""Spark session, source pings and output paths shared by every script that reads the source data.

This copy of the JogjaPulse processing code is set up for the synthetic sample: by default it runs a
local Spark session on sample/pings_synthetic.parquet (make_synthetic_sample.py) and writes to
sample_run/. The processing itself is that of the article; only these defaults differ.

Environment variables (all optional):
  JOGJAPULSE_SPARK   "local" (default): a local Spark 3.5.7 session (local[*]) with Sedona 1.8 from
                     Maven Central; needs Java 17 and requirements-spark.txt.
                     "connect": Spark Connect at SPARK_REMOTE (default sc://localhost:15002), the setup
                     used for the article (Spark 3.5.7 with Sedona 1.8 on Kubernetes).
  JOGJAPULSE_SOURCE  Parquet path of the source pings (default: the synthetic sample).
  JOGJAPULSE_OUT     Root folder for everything the scripts write (default sample_run/, see paths.py).
  JOGJAPULSE_LEGACY_HOME  Parquet path of the legacy home table (default under JOGJAPULSE_OUT).

Default coordinates: exact (latitude, longitude) pairs reported by >= DEFAULT_MIN_DEVICES distinct
devices over the whole period. GPS fixes of different devices do not coincide to the last decimal; the
25 such points in the source data all have coarse coordinates (1-4 decimals) and are likely fallback
locations (IP, cell or geocoded centroids), e.g. a point in the Yogyakarta city centre (Gondomanan)
reported by about 47,000 devices. They are listed in data/default_points.csv (find_default_points.py).
"""
import os
import sys
import time

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from paths import OUT_ROOT, out_dir, out_path  # noqa: F401  (re-exported for the Spark scripts)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SYNTHETIC_SAMPLE = os.path.join(ROOT, "sample", "pings_synthetic.parquet")
SOURCE = os.environ.get("JOGJAPULSE_SOURCE", SYNTHETIC_SAMPLE)
# Dataset v1.0 home per device (20:00-07:59 mode of 4-decimal bins), written by home_location_legacy.py
LEGACY_HOME = os.environ.get("JOGJAPULSE_LEGACY_HOME", os.path.join(OUT_ROOT, "legacy_home.parquet"))
DEFAULT_MIN_DEVICES = 10_000
DEFAULT_POINTS_CSV = os.path.join(HERE, "data", "default_points.csv")

SPARK_VERSION = "3.5.7"
SEDONA_PACKAGES = ("org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.8.0,"
                   "org.datasyslab:geotools-wrapper:1.8.0-33.1")


def get_spark(app="jogjapulse"):
    """Spark session (Connect or local, see module docstring); the time zone is checked before returning."""
    if os.environ.get("JOGJAPULSE_SPARK", "local") == "local":
        # Classic PySpark converts collected timestamps to the Python process time zone, and the JVM
        # defaults to the machine's zone; run both in UTC, as the cluster does.
        os.environ["TZ"] = "UTC"
        time.tzset()
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # Python workers = this interpreter
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        spark = (SparkSession.builder.master("local[*]").appName(app)
                 .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
                 .config("spark.jars.packages", SEDONA_PACKAGES)
                 .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions")
                 .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                 .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
                 .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
                 # A new local session would otherwise take the laptop's zone; the cluster session is UTC.
                 .config("spark.sql.session.timeZone", "UTC")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")
    else:
        spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://localhost:15002")).getOrCreate()
    check_session_time_zone(spark)
    assert spark.version.startswith(SPARK_VERSION), f"Spark {SPARK_VERSION} expected, got {spark.version}"
    return spark


def check_session_time_zone(spark):
    """datetime is WIB wall-clock time stored as a UTC timestamp; hour()/to_date() read it in the session
    time zone, so the session must be UTC (checked here, never changed on an existing session)."""
    assert spark.conf.get("spark.sql.session.timeZone") in ("UTC", "Etc/UTC", "GMT"), \
        "Spark session time zone must be UTC"


def raw_pings(spark):
    return spark.read.parquet(SOURCE)


def load_pings(spark):
    """Source pings without the default coordinates listed in data/default_points.csv."""
    pts = pd.read_csv(DEFAULT_POINTS_CSV)[["latitude", "longitude"]]
    pts_sdf = spark.createDataFrame(pts)
    return raw_pings(spark).join(F.broadcast(pts_sdf), ["latitude", "longitude"], "left_anti")
