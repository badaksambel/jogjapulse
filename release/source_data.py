"""Source pings with default coordinates removed; shared by every script that builds a released product.

Default coordinates: exact (latitude, longitude) pairs reported by >= DEFAULT_MIN_DEVICES distinct
devices over the whole period. GPS fixes of different devices do not coincide to the last decimal; the
25 such points in the source data all have coarse coordinates (1-4 decimals) and are likely fallback
locations (IP, cell or geocoded centroids), e.g. a point in the Yogyakarta city centre (Gondomanan)
reported by about 47,000 devices. They are listed in data/default_points.csv (find_default_points.py).
"""
import os

import pandas as pd
from pyspark.sql import functions as F

SOURCE = "s3a://ilum-jogjapulse/mobility/gps_limited_user_month"
DEFAULT_MIN_DEVICES = 10_000
DEFAULT_POINTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "default_points.csv")


def check_session_time_zone(spark):
    """datetime is WIB wall-clock time stored without a zone; hour()/to_date() read it in the session
    time zone, so the session must be UTC (checked here, never changed)."""
    assert spark.conf.get("spark.sql.session.timeZone") in ("UTC", "Etc/UTC", "GMT"), \
        "Spark session time zone must be UTC"


def raw_pings(spark):
    return spark.read.parquet(SOURCE)


def load_pings(spark):
    """Source pings without the default coordinates listed in data/default_points.csv."""
    pts = pd.read_csv(DEFAULT_POINTS_CSV)[["latitude", "longitude"]]
    pts_sdf = spark.createDataFrame(pts)
    return raw_pings(spark).join(F.broadcast(pts_sdf), ["latitude", "longitude"], "left_anti")
