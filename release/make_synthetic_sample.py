"""Synthetic source pings in the format of the private source data, for running the pipeline end to end.

The pings are random and say nothing about real mobility. They only reproduce the schema and the
features the pipeline relies on: devices with a home kapanewon (night pings near home), daytime visits
to other kapanewon, intermittent observation (few active days, bursts of pings) and the WIB/UTC time
convention of the source:
    maid       string     pseudonymous device id
    latitude   double     WGS 84
    longitude  double
    timestamp  double     Unix epoch seconds (UTC)
    datetime   timestamp  WIB wall-clock time (= timestamp + 7 h), read in a UTC Spark session
    month      int        YYYYMM of datetime

Usage (writes sample/pings_synthetic.parquet, about 0.9 million pings from 15,000 devices):
    .venv-spark/bin/python release/make_synthetic_sample.py
then run the pipeline on it with ./run_sample.sh (see README.md).
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely import wkt

SEED = 20211201
N_DEVICES = 15000
PERIOD = pd.date_range("2021-12-01", "2022-04-30", freq="D")
OUT = os.path.join("..", "sample", "pings_synthetic.parquet")  # sample/ in the repository root
DEG_PER_M = 1 / 111_000

rng = np.random.default_rng(SEED)
kap = pd.read_csv("data/kapanewon_wkt.csv")
pop = pd.read_csv("data/bps_population_kapanewon_2021.csv")
kap = kap.merge(pop, on=["KAB_KOTA", "KEC_KAPN"])
geoms = [wkt.loads(w) for w in kap["wkt"]]
cent = np.array([[g.centroid.x, g.centroid.y] for g in geoms])


def random_points(i, n):
    """n uniform random points inside kapanewon i (rejection sampling in its bounding box)."""
    g = geoms[i]
    x0, y0, x1, y1 = g.bounds
    out = np.empty((0, 2))
    while len(out) < n:
        xy = np.column_stack([rng.uniform(x0, x1, 4 * n), rng.uniform(y0, y1, 4 * n)])
        out = np.vstack([out, xy[shapely.contains_xy(g, xy[:, 0], xy[:, 1])]])
    return out[:n]


# Home kapanewon: more devices in populous (urban) kapanewon, as in the real panel
w = kap["pop_actual"].to_numpy(float) ** 1.3
home_k = rng.choice(len(kap), N_DEVICES, p=w / w.sum())
home_xy = np.vstack([random_points(k, 1) for k in home_k])

# One or two regular daytime destinations per device, chosen by a gravity rule (population / distance^2)
dist = np.hypot(*(cent[:, None, :] - cent[None, :, :]).transpose(2, 0, 1)) / DEG_PER_M / 1000 + 1.0
attract = kap["pop_actual"].to_numpy(float)[None, :] / dist ** 2
dest = [rng.choice(len(kap), rng.integers(1, 3), replace=False, p=attract[k] / attract[k].sum()) for k in home_k]

# Hour-of-day profile of pings (WIB): low at night, plateau 11-20 h
hour_w = np.array([2, 1.5, 1, 1, 1.5, 3, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 7, 5, 3], float)
hour_p = hour_w / hour_w.sum()

rows = []
for d in range(N_DEVICES):
    n_days = int(np.clip(rng.geometric(1 / 10), 1, len(PERIOD)))  # intermittent: median ~7 active days
    days = rng.choice(PERIOD, n_days, replace=False)
    for day in days:
        n = 1 + rng.poisson(5)
        hours = rng.choice(24, n, p=hour_p)
        secs = hours * 3600 + rng.integers(0, 3600, n)
        at_home = (hours >= 22) | (hours < 6) | (rng.random(n) < 0.5)
        k_out = rng.choice(dest[d], n)
        xy = np.where(at_home[:, None], home_xy[d] + rng.normal(0, 30 * DEG_PER_M, (n, 2)),
                      np.vstack([random_points(k, 1) for k in k_out]))
        for (x, y), s in zip(xy, secs):
            rows.append((d, y, x, pd.Timestamp(day) + pd.Timedelta(seconds=int(s))))

df = pd.DataFrame(rows, columns=["dev", "latitude", "longitude", "wib"])
df["maid"] = [f"synthetic-{i:05d}" for i in df["dev"]]
df["latitude"], df["longitude"] = df["latitude"].round(6), df["longitude"].round(6)
# datetime: WIB wall clock stored as a UTC instant (so a UTC session shows WIB); timestamp: true epoch
df["datetime"] = df["wib"].dt.tz_localize("UTC")
df["timestamp"] = (df["datetime"] - pd.Timedelta(hours=7)).astype("int64") / 1e9
df["month"] = (df["wib"].dt.year * 100 + df["wib"].dt.month).astype("int32")
df = df.sort_values(["maid", "datetime"])[["maid", "latitude", "longitude", "timestamp", "datetime", "month"]]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
schema = pa.schema([("maid", pa.string()), ("latitude", pa.float64()), ("longitude", pa.float64()),
                    ("timestamp", pa.float64()), ("datetime", pa.timestamp("us", tz="UTC")), ("month", pa.int32())])
pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), OUT)
print(f"Saved {OUT}: {len(df):,} pings from {df['maid'].nunique():,} synthetic devices, "
      f"{df['datetime'].min():%Y-%m-%d} to {df['datetime'].max():%Y-%m-%d}")
