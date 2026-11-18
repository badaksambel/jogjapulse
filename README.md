# JogjaPulse: processing code and synthetic data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23244139.svg)](https://doi.org/10.5281/zenodo.23244139)

Processing, validation and figure code of the JogjaPulse dataset (aggregated mobility for the Special
Region of Yogyakarta, Indonesia, December 2021 to April 2022). The device-level source data cannot be
shared, so the repository includes synthetic pings on which the whole pipeline runs on a laptop.

- Dataset (version 1.1): https://doi.org/10.5281/zenodo.24333942
- Code, all versions: https://doi.org/10.5281/zenodo.23244139

By default the scripts use a local Spark session, read the synthetic data and write to `sample_run/`.
The same scripts produced the dataset from the source data (see "Running on the source data").

**The synthetic pings are random.** They reproduce the schema, the time convention and the general
structure of the source data (devices with a home, daytime visits, intermittent observation), not
real mobility, so the numbers and figures produced here are not those of the article.

## Requirements

| Component | Version |
|---|---|
| Java | 17 (set `JAVA_HOME`) |
| Apache Spark (PySpark) | 3.5.7 |
| Apache Sedona | 1.8.0, jars fetched from Maven Central on first run |
| Python for Spark scripts | 3.10, [`requirements-spark.txt`](requirements-spark.txt) |
| Python for analysis and figures | 3.13, [`requirements.txt`](requirements.txt) |

```bash
python3.10 -m venv .venv-spark && .venv-spark/bin/pip install -r requirements-spark.txt
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
./run_sample.sh
```

This runs every step in order on the included synthetic data and takes a few minutes. Results:

| Folder | Content |
|---|---|
| `sample_run/output/` | the released files of the dataset, computed from the sample |
| `sample_run/output_k10/` | presence products with k = 10, for the choice of k |
| `sample_run/od_work/` | internal unsuppressed OD counts used for the OD validation (never released for real data) |
| `sample_run/figures/` | Figures 2-6 of the article and their statistics |
| `sample_run/sensitivity/` | sensitivity of homes and OD flows to the default-coordinate threshold |
| `sample_run/intermittency_*.csv` | observation intermittency statistics |

## Running on the source data

The dataset was produced with Spark Connect on a Kubernetes cluster (Spark 3.5.7, Sedona 1.8), with
the same scripts and these settings:

```bash
export JOGJAPULSE_SPARK=connect SPARK_REMOTE=sc://<host>:15002
export JOGJAPULSE_SOURCE=<parquet path of the source pings> JOGJAPULSE_OUT=<output folder>
```

## Synthetic data

The synthetic pings are included in `sample/pings_synthetic.parquet` (15,000 devices, 890,496 pings,
1 December 2021 to 30 April 2022). `release/make_synthetic_sample.py` regenerates exactly this file
(fixed seed). Same columns and types as the source data:

| Column | Type | Description |
|---|---|---|
| `maid` | string | Pseudonymous device identifier. |
| `latitude`, `longitude` | double | WGS84 coordinates. |
| `timestamp` | double | Unix epoch seconds (UTC). |
| `datetime` | timestamp | Local wall-clock time (WIB, UTC+7), read in a UTC Spark session; equal to `timestamp` + 7 h. |
| `month` | integer | `YYYYMM` of `datetime`. |

**Time zone.** The scripts take dates and hours from `datetime` with Spark's `hour()` and
`to_date()`, which read it in the session time zone. The session is created in UTC and every Spark
script checks this (`release/source_data.py`), so the hours are WIB.

## Settings

All optional, as environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `JOGJAPULSE_SPARK` | `local` (local Spark with Sedona) or `connect` (Spark Connect at `SPARK_REMOTE`) | `local` |
| `JOGJAPULSE_SOURCE` | Parquet path of the input pings | `sample/pings_synthetic.parquet` |
| `JOGJAPULSE_OUT` | Root folder for outputs | `sample_run/` |
| `JOGJAPULSE_FIG` | Folder for figures | `sample_run/figures/` |
| `JOGJAPULSE_LEGACY_HOME` | Parquet path of the dataset v1.0 home table | `sample_run/legacy_home.parquet` |
| `SPARK_DRIVER_MEMORY` | Driver memory of the local session | `4g` |
| `SPARK_PY`, `PY` | Interpreters used by `run_sample.sh` | `.venv-spark/bin/python`, `.venv/bin/python` |

## Scripts

| Step | Script | Article |
|---|---|---|
| Synthetic input | `release/make_synthetic_sample.py` | - |
| Source checks | `validation/source_data_checks.py` | Methods (schema, time zone) |
| Default coordinates | `release/find_default_points.py` (lists them; the 25 points removed from the source data are in `release/data/default_points.csv`) | Methods |
| Kapanewon polygons | `release/make_kapanewon_wkt.py` (output included in `release/data/`) | Methods |
| Home location | `release/home_location_full_period.py` | Methods, Figure 3, 4 |
| Presence products A, B | `release/tier1_presence.py`, `release/presence_aggregate.py` | Data Records, Privacy |
| OD product C | `release/tier1_od.py` | Data Records |
| OD validation counts | `release/od_feasibility.py`, `release/od_feasibility_analysis.py` | Technical Validation |
| Night presence validation | `release/validate_tier1.py` | Technical Validation |
| Panel and temporal patterns | `temporal/temporal_validasi.py`, `temporal/temporal_date_hour.py`, `temporal/plot_daily_coverage.py`, `temporal/plot_hourly_dow.py` | Figure 2, 5 |
| Home vs population | `validation/analyze_users_pop_kapanewon_dual_correlation.py`, `validation/static_choropleth_ratio_side_by_side.py` | Figure 3, 4 |
| Gravity, reciprocity, censoring | `validation_od/plot_od_validation.py`, `validation_od/od_censoring_sensitivity.py` | Figure 6, Usage Notes |
| Uniqueness test | `privacy/unicity_test.py` | Privacy |
| Observation intermittency | `validation/device_intermittency_stats.py` | Background |
| Earlier home definition (20:00-07:59, dataset v1.0) | `release/home_location_legacy.py`, `release/check_home_location.py` | Robustness check |
| Default-coordinate sensitivity | `validation/coarse_sensitivity.py`, `validation/coarse_sensitivity_analysis.py`, `validation_od/od_hotspot_sensitivity.py`, `validation_od/od_hotspot_sensitivity_analysis.py` | Dataset README, limitation 9 |

Input data in `release/data/`: kalurahan boundaries from Geoportal DIY (2023), the 78 kapanewon
polygons derived from them, BPS population per kapanewon, the list of removed default coordinates,
and the home-vs-population table of dataset v1.0. `validation_od/od_censoring_sensitivity.csv` holds
the censoring statistics obtained on the source data.

## Citation

Please cite the archived code (https://doi.org/10.5281/zenodo.23244139) and the dataset
(https://doi.org/10.5281/zenodo.24333942).

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).
