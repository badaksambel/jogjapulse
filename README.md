# JogjaPulse: processing code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23244140.svg)](https://doi.org/10.5281/zenodo.23244140)

Code used to build and validate aggregated mobility products for the Special Region of
Yogyakarta (DIY), Indonesia, from device-level GPS pings (December 2021 to April 2022).
All scripts are in [`release/`](release/).

## Products

| Product | Spatial unit | Time unit | Script |
|---|---|---|---|
| A. Presence | 500 m grid cell (UTM 49S) | day | `tier1_presence.py` |
| B. Presence | kapanewon (78) | hour | `tier1_presence.py` |
| C. Home-based OD flows | kapanewon to kapanewon | month | `tier1_od.py` |

Counts are numbers of distinct devices (MAIDs). Rows with fewer than k = 5 devices are
suppressed. Ping counts are published only in product B: they are additive, so publishing
them for grid cells as well would allow suppressed cells to be recovered by subtraction.

## Scripts

Run in this order (after `find_default_points.py`, whose output `data/default_points.csv` is included):

1. `make_kapanewon_wkt.py`: dissolves the 438 kalurahan of Geoportal DIY (2023) into
   78 kapanewon polygons and writes them as WKT (`data/kapanewon_wkt.csv`).
2. `tier1_presence.py`: assigns pings to kapanewon and grid cells, aggregates products A and
   B, applies suppression and writes a suppression summary. Set `TIER1_K` to use another k.
3. `home_location_full_period.py`: assigns each device one home kapanewon over the whole
   period (same rule as the monthly OD home, below) and writes
   `data/validasi_kapanewon_des2021_apr2022.csv` (home-based devices vs BPS population).
   `home_location_legacy.py` reproduces the earlier definition (20:00-07:59, ~11 m bins,
   no minimum nights) for comparison only.
   `validate_tier1.py`: compares night-time presence (22:00-05:59 WIB) per kapanewon with
   BPS population and with home-based device counts (Pearson correlation, linear and log).
4. `tier1_od.py`: assigns each device a monthly home and builds home-based OD flows
   (product C), with origin denominators.
5. `od_feasibility.py`: computes unsuppressed internal counts used to assess the OD product
   (homes at 1-3 minimum nights, coverage, monthly and weekly OD). These outputs are for
   analysis only and are not released.
6. `od_feasibility_analysis.py`: compares monthly home counts with BPS population, measures
   OD suppression at k = 5, and fits a gravity model
   (log T_ij = a + b log H_i + c log P_j - gamma log d_ij) with a symmetry check.

## Definitions

- **Time:** all dates and hours are local time (WIB, UTC+7), taken from the `datetime`
  column of the source data (see Data).
- **Home:** for each device, the kapanewon with the most distinct nights observed between
  22:00 and 05:59. Ties are broken by number of night pings. Pings before 06:00 belong to
  the previous night. A home requires at least 2 nights. The rule is applied per month for
  the OD product and once over the whole period for validation against BPS population.
- **Destination:** a kapanewon where the device is observed between 08:00 and 17:59 in the
  same month.
- **OD flow:** distinct devices with home i observed in j during the month (i = j included).
- **Reduced quality:** March and April 2022 are flagged because the device panel is small
  and the gravity-model fit is weak.

## Data

Included in [`release/data/`](release/data/):

- `diy_kalurahan_2023/`: kalurahan boundaries (shapefile, WGS84) from Geoportal DIY, 2023.
- `kapanewon_wkt.csv`: the 78 kapanewon polygons derived from them.
- `validasi_kapanewon_des2021_apr2022.csv`: per kapanewon, the number of devices with a
  full-period home there (`n_users`, 237,719 devices in total), BPS population
  (`pop_actual`), their ratio and the post-stratification weight `pop_actual / n_users`.

Not included: the device-level GPS pings (Parquet, 44,384,826 rows, 594,223 devices). They
cannot be redistributed because of privacy and licensing restrictions. Columns:

| Column | Type | Description |
|---|---|---|
| `maid` | string | Mobile advertising identifier of the device. |
| `latitude`, `longitude` | double | WGS84 coordinates. No horizontal-accuracy field is provided. |
| `timestamp` | double | Unix epoch seconds (UTC), whole seconds. |
| `datetime` | timestamp | Local wall-clock time (WIB), stored without a time zone. Equal to `timestamp` + 7 h in every row. |
| `month` | integer | `YYYYMM` of `datetime`. |

Time zone: the scripts take dates and hours from `datetime`. Spark reads a zone-less timestamp in
the session time zone, so the session must be UTC for `hour()` and `to_date()` to return WIB.
Every Spark script checks this at start-up and stops otherwise; it never changes the setting.
`validation/source_data_checks.py` verifies the schema, the `timestamp` + 7 h relation,
duplicates and coordinate quality.

Default coordinates: 25 exact coordinate pairs reported by at least 10,000 distinct devices
(all with 1-4 decimals, likely fallback locations rather than GPS fixes; 6,388,904 pings, 14.4%)
are removed before any aggregation. `find_default_points.py` lists them in
`data/default_points.csv`, and `source_data.py` (`load_pings`) reads the source pings without
them; every script below reads pings through this function. What remains: 37,995,922 pings of
559,027 devices.
Scripts 2, 4 and 5 read these data and cannot be run without them. Script 1 runs with the
included data alone. Script 3 also needs the output of script 2, and script 6 the
unsuppressed output of script 5.

## Requirements

- Apache Spark 3.5.7 with Apache Sedona 1.8 (scripts 2, 4 and 5), connected through Spark
  Connect. The endpoint is read from `SPARK_REMOTE` (default `sc://localhost:15002`).
  The Spark session time zone must be UTC so that WIB timestamps are not shifted (checked by
  each script).
- Python 3.10 with pandas, numpy, scipy, statsmodels, geopandas and pyspark 3.5.7.

Scripts resolve paths relative to `release/`. Outputs are written to `release/output/`
(released products) and `release/od_work/` (internal analysis).

## Citation

Please cite the archived release: https://doi.org/10.5281/zenodo.23244140

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).
