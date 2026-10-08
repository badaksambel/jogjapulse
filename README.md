# JogjaPulse: processing code

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

Run in this order:

1. `make_kapanewon_wkt.py`: dissolves the 438 kalurahan of Geoportal DIY (2023) into
   78 kapanewon polygons and writes them as WKT (`data/kapanewon_wkt.csv`).
2. `tier1_presence.py`: assigns pings to kapanewon and grid cells, aggregates products A and
   B, applies suppression and writes a suppression summary. Set `TIER1_K` to use another k.
3. `validate_tier1.py`: compares night-time presence (00:00-05:59 WIB) per kapanewon with
   BPS population and with home-based device counts (Pearson correlation on log values).
4. `tier1_od.py`: assigns each device a monthly home and builds home-based OD flows
   (product C), with origin denominators.
5. `od_feasibility.py`: computes unsuppressed internal counts used to assess the OD product
   (homes at 1-3 minimum nights, coverage, monthly and weekly OD). These outputs are for
   analysis only and are not released.
6. `od_feasibility_analysis.py`: compares monthly home counts with BPS population, measures
   OD suppression at k = 5, and fits a gravity model
   (log T_ij = a + b log H_i + c log P_j - gamma log d_ij) with a symmetry check.

## Definitions

- **Time:** timestamps are local time (WIB, UTC+7).
- **Home:** for each device and month, the kapanewon with the most distinct nights observed
  between 22:00 and 05:59. Ties are broken by number of night pings. Pings before 06:00
  belong to the previous night. A home requires at least 2 nights.
- **Destination:** a kapanewon where the device is observed between 08:00 and 17:59 in the
  same month.
- **OD flow:** distinct devices with home i observed in j during the month (i = j included).
- **Reduced quality:** March and April 2022 are flagged because the device panel is small
  and the gravity-model fit is weak.

## Data

Included in [`release/data/`](release/data/):

- `diy_kalurahan_2023/`: kalurahan boundaries (shapefile, WGS84) from Geoportal DIY, 2023.
- `kapanewon_wkt.csv`: the 78 kapanewon polygons derived from them.
- `validasi_kapanewon_des2021_apr2022.csv`: per kapanewon, the number of home-based devices
  (`n_users`), BPS population (`pop_actual`) and their ratio.

Not included: the device-level GPS pings (Parquet, columns `maid`, `latitude`, `longitude`,
`datetime`). They cannot be redistributed because of privacy and licensing restrictions.
Scripts 2, 4 and 5 read these data and cannot be run without them. Script 1 runs with the
included data alone. Script 3 also needs the output of script 2, and script 6 the
unsuppressed output of script 5.

## Requirements

- Apache Spark 3.5.7 with Apache Sedona 1.8 (scripts 2, 4 and 5), connected through Spark
  Connect. The endpoint is read from `SPARK_REMOTE` (default `sc://localhost:15002`).
  The Spark session time zone must be UTC so that WIB timestamps are not shifted.
- Python 3.10 with pandas, numpy, scipy, statsmodels, geopandas and pyspark 3.5.7.

Scripts resolve paths relative to `release/`. Outputs are written to `release/output/`
(released products) and `release/od_work/` (internal analysis).

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).
