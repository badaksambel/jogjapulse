#!/usr/bin/env bash
# Run the JogjaPulse pipeline, validation and figures on the synthetic sample.
# Outputs go to sample_run/ (released-format files in sample_run/output/, figures in sample_run/figures/).
#
# Needs Java 17 (JAVA_HOME) and the two environments of the README:
#   .venv-spark  (Python 3.10, requirements-spark.txt)  Spark scripts
#   .venv        (Python 3.13, requirements.txt)        analysis and figure scripts
# Other interpreters can be given as SPARK_PY=... PY=... ./run_sample.sh
set -euo pipefail
cd "$(dirname "$0")"

SPARK_PY=${SPARK_PY:-.venv-spark/bin/python}
PY=${PY:-.venv/bin/python}
export MPLBACKEND=Agg

step() { echo "== $*"; "$@"; }

[[ -f sample/pings_synthetic.parquet ]] || step "$SPARK_PY" release/make_synthetic_sample.py

# Pipeline (Spark): source checks, products, internal OD counts, suppression statistics
step "$SPARK_PY" validation/source_data_checks.py
step "$SPARK_PY" release/home_location_full_period.py
step "$SPARK_PY" release/tier1_presence.py
(export TIER1_K=10; step "$SPARK_PY" release/tier1_presence.py)
step "$SPARK_PY" release/tier1_od.py
step "$SPARK_PY" release/od_feasibility.py
step "$SPARK_PY" release/presence_aggregate.py
step "$SPARK_PY" temporal/temporal_validasi.py
step "$SPARK_PY" temporal/temporal_date_hour.py
step "$SPARK_PY" privacy/unicity_test.py
step "$SPARK_PY" validation/device_intermittency_stats.py
(export WRITE_LEGACY_HOME=1; step "$SPARK_PY" release/home_location_legacy.py)
step "$SPARK_PY" release/check_home_location.py
step "$SPARK_PY" validation/coarse_sensitivity.py
step "$SPARK_PY" validation_od/od_hotspot_sensitivity.py

# Validation and figures (pandas): Figure 2-6 of the article, on the sample
step "$PY" release/validate_tier1.py
step "$PY" release/od_feasibility_analysis.py
step "$PY" temporal/plot_daily_coverage.py
step "$PY" validation/analyze_users_pop_kapanewon_dual_correlation.py
step "$PY" validation/static_choropleth_ratio_side_by_side.py
step "$PY" temporal/plot_hourly_dow.py
step "$PY" validation_od/plot_od_validation.py
step "$PY" validation_od/od_censoring_sensitivity.py
step "$PY" validation/coarse_sensitivity_analysis.py
step "$PY" validation_od/od_hotspot_sensitivity_analysis.py

echo "Done: outputs in sample_run/output/, figures in sample_run/figures/"
