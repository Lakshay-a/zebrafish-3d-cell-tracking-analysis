#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="frozen_mmp_results_excluding_20240604_block02_20240702_block06"
EXCLUDED_FISH=("20240604_block02" "20240702_block06")
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/$RESULTS_ROOT/_mpl_cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$MPLCONFIGDIR"

run_application() {
  local name="$1"
  local treated_input="$2"

  "$PYTHON_BIN" -u 12_apply_frozen_untreated_model.py \
    --model "frozen_untreated_models/$name/frozen_untreated_model.joblib" \
    --treated-input "$treated_input" \
    --output-dir "$RESULTS_ROOT/$name" \
    --condition "MMP9_inhibited" \
    --exclude-fish "${EXCLUDED_FISH[@]}"
}

run_application \
  "musc_model_a_plus_injury" \
  "manual_injury_feature_outputs_time_corrected_mmp/musc/constrained_fish_level_musc_model_a_plus_injury.csv"

run_application \
  "macrophage_all_model_b_plus_injury" \
  "manual_injury_feature_outputs_time_corrected_mmp/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv"

run_application \
  "macrophage_outside_boundary_model_b" \
  "constrained_fish_features_time_corrected_mmp/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv"

run_application \
  "musc_model_a_no_injury_all_fish" \
  "constrained_fish_features_time_corrected_mmp/musc/constrained_fish_level_mean_median.csv"

run_application \
  "musc_model_a_no_injury_drop2" \
  "constrained_fish_features_time_corrected_mmp/musc/constrained_fish_level_mean_median.csv"

"$PYTHON_BIN" -u 13_analyze_frozen_mmp_results.py \
  --results-root "$RESULTS_ROOT" \
  --frozen-root "frozen_untreated_models" \
  --output-dir "$RESULTS_ROOT/combined_analysis"

echo "[DONE] Exclusion sensitivity analysis saved under:"
echo "$SCRIPT_DIR/$RESULTS_ROOT"
