#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="frozen_liraglutide_results"
FISH_ROOT="constrained_fish_features_time_corrected_liraglutide"
INJURY_ROOT="manual_injury_feature_outputs_time_corrected_liraglutide"
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
    --condition "Liraglutide_treated"
}

run_application \
  "musc_model_a_plus_injury" \
  "$INJURY_ROOT/musc/constrained_fish_level_musc_model_a_plus_injury.csv"

run_application \
  "macrophage_all_model_b_plus_injury" \
  "$INJURY_ROOT/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv"

run_application \
  "macrophage_outside_boundary_model_b" \
  "$FISH_ROOT/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv"

if [ -f "frozen_untreated_models/musc_model_a_no_injury_all_fish/frozen_untreated_model.joblib" ]; then
  run_application \
    "musc_model_a_no_injury_all_fish" \
    "$FISH_ROOT/musc/constrained_fish_level_mean_median.csv"
fi

if [ -f "frozen_untreated_models/musc_model_a_no_injury_drop2/frozen_untreated_model.joblib" ]; then
  run_application \
    "musc_model_a_no_injury_drop2" \
    "$FISH_ROOT/musc/constrained_fish_level_mean_median.csv"
fi

"$PYTHON_BIN" -u 13_analyze_frozen_mmp_results.py \
  --results-root "$RESULTS_ROOT" \
  --frozen-root "frozen_untreated_models" \
  --output-dir "$RESULTS_ROOT/combined_analysis" \
  --treated-condition "Liraglutide_treated" \
  --treated-label "Liraglutide treated" \
  --output-prefix "liraglutide"

"$PYTHON_BIN" -u 14_audit_injury_roi_consistency.py \
  --treated-condition "Liraglutide_treated" \
  --treated-label "Liraglutide treated" \
  --treated-annotation-root "manual_injury_annotations_liraglutide" \
  --treated-metadata "Liraglutide_metadata.csv" \
  --output-dir "$RESULTS_ROOT/combined_analysis"

echo "[DONE] Frozen Liraglutide analysis saved under:"
echo "$SCRIPT_DIR/$RESULTS_ROOT"
