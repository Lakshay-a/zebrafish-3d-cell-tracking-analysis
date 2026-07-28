#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_DIR="$ROOT/Feature extraction"
DATA_DIR="$ROOT/analysis_data"
OUTPUT_ROOT="${REPRODUCED_OUTPUT_DIR:-$ROOT/reproduced_analysis}"
RESULT_ROOT="$OUTPUT_ROOT/final_best_models_1000_perm"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PERMUTATIONS="${PERMUTATIONS:-2000}"
RANDOM_SEED="${RANDOM_SEED:-42}"

export PYTHONHASHSEED="$RANDOM_SEED"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_ROOT/_mpl_cache}"
mkdir -p "$MPLCONFIGDIR"

run_calibrated_svm() {
  local input="$1"
  local output="$2"
  local name="$3"
  "$PYTHON_BIN" -u "$FEATURE_DIR/06f_test_constrained_fish_separation_calibrated_linear_svm.py" \
    --input "$DATA_DIR/$input" \
    --output-dir "$RESULT_ROOT/$output" \
    --dataset-name "$name" \
    --random-seed "$RANDOM_SEED" \
    --permutations "$PERMUTATIONS"
}

run_calibrated_svm \
  "manual_injury_feature_outputs_time_corrected/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv" \
  "calibrated_linear_svm/macrophage_all_model_b_plus_injury" \
  "macrophage_all_model_b_plus_injury_calibrated_linear_svm_perm${PERMUTATIONS}"

run_calibrated_svm \
  "constrained_fish_features_time_corrected/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv" \
  "calibrated_linear_svm/macrophage_outside_boundary_model_b" \
  "macrophage_outside_boundary_model_b_calibrated_linear_svm_perm${PERMUTATIONS}"

"$PYTHON_BIN" -u "$FEATURE_DIR/06_test_constrained_fish_separation_final.py" \
  --input "$DATA_DIR/subset_sensitivity_results/musc_model_a_legacy_l1_top5_combo/_subset_inputs/drop_20240422_block06__20240422_block08.csv" \
  --output-dir "$RESULT_ROOT/legacy_l1/musc_model_a_drop_20240422_block06_20240422_block08" \
  --dataset-name "musc_model_a_l1_drop_20240422_block06_20240422_block08_perm${PERMUTATIONS}" \
  --random-seed "$RANDOM_SEED" \
  --permutations "$PERMUTATIONS"

run_calibrated_svm \
  "subset_sensitivity_results/musc_model_a_plus_injury_svm_linear_top4_combo/_subset_inputs/drop_260511_block07__260511_block05__260427_block02.csv" \
  "calibrated_linear_svm/musc_model_a_plus_injury_drop_260511_block07_260511_block05_260427_block02" \
  "musc_model_a_plus_injury_calibrated_linear_svm_drop_260511_block07_260511_block05_260427_block02_perm${PERMUTATIONS}"

"$PYTHON_BIN" -u "$FEATURE_DIR/06g_summarize_classifier_results.py" \
  --root "$RESULT_ROOT" \
  --output "$RESULT_ROOT/final_best_models_1000_perm_summary.csv"

echo "[DONE] Reproduced classifier tables: $RESULT_ROOT"

