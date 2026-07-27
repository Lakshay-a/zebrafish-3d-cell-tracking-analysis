#!/bin/bash
set -euo pipefail

# Run Model B and Model D feature construction and L1 nested-LOFO analyses
# for MUSC, macrophage-all, and macrophage outside-boundary.
#
# Usage:
#   chmod +x run_model_b_d_analyses.sh
#   ./run_model_b_d_analyses.sh
#
# Optional:
#   PERMUTATIONS=100 ./run_model_b_d_analyses.sh
#   PERMUTATIONS=2000 ./run_model_b_d_analyses.sh
#
# The default is 500 permutations.

PERMUTATIONS="${PERMUTATIONS:-0}"
MODEL_SCRIPT="04c_make_model_b_d_fish_features.py"
CLASSIFIER_SCRIPT="06_test_constrained_fish_separation_final.py"

run_one() {
    local model_letter="$1"
    local dataset_name="$2"
    local input_csv="$3"

    local feature_dir="constrained_fish_features/model_${model_letter}/${dataset_name}"
    local result_dir="constrained_separation_results/model_${model_letter}/${dataset_name}_l1"

    echo
    echo "============================================================"
    echo "MODEL $(printf '%s' "$model_letter" | tr '[:lower:]' '[:upper:]'): ${dataset_name}"
    echo "============================================================"

    python3 -u "$MODEL_SCRIPT" \
      --input "$input_csv" \
      --model "$model_letter" \
      --output-dir "$feature_dir" \
      --dataset-name "${dataset_name}_model_${model_letter}"

    python3 -u "$CLASSIFIER_SCRIPT" \
      --input "$feature_dir/constrained_fish_level_mean_median.csv" \
      --output-dir "$result_dir" \
      --dataset-name "${dataset_name}_model_${model_letter}_l1" \
      --model l1 \
      --permutations "$PERMUTATIONS"
}

run_one "b" "musc" \
  "qc_outlier_outputs/musc/cell_track_features_qc_filtered.csv"

run_one "b" "macrophage_all" \
  "qc_outlier_outputs/macrophage_all/cell_track_features_qc_filtered.csv"

run_one "b" "macrophage_outside_boundary" \
  "qc_outlier_outputs/macrophage_outside_boundary/cell_track_features_qc_filtered.csv"

run_one "d" "musc" \
  "qc_outlier_outputs/musc/cell_track_features_qc_filtered.csv"

run_one "d" "macrophage_all" \
  "qc_outlier_outputs/macrophage_all/cell_track_features_qc_filtered.csv"

run_one "d" "macrophage_outside_boundary" \
  "qc_outlier_outputs/macrophage_outside_boundary/cell_track_features_qc_filtered.csv"

echo
echo "============================================================"
echo "ALL MODEL B AND MODEL D L1 ANALYSES COMPLETED"
echo "Finished: $(date)"
echo "============================================================"
