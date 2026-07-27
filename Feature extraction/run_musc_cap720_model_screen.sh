#!/bin/bash
set -euo pipefail

# Test whether long elapsed-time MUSC tracks drive poor genotype prediction.
# This pipeline caps each raw MUSC trajectory at 720 elapsed minutes, then
# rebuilds fish-level Model A/B/D tables and reruns the same classifier screen.
#
# Outputs are separated from existing analyses:
#   qc_outlier_outputs_time_corrected_cap720/musc
#   constrained_fish_features_time_corrected_cap720/...
#   constrained_separation_results_time_corrected_cap720/...

PERMUTATIONS="${PERMUTATIONS:-0}"

# Run this script from the Feature extraction directory.

echo "============================================================"
echo "1. Recalculate MUSC track features capped at 720 minutes"
echo "============================================================"

python3 -u 03c_recalculate_time_corrected_track_features_capped.py \
  --root final_feature_outputs \
  --dataset musc \
  --metadata-file block_metadata.csv \
  --qc-track-table qc_outlier_outputs/musc/cell_track_features_qc_filtered.csv \
  --output-dir qc_outlier_outputs_time_corrected_cap720/musc \
  --max-elapsed-minutes 720 \
  --min-track-points 2

CAP_TRACK_TABLE="qc_outlier_outputs_time_corrected_cap720/musc/cell_track_features_time_corrected_cap720.csv"

echo "============================================================"
echo "2. Build cap720 MUSC Model A fish features"
echo "============================================================"

python3 -u 04b_make_constrained_fish_features.py \
  --input "$CAP_TRACK_TABLE" \
  --output-dir constrained_fish_features_time_corrected_cap720/musc \
  --dataset-name "musc_model_a_time_corrected_cap720"

echo "============================================================"
echo "3. Build cap720 MUSC Model B fish features"
echo "============================================================"

python3 -u 04c_make_model_b_d_fish_features.py \
  --input "$CAP_TRACK_TABLE" \
  --model b \
  --output-dir constrained_fish_features_time_corrected_cap720/model_b/musc \
  --dataset-name "musc_model_b_time_corrected_cap720" \
  --minimum-new-features 2

echo "============================================================"
echo "4. Build cap720 MUSC Model D fish features"
echo "============================================================"

python3 -u 04c_make_model_b_d_fish_features.py \
  --input "$CAP_TRACK_TABLE" \
  --model d \
  --output-dir constrained_fish_features_time_corrected_cap720/model_d/musc \
  --dataset-name "musc_model_d_time_corrected_cap720" \
  --minimum-new-features 2

run_l1() {
  local model="$1"
  local input="$2"

  python3 -u 06_test_constrained_fish_separation_final.py \
    --input "$input" \
    --output-dir "constrained_separation_results_time_corrected_cap720/$model/musc_l1" \
    --dataset-name "musc_${model}_time_corrected_cap720_l1" \
    --model l1 \
    --permutations "$PERMUTATIONS"
}

run_svm() {
  local kernel="$1"
  local model="$2"
  local input="$3"

  python3 -u 06c_test_constrained_fish_separation_svm.py \
    --input "$input" \
    --output-dir "constrained_separation_results_time_corrected_cap720/svm_${kernel}/$model/musc" \
    --dataset-name "musc_${model}_time_corrected_cap720_svm_${kernel}" \
    --kernel "$kernel" \
    --permutations "$PERMUTATIONS"
}

run_extra() {
  local method="$1"
  local script="$2"
  local model="$3"
  local input="$4"

  python3 -u "$script" \
    --input "$input" \
    --output-dir "constrained_separation_results_time_corrected_cap720/$method/$model/musc" \
    --dataset-name "musc_${model}_time_corrected_cap720_${method}" \
    --permutations "$PERMUTATIONS"
}

run_all_models_for_input() {
  local model="$1"
  local input="$2"

  echo "============================================================"
  echo "5. Run cap720 classifiers for $model"
  echo "============================================================"

  run_l1 "$model" "$input"
  run_svm linear "$model" "$input"
  run_svm rbf "$model" "$input"

  run_extra \
    l2_logistic \
    06d_test_constrained_fish_separation_l2_logistic.py \
    "$model" \
    "$input"

  run_extra \
    lda_shrinkage \
    06e_test_constrained_fish_separation_lda_shrinkage.py \
    "$model" \
    "$input"

  run_extra \
    calibrated_linear_svm \
    06f_test_constrained_fish_separation_calibrated_linear_svm.py \
    "$model" \
    "$input"
}

run_all_models_for_input \
  model_a \
  constrained_fish_features_time_corrected_cap720/musc/constrained_fish_level_mean_median.csv

run_all_models_for_input \
  model_b \
  constrained_fish_features_time_corrected_cap720/model_b/musc/constrained_fish_level_mean_median.csv

run_all_models_for_input \
  model_d \
  constrained_fish_features_time_corrected_cap720/model_d/musc/constrained_fish_level_mean_median.csv

echo "============================================================"
echo "6. Summarize cap720 MUSC screen"
echo "============================================================"

python3 -u 06g_summarize_classifier_results.py \
  --root constrained_separation_results_time_corrected_cap720 \
  --output constrained_separation_results_time_corrected_cap720/musc_cap720_model_summary.csv

echo
echo "============================================================"
echo "Cap720 MUSC analysis completed successfully"
echo "============================================================"
echo "Summary:"
echo "constrained_separation_results_time_corrected_cap720/musc_cap720_model_summary.csv"