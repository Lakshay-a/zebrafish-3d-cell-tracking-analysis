#!/bin/bash
set -e

ROOT="final_best_models_1000_perm"
mkdir -p "$ROOT"
export MPLCONFIGDIR="$ROOT/_mpl_cache"

python3 06f_test_constrained_fish_separation_calibrated_linear_svm.py \
  --input manual_injury_feature_outputs_time_corrected/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv \
  --output-dir "$ROOT/calibrated_linear_svm/macrophage_all_model_b_plus_injury" \
  --dataset-name "macrophage_all_model_b_plus_injury_calibrated_linear_svm_perm1000" \
  --permutations 2000

python3 06f_test_constrained_fish_separation_calibrated_linear_svm.py \
  --input constrained_fish_features_time_corrected/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv \
  --output-dir "$ROOT/calibrated_linear_svm/macrophage_outside_boundary_model_b" \
  --dataset-name "macrophage_outside_boundary_model_b_calibrated_linear_svm_perm1000" \
  --permutations 2000

# python3 06c_test_constrained_fish_separation_svm.py \
# --input.
#   --output-dir "$ROOT/svm_rbf/musc_model_b" \
#   --dataset-name "musc_model_b_svm_rbf_perm1000" \
#   --kernel rbf \
#   --permutations 1000

python3 06_test_constrained_fish_separation_final.py \
  --input subset_sensitivity_results/musc_model_a_legacy_l1_top5_combo/_subset_inputs/drop_20240422_block06__20240422_block08.csv \
  --output-dir "$ROOT/legacy_l1/musc_model_a_drop_20240422_block06_20240422_block08" \
  --dataset-name "musc_model_a_l1_drop_20240422_block06_20240422_block08_perm1000" \
  --permutations 2000

python3 06f_test_constrained_fish_separation_calibrated_linear_svm.py \
  --input subset_sensitivity_results/musc_model_a_plus_injury_svm_linear_top4_combo/_subset_inputs/drop_260511_block07__260511_block05__260427_block02.csv \
  --output-dir "$ROOT/calibrated_linear_svm/musc_model_a_plus_injury_drop_260511_block07_260511_block05_260427_block02" \
  --dataset-name "musc_model_a_plus_injury_calibrated_linear_svm_drop_260511_block07_260511_block05_260427_block02_perm1000" \
  --permutations 2000

python3 06g_summarize_classifier_results.py \
  --root "$ROOT" \
  --output "$ROOT/final_best_models_1000_perm_summary.csv"

echo "Done. Summary saved to:"
echo "$ROOT/final_best_models_1000_perm_summary.csv"


