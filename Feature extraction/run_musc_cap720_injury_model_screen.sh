#!/bin/bash
set -euo pipefail

# Run MUSC Model A + injury using injury features recalculated from only the
# first 720 elapsed minutes of each track.
#
# Outputs are kept separate from the original analyses:
#   manual_injury_feature_outputs_time_corrected_cap720/
#   constrained_separation_results_time_corrected_cap720/

PERMUTATIONS="${PERMUTATIONS:-0}"


echo "============================================================"
echo "1. Build MUSC cap720 injury features"
echo "============================================================"

python3 -u 10b_injury_roi_feature_extraction_musc_cap720.py \
  --model-a-table constrained_fish_features_time_corrected_cap720/musc/constrained_fish_level_mean_median.csv \
  --output-root manual_injury_feature_outputs_time_corrected_cap720 \
  --max-elapsed-minutes 720

INPUT="manual_injury_feature_outputs_time_corrected_cap720/musc/constrained_fish_level_musc_model_a_plus_injury.csv"

echo "============================================================"
echo "2. Run MUSC cap720 Model A + injury classifiers"
echo "============================================================"

echo "------------------------------------------------------------"
echo "2.1 L1 logistic regression"
echo "------------------------------------------------------------"

python3 -u 06_test_constrained_fish_separation_final.py \
  --input "$INPUT" \
  --output-dir constrained_separation_results_time_corrected_cap720/model_injury/musc_l1 \
  --dataset-name "musc_model_a_plus_injury_time_corrected_cap720_l1" \
  --model l1 \
  --permutations "$PERMUTATIONS"

echo "------------------------------------------------------------"
echo "2.2 Linear SVM"
echo "------------------------------------------------------------"

python3 -u 06c_test_constrained_fish_separation_svm.py \
  --input "$INPUT" \
  --output-dir constrained_separation_results_time_corrected_cap720/svm_linear/model_injury/musc \
  --dataset-name "musc_model_a_plus_injury_time_corrected_cap720_svm_linear" \
  --kernel linear \
  --permutations "$PERMUTATIONS"

echo "------------------------------------------------------------"
echo "2.3 RBF SVM"
echo "------------------------------------------------------------"

python3 -u 06c_test_constrained_fish_separation_svm.py \
  --input "$INPUT" \
  --output-dir constrained_separation_results_time_corrected_cap720/svm_rbf/model_injury/musc \
  --dataset-name "musc_model_a_plus_injury_time_corrected_cap720_svm_rbf" \
  --kernel rbf \
  --permutations "$PERMUTATIONS"

echo "------------------------------------------------------------"
echo "2.4 L2 logistic regression"
echo "------------------------------------------------------------"

python3 -u 06d_test_constrained_fish_separation_l2_logistic.py \
  --input "$INPUT" \
  --output-dir constrained_separation_results_time_corrected_cap720/l2_logistic/model_injury/musc \
  --dataset-name "musc_model_a_plus_injury_time_corrected_cap720_l2_logistic" \
  --permutations "$PERMUTATIONS"

echo "------------------------------------------------------------"
echo "2.5 Shrinkage LDA"
echo "------------------------------------------------------------"

python3 -u 06e_test_constrained_fish_separation_lda_shrinkage.py \
  --input "$INPUT" \
  --output-dir constrained_separation_results_time_corrected_cap720/lda_shrinkage/model_injury/musc \
  --dataset-name "musc_model_a_plus_injury_time_corrected_cap720_lda_shrinkage" \
  --permutations "$PERMUTATIONS"

echo "------------------------------------------------------------"
echo "2.6 Calibrated linear SVM"
echo "------------------------------------------------------------"

python3 -u 06f_test_constrained_fish_separation_calibrated_linear_svm.py \
  --input "$INPUT" \
  --output-dir constrained_separation_results_time_corrected_cap720/calibrated_linear_svm/model_injury/musc \
  --dataset-name "musc_model_a_plus_injury_time_corrected_cap720_calibrated_linear_svm" \
  --permutations "$PERMUTATIONS"

echo "============================================================"
echo "3. Summarize cap720 MUSC results including injury"
echo "============================================================"

python3 -u 06g_summarize_classifier_results.py \
  --root constrained_separation_results_time_corrected_cap720 \
  --output constrained_separation_results_time_corrected_cap720/musc_cap720_with_injury_model_summary.csv

echo
echo "============================================================"
echo "Cap720 MUSC injury analysis completed successfully"
echo "============================================================"
echo "Summary:"
echo "constrained_separation_results_time_corrected_cap720/musc_cap720_with_injury_model_summary.csv"

