#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
C_GRID="0.001,0.002848035868435802,0.008111308307896872,0.02310129700083159,0.0657933224657568,0.1873817422860383,0.5336699231206307,1.5199110829529332,4.328761281083057,12.32846739442066,35.11191734215127,100.0"

"$PYTHON_BIN" -u 11_fit_frozen_untreated_model.py \
  --input "constrained_fish_features_time_corrected/musc/constrained_fish_level_mean_median.csv" \
  --output-dir "frozen_untreated_models/musc_model_a_no_injury_all_fish" \
  --dataset-name "musc_model_a_no_injury_all_fish" \
  --classifier l1 \
  --c-grid "$C_GRID" \
  --evaluation-results-dir "constrained_separation_results_time_corrected/model_a/musc_l1"

"$PYTHON_BIN" -u 11_fit_frozen_untreated_model.py \
  --input "subset_sensitivity_results/musc_model_a_legacy_l1_top5_combo/_subset_inputs/drop_20240422_block06__20240422_block08.csv" \
  --output-dir "frozen_untreated_models/musc_model_a_no_injury_drop2" \
  --dataset-name "musc_model_a_no_injury_drop2" \
  --classifier l1 \
  --c-grid "$C_GRID" \
  --evaluation-results-dir "final_best_models_1000_perm/legacy_l1/musc_model_a_drop_20240422_block06_20240422_block08"

bash run_frozen_mmp_analysis.sh
