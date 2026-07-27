#!/bin/bash
set -euo pipefail

# Model C:
# outside-boundary Model A + cluster/accumulation dynamics + L1 nested LOFO

PERMUTATIONS="${PERMUTATIONS:-500}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

BLOCKS_ROOT="${BLOCKS_ROOT:-$PROJECT_ROOT/overnight_batch_outputs}"

MODEL_A_TABLE="constrained_fish_features/macrophage_outside_boundary/constrained_fish_level_mean_median.csv"

MODEL_C_FEATURE_DIR="constrained_fish_features/model_c/macrophage_outside_boundary"

MODEL_C_RESULT_DIR="constrained_separation_results/model_c/macrophage_outside_boundary_l1"

echo "============================================================"
echo "BUILDING MODEL C FEATURES"
echo "Started: $(date)"
echo "============================================================"

python3 -u 08_extract_model_c_cluster_dynamics.py \
  --blocks-root "$BLOCKS_ROOT" \
  --model-a-table "$MODEL_A_TABLE" \
  --output-dir "$MODEL_C_FEATURE_DIR" \
  --feature-set compact \
  --rolling-window 5

echo
echo "============================================================"
echo "RUNNING MODEL C L1 NESTED LOFO"
echo "Started: $(date)"
echo "============================================================"

python3 -u 06_test_constrained_fish_separation_final.py \
  --input "$MODEL_C_FEATURE_DIR/constrained_fish_level_model_c.csv" \
  --output-dir "$MODEL_C_RESULT_DIR" \
  --dataset-name "macrophage_outside_boundary_model_c_l1" \
  --model l1 \
  --permutations "$PERMUTATIONS"

echo
echo "============================================================"
echo "MODEL C COMPLETED"
echo "Finished: $(date)"
echo "============================================================"
