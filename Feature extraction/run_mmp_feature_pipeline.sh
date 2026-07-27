#!/bin/bash
set -euo pipefail

# Extract MMP features, combine all fish, apply track QC and time correction,
# then construct fish-level constrained feature tables for Models A, B and D.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

METADATA_CSV="$SCRIPT_DIR/MMP_metadata.csv"
ANALYSIS_METADATA_CSV="$SCRIPT_DIR/MMP_analysis_metadata.csv"
FEATURE_ROOT="$SCRIPT_DIR/final_feature_outputs_mmp"
COMBINED_DIR="$SCRIPT_DIR/final_combined_features_mmp"
QC_DIR="$SCRIPT_DIR/qc_outlier_outputs_mmp"
TIME_CORRECTED_DIR="$SCRIPT_DIR/qc_outlier_outputs_time_corrected_mmp"
FISH_FEATURE_DIR="$SCRIPT_DIR/constrained_fish_features_time_corrected_mmp"

DATASETS=(musc macrophage_all macrophage_outside_boundary)

combined_filename_for_dataset() {
  local dataset="$1"

  case "$dataset" in
    musc)
      echo "musc_all_blocks_cell_track_features.csv"
      ;;
    macrophage_all)
      echo "macrophage_all_blocks_cell_track_features.csv"
      ;;
    macrophage_outside_boundary)
      echo "macrophage_outside_boundary_all_blocks_cell_track_features.csv"
      ;;
    *)
      echo "[ERROR] Unknown dataset: $dataset" >&2
      return 1
      ;;
  esac
}

cd "$SCRIPT_DIR"

for required in \
  "$METADATA_CSV" \
  "$ANALYSIS_METADATA_CSV" \
  "$SCRIPT_DIR/feature_extraction.sh" \
  "$SCRIPT_DIR/combine_feature_outputs.py" \
  "$SCRIPT_DIR/02_filter_outliers.py" \
  "$SCRIPT_DIR/03b_recalculate_time_corrected_track_features.py" \
  "$SCRIPT_DIR/04b_make_constrained_fish_features.py" \
  "$SCRIPT_DIR/04c_make_model_b_d_fish_features.py"
do
  if [ ! -f "$required" ]; then
    echo "[ERROR] Missing required file: $required"
    exit 1
  fi
done

if ! awk -F',' '
  NR == 1 {
    for (i = 1; i <= NF; i++) {
      if ($i == "block_name") block_col = i
      if ($i == "time_interval_seconds") interval_col = i
    }
    next
  }
  {
    value = $interval_col
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    if (value == "" || value + 0 <= 0) {
      print $block_col
      bad = 1
    }
  }
  END {
    if (!block_col || !interval_col) exit 2
    exit bad
  }
' "$METADATA_CSV"
then
  echo "[ERROR] MMP_metadata.csv has missing/non-positive time intervals,"
  echo "        or lacks block_name/time_interval_seconds columns."
  exit 1
fi

mkdir -p \
  "$FEATURE_ROOT" \
  "$COMBINED_DIR" \
  "$QC_DIR" \
  "$TIME_CORRECTED_DIR" \
  "$FISH_FEATURE_DIR"

echo "============================================================"
echo "1. Extract per-block MMP object and track features"
echo "============================================================"
bash "$SCRIPT_DIR/feature_extraction.sh"

echo "============================================================"
echo "2. Combine every MMP fish within each dataset"
echo "============================================================"
"$PYTHON_BIN" -u "$SCRIPT_DIR/combine_feature_outputs.py" \
  --feature-root "$FEATURE_ROOT" \
  --output-dir "$COMBINED_DIR"

echo "============================================================"
echo "3. Apply track-level QC to each combined dataset"
echo "============================================================"
for dataset in "${DATASETS[@]}"; do
  combined_filename=$(combined_filename_for_dataset "$dataset")
  input="$COMBINED_DIR/$combined_filename"

  if [ ! -f "$input" ]; then
    echo "[ERROR] Missing combined input for $dataset: $input"
    exit 1
  fi

  "$PYTHON_BIN" -u "$SCRIPT_DIR/02_filter_outliers.py" \
    --features "$input" \
    --output-dir "$QC_DIR/$dataset"
done

echo "============================================================"
echo "4. Recalculate movement features using real frame intervals"
echo "============================================================"
for dataset in "${DATASETS[@]}"; do
  qc_table="$QC_DIR/$dataset/cell_track_features_qc_filtered.csv"

  "$PYTHON_BIN" -u "$SCRIPT_DIR/03b_recalculate_time_corrected_track_features.py" \
    --root "$FEATURE_ROOT" \
    --dataset "$dataset" \
    --metadata-file "$METADATA_CSV" \
    --qc-track-table "$qc_table" \
    --output-dir "$TIME_CORRECTED_DIR/$dataset"
done

echo "============================================================"
echo "5. Build constrained Model A fish-level tables"
echo "============================================================"
for dataset in "${DATASETS[@]}"; do
  input="$TIME_CORRECTED_DIR/$dataset/cell_track_features_time_corrected.csv"

  "$PYTHON_BIN" -u "$SCRIPT_DIR/04b_make_constrained_fish_features.py" \
    --input "$input" \
    --metadata-file "$ANALYSIS_METADATA_CSV" \
    --metadata-fish-col fish_id \
    --output-dir "$FISH_FEATURE_DIR/$dataset" \
    --dataset-name "${dataset}_mmp_model_a"
done

echo "============================================================"
echo "6. Build constrained Model B and D fish-level tables"
echo "============================================================"
for dataset in "${DATASETS[@]}"; do
  input="$TIME_CORRECTED_DIR/$dataset/cell_track_features_time_corrected.csv"

  for model in b d; do
    "$PYTHON_BIN" -u "$SCRIPT_DIR/04c_make_model_b_d_fish_features.py" \
      --input "$input" \
      --metadata-file "$ANALYSIS_METADATA_CSV" \
      --metadata-fish-col fish_id \
      --model "$model" \
      --output-dir "$FISH_FEATURE_DIR/model_$model/$dataset" \
      --dataset-name "${dataset}_mmp_model_$model"
  done
done

echo "============================================================"
echo "MMP FEATURE PIPELINE COMPLETE"
echo "Fish-level tables:"
echo "$FISH_FEATURE_DIR"
echo "============================================================"
