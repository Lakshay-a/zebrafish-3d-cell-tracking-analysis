#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_DIR="$ROOT/Feature extraction"
DATA_DIR="$ROOT/analysis_data"
OUTPUT_DIR="${REPRODUCED_PLOT_DIR:-$ROOT/reproduced_analysis/final_model_plots}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/reproduced_analysis/_mpl_cache}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" -u "$FEATURE_DIR/plot_scripts/generate_all_final_model_plots.py" \
  --feature-extraction-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "[DONE] Reproduced final-model plots: $OUTPUT_DIR"
