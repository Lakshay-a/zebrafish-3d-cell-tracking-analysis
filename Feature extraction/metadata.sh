#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERNIGHT_DIR="$PROJECT_ROOT/overnight_batch_outputs/Liraglutide"
FEATURE_DIR="$PROJECT_ROOT/Feature extraction"
OUT_CSV="$FEATURE_DIR/Liraglutide_metadata.csv"

mkdir -p "$FEATURE_DIR"

echo "block_name,fish_id,genotype,time_interval_seconds" > "$OUT_CSV"

find "$OVERNIGHT_DIR" -mindepth 1 -maxdepth 1 -type d \
  | sort \
  | while read -r block_dir; do
        block_name=$(basename "$block_dir")
        echo "$block_name,$block_name," >> "$OUT_CSV"
    done

echo "[DONE] Created metadata template:"
echo "$OUT_CSV"
echo
echo "Now fill the genotype column manually with WT or MUT."