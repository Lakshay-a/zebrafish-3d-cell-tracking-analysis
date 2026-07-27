#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export FEATURE_OVERNIGHT_DIR="$PROJECT_ROOT/overnight_batch_outputs/Liraglutide"
export FEATURE_METADATA_CSV="$SCRIPT_DIR/Liraglutide_metadata.csv"
export FEATURE_OUTPUT_DIR="$SCRIPT_DIR/final_feature_outputs_liraglutide"
export FEATURE_XY_PIXEL_SIZE_UM="0.7533114346590908"
export FEATURE_Z_STEP_SIZE_UM="1.0"

bash "$SCRIPT_DIR/feature_extraction.sh"
