#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERNIGHT_DIR="${FEATURE_OVERNIGHT_DIR:-$PROJECT_ROOT/overnight_batch_outputs/MMP}"
FEATURE_DIR="$PROJECT_ROOT/Feature extraction"

METADATA_CSV="${FEATURE_METADATA_CSV:-$FEATURE_DIR/MMP_metadata.csv}"
FEATURE_SCRIPT="$FEATURE_DIR/01_extract_tracked_cell_features.py"

FINAL_FEATURE_DIR="${FEATURE_OUTPUT_DIR:-$FEATURE_DIR/final_feature_outputs_mmp}"
mkdir -p "$FINAL_FEATURE_DIR"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "RUNNING FEATURE EXTRACTION FOR ALL BLOCKS"
echo "============================================================"
echo "Overnight dir: $OVERNIGHT_DIR"
echo "Metadata CSV:  $METADATA_CSV"
echo "Output dir:    $FINAL_FEATURE_DIR"
echo

if [ ! -f "$METADATA_CSV" ]; then
    echo "[ERROR] Missing metadata file:"
    echo "$METADATA_CSV"
    echo
    echo "Create block_metadata.csv with columns:"
    echo "block_name,fish_id,genotype"
    exit 1
fi

if [ ! -f "$FEATURE_SCRIPT" ]; then
    echo "[ERROR] Missing feature script:"
    echo "$FEATURE_SCRIPT"
    exit 1
fi


get_metadata_for_block() {
    local block_name="$1"

    META_LINE=$(awk -F',' -v b="$block_name" '
        NR > 1 && $1 == b {print $0}
    ' "$METADATA_CSV")

    if [ -z "$META_LINE" ]; then
        echo "[WARN] No metadata found for $block_name"
        return 1
    fi

    FISH_ID=$(echo "$META_LINE" | awk -F',' '{print $2}')
    GENOTYPE=$(echo "$META_LINE" | awk -F',' '{print $3}')

    if [ -z "$FISH_ID" ] || [ -z "$GENOTYPE" ]; then
        echo "[WARN] Incomplete metadata for $block_name"
        return 1
    fi

    return 0
}


get_pixel_sizes_for_block() {
    local block_name="$1"

    if [ -n "${FEATURE_XY_PIXEL_SIZE_UM:-}" ] && [ -n "${FEATURE_Z_STEP_SIZE_UM:-}" ]; then
        XY_PIXEL_SIZE_UM_VALUE="$FEATURE_XY_PIXEL_SIZE_UM"
        Z_STEP_SIZE_UM_VALUE="$FEATURE_Z_STEP_SIZE_UM"
        return 0
    fi

    # case "$block_name" in
    #     260420_block03)
    #         XY_PIXEL_SIZE_UM_VALUE="0.828642578125"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;

    # 260427_block02|260427_block03|260427_block06|260427_block07|260427_block08|260428_block02|260428.
    #         XY_PIXEL_SIZE_UM_VALUE="0.7533114346590908"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;

    #     20240325_block02)
    #         XY_PIXEL_SIZE_UM_VALUE="0.3452677408854165"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;

    #     20240422_block01|20240422_block06|20240422_block08|20240422_block11)
    #         XY_PIXEL_SIZE_UM_VALUE="0.4143212890625"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;

    #     *)
    #         XY_PIXEL_SIZE_UM_VALUE="0.7533114346590908"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         echo "[WARN] Using default pixel size for $block_name"
    #         ;;
    # esac

    # # mmp lookup
    # # All MMP CZI files have the same physical pixel sizes.
    # case "$block_name" in
    # 20240604_block02|20240604_block06|20240604_block07|20240702_block02|20240702_block03|20240702_bl.
    #         XY_PIXEL_SIZE_UM_VALUE="0.2959437779017855"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;

    #     *)
    #         echo "[BLOCK] WARNING: no MMP pixel size entry for $block_name."
    #         echo "        Skipping this block."
    #         return 1
    #         ;;
    # esac

    # Default MMP lookup. Other conditions use a dedicated launcher that
    # supplies FEATURE_XY_PIXEL_SIZE_UM and FEATURE_Z_STEP_SIZE_UM.
    case "$block_name" in
        20240604_block02|20240604_block06|20240604_block07|20240702_block02|20240702_block03|20240702_block04|20240702_block05|20240702_block06|20240702_block07)
            XY_PIXEL_SIZE_UM_VALUE="0.2959437779017855"
            Z_STEP_SIZE_UM_VALUE="1.0"
            ;;

        *)
            echo "[BLOCK] WARNING: no pixel size entry for $block_name."
            echo "        Skipping this block."
            return 1
            ;;
    esac
}


find_first_csv() {
    local folder="$1"
    local pattern="$2"

    find "$folder" -maxdepth 1 -name "$pattern" -type f | sort | head -n 1
}


run_features_for_one_mode() {
    local block_name="$1"
    local block_dir="$2"
    local cell_type="$3"
    local region_mode="$4"
    local tracking_dir="$5"
    local labels_tif="$6"
    local tracks_csv="$7"
    local include_cluster="$8"
    local output_subdir="$9"

    if [ ! -f "$labels_tif" ]; then
        echo "[SKIP] Missing labels: $labels_tif"
        return 0
    fi

    if [ ! -f "$tracks_csv" ]; then
        echo "[SKIP] Missing tracks: $tracks_csv"
        return 0
    fi

    local raw_tif="$block_dir/raw_TCZYX_drift_corrected.tif"

    if [ ! -f "$raw_tif" ]; then
        echo "[SKIP] Missing raw tif: $raw_tif"
        return 0
    fi

    local out_dir="$FINAL_FEATURE_DIR/$block_name/$output_subdir"
    mkdir -p "$out_dir"

    echo "------------------------------------------------------------"
    echo "[RUN] $block_name | $cell_type | $region_mode"
    echo "      fish_id:   $FISH_ID"
    echo "      genotype:  $GENOTYPE"
    echo "      tracks:    $tracks_csv"
    echo "      labels:    $labels_tif"
    echo "      output:    $out_dir"
    echo "------------------------------------------------------------"

    FEATURE_BLOCK_NAME="$block_name" \
    FEATURE_FISH_ID="$FISH_ID" \
    FEATURE_GENOTYPE="$GENOTYPE" \
    BATCH_CELL_TYPE="$cell_type" \
    MACROPHAGE_REGION_MODE="$region_mode" \
    FEATURE_INCLUDE_CLUSTER="$include_cluster" \
    FEATURE_TRACKS_CSV="$tracks_csv" \
    FEATURE_LABELS_TIF="$labels_tif" \
    FEATURE_RAW_TIF="$raw_tif" \
    FEATURE_OUTPUT_DIR="$out_dir" \
    FEATURE_OUTPUT_BASE_DIR="$tracking_dir" \
    FEATURE_XY_PIXEL_SIZE_UM="$XY_PIXEL_SIZE_UM_VALUE" \
    FEATURE_Z_STEP_SIZE_UM="$Z_STEP_SIZE_UM_VALUE" \
    python3 -u "$FEATURE_SCRIPT"
}


for block_dir in "$OVERNIGHT_DIR"/*/; do
    [ -d "$block_dir" ] || continue

    block_name=$(basename "$block_dir")

    echo
    echo "============================================================"
    echo "BLOCK: $block_name"
    echo "============================================================"

    if ! get_metadata_for_block "$block_name"; then
        echo "[SKIP] No genotype/fish metadata for $block_name"
        continue
    fi

    get_pixel_sizes_for_block "$block_name"

    export BATCH_XY_PIXEL_SIZE_UM="$XY_PIXEL_SIZE_UM_VALUE"
    export BATCH_Z_STEP_SIZE_UM="$Z_STEP_SIZE_UM_VALUE"


    # Musc


    MUSC_LABELS="$block_dir/musc_3d_labels_TZYX.tif"
    MUSC_TRACKING_DIR="$block_dir/musc_tracking_outputs"

    if [ -d "$MUSC_TRACKING_DIR" ]; then
        MUSC_TRACKS=$(find_first_csv "$MUSC_TRACKING_DIR" "musc_tracks_*good_filtered*.csv")

        if [ -z "$MUSC_TRACKS" ]; then
            MUSC_TRACKS=$(find_first_csv "$MUSC_TRACKING_DIR" "musc_tracks_*.csv")
        fi

        if [ -n "$MUSC_TRACKS" ]; then
            run_features_for_one_mode \
                "$block_name" \
                "$block_dir" \
                "musc" \
                "all" \
                "$MUSC_TRACKING_DIR" \
                "$MUSC_LABELS" \
                "$MUSC_TRACKS" \
                "0" \
                "musc"
        else
            echo "[SKIP] No MUSC track CSV found for $block_name"
        fi
    else
        echo "[SKIP] No MUSC tracking folder for $block_name"
    fi



    # Macrophage - all region mode


    MAC_LABELS="$block_dir/macrophage_3d_labels_TZYX.tif"
    MAC_ALL_TRACKING_DIR="$block_dir/macrophage_tracking_outputs_all"

    if [ -d "$MAC_ALL_TRACKING_DIR" ]; then
        MAC_ALL_TRACKS=$(find_first_csv "$MAC_ALL_TRACKING_DIR" "macrophage_tracks_*good_filtered*.csv")

        if [ -z "$MAC_ALL_TRACKS" ]; then
            MAC_ALL_TRACKS=$(find_first_csv "$MAC_ALL_TRACKING_DIR" "macrophage_tracks_*.csv")
        fi

        if [ -n "$MAC_ALL_TRACKS" ]; then
            run_features_for_one_mode \
                "$block_name" \
                "$block_dir" \
                "macrophage" \
                "all" \
                "$MAC_ALL_TRACKING_DIR" \
                "$MAC_LABELS" \
                "$MAC_ALL_TRACKS" \
                "0" \
                "macrophage_all"
        else
            echo "[SKIP] No macrophage ALL track CSV found for $block_name"
        fi
    else
        echo "[SKIP] No macrophage ALL tracking folder for $block_name"
    fi



    # Macrophage - outside boundary region mode


    MAC_OUT_TRACKING_DIR="$block_dir/macrophage_tracking_outputs_outside_boundary"
    MAC_CLUSTER_MASK="$block_dir/macrophage_cluster_tracking_exclusion_mask_TYX.tif"

    if [ -d "$MAC_OUT_TRACKING_DIR" ] && [ -f "$MAC_CLUSTER_MASK" ]; then
        MAC_OUT_TRACKS=$(find_first_csv "$MAC_OUT_TRACKING_DIR" "macrophage_tracks_*good_filtered*.csv")

        if [ -z "$MAC_OUT_TRACKS" ]; then
            MAC_OUT_TRACKS=$(find_first_csv "$MAC_OUT_TRACKING_DIR" "macrophage_tracks_*.csv")
        fi

        if [ -n "$MAC_OUT_TRACKS" ]; then
            run_features_for_one_mode \
                "$block_name" \
                "$block_dir" \
                "macrophage" \
                "outside_boundary" \
                "$MAC_OUT_TRACKING_DIR" \
                "$MAC_LABELS" \
                "$MAC_OUT_TRACKS" \
                "1" \
                "macrophage_outside_boundary"
        else
            echo "[SKIP] No macrophage OUTSIDE_BOUNDARY track CSV found for $block_name"
        fi
    else
        echo "[SKIP] No macrophage OUTSIDE_BOUNDARY tracking folder or cluster mask for $block_name"
    fi

done


echo
echo "============================================================"
echo "FEATURE EXTRACTION COMPLETE"
echo "============================================================"
echo "Outputs saved in:"
echo "$FINAL_FEATURE_DIR"
echo
