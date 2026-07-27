#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OVERNIGHT_DIR="${OVERNIGHT_DIR:-$SCRIPT_DIR/overnight_batch_outputs/Liraglutide}"

echo "============================================================"
echo "RUNNING 3D RECONSTRUCTION + TRACKING FOR ALL OVERNIGHT BLOCKS"
echo "RESUME MODE: completed outputs will be skipped"
echo "============================================================"
echo "Overnight folder: $OVERNIGHT_DIR"
echo



# Pixel size lookup


get_pixel_sizes_for_block() {
    local block_name="$1"

    # ORIGINAL UNTREATED LOOKUP (disabled while processing MMP)
    #
    # case "$block_name" in
    #     260420_block03)
    #         XY_PIXEL_SIZE_UM_VALUE="0.828642578125"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;
    #
    # 260427_block02|260427_block03|260427_block06|260427_block07|260427_block08|260428_block02|260428.
    #         XY_PIXEL_SIZE_UM_VALUE="0.7533114346590908"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;
    #
    #     20240325_block02)
    #         XY_PIXEL_SIZE_UM_VALUE="0.3452677408854165"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;
    #
    #     20240422_block01|20240422_block06|20240422_block08|20240422_block11)
    #         XY_PIXEL_SIZE_UM_VALUE="0.4143212890625"
    #         Z_STEP_SIZE_UM_VALUE="1.0"
    #         ;;
    #
    #     *)
    #         echo "[BLOCK] WARNING: no pixel size entry for $block_name."
    #         echo "        Skipping this block."
    #         return 1
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

    # liraglutide lookup
    case "$block_name" in
        260512_block06|260513_block08|260519_block01|260519_block05|260519_block07|260519_block08|260601_block01|260601_block03|260601_block05)
            XY_PIXEL_SIZE_UM_VALUE="0.7533114346590908"
            Z_STEP_SIZE_UM_VALUE="1.0"
            ;;

        *)
            echo "[BLOCK] WARNING: no Liraglutide pixel size entry for $block_name."
            echo "        Skipping this block."
            return 1
            ;;
    esac

}



# Safe run helper


run_step() {
    local log_file="$1"
    shift

    echo "[RUN] $*"
    echo "[LOG] $log_file"

    if "$@" 2>&1 | tee "$log_file"; then
        echo "[DONE] Step completed."
        return 0
    else
        echo "[ERROR] Step failed. See log:"
        echo "        $log_file"
        return 1
    fi
}



# Main loop


for BLOCK_DIR in "$OVERNIGHT_DIR"/*; do

    if [ ! -d "$BLOCK_DIR" ]; then
        continue
    fi

    BLOCK_NAME=$(basename "$BLOCK_DIR")

    echo
    echo "============================================================"
    echo "PROCESSING BLOCK: $BLOCK_NAME"
    echo "Path: $BLOCK_DIR"
    echo "============================================================"

    if ! get_pixel_sizes_for_block "$BLOCK_NAME"; then
        continue
    fi

    export BATCH_XY_PIXEL_SIZE_UM="$XY_PIXEL_SIZE_UM_VALUE"
    export BATCH_Z_STEP_SIZE_UM="$Z_STEP_SIZE_UM_VALUE"

    echo "[BLOCK] Pixel sizes for $BLOCK_NAME:"
    echo "        XY=$BATCH_XY_PIXEL_SIZE_UM"
    echo "        Z =$BATCH_Z_STEP_SIZE_UM"

    mkdir -p "$BLOCK_DIR/logs"



    # Musc


    MUSC_MASK="$BLOCK_DIR/musc_cellpose_masks_TZYX.tif"
    MUSC_LABELS="$BLOCK_DIR/musc_3d_labels_TZYX.tif"
    MUSC_FINAL="$BLOCK_DIR/musc_tracking_outputs/musc_track_filter_report_nearest.csv"

    if [ -f "$MUSC_MASK" ]; then
        echo
        echo "------------------------------"
        echo "MUSC found: $MUSC_MASK"
        echo "Tracking method: nearest"
        echo "------------------------------"

        if [ ! -f "$MUSC_LABELS" ]; then
            echo "[MUSC] 3D labels missing. Running reconstruction..."

            if ! run_step "$BLOCK_DIR/logs/musc_2_reconstruct.log" \
                env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                    BATCH_CELL_TYPE=musc \
                    TRACKING_METHOD=nearest \
                    python3 -u 2_reconstruct3D_objects.py
            then
                echo "[MUSC] Reconstruction failed. Skipping MUSC tracking for this block."
            fi

        else
            echo "[MUSC] 3D labels already exist. Skipping reconstruction:"
            echo "       $MUSC_LABELS"
        fi

        if [ -f "$MUSC_FINAL" ]; then
            echo "[MUSC] Final output already exists. Skipping scripts 03, 5, 6, 7:"
            echo "       $MUSC_FINAL"
        else
            echo "[MUSC] Final output missing. Running scripts 03, 5, 6, 7..."

            if [ -f "$MUSC_LABELS" ]; then
                run_step "$BLOCK_DIR/logs/musc_03_track_nearest.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=musc \
                        TRACKING_METHOD=nearest \
                        python3 -u 03_track_3d_objects.py

                run_step "$BLOCK_DIR/logs/musc_5_qc_nearest.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=musc \
                        TRACKING_METHOD=nearest \
                        python3 -u 5_track_qc_summary.py

                run_step "$BLOCK_DIR/logs/musc_6_features_nearest.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=musc \
                        TRACKING_METHOD=nearest \
                        python3 -u 6_track_quality_features.py

                run_step "$BLOCK_DIR/logs/musc_7_filter_nearest.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=musc \
                        TRACKING_METHOD=nearest \
                        python3 -u 7_filtered_good_tracks.py
            else
                echo "[MUSC] 3D labels still missing. Skipping MUSC tracking."
            fi
        fi

    else
        echo
        echo "[MUSC] No MUSC Cellpose mask found. Skipping MUSC for $BLOCK_NAME"
        echo "       Expected: $MUSC_MASK"
    fi



    # Macrophage


    MAC_MASK="$BLOCK_DIR/macrophage_cellpose_masks_TZYX.tif"
    MAC_LABELS="$BLOCK_DIR/macrophage_3d_labels_TZYX.tif"

    MAC_CLUSTER_EXCLUSION="$BLOCK_DIR/macrophage_cluster_tracking_exclusion_mask_TYX.tif"
    MAC_OUTSIDE_BOUNDARY_CSV="$BLOCK_DIR/macrophage_3d_object_features_outside_boundary.csv"

    MAC_ALL_FINAL="$BLOCK_DIR/macrophage_tracking_outputs_all/macrophage_track_filter_report_lap.csv"
    MAC_OB_FINAL="$BLOCK_DIR/macrophage_tracking_outputs_outside_boundary/macrophage_track_filter_report_lap.csv"

    if [ -f "$MAC_MASK" ]; then
        echo
        echo "------------------------------"
        echo "MACROPHAGE found: $MAC_MASK"
        echo "Tracking method: LAP"
        echo "Region modes: all + outside_boundary"
        echo "------------------------------"

        if [ ! -f "$MAC_LABELS" ]; then
            echo "[MACROPHAGE] 3D labels missing. Running reconstruction..."

            if ! run_step "$BLOCK_DIR/logs/macrophage_2_reconstruct.log" \
                env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                    BATCH_CELL_TYPE=macrophage \
                    TRACKING_METHOD=lap \
                    python3 -u 2_reconstruct3D_objects.py
            then
                echo "[MACROPHAGE] Reconstruction failed. Skipping macrophage tracking for this block."
                continue
            fi

        else
            echo "[MACROPHAGE] 3D labels already exist. Skipping reconstruction:"
            echo "             $MAC_LABELS"
        fi



        # 3b cluster detection
        # Only run if cluster exclusion mask does not exist


        echo
        echo "[MACROPHAGE] Checking cluster exclusion mask:"
        echo "             $MAC_CLUSTER_EXCLUSION"

        if [ ! -f "$MAC_CLUSTER_EXCLUSION" ]; then
            echo "[MACROPHAGE] Cluster tracking exclusion mask missing."
            echo "[MACROPHAGE] Running 3b cluster detection..."

            if ! run_step "$BLOCK_DIR/logs/macrophage_3b_cluster.log" \
                env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                    BATCH_CELL_TYPE=macrophage \
                    TRACKING_METHOD=lap \
                    python3 -u 3b_detect_macrophage_clusters.py
            then
                echo "[MACROPHAGE] 3b failed. Continuing with all mode only if possible."
            fi
        else
            echo "[MACROPHAGE] Cluster tracking exclusion mask already exists. Skipping 3b."
        fi



        # MACROPHAGE REGION MODE: all
        # Resume check: skip if final all output exists


        if [ -f "$MAC_ALL_FINAL" ]; then
            echo
            echo "[MACROPHAGE all] Final output already exists. Skipping scripts 03, 5, 6, 7:"
            echo "                 $MAC_ALL_FINAL"
        else
            echo
            echo "[MACROPHAGE all] Final output missing. Running scripts 03, 5, 6, 7..."

            if [ -f "$MAC_LABELS" ]; then
                run_step "$BLOCK_DIR/logs/macrophage_03_track_all_lap.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=macrophage \
                        MACROPHAGE_REGION_MODE=all \
                        TRACKING_METHOD=lap \
                        python3 -u 03_track_3d_objects.py

                run_step "$BLOCK_DIR/logs/macrophage_5_qc_all_lap.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=macrophage \
                        MACROPHAGE_REGION_MODE=all \
                        TRACKING_METHOD=lap \
                        python3 -u 5_track_qc_summary.py

                run_step "$BLOCK_DIR/logs/macrophage_6_features_all_lap.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=macrophage \
                        MACROPHAGE_REGION_MODE=all \
                        TRACKING_METHOD=lap \
                        python3 -u 6_track_quality_features.py

                run_step "$BLOCK_DIR/logs/macrophage_7_filter_all_lap.log" \
                    env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                        BATCH_CELL_TYPE=macrophage \
                        MACROPHAGE_REGION_MODE=all \
                        TRACKING_METHOD=lap \
                        python3 -u 7_filtered_good_tracks.py
            else
                echo "[MACROPHAGE all] 3D labels missing. Skipping all-mode tracking."
            fi
        fi



        # MACROPHAGE REGION MODE: outside_boundary
        # Only run if cluster outputs exist
        # Resume check: skip if final outside_boundary output exists


        if [ -f "$MAC_OB_FINAL" ]; then
            echo
            echo "[MACROPHAGE outside_boundary] Final output already exists. Skipping scripts 03, 5, 6, 7:"
            echo "                              $MAC_OB_FINAL"

        elif [ -f "$MAC_CLUSTER_EXCLUSION" ] && [ -f "$MAC_OUTSIDE_BOUNDARY_CSV" ]; then
            echo
            echo "[MACROPHAGE outside_boundary] Cluster outputs exist."
            echo "[MACROPHAGE outside_boundary] Running scripts 03, 5, 6, 7..."

            run_step "$BLOCK_DIR/logs/macrophage_03_track_outside_boundary_lap.log" \
                env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                    BATCH_CELL_TYPE=macrophage \
                    MACROPHAGE_REGION_MODE=outside_boundary \
                    TRACKING_METHOD=lap \
                    python3 -u 03_track_3d_objects.py

            run_step "$BLOCK_DIR/logs/macrophage_5_qc_outside_boundary_lap.log" \
                env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                    BATCH_CELL_TYPE=macrophage \
                    MACROPHAGE_REGION_MODE=outside_boundary \
                    TRACKING_METHOD=lap \
                    python3 -u 5_track_qc_summary.py

            run_step "$BLOCK_DIR/logs/macrophage_6_features_outside_boundary_lap.log" \
                env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                    BATCH_CELL_TYPE=macrophage \
                    MACROPHAGE_REGION_MODE=outside_boundary \
                    TRACKING_METHOD=lap \
                    python3 -u 6_track_quality_features.py

            run_step "$BLOCK_DIR/logs/macrophage_7_filter_outside_boundary_lap.log" \
                env BATCH_PROJECT_DIR="$BLOCK_DIR" \
                    BATCH_CELL_TYPE=macrophage \
                    MACROPHAGE_REGION_MODE=outside_boundary \
                    TRACKING_METHOD=lap \
                    python3 -u 7_filtered_good_tracks.py

        else
            echo
            echo "[MACROPHAGE outside_boundary] Missing cluster outputs. Skipping outside_boundary."
            echo "                              Expected mask: $MAC_CLUSTER_EXCLUSION"
            echo "                              Expected CSV:  $MAC_OUTSIDE_BOUNDARY_CSV"
        fi

    else
        echo
        echo "[MACROPHAGE] No macrophage Cellpose mask found. Skipping macrophage for $BLOCK_NAME"
        echo "             Expected: $MAC_MASK"
    fi

done

echo
echo "============================================================"
echo "ALL OVERNIGHT BLOCKS COMPLETE"
echo "============================================================"
