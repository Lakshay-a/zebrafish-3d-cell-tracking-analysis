from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import shift as ndi_shift

from config import (
    CELL_TYPE,
    PROJECT_DIR,
    CELLPOSE_MASKS_TZYX_TIF,
)

# CSV created by check_shift.py
SHIFT_CSV = (
    PROJECT_DIR
    / "mask_shift_debug"
    / f"{CELL_TYPE}_mask_xy_shift_scores.csv"
)

OUT_MASK = (
    PROJECT_DIR
    / f"{CELL_TYPE}_cellpose_masks_TZYX_time_shifted.tif"
)

# If True, round shifts to nearest integer.
# This is safest for label masks.
ROUND_TO_INTEGER = True


def main():
    print("=" * 70)
    print("APPLY TIME-DEPENDENT MASK SHIFT")
    print("=" * 70)

    print(f"Input mask: {CELLPOSE_MASKS_TZYX_TIF}")
    print(f"Shift CSV:  {SHIFT_CSV}")
    print(f"Output:     {OUT_MASK}")

    masks = tiff.imread(CELLPOSE_MASKS_TZYX_TIF)
    print("Mask shape:", masks.shape)

    if masks.ndim != 4:
        raise ValueError(f"Expected T,Z,Y,X mask. Got {masks.shape}")

    T, Z, Y, X = masks.shape

    shifts = pd.read_csv(SHIFT_CSV)

    required = ["time", "best_dy", "best_dx"]
    missing = [c for c in required if c not in shifts.columns]
    if missing:
        raise ValueError(f"Shift CSV missing columns: {missing}")

    # Median shift per sampled timepoint across tested Z slices
    per_t = (
        shifts
        .groupby("time")[["best_dy", "best_dx"]]
        .median()
        .reset_index()
        .sort_values("time")
    )

    print("\nMeasured median shifts per sampled time:")
    print(per_t)

    sampled_t = per_t["time"].to_numpy(dtype=float)
    sampled_dy = per_t["best_dy"].to_numpy(dtype=float)
    sampled_dx = per_t["best_dx"].to_numpy(dtype=float)

    all_t = np.arange(T, dtype=float)

    # Interpolate shifts across all timepoints.
    # Before first sampled time and after last sampled time, use edge values.
    dy_all = np.interp(all_t, sampled_t, sampled_dy)
    dx_all = np.interp(all_t, sampled_t, sampled_dx)

    if ROUND_TO_INTEGER:
        dy_all = np.rint(dy_all).astype(int)
        dx_all = np.rint(dx_all).astype(int)

    shift_table = pd.DataFrame(
        {
            "time": np.arange(T),
            "applied_dy": dy_all,
            "applied_dx": dx_all,
        }
    )

    shift_table_path = (
        PROJECT_DIR
        / "mask_shift_debug"
        / f"{CELL_TYPE}_applied_time_shifts.csv"
    )
    shift_table.to_csv(shift_table_path, index=False)

    print("\nApplied shifts:")
    print(shift_table.head(20))
    print("...")
    print(shift_table.tail(20))
    print(f"\n[SAVED] Shift table: {shift_table_path}")

    shifted = np.zeros_like(masks)

    for t in range(T):
        dy = float(dy_all[t])
        dx = float(dx_all[t])

        print(f"[INFO] Shifting T={t:03d}: dy={dy}, dx={dx}")

        for z in range(Z):
            shifted[t, z] = ndi_shift(
                masks[t, z],
                shift=(dy, dx),
                order=0,          # nearest neighbour, preserves label IDs
                mode="constant",
                cval=0,
                prefilter=False,
            )

    tiff.imwrite(
        OUT_MASK,
        shifted.astype(masks.dtype),
        imagej=True,
        metadata={"axes": "TZYX"},
    )

    print(f"\n[SAVED] Shifted mask: {OUT_MASK}")
    print("[DONE]")


if __name__ == "__main__":
    main()