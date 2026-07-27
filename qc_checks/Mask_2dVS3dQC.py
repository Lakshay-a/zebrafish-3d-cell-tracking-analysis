import numpy as np
import pandas as pd
import tifffile as tiff

from config import (
    CELL_TYPE,
    CELLPOSE_MASKS_TZYX_TIF,
    LABELS_3D_TZYX_TIF,
    PROJECT_DIR,
)

TIME_INDEX = 23

OUTPUT_CSV = PROJECT_DIR / f"{CELL_TYPE}_2d_vs_3d_counts_T{TIME_INDEX:04d}.csv"


def count_objects_2d(label_2d):
    labels = np.unique(label_2d)
    labels = labels[labels != 0]
    return len(labels)


def main():
    print("=" * 70)
    print("2D MASK VS 3D LABEL COUNT CHECK")
    print("=" * 70)

    print(f"Cell type: {CELL_TYPE}")
    print(f"2D masks:  {CELLPOSE_MASKS_TZYX_TIF}")
    print(f"3D labels: {LABELS_3D_TZYX_TIF}")
    print(f"Time:      {TIME_INDEX}")

    mask_tzyx = tiff.imread(CELLPOSE_MASKS_TZYX_TIF)
    label_tzyx = tiff.imread(LABELS_3D_TZYX_TIF)

    print(f"2D mask shape:  {mask_tzyx.shape}")
    print(f"3D label shape: {label_tzyx.shape}")

    if mask_tzyx.shape != label_tzyx.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"2D masks:  {mask_tzyx.shape}\n"
            f"3D labels: {label_tzyx.shape}"
        )

    T, Z, Y, X = mask_tzyx.shape

    if TIME_INDEX >= T:
        raise ValueError(f"TIME_INDEX={TIME_INDEX}, but T={T}")

    rows = []

    for z in range(Z):
        mask_2d = mask_tzyx[TIME_INDEX, z]
        label_2d = label_tzyx[TIME_INDEX, z]

        mask_objects = count_objects_2d(mask_2d)
        label_objects = count_objects_2d(label_2d)

        mask_pixels = int((mask_2d > 0).sum())
        label_pixels = int((label_2d > 0).sum())

        retained_pixel_fraction = (
            label_pixels / mask_pixels if mask_pixels > 0 else np.nan
        )

        rows.append(
            {
                "time": TIME_INDEX,
                "z": z,
                "mask_2d_objects": mask_objects,
                "label_3d_objects_visible_in_slice": label_objects,
                "mask_2d_pixels": mask_pixels,
                "label_3d_pixels_visible_in_slice": label_pixels,
                "retained_pixel_fraction": retained_pixel_fraction,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print(df)
    print()
    print("Summary:")
    print(f"Total 2D mask pixels at T={TIME_INDEX}: {df['mask_2d_pixels'].sum()}")
    print(f"Total 3D label pixels at T={TIME_INDEX}: {df['label_3d_pixels_visible_in_slice'].sum()}")
    print(f"Total 2D objects across Z slices: {df['mask_2d_objects'].sum()}")
    print(f"Total visible 3D label objects across Z slices: {df['label_3d_objects_visible_in_slice'].sum()}")

    print()
    print(f"[SAVED] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()