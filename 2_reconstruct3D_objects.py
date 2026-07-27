import pandas as pd
import tifffile as tiff
import numpy as np

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    RAW_TCZYX_TIF,
    CELLPOSE_MASKS_TZYX_TIF,
    LABELS_3D_TZYX_TIF,
    RECONSTRUCTION_CSV,
    OBJECT_FEATURES_CSV,
)

if CELL_TYPE=="musc":
    from utils_reconstruct_musc import (
        reconstruct_3d_objects_for_timepoint,
        extract_3d_features,
    )
elif CELL_TYPE=="macrophage":
    from utils_reconstruct_macrophage import (
        reconstruct_3d_objects_for_timepoint,
        extract_3d_features,
    )
# from utils_reconstruct import (
#     reconstruct_3d_objects_for_timepoint,
#     extract_3d_features,
# )


def load_selected_raw_channel():
    """Loads the shared raw stack and returns one selected channel as T,Z,Y,X."""
    print(f"[INFO] Loading raw stack: {RAW_TCZYX_TIF}")
    raw = tiff.imread(RAW_TCZYX_TIF)

    print(f"[INFO] Raw shape: {raw.shape}")

    if raw.ndim == 5:
        # T,c,z,y,x
        T, C, Z, Y, X = raw.shape

        if CHANNEL_INDEX >= C:
            raise ValueError(
                f"CHANNEL_INDEX={CHANNEL_INDEX}, but raw only has {C} channels. "
                f"Raw shape: {raw.shape}"
            )

        raw_tzyx = raw[:, CHANNEL_INDEX, :, :, :]

    elif raw.ndim == 4:
        # Already T,Z,Y,X
        raw_tzyx = raw

    else:
        raise ValueError(
            f"Expected raw shape T,C,Z,Y,X or T,Z,Y,X. Got {raw.shape}"
        )

    print(
        f"[INFO] Selected {CELL_TYPE} raw channel "
        f"CHANNEL_INDEX={CHANNEL_INDEX}, shape={raw_tzyx.shape}"
    )

    return raw_tzyx


def main():
    print("=" * 60)
    print(f"3D RECONSTRUCTION: {CELL_TYPE}")
    print("=" * 60)

    raw_4d = load_selected_raw_channel()

    print(f"[INFO] Loading 2D Cellpose masks: {CELLPOSE_MASKS_TZYX_TIF}")
    mask_4d = tiff.imread(CELLPOSE_MASKS_TZYX_TIF)

    print(f"[INFO] Selected raw shape: {raw_4d.shape}")
    print(f"[INFO] Mask shape:         {mask_4d.shape}")

    if mask_4d.ndim != 4:
        raise ValueError(
            f"Expected mask shape T,Z,Y,X. Got {mask_4d.shape}"
        )

    if raw_4d.shape != mask_4d.shape:
        raise ValueError(
            f"Raw and mask shape mismatch. raw={raw_4d.shape}, mask={mask_4d.shape}. "
            "This usually means the mask was generated from a different time range, "
            "different Z range, or different raw stack."
        )

    T, Z, Y, X = mask_4d.shape

    labels_4d = np.zeros_like(mask_4d, dtype=np.uint16)

    all_reconstruction_rows = []
    all_feature_rows = []

    file_name = f"{CELL_TYPE}_{RAW_TCZYX_TIF.stem}"

    for t_idx in range(T):
        print(f"[INFO] Reconstructing {CELL_TYPE} 3D objects at T={t_idx + 1}/{T}")

        label_3d, reconstruction_df = reconstruct_3d_objects_for_timepoint(
            mask_4d[t_idx]
        )

        labels_4d[t_idx] = label_3d.astype(np.uint16)

        feature_df = extract_3d_features(
            label_3d=label_3d,
            raw_3d=raw_4d[t_idx],
            file_name=file_name,
            time_index=t_idx,
        )

        if not reconstruction_df.empty:
            reconstruction_df["file"] = file_name
            reconstruction_df["cell_type"] = CELL_TYPE
            reconstruction_df["time"] = t_idx

        if not feature_df.empty:
            feature_df["cell_type"] = CELL_TYPE

        all_reconstruction_rows.append(reconstruction_df)
        all_feature_rows.append(feature_df)

    reconstruction_table = (
        pd.concat(all_reconstruction_rows, ignore_index=True)
        if all_reconstruction_rows
        else pd.DataFrame()
    )

    object_features = (
        pd.concat(all_feature_rows, ignore_index=True)
        if all_feature_rows
        else pd.DataFrame()
    )

    print(f"[INFO] Saving {CELL_TYPE} 3D labels: {LABELS_3D_TZYX_TIF}")
    tiff.imwrite(
        LABELS_3D_TZYX_TIF,
        labels_4d.astype(np.uint16),
        imagej=True,
        metadata={"axes": "TZYX"},
    )

    print(f"[INFO] Saving reconstruction table: {RECONSTRUCTION_CSV}")
    reconstruction_table.to_csv(RECONSTRUCTION_CSV, index=False)

    print(f"[INFO] Saving object features: {OBJECT_FEATURES_CSV}")
    object_features.to_csv(OBJECT_FEATURES_CSV, index=False)

    print("=" * 60)
    print("[DONE] 3D reconstruction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()