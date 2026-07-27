from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    RAW_TCZYX_TIF,
    LABELS_3D_TZYX_TIF,
    TRACKING_OUTPUT_DIR,
    CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF,
)



# User settings


N_GT_TRACKS = 10

OUTPUT_CSV = (
    TRACKING_OUTPUT_DIR
    / f"{CELL_TYPE}_manual_ground_truth_tracks.csv"
)

RAW_CONTRAST_PERCENTILES = (0.5, 99.8)

SHOW_LABELS = True
SHOW_CLUSTER_EXCLUSION_MASK = CELL_TYPE == "macrophage"



# Helper functions


def load_raw_as_tzyx(
    raw_path: Path,
    channel_index: int,
) -> np.ndarray:
    """Load the raw image and return a single-channel TZYX stack."""
    raw = tifffile.imread(raw_path)

    print(f"Loaded raw shape: {raw.shape}")

    if raw.ndim == 5:
        if not 0 <= channel_index < raw.shape[1]:
            raise ValueError(
                f"CHANNEL_INDEX={channel_index} is invalid for "
                f"raw shape {raw.shape}."
            )

        raw_tzyx = raw[:, channel_index, :, :, :]

        print(
            f"Selected channel {channel_index}. "
            f"New shape TZYX: {raw_tzyx.shape}"
        )

    elif raw.ndim == 4:
        raw_tzyx = raw
        print(f"Raw already appears to be TZYX: {raw_tzyx.shape}")

    else:
        raise ValueError(
            f"Unexpected raw image shape {raw.shape}. "
            "Expected TCZYX or TZYX."
        )

    return raw_tzyx


def load_cluster_mask_as_tzyx(
    mask_path: Path,
    raw_tzyx_shape: tuple,
) -> np.ndarray:
    """Load a cluster mask saved as TYX and expand it to TZYX."""
    cluster_tyx = tifffile.imread(mask_path)

    print(f"Loaded cluster mask shape: {cluster_tyx.shape}")

    if cluster_tyx.ndim != 3:
        raise ValueError(
            f"Expected cluster mask shape TYX. "
            f"Got {cluster_tyx.shape}"
        )

    t_raw, z_raw, y_raw, x_raw = raw_tzyx_shape
    t_mask, y_mask, x_mask = cluster_tyx.shape

    if (t_mask, y_mask, x_mask) != (t_raw, y_raw, x_raw):
        raise ValueError(
            "Cluster mask TYX does not match raw T/Y/X dimensions.\n"
            f"Raw TZYX:    {raw_tzyx_shape}\n"
            f"Cluster TYX: {cluster_tyx.shape}"
        )

    cluster_tzyx = np.repeat(
        cluster_tyx[:, np.newaxis, :, :],
        z_raw,
        axis=1,
    )

    return cluster_tzyx.astype(np.uint8)


def load_existing_gt_points(
    output_csv: Path,
) -> dict[str, np.ndarray]:
    """Load an existing manual GT CSV."""
    if not output_csv.exists():
        print("[INFO] No existing GT CSV found.")
        print("[INFO] Starting with empty GT layers.")
        return {}

    try:
        df = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
        print(
            "[WARNING] Existing GT CSV is completely empty. "
            "Starting with empty GT layers."
        )
        return {}

    if df.empty:
        print(
            "[WARNING] Existing GT CSV contains no rows. "
            "Starting with empty GT layers."
        )
        return {}

    required_columns = {
        "gt_track_id",
        "time",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Existing GT CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Avoid loading points belonging to another cell type.
    if "cell_type" in df.columns:
        matching_cell_type = (
            df["cell_type"]
            .astype(str)
            .str.lower()
            .str.strip()
            == CELL_TYPE.lower().strip()
        )

        n_mismatched = int((~matching_cell_type).sum())

        if n_mismatched > 0:
            print(
                f"[WARNING] Ignoring {n_mismatched} rows whose "
                f"cell_type is not '{CELL_TYPE}'."
            )

        df = df[matching_cell_type].copy()

    if df.empty:
        print(
            f"[WARNING] No existing rows match CELL_TYPE='{CELL_TYPE}'."
        )
        return {}

    numeric_columns = [
        "time",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    invalid_rows = df[numeric_columns].isna().any(axis=1)

    if invalid_rows.any():
        print(
            f"[WARNING] Ignoring {int(invalid_rows.sum())} existing "
            "rows with invalid coordinates."
        )
        df = df[~invalid_rows].copy()

    existing_points = {}

    for gt_track_id, group in df.groupby(
        "gt_track_id",
        sort=True,
    ):
        gt_track_id = str(gt_track_id).strip()

        if not gt_track_id.startswith("GT"):
            print(
                f"[WARNING] Ignoring unexpected track ID: "
                f"{gt_track_id}"
            )
            continue

        group = group.sort_values("time")

        points_tzyx = group[
            [
                "time",
                "centroid_z",
                "centroid_y",
                "centroid_x",
            ]
        ].to_numpy(dtype=float)

        existing_points[gt_track_id] = points_tzyx

    total_points = sum(
        len(points)
        for points in existing_points.values()
    )

    print("\nLoaded existing manual ground truth:")
    print(output_csv)
    print(f"Tracks: {len(existing_points)}")
    print(f"Rows:   {total_points}")

    for gt_track_id, points in existing_points.items():
        print(f"  {gt_track_id}: {len(points)} points")

    return existing_points


def gt_layer_sort_key(layer_name: str):
    """Sort GT001, GT002, ... ."""
    try:
        return int(layer_name.removeprefix("GT"))
    except ValueError:
        return float("inf")


def export_gt_points(
    viewer: napari.Viewer,
    output_csv: Path,
) -> bool:
    """Export all Points layers named GT001, GT002, ... ."""
    rows = []

    for layer in viewer.layers:
        if not isinstance(layer, napari.layers.Points):
            continue

        if not layer.name.startswith("GT"):
            continue

        gt_track_id = layer.name
        data = np.asarray(layer.data)

        if data.size == 0:
            continue

        if data.ndim != 2 or data.shape[1] != 4:
            print(
                f"[WARNING] Skipping {gt_track_id}; "
                f"expected points with shape (N, 4), got {data.shape}."
            )
            continue

        for point in data:
            t, z, y, x = point

            rows.append(
                {
                    "cell_type": CELL_TYPE,
                    "gt_track_id": gt_track_id,
                    "time": int(round(float(t))),
                    "centroid_z": float(z),
                    "centroid_y": float(y),
                    "centroid_x": float(x),
                    "visibility": "visible",
                    "note": "",
                }
            )

    df = pd.DataFrame(rows)

    # Critical protection against replacing valid GT with an empty file.
    if df.empty:
        print("\n[WARNING] No GT points are currently present.")

        if output_csv.exists():
            print("[WARNING] Existing CSV was NOT overwritten:")
            print(output_csv)
        else:
            print("[WARNING] No CSV was created.")

        return False

    df = df.sort_values(
        ["gt_track_id", "time"]
    ).reset_index(drop=True)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write through a temporary file to reduce the risk of a
    # partially written CSV.
    temporary_csv = output_csv.with_suffix(
        output_csv.suffix + ".tmp"
    )

    df.to_csv(
        temporary_csv,
        index=False,
    )

    temporary_csv.replace(output_csv)

    print("\nSaved manual ground truth:")
    print(output_csv)
    print(f"Rows: {len(df)}")
    print(df.groupby("gt_track_id")["time"].count())

    return True



# Main


def main():
    print("\n==============================")
    print("MANUAL GROUND TRUTH TRACKING")
    print("==============================")
    print(f"Cell type:   {CELL_TYPE}")
    print(f"Raw file:    {RAW_TCZYX_TIF}")
    print(f"Labels file: {LABELS_3D_TZYX_TIF}")
    print(f"Output CSV:  {OUTPUT_CSV}")

    raw_tzyx = load_raw_as_tzyx(
        RAW_TCZYX_TIF,
        CHANNEL_INDEX,
    )


    # Load cluster exclusion mask


    cluster_tzyx = None

    if (
        SHOW_CLUSTER_EXCLUSION_MASK
        and CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF.exists()
    ):
        cluster_tzyx = load_cluster_mask_as_tzyx(
            CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF,
            raw_tzyx.shape,
        )

    else:
        print("[INFO] Cluster mask not shown.")
        print(
            "Expected path: "
            f"{CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF}"
        )


    # Load 3D labels


    labels_tzyx = None

    if SHOW_LABELS and LABELS_3D_TZYX_TIF.exists():
        labels_tzyx = tifffile.imread(
            LABELS_3D_TZYX_TIF
        )

        print(
            f"Loaded labels shape: "
            f"{labels_tzyx.shape}"
        )

        if labels_tzyx.shape != raw_tzyx.shape:
            print(
                "[WARNING] Raw and label shapes "
                "do not match exactly."
            )
            print(f"Raw TZYX:    {raw_tzyx.shape}")
            print(f"Labels TZYX: {labels_tzyx.shape}")

    elif SHOW_LABELS:
        print(
            "[WARNING] Labels file does not exist:"
        )
        print(LABELS_3D_TZYX_TIF)


    # Load existing manual GT CSV


    existing_gt_points = load_existing_gt_points(
        OUTPUT_CSV
    )


    # Create Napari viewer


    p_low, p_high = np.percentile(
        raw_tzyx,
        RAW_CONTRAST_PERCENTILES,
    )

    viewer = napari.Viewer(ndisplay=2)

    viewer.add_image(
        raw_tzyx,
        name=f"raw_{CELL_TYPE}_TZYX",
        contrast_limits=(p_low, p_high),
        colormap="gray",
        blending="additive",
    )

    if cluster_tzyx is not None:
        viewer.add_labels(
            cluster_tzyx,
            name="cluster_exclusion_mask_TZYX",
            opacity=0.25,
        )

    if labels_tzyx is not None:
        viewer.add_labels(
            labels_tzyx,
            name=f"{CELL_TYPE}_3d_labels_TZYX",
            opacity=0.35,
        )

    try:
        viewer.dims.axis_labels = (
            "time",
            "z",
            "y",
            "x",
        )
    except Exception:
        pass


    # Create or restore GT point layers


    default_layer_names = {
        f"GT{i:03d}"
        for i in range(1, N_GT_TRACKS + 1)
    }

    # Also restore any existing layers beyond N_GT_TRACKS.
    all_gt_layer_names = (
        default_layer_names
        | set(existing_gt_points.keys())
    )

    all_gt_layer_names = sorted(
        all_gt_layer_names,
        key=gt_layer_sort_key,
    )

    for gt_layer_name in all_gt_layer_names:
        layer_data = existing_gt_points.get(
            gt_layer_name,
            np.empty((0, 4), dtype=float),
        )

        viewer.add_points(
            data=layer_data,
            ndim=4,
            name=gt_layer_name,
            size=8,
            face_color="red",
            opacity=1.0,
        )


    # Save shortcut


    @viewer.bind_key("Control-S")
    def save_gt_points(viewer):
        export_gt_points(
            viewer,
            OUTPUT_CSV,
        )

    print("\nInstructions:")
    print("1. Select a GT layer, e.g. GT001.")
    print("2. Select the point-add tool.")
    print("3. Go to the correct time and Z plane.")
    print("4. Click the centre of the same biological cell.")
    print("5. Move to the next timepoint and repeat.")
    print("6. Press Control-S to save the CSV.")
    print()
    print("Use one GT layer per manually tracked cell.")
    print(
        "Existing CSV points are automatically restored "
        "when the script is reopened."
    )
    print(
        "An existing CSV will not be overwritten when "
        "all GT layers are empty."
    )

    napari.run()


if __name__ == "__main__":
    main()