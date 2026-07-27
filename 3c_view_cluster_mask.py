import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import napari

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    RAW_TCZYX_TIF,
    LABELS_3D_TZYX_TIF,
    PROJECT_DIR,
)


# Override this when viewing a mask from a different output directory.
CLUSTER_MASK_TYX_TIF = Path(os.environ.get(
    "CLUSTER_MASK_TYX_TIF",
    PROJECT_DIR / f"{CELL_TYPE}_cluster_tracking_exclusion_mask_TYX.tif",
)).expanduser().resolve()

OBJECT_FEATURES_WITH_CLUSTER_CSV = (
    PROJECT_DIR / f"{CELL_TYPE}_3d_object_features_with_cluster_flags.csv"
)
TRACKS_CSV_TO_VIEW = (
    PROJECT_DIR
    / "macrophage_tracking_outputs_all"
    / "macrophage_tracks_lap.csv"
)

def load_raw_tzyx():
    raw = tiff.imread(RAW_TCZYX_TIF)

    if raw.ndim == 5:
        raw_tzyx = raw[:, CHANNEL_INDEX, :, :, :]
    elif raw.ndim == 4:
        raw_tzyx = raw
    else:
        raise ValueError(f"Unexpected raw shape: {raw.shape}")

    return raw_tzyx


def main():
    print("=" * 70)
    print("VIEW CLUSTER MASK IN NAPARI")
    print("=" * 70)

    print(f"Cell type:      {CELL_TYPE}")
    print(f"Raw:            {RAW_TCZYX_TIF}")
    print(f"3D labels:      {LABELS_3D_TZYX_TIF}")
    print(f"Cluster mask:   {CLUSTER_MASK_TYX_TIF}")
    print(f"Cluster table:  {OBJECT_FEATURES_WITH_CLUSTER_CSV}")

    raw_tzyx = load_raw_tzyx()
    labels_tzyx = tiff.imread(LABELS_3D_TZYX_TIF)
    cluster_tyx = tiff.imread(CLUSTER_MASK_TYX_TIF)

    print("Raw TZYX shape:      ", raw_tzyx.shape)
    print("Labels TZYX shape:   ", labels_tzyx.shape)
    print("Cluster TYX shape:   ", cluster_tyx.shape)

    if cluster_tyx.ndim != 3:
        raise ValueError(
            f"Expected cluster mask shape T,Y,X. Got {cluster_tyx.shape}"
        )

    # Z-projections for 2D time viewer
    raw_mip_tyx = raw_tzyx.max(axis=1)
    labels_mip_tyx = (labels_tzyx > 0).max(axis=1).astype(np.uint8)

    viewer = napari.Viewer(ndisplay=2)

    viewer.add_image(
        raw_mip_tyx,
        name=f"{CELL_TYPE} raw max projection",
        contrast_limits=[
            np.percentile(raw_mip_tyx, 1),
            np.percentile(raw_mip_tyx, 99.8),
        ],
    )

    viewer.add_labels(
        labels_mip_tyx,
        name=f"{CELL_TYPE} reconstructed label projection",
        opacity=0.25,
        visible=False,
    )

    viewer.add_labels(
        cluster_tyx.astype(np.uint16),
        name="detected cluster mask",
        opacity=0.45,
    )

    if TRACKS_CSV_TO_VIEW.exists():
        tracks = pd.read_csv(TRACKS_CSV_TO_VIEW)

        print("\n[NAPARI TRACK CHECK]")
        print(f"Loaded tracks from: {TRACKS_CSV_TO_VIEW}")
        print("Track rows:", len(tracks))
        print("Track IDs:", tracks["track_id"].nunique())

        if "cluster_region_class" in tracks.columns:
            print(tracks["cluster_region_class"].value_counts(dropna=False))

        required = {"track_id", "time", "centroid_y", "centroid_x"}

        if not required.issubset(tracks.columns):
            raise ValueError(
                f"Tracks file missing columns: {required - set(tracks.columns)}\n"
                f"Available columns: {list(tracks.columns)}"
            )

        # Napari tracks format for 2D + time:
        # track_id, time, y, x
        track_data = tracks[
            ["track_id", "time", "centroid_y", "centroid_x"]
        ].to_numpy(dtype=float)

        viewer.add_tracks(
            track_data,
            name="FINAL outside_only macrophage tracks",
            tail_length=1,
            tail_width=1,
            visible=False,   # hide lines by default
        )

        # Optional: also show final track points as dots
        track_points = tracks[
            ["time", "centroid_y", "centroid_x"]
        ].to_numpy(dtype=float)

        viewer.add_points(
            track_points,
            name="FINAL outside_only track points",
            size=3,
            ndim=3,
            opacity=0.9,
        )

    else:
        print(f"[WARNING] Track file not found: {TRACKS_CSV_TO_VIEW}")

    print("\nControls:")
    print("- Scroll time axis to inspect cluster over time.")
    print("- Toggle reconstructed label projection on/off.")
    print("- Toggle inside/boundary/outside centroid layers.")
    print("- The cluster mask is TYX, so this viewer uses Z-projections.")

    napari.run()


if __name__ == "__main__":
    main()