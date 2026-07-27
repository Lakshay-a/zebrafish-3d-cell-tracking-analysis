import numpy as np
import pandas as pd
import tifffile as tiff
import napari

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    RAW_TCZYX_TIF,
    LABELS_3D_TZYX_TIF,
    TRACKING_OUTPUT_DIR,
    VIEW_TRACKING_METHOD,
    MACROPHAGE_REGION_MODE,
    CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF,
)

FIND_TRACK_ID = None  # change this to the track you want to inspect
def load_selected_raw_channel():
    """Load shared raw stack and return selected channel as T,Z,Y,X."""
    print(f"[INFO] Loading raw stack: {RAW_TCZYX_TIF}")
    raw = tiff.imread(RAW_TCZYX_TIF)

    print(f"[INFO] Raw shape: {raw.shape}")

    if raw.ndim == 5:
        # T,c,z,y,x
        T, C, Z, Y, X = raw.shape

        if CHANNEL_INDEX >= C:
            raise ValueError(
                f"CHANNEL_INDEX={CHANNEL_INDEX}, but raw has only {C} channels. "
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

def focus_on_track(viewer, tracks_df, track_id):
    """Jump Napari view to a specific track_id and add a highlight points layer."""

    if "track_id" not in tracks_df.columns:
        raise ValueError("tracks_df must contain a 'track_id' column.")

    one_track = tracks_df[tracks_df["track_id"] == track_id].copy()

    if len(one_track) == 0:
        print(f"[WARNING] Track ID {track_id} not found.")
        return

    one_track = one_track.sort_values("time")

    # choose middle point in the track
    mid_row = one_track.iloc[len(one_track) // 2]

    t = int(round(mid_row["time"]))
    z = int(round(mid_row["centroid_z"]))
    y = float(mid_row["centroid_y"])
    x = float(mid_row["centroid_x"])

    print("\n==============================")
    print(f"FOCUSING ON TRACK {track_id}")
    print("==============================")
    print(f"Track length: {len(one_track)}")
    print(f"Time range:   {one_track['time'].min()} → {one_track['time'].max()}")
    print(f"Z range:      {one_track['centroid_z'].min():.2f} → {one_track['centroid_z'].max():.2f}")
    print(f"Jumping to:   T={t}, Z={z}, Y={y:.1f}, X={x:.1f}")

    # Set time and Z sliders
    # Assumes viewer dimensions are T, Z, Y, X
    viewer.dims.set_current_step(0, t)
    viewer.dims.set_current_step(1, z)

    # Center camera approximately on the object
    try:
        viewer.camera.center = (z, y, x)
    except Exception:
        try:
            viewer.camera.center = (y, x)
        except Exception:
            pass

    # Add highlighted points for this track
    points_data = one_track[["time", "centroid_z", "centroid_y", "centroid_x"]].to_numpy()

    if f"highlight_track_{track_id}" in viewer.layers:
        viewer.layers.remove(f"highlight_track_{track_id}")

    highlight_layer = viewer.add_points(
        points_data,
        name=f"highlight_track_{track_id}",
        size=10,
        face_color="red",
        opacity=0.9,
    )

    print(f"[INFO] Added highlight layer: highlight_track_{track_id}")

def main():
    print("=" * 60)
    print(f"NAPARI TRACK VIEWER: {CELL_TYPE}")
    print("=" * 60)

    raw = load_selected_raw_channel()

    print(f"[INFO] Loading 3D labels: {LABELS_3D_TZYX_TIF}")
    labels = tiff.imread(LABELS_3D_TZYX_TIF)

    print(f"[INFO] Label shape: {labels.shape}")

    if labels.shape != raw.shape:
        raise ValueError(
            f"Raw and label shape mismatch. raw={raw.shape}, labels={labels.shape}"
        )

    tracks_csv = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{VIEW_TRACKING_METHOD}_good_filtered.csv"
    # tracks_csv = TRACKING_OUTPUT_DIR /.
    # tracks_csv = TRACKING_OUTPUT_DIR /.

    print(f"[INFO] Loading tracks: {tracks_csv}")
    print(f"[CHECK] MACROPHAGE_REGION_MODE = {MACROPHAGE_REGION_MODE}")
    print(f"[CHECK] TRACKING_OUTPUT_DIR = {TRACKING_OUTPUT_DIR}")
    print(f"[CHECK] tracks_csv = {tracks_csv}")
    if not tracks_csv.exists():
        raise FileNotFoundError(
            f"Tracks file not found:\n{tracks_csv}\n"
            "Run 03_track_3d_objects.py first, or change VIEW_TRACKING_METHOD in config.py."
        )

    tracks_df = pd.read_csv(tracks_csv)

    viewer = napari.Viewer()

    viewer.add_image(
        raw,
        name=f"raw_{CELL_TYPE}_TZYX",
        contrast_limits=[
            np.percentile(raw, 1),
            np.percentile(raw, 99.8),
        ],
    )

    viewer.add_labels(
        labels,
        name=f"3D_reconstructed_{CELL_TYPE}_labels",
    )

    

    if not tracks_df.empty:
        required_cols = [
            "track_id",
            "time",
            "centroid_z",
            "centroid_y",
            "centroid_x",
        ]

        missing = [c for c in required_cols if c not in tracks_df.columns]

        if missing:
            raise ValueError(
                f"Tracks file is missing required columns: {missing}. "
                f"Available columns: {list(tracks_df.columns)}"
            )

        tracks_np = tracks_df[
            [
                "track_id",
                "time",
                "centroid_z",
                "centroid_y",
                "centroid_x",
            ]
        ].dropna().to_numpy(dtype=float)

        viewer.add_tracks(
            tracks_np,
            name=f"{CELL_TYPE}_tracks_{VIEW_TRACKING_METHOD}",
        )
        viewer.add_points(
            tracks_df[["time", "centroid_z", "centroid_y", "centroid_x"]].to_numpy(),
            name="TRACKED_CENTROID_POINTS_ONLY",
            size=4,
            face_color="red",
            opacity=0.9,
        )

    else:
        print("[WARNING] Tracks CSV is empty. Showing raw and labels only.")

    if FIND_TRACK_ID is not None:
        focus_on_track(viewer, tracks_df, FIND_TRACK_ID)

    napari.run()


if __name__ == "__main__":
    main()