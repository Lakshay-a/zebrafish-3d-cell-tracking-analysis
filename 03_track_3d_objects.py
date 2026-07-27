"""Run multiple tracking algorithms on reconstructed 3D object features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile as tiff

from tracking_utils import (
    canonicalize_features,
    default_params,
    run_keyhole_tracker,
    run_lap_tracker,
    run_nearest_tracker,
    track_summary,
)

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    OBJECT_FEATURES_CSV,
    TRACKING_FEATURES_CSV,
    TRACKING_OUTPUT_DIR,
    TRACKING_METHODS,
    TRACKS_CSV,
    FILTERED_TRACKS_CSV,
    MAX_TIME_GAP,
    MAX_TRACK_XY_DISTANCE,
    MAX_TRACK_Z_DISTANCE,
    MAX_VOLUME_RATIO,
    Z_DISTANCE_WEIGHT,
    VOLUME_COST_WEIGHT,
    INTENSITY_COST_WEIGHT,
    KEYHOLE_FORWARD_DISTANCE,
    KEYHOLE_BACK_RADIUS,
    KEYHOLE_ANGLE_DEGREES,
    MIN_TRACK_LENGTH,
    ASSIGNMENT_COST_CUTOFF,
    MACROPHAGE_REGION_MODE,
    CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF,
    SPLIT_TRACKS_CROSSING_CLUSTER_MASK,
    CLUSTER_CROSSING_LINE_SAMPLES,
)



# Basic helpers


def configure_tracker_params():
    """Build tracker parameters from config.py."""
    params = default_params(CELL_TYPE)

    params.z_scale = Z_DISTANCE_WEIGHT

    params.max_link_distance = MAX_TRACK_XY_DISTANCE
    params.max_z_step = MAX_TRACK_Z_DISTANCE
    params.max_gap = MAX_TIME_GAP

    params.volume_weight = VOLUME_COST_WEIGHT
    params.intensity_weight = INTENSITY_COST_WEIGHT

    params.keyhole_forward_distance = KEYHOLE_FORWARD_DISTANCE
    params.keyhole_back_radius = KEYHOLE_BACK_RADIUS
    params.keyhole_angle_degrees = KEYHOLE_ANGLE_DEGREES

    params.min_track_length = MIN_TRACK_LENGTH

    try:
        params.assignment_cost_cutoff = ASSIGNMENT_COST_CUTOFF
    except NameError:
        params.assignment_cost_cutoff = None

    return params



def filter_short_tracks(tracks: pd.DataFrame, min_track_length: int) -> pd.DataFrame:
    """Optional simple track-length filter."""
    if tracks.empty:
        return tracks

    lengths = tracks.groupby("track_id")["time"].nunique()
    good_ids = lengths[lengths >= min_track_length].index

    return tracks[tracks["track_id"].isin(good_ids)].copy()



# Cluster mask / strict censor helpers


def should_apply_macrophage_cluster_censor() -> bool:
    """True only for macrophage region-aware tracking."""
    return CELL_TYPE == "macrophage" and MACROPHAGE_REGION_MODE != "all"


def load_cluster_exclusion_mask_for_tracking():
    """Load the macrophage tracking exclusion mask."""
    if not should_apply_macrophage_cluster_censor():
        return None

    if not SPLIT_TRACKS_CROSSING_CLUSTER_MASK:
        print(
            "[WARNING] SPLIT_TRACKS_CROSSING_CLUSTER_MASK=False. "
            "Point-level cluster censoring will still use cluster flags if present, "
            "but link-crossing checks are disabled."
        )

    if not CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF.exists():
        print(
            "[WARNING] Cluster tracking exclusion mask not found. "
            "Centroid and link-crossing cluster checks cannot be applied."
        )
        print(f"[WARNING] Missing: {CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF}")
        return None

    print("[INFO] Loading cluster tracking exclusion mask:")
    print(f"       {CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF}")

    mask = tiff.imread(CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF)

    if mask.ndim != 3:
        raise ValueError(
            "Expected cluster tracking exclusion mask with shape T,Y,X. "
            f"Got {mask.shape}"
        )

    mask = mask > 0

    print(f"[INFO] Cluster exclusion mask shape: {mask.shape}")
    print(f"[INFO] Cluster exclusion mask pixels total: {int(mask.sum())}")

    return mask


def _as_bool(value) -> bool:
    """Robust conversion for CSV-loaded booleans."""
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coordinate_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Find time, y and x coordinate columns."""
    time_col = _find_first_column(df, ["time", "t", "frame"])
    y_col = _find_first_column(df, ["centroid_y", "y", "Y"])
    x_col = _find_first_column(df, ["centroid_x", "x", "X"])

    missing = []
    if time_col is None:
        missing.append("time")
    if y_col is None:
        missing.append("centroid_y/y")
    if x_col is None:
        missing.append("centroid_x/x")

    if missing:
        raise ValueError(
            f"Cannot apply cluster mask censoring. Missing coordinate columns: {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )

    return time_col, y_col, x_col


def point_inside_exclusion_mask(row, exclusion_mask_tyx: np.ndarray | None, df: pd.DataFrame) -> bool:
    """Check whether a detection centroid lies inside the TYX cluster exclusion mask."""
    if exclusion_mask_tyx is None:
        return False

    time_col, y_col, x_col = _coordinate_columns(df)

    t = int(round(_safe_float(row[time_col], default=-1)))
    y = int(round(_safe_float(row[y_col], default=-1)))
    x = int(round(_safe_float(row[x_col], default=-1)))

    T, Y, X = exclusion_mask_tyx.shape

    if t < 0 or t >= T or y < 0 or y >= Y or x < 0 or x >= X:
        return False

    return bool(exclusion_mask_tyx[t, y, x])


def row_has_cluster_overlap_flags(row) -> bool:
    """True if cluster-detection flags say this row should not be used for."""
    # outside_only must contain only true outside_cluster rows.
    # outside_boundary may keep cluster_boundary rows, but never overlap rows.
    if "cluster_region_class" in row.index:
        region = str(row["cluster_region_class"]).strip().lower()

        if MACROPHAGE_REGION_MODE == "outside_only" and region != "outside_cluster":
            return True

        if region == "inside_cluster":
            return True

    if "inside_cluster" in row.index and _as_bool(row["inside_cluster"]):
        return True

    if "overlap_cluster_mask" in row.index and _as_bool(row["overlap_cluster_mask"]):
        return True

    if "cluster_overlap_pixels" in row.index:
        if _safe_float(row["cluster_overlap_pixels"], default=0.0) > 0:
            return True

    if "cluster_overlap_fraction" in row.index:
        if _safe_float(row["cluster_overlap_fraction"], default=0.0) > 0:
            return True

    return False


def xy_line_crosses_mask(
    y0,
    x0,
    y1,
    x1,
    mask_xy,
    n_samples=80,
):
    """Check whether the straight XY line between two detections crosses."""
    if mask_xy is None or mask_xy.sum() == 0:
        return False

    Y, X = mask_xy.shape

    ys = np.linspace(float(y0), float(y1), int(n_samples))
    xs = np.linspace(float(x0), float(x1), int(n_samples))

    ys = np.rint(ys).astype(int)
    xs = np.rint(xs).astype(int)

    valid = (
        (ys >= 0)
        & (ys < Y)
        & (xs >= 0)
        & (xs < X)
    )

    if not np.any(valid):
        return False

    return bool(mask_xy[ys[valid], xs[valid]].any())


def link_crosses_exclusion_mask(
    prev_row,
    row,
    exclusion_mask_tyx: np.ndarray | None,
    df: pd.DataFrame,
) -> bool:
    """Check whether the XY line between two consecutive detections crosses."""
    if exclusion_mask_tyx is None:
        return False

    if not SPLIT_TRACKS_CROSSING_CLUSTER_MASK:
        return False

    time_col, y_col, x_col = _coordinate_columns(df)

    t0 = int(round(_safe_float(prev_row[time_col], default=-1)))
    t1 = int(round(_safe_float(row[time_col], default=-1)))

    y0 = _safe_float(prev_row[y_col])
    x0 = _safe_float(prev_row[x_col])
    y1 = _safe_float(row[y_col])
    x1 = _safe_float(row[x_col])

    if not all(np.isfinite(v) for v in [y0, x0, y1, x1]):
        return False

    T = exclusion_mask_tyx.shape[0]

    t_start = max(0, min(t0, t1))
    t_end = min(T - 1, max(t0, t1))

    if t_start > t_end:
        return False

    for tt in range(t_start, t_end + 1):
        if xy_line_crosses_mask(
            y0,
            x0,
            y1,
            x1,
            exclusion_mask_tyx[tt],
            n_samples=CLUSTER_CROSSING_LINE_SAMPLES,
        ):
            return True

    return False


def print_cluster_columns_diagnostics(df: pd.DataFrame, label: str) -> None:
    """Print useful diagnostics for checking whether cluster flags survived."""
    print(f"[CHECK] {label} columns:")
    print(f"        {list(df.columns)}")

    if "cluster_region_class" in df.columns:
        print(f"[CHECK] {label} cluster_region_class counts:")
        print(df["cluster_region_class"].value_counts(dropna=False))

    if "inside_cluster" in df.columns:
        print(f"[CHECK] {label} inside_cluster true count:")
        print(int(df["inside_cluster"].map(_as_bool).sum()))

    if "overlap_cluster_mask" in df.columns:
        print(f"[CHECK] {label} overlap_cluster_mask true count:")
        print(int(df["overlap_cluster_mask"].map(_as_bool).sum()))

    if "cluster_overlap_pixels" in df.columns:
        vals = pd.to_numeric(df["cluster_overlap_pixels"], errors="coerce").fillna(0)
        print(f"[CHECK] {label} rows with cluster_overlap_pixels > 0:")
        print(int((vals > 0).sum()))


def strict_filter_features_before_tracking(
    features: pd.DataFrame,
    exclusion_mask_tyx: np.ndarray | None,
) -> pd.DataFrame:
    """Remove cluster detections before running LAP/nearest/keyhole."""
    if not should_apply_macrophage_cluster_censor():
        return features

    if features.empty:
        return features

    print_cluster_columns_diagnostics(features, "tracking input BEFORE strict cluster censor")

    before = len(features)
    remove_by_flags = np.zeros(before, dtype=bool)
    remove_by_centroid = np.zeros(before, dtype=bool)

    for i, (_, row) in enumerate(features.iterrows()):
        if row_has_cluster_overlap_flags(row):
            remove_by_flags[i] = True

        if point_inside_exclusion_mask(row, exclusion_mask_tyx, features):
            remove_by_centroid[i] = True

    remove = remove_by_flags | remove_by_centroid
    out = features.loc[~remove].copy()

    print("[INFO] Strict pre-tracking cluster censor complete")
    print(f"       Input detections:          {before}")
    print(f"       Removed by cluster flags:  {int(remove_by_flags.sum())}")
    print(f"       Removed by centroid mask:  {int(remove_by_centroid.sum())}")
    print(f"       Total removed:             {int(remove.sum())}")
    print(f"       Remaining detections:      {len(out)}")

    if out.empty:
        raise RuntimeError(
            "Strict cluster censoring removed all macrophage detections. "
            "Check MACROPHAGE_REGION_MODE and the cluster exclusion mask."
        )

    print_cluster_columns_diagnostics(out, "tracking input AFTER strict cluster censor")

    return out


def stop_tracks_at_cluster_mask(
    tracks: pd.DataFrame,
    exclusion_mask_tyx: np.ndarray | None,
) -> pd.DataFrame:
    """Strict macrophage-only cluster censoring after tracking."""
    if not should_apply_macrophage_cluster_censor():
        return tracks

    if tracks.empty:
        return tracks

    if "track_id" not in tracks.columns:
        raise ValueError(
            "Cannot stop tracks at cluster mask. Tracks table is missing 'track_id'."
        )

    _coordinate_columns(tracks)  # validates time/y/x columns are available

    print_cluster_columns_diagnostics(tracks, "tracks BEFORE strict cluster stop")

    tracks = tracks.sort_values(["track_id", "time"]).copy()

    new_rows = []
    next_track_id = 1

    n_input_rows = len(tracks)
    n_removed_by_flags = 0
    n_removed_by_centroid = 0
    n_stopped_links = 0
    n_kept = 0

    for _, group in tracks.groupby("track_id", sort=False):
        group = group.sort_values("time")

        current_track_id = next_track_id
        next_track_id += 1

        previous_kept_row = None

        for _, row in group.iterrows():
            remove_this = False

            # Rule 1: trust cluster overlap flags from 3b_detect_macrophage_clusters.py.
            if row_has_cluster_overlap_flags(row):
                n_removed_by_flags += 1
                remove_this = True

            # Rule 2: also remove if the centroid is inside the TYX exclusion mask.
            if point_inside_exclusion_mask(row, exclusion_mask_tyx, tracks):
                n_removed_by_centroid += 1
                remove_this = True

            if remove_this:
                previous_kept_row = None

                # Any future outside detection must become a new independent track.
                current_track_id = next_track_id
                next_track_id += 1
                continue

            # Rule 3: if the link from the previous kept outside point crosses
            # the cluster mask, stop the previous track and begin a new one here.
            if previous_kept_row is not None:
                if link_crosses_exclusion_mask(
                    previous_kept_row,
                    row,
                    exclusion_mask_tyx,
                    tracks,
                ):
                    n_stopped_links += 1
                    current_track_id = next_track_id
                    next_track_id += 1

            new_row = row.copy()
            new_row["track_id"] = current_track_id
            new_rows.append(new_row)

            previous_kept_row = row
            n_kept += 1

    if len(new_rows) == 0:
        print("[WARNING] Strict cluster censoring removed all track detections.")
        return tracks.iloc[0:0].copy()

    out = pd.DataFrame(new_rows)

    print("[INFO] Strict cluster stop/censor complete")
    print(f"       Input detections:          {n_input_rows}")
    print(f"       Kept detections:           {n_kept}")
    print(f"       Removed by cluster flags:  {n_removed_by_flags}")
    print(f"       Removed by centroid mask:  {n_removed_by_centroid}")
    print(f"       Links stopped at cluster:  {n_stopped_links}")
    print(f"       Tracks after censoring:    {out['track_id'].nunique()}")

    validate_no_track_points_inside_cluster(out, exclusion_mask_tyx)

    return out


def validate_no_track_points_inside_cluster(
    tracks: pd.DataFrame,
    exclusion_mask_tyx: np.ndarray | None,
) -> None:
    """Final sanity check: no kept track centroid should lie inside the TYX mask,."""
    if not should_apply_macrophage_cluster_censor():
        return

    if tracks.empty:
        return

    bad_by_flags = 0
    bad_by_centroid = 0

    for _, row in tracks.iterrows():
        if row_has_cluster_overlap_flags(row):
            bad_by_flags += 1

        if point_inside_exclusion_mask(row, exclusion_mask_tyx, tracks):
            bad_by_centroid += 1

    print("[CHECK] Final cluster-censor validation")
    print(f"        Rows still bad by cluster flags: {bad_by_flags}")
    print(f"        Rows still inside mask centroid: {bad_by_centroid}")

    if bad_by_flags > 0 or bad_by_centroid > 0:
        raise RuntimeError(
            "Strict cluster censoring failed: some track rows are still inside/overlapping "
            "the cluster exclusion mask. Do not use these tracks until this is fixed."
        )



# Main


def main() -> None:
    TRACKING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3D TRACKING BENCHMARK")
    print("=" * 60)
    print(f"Cell type:              {CELL_TYPE}")
    print(f"Channel index:          {CHANNEL_INDEX}")
    print(f"Input features:         {TRACKING_FEATURES_CSV}")
    print(f"Output folder:          {TRACKING_OUTPUT_DIR}")
    print(f"Tracking methods:       {TRACKING_METHODS}")
    print(f"Default tracks output:  {TRACKS_CSV}")

    if CELL_TYPE == "macrophage":
        print(f"Macrophage region mode: {MACROPHAGE_REGION_MODE}")
        print(f"Cluster mask:           {CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF}")

    print()

    if not TRACKING_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Tracking features CSV not found:\n{TRACKING_FEATURES_CSV}\n"
            "Run 02_reconstruct_3d_objects.py and 3b_detect_macrophage_clusters.py first."
        )

    features = pd.read_csv(TRACKING_FEATURES_CSV)
    features = canonicalize_features(features)

    if "cell_type" not in features.columns:
        features["cell_type"] = CELL_TYPE

    cluster_exclusion_mask_tyx = load_cluster_exclusion_mask_for_tracking()

    # Critical change: remove cluster detections BEFORE the tracker sees them.
    features = strict_filter_features_before_tracking(
        features,
        cluster_exclusion_mask_tyx,
    )

    params = configure_tracker_params()

    print("[INFO] Feature table loaded")
    print(f"Timepoints:             {features['time'].nunique()}")
    print(f"Objects used:           {len(features)}")
    print(f"Tracker parameters:     {params}")
    print()

    # Save the exact table used for tracking for debugging/reproducibility.
    if CELL_TYPE == "macrophage":
        used_features_path = (
            TRACKING_OUTPUT_DIR
            / f"{CELL_TYPE}_tracking_input_used_{MACROPHAGE_REGION_MODE}.csv"
        )
        features.to_csv(used_features_path, index=False)
        print(f"[SAVED] Actual tracking input used: {used_features_path}")
        print()

    default_tracks_saved = False

    for method in TRACKING_METHODS:
        print("-" * 60)
        print(f"Running tracker: {method}")
        print("-" * 60)

        if method == "nearest":
            tracks = run_nearest_tracker(features, params)

        elif method == "lap":
            tracks = run_lap_tracker(features, params)

        elif method == "keyhole":
            tracks = run_keyhole_tracker(features, params)

        else:
            raise ValueError(f"Unknown tracking method: {method}")

        # Critical change: strict post-tracking cluster stop/censor.
        # This removes bad detections and breaks tracks at the cluster boundary.
        tracks = stop_tracks_at_cluster_mask(
            tracks,
            cluster_exclusion_mask_tyx,
        )

        if tracks.empty:
            print(f"[WARNING] {method} returned no usable tracks after cluster censoring.")
            continue

        tracks["cell_type"] = CELL_TYPE
        tracks["tracking_method"] = method

        tracks_path = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{method}.csv"
        summary_path = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_track_summary_{method}.csv"

        tracks.to_csv(tracks_path, index=False)
        track_summary(tracks).to_csv(summary_path, index=False)

        filtered_tracks = filter_short_tracks(
            tracks,
            min_track_length=MIN_TRACK_LENGTH,
        )

        filtered_path = (
            TRACKING_OUTPUT_DIR
            / f"{CELL_TYPE}_tracks_{method}_filtered_minlen{MIN_TRACK_LENGTH}.csv"
        )
        filtered_tracks.to_csv(filtered_path, index=False)

        n_tracks = tracks["track_id"].nunique()
        n_filtered_tracks = filtered_tracks["track_id"].nunique()

        print(f"[SAVED] Tracks:          {tracks_path}")
        print(f"[SAVED] Summary:         {summary_path}")
        print(f"[SAVED] Filtered tracks: {filtered_path}")
        print(f"[INFO] Tracks:           {n_tracks}")
        print(f"[INFO] Filtered tracks:  {n_filtered_tracks}")
        print()

        # Save keyhole as the default downstream track file.
        # If keyhole is not being run, save the first completed method.
        if method == "keyhole":
            tracks.to_csv(TRACKS_CSV, index=False)
            filtered_tracks.to_csv(FILTERED_TRACKS_CSV, index=False)
            default_tracks_saved = True

            print(f"[SAVED] Default tracks:          {TRACKS_CSV}")
            print(f"[SAVED] Default filtered tracks: {FILTERED_TRACKS_CSV}")

        elif not default_tracks_saved:
            tracks.to_csv(TRACKS_CSV, index=False)
            filtered_tracks.to_csv(FILTERED_TRACKS_CSV, index=False)
            default_tracks_saved = True

            print(f"[SAVED] Default tracks:          {TRACKS_CSV}")
            print(f"[SAVED] Default filtered tracks: {FILTERED_TRACKS_CSV}")

    print("=" * 60)
    print("[DONE] 3D tracking benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
