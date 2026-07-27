import numpy as np
import pandas as pd

from config import (
    CELL_TYPE,
    TRACKING_OUTPUT_DIR,
    FEATURE_TRACKING_METHOD,
    TRACK_FEATURES_CSV,
    Z_DISTANCE_WEIGHT,
    STATIC_DISPLACEMENT_THRESHOLD,
    LONG_TRACK_LENGTH,
    VERY_LONG_TRACK_LENGTH,
    SUSPICIOUS_JUMP_THRESHOLD,
)

# Optional split/merge settings. If these are not yet present in config.py,
# the script will still run using the defaults below.
try:
    from config import (
        ENABLE_SPLIT_MERGE_FLAGS,
        MAX_VOLUME_FOLD_CHANGE_FOR_SPLIT_MERGE,
        MAX_INTENSITY_FOLD_CHANGE_FOR_SPLIT_MERGE,
        MAX_VOLUME_CV_FOR_SPLIT_MERGE,
        MAX_STEP_TO_MEDIAN_RATIO_FOR_SPLIT_MERGE,
        MIN_ABSOLUTE_STEP_FOR_SPLIT_MERGE,
    )
except ImportError:
    ENABLE_SPLIT_MERGE_FLAGS = True
    MAX_VOLUME_FOLD_CHANGE_FOR_SPLIT_MERGE = 4.0
    MAX_INTENSITY_FOLD_CHANGE_FOR_SPLIT_MERGE = 4.0
    MAX_VOLUME_CV_FOR_SPLIT_MERGE = 1.0
    MAX_STEP_TO_MEDIAN_RATIO_FOR_SPLIT_MERGE = 8.0
    MIN_ABSOLUTE_STEP_FOR_SPLIT_MERGE = 10.0


# Trajectory-analysis reference: https://pubmed.ncbi.nlm.nih.gov/27713081/
def add_step_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add frame-to-frame 3D movement features for one track."""
    g = g.sort_values("time").copy()

    g["prev_z"] = g["centroid_z"].shift(1)
    g["prev_y"] = g["centroid_y"].shift(1)
    g["prev_x"] = g["centroid_x"].shift(1)
    g["prev_time"] = g["time"].shift(1)

    dz = (g["centroid_z"] - g["prev_z"]) * Z_DISTANCE_WEIGHT
    dy = g["centroid_y"] - g["prev_y"]
    dx = g["centroid_x"] - g["prev_x"]

    g["step_distance_3d"] = np.sqrt(dx**2 + dy**2 + dz**2)
    g["step_distance_xy"] = np.sqrt(dx**2 + dy**2)
    g["step_z_slices"] = g["centroid_z"] - g["prev_z"]
    g["time_gap"] = g["time"] - g["prev_time"]

    return g


def add_split_merge_flags_to_quality_features(
    tracks_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    z_distance_weight: float,
    max_allowed_volume_fold_change: float = 4.0,
    max_allowed_intensity_fold_change: float = 4.0,
    max_allowed_volume_cv: float = 1.0,
    max_allowed_step_to_median_ratio: float = 8.0,
    min_absolute_step: float = 10.0,
) -> pd.DataFrame:
    """Adds split/merge warning columns to track-level quality features."""

    required_cols = ["track_id", "time", "centroid_z", "centroid_y", "centroid_x"]
    missing = [c for c in required_cols if c not in tracks_df.columns]
    if missing:
        raise ValueError(f"tracks_df missing required columns: {missing}")

    has_volume = "volume_voxels" in tracks_df.columns
    has_intensity = "mean_intensity" in tracks_df.columns

    eps = 1e-6
    flag_rows = []

    for track_id, g in tracks_df.groupby("track_id"):
        g = g.sort_values("time").copy()
        reasons = []


        # 1. Step-distance spike

        if len(g) >= 2:
            z = g["centroid_z"].to_numpy(dtype=float) * float(z_distance_weight)
            y = g["centroid_y"].to_numpy(dtype=float)
            x = g["centroid_x"].to_numpy(dtype=float)

            dz = np.diff(z)
            dy = np.diff(y)
            dx = np.diff(x)
            steps = np.sqrt(dx**2 + dy**2 + dz**2)

            max_step = float(np.nanmax(steps)) if len(steps) else 0.0
            median_step = float(np.nanmedian(steps)) if len(steps) else 0.0

            if median_step <= eps:
                step_to_median_ratio = np.nan
                step_spike = False
            else:
                step_to_median_ratio = max_step / median_step
                step_spike = (
                    step_to_median_ratio > max_allowed_step_to_median_ratio
                    and max_step > min_absolute_step
                )
        else:
            max_step = 0.0
            median_step = 0.0
            step_to_median_ratio = np.nan
            step_spike = False

        if step_spike:
            reasons.append("step_spike")


        # 2. Volume jump / volume instability

        if has_volume:
            volumes = g["volume_voxels"].to_numpy(dtype=float)
            volumes = volumes[np.isfinite(volumes)]

            if len(volumes) >= 2 and np.nanmean(volumes) > eps:
                track_volume_cv = float(np.nanstd(volumes) / (np.nanmean(volumes) + eps))

                v1 = volumes[:-1]
                v2 = volumes[1:]
                valid = (v1 > eps) & (v2 > eps)

                if valid.any():
                    volume_ratios = np.maximum(
                        v2[valid] / v1[valid],
                        v1[valid] / v2[valid],
                    )
                    max_observed_volume_fold_change = float(np.nanmax(volume_ratios))
                else:
                    max_observed_volume_fold_change = np.nan
            else:
                track_volume_cv = np.nan
                max_observed_volume_fold_change = np.nan

            volume_jump = (
                np.isfinite(max_observed_volume_fold_change)
                and max_observed_volume_fold_change > max_allowed_volume_fold_change
            )

            high_volume_cv = (
                np.isfinite(track_volume_cv)
                and track_volume_cv > max_allowed_volume_cv
            )
        else:
            track_volume_cv = np.nan
            max_observed_volume_fold_change = np.nan
            volume_jump = False
            high_volume_cv = False

        if volume_jump:
            reasons.append("volume_jump")
        if high_volume_cv:
            reasons.append("high_volume_cv")


        # 3. Intensity jump

        if has_intensity:
            intensities = g["mean_intensity"].to_numpy(dtype=float)
            intensities = intensities[np.isfinite(intensities)]

            if len(intensities) >= 2:
                i1 = intensities[:-1]
                i2 = intensities[1:]
                valid = (i1 > eps) & (i2 > eps)

                if valid.any():
                    intensity_ratios = np.maximum(
                        i2[valid] / i1[valid],
                        i1[valid] / i2[valid],
                    )
                    max_observed_intensity_fold_change = float(np.nanmax(intensity_ratios))
                else:
                    max_observed_intensity_fold_change = np.nan
            else:
                max_observed_intensity_fold_change = np.nan

            intensity_jump = (
                np.isfinite(max_observed_intensity_fold_change)
                and max_observed_intensity_fold_change > max_allowed_intensity_fold_change
            )
        else:
            max_observed_intensity_fold_change = np.nan
            intensity_jump = False

        if intensity_jump:
            reasons.append("intensity_jump")

        split_merge_warning = len(reasons) > 0

        flag_rows.append(
            {
                "track_id": track_id,
                "max_observed_step_distance_3d_for_split_merge": max_step,
                "median_step_distance_3d_for_split_merge": median_step,
                "max_step_to_median_ratio": step_to_median_ratio,
                "track_volume_cv_for_split_merge": track_volume_cv,
                "max_observed_volume_fold_change": max_observed_volume_fold_change,
                "max_observed_intensity_fold_change": max_observed_intensity_fold_change,
                "split_merge_step_spike": bool(step_spike),
                "split_merge_volume_jump": bool(volume_jump),
                "split_merge_high_volume_cv": bool(high_volume_cv),
                "split_merge_intensity_jump": bool(intensity_jump),
                "split_merge_warning": bool(split_merge_warning),
                "split_merge_reason": ";".join(reasons) if reasons else "",
            }
        )

    flags_df = pd.DataFrame(flag_rows)
    quality_df = quality_df.merge(flags_df, on="track_id", how="left")

    bool_cols = [
        "split_merge_step_spike",
        "split_merge_volume_jump",
        "split_merge_high_volume_cv",
        "split_merge_intensity_jump",
        "split_merge_warning",
    ]

    for col in bool_cols:
        if col in quality_df.columns:
            quality_df[col] = quality_df[col].fillna(False).astype(bool)

    if "split_merge_reason" in quality_df.columns:
        quality_df["split_merge_reason"] = quality_df["split_merge_reason"].fillna("")

    return quality_df


def add_clean_feature_exclusion_flags(quality_df: pd.DataFrame) -> pd.DataFrame:
    """Creates clean-feature exclusion columns using only automated split/merge flags."""

    # Ensure expected columns exist and are boolean-safe.
    bool_cols = [
        "split_merge_warning",
        "split_merge_step_spike",
        "split_merge_volume_jump",
        "split_merge_high_volume_cv",
        "split_merge_intensity_jump",
    ]

    for col in bool_cols:
        if col not in quality_df.columns:
            quality_df[col] = False
        quality_df[col] = quality_df[col].fillna(False).astype(bool)

    if "split_merge_reason" not in quality_df.columns:
        quality_df["split_merge_reason"] = ""
    quality_df["split_merge_reason"] = quality_df["split_merge_reason"].fillna("")

    # Step spike alone is retained as a warning, not used as automatic exclusion.
    quality_df["step_spike_only_warning"] = (
        quality_df["split_merge_step_spike"]
        & ~quality_df["split_merge_volume_jump"]
        & ~quality_df["split_merge_high_volume_cv"]
        & ~quality_df["split_merge_intensity_jump"]
    )

    if CELL_TYPE == "macrophage":
        quality_df["exclude_from_clean_features"] = (
            quality_df["split_merge_high_volume_cv"]
            | quality_df["split_merge_intensity_jump"]
            | (
                quality_df["split_merge_volume_jump"]
                & quality_df["split_merge_step_spike"]
            )
        )
    elif CELL_TYPE == "musc":
        quality_df["exclude_from_clean_features"] = (
            quality_df["split_merge_volume_jump"]
            | quality_df["split_merge_high_volume_cv"]
            | quality_df["split_merge_intensity_jump"]
        )

    def _make_exclusion_reason(row):
        reasons = []

        if CELL_TYPE == "macrophage":
            if bool(row.get("split_merge_high_volume_cv", False)):
                reasons.append("high_volume_cv")
            if bool(row.get("split_merge_intensity_jump", False)):
                reasons.append("intensity_jump")
            if (
                bool(row.get("split_merge_volume_jump", False))
                and bool(row.get("split_merge_step_spike", False))
            ):
                reasons.append("volume_jump_with_step_spike")
        else:
            if bool(row.get("split_merge_volume_jump", False)):
                reasons.append("volume_jump")
            if bool(row.get("split_merge_high_volume_cv", False)):
                reasons.append("high_volume_cv")
            if bool(row.get("split_merge_intensity_jump", False)):
                reasons.append("intensity_jump")

        return ";".join(reasons)

    quality_df["clean_feature_exclusion_reason"] = quality_df.apply(
        _make_exclusion_reason,
        axis=1,
    )

    return quality_df


def main():
    tracks_csv = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{FEATURE_TRACKING_METHOD}.csv"

    print("\n==============================")
    print("TRACK QUALITY FEATURE EXTRACTION")
    print("==============================\n")

    print(f"Cell type:       {CELL_TYPE}")
    print(f"Tracking method: {FEATURE_TRACKING_METHOD}")
    print(f"Input tracks:    {tracks_csv}")
    print(f"Output features: {TRACK_FEATURES_CSV}")

    if not tracks_csv.exists():
        raise FileNotFoundError(
            f"Tracks file not found:\n{tracks_csv}\n"
            "Run 03_track_3d_objects.py first, or change FEATURE_TRACKING_METHOD in config.py."
        )

    df = pd.read_csv(tracks_csv)

    if df.empty:
        print("[ERROR] Tracks CSV is empty.")
        return

    required = [
        "track_id",
        "time",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    for track_id, g in df.groupby("track_id"):
        g = add_step_features(g)

        length = int(g["time"].nunique())

        first = g.iloc[0]
        last = g.iloc[-1]

        z0, y0, x0 = first[["centroid_z", "centroid_y", "centroid_x"]]
        z1, y1, x1 = last[["centroid_z", "centroid_y", "centroid_x"]]

        net_displacement_3d = np.sqrt(
            (x1 - x0) ** 2
            + (y1 - y0) ** 2
            + ((z1 - z0) * Z_DISTANCE_WEIGHT) ** 2
        )

        net_displacement_xy = np.sqrt(
            (x1 - x0) ** 2
            + (y1 - y0) ** 2
        )

        net_z_displacement_slices = z1 - z0

        path_length_3d = g["step_distance_3d"].fillna(0).sum()
        path_length_xy = g["step_distance_xy"].fillna(0).sum()

        mean_step_3d = g["step_distance_3d"].mean()
        max_step_3d = g["step_distance_3d"].max()

        mean_step_xy = g["step_distance_xy"].mean()
        max_step_xy = g["step_distance_xy"].max()

        mean_abs_z_step = g["step_z_slices"].abs().mean()
        max_abs_z_step = g["step_z_slices"].abs().max()

        directionality_3d = net_displacement_3d / path_length_3d if path_length_3d > 0 else np.nan
        directionality_xy = net_displacement_xy / path_length_xy if path_length_xy > 0 else np.nan

        z_range = g["centroid_z"].max() - g["centroid_z"].min()
        y_range = g["centroid_y"].max() - g["centroid_y"].min()
        x_range = g["centroid_x"].max() - g["centroid_x"].min()

        volume_mean = g["volume_voxels"].mean() if "volume_voxels" in g.columns else np.nan
        volume_std = g["volume_voxels"].std() if "volume_voxels" in g.columns else np.nan
        volume_min = g["volume_voxels"].min() if "volume_voxels" in g.columns else np.nan
        volume_max = g["volume_voxels"].max() if "volume_voxels" in g.columns else np.nan

        volume_cv = (
            volume_std / volume_mean
            if pd.notna(volume_mean) and volume_mean > 0
            else np.nan
        )

        mean_intensity = g["mean_intensity"].mean() if "mean_intensity" in g.columns else np.nan
        max_intensity = g["max_intensity"].max() if "max_intensity" in g.columns else np.nan

        start_time = int(g["time"].min())
        end_time = int(g["time"].max())
        duration = end_time - start_time + 1
        completeness = length / max(duration, 1)

        suspicious_jump = (
            max_step_3d > SUSPICIOUS_JUMP_THRESHOLD
            if pd.notna(max_step_3d)
            else False
        )

        possibly_static = (
            net_displacement_3d < STATIC_DISPLACEMENT_THRESHOLD
            and length >= LONG_TRACK_LENGTH
        )

        rows.append(
            {
                "cell_type": CELL_TYPE,
                "tracking_method": FEATURE_TRACKING_METHOD,
                "track_id": track_id,
                "track_length": length,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "completeness": completeness,
                "net_displacement_3d": net_displacement_3d,
                "net_displacement_xy": net_displacement_xy,
                "net_z_displacement_slices": net_z_displacement_slices,
                "path_length_3d": path_length_3d,
                "path_length_xy": path_length_xy,
                "directionality_3d": directionality_3d,
                "directionality_xy": directionality_xy,
                "mean_step_distance_3d": mean_step_3d,
                "max_step_distance_3d": max_step_3d,
                "mean_step_distance_xy": mean_step_xy,
                "max_step_distance_xy": max_step_xy,
                "z_range": z_range,
                "y_range": y_range,
                "x_range": x_range,
                "mean_z": g["centroid_z"].mean(),
                "mean_y": g["centroid_y"].mean(),
                "mean_x": g["centroid_x"].mean(),
                "mean_abs_z_step": mean_abs_z_step,
                "max_abs_z_step": max_abs_z_step,
                "mean_volume": volume_mean,
                "min_volume": volume_min,
                "max_volume": volume_max,
                "volume_cv": volume_cv,
                "mean_intensity": mean_intensity,
                "max_intensity": max_intensity,
                "possibly_static": possibly_static,
                "long_track": length >= LONG_TRACK_LENGTH,
                "very_long_track": length >= VERY_LONG_TRACK_LENGTH,
                "suspicious_jump": suspicious_jump,
            }
        )

    summary = pd.DataFrame(rows)

    if ENABLE_SPLIT_MERGE_FLAGS:
        print("\n[INFO] Adding split/merge warning flags...")
        summary = add_split_merge_flags_to_quality_features(
            tracks_df=df,
            quality_df=summary,
            z_distance_weight=Z_DISTANCE_WEIGHT,
            max_allowed_volume_fold_change=MAX_VOLUME_FOLD_CHANGE_FOR_SPLIT_MERGE,
            max_allowed_intensity_fold_change=MAX_INTENSITY_FOLD_CHANGE_FOR_SPLIT_MERGE,
            max_allowed_volume_cv=MAX_VOLUME_CV_FOR_SPLIT_MERGE,
            max_allowed_step_to_median_ratio=MAX_STEP_TO_MEDIAN_RATIO_FOR_SPLIT_MERGE,
            min_absolute_step=MIN_ABSOLUTE_STEP_FOR_SPLIT_MERGE,
        )

        n_warn = int(summary["split_merge_warning"].sum())
        print(f"[INFO] Tracks with split/merge warning: {n_warn}/{len(summary)}")

        if n_warn > 0:
            print("\nTop split/merge warning reasons:")
            print(
                summary.loc[summary["split_merge_warning"], "split_merge_reason"]
                .value_counts()
                .head(10)
            )
    else:
        summary["split_merge_warning"] = False
        summary["split_merge_reason"] = ""

    summary = add_clean_feature_exclusion_flags(summary)
    n_clean_exclude = int(summary["exclude_from_clean_features"].sum())
    n_step_spike_only = int(summary["step_spike_only_warning"].sum())

    print(f"\n[INFO] Tracks excluded from clean features: {n_clean_exclude}/{len(summary)}")
    print(
        "[INFO] Step-spike-only warnings retained for clean features: "
        f"{n_step_spike_only}/{len(summary)}"
    )

    TRACK_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TRACK_FEATURES_CSV, index=False)

    print("\nSaved:", TRACK_FEATURES_CSV)

    print("\nTrack quality summary:")
    print(summary.describe(include="all"))

    print("\nLong tracks:", int(summary["long_track"].sum()))
    print("Very long tracks:", int(summary["very_long_track"].sum()))
    print("Possibly static long tracks:", int(summary["possibly_static"].sum()))
    print(
        f"Tracks with suspicious jump > {SUSPICIOUS_JUMP_THRESHOLD} px-equivalent:",
        int(summary["suspicious_jump"].sum()),
    )
    print("Tracks with split/merge warning:", int(summary["split_merge_warning"].sum()))
    print("Step-spike-only warnings retained:", int(summary["step_spike_only_warning"].sum()))
    print("Tracks excluded from clean features:", int(summary["exclude_from_clean_features"].sum()))

    print("\nTop 20 longest tracks:")
    display_cols = [
        "track_id",
        "track_length",
        "net_displacement_3d",
        "path_length_3d",
        "directionality_3d",
        "z_range",
        "mean_volume",
        "volume_cv",
        "split_merge_warning",
        "split_merge_reason",
        "step_spike_only_warning",
        "exclude_from_clean_features",
        "clean_feature_exclusion_reason",
        "possibly_static",
        "suspicious_jump",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]

    print(
        summary.sort_values("track_length", ascending=False)
        .head(20)[display_cols]
    )

    flagged = summary[summary["split_merge_warning"]].copy()
    if not flagged.empty:
        flagged_cols = [
            "track_id",
            "track_length",
            "net_displacement_3d",
            "path_length_3d",
            "z_range",
            "mean_volume",
            "volume_cv",
            "max_observed_volume_fold_change",
            "max_observed_intensity_fold_change",
            "max_step_to_median_ratio",
            "step_spike_only_warning",
            "exclude_from_clean_features",
            "clean_feature_exclusion_reason",
            "split_merge_reason",
        ]
        flagged_cols = [c for c in flagged_cols if c in summary.columns]

        print("\nTop 20 split/merge-warning tracks:")
        print(
            flagged.sort_values(
                ["track_length", "net_displacement_3d"],
                ascending=[False, False],
            )
            .head(20)[flagged_cols]
        )


if __name__ == "__main__":
    main()
