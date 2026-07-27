import pandas as pd

from config import (
    CELL_TYPE,
    TRACKING_OUTPUT_DIR,
    FEATURE_TRACKING_METHOD,
    TRACK_FEATURES_CSV,
    MIN_GOOD_FILTER_TRACK_LENGTH,
    MAX_GOOD_FILTER_STEP_DISTANCE_3D,
    MAX_GOOD_FILTER_Z_RANGE,
    REMOVE_STATIC_TRACKS,
)

try:
    from config import REMOVE_SPLIT_MERGE_WARNINGS
except ImportError:
    REMOVE_SPLIT_MERGE_WARNINGS = True


def as_bool_series(s: pd.Series) -> pd.Series:
    """Safely convert bool/string/numeric columns to boolean."""
    if s.dtype == bool:
        return s.fillna(False)

    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def append_filter_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Append a semicolon-separated reason to filter_reason."""
    if int(mask.sum()) == 0:
        return

    df.loc[mask, "filter_reason"] = df.loc[mask, "filter_reason"].apply(
        lambda x: reason if x == "" else f"{x};{reason}"
    )


def main():
    tracks_csv = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{FEATURE_TRACKING_METHOD}.csv"

    good_tracks_csv = (
        TRACKING_OUTPUT_DIR
        / f"{CELL_TYPE}_tracks_{FEATURE_TRACKING_METHOD}_good_filtered.csv"
    )

    clean_tracks_csv = (
        TRACKING_OUTPUT_DIR
        / f"{CELL_TYPE}_tracks_{FEATURE_TRACKING_METHOD}_clean_for_features.csv"
    )

    good_track_ids_csv = (
        TRACKING_OUTPUT_DIR
        / f"{CELL_TYPE}_good_track_ids_{FEATURE_TRACKING_METHOD}.csv"
    )

    clean_track_ids_csv = (
        TRACKING_OUTPUT_DIR
        / f"{CELL_TYPE}_clean_track_ids_{FEATURE_TRACKING_METHOD}.csv"
    )

    good_quality_csv = (
        TRACKING_OUTPUT_DIR
        / f"{CELL_TYPE}_good_track_quality_{FEATURE_TRACKING_METHOD}.csv"
    )

    clean_quality_csv = (
        TRACKING_OUTPUT_DIR
        / f"{CELL_TYPE}_clean_track_quality_{FEATURE_TRACKING_METHOD}.csv"
    )

    filter_report_csv = (
        TRACKING_OUTPUT_DIR
        / f"{CELL_TYPE}_track_filter_report_{FEATURE_TRACKING_METHOD}.csv"
    )

    print("\n==============================")
    print("GOOD / CLEAN TRACK FILTER")
    print("==============================")
    print(f"Cell type:        {CELL_TYPE}")
    print(f"Tracking method:  {FEATURE_TRACKING_METHOD}")
    print(f"Input tracks:     {tracks_csv}")
    print(f"Quality features: {TRACK_FEATURES_CSV}")

    if not tracks_csv.exists():
        raise FileNotFoundError(
            f"Tracks file not found:\n{tracks_csv}\n"
            "Run 03_track_3d_objects.py first."
        )

    if not TRACK_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Track quality feature file not found:\n{TRACK_FEATURES_CSV}\n"
            "Run 06_track_quality_features.py first."
        )

    tracks = pd.read_csv(tracks_csv)
    quality = pd.read_csv(TRACK_FEATURES_CSV)

    required_quality_cols = [
        "track_id",
        "track_length",
        "max_step_distance_3d",
        "possibly_static",
        "z_range",
    ]

    missing = [c for c in required_quality_cols if c not in quality.columns]

    if missing:
        raise ValueError(
            f"Missing required quality columns: {missing}. "
            f"Available columns: {list(quality.columns)}"
        )

    quality["filter_reason"] = ""


    # Basic good-track filter

    short_mask = quality["track_length"] < MIN_GOOD_FILTER_TRACK_LENGTH
    append_filter_reason(quality, short_mask, "short_track")

    high_step_mask = quality["max_step_distance_3d"] > MAX_GOOD_FILTER_STEP_DISTANCE_3D
    append_filter_reason(quality, high_step_mask, "high_step_distance")

    high_z_mask = quality["z_range"] > MAX_GOOD_FILTER_Z_RANGE
    append_filter_reason(quality, high_z_mask, "high_z_range")

    if REMOVE_STATIC_TRACKS:
        static_mask = as_bool_series(quality["possibly_static"])
        append_filter_reason(quality, static_mask, "possibly_static")

    good_quality = quality[quality["filter_reason"] == ""].copy()
    good_track_ids = set(good_quality["track_id"].tolist())
    good_tracks = tracks[tracks["track_id"].isin(good_track_ids)].copy()


    # Clean-for-features filter: good tracks minus automated clean-feature exclusions

    clean_quality = good_quality.copy()
    clean_feature_removed = 0
    split_merge_warning_in_good = 0
    step_spike_only_retained_in_good = 0

    if "split_merge_warning" in clean_quality.columns:
        split_merge_warning_in_good = int(as_bool_series(clean_quality["split_merge_warning"]).sum())

    if "step_spike_only_warning" in clean_quality.columns:
        step_spike_only_retained_in_good = int(
            as_bool_series(clean_quality["step_spike_only_warning"]).sum()
        )

    if REMOVE_SPLIT_MERGE_WARNINGS:
        if "exclude_from_clean_features" in clean_quality.columns:
            clean_exclude_mask = as_bool_series(clean_quality["exclude_from_clean_features"])
        elif "split_merge_warning" in clean_quality.columns:
            clean_exclude_mask = as_bool_series(clean_quality["split_merge_warning"])
            print(
                "[WARNING] 'exclude_from_clean_features' was not found, so falling back "
                "to 'split_merge_warning'. This may be over-strict. Re-run the updated "
                "6_track_quality_features.py to use the balanced exclusion rule."
            )
        else:
            clean_exclude_mask = pd.Series(False, index=clean_quality.index)
            print(
                "[WARNING] REMOVE_SPLIT_MERGE_WARNINGS=True, but neither "
                "'exclude_from_clean_features' nor 'split_merge_warning' was found. "
                "Run 6_track_quality_features.py after adding split/merge flags."
            )

        clean_feature_removed = int(clean_exclude_mask.sum())
        clean_quality = clean_quality[~clean_exclude_mask].copy()

    clean_track_ids = set(clean_quality["track_id"].tolist())
    clean_tracks = tracks[tracks["track_id"].isin(clean_track_ids)].copy()


    # Save outputs

    good_tracks_csv.parent.mkdir(parents=True, exist_ok=True)

    good_tracks.to_csv(good_tracks_csv, index=False)
    good_quality[["track_id"]].to_csv(good_track_ids_csv, index=False)
    good_quality.to_csv(good_quality_csv, index=False)

    clean_tracks.to_csv(clean_tracks_csv, index=False)
    clean_quality[["track_id"]].to_csv(clean_track_ids_csv, index=False)
    clean_quality.to_csv(clean_quality_csv, index=False)

    quality.to_csv(filter_report_csv, index=False)


    # Print summary

    print("\n==============================")
    print("GOOD / CLEAN TRACK FILTER SUMMARY")
    print("==============================")
    print(f"Total tracks before filtering:     {quality['track_id'].nunique()}")
    print(f"Good tracks after basic filtering: {len(good_track_ids)}")
    print(f"Clean tracks for features:         {len(clean_track_ids)}")
    print(f"Total detections before:           {len(tracks)}")
    print(f"Good detections after filtering:   {len(good_tracks)}")
    print(f"Clean detections for features:     {len(clean_tracks)}")

    print("\nFilter settings:")
    print(f"Minimum track length:              {MIN_GOOD_FILTER_TRACK_LENGTH}")
    print(f"Maximum 3D step distance:          {MAX_GOOD_FILTER_STEP_DISTANCE_3D}")
    print(f"Maximum Z range:                   {MAX_GOOD_FILTER_Z_RANGE}")
    print(f"Remove static tracks:              {REMOVE_STATIC_TRACKS}")
    print(f"Remove clean-feature exclusions:   {REMOVE_SPLIT_MERGE_WARNINGS}")
    print(f"Good tracks with any split/merge warning: {split_merge_warning_in_good}")
    print(f"Good tracks with step-spike-only warning retained: {step_spike_only_retained_in_good}")
    print(f"Good tracks removed from clean features: {clean_feature_removed}")

    print("\nBasic rejection reason counts:")
    rejected = quality[quality["filter_reason"] != ""].copy()

    if rejected.empty:
        print("No tracks rejected by basic filters.")
    else:
        print(
            rejected["filter_reason"]
            .str.split(";")
            .explode()
            .value_counts()
        )

    if REMOVE_SPLIT_MERGE_WARNINGS and "clean_feature_exclusion_reason" in good_quality.columns:
        excluded_clean = good_quality[
            good_quality["clean_feature_exclusion_reason"].fillna("").astype(str).str.strip() != ""
        ]
        if not excluded_clean.empty:
            print("\nClean-feature exclusion reason counts among basic-good tracks:")
            print(
                excluded_clean["clean_feature_exclusion_reason"]
                .str.split(";")
                .explode()
                .value_counts()
            )

    if not clean_quality.empty:
        print("\nClean track length summary:")
        print(clean_quality["track_length"].describe())

        if "net_displacement_3d" in clean_quality.columns:
            print("\nClean track 3D displacement summary:")
            print(clean_quality["net_displacement_3d"].describe())

        print("\nClean track Z range summary:")
        print(clean_quality["z_range"].describe())
    else:
        print("\n[WARNING] No tracks passed the final clean filter.")

    print("\nSaved:")
    print(good_tracks_csv)
    print(good_track_ids_csv)
    print(good_quality_csv)
    print(clean_tracks_csv)
    print(clean_track_ids_csv)
    print(clean_quality_csv)
    print(filter_report_csv)


if __name__ == "__main__":
    main()
