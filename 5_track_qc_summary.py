import numpy as np
import pandas as pd

from config import (
    CELL_TYPE,
    TRACKING_OUTPUT_DIR,
    QC_TRACKING_METHOD,
    MIN_GOOD_TRACK_LENGTH,
    LARGE_JUMP_THRESHOLD,
    Z_DISTANCE_WEIGHT,
)


# Trajectory-analysis reference: https://pubmed.ncbi.nlm.nih.gov/27713081/
def add_step_distances_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 3D step distances if they are not already present."""
    df = df.copy()

    required = ["track_id", "time", "centroid_z", "centroid_y", "centroid_x"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"[WARNING] Cannot calculate step distances. Missing columns: {missing}")
        return df

    df = df.sort_values(["track_id", "time"]).reset_index(drop=True)

    df["prev_z"] = df.groupby("track_id")["centroid_z"].shift(1)
    df["prev_y"] = df.groupby("track_id")["centroid_y"].shift(1)
    df["prev_x"] = df.groupby("track_id")["centroid_x"].shift(1)
    df["prev_time"] = df.groupby("track_id")["time"].shift(1)

    dz = (df["centroid_z"] - df["prev_z"]) * Z_DISTANCE_WEIGHT
    dy = df["centroid_y"] - df["prev_y"]
    dx = df["centroid_x"] - df["prev_x"]

    df["step_distance_3d"] = np.sqrt(dx**2 + dy**2 + dz**2)
    df["step_distance_xy"] = np.sqrt(dx**2 + dy**2)
    df["step_z_slices"] = df["centroid_z"] - df["prev_z"]
    df["time_gap"] = df["time"] - df["prev_time"]

    return df


def main():
    tracks_csv = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{QC_TRACKING_METHOD}.csv"

    print("\n==============================")
    print("TRACKING QC SUMMARY")
    print("==============================\n")

    print(f"Cell type:       {CELL_TYPE}")
    print(f"Tracking method: {QC_TRACKING_METHOD}")
    print(f"Input tracks:    {tracks_csv}")

    if not tracks_csv.exists():
        raise FileNotFoundError(
            f"Tracks file not found:\n{tracks_csv}\n"
            "Run 03_track_3d_objects.py first, or change QC_TRACKING_METHOD in config.py."
        )

    df = pd.read_csv(tracks_csv)

    if df.empty:
        print("[ERROR] Tracks CSV is empty.")
        return

    df = add_step_distances_if_missing(df)

    n_objects = len(df)
    n_tracks = df["track_id"].nunique()
    n_timepoints = df["time"].nunique()

    print(f"\nTotal detections in tracks: {n_objects}")
    print(f"Total tracks:               {n_tracks}")
    print(f"Timepoints represented:     {n_timepoints}")

    track_lengths = df.groupby("track_id")["time"].nunique()

    print("\nTrack length summary:")
    print(track_lengths.describe())

    good_tracks = track_lengths[track_lengths >= MIN_GOOD_TRACK_LENGTH]

    print(f"\nTracks with length >= {MIN_GOOD_TRACK_LENGTH}: {len(good_tracks)}")

    if n_tracks > 0:
        print(f"Percentage useful-length tracks: {100 * len(good_tracks) / n_tracks:.2f}%")


    # Step-distance QC

    if "step_distance_3d" in df.columns:
        valid_steps = df["step_distance_3d"].dropna()

        print("\n3D step distance summary:")
        print(valid_steps.describe())

        large_jumps = df[df["step_distance_3d"] > LARGE_JUMP_THRESHOLD]

        print(f"\nLarge 3D jumps > {LARGE_JUMP_THRESHOLD} px-equivalent: {len(large_jumps)}")

        if len(large_jumps) > 0:
            print("\nExample large jumps:")
            cols = [
                "track_id",
                "time",
                "time_gap",
                "centroid_z",
                "centroid_y",
                "centroid_x",
                "step_z_slices",
                "step_distance_xy",
                "step_distance_3d",
            ]
            cols = [c for c in cols if c in large_jumps.columns]
            print(large_jumps[cols].head(20))

    if "step_distance_xy" in df.columns:
        print("\nXY-only step distance summary:")
        print(df["step_distance_xy"].dropna().describe())

    if "step_z_slices" in df.columns:
        print("\nZ-step summary, in raw Z slices:")
        print(df["step_z_slices"].dropna().describe())


    # Z movement / range QC

    if "centroid_z" in df.columns:
        z_ranges = df.groupby("track_id")["centroid_z"].agg(lambda x: x.max() - x.min())

        print("\nTrack Z-range summary, in raw Z slices:")
        print(z_ranges.describe())


    # Detection counts per timepoint

    print("\nDetections per timepoint:")
    print(df.groupby("time").size())


    # Save outputs

    output_path = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_track_lengths_summary_{QC_TRACKING_METHOD}.csv"
    track_lengths.reset_index(name="track_length").to_csv(output_path, index=False)

    print(f"\nSaved track length summary to: {output_path}")

    qc_tracks_path = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{QC_TRACKING_METHOD}_with_step_qc.csv"
    df.to_csv(qc_tracks_path, index=False)

    print(f"Saved tracks with step QC to: {qc_tracks_path}")


if __name__ == "__main__":
    main()
