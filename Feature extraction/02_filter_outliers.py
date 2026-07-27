from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTLIER_FEATURES = [
    "max_step_distance_3d_um",
    "mean_speed_um_per_frame",
    "max_speed_um_per_frame",
    "total_path_length_3d_um",
    "net_displacement_3d_um",
    "max_volume_fold_change",
    "max_intensity_fold_change",
]


# Median absolute deviation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html
def robust_upper_threshold(values: pd.Series, mad_multiplier: float = 6.0) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) < 8:
        return np.nan

    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad > 0:
        return float(median + mad_multiplier * 1.4826 * mad)

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    if iqr > 0:
        return float(q3 + 3.0 * iqr)

    return np.nan


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        return numeric.fillna(0) > 0

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "fish_id" not in df.columns:
        if "file" in df.columns:
            df["fish_id"] = df["file"].astype(str)
        else:
            df["fish_id"] = "unknown_fish"

    if "track_id" not in df.columns:
        raise ValueError("Feature file must contain track_id.")

    return df


def add_outlier_flags(
    df: pd.DataFrame,
    group_col: str,
    min_track_length: int,
    mad_multiplier: float,
    hard_max_step_um: float | None,
    hard_max_speed_um_per_frame: float | None,
    max_volume_fold_change: float | None,
    max_intensity_fold_change: float | None,
) -> pd.DataFrame:
    df = df.copy()

    for col in DEFAULT_OUTLIER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["qc_exclude"] = False
    df["qc_reason"] = ""


    # 1. Anatomical/body-region flags, if already present

    if "outside_body_fraction" in df.columns:
        outside = pd.to_numeric(df["outside_body_fraction"], errors="coerce")
        flag = outside > 0.20
        df.loc[flag, "qc_exclude"] = True
        df.loc[flag, "qc_reason"] += "outside_body_fraction>0.20; "

    if "inside_body_fraction" in df.columns:
        inside = pd.to_numeric(df["inside_body_fraction"], errors="coerce")
        flag = inside < 0.80
        df.loc[flag, "qc_exclude"] = True
        df.loc[flag, "qc_reason"] += "inside_body_fraction<0.80; "


    # 2. Hard biological/technical caps

    if hard_max_step_um is not None and "max_step_distance_3d_um" in df.columns:
        flag = df["max_step_distance_3d_um"] > hard_max_step_um
        df.loc[flag, "qc_exclude"] = True
        df.loc[flag, "qc_reason"] += f"hard_max_step>{hard_max_step_um}um; "

    if hard_max_speed_um_per_frame is not None and "max_speed_um_per_frame" in df.columns:
        flag = df["max_speed_um_per_frame"] > hard_max_speed_um_per_frame
        df.loc[flag, "qc_exclude"] = True
        df.loc[flag, "qc_reason"] += f"hard_max_speed>{hard_max_speed_um_per_frame}um/frame; "

    if max_volume_fold_change is not None and "max_volume_fold_change" in df.columns:
        flag = df["max_volume_fold_change"] > max_volume_fold_change
        df.loc[flag, "qc_warning_volume"] = True
        df.loc[flag, "qc_reason"] += f"volume_fold_change>{max_volume_fold_change}; "

    if max_volume_fold_change is not None and "max_volume_fold_change" in df.columns:
        flag = df["max_volume_fold_change"] > max_volume_fold_change
        df.loc[flag, "qc_warning_volume"] = True
        df.loc[flag, "qc_reason"] += f"volume_fold_change>{max_volume_fold_change}; "


    # 3. Per-fish robust outlier thresholds

    threshold_rows = []

    for fish_id, group in df.groupby(group_col):
        for feature in DEFAULT_OUTLIER_FEATURES:
            if feature not in group.columns:
                continue

            threshold = robust_upper_threshold(group[feature], mad_multiplier=mad_multiplier)

            if not np.isfinite(threshold):
                continue

            threshold_rows.append(
                {
                    group_col: fish_id,
                    "feature": feature,
                    "robust_upper_threshold": threshold,
                }
            )

            flag = (
                (df[group_col] == fish_id)
                & (pd.to_numeric(df[feature], errors="coerce") > threshold)
                & (pd.to_numeric(df.get("track_length", np.nan), errors="coerce") >= min_track_length)
            )

            # max step and speed are strong evidence of tracking error.
            if feature in {"max_step_distance_3d_um", "max_speed_um_per_frame"}:
                df.loc[flag, "qc_exclude"] = True
                df.loc[flag, "qc_reason"] += f"{feature}>per_fish_threshold; "

            # total path / net displacement alone are suspicious, but not always wrong.
            # Exclude only if the track is short or directionality is very high.
            elif feature in {"total_path_length_3d_um", "net_displacement_3d_um"}:
                short_track = pd.to_numeric(df.get("track_length", np.nan), errors="coerce") < 10
                high_direct = pd.to_numeric(df.get("directionality_ratio", np.nan), errors="coerce") > 0.85

                strong_flag = flag & (short_track | high_direct)

                df.loc[strong_flag, "qc_exclude"] = True
                df.loc[strong_flag, "qc_reason"] += f"{feature}>per_fish_threshold_with_short_or_direct_track; "

            # volume/intensity fold change often means a bad identity switch.
            else:
                if feature in {"max_volume_fold_change", "max_intensity_fold_change"}:
                    df.loc[flag, "qc_reason"] += f"{feature}>per_fish_threshold_warning; "
                    if feature == "max_volume_fold_change":
                        df.loc[flag, "qc_warning_volume"] = True
                    if feature == "max_intensity_fold_change":
                        df.loc[flag, "qc_warning_intensity"] = True
                else:
                    df.loc[flag, "qc_exclude"] = True
                    df.loc[flag, "qc_reason"] += f"{feature}>per_fish_threshold; "

    threshold_df = pd.DataFrame(threshold_rows)

    df["qc_keep"] = ~df["qc_exclude"]

    return df, threshold_df


def filter_original_tracks(
    tracks_csv: Path,
    flagged_features: pd.DataFrame,
    output_path: Path,
):
    tracks = pd.read_csv(tracks_csv)

    bad_ids = set(flagged_features.loc[flagged_features["qc_exclude"], "track_id"].astype(str))

    before = len(tracks)
    tracks = tracks[~tracks["track_id"].astype(str).isin(bad_ids)].copy()
    after = len(tracks)

    tracks.to_csv(output_path, index=False)

    print(f"[SAVED] Filtered track CSV: {output_path}")
    print(f"[INFO] Track rows removed: {before - after}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", required=True, help="Cell-level track feature CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracks-csv", default=None, help="Optional original tracked CSV to filter for plotting.")
    parser.add_argument("--group-col", default="fish_id")
    parser.add_argument("--min-track-length", type=int, default=5)
    parser.add_argument("--mad-multiplier", type=float, default=6.0)

    parser.add_argument("--hard-max-step-um", type=float, default=25.0)
    parser.add_argument("--hard-max-speed-um-per-frame", type=float, default=25.0)
    parser.add_argument("--max-volume-fold-change", type=float, default=8.0)
    parser.add_argument("--max-intensity-fold-change", type=float, default=10.0)

    args = parser.parse_args()

    feature_path = Path(args.features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_path)
    df = normalise_columns(df)

    if args.group_col not in df.columns:
        print(f"[WARN] group column {args.group_col} not found. Using fish_id.")
        args.group_col = "fish_id"

    flagged, thresholds = add_outlier_flags(
        df=df,
        group_col=args.group_col,
        min_track_length=args.min_track_length,
        mad_multiplier=args.mad_multiplier,
        hard_max_step_um=args.hard_max_step_um,
        hard_max_speed_um_per_frame=args.hard_max_speed_um_per_frame,
        max_volume_fold_change=args.max_volume_fold_change,
        max_intensity_fold_change=args.max_intensity_fold_change,
    )

    flagged_out = output_dir / "cell_track_features_with_qc_flags.csv"
    clean_out = output_dir / "cell_track_features_qc_filtered.csv"
    removed_out = output_dir / "removed_track_qc_outliers.csv"
    thresholds_out = output_dir / "per_fish_outlier_thresholds.csv"

    flagged.to_csv(flagged_out, index=False)
    flagged[flagged["qc_keep"]].to_csv(clean_out, index=False)
    flagged[flagged["qc_exclude"]].to_csv(removed_out, index=False)
    thresholds.to_csv(thresholds_out, index=False)

    print(f"[SAVED] All features with QC flags: {flagged_out}")
    print(f"[SAVED] QC-filtered features: {clean_out}")
    print(f"[SAVED] Removed tracks: {removed_out}")
    print(f"[SAVED] Per-fish thresholds: {thresholds_out}")

    print("\n[INFO] QC summary:")
    print(flagged["qc_exclude"].value_counts(dropna=False))

    if "qc_reason" in flagged.columns:
        print("\n[INFO] Top removal reasons:")
        print(flagged.loc[flagged["qc_exclude"], "qc_reason"].value_counts().head(20))

    if args.tracks_csv:
        tracks_csv = Path(args.tracks_csv)
        filtered_tracks_out = output_dir / "tracks_qc_filtered.csv"
        filter_original_tracks(
            tracks_csv=tracks_csv,
            flagged_features=flagged,
            output_path=filtered_tracks_out,
        )


if __name__ == "__main__":
    main()
