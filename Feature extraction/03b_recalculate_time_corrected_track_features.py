#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATTERNS = {
    "musc": "musc/*object_timepoint_features.csv",
    "macrophage_all": "macrophage_all/*object_timepoint_features.csv",
    "macrophage_outside_boundary": "macrophage_outside_boundary/*object_timepoint_features.csv",
}

SHAPE_FEATURES = [
    "sphericity",
    "elongation",
    "volume_um3",
    "surface_area_um2",
    "surface_area_to_volume",
    "aspect_ratio_3d",
    "solidity_3d",
    "extent_3d",
    "compactness_3d",
    "flatness",
    "prolate_ellipticity",
    "oblate_ellipticity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate track-level movement features using real frame "
            "intervals from block metadata."
        )
    )
    parser.add_argument("--root", default="final_feature_outputs")
    parser.add_argument("--dataset", choices=sorted(DATASET_PATTERNS), required=True)
    parser.add_argument("--metadata-file", default="block_metadata.csv")
    parser.add_argument("--metadata-block-col", default="block_name")
    parser.add_argument("--time-interval-col", default="time_interval_seconds")
    parser.add_argument("--qc-track-table", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-track-points", type=int, default=2)
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def read_metadata(path: Path, block_col: str, interval_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = pd.read_csv(path, low_memory=False)
    missing = [column for column in [block_col, interval_col] if column not in metadata.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. Add a positive {interval_col} value for each block."
        )
    metadata = metadata[[block_col, interval_col]].copy()
    metadata[block_col] = metadata[block_col].astype(str).str.strip()
    metadata[interval_col] = numeric(metadata[interval_col])
    bad = metadata[metadata[interval_col].isna() | (metadata[interval_col] <= 0)]
    if not bad.empty:
        raise ValueError(
            "Metadata has missing/non-positive time intervals for blocks: "
            + ", ".join(bad[block_col].astype(str).head(20))
        )
    duplicated = metadata[metadata[block_col].duplicated()][block_col].unique()
    if len(duplicated):
        raise ValueError(
            "Metadata has duplicate block_name rows: " + ", ".join(map(str, duplicated[:20]))
        )
    return metadata


def load_timepoints(root: Path, dataset: str) -> pd.DataFrame:
    files = sorted(root.glob(f"*/{DATASET_PATTERNS[dataset]}"))
    if not files:
        raise FileNotFoundError(f"No object timepoint files found for {dataset} under {root}")

    base_needed = {
        "time",
        "track_id",
        "block_name",
        "fish_id",
        "genotype",
        "centroid_x_um",
        "centroid_y_um",
        "centroid_z_um",
    }
    frames: list[pd.DataFrame] = []
    for path in files:
        header = pd.read_csv(path, nrows=0)
        usecols = [column for column in header.columns if column in base_needed or column in SHAPE_FEATURES]
        missing = base_needed - set(usecols)
        if missing:
            print(f"[WARN] Skipping {path}; missing {sorted(missing)}")
            continue
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No usable object timepoint files found for {dataset}")

    data = pd.concat(frames, ignore_index=True)
    data["block_name"] = data["block_name"].astype(str).str.strip()
    data["fish_id"] = data["fish_id"].astype(str).str.strip()
    data["track_id"] = data["track_id"].astype(str).str.strip()
    data["genotype"] = data["genotype"].map(normalise_genotype)
    for column in ["time", "centroid_x_um", "centroid_y_um", "centroid_z_um"] + [
        column for column in SHAPE_FEATURES if column in data.columns
    ]:
        data[column] = numeric(data[column])
    data = data.dropna(subset=["time", "centroid_x_um", "centroid_y_um", "centroid_z_um"])
    data = data[data["genotype"].isin(["WT", "MUT"])].copy()
    data["global_track_id"] = data["block_name"] + "::" + data["track_id"]
    return data.sort_values(["block_name", "track_id", "time"]).reset_index(drop=True)


def load_qc_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"QC track table not found: {path}")
    qc = pd.read_csv(path, low_memory=False)
    needed = {"block_name", "track_id"}
    if not needed.issubset(qc.columns):
        raise ValueError(f"QC table must have block_name and track_id columns: {path}")
    qc = qc.copy()
    qc["block_name"] = qc["block_name"].astype(str).str.strip()
    qc["track_id"] = qc["track_id"].astype(str).str.strip()
    return qc


def summarize_track(track: pd.DataFrame, interval_seconds: float) -> dict[str, object] | None:
    track = track.sort_values("time")
    if len(track) < 2:
        return None

    coords = track[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy(float)
    frame_times = track["time"].to_numpy(float)
    deltas = np.diff(coords, axis=0)
    frame_deltas = np.diff(frame_times)
    valid = np.isfinite(frame_deltas) & (frame_deltas > 0)
    if not valid.any():
        return None

    step_lengths = np.sqrt((deltas**2).sum(axis=1))
    delta_seconds = frame_deltas * interval_seconds
    valid_steps = valid & np.isfinite(step_lengths) & (delta_seconds > 0)
    if not valid_steps.any():
        return None

    step_lengths = step_lengths[valid_steps]
    delta_seconds = delta_seconds[valid_steps]
    speeds_per_second = step_lengths / delta_seconds
    speeds_per_min = speeds_per_second * 60.0
    squared_per_min = (step_lengths**2) / (delta_seconds / 60.0)

    total_path = float(np.nansum(step_lengths))
    net = float(np.sqrt(((coords[-1] - coords[0]) ** 2).sum()))
    z_values = track["centroid_z_um"].to_numpy(float)
    z_steps = np.abs(np.diff(z_values)[valid_steps])
    z_speeds_per_min = z_steps / delta_seconds * 60.0

    row: dict[str, object] = {
        "block_name": str(track["block_name"].iloc[0]),
        "fish_id": str(track["fish_id"].iloc[0]),
        "genotype": str(track["genotype"].iloc[0]),
        "track_id": str(track["track_id"].iloc[0]),
        "global_track_id": str(track["global_track_id"].iloc[0]),
        "time_interval_seconds": float(interval_seconds),
        "n_timepoints": int(len(track)),
        "track_duration_seconds": float((frame_times[-1] - frame_times[0]) * interval_seconds),
        "track_duration_minutes": float((frame_times[-1] - frame_times[0]) * interval_seconds / 60.0),
        "net_displacement_3d_um": net,
        "total_path_length_3d_um": total_path,
        "directionality_ratio": float(net / total_path) if total_path > 0 else np.nan,
        "tortuosity": float(total_path / net) if net > 0 else np.nan,
        "mean_squared_displacement_3d_um2": float(np.nanmean(step_lengths**2)),
        "mean_squared_displacement_3d_um2_per_min": float(np.nanmean(squared_per_min)),
        "mean_speed_um_per_second": float(np.nanmean(speeds_per_second)),
        "median_speed_um_per_second": float(np.nanmedian(speeds_per_second)),
        "mean_speed_um_per_min": float(np.nanmean(speeds_per_min)),
        "median_speed_um_per_min": float(np.nanmedian(speeds_per_min)),
        "z_range_um": float(np.nanmax(z_values) - np.nanmin(z_values)),
        "z_displacement_um": float(z_values[-1] - z_values[0]),
        "z_path_length_um": float(np.nansum(z_steps)),
        "mean_velocity_z_um_per_min": float(np.nanmean(z_speeds_per_min)),
    }

    for feature in SHAPE_FEATURES:
        if feature not in track.columns:
            continue
        values = numeric(track[feature]).dropna().to_numpy(float)
        if len(values) == 0:
            continue
        row[f"mean_{feature}"] = float(np.nanmean(values))
        row[f"median_{feature}"] = float(np.nanmedian(values))
        if len(values) > 1:
            mean = np.nanmean(values)
            row[f"{feature}_cv"] = float(np.nanstd(values, ddof=1) / mean) if mean else np.nan
            row[f"median_absolute_{feature}_change"] = float(np.nanmedian(np.abs(np.diff(values))))
            feature_table = track[["time", feature]].dropna().copy()
            if len(feature_table) > 1:
                x_minutes = (
                    feature_table["time"].to_numpy(float) - float(feature_table["time"].iloc[0])
                ) * interval_seconds / 60.0
                y_values = feature_table[feature].to_numpy(float)
                if np.nanmax(x_minutes) > np.nanmin(x_minutes):
                    slope, _ = np.polyfit(x_minutes, y_values, 1)
                    row[f"{feature}_slope_per_min"] = float(slope)
    return row


def recalculate(data: pd.DataFrame, metadata: pd.DataFrame, block_col: str, interval_col: str, min_track_points: int) -> pd.DataFrame:
    merged = data.merge(
        metadata.rename(columns={block_col: "block_name"}),
        on="block_name",
        how="left",
        validate="many_to_one",
    )
    missing = sorted(merged.loc[merged[interval_col].isna(), "block_name"].unique())
    if missing:
        raise ValueError(
            "No time_interval_seconds found in metadata for blocks: " + ", ".join(missing[:50])
        )

    rows: list[dict[str, object]] = []
    for _, track in merged.groupby("global_track_id", sort=False):
        if len(track) < min_track_points:
            continue
        interval = float(track[interval_col].iloc[0])
        row = summarize_track(track, interval)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No usable tracks remained after recalculation.")
    return pd.DataFrame(rows)


def merge_with_qc(recalculated: pd.DataFrame, qc: pd.DataFrame | None) -> pd.DataFrame:
    if qc is None:
        return recalculated

    corrected_columns = [
        column
        for column in recalculated.columns
        if column not in {"block_name", "track_id", "fish_id", "genotype"}
    ]
    drop_existing = [column for column in corrected_columns if column in qc.columns]
    qc_base = qc.drop(columns=drop_existing).copy()
    merged = qc_base.merge(
        recalculated[["block_name", "track_id"] + corrected_columns],
        on=["block_name", "track_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("QC merge produced zero rows. Check block_name/track_id matching.")
    return merged


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(Path(args.metadata_file), args.metadata_block_col, args.time_interval_col)
    timepoints = load_timepoints(Path(args.root), args.dataset)
    qc = load_qc_table(Path(args.qc_track_table)) if args.qc_track_table else None
    recalculated = recalculate(
        timepoints,
        metadata,
        args.metadata_block_col,
        args.time_interval_col,
        args.min_track_points,
    )
    output = merge_with_qc(recalculated, qc)

    output_path = output_dir / "cell_track_features_time_corrected.csv"
    output.to_csv(output_path, index=False)
    recalculated.to_csv(output_dir / "all_recalculated_track_features_time_corrected.csv", index=False)
    summary = (
        output.groupby(["block_name", "genotype"], dropna=False)
        .size()
        .reset_index(name="n_tracks")
        .merge(metadata.rename(columns={args.metadata_block_col: "block_name"}), on="block_name", how="left")
    )
    summary.to_csv(output_dir / "time_correction_summary_by_block.csv", index=False)

    print(f"[DONE] Wrote corrected track table: {output_path}")
    print(f"[INFO] Tracks retained: {len(output):,}")
    print(f"[INFO] Blocks: {output['block_name'].nunique():,}")


if __name__ == "__main__":
    main()
