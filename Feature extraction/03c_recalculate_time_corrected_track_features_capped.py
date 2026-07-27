#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


HELPER_PATH = Path(__file__).with_name("03b_recalculate_time_corrected_track_features.py")
spec = importlib.util.spec_from_file_location("time_corrected_helpers", HELPER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not import helper functions from {HELPER_PATH}")
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate track-level features after capping each track at a "
            "maximum elapsed real time."
        )
    )
    parser.add_argument("--root", default="final_feature_outputs")
    parser.add_argument("--dataset", choices=sorted(helpers.DATASET_PATTERNS), required=True)
    parser.add_argument("--metadata-file", default="block_metadata.csv")
    parser.add_argument("--metadata-block-col", default="block_name")
    parser.add_argument("--time-interval-col", default="time_interval_seconds")
    parser.add_argument("--qc-track-table", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-track-points", type=int, default=2)
    parser.add_argument("--max-elapsed-minutes", type=float, default=720.0)
    return parser.parse_args()


def cap_track(track: pd.DataFrame, interval_seconds: float, max_elapsed_minutes: float) -> pd.DataFrame:
    track = track.sort_values("time").copy()
    start_time = float(track["time"].iloc[0])
    elapsed_minutes = (track["time"].astype(float) - start_time) * interval_seconds / 60.0
    return track.loc[elapsed_minutes <= max_elapsed_minutes].copy()


def cap_timepoints(
    timepoints: pd.DataFrame,
    metadata: pd.DataFrame,
    block_col: str,
    interval_col: str,
    max_elapsed_minutes: float,
    min_track_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = timepoints.merge(
        metadata.rename(columns={block_col: "block_name"}),
        on="block_name",
        how="left",
        validate="many_to_one",
    )
    missing = sorted(merged.loc[merged[interval_col].isna(), "block_name"].unique())
    if missing:
        raise ValueError(
            "No time interval found in metadata for blocks: " + ", ".join(missing[:50])
        )

    capped_tracks: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []

    for global_track_id, track in merged.groupby("global_track_id", sort=False):
        interval_seconds = float(track[interval_col].iloc[0])
        original_n = len(track)
        original_duration = (
            float(track["time"].max()) - float(track["time"].min())
        ) * interval_seconds / 60.0
        capped = cap_track(track, interval_seconds, max_elapsed_minutes)
        retained = len(capped) >= min_track_points
        if retained:
            capped_tracks.append(capped.drop(columns=[interval_col]))
        report_rows.append(
            {
                "global_track_id": global_track_id,
                "block_name": str(track["block_name"].iloc[0]),
                "fish_id": str(track["fish_id"].iloc[0]),
                "genotype": str(track["genotype"].iloc[0]),
                "track_id": str(track["track_id"].iloc[0]),
                "time_interval_seconds": interval_seconds,
                "original_n_timepoints": original_n,
                "capped_n_timepoints": len(capped),
                "original_duration_minutes": original_duration,
                "cap_minutes": max_elapsed_minutes,
                "retained_after_cap": retained,
            }
        )

    if not capped_tracks:
        raise ValueError("No tracks remained after elapsed-time cap.")
    return pd.concat(capped_tracks, ignore_index=True), pd.DataFrame(report_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = helpers.read_metadata(
        Path(args.metadata_file),
        args.metadata_block_col,
        args.time_interval_col,
    )
    timepoints = helpers.load_timepoints(Path(args.root), args.dataset)
    capped_timepoints, cap_report = cap_timepoints(
        timepoints,
        metadata,
        args.metadata_block_col,
        args.time_interval_col,
        args.max_elapsed_minutes,
        args.min_track_points,
    )
    qc = helpers.load_qc_table(Path(args.qc_track_table)) if args.qc_track_table else None
    recalculated = helpers.recalculate(
        capped_timepoints,
        metadata,
        args.metadata_block_col,
        args.time_interval_col,
        args.min_track_points,
    )
    output = helpers.merge_with_qc(recalculated, qc)

    output_path = output_dir / "cell_track_features_time_corrected_cap720.csv"
    output.to_csv(output_path, index=False)
    recalculated.to_csv(output_dir / "all_recalculated_track_features_time_corrected_cap720.csv", index=False)
    cap_report.to_csv(output_dir / "track_cap720_retention_report.csv", index=False)

    summary = (
        output.groupby(["block_name", "genotype"], dropna=False)
        .size()
        .reset_index(name="n_tracks")
        .merge(
            metadata.rename(columns={args.metadata_block_col: "block_name"}),
            on="block_name",
            how="left",
        )
    )
    summary.to_csv(output_dir / "time_correction_cap720_summary_by_block.csv", index=False)

    print(f"[DONE] Wrote capped corrected track table: {output_path}")
    print(f"[INFO] Tracks retained: {len(output):,}")
    print(f"[INFO] Blocks: {output['block_name'].nunique():,}")
    print(f"[INFO] Cap: {args.max_elapsed_minutes:g} elapsed minutes per track")


if __name__ == "__main__":
    main()
