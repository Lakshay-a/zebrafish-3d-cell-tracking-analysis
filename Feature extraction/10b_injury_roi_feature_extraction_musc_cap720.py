#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


HELPER_PATH = Path(__file__).with_name("10_injury_roi_feature_extraction.py")
spec = importlib.util.spec_from_file_location("injury_helpers", HELPER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not import helper functions from {HELPER_PATH}")
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MUSC manual-injury features after capping tracks at elapsed time."
    )
    parser.add_argument(
        "--model-a-table",
        default=(
            "constrained_fish_features_time_corrected_cap720/"
            "musc/constrained_fish_level_mean_median.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="manual_injury_feature_outputs_time_corrected_cap720",
    )
    parser.add_argument("--max-elapsed-minutes", type=float, default=720.0)
    parser.add_argument("--min-track-points", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_a_path = Path(args.model_a_table)
    output_root = Path(args.output_root)

    if not model_a_path.exists():
        raise FileNotFoundError(
            f"Cap720 Model A table not found: {model_a_path}. "
            "Run run_musc_cap720_model_screen.sh first through feature generation."
        )

    if not helpers.BLOCKS_ROOT.exists():
        raise FileNotFoundError(helpers.BLOCKS_ROOT)
    if not helpers.ANNOTATION_ROOT.exists():
        raise FileNotFoundError(helpers.ANNOTATION_ROOT)
    if not helpers.FRAME_INTERVAL_METADATA.exists():
        raise FileNotFoundError(helpers.FRAME_INTERVAL_METADATA)

    original_enrich = helpers.enrich_detections

    def capped_enrich_detections(*args_, **kwargs_):
        detections, aliases = original_enrich(*args_, **kwargs_)
        track_col = aliases["track"]
        detections = detections[
            detections["_elapsed_time_minutes"] <= args.max_elapsed_minutes
        ].copy()
        counts = detections.groupby(track_col).size()
        keep_tracks = counts[counts >= args.min_track_points].index
        detections = detections[detections[track_col].isin(keep_tracks)].copy()
        if detections.empty:
            raise ValueError("No injury detections remained after cap720 filtering.")
        return detections, aliases

    helpers.enrich_detections = capped_enrich_detections
    helpers.OUTPUT_ROOT = output_root
    helpers.MUSC_MODEL_A_TABLE = model_a_path

    frame_intervals_seconds = helpers.load_frame_interval_metadata(
        helpers.FRAME_INTERVAL_METADATA
    )
    musc_model = pd.read_csv(model_a_path, low_memory=False)
    musc_fish_col = helpers.model_a_fish_column(musc_model)
    musc_fish_names = sorted(
        musc_model[musc_fish_col]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    output_root.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []

    helpers.process_channel(
        channel="musc",
        track_names=helpers.MUSC_TRACK_NAMES,
        model_a_targets=[
            ("musc", model_a_path),
        ],
        fish_names=musc_fish_names,
        frame_intervals_seconds=frame_intervals_seconds,
        audit_rows=audit_rows,
    )

    audit = pd.DataFrame(audit_rows)
    audit["max_elapsed_minutes"] = args.max_elapsed_minutes
    audit.to_csv(output_root / "injury_feature_audit.csv", index=False)

    print()
    print("============================================================")
    print("MUSC CAP720 MANUAL-INJURY FEATURE EXTRACTION COMPLETE")
    print("============================================================")
    print(output_root / "musc" / "constrained_fish_level_musc_model_a_plus_injury.csv")


if __name__ == "__main__":
    main()
