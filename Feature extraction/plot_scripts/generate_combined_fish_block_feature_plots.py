#!/usr/bin/env python3
"""Generate combined untreated, MMP and liraglutide fish-block plots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from generate_all_final_model_plots import MODELS


TRACK_DATASET = {
    "musc_calibrated_svm_plus_injury": "musc",
    "musc_legacy_l1": "musc",
    "macrophage_all_calibrated_svm_plus_injury": "macrophage_all",
    "macrophage_outside_boundary_calibrated_svm": "macrophage_outside_boundary",
}
TRACK_ROOT = {
    "untreated": "qc_outlier_outputs_time_corrected",
    "mmp": "qc_outlier_outputs_time_corrected_mmp",
    "liraglutide": "qc_outlier_outputs_time_corrected_liraglutide",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    script = Path(__file__).with_name("make_combined_fish_block_feature_plots.py")
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    # These commands read frozen feature lists and final track tables only.
    # They do not fit, tune or otherwise modify any classifier.
    for model in MODELS:
        dataset = TRACK_DATASET[model]
        command = [sys.executable, str(script)]
        for condition, root in TRACK_ROOT.items():
            table = (
                args.feature_extraction_dir
                / root
                / dataset
                / "cell_track_features_time_corrected.csv"
            )
            command.extend(["--table", f"{condition}={table}"])
        command.extend(
            [
                "--feature-file",
                str(args.feature_root / model / "cv_union_selected_features.csv"),
                "--dataset-name",
                model,
                "--output-dir",
                str(args.output_dir / model),
            ]
        )
        print("[RUN]", " ".join(map(str, command)), flush=True)
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
