#!/usr/bin/env python3
"""Generate exploratory plots for features outside the CV-selected union.

These figures are deliberately kept separate from the final-model plots. A
feature is included only when it is available and numeric in untreated, MMP
and liraglutide fish tables, and was never selected in the completed outer-CV
folds. The script reads saved tables and feature lists; it does not train or
modify a model.

Implementation references
-------------------------
Pandas numeric conversion:
https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html

Subprocess execution:
https://docs.python.org/3/library/subprocess.html#subprocess.run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from generate_all_final_model_plots import MODELS


TRACK_ROOTS = {
    "untreated": "qc_outlier_outputs_time_corrected",
    "mmp": "qc_outlier_outputs_time_corrected_mmp",
    "liraglutide": "qc_outlier_outputs_time_corrected_liraglutide",
}
TRACK_DATASET = {
    "macrophage_all_calibrated_svm_plus_injury": "macrophage_all",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--selected-feature-root", type=Path, required=True)
    parser.add_argument("--six-group-output", type=Path, required=True)
    parser.add_argument("--fish-block-output", type=Path, required=True)
    return parser.parse_args()


def common_nonselected_features(
    tables: dict[str, Path], selected_path: Path
) -> list[str]:
    """Return usable fish summaries shared by all three experimental cohorts."""
    shared: set[str] | None = None
    for path in tables.values():
        frame = pd.read_csv(path, low_memory=False)
        candidates = {
            column
            for column in frame.columns
            if column.startswith(("fish_mean__", "fish_median__"))
            and pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2
        }
        shared = candidates if shared is None else shared.intersection(candidates)

    selected_table = pd.read_csv(selected_path)
    feature_column = (
        "feature" if "feature" in selected_table.columns else selected_table.columns[0]
    )
    selected = set(selected_table[feature_column].dropna().astype(str))
    return sorted((shared or set()).difference(selected))


def run(command: list[str], working_dir: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=working_dir, env=environment, check=True)


def main() -> None:
    args = arguments()
    feature_dir = args.feature_extraction_dir.resolve()
    selected_root = args.selected_feature_root.resolve()
    six_group_output = args.six_group_output.resolve()
    fish_block_output = args.fish_block_output.resolve()
    scripts = Path(__file__).resolve().parent
    manifest_rows: list[dict[str, str]] = []

    for model, config in MODELS.items():
        tables = {
            condition: feature_dir / relative
            for condition, relative in config["tables"].items()
        }
        selected_path = (
            selected_root / model / "cv_union_selected_features.csv"
        )
        features = common_nonselected_features(tables, selected_path)
        model_output = six_group_output / model
        model_output.mkdir(parents=True, exist_ok=True)
        feature_file = model_output / "nonselected_exploratory_features.csv"
        pd.DataFrame({"feature": features}).to_csv(feature_file, index=False)
        manifest_rows.extend(
            {
                "model": model,
                "feature": feature,
                "status": "not selected in any completed outer-CV fold",
            }
            for feature in features
        )

        command = [
            sys.executable,
            str(scripts / "make_six_group_fish_distributions.py"),
        ]
        for condition, path in tables.items():
            command.extend(["--table", f"{condition}={path}"])
        command.extend(
            [
                "--feature-file",
                str(feature_file),
                "--dataset-name",
                f"{model} (exploratory non-selected features)",
                "--output-dir",
                str(model_output),
            ]
        )
        run(command, feature_dir)

        # Only the requested macrophage-all folder receives the more detailed
        # track-level fish-block views. Features without a matching track
        # variable remain available in the six-group fish-level collection.
        if model in TRACK_DATASET:
            track_command = [
                sys.executable,
                str(scripts / "make_combined_fish_block_feature_plots.py"),
            ]
            dataset = TRACK_DATASET[model]
            for condition, root in TRACK_ROOTS.items():
                track_path = (
                    feature_dir
                    / root
                    / dataset
                    / "cell_track_features_time_corrected.csv"
                )
                track_command.extend(["--table", f"{condition}={track_path}"])
            track_command.extend(
                [
                    "--feature-file",
                    str(feature_file),
                    "--dataset-name",
                    f"{model} (exploratory non-selected features)",
                    "--output-dir",
                    str(fish_block_output / model),
                ]
            )
            run(track_command, feature_dir)

    six_group_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(
        six_group_output / "nonselected_feature_manifest.csv", index=False
    )


if __name__ == "__main__":
    main()
