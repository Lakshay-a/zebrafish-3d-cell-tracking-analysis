#!/usr/bin/env python3
"""Plot every feature selected in at least one completed outer CV fold."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from generate_all_final_model_plots import MODELS


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> None:
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.feature_extraction_dir.resolve()
    output = args.output_dir.resolve()
    scripts = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")

    for model_name, config in MODELS.items():
        stability_path = root / str(config["result"]) / "feature_selection_stability.csv"
        stability = pd.read_csv(stability_path)
        frequency_column = (
            "model_feature_frequency"
            if "model_feature_frequency" in stability.columns
            else "nonzero_selection_frequency"
        )
        frequency = pd.to_numeric(stability[frequency_column], errors="coerce").fillna(0)
        selected = stability.loc[frequency.gt(0), ["feature", frequency_column]].copy()
        selected = selected.sort_values(frequency_column, ascending=False).reset_index(drop=True)
        selected.insert(0, "selected_order", range(1, len(selected) + 1))

        model_output = output / model_name
        feature_file = model_output / "cv_union_selected_features.csv"
        model_output.mkdir(parents=True, exist_ok=True)
        selected.to_csv(feature_file, index=False)

        tables = dict(config["tables"])
        for condition, relative_table in tables.items():
            condition_output = model_output / "condition_specific_pca" / condition
            label = f"{model_name}: {condition}: all CV-selected features"
            for plot_script in (
                "make_pca_component_loading_plots.py",
                "make_pca_feature_contribution_plots.py",
            ):
                run(
                    [
                        sys.executable,
                        str(scripts / plot_script),
                        "--input",
                        str(root / relative_table),
                        "--feature-file",
                        str(feature_file),
                        "--dataset-name",
                        label,
                        "--output-dir",
                        str(condition_output),
                        "--top-n-features",
                        str(len(selected)),
                    ],
                    root,
                    environment,
                )

        run(
            [
                sys.executable,
                str(scripts / "make_frozen_untreated_pca_projection.py"),
                "--untreated",
                str(root / tables["untreated"]),
                "--treated",
                f"mmp={root / tables['mmp']}",
                "--treated",
                f"liraglutide={root / tables['liraglutide']}",
                "--coefficient-file",
                str(feature_file),
                "--dataset-name",
                f"{model_name}: all CV-selected features",
                "--feature-set-label",
                "Every feature selected in at least one completed outer CV fold",
                "--output-prefix",
                "cv_union_untreated_pca",
                "--output-dir",
                str(model_output / "fixed_untreated_pca"),
            ],
            root,
            environment,
        )


if __name__ == "__main__":
    main()
