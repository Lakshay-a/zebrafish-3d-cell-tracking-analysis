#!/usr/bin/env python3
"""Run untreated-fitted PCA projections for the four final frozen models."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from generate_all_final_model_plots import MODELS


COEFFICIENT_DIRS = {
    "musc_calibrated_svm_plus_injury": "musc_model_a_plus_injury",
    "macrophage_outside_boundary_calibrated_svm": "macrophage_outside_boundary_model_b",
    "macrophage_all_calibrated_svm_plus_injury": "macrophage_all_model_b_plus_injury",
    "musc_legacy_l1": "musc_model_a_no_injury_drop2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.feature_extraction_dir.resolve()
    script = Path(__file__).resolve().parent / "make_frozen_untreated_pca_projection.py"
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

    for model_name, config in MODELS.items():
        tables = dict(config["tables"])
        coefficients = (
            root
            / "frozen_untreated_models"
            / COEFFICIENT_DIRS[model_name]
            / "frozen_model_coefficients.csv"
        )
        command = [
            sys.executable,
            str(script),
            "--untreated",
            str(root / tables["untreated"]),
            "--treated",
            f"mmp={root / tables['mmp']}",
            "--treated",
            f"liraglutide={root / tables['liraglutide']}",
            "--coefficient-file",
            str(coefficients),
            "--dataset-name",
            model_name,
            "--output-dir",
            str(args.output_dir.resolve() / model_name / "frozen_untreated_pca"),
        ]
        print(f"[RUN] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
