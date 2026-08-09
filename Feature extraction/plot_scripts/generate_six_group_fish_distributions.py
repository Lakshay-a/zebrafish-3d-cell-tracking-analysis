#!/usr/bin/env python3
"""Generate six-group fish distributions for all four final model definitions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from generate_all_final_model_plots import MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.feature_extraction_dir.resolve()
    feature_root = args.feature_root.resolve()
    output = args.output_dir.resolve()
    script = Path(__file__).resolve().parent / "make_six_group_fish_distributions.py"
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")

    for model_name, config in MODELS.items():
        tables = dict(config["tables"])
        feature_file = feature_root / model_name / "cv_union_selected_features.csv"
        command = [
            sys.executable,
            str(script),
            "--table",
            f"untreated={root / tables['untreated']}",
            "--table",
            f"mmp={root / tables['mmp']}",
            "--table",
            f"liraglutide={root / tables['liraglutide']}",
            "--feature-file",
            str(feature_file),
            "--dataset-name",
            model_name,
            "--output-dir",
            str(output / model_name),
        ]
        print(f"[RUN] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
