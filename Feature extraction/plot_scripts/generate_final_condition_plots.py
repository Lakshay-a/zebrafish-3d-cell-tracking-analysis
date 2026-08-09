#!/usr/bin/env python3
"""Regenerate the dissertation plot set from the final analysis tables."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ("musc", "macrophage_all", "macrophage_outside_boundary")
CONDITIONS = {
    "untreated": {
        "feature_root": "final_feature_outputs",
        "track_root": "qc_outlier_outputs_time_corrected",
        # The retained untreated track keys are stored directly in the final
        # time-corrected tables; the earlier intermediate QC folder was not kept.
        "qc_root": None,
        "metadata": "block_metadata.csv",
        "fish_tables": {
            "musc": "constrained_fish_features_time_corrected/musc/constrained_fish_level_mean_median.csv",
            "macrophage_all": "manual_injury_feature_outputs_time_corrected/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv",
            "macrophage_outside_boundary": "constrained_fish_features_time_corrected/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv",
        },
    },
    "mmp": {
        "feature_root": "final_feature_outputs_mmp",
        "track_root": "qc_outlier_outputs_time_corrected_mmp",
        "qc_root": "qc_outlier_outputs_mmp",
        "metadata": "MMP_metadata.csv",
        "fish_tables": {
            "musc": "constrained_fish_features_time_corrected_mmp/musc/constrained_fish_level_mean_median.csv",
            "macrophage_all": "manual_injury_feature_outputs_time_corrected_mmp/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv",
            "macrophage_outside_boundary": "constrained_fish_features_time_corrected_mmp/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv",
        },
    },
    "liraglutide": {
        "feature_root": "final_feature_outputs_liraglutide",
        "track_root": "qc_outlier_outputs_time_corrected_liraglutide",
        "qc_root": "qc_outlier_outputs_liraglutide",
        "metadata": "Liraglutide_metadata.csv",
        "fish_tables": {
            "musc": "constrained_fish_features_time_corrected_liraglutide/musc/constrained_fish_level_mean_median.csv",
            "macrophage_all": "manual_injury_feature_outputs_time_corrected_liraglutide/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv",
            "macrophage_outside_boundary": "constrained_fish_features_time_corrected_liraglutide/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv",
        },
    },
}

MODEL_FEATURES = {
    "musc": "frozen_untreated_models/musc_model_a_no_injury_all_fish/frozen_model_coefficients.csv",
    "macrophage_all": "frozen_untreated_models/macrophage_all_model_b_plus_injury/frozen_model_coefficients.csv",
    "macrophage_outside_boundary": "frozen_untreated_models/macrophage_outside_boundary_model_b/frozen_model_coefficients.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matched plots for untreated, MMP-inhibited and liraglutide cohorts."
    )
    parser.add_argument(
        "--feature-extraction-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing the final feature tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: <feature-extraction-dir>/plots_final_conditions).",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=sorted(CONDITIONS),
        help="Condition(s) to run; omit to run all three.",
    )
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required final analysis files are missing:\n{joined}")


def run(command: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_extraction_dir.resolve()
    scripts = Path(__file__).resolve().parent
    output_root = (args.output_dir or feature_dir / "plots_final_conditions").resolve()
    conditions = args.condition or list(CONDITIONS)

    for condition in conditions:
        config = CONDITIONS[condition]
        condition_out = output_root / condition
        feature_root = feature_dir / str(config["feature_root"])
        metadata = feature_dir / str(config["metadata"])
        track_tables = {
            dataset: feature_dir
            / str(config["track_root"])
            / dataset
            / "cell_track_features_time_corrected.csv"
            for dataset in DATASETS
        }
        qc_tables = (
            {
                dataset: feature_dir
                / str(config["qc_root"])
                / dataset
                / "cell_track_features_qc_filtered.csv"
                for dataset in DATASETS
            }
            if config["qc_root"] is not None
            else track_tables
        )
        fish_tables = {
            dataset: feature_dir / relative
            for dataset, relative in dict(config["fish_tables"]).items()
        }
        require_files(
            [metadata, *track_tables.values(), *qc_tables.values(), *fish_tables.values()]
        )

        common_dataset_args = [
            item for dataset in DATASETS for item in ("--dataset", dataset)
        ]
        qc_args = [
            item
            for dataset in DATASETS
            for item in ("--qc-track-table", f"{dataset}={qc_tables[dataset]}")
        ]

        run(
            [
                sys.executable,
                str(scripts / "make_msd_directionality_plots.py"),
                "--root",
                str(feature_root),
                *common_dataset_args,
                *qc_args,
                "--metadata-file",
                str(metadata),
                "--output-dir",
                str(condition_out / "msd_directionality"),
            ],
            feature_dir,
        )
        for script, subdir in (
            ("make_shape_over_time_plots.py", "shape_over_time"),
            ("make_feret_proxy_plots.py", "feret_proxy"),
        ):
            run(
                [
                    sys.executable,
                    str(scripts / script),
                    "--root",
                    str(feature_root),
                    *common_dataset_args,
                    *qc_args,
                    "--output-dir",
                    str(condition_out / subdir),
                ],
                feature_dir,
            )

        labelled_tracks = [
            item
            for dataset in DATASETS
            for item in ("--dataset", f"{dataset}={track_tables[dataset]}")
        ]
        run(
            [
                sys.executable,
                str(scripts / "make_shape_change_plots.py"),
                *labelled_tracks,
                "--output-dir",
                str(condition_out / "shape_change"),
            ],
            feature_dir,
        )
        run(
            [
                sys.executable,
                str(scripts / "make_fish_variability_plots.py"),
                *labelled_tracks,
                "--output-dir",
                str(condition_out / "fish_variability"),
            ],
            feature_dir,
        )

        cell_args = [
            item
            for dataset in DATASETS
            for item in ("--cell-table", f"{dataset}={track_tables[dataset]}")
        ]
        fish_args = [
            item
            for dataset in DATASETS
            for item in ("--fish-table", f"{dataset}={fish_tables[dataset]}")
        ]
        run(
            [
                sys.executable,
                str(scripts / "make_z_movement_and_grouped_fish_plots.py"),
                *cell_args,
                *fish_args,
                "--output-dir",
                str(condition_out / "z_movement_and_fish_features"),
            ],
            feature_dir,
        )

        for dataset in DATASETS:
            feature_file = feature_dir / MODEL_FEATURES[dataset]
            require_files([feature_file])
            for script in (
                "make_pca_component_loading_plots.py",
                "make_pca_feature_contribution_plots.py",
            ):
                run(
                    [
                        sys.executable,
                        str(scripts / script),
                        "--input",
                        str(fish_tables[dataset]),
                        "--feature-file",
                        str(feature_file),
                        "--dataset-name",
                        f"{condition}: {dataset}",
                        "--output-dir",
                        str(condition_out / "final_model_pca"),
                    ],
                    feature_dir,
                )

    print(f"[DONE] Final-condition plots saved under: {output_root}")


if __name__ == "__main__":
    main()
