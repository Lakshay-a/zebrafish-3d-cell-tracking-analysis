#!/usr/bin/env python3
"""Generate static plots from the four completed best-model result sets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MODELS = {
    "musc_calibrated_svm_plus_injury": {
        "result": "final_best_models_1000_perm/calibrated_linear_svm/musc_model_a_plus_injury_drop_260511_block07_260511_block05_260427_block02",
        "tables": {
            "untreated": "subset_sensitivity_results/musc_model_a_plus_injury_svm_linear_top4_combo/_subset_inputs/drop_260511_block07__260511_block05__260427_block02.csv",
            "mmp": "manual_injury_feature_outputs_time_corrected_mmp/musc/constrained_fish_level_musc_model_a_plus_injury.csv",
            "liraglutide": "manual_injury_feature_outputs_time_corrected_liraglutide/musc/constrained_fish_level_musc_model_a_plus_injury.csv",
        },
    },
    "macrophage_outside_boundary_calibrated_svm": {
        "result": "final_best_models_1000_perm/calibrated_linear_svm/macrophage_outside_boundary_model_b",
        "tables": {
            "untreated": "constrained_fish_features_time_corrected/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv",
            "mmp": "constrained_fish_features_time_corrected_mmp/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv",
            "liraglutide": "constrained_fish_features_time_corrected_liraglutide/model_b/macrophage_outside_boundary/constrained_fish_level_mean_median.csv",
        },
    },
    "macrophage_all_calibrated_svm_plus_injury": {
        "result": "final_best_models_1000_perm/calibrated_linear_svm/macrophage_all_model_b_plus_injury",
        "tables": {
            "untreated": "manual_injury_feature_outputs_time_corrected/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv",
            "mmp": "manual_injury_feature_outputs_time_corrected_mmp/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv",
            "liraglutide": "manual_injury_feature_outputs_time_corrected_liraglutide/macrophage/constrained_fish_level_macrophage_all_model_b_plus_injury.csv",
        },
    },
    "musc_legacy_l1": {
        "result": "final_best_models_1000_perm/legacy_l1/musc_model_a_drop_20240422_block06_20240422_block08",
        "tables": {
            "untreated": "subset_sensitivity_results/musc_model_a_legacy_l1_top5_combo/_subset_inputs/drop_20240422_block06__20240422_block08.csv",
            "mmp": "constrained_fish_features_time_corrected_mmp/musc/constrained_fish_level_mean_median.csv",
            "liraglutide": "constrained_fish_features_time_corrected_liraglutide/musc/constrained_fish_level_mean_median.csv",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    args = parse_args()
    root = args.feature_extraction_dir.resolve()
    output = args.output_dir.resolve()
    scripts = Path(__file__).resolve().parent

    for model_name, config in MODELS.items():
        result_dir = root / str(config["result"])
        stability = result_dir / "feature_selection_stability.csv"
        selected = result_dir / "stable_selected_features.csv"
        for required in (stability, selected):
            if not required.is_file():
                raise FileNotFoundError(required)

        model_out = output / model_name
        run(
            [
                sys.executable,
                str(scripts / "make_feature_selection_stability_plots.py"),
                "--stability",
                str(stability),
                "--dataset-name",
                model_name,
                "--output-dir",
                str(model_out / "untreated_training_stability"),
            ],
            root,
        )

        for condition, relative_table in dict(config["tables"]).items():
            table = root / relative_table
            if not table.is_file():
                raise FileNotFoundError(table)
            label = f"{model_name}: {condition}"
            condition_out = model_out / condition

            run(
                [
                    sys.executable,
                    str(scripts / "make_directional_feature_stability_plots.py"),
                    "--stability",
                    str(stability),
                    "--feature-table",
                    str(table),
                    "--dataset-name",
                    label,
                    "--output-dir",
                    str(condition_out / "directional_stability"),
                ],
                root,
            )
            for script in (
                "make_pca_component_loading_plots.py",
                "make_pca_feature_contribution_plots.py",
            ):
                run(
                    [
                        sys.executable,
                        str(scripts / script),
                        "--input",
                        str(table),
                        "--feature-file",
                        str(selected),
                        "--dataset-name",
                        label,
                        "--output-dir",
                        str(condition_out / "pca"),
                    ],
                    root,
                )

    print(f"[DONE] All final-model plots saved under: {output}")


if __name__ == "__main__":
    main()
