#!/usr/bin/env python3
"""Experiment 4."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of finite numbers."
        )
    return values


def float_list_to_text(values: list[float]) -> str:
    return ",".join(str(value) for value in values)


def parse_labelled_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=CSV, for example untreated=fish_table.csv"
        )
    label, path_text = text.split("=", 1)
    label = label.strip()
    path = Path(path_text.strip())
    if not label:
        raise argparse.ArgumentTypeError("Condition label cannot be empty.")
    if not path_text.strip():
        raise argparse.ArgumentTypeError("CSV path cannot be empty.")
    return label, path


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return cleaned.strip("_") or "condition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run separate nested-LOFO WT-versus-MUT models within each "
            "biological condition."
        )
    )
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        type=parse_labelled_path,
        metavar="LABEL=CSV",
        help="Condition-labelled fish-level CSV. Repeat as needed.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-prefix", default="dataset")
    parser.add_argument(
        "--classifier-script",
        default="06_test_constrained_fish_separation_final.py",
    )
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument(
        "--model",
        choices=["l1", "elasticnet"],
        default="l1",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)
    parser.add_argument(
        "--selection-frequency-threshold",
        type=float,
        default=0.40,
    )
    parser.add_argument("--max-stable-features", type=int, default=8)
    parser.add_argument(
        "--c-grid",
        type=parse_float_list,
        default=parse_float_list("0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100"),
    )
    parser.add_argument(
        "--l1-ratio-grid",
        type=parse_float_list,
        default=parse_float_list("0.10,0.25,0.50,0.75,0.90"),
    )
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--show-convergence-warnings", action="store_true")
    return parser.parse_args()


def load_classifier_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"Classifier script was not found: {script_path.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        "existing_fish_classifier_for_wrapper",
        script_path.resolve(),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import classifier script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_condition_tables(paths: list[Path], label: str) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path, low_memory=False)
        if table.empty:
            raise ValueError(f"Input table is empty: {path}")
        table = table.copy()
        table["__source_csv"] = str(path)
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True, sort=False)
    print(
        f"[INFO] {label}: loaded {len(combined)} rows "
        f"from {len(paths)} file(s)."
    )
    return combined


def read_single_row(path: Path) -> dict[str, object]:
    table = pd.read_csv(path)
    if table.empty:
        return {}
    return table.iloc[0].to_dict()


def selected_feature_set(
    stability: pd.DataFrame,
    threshold: float,
) -> set[str]:
    if stability.empty:
        return set()
    selected = stability.loc[
        stability["nonzero_selection_frequency"] >= threshold,
        "feature",
    ]
    return set(selected.astype(str))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    classifier_script = Path(args.classifier_script)
    classifier = load_classifier_module(classifier_script)

    condition_paths: dict[str, list[Path]] = {}
    for label, path in args.condition:
        condition_paths.setdefault(label, []).append(path)

    audit_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    combined_stability_tables: list[pd.DataFrame] = []
    combined_prediction_tables: list[pd.DataFrame] = []
    condition_selected_sets: dict[str, set[str]] = {}

    for condition_index, (condition_label, paths) in enumerate(
        condition_paths.items(),
        start=1,
    ):
        condition_name = safe_name(condition_label)
        condition_dir = output_root / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        table = load_condition_tables(paths, condition_label)
        fish_col = classifier.detect_column(
            table,
            args.fish_col,
            classifier.FISH_COLUMN_CANDIDATES,
            "fish",
        )
        genotype_col = classifier.detect_column(
            table,
            args.genotype_col,
            classifier.GENOTYPE_COLUMN_CANDIDATES,
            "genotype",
        )

        table = table.copy()
        table[fish_col] = table[fish_col].astype(str).str.strip()
        table[genotype_col] = table[genotype_col].map(
            classifier.normalise_genotype
        )
        table = table[
            table[genotype_col].isin([args.group_a, args.group_b])
        ].copy()
        table = table.sort_values(
            [genotype_col, fish_col]
        ).reset_index(drop=True)

        counts = table[genotype_col].value_counts().to_dict()
        n_group_a = int(counts.get(args.group_a, 0))
        n_group_b = int(counts.get(args.group_b, 0))
        warning = ""

        if min(n_group_a, n_group_b) < 2:
            status = "not_run_insufficient_fish"
            warning = (
                "Nested LOFO requires at least two fish in each genotype so "
                "that both classes remain in every outer training fold."
            )
            audit_records.append(
                {
                    "condition": condition_label,
                    "status": status,
                    "n_fish": len(table),
                    f"n_{args.group_a}": n_group_a,
                    f"n_{args.group_b}": n_group_b,
                    "warning": warning,
                    "input_files": "|".join(str(path) for path in paths),
                }
            )
            print(f"[WARN] {condition_label}: {warning}")
            continue

        if min(n_group_a, n_group_b) < 5:
            warning = (
                "Model run completed, but feature selection and performance "
                "should be described as exploratory because one or both "
                "genotype groups contain fewer than five fish."
            )

        condition_input = condition_dir / "condition_fish_table.csv"
        table.to_csv(condition_input, index=False)

        dataset_name = (
            f"{args.dataset_prefix}_{condition_name}_{args.model}"
        )
        command = [
            sys.executable,
            "-u",
            str(classifier_script),
            "--input",
            str(condition_input),
            "--output-dir",
            str(condition_dir),
            "--dataset-name",
            dataset_name,
            "--fish-col",
            fish_col,
            "--genotype-col",
            genotype_col,
            "--group-a",
            args.group_a,
            "--group-b",
            args.group_b,
            "--model",
            args.model,
            "--correlation-threshold",
            str(args.correlation_threshold),
            "--top-k",
            str(args.top_k),
            "--max-missing-fraction",
            str(args.max_missing_fraction),
            "--selection-frequency-threshold",
            str(args.selection_frequency_threshold),
            "--max-stable-features",
            str(args.max_stable_features),
            "--c-grid",
            float_list_to_text(args.c_grid),
            "--l1-ratio-grid",
            float_list_to_text(args.l1_ratio_grid),
            "--permutations",
            str(args.permutations),
            "--random-seed",
            str(args.random_seed + condition_index - 1),
        ]
        if args.no_progress:
            command.append("--no-progress")
        if args.show_convergence_warnings:
            command.append("--show-convergence-warnings")

        print()
        print("=" * 72)
        print(f"RUNNING CONDITION MODEL: {condition_label}")
        print("=" * 72)
        subprocess.run(command, check=True)

        metrics_path = condition_dir / "nested_lofo_metrics.csv"
        permutation_path = condition_dir / "permutation_test.csv"
        stability_path = condition_dir / "feature_selection_stability.csv"
        predictions_path = condition_dir / "nested_lofo_predictions.csv"

        metrics = read_single_row(metrics_path)
        permutation = read_single_row(permutation_path)
        stability = pd.read_csv(stability_path)
        predictions = pd.read_csv(predictions_path)

        stability.insert(0, "condition", condition_label)
        predictions.insert(0, "condition", condition_label)
        combined_stability_tables.append(stability)
        combined_prediction_tables.append(predictions)

        selected_set = selected_feature_set(
            stability,
            args.selection_frequency_threshold,
        )
        condition_selected_sets[condition_label] = selected_set

        summary_records.append(
            {
                "condition": condition_label,
                "model_type": args.model,
                "n_fish": int(metrics.get("n_fish", len(table))),
                f"n_{args.group_a}": n_group_a,
                f"n_{args.group_b}": n_group_b,
                "balanced_accuracy": metrics.get(
                    "balanced_accuracy",
                    np.nan,
                ),
                "roc_auc": metrics.get("roc_auc", np.nan),
                "permutation_p_value": permutation.get(
                    "permutation_p_value",
                    np.nan,
                ),
                "stable_feature_count_at_threshold": len(selected_set),
                "stable_features_at_threshold": "|".join(
                    sorted(selected_set)
                ),
                "interpretation_warning": warning,
            }
        )
        audit_records.append(
            {
                "condition": condition_label,
                "status": "completed",
                "n_fish": len(table),
                f"n_{args.group_a}": n_group_a,
                f"n_{args.group_b}": n_group_b,
                "warning": warning,
                "input_files": "|".join(str(path) for path in paths),
            }
        )

    audit = pd.DataFrame(audit_records)
    audit.to_csv(
        output_root / "condition_input_audit.csv",
        index=False,
    )

    summary = pd.DataFrame(summary_records)
    summary.to_csv(
        output_root / "condition_model_summary.csv",
        index=False,
    )

    if combined_stability_tables:
        combined_stability = pd.concat(
            combined_stability_tables,
            ignore_index=True,
            sort=False,
        )
        combined_stability.to_csv(
            output_root / "combined_feature_stability.csv",
            index=False,
        )

        frequency_wide = combined_stability.pivot_table(
            index="feature",
            columns="condition",
            values="nonzero_selection_frequency",
            aggfunc="first",
        ).reset_index()
        frequency_wide.to_csv(
            output_root / "feature_selection_frequency_by_condition.csv",
            index=False,
        )

        coefficient_wide = combined_stability.pivot_table(
            index="feature",
            columns="condition",
            values="mean_signed_coefficient_when_selected",
            aggfunc="first",
        ).reset_index()
        coefficient_wide.to_csv(
            output_root / "mean_coefficient_by_condition.csv",
            index=False,
        )

    if combined_prediction_tables:
        combined_predictions = pd.concat(
            combined_prediction_tables,
            ignore_index=True,
            sort=False,
        )
        combined_predictions.to_csv(
            output_root / "combined_nested_lofo_predictions.csv",
            index=False,
        )

    overlap_records: list[dict[str, object]] = []
    for condition_a, condition_b in itertools.combinations(
        condition_selected_sets,
        2,
    ):
        features_a = condition_selected_sets[condition_a]
        features_b = condition_selected_sets[condition_b]
        union = features_a | features_b
        intersection = features_a & features_b
        jaccard = (
            len(intersection) / len(union)
            if union
            else np.nan
        )
        overlap_records.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "selected_count_a": len(features_a),
                "selected_count_b": len(features_b),
                "shared_selected_count": len(intersection),
                "union_selected_count": len(union),
                "jaccard_overlap": jaccard,
                "shared_selected_features": "|".join(
                    sorted(intersection)
                ),
                "only_condition_a": "|".join(
                    sorted(features_a - features_b)
                ),
                "only_condition_b": "|".join(
                    sorted(features_b - features_a)
                ),
            }
        )

    pd.DataFrame(overlap_records).to_csv(
        output_root / "pairwise_selected_feature_overlap.csv",
        index=False,
    )

    print()
    print("[DONE] Experiment 4 outputs saved to:")
    print(f"       {output_root}")
    if not summary.empty:
        print()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
