from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize nested LOFO classifier result folders."
    )
    parser.add_argument(
        "--root",
        default="constrained_separation_results_time_corrected",
        help="Root results folder to scan.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for the combined summary table.",
    )
    return parser.parse_args()


def read_permutation(metrics_path: Path) -> float:
    permutation_path = metrics_path.parent / "permutation_test.csv"

    if not permutation_path.exists():
        return float("nan")

    if permutation_path.stat().st_size == 0:
        return float("nan")

    try:
        table = pd.read_csv(permutation_path)
    except pd.errors.EmptyDataError:
        return float("nan")
    except Exception as exc:
        print(
            f"[WARN] Could not read permutation file "
            f"{permutation_path}: {exc}"
        )
        return float("nan")

    if table.empty:
        return float("nan")

    possible_columns = [
        "permutation_p_value",
        "p_value",
        "permutation_p",
    ]

    for column in possible_columns:
        if column in table.columns:
            value = pd.to_numeric(
                pd.Series([table.iloc[0][column]]),
                errors="coerce",
            ).iloc[0]
            return float(value) if pd.notna(value) else float("nan")

    return float("nan")


def print_plain_table(table: pd.DataFrame) -> None:
    text_table = table.fillna("").astype(str)
    widths = {
        column: max(len(column), int(text_table[column].map(len).max()))
        for column in text_table.columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in text_table.columns)
    rule = "-+-".join("-" * widths[column] for column in text_table.columns)
    print(header)
    print(rule)
    for _, row in text_table.iterrows():
        print(" | ".join(row[column].ljust(widths[column]) for column in text_table.columns))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: list[dict[str, object]] = []

    metrics_files = sorted(root.rglob("nested_lofo_metrics.csv"))

    print(f"[INFO] Searching under: {root.resolve()}")
    print(f"[INFO] Metrics files found: {len(metrics_files)}")

    for metrics_path in metrics_files:
        parts = metrics_path.relative_to(root).parts[:-1]

        # Current structure:
        # root / method / dataset / nested_lofo_metrics.csv
        if len(parts) == 2:
            method, dataset = parts
            model = method

        # Also support:
        # root / method / model / dataset / nested_lofo_metrics.csv
        elif len(parts) >= 3:
            method = parts[0]
            model = parts[1]
            dataset = "/".join(parts[2:])

        else:
            print(
                f"[WARN] Unexpected results path structure: {metrics_path}"
            )
            continue

        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            continue

        row = metrics.iloc[0].to_dict()
        row.update(
            {
                "method": method,
                "model": model,
                "dataset": dataset,
                "permutation_p_value": read_permutation(metrics_path),
                "results_dir": str(metrics_path.parent),
            }
        )
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No nested_lofo_metrics.csv files found under {root.resolve()}"
        )

    summary = pd.DataFrame(rows)
    display_cols = [
        "method",
        "model",
        "dataset",
        "n_fish",
        "balanced_accuracy",
        "roc_auc",
        "permutation_p_value",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "results_dir",
    ]
    display_cols = [col for col in display_cols if col in summary.columns]
    summary = summary[display_cols].sort_values(
        ["balanced_accuracy", "roc_auc"],
        ascending=False,
    )

    for col in ["balanced_accuracy", "roc_auc", "permutation_p_value"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").round(3)

    print_plain_table(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
