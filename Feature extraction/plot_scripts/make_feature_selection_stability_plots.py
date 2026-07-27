#!/usr/bin/env python3
"""Plot nested-LOFO feature-selection frequencies from a completed model run."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot feature-selection stability from an existing result CSV."
    )
    parser.add_argument("--stability", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def feature_label(feature: str) -> str:
    text = feature.replace("fish_mean__", "Mean: ").replace(
        "fish_median__", "Median: "
    )
    replacements = {
        "_um2_per_min": " (µm²/min)",
        "_um_per_min": " (µm/min)",
        "_um3": " (µm³)",
        "_um2": " (µm²)",
        "_um": " (µm)",
        "_3d": " 3D",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.stability)
    frequency_columns = [
        column
        for column in [
            "nonzero_selection_frequency",
            "model_feature_frequency",
            "top_k_frequency",
        ]
        if column in table.columns
    ]
    if not frequency_columns or "feature" not in table.columns:
        raise ValueError("Stability CSV needs a feature and selection-frequency column.")

    primary = frequency_columns[0]
    table = table.copy()
    for column in frequency_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0)
    table = table.sort_values(frequency_columns, ascending=False).head(args.top_n)
    table = table.sort_values(primary, ascending=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = safe_name(args.dataset_name)
    table.to_csv(
        output_dir / f"feature_selection_stability__{basename}.csv", index=False
    )

    fig, ax = plt.subplots(
        figsize=(10.5, max(5.8, 0.42 * len(table) + 1.8))
    )
    colors = [
        "#2f7d32" if value >= 0.8 else "#e69f00" if value >= 0.5 else "#8c8c8c"
        for value in table[primary]
    ]
    bars = ax.barh(
        range(len(table)), table[primary], color=colors, edgecolor="black", linewidth=0.4
    )
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(0.8, color="#1b5e20", linestyle=":", linewidth=1.0)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([feature_label(value) for value in table["feature"]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Outer-fold selection frequency")
    ax.set_title(f"{args.dataset_name}: feature-selection stability")
    ax.grid(axis="x", alpha=0.2)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in table[primary]], padding=3)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"feature_selection_stability__{basename}.png",
        dpi=280,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
