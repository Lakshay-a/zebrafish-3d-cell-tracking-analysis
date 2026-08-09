#!/usr/bin/env python3
"""Show how often selected features favour WT or mutant fish across CV folds."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


GROUP_COLORS = {
    "WT": "#1f77b4",
    "MUT": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot feature-selection stability with WT/MUT coefficient direction."
    )
    parser.add_argument("--stability", required=True)
    parser.add_argument("--feature-table", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument("--top-n", type=int, default=18)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


def feature_label(feature: str) -> str:
    text = feature.replace("fish_mean__", "Mean: ").replace("fish_median__", "Median: ")
    replacements = {
        "_um2_per_min": " (um2/min)",
        "_um_per_min": " (um/min)",
        "_um_per_frame": " (um/frame)",
        "_um3": " (um3)",
        "_um2": " (um2)",
        "_um": " (um)",
        "_3d": " 3D",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def make_plot(
    stability_path: Path,
    feature_table_path: Path | None,
    dataset_name: str,
    output_dir: Path,
    group_a: str,
    group_b: str,
    top_n: int,
) -> Path:
    table = pd.read_csv(stability_path)
    required = {
        "feature",
        "nonzero_selection_frequency",
        "mean_absolute_coefficient_when_selected",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{stability_path} is missing {sorted(missing)}. "
        )

    table = table.copy()
    table["selection_frequency"] = pd.to_numeric(
        table["nonzero_selection_frequency"], errors="coerce"
    ).fillna(0)
    table["abs_coefficient"] = pd.to_numeric(
        table["mean_absolute_coefficient_when_selected"], errors="coerce"
    ).fillna(0)
    direction_label = "coefficient"
    direction_column = "signed_coefficient"
    if feature_table_path is not None:
        features = pd.read_csv(feature_table_path, low_memory=False)
        genotype_col = next(
            (column for column in ["genotype", "group", "condition", "class", "label"] if column in features.columns),
            None,
        )
        if genotype_col is None:
            raise ValueError(f"Could not find genotype column in {feature_table_path}")
        features[genotype_col] = features[genotype_col].astype(str).str.upper().str.replace("WILD TYPE", "WT")
        direction_records: list[dict[str, float | str]] = []
        for feature in table["feature"]:
            if feature not in features.columns:
                direction_records.append({"feature": feature, "direction_value": 0.0, "effect_size_for_point": 0.0})
                continue
            values = pd.to_numeric(features[feature], errors="coerce")
            a = values[features[genotype_col].eq(group_a.upper())].dropna().to_numpy(float)
            b = values[features[genotype_col].eq(group_b.upper())].dropna().to_numpy(float)
            if len(a) == 0 or len(b) == 0:
                direction_records.append({"feature": feature, "direction_value": 0.0, "effect_size_for_point": 0.0})
                continue
            pooled_iqr = np.nanpercentile(np.concatenate([a, b]), 75) - np.nanpercentile(np.concatenate([a, b]), 25)
            if not np.isfinite(pooled_iqr) or pooled_iqr <= 0:
                pooled_iqr = np.nanstd(np.concatenate([a, b]))
            if not np.isfinite(pooled_iqr) or pooled_iqr <= 0:
                pooled_iqr = 1.0
            median_difference = float(np.nanmedian(b) - np.nanmedian(a))
            direction_records.append(
                {
                    "feature": feature,
                    "direction_value": median_difference,
                    "effect_size_for_point": abs(median_difference) / pooled_iqr,
                }
            )
        direction_table = pd.DataFrame(direction_records)
        table = table.merge(direction_table, on="feature", how="left")
        direction_label = "genotype median"
        direction_column = "median_difference_group_b_minus_group_a"
    elif "mean_signed_coefficient_when_selected" in table.columns:
        table["direction_value"] = pd.to_numeric(
            table["mean_signed_coefficient_when_selected"], errors="coerce"
        ).fillna(0)
        table["effect_size_for_point"] = table["abs_coefficient"]
        direction_label = "coefficient"
        direction_column = "signed_coefficient"
    else:
        raise ValueError(
            f"{stability_path} has no signed coefficient column. Provide --feature-table "
            "to plot WT/MUT direction by feature median difference."
        )
    table["direction_value"] = table["direction_value"].fillna(0)
    table["effect_size_for_point"] = table["effect_size_for_point"].fillna(0)
    table["directional_frequency"] = np.sign(table["direction_value"]) * table["selection_frequency"]
    table = table.sort_values(
        ["selection_frequency", "abs_coefficient"], ascending=False
    ).head(top_n)
    table = table.sort_values("directional_frequency")

    y = np.arange(len(table))
    colors = [
        GROUP_COLORS.get(group_b, "#d62728") if value > 0 else GROUP_COLORS.get(group_a, "#1f77b4")
        for value in table["direction_value"]
    ]
    sizes = 55 + 190 * (
        table["effect_size_for_point"] / table["effect_size_for_point"].max()
        if table["effect_size_for_point"].max() > 0
        else 0
    )

    fig, ax = plt.subplots(figsize=(10.8, max(6.0, 0.42 * len(table) + 2.0)))
    ax.axvline(0, color="black", linewidth=1.1)
    ax.hlines(y, 0, table["directional_frequency"], color=colors, linewidth=2.4, alpha=0.85)
    ax.scatter(
        table["directional_frequency"],
        y,
        s=sizes,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        zorder=3,
    )

    for yy, row in zip(y, table.itertuples(index=False)):
        ax.text(
            row.directional_frequency,
            yy + 0.17,
            f"{row.selection_frequency:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(feature) for feature in table["feature"]])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(
        f"Selection frequency with genotype direction "
        f"(left = {group_a}-higher, right = {group_b}-higher)"
    )
    ax.set_title(f"{dataset_name}: directional feature-selection stability")
    ax.grid(axis="x", alpha=0.22)
    handles = [
        Line2D([0], [0], color=GROUP_COLORS.get(group_a, "#1f77b4"), marker="o",
               markerfacecolor=GROUP_COLORS.get(group_a, "#1f77b4"),
               markeredgecolor="black", linewidth=2.4, label=f"{group_a}-higher feature"),
        Line2D([0], [0], color=GROUP_COLORS.get(group_b, "#d62728"), marker="o",
               markerfacecolor=GROUP_COLORS.get(group_b, "#d62728"),
               markeredgecolor="black", linewidth=2.4, label=f"{group_b}-higher feature"),
        Line2D([0], [0], color="black", marker="o", markerfacecolor="white",
               linestyle="None", markersize=8, label=f"Point size = {direction_label} effect"),
    ]
    ax.legend(handles=handles, frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"directional_feature_stability__{safe_name(dataset_name)}.png"
    fig.savefig(output_path, dpi=260)
    plt.close(fig)

    table[
        [
            "feature",
            "selection_frequency",
            "direction_value",
            "abs_coefficient",
            "effect_size_for_point",
            "directional_frequency",
        ]
    ].rename(columns={"direction_value": direction_column}).to_csv(
        output_dir / f"directional_feature_stability__{safe_name(dataset_name)}.csv", index=False
    )
    return output_path


def main() -> None:
    args = parse_args()
    output_path = make_plot(
        Path(args.stability),
        Path(args.feature_table) if args.feature_table else None,
        args.dataset_name,
        Path(args.output_dir),
        args.group_a,
        args.group_b,
        args.top_n,
    )
    print(f"[DONE] Saved {output_path}")


if __name__ == "__main__":
    main()
