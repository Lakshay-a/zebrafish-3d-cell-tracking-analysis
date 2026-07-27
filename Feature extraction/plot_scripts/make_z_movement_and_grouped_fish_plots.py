#!/usr/bin/env python3
"""Plot axial movement and fish-level genotype summaries from final tables."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FISH_CANDIDATES = ["fish_id", "block_name", "block", "source_block", "sample_id"]
GENOTYPE_CANDIDATES = ["genotype", "group", "condition", "class", "label"]

GENOTYPE_COLORS = {
    "WT": "#1f77b4",
    "MUT": "#d62728",
}

Z_FEATURES = [
    ("z_range_um", "Z range (um)"),
    ("z_displacement_um", "Net Z displacement (um)"),
    ("z_path_length_um", "Total Z path length (um)"),
    ("mean_velocity_z_um_per_min", "Mean Z velocity (um/min)"),
]

DEFAULT_FISH_FEATURE_GROUPS = [
    [
        "fish_mean__net_displacement_3d_um",
        "fish_mean__mean_squared_displacement_3d_um2_per_min",
        "fish_mean__mean_speed_um_per_min",
        "fish_mean__median_speed_um_per_min",
        "fish_mean__tortuosity",
    ],
    [
        "fish_median__net_displacement_3d_um",
        "fish_median__mean_squared_displacement_3d_um2_per_min",
        "fish_median__mean_speed_um_per_min",
        "fish_median__median_speed_um_per_min",
        "fish_median__tortuosity",
    ],
    [
        "fish_mean__directionality_ratio",
        "fish_mean__mean_sphericity",
        "fish_mean__mean_elongation",
        "fish_mean__mean_volume_um3",
        "fish_median__mean_sphericity",
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make z-axis movement plots and grouped fish-level feature panels."
    )
    parser.add_argument("--cell-table", action="append", default=[], metavar="LABEL=CSV")
    parser.add_argument("--fish-table", action="append", default=[], metavar="LABEL=CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-order", nargs="*", default=["WT", "MUT"])
    parser.add_argument("--max-cell-points-per-fish", type=int, default=120)
    parser.add_argument("--histogram-bins", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def parse_mapping(values: list[str], role: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{role} must use LABEL=CSV format: {value}")
        label, raw_path = value.split("=", 1)
        result[label.strip()] = Path(raw_path.strip())
    return result


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} column '{explicit}' not found.")
        return explicit
    lower_map = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not detect {role} column.")


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def feature_label(feature: str) -> str:
    text = feature.replace("fish_mean__", "Mean: ").replace("fish_median__", "Median: ")
    replacements = {
        "_um2_per_min": " (um2/min)",
        "_um_per_min": " (um/min)",
        "_um_per_frame": " (um/frame)",
        "_um3": " (um3)",
        "_um2": " (um2)",
        "_um": " (um)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


# Cliff's delta reports stochastic dominance without assuming normality:
# https://revistas.javeriana.edu.co/index.php/revPsycho/article/view/643
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    differences = a[:, None] - b[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def prepare_identity(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    group_order: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df[fish_col] = df[fish_col].astype(str).str.strip()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    return df[df[genotype_col].isin(group_order)].copy()


def fish_order_table(df: pd.DataFrame, fish_col: str, genotype_col: str) -> pd.DataFrame:
    return (
        df[[fish_col, genotype_col]]
        .drop_duplicates()
        .sort_values([genotype_col, fish_col])
        .reset_index(drop=True)
    )


def plot_z_movement_panel(
    label: str,
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    out: Path,
    max_points_per_fish: int,
    seed: int,
) -> list[dict[str, object]]:
    available = [(feature, display) for feature, display in Z_FEATURES if feature in df.columns]
    if not available:
        print(f"[WARN] No z movement features found for {label}.")
        return []

    order = fish_order_table(df, fish_col, genotype_col)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        len(available),
        1,
        figsize=(max(10.5, 0.46 * len(order) + 3.0), 3.2 * len(available)),
        sharex=True,
    )
    if len(available) == 1:
        axes = [axes]

    summary: list[dict[str, object]] = []
    for ax, (feature, display) in zip(axes, available):
        values_all = numeric(df[feature])
        plot_table = df[[fish_col, genotype_col]].copy()
        plot_table["_value"] = values_all
        plot_table = plot_table.dropna(subset=["_value"])

        for x_pos, row in order.iterrows():
            fish = str(row[fish_col])
            genotype = str(row[genotype_col])
            values = plot_table.loc[plot_table[fish_col].astype(str).eq(fish), "_value"].to_numpy(float)
            if len(values) == 0:
                continue
            plotted = values
            if max_points_per_fish > 0 and len(values) > max_points_per_fish:
                plotted = rng.choice(values, size=max_points_per_fish, replace=False)
            jitter = rng.normal(0, 0.055, size=len(plotted))
            ax.scatter(
                x_pos + jitter,
                plotted,
                s=13,
                alpha=0.48,
                color=GENOTYPE_COLORS.get(genotype, "#666666"),
                linewidths=0,
            )
            ax.hlines(
                np.nanmedian(values),
                x_pos - 0.22,
                x_pos + 0.22,
                color="black",
                linewidth=2.3,
            )
            summary.append(
                {
                    "dataset": label,
                    "fish_id": fish,
                    "genotype": genotype,
                    "feature": feature,
                    "n_cell_tracks": int(len(values)),
                    "median": float(np.nanmedian(values)),
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                }
            )
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.55)
        ax.set_ylabel(display)
        ax.grid(axis="y", alpha=0.22)

    labels = [f"{row[fish_col]}\n{row[genotype_col]}" for _, row in order.iterrows()]
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    fig.suptitle(f"{label}: z-axis movement of tracked cells", y=0.995)
    fig.tight_layout()
    dataset_out = ensure_dir(out / "z_movement_by_dataset" / safe_name(label))
    fig.savefig(dataset_out / "z_movement_by_fish.png", dpi=260)
    plt.close(fig)
    return summary


def plot_z_range_histogram(
    label: str,
    df: pd.DataFrame,
    genotype_col: str,
    out: Path,
    group_order: list[str],
    bins: int,
) -> list[dict[str, object]]:
    feature = "z_range_um"
    if feature not in df.columns:
        print(f"[WARN] z_range_um not found for {label}; skipping z-range histogram.")
        return []

    table = df[[genotype_col, feature]].copy()
    table[feature] = numeric(table[feature])
    table = table.dropna(subset=[feature])
    if table.empty:
        return []

    arrays = [
        table.loc[table[genotype_col].eq(genotype), feature].to_numpy(float)
        for genotype in group_order
    ]
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays if len(array)])
    if len(finite) == 0:
        return []

    upper = float(np.nanpercentile(finite, 99.5))
    lower = max(0.0, float(np.nanmin(finite)))
    if not np.isfinite(upper) or upper <= lower:
        upper = float(np.nanmax(finite))
    bin_edges = np.linspace(lower, upper, max(5, bins) + 1)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    summary: list[dict[str, object]] = []
    for genotype, values in zip(group_order, arrays):
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        clipped = values[(values >= lower) & (values <= upper)]
        ax.hist(
            clipped,
            bins=bin_edges,
            density=True,
            alpha=0.48,
            color=GENOTYPE_COLORS.get(genotype, "#666666"),
            edgecolor="white",
            linewidth=0.7,
            label=f"{genotype} cells (n={len(values)})",
        )
        ax.axvline(
            np.nanmedian(values),
            color=GENOTYPE_COLORS.get(genotype, "#666666"),
            linewidth=2.2,
            linestyle="-",
        )
        summary.append(
            {
                "dataset": label,
                "genotype": genotype,
                "feature": feature,
                "n_cell_tracks": int(len(values)),
                "median": float(np.nanmedian(values)),
                "mean": float(np.nanmean(values)),
                "p75": float(np.nanpercentile(values, 75)),
                "p90": float(np.nanpercentile(values, 90)),
            }
        )

    ax.set_xlabel("Z range movement (um)")
    ax.set_ylabel("Normalized density of cell tracks")
    ax.set_title(f"{label}: normalized distribution of z-range movement")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    dataset_out = ensure_dir(out / "z_movement_by_dataset" / safe_name(label))
    fig.savefig(dataset_out / "z_range_histogram.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_grouped_fish_features(
    label: str,
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    out: Path,
    group_order: list[str],
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    groups = []
    for feature_group in DEFAULT_FISH_FEATURE_GROUPS:
        present = [feature for feature in feature_group if feature in df.columns]
        if present:
            groups.append(present)

    for group_index, features in enumerate(groups, start=1):
        fig, axes = plt.subplots(1, len(features), figsize=(3.25 * len(features), 5.3), sharey=False)
        if len(features) == 1:
            axes = [axes]

        for ax, feature in zip(axes, features):
            values = numeric(df[feature])
            plot_table = df[[fish_col, genotype_col]].copy()
            plot_table["_value"] = values
            plot_table = plot_table.dropna(subset=["_value"])

            arrays = []
            for x_pos, genotype in enumerate(group_order):
                sub = plot_table[plot_table[genotype_col].eq(genotype)]
                arr = sub["_value"].to_numpy(float)
                arrays.append(arr)
                if len(arr) == 0:
                    continue
                ax.boxplot(
                    [arr],
                    positions=[x_pos],
                    widths=0.46,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={"facecolor": "white", "edgecolor": GENOTYPE_COLORS.get(genotype, "black")},
                    medianprops={"color": "black", "linewidth": 2},
                )
                jitter = rng.normal(0, 0.055, size=len(arr))
                ax.scatter(
                    x_pos + jitter,
                    arr,
                    s=42,
                    alpha=0.85,
                    color=GENOTYPE_COLORS.get(genotype, "#666666"),
                    edgecolors="white",
                    linewidths=0.35,
                )

            delta = cliffs_delta(arrays[0], arrays[1]) if len(arrays) >= 2 else np.nan
            ax.set_xticks(np.arange(len(group_order)))
            ax.set_xticklabels(group_order)
            ax.set_title(f"{feature_label(feature)}\nCliff's delta={delta:.2f}" if np.isfinite(delta) else feature_label(feature), fontsize=9)
            ax.grid(axis="y", alpha=0.22)
            records.append(
                {
                    "dataset": label,
                    "feature": feature,
                    "cliffs_delta_first_group_vs_second": delta,
                }
            )

        fig.suptitle(f"{label}: grouped fish-level feature distributions", y=1.02)
        fig.tight_layout()
        dataset_out = ensure_dir(out / "fish_feature_groups_by_dataset" / safe_name(label))
        fig.savefig(dataset_out / f"grouped_fish_features_{group_index:02d}.png", dpi=260, bbox_inches="tight")
        plt.close(fig)
    return records


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    z_records: list[dict[str, object]] = []
    z_histogram_records: list[dict[str, object]] = []
    for index, (label, path) in enumerate(parse_mapping(args.cell_table, "cell-table").items()):
        df = pd.read_csv(path, low_memory=False)
        fish_col = detect_column(df, args.fish_col, FISH_CANDIDATES, "fish")
        genotype_col = detect_column(df, args.genotype_col, GENOTYPE_CANDIDATES, "genotype")
        df = prepare_identity(df, fish_col, genotype_col, args.group_order)
        z_records.extend(
            plot_z_movement_panel(
                label,
                df,
                fish_col,
                genotype_col,
                out,
                args.max_cell_points_per_fish,
                args.random_seed + index,
            )
        )
        z_histogram_records.extend(
            plot_z_range_histogram(
                label,
                df,
                genotype_col,
                out,
                args.group_order,
                args.histogram_bins,
            )
        )

    fish_records: list[dict[str, object]] = []
    for index, (label, path) in enumerate(parse_mapping(args.fish_table, "fish-table").items()):
        df = pd.read_csv(path, low_memory=False)
        fish_col = detect_column(df, args.fish_col, FISH_CANDIDATES, "fish")
        genotype_col = detect_column(df, args.genotype_col, GENOTYPE_CANDIDATES, "genotype")
        df = prepare_identity(df, fish_col, genotype_col, args.group_order)
        fish_records.extend(
            plot_grouped_fish_features(
                label,
                df,
                fish_col,
                genotype_col,
                out,
                args.group_order,
                args.random_seed + 100 + index,
            )
        )

    summary_out = ensure_dir(out / "summary_tables")
    if z_records:
        pd.DataFrame(z_records).to_csv(summary_out / "z_movement_by_fish_summary.csv", index=False)
    if z_histogram_records:
        pd.DataFrame(z_histogram_records).to_csv(summary_out / "z_range_histogram_summary.csv", index=False)
    if fish_records:
        pd.DataFrame(fish_records).to_csv(summary_out / "grouped_fish_feature_effects.csv", index=False)
    print(f"[DONE] Z movement and grouped fish plots saved to: {out}")


if __name__ == "__main__":
    main()
