#!/usr/bin/env python3
"""Plot changes in cell-shape measurements using saved track-level summaries."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse


FISH_CANDIDATES = ["fish_id", "block_name", "block", "source_block", "sample_id"]
GENOTYPE_CANDIDATES = ["genotype", "group", "condition", "class", "label"]

GENOTYPE_COLORS = {
    "WT": "#1f77b4",
    "MUT": "#d62728",
}

CHANGE_FEATURES = [
    ("sphericity_change_start_to_end", "Delta sphericity"),
    ("elongation_change_start_to_end", "Delta elongation"),
    ("flatness_change_start_to_end", "Delta flatness"),
    ("aspect_ratio_3d_change_start_to_end", "Delta aspect ratio"),
    ("volume_um3_change_start_to_end", "Delta volume (um3)"),
    ("surface_area_to_volume_change_start_to_end", "Delta surface area:volume"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual plots showing how cell shape changes over tracks."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Repeat for each QC-filtered cell/track feature table.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-order", nargs="*", default=["WT", "MUT"])
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--max-arrows", type=int, default=450)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def parse_datasets(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Use LABEL=CSV format: {value}")
        label, path = value.split("=", 1)
        result[label.strip()] = Path(path.strip())
    return result


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "dataset"


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


def ordered_groups(series: pd.Series, preferred: list[str]) -> list[str]:
    present = [str(value) for value in series.dropna().unique()]
    groups = [group for group in preferred if group in present]
    groups.extend(group for group in present if group not in groups)
    return groups


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sample_table(table: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if max_points <= 0 or len(table) <= max_points:
        return table
    return table.sample(n=max_points, random_state=seed)


def robust_z(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    center = np.nanmedian(array)
    spread = np.nanpercentile(array, 75) - np.nanpercentile(array, 25)
    if not np.isfinite(spread) or spread <= 0:
        spread = np.nanstd(array)
    if not np.isfinite(spread) or spread <= 0:
        spread = 1.0
    return (array - center) / spread


def add_shape_icons(ax: plt.Axes) -> None:
    circle = Circle(
        (0.08, 0.11),
        0.035,
        transform=ax.transAxes,
        fill=False,
        color="black",
        linewidth=1.2,
        alpha=0.75,
    )
    ellipse = Ellipse(
        (0.88, 0.86),
        0.11,
        0.04,
        angle=28,
        transform=ax.transAxes,
        fill=False,
        color="black",
        linewidth=1.2,
        alpha=0.75,
    )
    ax.add_patch(circle)
    ax.add_patch(ellipse)
    ax.text(0.13, 0.11, "rounder", transform=ax.transAxes, va="center", fontsize=9)
    ax.text(0.74, 0.86, "elongated", transform=ax.transAxes, va="center", fontsize=9)


def plot_shape_change_plane(
    label: str,
    table: pd.DataFrame,
    genotype_col: str,
    group_order: list[str],
    output_path: Path,
    max_points: int,
    seed: int,
) -> None:
    required = ["sphericity_change_start_to_end", "elongation_change_start_to_end"]
    if not set(required).issubset(table.columns):
        print(f"[WARN] Skipping change plane for {label}; missing {required}.")
        return

    data = table[[genotype_col] + required].copy()
    for column in required:
        data[column] = numeric(data[column])
    data = data.dropna()
    data = sample_table(data, max_points, seed)

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    for group in ordered_groups(data[genotype_col], group_order):
        sub = data[data[genotype_col] == group]
        ax.scatter(
            sub["sphericity_change_start_to_end"],
            sub["elongation_change_start_to_end"],
            s=13,
            alpha=0.34,
            color=GENOTYPE_COLORS.get(group, "#666666"),
            label=f"{group} tracks",
            linewidths=0,
        )

    ax.axhline(0, color="black", linewidth=1.1)
    ax.axvline(0, color="black", linewidth=1.1)
    ax.text(0.98, 0.98, "more elongated", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold")
    ax.text(0.02, 0.02, "less spherical", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.98, 0.02, "rounder", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Change in sphericity from track start to end")
    ax.set_ylabel("Change in elongation from track start to end")
    ax.set_title(f"{label}: track-level shape-change direction")
    ax.legend(frameon=True)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_shape_state_arrows(
    label: str,
    table: pd.DataFrame,
    genotype_col: str,
    group_order: list[str],
    output_path: Path,
    max_arrows: int,
    seed: int,
) -> None:
    required = [
        "mean_sphericity",
        "mean_elongation",
        "sphericity_change_start_to_end",
        "elongation_change_start_to_end",
    ]
    if not set(required).issubset(table.columns):
        print(f"[WARN] Skipping shape arrows for {label}; missing {required}.")
        return

    data = table[[genotype_col] + required].copy()
    for column in required:
        data[column] = numeric(data[column])
    data = data.dropna()
    data = sample_table(data, max_arrows, seed)

    fig, ax = plt.subplots(figsize=(7.8, 6.9))
    for group in ordered_groups(data[genotype_col], group_order):
        sub = data[data[genotype_col] == group]
        color = GENOTYPE_COLORS.get(group, "#666666")
        start_x = sub["mean_sphericity"] - 0.5 * sub["sphericity_change_start_to_end"]
        start_y = sub["mean_elongation"] - 0.5 * sub["elongation_change_start_to_end"]
        dx = sub["sphericity_change_start_to_end"]
        dy = sub["elongation_change_start_to_end"]
        ax.quiver(
            start_x,
            start_y,
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=0.28,
            width=0.003,
            label=f"{group} estimated start-to-end",
        )

    ax.text(0.98, 0.05, "sphere-like", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.05, 0.95, "elongated", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold")
    ax.set_xlabel("Sphericity")
    ax.set_ylabel("Elongation")
    ax.set_title(
        f"{label}: estimated shape-state trajectories\n"
        "arrows use mean shape +/- half of start-to-end change"
    )
    ax.legend(frameon=True, fontsize=8)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_shape_morphospace_density(
    label: str,
    table: pd.DataFrame,
    genotype_col: str,
    group_order: list[str],
    output_path: Path,
    max_points: int,
    seed: int,
) -> None:
    required = [
        "mean_sphericity",
        "mean_elongation",
        "sphericity_change_start_to_end",
        "elongation_change_start_to_end",
    ]
    if not set(required).issubset(table.columns):
        print(f"[WARN] Skipping morphospace density for {label}; missing {required}.")
        return

    data = table[[genotype_col] + required].copy()
    for column in required:
        data[column] = numeric(data[column])
    data = data.dropna()
    if data.empty:
        return
    sampled = sample_table(data, max_points, seed)

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    for group in ordered_groups(data[genotype_col], group_order):
        sub = sampled[sampled[genotype_col] == group]
        if sub.empty:
            continue
        color = GENOTYPE_COLORS.get(group, "#666666")
        ax.scatter(
            sub["mean_sphericity"],
            sub["mean_elongation"],
            s=10,
            alpha=0.18,
            color=color,
            linewidths=0,
            label=f"{group} tracks",
        )

        full = data[data[genotype_col] == group]
        if len(full) >= 20:
            ax.hist2d(
                full["mean_sphericity"],
                full["mean_elongation"],
                bins=34,
                range=[
                    [data["mean_sphericity"].quantile(0.01), data["mean_sphericity"].quantile(0.99)],
                    [data["mean_elongation"].quantile(0.01), data["mean_elongation"].quantile(0.99)],
                ],
                cmap="Greys",
                alpha=0.08,
                cmin=1,
            )

        start_x = full["mean_sphericity"] - 0.5 * full["sphericity_change_start_to_end"]
        start_y = full["mean_elongation"] - 0.5 * full["elongation_change_start_to_end"]
        end_x = full["mean_sphericity"] + 0.5 * full["sphericity_change_start_to_end"]
        end_y = full["mean_elongation"] + 0.5 * full["elongation_change_start_to_end"]
        ax.annotate(
            "",
            xy=(np.nanmedian(end_x), np.nanmedian(end_y)),
            xytext=(np.nanmedian(start_x), np.nanmedian(start_y)),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 3,
                "color": color,
                "alpha": 0.9,
            },
        )
        ax.scatter(
            [np.nanmedian(start_x), np.nanmedian(end_x)],
            [np.nanmedian(start_y), np.nanmedian(end_y)],
            s=[52, 72],
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )

    add_shape_icons(ax)
    ax.set_xlabel("Mean sphericity")
    ax.set_ylabel("Mean elongation")
    ax.set_title(f"{label}: cell-shape morphospace")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_shape_state_transition(
    label: str,
    table: pd.DataFrame,
    genotype_col: str,
    group_order: list[str],
    output_path: Path,
) -> pd.DataFrame:
    required = [
        "mean_sphericity",
        "mean_elongation",
        "sphericity_change_start_to_end",
        "elongation_change_start_to_end",
    ]
    if not set(required).issubset(table.columns):
        return pd.DataFrame()

    data = table[[genotype_col] + required].copy()
    for column in required:
        data[column] = numeric(data[column])
    data = data.dropna()
    if data.empty:
        return pd.DataFrame()

    start_sphericity = data["mean_sphericity"] - 0.5 * data["sphericity_change_start_to_end"]
    end_sphericity = data["mean_sphericity"] + 0.5 * data["sphericity_change_start_to_end"]
    start_elongation = data["mean_elongation"] - 0.5 * data["elongation_change_start_to_end"]
    end_elongation = data["mean_elongation"] + 0.5 * data["elongation_change_start_to_end"]

    all_sphericity = pd.concat([start_sphericity, end_sphericity], ignore_index=True)
    all_elongation = pd.concat([start_elongation, end_elongation], ignore_index=True)
    start_score = robust_z(start_elongation) - robust_z(start_sphericity)
    end_score = robust_z(end_elongation) - robust_z(end_sphericity)
    combined_score = robust_z(all_elongation) - robust_z(all_sphericity)
    low, high = np.nanpercentile(combined_score, [33.3, 66.7])

    labels = ["round-like", "intermediate", "elongated-like"]

    def state(score: np.ndarray) -> pd.Categorical:
        values = np.where(score <= low, labels[0], np.where(score >= high, labels[2], labels[1]))
        return pd.Categorical(values, categories=labels, ordered=True)

    data = data.copy()
    data["start_shape_state"] = state(start_score)
    data["end_shape_state"] = state(end_score)

    groups = ordered_groups(data[genotype_col], group_order)
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(5.1 * len(groups), 4.8),
        squeeze=False,
    )
    records: list[dict[str, object]] = []
    for ax, group in zip(axes[0], groups):
        sub = data[data[genotype_col] == group]
        matrix = pd.crosstab(
            sub["start_shape_state"],
            sub["end_shape_state"],
            dropna=False,
        ).reindex(index=labels, columns=labels, fill_value=0)
        values = matrix.to_numpy(float)
        total = values.sum()
        proportions = values / total if total > 0 else values
        image = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=max(0.01, proportions.max()))
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{proportions[row, col]:.0%}\n(n={int(values[row, col])})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
                records.append(
                    {
                        "dataset": label,
                        "genotype": group,
                        "start_shape_state": labels[row],
                        "end_shape_state": labels[col],
                        "n_tracks": int(values[row, col]),
                        "proportion_within_genotype": float(proportions[row, col]),
                    }
                )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Estimated end shape state")
        ax.set_ylabel("Estimated start shape state")
        ax.set_title(f"{group} transitions")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Proportion of tracks")
    fig.suptitle(f"{label}: round-to-elongated shape-state transitions", y=1.02)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(records)


def plot_fish_change_summary(
    label: str,
    table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    group_order: list[str],
    output_path: Path,
) -> pd.DataFrame:
    needed = ["sphericity_change_start_to_end", "elongation_change_start_to_end"]
    if not set(needed).issubset(table.columns):
        return pd.DataFrame()

    data = table[[fish_col, genotype_col] + needed].copy()
    for column in needed:
        data[column] = numeric(data[column])
    data = data.dropna()
    summary = (
        data.groupby([genotype_col, fish_col], dropna=False)
        .agg(
            median_delta_sphericity=("sphericity_change_start_to_end", "median"),
            median_delta_elongation=("elongation_change_start_to_end", "median"),
            n_tracks=("sphericity_change_start_to_end", "size"),
        )
        .reset_index()
        .sort_values([genotype_col, fish_col])
    )
    if summary.empty:
        return summary

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(9.5, 0.48 * len(summary) + 3), 5.9))
    for group in ordered_groups(summary[genotype_col], group_order):
        mask = summary[genotype_col].eq(group).to_numpy()
        ax.scatter(
            x[mask] - 0.08,
            summary.loc[mask, "median_delta_sphericity"],
            s=58,
            marker="o",
            color=GENOTYPE_COLORS.get(group, "#666666"),
            label=f"{group} median delta sphericity",
        )
        ax.scatter(
            x[mask] + 0.08,
            summary.loc[mask, "median_delta_elongation"],
            s=58,
            marker="^",
            color=GENOTYPE_COLORS.get(group, "#666666"),
            label=f"{group} median delta elongation",
        )
    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row[fish_col]}\n{row[genotype_col]}" for _, row in summary.iterrows()],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Median start-to-end shape change per fish")
    ax.set_title(f"{label}: fish-level shape-change summaries")
    ax.legend(frameon=True, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)
    return summary


def plot_change_heatmap(
    label: str,
    table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    output_path: Path,
) -> pd.DataFrame:
    available = [(column, display) for column, display in CHANGE_FEATURES if column in table.columns]
    if len(available) < 2:
        return pd.DataFrame()

    columns = [column for column, _ in available]
    data = table[[fish_col, genotype_col] + columns].copy()
    for column in columns:
        data[column] = numeric(data[column])
    summary = (
        data.groupby([genotype_col, fish_col], dropna=False)[columns]
        .median()
        .reset_index()
        .sort_values([genotype_col, fish_col])
    )
    if summary.empty:
        return summary

    values = summary[columns].to_numpy(float)
    scaled = np.zeros_like(values)
    for idx in range(values.shape[1]):
        column = values[:, idx]
        center = np.nanmedian(column)
        spread = np.nanpercentile(column, 75) - np.nanpercentile(column, 25)
        spread = spread if np.isfinite(spread) and spread > 0 else np.nanstd(column)
        spread = spread if np.isfinite(spread) and spread > 0 else 1.0
        scaled[:, idx] = (column - center) / spread

    fig, ax = plt.subplots(figsize=(max(9.5, 0.42 * len(summary) + 3), 5.5))
    image = ax.imshow(scaled.T, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels(
        [f"{row[fish_col]}\n{row[genotype_col]}" for _, row in summary.iterrows()],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(available)))
    ax.set_yticklabels([display for _, display in available], fontsize=8)
    ax.set_title(f"{label}: median shape-change profile per fish")
    fig.colorbar(image, ax=ax, label="Robust-scaled median change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)
    return summary


def run_dataset(
    label: str,
    path: Path,
    output_root: Path,
    args: argparse.Namespace,
    dataset_index: int,
) -> None:
    df = pd.read_csv(path, low_memory=False)
    fish_col = detect_column(df, args.fish_col, FISH_CANDIDATES, "fish")
    genotype_col = detect_column(df, args.genotype_col, GENOTYPE_CANDIDATES, "genotype")
    df = df.copy()
    df[fish_col] = df[fish_col].astype(str).str.strip()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    df = df[df[genotype_col].isin(args.group_order)].copy()

    out = output_root / safe_name(label)
    out.mkdir(parents=True, exist_ok=True)

    plot_shape_change_plane(
        label,
        df,
        genotype_col,
        args.group_order,
        out / "01_shape_change_plane.png",
        args.max_points,
        args.random_seed + dataset_index,
    )
    plot_shape_state_arrows(
        label,
        df,
        genotype_col,
        args.group_order,
        out / "02_shape_state_arrows.png",
        args.max_arrows,
        args.random_seed + 100 + dataset_index,
    )
    plot_shape_morphospace_density(
        label,
        df,
        genotype_col,
        args.group_order,
        out / "03_shape_morphospace_density.png",
        args.max_points,
        args.random_seed + 200 + dataset_index,
    )
    transition_summary = plot_shape_state_transition(
        label,
        df,
        genotype_col,
        args.group_order,
        out / "04_shape_state_transition_matrix.png",
    )
    if not transition_summary.empty:
        transition_summary.to_csv(out / "shape_state_transition_summary.csv", index=False)
    fish_summary = plot_fish_change_summary(
        label,
        df,
        fish_col,
        genotype_col,
        args.group_order,
        out / "05_fish_shape_change_summary.png",
    )
    if not fish_summary.empty:
        fish_summary.to_csv(out / "fish_shape_change_summary.csv", index=False)

    heatmap_summary = plot_change_heatmap(
        label,
        df,
        fish_col,
        genotype_col,
        out / "06_fish_shape_change_heatmap.png",
    )
    if not heatmap_summary.empty:
        heatmap_summary.to_csv(out / "fish_shape_change_heatmap_values.csv", index=False)

    print(f"[DONE] {label}: shape-change plots saved to {out}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for index, (label, path) in enumerate(parse_datasets(args.dataset).items()):
        run_dataset(label, path, output_root, args, index)
    print(f"[DONE] All shape-change plots saved under: {output_root}")


if __name__ == "__main__":
    main()
