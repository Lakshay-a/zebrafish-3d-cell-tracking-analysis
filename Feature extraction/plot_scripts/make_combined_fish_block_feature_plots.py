#!/usr/bin/env python3
"""Plot track-level feature distributions for every fish across all conditions.

The violin shapes use Matplotlib's documented kernel-density visualization:
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violinplot.html

Displayed quantile limits use NumPy percentile calculations:
https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CONDITION_ORDER = ("untreated", "mmp", "liraglutide")
CONDITION_LABELS = {
    "untreated": "Untreated",
    "mmp": "MMP",
    "liraglutide": "Liraglutide",
}
GROUP_COLORS = {
    ("untreated", "WT"): "#4C78A8",
    ("untreated", "MUT"): "#E45756",
    ("mmp", "WT"): "#F28E2B",
    ("mmp", "MUT"): "#B279A2",
    ("liraglutide", "WT"): "#59A14F",
    ("liraglutide", "MUT"): "#9D755D",
}
GENOTYPE_MARKERS = {"WT": "o", "MUT": "^"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="append", required=True, metavar="CONDITION=CSV")
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points-per-fish", type=int, default=160)
    parser.add_argument("--display-quantile", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mapping(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        condition, path = item.split("=", 1)
        result[condition.strip().lower()] = Path(path)
    return result


def genotype(value: object) -> str:
    text = str(value).upper()
    if "MUT" in text:
        return "MUT"
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", text) or "WILD TYPE" in text:
        return "WT"
    return str(value)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def display_name(feature: str) -> str:
    return feature.replace("_um2_per_min", " (µm²/min)").replace(
        "_um_per_min", " (µm/min)"
    ).replace("_um3", " (µm³)").replace("_um", " (µm)").replace("_", " ").capitalize()


def raw_feature(feature: str) -> str:
    return re.sub(r"^fish_(mean|median)__", "", feature)


def load_features(path: Path) -> list[tuple[str, str]]:
    table = pd.read_csv(path)
    column = "feature" if "feature" in table.columns else table.columns[0]
    pairs: list[tuple[str, str]] = []
    # A fish mean and fish median can refer to the same track-level variable.
    # Plot that underlying distribution once rather than presenting duplicates.
    for selected in table[column].dropna().astype(str):
        raw = raw_feature(selected)
        if raw not in [item[1] for item in pairs]:
            pairs.append((selected, raw))
    return pairs


def main() -> None:
    args = arguments()
    paths = mapping(args.table)
    features = load_features(args.feature_file)
    frames = []
    for condition in CONDITION_ORDER:
        table = pd.read_csv(paths[condition], low_memory=False)
        required = {"fish_id", "block_name", "genotype"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{paths[condition]} lacks columns: {sorted(missing)}")
        table = table.copy()
        table["condition"] = condition
        table["genotype"] = table["genotype"].map(genotype)
        table["fish_key"] = (
            table["condition"].astype(str)
            + "::"
            + table["block_name"].astype(str)
            + "::"
            + table["fish_id"].astype(str)
        )
        frames.append(table)
    data = pd.concat(frames, ignore_index=True, sort=False)
    available = [(selected, raw) for selected, raw in features if raw in data.columns]
    if not available:
        raise ValueError("None of the selected model features exist in the track tables.")

    order = (
        data[["condition", "genotype", "block_name", "fish_id", "fish_key"]]
        .drop_duplicates()
        .assign(
            condition_rank=lambda x: x["condition"].map(
                {name: index for index, name in enumerate(CONDITION_ORDER)}
            ),
            genotype_rank=lambda x: x["genotype"].map({"WT": 0, "MUT": 1}).fillna(2),
        )
        .sort_values(
            ["condition_rank", "genotype_rank", "block_name", "fish_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    output = args.output_dir
    points_dir = output / "points_plus_fish_summary"
    summary_dir = output / "fish_median_iqr_summary"
    violin_dir = output / "violin_summary"
    points_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    violin_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []

    for feature_index, (selected, feature) in enumerate(available, start=1):
        values = pd.to_numeric(data[feature], errors="coerce")
        feature_data = data.assign(_value=values).dropna(subset=["_value"])
        all_values = feature_data["_value"].to_numpy(float)
        # Quantile limits only change the displayed range. Every observation is
        # retained when calculating the fish median, IQR and exported summaries.
        lower_quantile = max(0.0, 1.0 - args.display_quantile)
        visible_lower = float(np.quantile(all_values, lower_quantile))
        visible_upper = float(np.quantile(all_values, args.display_quantile))
        if np.nanmin(all_values) >= 0:
            visible_lower = 0.0
        visible_span = visible_upper - visible_lower
        visible_pad = 0.06 * visible_span if visible_span > 0 else 1.0
        summaries = []
        for x_pos, fish in order.iterrows():
            fish_values = feature_data.loc[
                feature_data["fish_key"].eq(fish["fish_key"]), "_value"
            ].to_numpy(float)
            if not len(fish_values):
                continue
            q25, median, q75 = np.percentile(fish_values, [25, 50, 75])
            summaries.append((x_pos, fish, q25, median, q75, len(fish_values)))
            records.append(
                {
                    "selected_model_feature": selected,
                    "track_feature": feature,
                    "condition": fish["condition"],
                    "genotype": fish["genotype"],
                    "block_name": fish["block_name"],
                    "fish_id": fish["fish_id"],
                    "n_tracks": len(fish_values),
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                }
            )

        width = max(13, 0.34 * len(order) + 4)
        fig, ax = plt.subplots(figsize=(width, 6.4))
        for x_pos, fish, q25, median, q75, _ in summaries:
            fish_values = feature_data.loc[
                feature_data["fish_key"].eq(fish["fish_key"]), "_value"
            ].to_numpy(float)
            if len(fish_values) > args.max_points_per_fish:
                fish_values = rng.choice(
                    fish_values, size=args.max_points_per_fish, replace=False
                )
            color = GROUP_COLORS.get((fish["condition"], fish["genotype"]), "#777777")
            ax.scatter(
                x_pos + rng.normal(0, 0.055, len(fish_values)),
                fish_values,
                s=10,
                alpha=0.22,
                color=color,
                marker=GENOTYPE_MARKERS.get(fish["genotype"], "o"),
                edgecolors="none",
            )
            ax.vlines(x_pos, q25, q75, color="black", linewidth=3, alpha=0.55)
            ax.hlines(median, x_pos - 0.23, x_pos + 0.23, color="black", linewidth=2.4)
        ax.set_ylabel(display_name(feature))
        ax.set_ylim(visible_lower - visible_pad, visible_upper + visible_pad)
        ax.set_title(
            f"{args.dataset_name}: {display_name(feature)} by fish block "
            f"(display zoom: central {args.display_quantile - lower_quantile:.0%})"
        )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [
                f"{CONDITION_LABELS[row.condition]} | {row.block_name}\n"
                f"{row.fish_id} ({row.genotype})"
                for row in order.itertuples()
            ],
            rotation=72,
            ha="right",
            fontsize=6.5,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="w",
                    markerfacecolor=color,
                    label=f"{CONDITION_LABELS[condition]} {group}",
                    marker=GENOTYPE_MARKERS[group],
                    markersize=7,
                )
                for (condition, group), color in GROUP_COLORS.items()
            ]
            + [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2.5,
                    label="Fish median; vertical bar = IQR",
                )
            ],
            fontsize=7,
            ncol=3,
            loc="upper right",
            frameon=True,
        )
        fig.tight_layout()
        filename = f"feature__{feature_index:02d}__{safe_name(feature)}.png"
        fig.savefig(points_dir / filename, dpi=260, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(width, 6.4))
        for x_pos, fish, q25, median, q75, count in summaries:
            color = GROUP_COLORS.get((fish["condition"], fish["genotype"]), "#777777")
            ax.vlines(x_pos, q25, q75, color=color, linewidth=8, alpha=0.45)
            ax.scatter(
                x_pos,
                median,
                s=70,
                marker=GENOTYPE_MARKERS.get(fish["genotype"], "o"),
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
            ax.text(x_pos, q75, f"n={count}", rotation=90, fontsize=6, va="bottom")
        ax.set_ylabel(display_name(feature))
        ax.set_title(f"{args.dataset_name}: fish medians and IQR for {display_name(feature)}")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [
                f"{CONDITION_LABELS[row.condition]} | {row.block_name}\n"
                f"{row.fish_id} ({row.genotype})"
                for row in order.itertuples()
            ],
            rotation=72,
            ha="right",
            fontsize=6.5,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker=GENOTYPE_MARKERS[group],
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    label=f"{CONDITION_LABELS[condition]} {group}",
                    markersize=7,
                )
                for (condition, group), color in GROUP_COLORS.items()
            ]
            + [
                Line2D(
                    [0],
                    [0],
                    color="#555555",
                    linewidth=8,
                    alpha=0.55,
                    label="Vertical bar = fish IQR",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markerfacecolor="white",
                    label="Point = fish median",
                    markersize=7,
                ),
            ],
            fontsize=7,
            ncol=3,
            loc="upper right",
            frameon=True,
        )
        fig.tight_layout()
        fig.savefig(summary_dir / filename, dpi=260, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(width, 6.4))
        violin_values = []
        violin_positions = []
        violin_colors = []
        for x_pos, fish, _, median, _, _ in summaries:
            fish_values = feature_data.loc[
                feature_data["fish_key"].eq(fish["fish_key"]), "_value"
            ].to_numpy(float)
            visible = fish_values[
                (fish_values >= visible_lower) & (fish_values <= visible_upper)
            ]
            if len(visible) >= 2:
                violin_values.append(visible)
                violin_positions.append(x_pos)
                violin_colors.append(
                    GROUP_COLORS.get((fish["condition"], fish["genotype"]), "#777777")
                )
            ax.scatter(
                x_pos,
                median,
                s=66,
                marker=GENOTYPE_MARKERS.get(fish["genotype"], "o"),
                color=GROUP_COLORS.get(
                    (fish["condition"], fish["genotype"]), "#777777"
                ),
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
            )
            ax.hlines(median, x_pos - 0.22, x_pos + 0.22, color="black", linewidth=2)
        if violin_values:
            parts = ax.violinplot(
                violin_values,
                positions=violin_positions,
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, color in zip(parts["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.38)
                body.set_linewidth(0.7)
        ax.set_ylim(visible_lower - visible_pad, visible_upper + visible_pad)
        ax.set_ylabel(display_name(feature))
        ax.set_title(
            f"{args.dataset_name}: {display_name(feature)} distributions by fish block "
            f"(display zoom: central {args.display_quantile - lower_quantile:.0%})"
        )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [
                f"{CONDITION_LABELS[row.condition]} | {row.block_name}\n"
                f"{row.fish_id} ({row.genotype})"
                for row in order.itertuples()
            ],
            rotation=72,
            ha="right",
            fontsize=6.5,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker=GENOTYPE_MARKERS[group],
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    label=f"{CONDITION_LABELS[condition]} {group}",
                    markersize=7,
                )
                for (condition, group), color in GROUP_COLORS.items()
            ],
            fontsize=7,
            ncol=3,
            loc="upper right",
            frameon=True,
        )
        fig.tight_layout()
        fig.savefig(violin_dir / filename, dpi=260, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(records).to_csv(output / "fish_block_feature_summaries.csv", index=False)
    pd.DataFrame(
        [{"selected_model_feature": selected, "track_feature": raw} for selected, raw in available]
    ).to_csv(output / "plotted_feature_mapping.csv", index=False)


if __name__ == "__main__":
    main()
