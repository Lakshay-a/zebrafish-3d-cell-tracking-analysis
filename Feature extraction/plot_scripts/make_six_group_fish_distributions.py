#!/usr/bin/env python3
"""Compare individual-fish feature values across three experimental conditions."""

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
from matplotlib.lines import Line2D


GENOTYPE_COLUMNS = ("genotype", "group", "condition", "class", "label")
FISH_COLUMNS = ("fish_id", "block_name", "block", "source_block", "sample_id")
GROUPS = (
    ("untreated", "WT"),
    ("untreated", "MUT"),
    ("mmp", "WT"),
    ("mmp", "MUT"),
    ("liraglutide", "WT"),
    ("liraglutide", "MUT"),
)
CONDITION_COLORS = {
    "untreated": "#4c78a8",
    "mmp": "#f58518",
    "liraglutide": "#54a24b",
}
GENOTYPE_MARKERS = {"WT": "o", "MUT": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="append", required=True, metavar="CONDITION=CSV")
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overview-columns", type=int, default=3)
    parser.add_argument("--overview-features-per-page", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=CSV, received: {value}")
        label, raw_path = value.split("=", 1)
        mapping[label.strip()] = Path(raw_path.strip())
    missing = {condition for condition, _ in GROUPS} - set(mapping)
    if missing:
        raise ValueError(f"Missing condition tables: {sorted(missing)}")
    return mapping


def detect_column(table: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    lookup = {str(column).lower(): str(column) for column in table.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(f"Could not detect {role} column.")


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if "WILD TYPE" in upper or re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper):
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "feature"


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


def load_fish_tables(paths: dict[str, Path], features: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition, path in paths.items():
        table = pd.read_csv(path, low_memory=False)
        missing = [feature for feature in features if feature not in table.columns]
        if missing:
            raise ValueError(f"{path} is missing features: {missing}")
        fish_col = detect_column(table, FISH_COLUMNS, "fish")
        genotype_col = detect_column(table, GENOTYPE_COLUMNS, "genotype")
        frame = table[[fish_col, genotype_col, *features]].copy()
        frame = frame.rename(columns={fish_col: "fish_id", genotype_col: "genotype"})
        frame["fish_id"] = frame["fish_id"].astype(str)
        frame["genotype"] = frame["genotype"].map(normalise_genotype)
        frame["condition"] = condition
        frames.append(frame[frame["genotype"].isin(["WT", "MUT"])])
    return pd.concat(frames, ignore_index=True)


def draw_feature(
    axis: plt.Axes,
    table: pd.DataFrame,
    feature: str,
    rng: np.random.Generator,
    compact: bool = False,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for position, (condition, genotype) in enumerate(GROUPS):
        subset = table[
            table["condition"].eq(condition) & table["genotype"].eq(genotype)
        ]
        values = pd.to_numeric(subset[feature], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        values = values[valid].to_numpy(float)
        fish_ids = subset.loc[valid, "fish_id"].astype(str).to_numpy()
        if len(values) == 0:
            continue

        jitter = rng.normal(0, 0.055, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=42 if compact else 58,
            color=CONDITION_COLORS[condition],
            marker=GENOTYPE_MARKERS[genotype],
            edgecolor="black",
            linewidth=0.55,
            alpha=0.82,
            zorder=3,
        )
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        axis.vlines(position, q25, q75, color="#555555", linewidth=5, alpha=0.55)
        axis.hlines(median, position - 0.18, position + 0.18, color="black", linewidth=2.4)
        for fish_id, value in zip(fish_ids, values):
            records.append(
                {
                    "feature": feature,
                    "condition": condition,
                    "genotype": genotype,
                    "fish_id": fish_id,
                    "value": float(value),
                    "group_median": median,
                    "group_q25": float(q25),
                    "group_q75": float(q75),
                }
            )

    axis.set_xticks(range(len(GROUPS)))
    axis.set_xticklabels(
        [f"{condition.title()}\n{genotype}" for condition, genotype in GROUPS],
        rotation=0,
    )
    axis.set_title(feature_label(feature), fontsize=10 if compact else 14)
    axis.grid(axis="y", alpha=0.22)
    return records


def main() -> None:
    args = parse_args()
    feature_table = pd.read_csv(args.feature_file)
    if "feature" not in feature_table.columns:
        raise ValueError("Feature file must contain a feature column.")
    features = feature_table["feature"].astype(str).drop_duplicates().tolist()
    data = load_fish_tables(parse_mapping(args.table), features)
    output = Path(args.output_dir)
    per_feature = output / "per_feature_raw_units"
    overview = output / "overview_pages_raw_units"
    per_feature.mkdir(parents=True, exist_ok=True)
    overview.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    all_records: list[dict[str, object]] = []

    for index, feature in enumerate(features, start=1):
        fig, axis = plt.subplots(figsize=(9.2, 6.4))
        records = draw_feature(axis, data, feature, rng)
        all_records.extend(records)
        axis.set_ylabel(feature_label(feature))
        axis.set_title(f"{args.dataset_name}\n{feature_label(feature)}")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="black",
                label=condition.title(),
            )
            for condition, color in CONDITION_COLORS.items()
        ] + [
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                color="black",
                markerfacecolor="white",
                label=genotype,
            )
            for genotype, marker in GENOTYPE_MARKERS.items()
        ]
        axis.legend(handles=handles, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(
            per_feature / f"feature__{index:02d}__{safe_name(feature)}.png",
            dpi=280,
            bbox_inches="tight",
        )
        plt.close(fig)

    page_size = max(1, args.overview_features_per_page)
    for page_start in range(0, len(features), page_size):
        page_features = features[page_start : page_start + page_size]
        columns = max(1, args.overview_columns)
        rows = math.ceil(len(page_features) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.0 * rows))
        axes_array = np.atleast_1d(axes).ravel()
        for axis, feature in zip(axes_array, page_features):
            draw_feature(axis, data, feature, rng, compact=True)
            axis.tick_params(axis="x", labelsize=7)
        for axis in axes_array[len(page_features) :]:
            axis.axis("off")
        fig.suptitle(
            f"{args.dataset_name}: individual-fish distributions across six groups",
            fontsize=15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        page_number = page_start // page_size + 1
        fig.savefig(
            overview / f"all_features_page_{page_number:02d}.png",
            dpi=260,
            bbox_inches="tight",
        )
        plt.close(fig)

    pd.DataFrame(all_records).to_csv(
        output / "individual_fish_values_and_group_summaries.csv", index=False
    )


if __name__ == "__main__":
    main()
