#!/usr/bin/env python3
"""Fit PCA on untreated model features and project treated fish without refitting."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


GENOTYPE_COLUMNS = ("genotype", "group", "condition", "class", "label")
FISH_COLUMNS = ("fish_id", "block_name", "block", "source_block", "sample_id")
CONDITION_COLORS = {
    "untreated": "#4c78a8",
    "mmp": "#f58518",
    "liraglutide": "#54a24b",
}
GENOTYPE_MARKERS = {"WT": "o", "MUT": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit preprocessing and PCA on untreated fish, then project treated fish."
    )
    parser.add_argument("--untreated", required=True)
    parser.add_argument(
        "--treated", action="append", default=[], metavar="CONDITION=CSV"
    )
    parser.add_argument("--coefficient-file", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-set-label",
        default="All features with non-zero coefficients in the frozen classifier",
    )
    parser.add_argument("--output-prefix", default="frozen_untreated_pca")
    return parser.parse_args()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=CSV, received: {value}")
        condition, raw_path = value.split("=", 1)
        mapping[condition.strip()] = Path(raw_path.strip())
    return mapping


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


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


def load_table(path: Path, condition: str, features: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    missing = [feature for feature in features if feature not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing frozen-model features: {missing}")
    genotype_col = detect_column(table, GENOTYPE_COLUMNS, "genotype")
    fish_col = detect_column(table, FISH_COLUMNS, "fish")
    result = table[[fish_col, genotype_col, *features]].copy()
    result = result.rename(columns={fish_col: "fish_id", genotype_col: "genotype"})
    result["fish_id"] = result["fish_id"].astype(str)
    result["genotype"] = result["genotype"].map(normalise_genotype)
    result["condition"] = condition
    return result[result["genotype"].isin(["WT", "MUT"])].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    coefficient_table = pd.read_csv(args.coefficient_file)
    if "feature" not in coefficient_table.columns:
        raise ValueError("Coefficient file must contain a feature column.")
    features = coefficient_table["feature"].astype(str).drop_duplicates().tolist()
    if len(features) < 2:
        raise ValueError("At least two frozen-model features are required for PCA.")

    untreated = load_table(Path(args.untreated), "untreated", features)
    treated = {
        condition: load_table(path, condition, features)
        for condition, path in parse_mapping(args.treated).items()
    }

    untreated_matrix = untreated[features].apply(pd.to_numeric, errors="coerce")
    # Missing values are handled exactly once using untreated medians:
    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    imputer = SimpleImputer(strategy="median")
    untreated_imputed = imputer.fit_transform(untreated_matrix)

    # The untreated mean and variance define the common standardized feature space:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    untreated_scaled = scaler.fit_transform(untreated_imputed)

    # PCA is fitted only to untreated fish; treated tables are transformed with
    # the same axes, following the fit/transform API documented here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=min(2, untreated_scaled.shape[0], untreated_scaled.shape[1]))
    untreated_scores = pca.fit_transform(untreated_scaled)
    if untreated_scores.shape[1] < 2:
        raise ValueError("Untreated data did not support both PC1 and PC2.")

    score_frames: list[pd.DataFrame] = []

    def add_scores(table: pd.DataFrame, scores: np.ndarray) -> None:
        score_frames.append(
            pd.DataFrame(
                {
                    "fish_id": table["fish_id"],
                    "genotype": table["genotype"],
                    "condition": table["condition"],
                    "PC1": scores[:, 0],
                    "PC2": scores[:, 1],
                }
            )
        )

    add_scores(untreated, untreated_scores)
    for condition, table in treated.items():
        matrix = table[features].apply(pd.to_numeric, errors="coerce")
        scores = pca.transform(scaler.transform(imputer.transform(matrix)))
        add_scores(table, scores)
    all_scores = pd.concat(score_frames, ignore_index=True)

    loadings = pd.DataFrame(
        {
            "feature": features,
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
        }
    )
    for component in ("PC1", "PC2"):
        squared = loadings[f"{component}_loading"].pow(2)
        loadings[f"{component}_contribution_percent"] = squared / squared.sum() * 100

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    basename = safe_name(args.dataset_name)
    prefix = safe_name(args.output_prefix)
    all_scores.to_csv(output / f"{prefix}_scores__{basename}.csv", index=False)
    loadings.to_csv(output / f"{prefix}_loadings__{basename}.csv", index=False)
    pd.DataFrame(
        {
            "principal_component": ["PC1", "PC2"],
            "untreated_explained_variance_ratio": pca.explained_variance_ratio_[:2],
        }
    ).to_csv(output / f"{prefix}_explained_variance__{basename}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    for condition in ["untreated", *treated]:
        for genotype in ("WT", "MUT"):
            subset = all_scores[
                all_scores["condition"].eq(condition)
                & all_scores["genotype"].eq(genotype)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                color=CONDITION_COLORS.get(condition, "#777777"),
                marker=GENOTYPE_MARKERS[genotype],
                s=75,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.82,
            )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(f"Untreated PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"Untreated PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"{args.dataset_name}: treated fish projected into untreated PCA space")
    condition_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=CONDITION_COLORS.get(condition, "#777777"),
            markeredgecolor="black",
            label=condition,
        )
        for condition in ["untreated", *treated]
    ]
    genotype_handles = [
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
    first = ax.legend(handles=condition_handles, title="Condition", loc="best")
    ax.add_artist(first)
    ax.legend(handles=genotype_handles, title="Genotype", loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        output / f"{prefix}_projection__{basename}.png",
        dpi=280,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(14.5, max(5.8, 0.45 * len(features) + 1.8)), sharey=True
    )
    y = np.arange(len(loadings))
    for axis, component, color in (
        (axes[0], "PC1", "#4c78a8"),
        (axes[1], "PC2", "#f58518"),
    ):
        values = loadings[f"{component}_contribution_percent"]
        axis.barh(y, values, color=color, edgecolor="black", linewidth=0.4)
        axis.set_xlabel(f"Contribution to untreated {component} (%)")
        axis.set_title(
            f"{component}: {pca.explained_variance_ratio_[int(component[-1]) - 1] * 100:.1f}%"
        )
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([feature_label(feature) for feature in features])
    fig.suptitle(
        f"{args.dataset_name}: fixed untreated PCA composition\n"
        f"{args.feature_set_label}"
    )
    fig.tight_layout()
    fig.savefig(
        output / f"{prefix}_composition__{basename}.png",
        dpi=280,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
