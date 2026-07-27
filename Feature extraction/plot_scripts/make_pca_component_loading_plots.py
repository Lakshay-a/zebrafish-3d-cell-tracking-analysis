#!/usr/bin/env python3
"""Plot PCA component loadings for features retained by a completed model run."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


GENOTYPE_COLUMNS = ["genotype", "group", "condition", "class", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PC1 and PC2 loadings for selected final-model features."
    )
    parser.add_argument("--input", required=True, help="Fish-level feature table.")
    parser.add_argument("--feature-file", required=True, help="CSV with a feature column.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n-features", type=int, default=12)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


def detect_genotype_column(df: pd.DataFrame) -> str:
    lookup = {str(column).lower(): str(column) for column in df.columns}
    for candidate in GENOTYPE_COLUMNS:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError("Could not detect genotype column.")


def feature_label(feature: str) -> str:
    text = feature.replace("fish_mean__", "Mean: ").replace("fish_median__", "Median: ")
    replacements = {
        "_um2_per_min": " (um2/min)",
        "_um_per_min": " (um/min)",
        "_um3": " (um3)",
        "_um2": " (um2)",
        "_um": " (um)",
        "_3d": " 3D",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def load_features(df: pd.DataFrame, feature_file: Path, top_n: int) -> list[str]:
    features = pd.read_csv(feature_file)
    if "feature" not in features.columns:
        raise ValueError(f"{feature_file} must contain a feature column.")
    if "selected_order" in features.columns:
        features = features.sort_values("selected_order", ascending=True)
    else:
        sort_cols = [
            col
            for col in [
                "nonzero_selection_frequency",
                "model_feature_frequency",
                "mean_absolute_coefficient_when_selected",
                "mean_absolute_coefficient_when_used",
            ]
            if col in features.columns
        ]
        if sort_cols:
            features = features.sort_values(sort_cols, ascending=False)

    usable: list[str] = []
    for feature in features["feature"].astype(str).tolist():
        if feature not in df.columns or feature in usable:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) >= 2:
            usable.append(feature)
        if len(usable) >= top_n:
            break
    if len(usable) < 2:
        raise ValueError("Need at least two usable features for PCA.")
    return usable


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    feature_file = Path(args.feature_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    genotype_col = detect_genotype_column(df)
    df = df.copy()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    df = df[df[genotype_col].isin(["WT", "MUT"])].reset_index(drop=True)
    features = load_features(df, feature_file, args.top_n_features)

    matrix = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    scaled = StandardScaler().fit_transform(imputed)
    # PCA implementation and component conventions:
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=min(2, scaled.shape[1], scaled.shape[0]))
    pca.fit(scaled)
    if pca.components_.shape[0] < 2:
        raise ValueError("PCA did not produce PC1 and PC2.")

    loadings = pd.DataFrame(
        {
            "feature": features,
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
        }
    )
    loadings["abs_PC1_loading"] = loadings["PC1_loading"].abs()
    loadings["abs_PC2_loading"] = loadings["PC2_loading"].abs()
    loadings["combined_abs_loading"] = loadings["abs_PC1_loading"] + loadings["abs_PC2_loading"]
    loadings = loadings.sort_values("combined_abs_loading", ascending=False)

    basename = safe_name(args.dataset_name)
    loadings.to_csv(output_dir / f"pca_component_loadings__{basename}.csv", index=False)
    pd.DataFrame(
        {
            "principal_component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_[:2],
        }
    ).to_csv(output_dir / f"pca_explained_variance__{basename}.csv", index=False)

    plot_table = loadings.sort_values("combined_abs_loading", ascending=True)
    labels = [feature_label(feature) for feature in plot_table["feature"]]
    y = np.arange(len(plot_table))
    height = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14.0, max(5.8, 0.45 * len(plot_table) + 2.0)), sharey=True)
    for ax, pc, color in [
        (axes[0], "PC1_loading", "#3b73b9"),
        (axes[1], "PC2_loading", "#d95f02"),
    ]:
        values = plot_table[pc].to_numpy(float)
        bar_colors = [color if value >= 0 else "#8a8a8a" for value in values]
        ax.barh(y, values, height=height, color=bar_colors, edgecolor="black", linewidth=0.35)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.grid(axis="x", alpha=0.20)
        ax.set_xlabel(f"{pc.replace('_loading', '')} loading")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    axes[1].set_title(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    fig.suptitle(f"{args.dataset_name}: features contributing to PC1 and PC2", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_dir / f"pca_component_loadings__{basename}.png", dpi=280)
    plt.close(fig)

    scatter = loadings.copy()
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.65)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.65)
    sizes = 80 + 360 * scatter["combined_abs_loading"] / scatter["combined_abs_loading"].max()
    ax.scatter(scatter["PC1_loading"], scatter["PC2_loading"], s=sizes, color="#4c78a8", edgecolor="black", alpha=0.82)
    for row in scatter.itertuples(index=False):
        ax.text(row.PC1_loading * 1.04, row.PC2_loading * 1.04, feature_label(row.feature), fontsize=8)
    ax.set_xlabel(f"PC1 loading ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 loading ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(f"{args.dataset_name}: PC loading map")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_dir / f"pca_loading_map__{basename}.png", dpi=280)
    plt.close(fig)

    print(f"[DONE] Saved PCA loading plots for {args.dataset_name} to {output_dir}")


if __name__ == "__main__":
    main()
