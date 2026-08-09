#!/usr/bin/env python3
"""Express squared PCA loadings as percentage contributions to each component."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


GENOTYPE_COLUMNS = ["genotype", "group", "condition", "class", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot feature percentage contributions to PC1 and PC2."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n-features", type=int, default=12)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


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


def detect_genotype_column(df: pd.DataFrame) -> str:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for candidate in GENOTYPE_COLUMNS:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError("Could not detect genotype column.")


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def load_features(df: pd.DataFrame, feature_file: Path, top_n: int) -> list[str]:
    table = pd.read_csv(feature_file)
    if "feature" not in table.columns:
        raise ValueError(f"{feature_file} must contain a feature column.")
    if "selected_order" in table.columns:
        table = table.sort_values("selected_order", ascending=True)
    else:
        sort_cols = [
            c
            for c in [
                "nonzero_selection_frequency",
                "model_feature_frequency",
                "mean_absolute_coefficient_when_selected",
                "mean_absolute_coefficient_when_used",
            ]
            if c in table.columns
        ]
        if sort_cols:
            table = table.sort_values(sort_cols, ascending=False)

    features: list[str] = []
    for feature in table["feature"].astype(str):
        if feature in features or feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) >= 2:
            features.append(feature)
        if len(features) >= top_n:
            break
    if len(features) < 2:
        raise ValueError("Need at least two usable features.")
    return features


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    genotype_col = detect_genotype_column(df)
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    df = df[df[genotype_col].isin(["WT", "MUT"])].reset_index(drop=True)
    features = load_features(df, Path(args.feature_file), args.top_n_features)

    matrix = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    scaled = StandardScaler().fit_transform(imputed)
    # PCA implementation and component conventions:
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=min(2, scaled.shape[0], scaled.shape[1]))
    pca.fit(scaled)
    if pca.components_.shape[0] < 2:
        raise ValueError("PCA did not produce PC1 and PC2.")

    rows = []
    for pc_idx, pc_name in enumerate(["PC1", "PC2"]):
        loadings = pca.components_[pc_idx]
        contribution = (loadings**2) / np.sum(loadings**2) * 100.0
        total_variance_contribution = contribution * pca.explained_variance_ratio_[pc_idx] / 100.0
        for feature, loading, pct, total_pct in zip(features, loadings, contribution, total_variance_contribution):
            rows.append(
                {
                    "principal_component": pc_name,
                    "feature": feature,
                    "loading": loading,
                    "feature_contribution_to_pc_percent": pct,
                    "pc_explained_variance_percent": pca.explained_variance_ratio_[pc_idx] * 100.0,
                    "feature_contribution_to_total_variance_percent": total_pct * 100.0,
                }
            )
    contrib = pd.DataFrame(rows)
    basename = safe_name(args.dataset_name)
    contrib.to_csv(out / f"pca_feature_contributions__{basename}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, max(5.8, 0.43 * len(features) + 2.0)), sharey=False)
    for ax, pc_name, color in [(axes[0], "PC1", "#3b73b9"), (axes[1], "PC2", "#d95f02")]:
        sub = contrib[contrib["principal_component"].eq(pc_name)].sort_values(
            "feature_contribution_to_pc_percent", ascending=True
        )
        y = np.arange(len(sub))
        ax.barh(y, sub["feature_contribution_to_pc_percent"], color=color, edgecolor="black", linewidth=0.35)
        ax.set_yticks(y)
        ax.set_yticklabels([feature_label(f) for f in sub["feature"]], fontsize=8)
        ax.set_xlabel(f"Contribution to {pc_name} (%)")
        explained = sub["pc_explained_variance_percent"].iloc[0]
        ax.set_title(f"{pc_name} explains {explained:.1f}% variance")
        ax.grid(axis="x", alpha=0.20)
        for yy, value in zip(y, sub["feature_contribution_to_pc_percent"]):
            ax.text(value + 0.6, yy, f"{value:.1f}%", va="center", fontsize=7)
        ax.set_xlim(0, max(35, sub["feature_contribution_to_pc_percent"].max() * 1.22))

    fig.suptitle(f"{args.dataset_name}: feature share of PC1 and PC2", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out / f"pca_feature_contributions__{basename}.png", dpi=280)
    plt.close(fig)

    print(f"[DONE] Saved PCA contribution plot for {args.dataset_name} to {out}")


if __name__ == "__main__":
    main()
