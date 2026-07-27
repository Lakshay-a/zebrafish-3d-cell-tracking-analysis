#!/usr/bin/env python3
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


FISH_CANDIDATES = ["fish_id", "block_name", "block", "source_block", "sample_id"]
GENOTYPE_CANDIDATES = ["genotype", "group", "condition", "class", "label"]
TRACK_CANDIDATES = ["track_id", "global_track_id", "cell_track_id", "cell_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a clear PCA biplot from the best/stable features."
    )
    parser.add_argument("--input", required=True, help="Feature CSV.")
    parser.add_argument(
        "--feature-file",
        default=None,
        help=(
            "CSV containing a feature column. If it has selection-frequency "
            "columns, the top features are selected automatically."
        ),
    )
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--top-n-features", type=int, default=10)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--track-col", default=None)
    parser.add_argument("--group-order", nargs="*", default=["WT", "MUT"])
    parser.add_argument("--point-alpha", type=float, default=0.35)
    parser.add_argument("--point-size", type=float, default=14.0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} column '{explicit}' not found.")
        return explicit
    lower_map = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(f"Could not detect {role} column.")
    return None


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def feature_label(feature: str) -> str:
    text = feature.replace("fish_mean__", "Mean: ").replace("fish_median__", "Median: ")
    replacements = {
        "_um_per_frame": " (um/frame)",
        "_um3": " (um3)",
        "_um2": " (um2)",
        "_um": " (um)",
        "_px": " (px)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def ordered_groups(series: pd.Series, preferred: list[str]) -> list[str]:
    present = [str(value) for value in series.dropna().unique()]
    groups = [group for group in preferred if group in present]
    groups.extend(group for group in present if group not in groups)
    return groups


def load_feature_list(
    input_df: pd.DataFrame,
    feature_file: str | None,
    explicit_features: list[str],
    top_n: int,
) -> list[str]:
    if explicit_features:
        features = explicit_features
    elif feature_file:
        table = pd.read_csv(feature_file)
        if "feature" not in table.columns:
            raise ValueError(f"Feature file must contain a 'feature' column: {feature_file}")
        sort_columns = [
            column
            for column in [
                "nonzero_selection_frequency",
                "top_k_frequency",
                "mean_absolute_coefficient_when_selected",
                "selected_order",
            ]
            if column in table.columns
        ]
        if "selected_order" in sort_columns:
            table = table.sort_values("selected_order", ascending=True)
        elif sort_columns:
            table = table.sort_values(sort_columns, ascending=False)
        features = table["feature"].astype(str).tolist()[:top_n]
    else:
        raise ValueError("Provide --feature-file or at least one --feature.")

    missing = [feature for feature in features if feature not in input_df.columns]
    if missing:
        raise ValueError(f"Features missing from input table: {missing}")

    usable: list[str] = []
    for feature in features:
        values = pd.to_numeric(input_df[feature], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) >= 2:
            usable.append(feature)
    if len(usable) < 2:
        raise ValueError("Fewer than two usable features for PCA.")
    return usable


# PCA API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    fish_col = detect_column(df, args.fish_col, FISH_CANDIDATES, "fish", required=False)
    genotype_col = detect_column(df, args.genotype_col, GENOTYPE_CANDIDATES, "genotype")
    track_col = detect_column(df, args.track_col, TRACK_CANDIDATES, "track", required=False)
    features = load_feature_list(df, args.feature_file, args.feature, args.top_n_features)

    metadata_cols = [column for column in [fish_col, genotype_col, track_col] if column]
    table = df[metadata_cols + features].copy()
    table[genotype_col] = table[genotype_col].map(normalise_genotype)
    table = table[table[genotype_col].notna() & table[genotype_col].ne("")]
    table = table.reset_index(drop=True)

    matrix = table[features].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    scaled = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=min(scaled.shape))
    coordinates = pca.fit_transform(scaled)
    scores = pd.DataFrame(
        coordinates,
        columns=[f"PC{i + 1}" for i in range(coordinates.shape[1])],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=scores.columns,
    )

    pd.concat([table[metadata_cols].reset_index(drop=True), scores], axis=1).to_csv(
        out / "best_feature_pca_scores.csv",
        index=False,
    )
    loadings.reset_index(names="feature").to_csv(
        out / "best_feature_pca_loadings.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "principal_component": scores.columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(out / "best_feature_pca_explained_variance.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out / "best_features_used.csv", index=False)

    colors = {"WT": "#1f77b4", "MUT": "#d62728"}
    groups = ordered_groups(table[genotype_col], args.group_order)
    fig, ax = plt.subplots(figsize=(8.6, 7.4))

    for group in groups:
        mask = table[genotype_col].astype(str).eq(group).to_numpy()
        ax.scatter(
            scores.loc[mask, "PC1"],
            scores.loc[mask, "PC2"],
            s=args.point_size,
            alpha=args.point_alpha,
            color=colors.get(group, "#666666"),
            label=f"{group} (n={int(mask.sum())})",
            linewidths=0,
        )

    radius = float(np.nanpercentile(np.sqrt(scores.PC1**2 + scores.PC2**2), 92))
    loading_xy = loadings[["PC1", "PC2"]].copy()
    loading_xy["magnitude"] = np.sqrt(loading_xy.PC1**2 + loading_xy.PC2**2)
    scale = 0.76 * radius / float(loading_xy["magnitude"].max())

    for feature, row in loading_xy.sort_values("magnitude", ascending=False).iterrows():
        x = float(row.PC1 * scale)
        y = float(row.PC2 * scale)
        ax.arrow(
            0,
            0,
            x,
            y,
            color="#2f4b7c",
            width=0.004 * max(radius, 1),
            head_width=0.04 * max(radius, 1),
            head_length=0.06 * max(radius, 1),
            alpha=0.85,
            length_includes_head=True,
        )
        ax.text(
            x * 1.10,
            y * 1.10,
            feature_label(feature),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.55)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(f"{args.dataset_name}: best-feature PCA biplot")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(out / "best_feature_pca_biplot.png", dpi=280)
    plt.close(fig)
    print(f"[DONE] Saved best-feature PCA biplot to: {out}")


if __name__ == "__main__":
    main()
