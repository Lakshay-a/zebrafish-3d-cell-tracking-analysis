#!/usr/bin/env python3
"""Create paper-style PCA plots from fish-level or cell/track-level feature tables."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FISH_CANDIDATES = [
    "fish_id", "block_name", "block", "source_block", "sample_id",
    "dataset_id", "czi_name", "file",
]
GENOTYPE_CANDIDATES = ["genotype", "group", "condition", "class", "label"]
TRACK_CANDIDATES = [
    "track_id", "global_track_id", "cell_track_id", "cell_id", "object_track_id"
]
NON_FEATURE_EXACT = {
    "time", "frame", "object_label", "cell_label", "track_length",
    "first_time", "last_time", "start_time", "end_time", "duration",
    "true_binary", "predicted_binary", "probability_group_b",
}
CHI2_50_DF2 = 1.38629436112
CHI2_95_DF2 = 5.99146454711


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style PCA plots")
    p.add_argument("--dataset", action="append", required=True, metavar="LABEL=CSV")
    p.add_argument("--level", choices=["auto", "fish", "cell"], default="auto")
    p.add_argument("--view", choices=["both", "mean", "median"], default="both")
    p.add_argument("--feature", action="append", default=[])
    p.add_argument("--fish-col", default=None)
    p.add_argument("--genotype-col", default=None)
    p.add_argument("--track-col", default=None)
    p.add_argument("--group-order", nargs="*", default=["WT", "MUT"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-loading-arrows", type=int, default=10)
    p.add_argument("--cluster-min", type=int, default=2)
    p.add_argument("--cluster-max", type=int, default=6)
    p.add_argument("--no-clustering", action="store_true")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def parse_datasets(values: list[str]) -> dict[str, Path]:
    result = {}
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
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} column '{explicit}' not found")
        return explicit
    lower = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise ValueError(f"Could not detect {role} column")
    return None


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def infer_level(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    is_fish = any(
        str(c).startswith("fish_mean__") or str(c).startswith("fish_median__")
        for c in df.columns
    )
    return "fish" if is_fish else "cell"


def usable_numeric(df: pd.DataFrame, column: str, identity: set[str]) -> bool:
    if column in identity or column in NON_FEATURE_EXACT:
        return False
    if column.startswith(("n_", "count_", "num_")):
        return False
    values = pd.to_numeric(df[column], errors="coerce")
    return values.notna().sum() >= 3 and values.nunique(dropna=True) >= 2


def choose_features(
    df: pd.DataFrame,
    level: str,
    view: str,
    custom: list[str],
    identity: set[str],
) -> list[str]:
    if custom:
        missing = [f for f in custom if f not in df.columns]
        if missing:
            raise ValueError(f"Missing requested PCA features: {missing}")
        features = [f for f in custom if usable_numeric(df, f, identity)]
    elif level == "fish":
        features = []
        if view in {"both", "mean"}:
            features.extend(c for c in df.columns if str(c).startswith("fish_mean__"))
        if view in {"both", "median"}:
            features.extend(c for c in df.columns if str(c).startswith("fish_median__"))
        features = [str(f) for f in features if usable_numeric(df, str(f), identity)]
    else:
        features = [str(c) for c in df.columns if usable_numeric(df, str(c), identity)]
    if len(features) < 2:
        raise ValueError("Fewer than two usable PCA features were found")
    return features


def feature_label(feature: str) -> str:
    text = feature.replace("fish_mean__", "Mean: ").replace("fish_median__", "Median: ")
    replacements = {
        "_um_per_frame": " (µm/frame)",
        "_um3": " (µm³)",
        "_um2": " (µm²)",
        "_um": " (µm)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def ordered_groups(series: pd.Series, preferred: list[str]) -> list[str]:
    present = [str(v) for v in series.dropna().unique()]
    groups = [g for g in preferred if g in present]
    groups.extend(g for g in present if g not in groups)
    return groups


def add_ellipse(ax, x: np.ndarray, y: np.ndarray, probability: float, linestyle: str):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return
    cov = np.cov(np.column_stack([x, y]), rowvar=False)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    q = CHI2_50_DF2 if probability == 0.50 else CHI2_95_DF2
    width, height = 2 * np.sqrt(vals * q)
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(
        (float(np.mean(x)), float(np.mean(y))),
        float(width), float(height), angle=float(angle), fill=False,
        linestyle=linestyle, linewidth=1.3 if probability == 0.50 else 1.0,
        alpha=0.85,
    ))


def prepare_pca(df: pd.DataFrame, features: list[str]):
    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(numeric)
    scaled = StandardScaler().fit_transform(imputed)
    n_components = min(scaled.shape)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)
    score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    score_df = pd.DataFrame(scores, columns=score_cols)
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=score_cols)
    return pca, score_df, loadings


def plot_scree(label: str, pca: PCA, path: Path):
    individual = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(individual)
    x = np.arange(1, len(individual) + 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    ax.bar(x, individual, label="Individual")
    ax.plot(x, cumulative, marker="o", linewidth=1.7, label="Cumulative")
    ax.axhline(80, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"{label}: PCA explained variance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_by_group(
    label: str,
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    pca: PCA,
    fish_col: str,
    genotype_col: str,
    group_order: list[str],
    path: Path,
    show_labels: bool,
    pc_x: int = 1,
    pc_y: int = 2,
):
    xcol, ycol = f"PC{pc_x}", f"PC{pc_y}"
    if xcol not in scores or ycol not in scores:
        return
    table = pd.concat([metadata.reset_index(drop=True), scores], axis=1)
    groups = ordered_groups(table[genotype_col], group_order)
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, ax = plt.subplots(figsize=(8.2, 6.5))
    for i, group in enumerate(groups):
        sub = table[table[genotype_col] == group]
        ax.scatter(sub[xcol], sub[ycol], s=70, marker=markers[i % len(markers)],
                   alpha=0.9, label=f"{group} (n={len(sub)})")
        add_ellipse(ax, sub[xcol].to_numpy(float), sub[ycol].to_numpy(float), 0.50, "-")
        add_ellipse(ax, sub[xcol].to_numpy(float), sub[ycol].to_numpy(float), 0.95, "--")
        if show_labels:
            for _, row in sub.iterrows():
                ax.annotate(str(row[fish_col]), (row[xcol], row[ycol]),
                            xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(f"{xcol} ({pca.explained_variance_ratio_[pc_x-1]*100:.1f}% variance)")
    ax.set_ylabel(f"{ycol} ({pca.explained_variance_ratio_[pc_y-1]*100:.1f}% variance)")
    ax.set_title(f"{label}: PCA by genotype\nSolid ellipse = 50%; dashed ellipse = 95%")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_panels(
    label: str,
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    pca: PCA,
    fish_col: str,
    genotype_col: str,
    group_order: list[str],
    path: Path,
    show_labels: bool,
):
    if "PC1" not in scores or "PC2" not in scores:
        return
    table = pd.concat([metadata.reset_index(drop=True), scores], axis=1)
    groups = ordered_groups(table[genotype_col], group_order)
    fig, axes = plt.subplots(1, len(groups), figsize=(6.0*len(groups), 5.2), squeeze=False,
                             sharex=True, sharey=True)
    xmin, xmax = scores.PC1.min(), scores.PC1.max()
    ymin, ymax = scores.PC2.min(), scores.PC2.max()
    xm, ym = max(0.5, 0.08*(xmax-xmin)), max(0.5, 0.08*(ymax-ymin))
    for ax, group in zip(axes.ravel(), groups):
        sub = table[table[genotype_col] == group]
        ax.scatter(sub.PC1, sub.PC2, s=70, alpha=0.9)
        add_ellipse(ax, sub.PC1.to_numpy(float), sub.PC2.to_numpy(float), 0.50, "-")
        add_ellipse(ax, sub.PC1.to_numpy(float), sub.PC2.to_numpy(float), 0.95, "--")
        if show_labels:
            for _, row in sub.iterrows():
                ax.annotate(str(row[fish_col]), (row.PC1, row.PC2), xytext=(4, 3),
                            textcoords="offset points", fontsize=7)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlim(xmin-xm, xmax+xm)
        ax.set_ylim(ymin-ym, ymax+ym)
        ax.set_title(f"{group} (n={len(sub)})")
        ax.grid(alpha=0.2)
    fig.supxlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    fig.supylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    fig.suptitle(f"{label}: genotype-specific PCA distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_biplot(
    label: str,
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    pca: PCA,
    genotype_col: str,
    group_order: list[str],
    path: Path,
    top_n: int,
):
    table = pd.concat([metadata.reset_index(drop=True), scores], axis=1)
    groups = ordered_groups(table[genotype_col], group_order)
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    for group in groups:
        sub = table[table[genotype_col] == group]
        ax.scatter(sub.PC1, sub.PC2, s=60, alpha=0.75, label=group)
    top = loadings[["PC1", "PC2"]].copy()
    top["magnitude"] = np.sqrt(top.PC1**2 + top.PC2**2)
    top = top.sort_values("magnitude", ascending=False).head(max(1, top_n))
    score_radius = float(np.nanpercentile(np.sqrt(scores.PC1**2 + scores.PC2**2), 92))
    scale = 0.78*score_radius/float(top.magnitude.max()) if top.magnitude.max() > 0 else 1.0
    for feature, row in top.iterrows():
        x, y = float(row.PC1*scale), float(row.PC2*scale)
        ax.arrow(0, 0, x, y, width=0.006*max(score_radius, 1),
                 head_width=0.05*max(score_radius, 1),
                 head_length=0.07*max(score_radius, 1),
                 length_includes_head=True, alpha=0.8)
        ax.text(x*1.08, y*1.08, feature_label(feature), fontsize=8,
                ha="center", va="center")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title(f"{label}: PCA biplot\nTop {len(top)} feature-loading vectors")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_loading_heatmap(label: str, loadings: pd.DataFrame, path: Path):
    pcs = [c for c in loadings.columns if re.fullmatch(r"PC\d+", c)][:5]
    values = loadings[pcs].to_numpy(float)
    labels = [feature_label(f) for f in loadings.index]
    fig, ax = plt.subplots(figsize=(8.5, max(6.0, 0.42*len(labels)+2)))
    image = ax.imshow(values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pcs)))
    ax.set_xticklabels(pcs)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{label}: PCA feature loadings")
    fig.colorbar(image, ax=ax, label="Loading")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def choose_clusters(scores_2d: np.ndarray, k_min: int, k_max: int, seed: int):
    upper = min(k_max, len(scores_2d)-1)
    lower = max(2, k_min)
    results = {}
    for k in range(lower, upper+1):
        model = KMeans(n_clusters=k, n_init=50, random_state=seed)
        labels = model.fit_predict(scores_2d)
        if len(np.unique(labels)) >= 2:
            results[k] = float(silhouette_score(scores_2d, labels))
    return (max(results, key=results.get), results) if results else (None, {})


def plot_silhouette(label: str, results: dict[int, float], best_k: int, path: Path):
    ks = sorted(results)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(ks, [results[k] for k in ks], marker="o")
    ax.axvline(best_k, linestyle="--", linewidth=1)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of K-means clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"{label}: cluster-number selection (best k={best_k})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_clusters(label: str, scores: pd.DataFrame, labels: np.ndarray, pca: PCA, path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    for cluster in sorted(np.unique(labels)):
        sub = scores.loc[labels == cluster]
        ax.scatter(sub.PC1, sub.PC2, s=70, alpha=0.9, label=f"Cluster {cluster+1}")
        ax.text(sub.PC1.mean(), sub.PC2.mean(), str(cluster+1), fontsize=14,
                fontweight="bold", ha="center", va="center")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title(f"{label}: K-means groups in PCA space")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_composition(
    label: str,
    metadata: pd.DataFrame,
    genotype_col: str,
    labels: np.ndarray,
    group_order: list[str],
    path: Path,
) -> pd.DataFrame:
    table = metadata.reset_index(drop=True).copy()
    table["cluster"] = labels + 1
    counts = table.groupby([genotype_col, "cluster"]).size().unstack(fill_value=0)
    groups = [g for g in group_order if g in counts.index]
    groups.extend(g for g in counts.index if g not in groups)
    counts = counts.loc[groups]
    prop = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    bottom = np.zeros(len(prop))
    x = np.arange(len(prop))
    for cluster in prop.columns:
        values = prop[cluster].to_numpy(float)
        ax.bar(x, values, bottom=bottom, label=f"Cluster {cluster}")
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(prop.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of observations")
    ax.set_title(f"{label}: PCA-cluster composition by genotype")
    ax.legend(title="Phenotypic cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return table


def run_dataset(label: str, csv_path: Path, output_root: Path, args: argparse.Namespace):
    if not csv_path.exists():
        print(f"[WARN] Missing file for {label}: {csv_path}")
        return
    df = pd.read_csv(csv_path, low_memory=False)
    level = infer_level(df, args.level)
    fish_col = detect_column(df, args.fish_col, FISH_CANDIDATES, "fish")
    genotype_col = detect_column(df, args.genotype_col, GENOTYPE_CANDIDATES, "genotype")
    track_col = detect_column(df, args.track_col, TRACK_CANDIDATES, "track", required=False)
    df = df.copy()
    df[fish_col] = df[fish_col].astype(str).str.strip()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    identity = {fish_col, genotype_col}
    if track_col:
        identity.add(track_col)
    features = choose_features(df, level, args.view, args.feature, identity)
    metadata_cols = [fish_col, genotype_col] + ([track_col] if track_col else [])
    working = df[metadata_cols + features].copy()
    working = working[
        working[fish_col].notna() & working[genotype_col].notna() &
        working[fish_col].ne("") & working[genotype_col].ne("")
    ].reset_index(drop=True)
    if len(working) < 4:
        print(f"[WARN] Too few observations for {label}")
        return

    out = output_root / safe_name(label)
    out.mkdir(parents=True, exist_ok=True)
    metadata = working[metadata_cols].copy()
    pca, scores, loadings = prepare_pca(working, features)

    pd.concat([metadata.reset_index(drop=True), scores], axis=1).to_csv(out / "pca_scores.csv", index=False)
    loadings.reset_index(names="feature").to_csv(out / "pca_loadings.csv", index=False)
    pd.DataFrame({
        "principal_component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(out / "pca_explained_variance.csv", index=False)
    pd.DataFrame({"feature": features, "display_name": [feature_label(f) for f in features]}).to_csv(
        out / "pca_features_used.csv", index=False
    )

    plot_scree(label, pca, out / "01_scree.png")
    plot_by_group(label, metadata, scores, pca, fish_col, genotype_col,
                  args.group_order, out / "02_pca_by_genotype.png",
                  level == "fish" and not args.no_labels)
    plot_panels(label, metadata, scores, pca, fish_col, genotype_col,
                args.group_order, out / "03_genotype_panels.png",
                level == "fish" and not args.no_labels)
    plot_biplot(label, metadata, scores, loadings, pca, genotype_col,
                args.group_order, out / "04_biplot.png", args.top_loading_arrows)
    plot_loading_heatmap(label, loadings, out / "05_loading_heatmap.png")
    if scores.shape[1] >= 3:
        plot_by_group(label, metadata, scores, pca, fish_col, genotype_col,
                      args.group_order, out / "06_pc1_pc3.png",
                      level == "fish" and not args.no_labels, pc_x=1, pc_y=3)

    if not args.no_clustering and scores.shape[1] >= 2:
        score_2d = scores[["PC1", "PC2"]].to_numpy(float)
        best_k, results = choose_clusters(score_2d, args.cluster_min,
                                          args.cluster_max, args.random_seed)
        if best_k is not None:
            plot_silhouette(label, results, best_k, out / "07_silhouette.png")
            kmeans = KMeans(n_clusters=best_k, n_init=100, random_state=args.random_seed)
            cluster_labels = kmeans.fit_predict(score_2d)
            plot_clusters(label, scores, cluster_labels, pca, out / "08_kmeans_clusters.png")
            assignments = plot_composition(label, metadata, genotype_col, cluster_labels,
                                           args.group_order, out / "09_cluster_composition.png")
            assignments.to_csv(out / "pca_cluster_assignments.csv", index=False)
            pd.DataFrame({
                "k": sorted(results),
                "silhouette_score": [results[k] for k in sorted(results)],
                "selected": [k == best_k for k in sorted(results)],
            }).to_csv(out / "kmeans_silhouette_scores.csv", index=False)

    print(f"[DONE] {label}: level={level}, observations={len(working)}, features={len(features)}")
    print(f"       Saved to: {out}")


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for label, path in parse_datasets(args.dataset).items():
        run_dataset(label, path, output_root, args)
    print(f"[DONE] All PCA outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
