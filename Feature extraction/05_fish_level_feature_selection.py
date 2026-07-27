from __future__ import annotations

import argparse
import itertools
import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = {
    "musc": [
        "directionality_ratio",
        "mean_elongation",
        "mean_sphericity",
        "mean_speed_um_per_frame",
        "mean_volume_um3",
        "net_displacement_3d_um",
        "total_path_length_3d_um",
        "z_range_um",
    ],
    "macrophage_all": [
        "mean_elongation",
        "median_elongation",
        "mean_sphericity",
        "mean_speed_um_per_frame",
        "mean_step_distance_3d_um",
        "mean_volume_um3",
        "net_displacement_3d_um",
        "total_path_length_3d_um",
        "directionality_ratio",
        "moving_step_fraction",
        "z_range_um",
    ],
    "macrophage_outside_boundary": [
        "mean_elongation",
        "median_elongation",
        "mean_sphericity",
        "mean_speed_um_per_frame",
        "mean_step_distance_3d_um",
        "mean_volume_um3",
        "net_displacement_3d_um",
        "total_path_length_3d_um",
        "directionality_ratio",
        "moving_step_fraction",
        "z_range_um",
        "near_cluster_boundary_fraction",
        "mean_distance_to_cluster_boundary_px",
        "min_distance_to_cluster_boundary_px",
    ],
}

FEATURE_FAMILIES = {
    "morphology": [
        "mean_elongation",
        "median_elongation",
        "mean_sphericity",
        "mean_volume_um3",
    ],
    "motility": [
        "directionality_ratio",
        "mean_speed_um_per_frame",
        "mean_step_distance_3d_um",
        "net_displacement_3d_um",
        "total_path_length_3d_um",
        "moving_step_fraction",
        "z_range_um",
    ],
    "spatial_cluster": [
        "near_cluster_boundary_fraction",
        "mean_distance_to_cluster_boundary_px",
        "min_distance_to_cluster_boundary_px",
        "mean_distance_to_cluster_boundary_um",
        "min_distance_to_cluster_boundary_um",
    ],
}

FISH_COLUMN_CANDIDATES = [
    "fish_id",
    "block_name",
    "block",
    "sample_id",
]

GENOTYPE_COLUMN_CANDIDATES = [
    "genotype",
    "group",
    "condition",
    "class",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fish-level PCA subset and feature-ablation analysis."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(BASE_FEATURES),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Optional comma-separated base feature names. "
            "The script automatically uses median_<feature> columns."
        ),
    )
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--max-pruned-features",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Number of label permutations per tested subset.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--run-exhaustive",
        action="store_true",
        help=(
            "Also test all feature subsets from size 2 to "
            "--exhaustive-max-size. Exploratory only."
        ),
    )
    parser.add_argument(
        "--exhaustive-max-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--top-exhaustive-plots",
        type=int,
        default=10,
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "unnamed"


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"{role} column '{explicit}' was not found."
            )
        return explicit

    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not detect {role} column. "
        f"Use --{role.replace('_', '-')}-col."
    )


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()

    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"

    return text


def resolve_feature_column(df: pd.DataFrame, base_feature: str) -> str | None:
    candidates = [
        f"median_{base_feature}",
        base_feature,
    ]

    for candidate in candidates:
        if candidate in df.columns:
            numeric = pd.to_numeric(df[candidate], errors="coerce")
            if numeric.notna().sum() >= 3 and numeric.nunique(dropna=True) >= 2:
                df[candidate] = numeric
                return candidate

    return None


# Cliff's delta reference: https://revistas.javeriana.edu.co/index.php/revPsycho/article/view/643
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    differences = a[:, None] - b[None, :]
    wins = np.sum(differences > 0)
    losses = np.sum(differences < 0)
    return float((wins - losses) / (len(a) * len(b)))


def calculate_univariate_effects(
    df: pd.DataFrame,
    genotype_col: str,
    feature_columns: dict[str, str],
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for base_feature, column in feature_columns.items():
        a = (
            df.loc[df[genotype_col] == group_a, column]
            .dropna()
            .to_numpy(dtype=float)
        )
        b = (
            df.loc[df[genotype_col] == group_b, column]
            .dropna()
            .to_numpy(dtype=float)
        )

        delta = cliffs_delta(a, b)

        records.append(
            {
                "feature": base_feature,
                "column": column,
                "cliffs_delta": delta,
                "absolute_cliffs_delta": abs(delta)
                if np.isfinite(delta)
                else np.nan,
                f"median_{safe_name(group_a)}": np.median(a)
                if len(a)
                else np.nan,
                f"median_{safe_name(group_b)}": np.median(b)
                if len(b)
                else np.nan,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("absolute_cliffs_delta", ascending=False)
        .reset_index(drop=True)
    )


def make_correlation_matrix(
    df: pd.DataFrame,
    feature_columns: dict[str, str],
) -> pd.DataFrame:
    base_order = list(feature_columns)
    matrix = df[[feature_columns[f] for f in base_order]].copy()
    matrix.columns = base_order
    return matrix.corr(method="spearman")


def effect_correlation_prune(
    effect_table: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    threshold: float,
    max_features: int,
) -> list[str]:
    selected: list[str] = []

    for feature in effect_table["feature"]:
        if feature not in correlation_matrix.columns:
            continue

        too_correlated = False
        for kept in selected:
            correlation = correlation_matrix.loc[feature, kept]
            if np.isfinite(correlation) and abs(correlation) >= threshold:
                too_correlated = True
                break

        if not too_correlated:
            selected.append(feature)

        if len(selected) >= max_features:
            break

    return selected


def prepare_matrix(
    df: pd.DataFrame,
    base_features: list[str],
    feature_columns: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    features = [
        feature for feature in base_features
        if feature in feature_columns
    ]
    if not features:
        return np.empty((len(df), 0)), []

    matrix = df[[feature_columns[f] for f in features]].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    scaled = StandardScaler().fit_transform(imputed)

    return scaled, features


# PCA API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def pca_metrics(
    matrix: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], PCA]:
    if matrix.shape[1] < 2:
        raise ValueError("At least two features are required for PCA.")

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(matrix)

    groups = np.unique(y)
    if len(groups) != 2:
        raise ValueError("Exactly two genotype groups are required.")

    centroids = {}
    within_distances = []

    for group in groups:
        group_coordinates = coordinates[y == group]
        centroid = group_coordinates.mean(axis=0)
        centroids[group] = centroid
        within_distances.extend(
            np.linalg.norm(group_coordinates - centroid, axis=1)
        )

    centroid_distance = float(
        np.linalg.norm(centroids[groups[0]] - centroids[groups[1]])
    )
    within_scatter = float(np.mean(within_distances))
    separation_ratio = centroid_distance / (within_scatter + 1e-12)

    if (
        len(coordinates) >= 4
        and all(np.sum(y == group) >= 2 for group in groups)
    ):
        try:
            silhouette = float(silhouette_score(coordinates, y))
        except ValueError:
            silhouette = np.nan
    else:
        silhouette = np.nan

    metrics = {
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
        "pc1_pc2_total_variance": float(
            pca.explained_variance_ratio_[:2].sum()
        ),
        "centroid_distance": centroid_distance,
        "within_group_scatter": within_scatter,
        "pca_separation_ratio": separation_ratio,
        "genotype_silhouette": silhouette,
    }

    return coordinates, metrics, pca


def make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


# Leave-one-out CV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html
def leave_one_fish_out_predictions(
    matrix: np.ndarray,
    y_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    loo = LeaveOneOut()
    predicted = np.zeros(len(y_binary), dtype=int)
    probabilities = np.zeros(len(y_binary), dtype=float)

    for train_index, test_index in loo.split(matrix):
        x_train = matrix[train_index]
        x_test = matrix[test_index]
        y_train = y_binary[train_index]

        if len(np.unique(y_train)) < 2:
            majority = int(np.round(np.mean(y_train)))
            predicted[test_index[0]] = majority
            probabilities[test_index[0]] = float(majority)
            continue

        model = make_classifier()
        model.fit(x_train, y_train)

        predicted[test_index[0]] = int(model.predict(x_test)[0])
        probabilities[test_index[0]] = float(
            model.predict_proba(x_test)[0, 1]
        )

    return predicted, probabilities


def classifier_metrics(
    matrix: np.ndarray,
    y_binary: np.ndarray,
) -> dict[str, float]:
    predicted, probabilities = leave_one_fish_out_predictions(
        matrix, y_binary
    )

    balanced_accuracy = float(
        balanced_accuracy_score(y_binary, predicted)
    )

    if len(np.unique(y_binary)) == 2:
        try:
            auc = float(roc_auc_score(y_binary, probabilities))
        except ValueError:
            auc = np.nan
    else:
        auc = np.nan

    return {
        "lofo_balanced_accuracy": balanced_accuracy,
        "lofo_roc_auc": auc,
    }


# Permutation-test method: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
def permutation_p_value(
    matrix: np.ndarray,
    y_binary: np.ndarray,
    observed_balanced_accuracy: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if n_permutations <= 0:
        return np.nan, np.nan, np.nan

    scores = np.empty(n_permutations, dtype=float)

    for index in range(n_permutations):
        permuted = rng.permutation(y_binary)
        metrics = classifier_metrics(matrix, permuted)
        scores[index] = metrics["lofo_balanced_accuracy"]

    p_value = float(
        (1 + np.sum(scores >= observed_balanced_accuracy))
        / (n_permutations + 1)
    )

    return (
        p_value,
        float(np.mean(scores)),
        float(np.std(scores, ddof=1)),
    )


def plot_pca(
    coordinates: np.ndarray,
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    subset_name: str,
    features: list[str],
    pca: PCA,
    output_path: Path,
    group_a: str,
    group_b: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.4))

    groups = [
        group for group in [group_a, group_b]
        if group in set(df[genotype_col])
    ]
    groups.extend(
        sorted(
            group for group in set(df[genotype_col])
            if group not in groups
        )
    )

    for group in groups:
        mask = df[genotype_col].to_numpy() == group
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=72,
            alpha=0.9,
            label=group,
        )

        for x, y, fish in zip(
            coordinates[mask, 0],
            coordinates[mask, 1],
            df.loc[mask, fish_col],
        ):
            ax.annotate(
                str(fish),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title(
        f"{subset_name}\n"
        f"{len(features)} features: {', '.join(features)}"
    )
    ax.axhline(0, linewidth=0.8, alpha=0.35)
    ax.axvline(0, linewidth=0.8, alpha=0.35)
    ax.grid(alpha=0.2)
    ax.legend(title="Genotype")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def analyse_subset(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    feature_columns: dict[str, str],
    subset_name: str,
    features: list[str],
    group_a: str,
    group_b: str,
    n_permutations: int,
    rng: np.random.Generator,
    plot_dir: Path | None,
) -> dict[str, object] | None:
    features = [
        feature for feature in features
        if feature in feature_columns
    ]

    if len(features) < 2:
        return None

    matrix, features = prepare_matrix(
        df,
        features,
        feature_columns,
    )

    if matrix.shape[1] < 2:
        return None

    y = df[genotype_col].astype(str).to_numpy()
    y_binary = (y == group_b).astype(int)

    if len(np.unique(y_binary)) != 2:
        return None

    coordinates, pca_result, pca = pca_metrics(matrix, y)
    classifier_result = classifier_metrics(matrix, y_binary)

    (
        permutation_p,
        permutation_mean,
        permutation_std,
    ) = permutation_p_value(
        matrix,
        y_binary,
        classifier_result["lofo_balanced_accuracy"],
        n_permutations,
        rng,
    )

    record: dict[str, object] = {
        "subset_name": subset_name,
        "n_features": len(features),
        "features": "|".join(features),
        **pca_result,
        **classifier_result,
        "permutation_p_balanced_accuracy": permutation_p,
        "permutation_mean_balanced_accuracy": permutation_mean,
        "permutation_std_balanced_accuracy": permutation_std,
    }

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_pca(
            coordinates,
            df,
            fish_col,
            genotype_col,
            subset_name,
            features,
            pca,
            plot_dir / f"{safe_name(subset_name)}.png",
            group_a,
            group_b,
        )

    return record


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    if correlation_matrix.empty:
        return

    size = max(7.0, 0.65 * len(correlation_matrix) + 3.5)
    fig, ax = plt.subplots(figsize=(size, size))

    image = ax.imshow(
        correlation_matrix.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    labels = list(correlation_matrix.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Fish-level Spearman feature correlations")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_subset_ranking(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    table = summary.dropna(subset=[metric]).copy()
    if table.empty:
        return

    table = table.sort_values(metric, ascending=True)
    fig_height = max(5.0, 0.42 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    y = np.arange(len(table))
    ax.scatter(table[metric], y, s=58)
    ax.set_yticks(y)
    ax.set_yticklabels(table["subset_name"])
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_predefined_subsets(
    candidate_features: list[str],
    pruned_features: list[str],
) -> dict[str, list[str]]:
    subsets: dict[str, list[str]] = {
        "all_candidates": list(candidate_features),
    }

    for family_name, family_features in FEATURE_FAMILIES.items():
        available = [
            feature for feature in family_features
            if feature in candidate_features
        ]
        if len(available) >= 2:
            subsets[family_name] = available

    if len(pruned_features) >= 2:
        subsets["effect_correlation_pruned"] = pruned_features

    for removed_feature in candidate_features:
        remaining = [
            feature for feature in candidate_features
            if feature != removed_feature
        ]
        if len(remaining) >= 2:
            subsets[
                f"leave_one_out__remove_{removed_feature}"
            ] = remaining

    return subsets


def build_exhaustive_subsets(
    candidate_features: list[str],
    max_size: int,
) -> dict[str, list[str]]:
    subsets: dict[str, list[str]] = {}

    maximum = min(max_size, len(candidate_features))
    for size in range(2, maximum + 1):
        for combination in itertools.combinations(
            candidate_features, size
        ):
            name = f"exhaustive_{size}__" + "__".join(combination)
            subsets[name] = list(combination)

    return subsets


def add_ablation_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    baseline = summary.loc[
        summary["subset_name"] == "all_candidates"
    ]

    if baseline.empty:
        return summary

    baseline_row = baseline.iloc[0]
    for metric in [
        "pca_separation_ratio",
        "genotype_silhouette",
        "lofo_balanced_accuracy",
        "lofo_roc_auc",
    ]:
        summary[f"delta_vs_all__{metric}"] = (
            summary[metric] - baseline_row[metric]
        )

    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        raise ValueError("Input fish-level table is empty.")

    fish_col = detect_column(
        df,
        args.fish_col,
        FISH_COLUMN_CANDIDATES,
        "fish",
    )
    genotype_col = detect_column(
        df,
        args.genotype_col,
        GENOTYPE_COLUMN_CANDIDATES,
        "genotype",
    )

    df = df.copy()
    df[fish_col] = df[fish_col].astype(str).str.strip()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)

    df = df[
        df[genotype_col].isin([args.group_a, args.group_b])
    ].copy()
    df = df.sort_values([genotype_col, fish_col]).reset_index(drop=True)

    if args.features:
        requested_features = [
            feature.strip()
            for feature in args.features.split(",")
            if feature.strip()
        ]
    else:
        requested_features = BASE_FEATURES[args.dataset]

    feature_columns: dict[str, str] = {}
    missing_features: list[str] = []

    for base_feature in requested_features:
        column = resolve_feature_column(df, base_feature)
        if column is None:
            missing_features.append(base_feature)
        else:
            feature_columns[base_feature] = column

    if missing_features:
        print("[WARN] Unavailable features:")
        for feature in missing_features:
            print(f"       - {feature}")

    candidate_features = list(feature_columns)
    if len(candidate_features) < 2:
        raise ValueError("Fewer than two usable candidate features.")

    print(f"[INFO] Fish: {len(df)}")
    print(
        f"[INFO] Genotypes: "
        f"{df[genotype_col].value_counts().to_dict()}"
    )
    print("[INFO] Candidate features:")
    for feature in candidate_features:
        print(f"       - {feature}")

    effect_table = calculate_univariate_effects(
        df,
        genotype_col,
        feature_columns,
        args.group_a,
        args.group_b,
    )
    effect_table.to_csv(
        output_dir / "univariate_effect_sizes.csv",
        index=False,
    )

    correlation_matrix = make_correlation_matrix(
        df,
        feature_columns,
    )
    correlation_matrix.to_csv(
        output_dir / "feature_spearman_correlations.csv"
    )
    plot_correlation_heatmap(
        correlation_matrix,
        output_dir / "feature_spearman_correlations.png",
    )

    pruned_features = effect_correlation_prune(
        effect_table,
        correlation_matrix,
        args.correlation_threshold,
        args.max_pruned_features,
    )

    pd.DataFrame(
        {
            "selected_order": np.arange(1, len(pruned_features) + 1),
            "feature": pruned_features,
        }
    ).to_csv(
        output_dir / "effect_correlation_pruned_features.csv",
        index=False,
    )

    print("[INFO] Effect/correlation-pruned subset:")
    for feature in pruned_features:
        print(f"       - {feature}")

    rng = np.random.default_rng(args.random_seed)
    plot_dir = output_dir / "pca_subsets"

    predefined_subsets = build_predefined_subsets(
        candidate_features,
        pruned_features,
    )

    records: list[dict[str, object]] = []

    for subset_name, features in predefined_subsets.items():
        print(
            f"[INFO] Testing {subset_name}: "
            f"{len(features)} features"
        )
        record = analyse_subset(
            df,
            fish_col,
            genotype_col,
            feature_columns,
            subset_name,
            features,
            args.group_a,
            args.group_b,
            args.permutations,
            rng,
            plot_dir,
        )
        if record is not None:
            records.append(record)

    exhaustive_records: list[dict[str, object]] = []
    if args.run_exhaustive:
        exhaustive_subsets = build_exhaustive_subsets(
            candidate_features,
            args.exhaustive_max_size,
        )

        print(
            f"[INFO] Testing {len(exhaustive_subsets)} exhaustive subsets."
        )

        for subset_name, features in exhaustive_subsets.items():
            record = analyse_subset(
                df,
                fish_col,
                genotype_col,
                feature_columns,
                subset_name,
                features,
                args.group_a,
                args.group_b,
                args.permutations,
                rng,
                plot_dir=None,
            )
            if record is not None:
                exhaustive_records.append(record)

        exhaustive_summary = pd.DataFrame(exhaustive_records)
        if not exhaustive_summary.empty:
            exhaustive_summary = exhaustive_summary.sort_values(
                [
                    "lofo_balanced_accuracy",
                    "pca_separation_ratio",
                ],
                ascending=False,
            ).reset_index(drop=True)
            exhaustive_summary.to_csv(
                output_dir / "exhaustive_subset_summary.csv",
                index=False,
            )

            top_plot_dir = output_dir / "top_exhaustive_pca"
            for _, row in exhaustive_summary.head(
                args.top_exhaustive_plots
            ).iterrows():
                features = str(row["features"]).split("|")
                analyse_subset(
                    df,
                    fish_col,
                    genotype_col,
                    feature_columns,
                    str(row["subset_name"]),
                    features,
                    args.group_a,
                    args.group_b,
                    n_permutations=0,
                    rng=rng,
                    plot_dir=top_plot_dir,
                )

    summary = pd.DataFrame(records)
    if summary.empty:
        raise ValueError("No valid subsets could be analysed.")

    summary = add_ablation_deltas(summary)
    summary = summary.sort_values(
        [
            "lofo_balanced_accuracy",
            "pca_separation_ratio",
        ],
        ascending=False,
    ).reset_index(drop=True)

    summary.to_csv(
        output_dir / "predefined_and_ablation_summary.csv",
        index=False,
    )

    leave_one_out = summary[
        summary["subset_name"].str.startswith(
            "leave_one_out__remove_"
        )
    ].copy()
    leave_one_out.to_csv(
        output_dir / "leave_one_feature_out_summary.csv",
        index=False,
    )

    plot_subset_ranking(
        summary,
        "pca_separation_ratio",
        output_dir / "subset_ranking_pca_separation.png",
        "Feature subsets ranked by PCA separation ratio",
    )
    plot_subset_ranking(
        summary,
        "lofo_balanced_accuracy",
        output_dir / "subset_ranking_lofo_balanced_accuracy.png",
        "Feature subsets ranked by leave-one-fish-out balanced accuracy",
    )
    plot_subset_ranking(
        summary,
        "genotype_silhouette",
        output_dir / "subset_ranking_genotype_silhouette.png",
        "Feature subsets ranked by genotype silhouette",
    )

    run_information = [
        f"input={args.input}",
        f"dataset={args.dataset}",
        f"fish_col={fish_col}",
        f"genotype_col={genotype_col}",
        f"group_a={args.group_a}",
        f"group_b={args.group_b}",
        f"correlation_threshold={args.correlation_threshold}",
        f"max_pruned_features={args.max_pruned_features}",
        f"permutations={args.permutations}",
        "candidate_features=" + ",".join(candidate_features),
        "pruned_features=" + ",".join(pruned_features),
    ]
    (output_dir / "run_information.txt").write_text(
        "\n".join(run_information) + "\n",
        encoding="utf-8",
    )

    print()
    print("[DONE] Feature-selection analysis complete.")
    print(
        f"[SAVED] {output_dir / 'predefined_and_ablation_summary.csv'}"
    )
    print(
        f"[SAVED] {output_dir / 'leave_one_feature_out_summary.csv'}"
    )
    print(f"[SAVED] PCA plots: {plot_dir}")


if __name__ == "__main__":
    main()
