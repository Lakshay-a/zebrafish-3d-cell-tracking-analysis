from __future__ import annotations

import argparse
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler


FISH_COLUMN_CANDIDATES = [
    "fish_id",
    "block_name",
    "block",
    "source_block",
    "sample_id",
    "dataset_id",
    "czi_name",
    "file",
]

GENOTYPE_COLUMN_CANDIDATES = [
    "genotype",
    "group",
    "condition",
    "class",
    "label",
]

TRACK_COLUMN_CANDIDATES = [
    "track_id",
    "global_track_id",
    "cell_track_id",
    "cell_id",
    "object_track_id",
]

# These columns/patterns are identifiers or direct outcome leakage, not
# candidate biological features. In all_numeric mode, only these are removed.
IDENTIFIER_EXACT_NAMES = {
    "genotype",
    "group",
    "condition",
    "class",
    "label",
    "fish_id",
    "block",
    "block_name",
    "source_block",
    "sample_id",
    "dataset_id",
    "czi_name",
    "file",
    "filename",
    "path",
    "track_id",
    "global_track_id",
    "cell_track_id",
    "cell_id",
    "object_track_id",
    "object_label",
    "index",
    "row_index",
}

IDENTIFIER_PATTERNS = [
    r"(^|_)genotype($|_)",
    r"(^|_)fish_?id($|_)",
    r"(^|_)track_?id($|_)",
    r"(^|_)cell_?id($|_)",
    r"(^|_)object_?id($|_)",
    r"(^|_)object_?label($|_)",
    r"(^|_)block_?name($|_)",
    r"(^|_)sample_?id($|_)",
    r"(^|_)dataset_?id($|_)",
]

# Optional extra exclusion for the recommended "biological" mode.
# This prevents raw position, acquisition metadata and explicit QC fields from
# being selected as genotype predictors.
TECHNICAL_PATTERNS = [
    r"(^|_)(start|end|centroid|bbox)_[xyz]($|_)",
    r"(^|_)[xyz]_(min|max)$",
    r"^(x|y|z)$",
    r"(^|_)(time|frame|timepoint)($|_)",
    r"(^|_)(qc|outlier|removed|exclude|exclusion|flag)($|_)",
    r"(^|_)(pixel_size|voxel_size|z_step|xy_pixel)($|_)",
    r"(^|_)(source|filename|filepath|folder|path)($|_)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "All-numeric fish-level aggregation, nested LOFO feature "
            "selection, PCA and backward elimination."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--track-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")

    parser.add_argument(
        "--candidate-mode",
        choices=["all_numeric", "biological"],
        default="all_numeric",
        help=(
            "all_numeric: use every numeric non-identifier column. "
            "biological: additionally exclude raw coordinates, acquisition "
            "metadata and explicit QC fields."
        ),
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=[],
        help=(
            "Additional regex pattern to exclude. May be supplied multiple "
            "times."
        ),
    )
    parser.add_argument(
        "--include-regex",
        default=None,
        help=(
            "Optional regex: only numeric columns matching this pattern are "
            "considered."
        ),
    )

    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--min-unique-fish-values",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--inner-top-k",
        type=int,
        default=25,
        help=(
            "Within each outer training fold, rank all surviving features "
            "and pass at most this many to L1 logistic regression."
        ),
    )
    parser.add_argument(
        "--selection-frequency-threshold",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--max-final-features",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--min-final-features",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help=(
            "Label permutations for the complete nested LOFO pipeline. "
            "Use 0 to skip."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "unnamed"


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested {role} column '{explicit}' was not found."
            )
        return explicit

    lower_map = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not detect {role} column. "
            f"Provide --{role.replace('_', '-')}-col."
        )
    return None


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()

    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"

    return text


def matches_any_pattern(column: str, patterns: list[str]) -> bool:
    return any(
        re.search(pattern, column, flags=re.IGNORECASE)
        for pattern in patterns
    )


def identify_numeric_candidates(
    df: pd.DataFrame,
    protected_columns: set[str],
    candidate_mode: str,
    include_regex: str | None,
    extra_exclude_patterns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    converted = pd.DataFrame(index=df.index)
    audit_records: list[dict[str, object]] = []

    for column in df.columns:
        column_str = str(column)
        lower = column_str.lower()

        if column_str in protected_columns or lower in IDENTIFIER_EXACT_NAMES:
            audit_records.append(
                {
                    "feature": column_str,
                    "stage": "candidate_detection",
                    "kept": False,
                    "reason": "identifier_or_protected_column",
                }
            )
            continue

        if matches_any_pattern(column_str, IDENTIFIER_PATTERNS):
            audit_records.append(
                {
                    "feature": column_str,
                    "stage": "candidate_detection",
                    "kept": False,
                    "reason": "identifier_pattern",
                }
            )
            continue

        if (
            candidate_mode == "biological"
            and matches_any_pattern(column_str, TECHNICAL_PATTERNS)
        ):
            audit_records.append(
                {
                    "feature": column_str,
                    "stage": "candidate_detection",
                    "kept": False,
                    "reason": "technical_pattern_biological_mode",
                }
            )
            continue

        if extra_exclude_patterns and matches_any_pattern(
            column_str, extra_exclude_patterns
        ):
            audit_records.append(
                {
                    "feature": column_str,
                    "stage": "candidate_detection",
                    "kept": False,
                    "reason": "user_exclude_regex",
                }
            )
            continue

        if include_regex and not re.search(
            include_regex, column_str, flags=re.IGNORECASE
        ):
            audit_records.append(
                {
                    "feature": column_str,
                    "stage": "candidate_detection",
                    "kept": False,
                    "reason": "did_not_match_include_regex",
                }
            )
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        nonmissing_count = int(numeric.notna().sum())

        if nonmissing_count < 2:
            audit_records.append(
                {
                    "feature": column_str,
                    "stage": "candidate_detection",
                    "kept": False,
                    "reason": "not_numeric_or_too_few_numeric_values",
                }
            )
            continue

        converted[column_str] = numeric.replace([np.inf, -np.inf], np.nan)
        audit_records.append(
            {
                "feature": column_str,
                "stage": "candidate_detection",
                "kept": True,
                "reason": "numeric_candidate",
            }
        )

    return converted, pd.DataFrame(audit_records)


def aggregate_to_fish_medians(
    identity_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    track_col: str | None,
) -> pd.DataFrame:
    combined = pd.concat(
        [identity_df[[fish_col, genotype_col]], numeric_df],
        axis=1,
    )

    grouped = combined.groupby(
        [fish_col, genotype_col],
        dropna=False,
        sort=True,
    )

    fish_table = grouped.median(numeric_only=True).reset_index()
    row_counts = grouped.size().reset_index(name="n_cell_track_rows")
    fish_table = fish_table.merge(
        row_counts,
        on=[fish_col, genotype_col],
        how="left",
    )

    if track_col and track_col in identity_df.columns:
        track_counts = (
            identity_df.groupby([fish_col, genotype_col])[track_col]
            .nunique(dropna=True)
            .reset_index(name="n_unique_tracks")
        )
        fish_table = fish_table.merge(
            track_counts,
            on=[fish_col, genotype_col],
            how="left",
        )

    return fish_table.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)


def median_absolute_deviation(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def unsupervised_filter_features(
    fish_table: pd.DataFrame,
    candidate_features: list[str],
    max_missing_fraction: float,
    min_unique_values: int,
    correlation_threshold: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    audit_records: list[dict[str, object]] = []
    survivors: list[str] = []

    for feature in candidate_features:
        values = pd.to_numeric(
            fish_table[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        missing_fraction = float(values.isna().mean())
        unique_count = int(values.nunique(dropna=True))
        mad = median_absolute_deviation(values)

        if missing_fraction > max_missing_fraction:
            kept = False
            reason = "excessive_fish_level_missingness"
        elif unique_count < min_unique_values:
            kept = False
            reason = "too_few_unique_fish_values"
        elif math.isclose(mad, 0.0, abs_tol=1e-12):
            kept = False
            reason = "zero_or_near_zero_variability"
        else:
            kept = True
            reason = "passed_missingness_and_variance"

        audit_records.append(
            {
                "feature": feature,
                "stage": "missingness_variance_filter",
                "kept": kept,
                "reason": reason,
                "fish_missing_fraction": missing_fraction,
                "fish_unique_values": unique_count,
                "fish_mad": mad,
            }
        )
        if kept:
            survivors.append(feature)

    if not survivors:
        raise ValueError(
            "No features survived missingness/variance filtering."
        )

    matrix = fish_table[survivors].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    correlation_matrix = matrix.corr(method="spearman").abs()

    feature_quality = {}
    for feature in survivors:
        values = matrix[feature]
        feature_quality[feature] = {
            "missing": float(values.isna().mean()),
            "mad": median_absolute_deviation(values),
        }

    removed_for_correlation: set[str] = set()
    correlated_pairs: list[dict[str, object]] = []

    for i, feature_a in enumerate(survivors):
        if feature_a in removed_for_correlation:
            continue

        for feature_b in survivors[i + 1 :]:
            if feature_b in removed_for_correlation:
                continue

            correlation = correlation_matrix.loc[feature_a, feature_b]
            if not np.isfinite(correlation):
                continue
            if correlation < correlation_threshold:
                continue

            quality_a = feature_quality[feature_a]
            quality_b = feature_quality[feature_b]

            if quality_a["missing"] < quality_b["missing"]:
                keep, remove = feature_a, feature_b
            elif quality_b["missing"] < quality_a["missing"]:
                keep, remove = feature_b, feature_a
            elif quality_a["mad"] >= quality_b["mad"]:
                keep, remove = feature_a, feature_b
            else:
                keep, remove = feature_b, feature_a

            removed_for_correlation.add(remove)
            correlated_pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "absolute_spearman": float(correlation),
                    "kept_feature": keep,
                    "removed_feature": remove,
                }
            )

    final_features = [
        feature for feature in survivors
        if feature not in removed_for_correlation
    ]

    for feature in survivors:
        audit_records.append(
            {
                "feature": feature,
                "stage": "correlation_filter",
                "kept": feature in final_features,
                "reason": (
                    "passed_correlation_filter"
                    if feature in final_features
                    else "removed_as_correlated_duplicate"
                ),
            }
        )

    return (
        final_features,
        pd.DataFrame(audit_records),
        pd.DataFrame(correlated_pairs),
    )


# Cliff's delta reference: https://revistas.javeriana.edu.co/index.php/revPsycho/article/view/643
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return 0.0

    differences = a[:, None] - b[None, :]
    wins = np.sum(differences > 0)
    losses = np.sum(differences < 0)
    return float((wins - losses) / (len(a) * len(b)))


def rank_features_within_training(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    top_k: int,
) -> list[str]:
    scores = []

    for feature in x_train.columns:
        values = pd.to_numeric(
            x_train[feature], errors="coerce"
        ).to_numpy(dtype=float)

        a = values[y_train == 0]
        b = values[y_train == 1]
        score = abs(cliffs_delta(a, b))
        scores.append((feature, score))

    scores.sort(key=lambda item: (-item[1], item[0]))
    k = min(top_k, len(scores))
    return [feature for feature, _ in scores[:k]]


def choose_regularisation_c(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_seed: int,
) -> float:
    class_counts = np.bincount(y_train)
    positive_counts = class_counts[class_counts > 0]

    if len(positive_counts) < 2:
        return 1.0

    min_class = int(positive_counts.min())
    n_splits = min(3, min_class)

    if n_splits < 2:
        return 1.0

    c_grid = np.logspace(-3, 2, 12)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    mean_scores = []

    for c_value in c_grid:
        fold_scores = []

        for train_index, validation_index in splitter.split(
            x_train, y_train
        ):
            train_fold = x_train.iloc[train_index]
            validation_fold = x_train.iloc[validation_index]
            y_fold = y_train[train_index]
            y_validation = y_train[validation_index]

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            x_fold = imputer.fit_transform(train_fold)
            x_validation = imputer.transform(validation_fold)

            x_fold = scaler.fit_transform(x_fold)
            x_validation = scaler.transform(x_validation)

            model = LogisticRegression(
                l1_ratio=1.0,
                solver="liblinear",
                C=c_value,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_seed,
            )
            model.fit(x_fold, y_fold)
            predictions = model.predict(x_validation)
            fold_scores.append(
                balanced_accuracy_score(y_validation, predictions)
            )

        mean_scores.append(float(np.mean(fold_scores)))

    best_score = max(mean_scores)
    candidate_indices = [
        index for index, score in enumerate(mean_scores)
        if math.isclose(score, best_score, abs_tol=1e-12)
    ]

    # Prefer stronger regularisation when tied.
    best_index = min(candidate_indices)
    return float(c_grid[best_index])


def fit_predict_outer_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    top_k: int,
    random_seed: int,
) -> tuple[int, float, list[str], dict[str, float], float]:
    ranked_features = rank_features_within_training(
        x_train, y_train, top_k
    )

    x_train_selected = x_train[ranked_features]
    x_test_selected = x_test[ranked_features]

    c_value = choose_regularisation_c(
        x_train_selected,
        y_train,
        random_seed,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train_array = imputer.fit_transform(x_train_selected)
    x_test_array = imputer.transform(x_test_selected)

    x_train_array = scaler.fit_transform(x_train_array)
    x_test_array = scaler.transform(x_test_array)

    model = LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=c_value,
        class_weight="balanced",
        max_iter=5000,
        random_state=random_seed,
    )
    model.fit(x_train_array, y_train)

    prediction = int(model.predict(x_test_array)[0])
    probability = float(model.predict_proba(x_test_array)[0, 1])

    coefficients = model.coef_[0]
    nonzero_features = []
    coefficient_map = {}

    for feature, coefficient in zip(ranked_features, coefficients):
        coefficient_map[feature] = float(coefficient)
        if not math.isclose(coefficient, 0.0, abs_tol=1e-10):
            nonzero_features.append(feature)

    return (
        prediction,
        probability,
        nonzero_features,
        coefficient_map,
        c_value,
    )


# Nested CV reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
def run_nested_lofo(
    fish_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_b: str,
    top_k: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    x = fish_table[features].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    y = (fish_table[genotype_col].astype(str) == group_b).astype(int).to_numpy()

    loo = LeaveOneOut()

    selection_counts = Counter()
    ranking_counts = Counter()
    coefficient_abs_sums = defaultdict(float)
    coefficient_signed_sums = defaultdict(float)

    prediction_records = []

    for fold_index, (train_index, test_index) in enumerate(
        loo.split(x), start=1
    ):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        (
            prediction,
            probability,
            nonzero_features,
            coefficient_map,
            c_value,
        ) = fit_predict_outer_fold(
            x_train,
            y_train,
            x_test,
            top_k,
            random_seed + fold_index,
        )

        ranked_features = rank_features_within_training(
            x_train, y_train, top_k
        )

        for feature in ranked_features:
            ranking_counts[feature] += 1

        for feature in nonzero_features:
            selection_counts[feature] += 1
            coefficient_abs_sums[feature] += abs(
                coefficient_map[feature]
            )
            coefficient_signed_sums[feature] += coefficient_map[feature]

        row = fish_table.iloc[test_index[0]]
        prediction_records.append(
            {
                fish_col: row[fish_col],
                genotype_col: row[genotype_col],
                "true_binary": int(y_test[0]),
                "predicted_binary": prediction,
                "probability_group_b": probability,
                "correct": int(prediction == y_test[0]),
                "selected_nonzero_count": len(nonzero_features),
                "selected_nonzero_features": "|".join(nonzero_features),
                "inner_selected_C": c_value,
            }
        )

    predictions = pd.DataFrame(prediction_records)
    n_folds = len(predictions)

    stability_records = []
    for feature in features:
        selected_count = selection_counts[feature]
        ranked_count = ranking_counts[feature]

        stability_records.append(
            {
                "feature": feature,
                "outer_folds": n_folds,
                "top_k_frequency": ranked_count / n_folds,
                "nonzero_selection_frequency": selected_count / n_folds,
                "mean_absolute_coefficient_when_selected": (
                    coefficient_abs_sums[feature] / selected_count
                    if selected_count
                    else 0.0
                ),
                "mean_signed_coefficient_when_selected": (
                    coefficient_signed_sums[feature] / selected_count
                    if selected_count
                    else 0.0
                ),
                "stability_importance_score": (
                    (selected_count / n_folds)
                    * (
                        coefficient_abs_sums[feature] / selected_count
                        if selected_count
                        else 0.0
                    )
                ),
            }
        )

    stability = pd.DataFrame(stability_records).sort_values(
        [
            "nonzero_selection_frequency",
            "stability_importance_score",
        ],
        ascending=False,
    ).reset_index(drop=True)

    true = predictions["true_binary"].to_numpy()
    predicted = predictions["predicted_binary"].to_numpy()
    probabilities = predictions["probability_group_b"].to_numpy()

    metrics = {
        "n_fish": int(len(predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true, predicted)
        ),
        "roc_auc": float(roc_auc_score(true, probabilities))
        if len(np.unique(true)) == 2
        else np.nan,
    }

    tn, fp, fn, tp = confusion_matrix(
        true, predicted, labels=[0, 1]
    ).ravel()
    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )

    return predictions, stability, metrics


def select_stable_features(
    stability: pd.DataFrame,
    frequency_threshold: float,
    max_features: int,
    min_features: int,
) -> list[str]:
    selected = stability.loc[
        stability["nonzero_selection_frequency"] >= frequency_threshold,
        "feature",
    ].tolist()

    selected = selected[:max_features]

    if len(selected) < min_features:
        selected = stability["feature"].head(min_features).tolist()

    return selected


def pca_coordinates_and_metrics(
    fish_table: pd.DataFrame,
    features: list[str],
    genotype_col: str,
) -> tuple[np.ndarray, PCA, dict[str, float]]:
    matrix = fish_table[features].apply(
        pd.to_numeric, errors="coerce"
    )

    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    scaled = StandardScaler().fit_transform(imputed)

    n_components = min(2, scaled.shape[1])
    pca = PCA(n_components=n_components)
    coordinates = pca.fit_transform(scaled)

    if n_components == 1:
        coordinates = np.column_stack(
            [coordinates[:, 0], np.zeros(len(coordinates))]
        )

    labels = fish_table[genotype_col].astype(str).to_numpy()
    groups = np.unique(labels)

    centroid_distance = np.nan
    within_scatter = np.nan
    separation_ratio = np.nan
    silhouette = np.nan

    if len(groups) == 2:
        centroids = {}
        within_distances = []

        for group in groups:
            group_coordinates = coordinates[labels == group]
            centroid = group_coordinates.mean(axis=0)
            centroids[group] = centroid
            within_distances.extend(
                np.linalg.norm(group_coordinates - centroid, axis=1)
            )

        centroid_distance = float(
            np.linalg.norm(
                centroids[groups[0]] - centroids[groups[1]]
            )
        )
        within_scatter = float(np.mean(within_distances))
        separation_ratio = centroid_distance / (
            within_scatter + 1e-12
        )

        if all(np.sum(labels == group) >= 2 for group in groups):
            try:
                silhouette = float(
                    silhouette_score(coordinates, labels)
                )
            except ValueError:
                silhouette = np.nan

    explained = pca.explained_variance_ratio_
    metrics = {
        "n_features": len(features),
        "pc1_variance": float(explained[0])
        if len(explained) >= 1
        else np.nan,
        "pc2_variance": float(explained[1])
        if len(explained) >= 2
        else 0.0,
        "pc1_pc2_total_variance": float(explained[:2].sum()),
        "centroid_distance": centroid_distance,
        "within_group_scatter": within_scatter,
        "pca_separation_ratio": separation_ratio,
        "genotype_silhouette": silhouette,
    }

    return coordinates, pca, metrics


def plot_pca(
    fish_table: pd.DataFrame,
    coordinates: np.ndarray,
    pca: PCA,
    features: list[str],
    fish_col: str,
    genotype_col: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.4))

    for genotype, group in fish_table.groupby(genotype_col):
        indices = group.index.to_numpy()
        ax.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            s=70,
            alpha=0.9,
            label=str(genotype),
        )

        for index in indices:
            ax.annotate(
                str(fish_table.loc[index, fish_col]),
                (
                    coordinates[index, 0],
                    coordinates[index, 1],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    explained = pca.explained_variance_ratio_ * 100
    pc1 = explained[0] if len(explained) >= 1 else 0.0
    pc2 = explained[1] if len(explained) >= 2 else 0.0

    ax.set_xlabel(f"PC1 ({pc1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% variance)")
    ax.set_title(f"{title}\n{len(features)} features")
    ax.axhline(0, linewidth=0.8, alpha=0.35)
    ax.axvline(0, linewidth=0.8, alpha=0.35)
    ax.grid(alpha=0.2)
    ax.legend(title="Genotype")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_stability(
    stability: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> None:
    table = stability.head(top_n).sort_values(
        "nonzero_selection_frequency"
    )

    fig_height = max(5.5, 0.37 * len(table) + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    y = np.arange(len(table))
    ax.scatter(
        table["nonzero_selection_frequency"],
        y,
        s=58,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table["feature"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(
        "Non-zero L1 selection frequency across held-out-fish folds"
    )
    ax.set_title("Fish-level feature stability ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def fixed_subset_lofo_accuracy(
    fish_table: pd.DataFrame,
    features: list[str],
    genotype_col: str,
    group_b: str,
    random_seed: int,
) -> tuple[float, float]:
    x = fish_table[features].apply(
        pd.to_numeric, errors="coerce"
    )
    y = (fish_table[genotype_col].astype(str) == group_b).astype(int).to_numpy()

    loo = LeaveOneOut()
    predicted = np.zeros(len(y), dtype=int)
    probabilities = np.zeros(len(y), dtype=float)

    for fold_index, (train_index, test_index) in enumerate(
        loo.split(x), start=1
    ):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y[train_index]

        c_value = choose_regularisation_c(
            x_train,
            y_train,
            random_seed + fold_index,
        )

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        train_array = imputer.fit_transform(x_train)
        test_array = imputer.transform(x_test)

        train_array = scaler.fit_transform(train_array)
        test_array = scaler.transform(test_array)

        model = LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_seed + fold_index,
        )
        model.fit(train_array, y_train)

        predicted[test_index[0]] = int(
            model.predict(test_array)[0]
        )
        probabilities[test_index[0]] = float(
            model.predict_proba(test_array)[0, 1]
        )

    balanced_accuracy = float(
        balanced_accuracy_score(y, predicted)
    )
    auc = float(roc_auc_score(y, probabilities))

    return balanced_accuracy, auc


def backward_elimination(
    fish_table: pd.DataFrame,
    starting_features: list[str],
    genotype_col: str,
    group_b: str,
    random_seed: int,
) -> pd.DataFrame:
    current = list(starting_features)
    records = []
    step = 0

    while len(current) >= 2:
        baseline_accuracy, baseline_auc = fixed_subset_lofo_accuracy(
            fish_table,
            current,
            genotype_col,
            group_b,
            random_seed + step,
        )

        if len(current) == 2:
            records.append(
                {
                    "step": step,
                    "n_features_before": len(current),
                    "features_before": "|".join(current),
                    "removed_feature": "",
                    "balanced_accuracy_before": baseline_accuracy,
                    "roc_auc_before": baseline_auc,
                    "balanced_accuracy_after": np.nan,
                    "roc_auc_after": np.nan,
                    "delta_balanced_accuracy": np.nan,
                    "delta_roc_auc": np.nan,
                }
            )
            break

        candidates = []

        for feature in current:
            remaining = [
                item for item in current if item != feature
            ]
            accuracy, auc = fixed_subset_lofo_accuracy(
                fish_table,
                remaining,
                genotype_col,
                group_b,
                random_seed + step,
            )
            candidates.append(
                {
                    "removed_feature": feature,
                    "remaining": remaining,
                    "balanced_accuracy_after": accuracy,
                    "roc_auc_after": auc,
                    "delta_balanced_accuracy": (
                        accuracy - baseline_accuracy
                    ),
                    "delta_roc_auc": auc - baseline_auc,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["balanced_accuracy_after"],
                -item["roc_auc_after"],
                item["removed_feature"],
            )
        )
        best = candidates[0]

        records.append(
            {
                "step": step,
                "n_features_before": len(current),
                "features_before": "|".join(current),
                "removed_feature": best["removed_feature"],
                "balanced_accuracy_before": baseline_accuracy,
                "roc_auc_before": baseline_auc,
                "balanced_accuracy_after": best[
                    "balanced_accuracy_after"
                ],
                "roc_auc_after": best["roc_auc_after"],
                "delta_balanced_accuracy": best[
                    "delta_balanced_accuracy"
                ],
                "delta_roc_auc": best["delta_roc_auc"],
            }
        )

        current = best["remaining"]
        step += 1

    return pd.DataFrame(records)


# Permutation-test method: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
def permutation_test_nested_pipeline(
    fish_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_b: str,
    top_k: int,
    observed_accuracy: float,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    original_labels = fish_table[genotype_col].copy()
    scores = []

    for permutation_index in range(n_permutations):
        permuted_table = fish_table.copy()
        permuted_table[genotype_col] = rng.permutation(
            original_labels.to_numpy()
        )

        _, _, metrics = run_nested_lofo(
            permuted_table,
            fish_col,
            genotype_col,
            features,
            group_b,
            top_k,
            random_seed + permutation_index + 1000,
        )
        scores.append(metrics["balanced_accuracy"])

    scores_array = np.asarray(scores, dtype=float)
    p_value = float(
        (1 + np.sum(scores_array >= observed_accuracy))
        / (n_permutations + 1)
    )

    return pd.DataFrame(
        {
            "observed_balanced_accuracy": [observed_accuracy],
            "permutation_count": [n_permutations],
            "permutation_mean": [float(scores_array.mean())],
            "permutation_std": [
                float(scores_array.std(ddof=1))
                if len(scores_array) > 1
                else 0.0
            ],
            "permutation_p_value": [p_value],
        }
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw = pd.read_csv(input_path, low_memory=False)
    if raw.empty:
        raise ValueError("Input table is empty.")

    fish_col = detect_column(
        raw, args.fish_col, FISH_COLUMN_CANDIDATES, "fish"
    )
    genotype_col = detect_column(
        raw,
        args.genotype_col,
        GENOTYPE_COLUMN_CANDIDATES,
        "genotype",
    )
    track_col = detect_column(
        raw,
        args.track_col,
        TRACK_COLUMN_CANDIDATES,
        "track",
        required=False,
    )

    raw = raw.copy()
    raw[fish_col] = raw[fish_col].astype(str).str.strip()
    raw[genotype_col] = raw[genotype_col].map(normalise_genotype)
    raw = raw[
        raw[genotype_col].isin([args.group_a, args.group_b])
    ].copy()

    protected = {fish_col, genotype_col}
    if track_col:
        protected.add(track_col)

    numeric_candidates, detection_audit = identify_numeric_candidates(
        raw,
        protected,
        args.candidate_mode,
        args.include_regex,
        args.exclude_regex,
    )

    if numeric_candidates.shape[1] < 2:
        raise ValueError(
            "Fewer than two eligible numeric feature columns were found."
        )

    identity_columns = [fish_col, genotype_col]
    if track_col:
        identity_columns.append(track_col)

    fish_table = aggregate_to_fish_medians(
        raw[identity_columns],
        numeric_candidates,
        fish_col,
        genotype_col,
        track_col,
    )
    fish_table.to_csv(
        output_dir / "all_numeric_fish_medians_before_filtering.csv",
        index=False,
    )

    candidate_features = list(numeric_candidates.columns)

    filtered_features, filter_audit, correlated_pairs = (
        unsupervised_filter_features(
            fish_table,
            candidate_features,
            args.max_missing_fraction,
            args.min_unique_fish_values,
            args.correlation_threshold,
        )
    )

    complete_audit = pd.concat(
        [detection_audit, filter_audit],
        ignore_index=True,
        sort=False,
    )
    complete_audit.to_csv(
        output_dir / "feature_filter_audit.csv",
        index=False,
    )
    correlated_pairs.to_csv(
        output_dir / "correlated_feature_pairs_removed.csv",
        index=False,
    )

    selected_columns = [
        fish_col,
        genotype_col,
        "n_cell_track_rows",
    ]
    if "n_unique_tracks" in fish_table.columns:
        selected_columns.append("n_unique_tracks")

    filtered_fish_table = fish_table[
        selected_columns + filtered_features
    ].copy()
    filtered_fish_table.to_csv(
        output_dir / "fish_level_features_after_unsupervised_filtering.csv",
        index=False,
    )

    print(f"[INFO] Input rows: {len(raw)}")
    print(
        f"[INFO] Fish counts: "
        f"{fish_table[genotype_col].value_counts().to_dict()}"
    )
    print(
        f"[INFO] Numeric candidate features detected: "
        f"{len(candidate_features)}"
    )
    print(
        f"[INFO] Features after unsupervised filtering: "
        f"{len(filtered_features)}"
    )

    predictions, stability, nested_metrics = run_nested_lofo(
        filtered_fish_table,
        fish_col,
        genotype_col,
        filtered_features,
        args.group_b,
        args.inner_top_k,
        args.random_seed,
    )
    predictions.to_csv(
        output_dir / "nested_lofo_predictions.csv",
        index=False,
    )
    stability.to_csv(
        output_dir / "nested_lofo_feature_stability.csv",
        index=False,
    )
    pd.DataFrame([nested_metrics]).to_csv(
        output_dir / "nested_lofo_metrics.csv",
        index=False,
    )

    stable_features = select_stable_features(
        stability,
        args.selection_frequency_threshold,
        args.max_final_features,
        args.min_final_features,
    )
    pd.DataFrame(
        {
            "selected_order": np.arange(
                1, len(stable_features) + 1
            ),
            "feature": stable_features,
        }
    ).to_csv(
        output_dir / "stable_selected_features.csv",
        index=False,
    )

    plot_stability(
        stability,
        output_dir / "feature_stability_ranking.png",
    )

    pca_records = []

    all_coordinates, all_pca, all_metrics = (
        pca_coordinates_and_metrics(
            filtered_fish_table,
            filtered_features,
            genotype_col,
        )
    )
    plot_pca(
        filtered_fish_table.reset_index(drop=True),
        all_coordinates,
        all_pca,
        filtered_features,
        fish_col,
        genotype_col,
        f"{args.dataset_name}: PCA using all filtered numeric features",
        output_dir / "pca_all_filtered_features.png",
    )
    pca_records.append(
        {
            "subset": "all_filtered_features",
            "features": "|".join(filtered_features),
            **all_metrics,
        }
    )

    selected_coordinates, selected_pca, selected_metrics = (
        pca_coordinates_and_metrics(
            filtered_fish_table,
            stable_features,
            genotype_col,
        )
    )
    plot_pca(
        filtered_fish_table.reset_index(drop=True),
        selected_coordinates,
        selected_pca,
        stable_features,
        fish_col,
        genotype_col,
        f"{args.dataset_name}: PCA using stable selected features",
        output_dir / "pca_stable_selected_features.png",
    )
    pca_records.append(
        {
            "subset": "stable_selected_features",
            "features": "|".join(stable_features),
            **selected_metrics,
        }
    )

    pd.DataFrame(pca_records).to_csv(
        output_dir / "pca_subset_metrics.csv",
        index=False,
    )

    backward = backward_elimination(
        filtered_fish_table,
        stable_features,
        genotype_col,
        args.group_b,
        args.random_seed,
    )
    backward.to_csv(
        output_dir / "stable_subset_backward_elimination.csv",
        index=False,
    )

    permutation = permutation_test_nested_pipeline(
        filtered_fish_table,
        fish_col,
        genotype_col,
        filtered_features,
        args.group_b,
        args.inner_top_k,
        nested_metrics["balanced_accuracy"],
        args.permutations,
        args.random_seed,
    )
    permutation.to_csv(
        output_dir / "nested_pipeline_permutation_test.csv",
        index=False,
    )

    run_information = [
        f"input={args.input}",
        f"dataset_name={args.dataset_name}",
        f"fish_col={fish_col}",
        f"genotype_col={genotype_col}",
        f"track_col={track_col}",
        f"candidate_mode={args.candidate_mode}",
        f"numeric_candidates={len(candidate_features)}",
        f"after_unsupervised_filtering={len(filtered_features)}",
        f"inner_top_k={args.inner_top_k}",
        f"selection_frequency_threshold={args.selection_frequency_threshold}",
        f"stable_selected_count={len(stable_features)}",
        "stable_selected_features=" + ",".join(stable_features),
        f"nested_lofo_balanced_accuracy={nested_metrics['balanced_accuracy']}",
        f"nested_lofo_roc_auc={nested_metrics['roc_auc']}",
    ]
    (output_dir / "run_information.txt").write_text(
        "\n".join(run_information) + "\n",
        encoding="utf-8",
    )

    print("[INFO] Stable selected features:")
    for feature in stable_features:
        print(f"       - {feature}")

    print(
        f"[RESULT] Nested LOFO balanced accuracy: "
        f"{nested_metrics['balanced_accuracy']:.3f}"
    )
    print(
        f"[RESULT] Nested LOFO ROC AUC: "
        f"{nested_metrics['roc_auc']:.3f}"
    )
    print()
    print(f"[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
