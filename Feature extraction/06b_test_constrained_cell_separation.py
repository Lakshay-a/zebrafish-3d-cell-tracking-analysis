from __future__ import annotations

import argparse
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


CORE_FEATURES = [
    "net_displacement_3d_um",
    "directionality_ratio",
    "tortuosity",
    "mean_squared_displacement_3d_um2_per_min",
    "mean_speed_um_per_min",
    "median_speed_um_per_min",
    "mean_sphericity",
    "mean_elongation",
    "mean_volume_um3",
]

FISH_COLUMN_CANDIDATES = ["fish_id", "block_name", "block", "sample_id"]
GENOTYPE_COLUMN_CANDIDATES = ["genotype", "group", "condition", "class"]
TRACK_COLUMN_CANDIDATES = ["track_id", "global_track_id", "cell_track_id", "cell_id"]


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Expected comma-separated finite numbers.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cell/track-level genotype separation with whole-fish LOFO testing."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--track-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument(
        "--c-grid",
        type=parse_float_list,
        default=parse_float_list("0.001,0.003,0.01,0.03,0.1,0.3,1,3,10"),
    )
    parser.add_argument("--permutations", type=int, default=200)
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
            raise ValueError(f"Requested {role} column '{explicit}' was not found.")
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


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def choose_features(df: pd.DataFrame, requested: list[str]) -> list[str]:
    candidates = requested if requested else CORE_FEATURES
    missing = [feature for feature in candidates if feature not in df.columns]
    if missing and requested:
        raise ValueError(f"Requested cell features missing: {missing}")
    usable: list[str] = []
    for feature in candidates:
        if feature not in df.columns:
            continue
        values = safe_numeric(df[feature])
        if values.notna().sum() >= 10 and values.nunique(dropna=True) >= 2:
            df[feature] = values
            usable.append(feature)
    if len(usable) < 2:
        raise ValueError("Fewer than two usable cell-level predictors.")
    return usable


# Cliff's delta reference: https://revistas.javeriana.edu.co/index.php/revPsycho/article/view/643
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    differences = a[:, None] - b[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def filter_features(
    x_train: pd.DataFrame,
    max_missing_fraction: float,
    correlation_threshold: float,
) -> list[str]:
    survivors: list[str] = []
    for feature in x_train.columns:
        values = safe_numeric(x_train[feature])
        if values.isna().mean() > max_missing_fraction:
            continue
        if values.nunique(dropna=True) < 2:
            continue
        std = float(values.std(skipna=True))
        if np.isfinite(std) and not math.isclose(std, 0.0, abs_tol=1e-12):
            survivors.append(feature)
    if len(survivors) <= 1:
        return survivors

    corr = x_train[survivors].apply(safe_numeric).corr(method="spearman").abs()
    removed: set[str] = set()
    for i, feature_a in enumerate(survivors):
        if feature_a in removed:
            continue
        for feature_b in survivors[i + 1:]:
            if feature_b in removed:
                continue
            value = corr.loc[feature_a, feature_b]
            if np.isfinite(value) and value >= correlation_threshold:
                removed.add(feature_b)
    return [feature for feature in survivors if feature not in removed]


def rank_features(x_train: pd.DataFrame, y_train: np.ndarray, top_k: int) -> list[str]:
    scores: list[tuple[str, float]] = []
    for feature in x_train.columns:
        values = safe_numeric(x_train[feature]).to_numpy(float)
        scores.append((feature, abs(cliffs_delta(values[y_train == 0], values[y_train == 1]))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return [feature for feature, _ in scores[:min(top_k, len(scores))]]


def fit_model(x_train: pd.DataFrame, y_train: np.ndarray, c_value: float) -> tuple[LogisticRegression, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train = imputer.fit_transform(x_train)
    train = scaler.fit_transform(train)
    model = LogisticRegression(
        solver="liblinear",
        penalty="l1",
        C=float(c_value),
        class_weight="balanced",
        max_iter=5000,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(train, y_train)
    return model, imputer, scaler


def choose_c(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    fish_train: np.ndarray,
    c_grid: list[float],
) -> float:
    unique_fish = np.unique(fish_train)
    if len(unique_fish) < 4 or len(np.unique(y_train)) < 2:
        return 1.0
    n_splits = min(3, len(unique_fish))
    splitter = GroupKFold(n_splits=n_splits)
    results: list[tuple[float, float]] = []
    for c_value in c_grid:
        scores: list[float] = []
        for inner_train, inner_valid in splitter.split(x_train, y_train, groups=fish_train):
            if len(np.unique(y_train[inner_train])) < 2 or len(np.unique(y_train[inner_valid])) < 2:
                continue
            model, imputer, scaler = fit_model(x_train.iloc[inner_train], y_train[inner_train], c_value)
            valid = scaler.transform(imputer.transform(x_train.iloc[inner_valid]))
            predicted = model.predict(valid)
            scores.append(balanced_accuracy_score(y_train[inner_valid], predicted))
        if scores:
            results.append((float(np.mean(scores)), float(c_value)))
    if not results:
        return 1.0
    results.sort(key=lambda item: (-item[0], item[1]))
    return results[0][1]


# Grouped validation concept: https://scikit-learn.org/stable/modules/cross_validation.html
def run_cell_lofo(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_b: str,
    top_k: int,
    max_missing_fraction: float,
    correlation_threshold: float,
    c_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    x = df[features].apply(safe_numeric)
    y = (df[genotype_col].astype(str) == group_b).astype(int).to_numpy()
    fish = df[fish_col].astype(str).to_numpy()
    logo = LeaveOneGroupOut()

    selected_counts: Counter[str] = Counter()
    abs_coef: defaultdict[str, float] = defaultdict(float)
    signed_coef: defaultdict[str, float] = defaultdict(float)
    fish_records: list[dict[str, object]] = []
    cell_records: list[pd.DataFrame] = []

    for fold_number, (train_index, test_index) in enumerate(logo.split(x, y, groups=fish), start=1):
        x_train = x.iloc[train_index]
        y_train = y[train_index]
        fish_train = fish[train_index]
        filtered = filter_features(x_train, max_missing_fraction, correlation_threshold)
        if not filtered or len(np.unique(y_train)) < 2:
            probability = np.repeat(float(np.mean(y_train)), len(test_index))
            ranked: list[str] = []
            c_value = np.nan
            selected: list[str] = []
        else:
            ranked = rank_features(x_train[filtered], y_train, top_k)
            c_value = choose_c(x_train[ranked], y_train, fish_train, c_grid)
            model, imputer, scaler = fit_model(x_train[ranked], y_train, c_value)
            test = scaler.transform(imputer.transform(x.iloc[test_index][ranked]))
            probability = model.predict_proba(test)[:, 1]
            selected = []
            for feature, coefficient in zip(ranked, model.coef_[0]):
                coefficient = float(coefficient)
                if not math.isclose(coefficient, 0.0, abs_tol=1e-10):
                    selected.append(feature)
                    selected_counts[feature] += 1
                    abs_coef[feature] += abs(coefficient)
                    signed_coef[feature] += coefficient

        held = df.iloc[test_index].copy()
        held["cell_probability_group_b"] = probability
        held["cell_predicted_binary"] = (probability >= 0.5).astype(int)
        cell_records.append(
            held[[fish_col, genotype_col, "cell_probability_group_b", "cell_predicted_binary"]]
        )

        row = df.iloc[test_index[0]]
        fish_probability = float(np.mean(probability))
        fish_prediction = int(fish_probability >= 0.5)
        true_binary = int(row[genotype_col] == group_b)
        fish_records.append(
            {
                fish_col: row[fish_col],
                genotype_col: row[genotype_col],
                "true_binary": true_binary,
                "predicted_binary": fish_prediction,
                "probability_group_b": fish_probability,
                "correct": int(fish_prediction == true_binary),
                "n_heldout_cells": len(test_index),
                "n_ranked_features": len(ranked),
                "ranked_features": "|".join(ranked),
                "n_selected_features": len(selected),
                "selected_features": "|".join(selected),
                "inner_selected_C": c_value,
            }
        )

    predictions = pd.DataFrame(fish_records)
    cell_predictions = pd.concat(cell_records, ignore_index=True)
    n_folds = len(predictions)
    stability = []
    for feature in features:
        count = selected_counts[feature]
        stability.append(
            {
                "feature": feature,
                "outer_folds": n_folds,
                "nonzero_selection_frequency": count / n_folds,
                "mean_absolute_coefficient_when_selected": abs_coef[feature] / count if count else 0.0,
                "mean_signed_coefficient_when_selected": signed_coef[feature] / count if count else 0.0,
            }
        )
    stability_df = pd.DataFrame(stability).sort_values(
        ["nonzero_selection_frequency", "mean_absolute_coefficient_when_selected"],
        ascending=False,
    )

    true = predictions["true_binary"].to_numpy(int)
    predicted = predictions["predicted_binary"].to_numpy(int)
    probabilities = predictions["probability_group_b"].to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    metrics = {
        "n_fish": n_folds,
        "n_cells": len(df),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "roc_auc": float(roc_auc_score(true, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return predictions, cell_predictions, stability_df, metrics


# Permutation-test method: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
def permutation_test(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_b: str,
    observed_accuracy: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.permutations <= 0:
        return pd.DataFrame(
            {
                "observed_balanced_accuracy": [observed_accuracy],
                "n_permutations": [0],
                "permutation_mean": [np.nan],
                "permutation_std": [np.nan],
                "permutation_p_value": [np.nan],
            }
        )
    rng = np.random.default_rng(args.random_seed)
    fish_labels = df[[fish_col, genotype_col]].drop_duplicates().reset_index(drop=True)
    scores: list[float] = []
    for index in range(args.permutations):
        mapping = dict(
            zip(
                fish_labels[fish_col].astype(str),
                rng.permutation(fish_labels[genotype_col].to_numpy()),
            )
        )
        permuted = df.copy()
        permuted[genotype_col] = permuted[fish_col].astype(str).map(mapping)
        _, _, _, metrics = run_cell_lofo(
            permuted,
            fish_col,
            genotype_col,
            features,
            group_b,
            args.top_k,
            args.max_missing_fraction,
            args.correlation_threshold,
            args.c_grid,
        )
        scores.append(float(metrics["balanced_accuracy"]))
        print(f"[PERM] {index + 1}/{args.permutations}: {scores[-1]:.3f}")
    values = np.asarray(scores, dtype=float)
    p_value = float((1 + np.sum(values >= observed_accuracy)) / (len(values) + 1))
    return pd.DataFrame(
        {
            "observed_balanced_accuracy": [observed_accuracy],
            "n_permutations": [args.permutations],
            "permutation_mean": [float(values.mean())],
            "permutation_std": [float(values.std(ddof=1)) if len(values) > 1 else 0.0],
            "permutation_p_value": [p_value],
        }
    )


def plot_probability(predictions: pd.DataFrame, genotype_col: str, output_path: Path, group_b: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for x_pos, genotype in enumerate(sorted(predictions[genotype_col].unique())):
        sub = predictions[predictions[genotype_col] == genotype]
        values = sub["probability_group_b"].to_numpy(float)
        ax.boxplot([values], positions=[x_pos], widths=0.42, showfliers=False)
        ax.scatter(np.repeat(x_pos, len(values)), values, s=48, alpha=0.85)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(sorted(predictions[genotype_col].unique()))))
    ax.set_xticklabels(sorted(predictions[genotype_col].unique()))
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(f"Held-out fish mean cell probability of {group_b}")
    ax.set_title("Cell-level model aggregated to held-out fish")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    fish_col = detect_column(df, args.fish_col, FISH_COLUMN_CANDIDATES, "fish")
    genotype_col = detect_column(df, args.genotype_col, GENOTYPE_COLUMN_CANDIDATES, "genotype")
    track_col = detect_column(df, args.track_col, TRACK_COLUMN_CANDIDATES, "track", required=False)
    df = df.copy()
    df[fish_col] = df[fish_col].astype(str).str.strip()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    df = df[df[genotype_col].isin([args.group_a, args.group_b])].copy()
    if track_col:
        df = df.drop_duplicates([fish_col, track_col]).copy()
    df = df.sort_values([genotype_col, fish_col]).reset_index(drop=True)

    features = choose_features(df, args.feature)
    print(f"[INFO] {args.dataset_name}: {df[fish_col].nunique()} fish, {len(df)} cell/track rows, {len(features)} features")

    predictions, cell_predictions, stability, metrics = run_cell_lofo(
        df,
        fish_col,
        genotype_col,
        features,
        args.group_b,
        args.top_k,
        args.max_missing_fraction,
        args.correlation_threshold,
        args.c_grid,
    )
    predictions.to_csv(output_dir / "nested_lofo_predictions.csv", index=False)
    cell_predictions.to_csv(output_dir / "cell_level_predictions.csv", index=False)
    stability.to_csv(output_dir / "feature_selection_stability.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "nested_lofo_metrics.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(output_dir / "cell_features_used.csv", index=False)

    permutation = permutation_test(
        df,
        fish_col,
        genotype_col,
        features,
        args.group_b,
        float(metrics["balanced_accuracy"]),
        args,
    )
    permutation.to_csv(output_dir / "permutation_test.csv", index=False)
    plot_probability(predictions, genotype_col, output_dir / "predicted_probability_by_fish.png", args.group_b)

    (output_dir / "run_information.txt").write_text(
        "\n".join(
            [
                f"input={args.input}",
                f"dataset_name={args.dataset_name}",
                f"fish_count={df[fish_col].nunique()}",
                f"cell_track_count={len(df)}",
                "features=" + ",".join(features),
                f"balanced_accuracy={metrics['balanced_accuracy']}",
                f"roc_auc={metrics['roc_auc']}",
                f"permutation_p_value={permutation.iloc[0]['permutation_p_value']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[RESULT] Cell-level LOFO balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"[RESULT] Cell-level LOFO ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
