from __future__ import annotations

import argparse
import math
import re
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
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler


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
        description="Test WT/MUT separability using constrained fish features."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Maximum features passed to L1 logistic regression per fold.",
    )
    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--selection-frequency-threshold",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--max-stable-features",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested {role} column '{explicit}' was not found."
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


def get_candidate_features(df: pd.DataFrame) -> list[str]:
    candidates = [
        column
        for column in df.columns
        if str(column).startswith("fish_mean__")
        or str(column).startswith("fish_median__")
    ]

    usable = []
    for column in candidates:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) >= 2:
            df[column] = values.replace([np.inf, -np.inf], np.nan)
            usable.append(column)

    return usable


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


def training_fold_filter(
    x_train: pd.DataFrame,
    max_missing_fraction: float,
    correlation_threshold: float,
) -> list[str]:
    survivors = []

    for feature in x_train.columns:
        values = pd.to_numeric(x_train[feature], errors="coerce")

        if float(values.isna().mean()) > max_missing_fraction:
            continue
        if values.nunique(dropna=True) < 2:
            continue
        if math.isclose(
            float(values.std(skipna=True)),
            0.0,
            abs_tol=1e-12,
        ):
            continue

        survivors.append(feature)

    if len(survivors) <= 1:
        return survivors

    matrix = x_train[survivors].copy()
    correlation = matrix.corr(method="spearman").abs()

    removed = set()

    for i, feature_a in enumerate(survivors):
        if feature_a in removed:
            continue

        for feature_b in survivors[i + 1:]:
            if feature_b in removed:
                continue

            value = correlation.loc[feature_a, feature_b]
            if not np.isfinite(value):
                continue
            if value < correlation_threshold:
                continue

            missing_a = x_train[feature_a].isna().mean()
            missing_b = x_train[feature_b].isna().mean()

            if missing_a < missing_b:
                removed.add(feature_b)
            elif missing_b < missing_a:
                removed.add(feature_a)
                break
            else:
                variance_a = x_train[feature_a].var(skipna=True)
                variance_b = x_train[feature_b].var(skipna=True)

                if variance_a >= variance_b:
                    removed.add(feature_b)
                else:
                    removed.add(feature_a)
                    break

    return [
        feature for feature in survivors
        if feature not in removed
    ]


def rank_features(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    top_k: int,
) -> list[str]:
    scores = []

    for feature in x_train.columns:
        values = pd.to_numeric(
            x_train[feature], errors="coerce"
        ).to_numpy(dtype=float)

        group_a = values[y_train == 0]
        group_b = values[y_train == 1]
        score = abs(cliffs_delta(group_a, group_b))
        scores.append((feature, score))

    scores.sort(key=lambda item: (-item[1], item[0]))
    return [feature for feature, _ in scores[: min(top_k, len(scores))]]


def choose_c(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_seed: int,
) -> float:
    class_counts = np.bincount(y_train)
    positive_counts = class_counts[class_counts > 0]

    if len(positive_counts) < 2:
        return 1.0

    n_splits = min(3, int(positive_counts.min()))
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
            x_train,
            y_train,
        ):
            train_fold = x_train.iloc[train_index]
            validation_fold = x_train.iloc[validation_index]
            y_fold = y_train[train_index]
            y_validation = y_train[validation_index]

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            train_array = imputer.fit_transform(train_fold)
            validation_array = imputer.transform(validation_fold)

            train_array = scaler.fit_transform(train_array)
            validation_array = scaler.transform(validation_array)

            model = LogisticRegression(
                l1_ratio=1.0,
                solver="liblinear",
                C=c_value,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_seed,
            )
            model.fit(train_array, y_fold)

            prediction = model.predict(validation_array)
            fold_scores.append(
                balanced_accuracy_score(y_validation, prediction)
            )

        mean_scores.append(float(np.mean(fold_scores)))

    best_score = max(mean_scores)
    best_indices = [
        i for i, score in enumerate(mean_scores)
        if math.isclose(score, best_score, abs_tol=1e-12)
    ]

    # Prefer stronger regularisation when tied.
    return float(c_grid[min(best_indices)])


# Nested CV reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
def run_nested_lofo(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_b: str,
    correlation_threshold: float,
    max_missing_fraction: float,
    top_k: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    x = df[features].copy()
    y = (
        df[genotype_col].astype(str) == group_b
    ).astype(int).to_numpy()

    loo = LeaveOneOut()
    prediction_records = []

    selected_counts = Counter()
    ranked_counts = Counter()
    coefficient_sums = defaultdict(float)

    for fold_number, (train_index, test_index) in enumerate(
        loo.split(x),
        start=1,
    ):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        filtered = training_fold_filter(
            x_train,
            max_missing_fraction,
            correlation_threshold,
        )

        if not filtered:
            probability = float(np.mean(y_train))
            prediction = int(probability >= 0.5)
            nonzero_features = []
            c_value = np.nan
        else:
            ranked = rank_features(
                x_train[filtered],
                y_train,
                top_k,
            )
            for feature in ranked:
                ranked_counts[feature] += 1

            c_value = choose_c(
                x_train[ranked],
                y_train,
                random_seed + fold_number,
            )

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            train_array = imputer.fit_transform(x_train[ranked])
            test_array = imputer.transform(x_test[ranked])

            train_array = scaler.fit_transform(train_array)
            test_array = scaler.transform(test_array)

            model = LogisticRegression(
                l1_ratio=1.0,
                solver="liblinear",
                C=c_value,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_seed + fold_number,
            )
            model.fit(train_array, y_train)

            probability = float(
                model.predict_proba(test_array)[0, 1]
            )
            prediction = int(
                model.predict(test_array)[0]
            )

            nonzero_features = []
            for feature, coefficient in zip(
                ranked,
                model.coef_[0],
            ):
                if not math.isclose(
                    float(coefficient),
                    0.0,
                    abs_tol=1e-10,
                ):
                    nonzero_features.append(feature)
                    selected_counts[feature] += 1
                    coefficient_sums[feature] += abs(
                        float(coefficient)
                    )

        row = df.iloc[test_index[0]]
        prediction_records.append(
            {
                fish_col: row[fish_col],
                genotype_col: row[genotype_col],
                "true_binary": int(y_test[0]),
                "predicted_binary": prediction,
                "probability_group_b": probability,
                "correct": int(prediction == y_test[0]),
                "selected_features": "|".join(nonzero_features),
                "n_selected_features": len(nonzero_features),
                "inner_selected_C": c_value,
            }
        )

    predictions = pd.DataFrame(prediction_records)
    n_folds = len(predictions)

    stability_records = []
    for feature in features:
        selected_count = selected_counts[feature]
        stability_records.append(
            {
                "feature": feature,
                "outer_folds": n_folds,
                "top_k_frequency": ranked_counts[feature] / n_folds,
                "nonzero_selection_frequency": selected_count / n_folds,
                "mean_absolute_coefficient_when_selected": (
                    coefficient_sums[feature] / selected_count
                    if selected_count
                    else 0.0
                ),
            }
        )

    stability = pd.DataFrame(stability_records).sort_values(
        [
            "nonzero_selection_frequency",
            "mean_absolute_coefficient_when_selected",
        ],
        ascending=False,
    ).reset_index(drop=True)

    true = predictions["true_binary"].to_numpy()
    predicted = predictions["predicted_binary"].to_numpy()
    probabilities = predictions["probability_group_b"].to_numpy()

    balanced_accuracy = float(
        balanced_accuracy_score(true, predicted)
    )
    roc_auc = float(
        roc_auc_score(true, probabilities)
    )

    tn, fp, fn, tp = confusion_matrix(
        true,
        predicted,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "n_fish": len(predictions),
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return predictions, stability, metrics


def select_stable_features(
    stability: pd.DataFrame,
    threshold: float,
    max_features: int,
) -> list[str]:
    selected = stability.loc[
        stability["nonzero_selection_frequency"] >= threshold,
        "feature",
    ].head(max_features).tolist()

    if len(selected) < 2:
        selected = stability["feature"].head(2).tolist()

    return selected


def make_pca(
    df: pd.DataFrame,
    features: list[str],
    fish_col: str,
    genotype_col: str,
    title: str,
    output_path: Path,
) -> None:
    matrix = df[features].apply(
        pd.to_numeric,
        errors="coerce",
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

    fig, ax = plt.subplots(figsize=(8, 6.4))

    for genotype, group in df.groupby(genotype_col):
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
                str(df.loc[index, fish_col]),
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


# Permutation-test method: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
def permutation_test(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_b: str,
    observed_accuracy: float,
    correlation_threshold: float,
    max_missing_fraction: float,
    top_k: int,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    original_labels = df[genotype_col].to_numpy()
    scores = []

    for permutation_index in range(n_permutations):
        permuted = df.copy()
        permuted[genotype_col] = rng.permutation(original_labels)

        _, _, metrics = run_nested_lofo(
            permuted,
            fish_col,
            genotype_col,
            features,
            group_b,
            correlation_threshold,
            max_missing_fraction,
            top_k,
            random_seed + 1000 + permutation_index,
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
            "n_permutations": [n_permutations],
            "permutation_mean": [float(scores_array.mean())],
            "permutation_std": [
                float(scores_array.std(ddof=1))
            ],
            "permutation_p_value": [p_value],
        }
    )


def plot_stability(
    stability: pd.DataFrame,
    output_path: Path,
) -> None:
    table = stability.sort_values(
        "nonzero_selection_frequency"
    )

    fig_height = max(5.5, 0.4 * len(table) + 2)
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
        "Selection frequency across held-out-fish folds"
    )
    ax.set_title("Constrained feature-selection stability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    df[genotype_col] = df[genotype_col].map(
        normalise_genotype
    )
    df = df[
        df[genotype_col].isin([args.group_a, args.group_b])
    ].copy()
    df = df.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)

    features = get_candidate_features(df)
    if len(features) < 2:
        raise ValueError(
            "Fewer than two usable fish_mean__/fish_median__ predictors."
        )

    print(f"[INFO] Fish count: {len(df)}")
    print(
        f"[INFO] Genotypes: "
        f"{df[genotype_col].value_counts().to_dict()}"
    )
    print(f"[INFO] Constrained predictors detected: {len(features)}")

    predictions, stability, metrics = run_nested_lofo(
        df,
        fish_col,
        genotype_col,
        features,
        args.group_b,
        args.correlation_threshold,
        args.max_missing_fraction,
        args.top_k,
        args.random_seed,
    )

    predictions.to_csv(
        output_dir / "nested_lofo_predictions.csv",
        index=False,
    )
    stability.to_csv(
        output_dir / "feature_selection_stability.csv",
        index=False,
    )
    pd.DataFrame([metrics]).to_csv(
        output_dir / "nested_lofo_metrics.csv",
        index=False,
    )

    stable_features = select_stable_features(
        stability,
        args.selection_frequency_threshold,
        args.max_stable_features,
    )

    pd.DataFrame(
        {
            "selected_order": np.arange(
                1,
                len(stable_features) + 1,
            ),
            "feature": stable_features,
        }
    ).to_csv(
        output_dir / "stable_selected_features.csv",
        index=False,
    )

    make_pca(
        df,
        features,
        fish_col,
        genotype_col,
        f"{args.dataset_name}: all constrained fish features",
        output_dir / "pca_all_constrained_features.png",
    )

    make_pca(
        df,
        stable_features,
        fish_col,
        genotype_col,
        f"{args.dataset_name}: stable constrained features",
        output_dir / "pca_stable_constrained_features.png",
    )

    plot_stability(
        stability,
        output_dir / "feature_selection_stability.png",
    )

    permutation = permutation_test(
        df,
        fish_col,
        genotype_col,
        features,
        args.group_b,
        metrics["balanced_accuracy"],
        args.correlation_threshold,
        args.max_missing_fraction,
        args.top_k,
        args.permutations,
        args.random_seed,
    )
    permutation.to_csv(
        output_dir / "permutation_test.csv",
        index=False,
    )

    run_lines = [
        f"input={input_path}",
        f"dataset_name={args.dataset_name}",
        f"fish_count={len(df)}",
        f"candidate_feature_count={len(features)}",
        f"balanced_accuracy={metrics['balanced_accuracy']}",
        f"roc_auc={metrics['roc_auc']}",
        "stable_features=" + ",".join(stable_features),
    ]
    (output_dir / "run_information.txt").write_text(
        "\n".join(run_lines) + "\n",
        encoding="utf-8",
    )

    print("[INFO] Stable constrained features:")
    for feature in stable_features:
        print(f"       - {feature}")

    print(
        f"[RESULT] Nested LOFO balanced accuracy: "
        f"{metrics['balanced_accuracy']:.3f}"
    )
    print(
        f"[RESULT] Nested LOFO ROC AUC: "
        f"{metrics['roc_auc']:.3f}"
    )

    if not permutation.empty:
        p_value = float(
            permutation.iloc[0]["permutation_p_value"]
        )
        print(
            f"[RESULT] Permutation p-value: {p_value:.4f}"
        )

    print(f"[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
