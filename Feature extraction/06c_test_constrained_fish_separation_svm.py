from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        print("[WARN] tqdm is not installed; progress bars are disabled.")
        return iterable


HELPER_PATH = Path(__file__).with_name("06_test_constrained_fish_separation_final.py")
spec = importlib.util.spec_from_file_location("constrained_logistic_helpers", HELPER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not import helper functions from {HELPER_PATH}")
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)


FISH_COLUMN_CANDIDATES = helpers.FISH_COLUMN_CANDIDATES
GENOTYPE_COLUMN_CANDIDATES = helpers.GENOTYPE_COLUMN_CANDIDATES


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of finite numbers.")
    return values


def parse_gamma_grid(text: str) -> list[str | float]:
    values: list[str | float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item in {"scale", "auto"}:
            values.append(item)
        else:
            value = float(item)
            if not np.isfinite(value) or value <= 0:
                raise argparse.ArgumentTypeError("Gamma values must be positive, scale, or auto.")
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one gamma value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test WT/MUT separability using constrained fish-level features and SVM."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument(
        "--kernel",
        choices=["linear", "rbf"],
        required=True,
        help="SVM kernel to evaluate. Run this script once for linear and once for rbf.",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Maximum features passed to the SVM in each outer fold.",
    )
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)
    parser.add_argument("--selection-frequency-threshold", type=float, default=0.40)
    parser.add_argument("--max-stable-features", type=int, default=8)
    parser.add_argument(
        "--c-grid",
        type=parse_float_list,
        default=parse_float_list("0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100"),
        help="Comma-separated SVM C values tested inside inner CV.",
    )
    parser.add_argument(
        "--gamma-grid",
        type=parse_gamma_grid,
        default=parse_gamma_grid("scale,auto,0.001,0.003,0.01,0.03,0.1,0.3,1"),
        help="Comma-separated gamma values for RBF SVM. Ignored for linear SVM.",
    )
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def build_model(
    kernel: str,
    c_value: float,
    gamma: str | float,
    random_seed: int,
) -> SVC:
    return SVC(
        kernel=kernel,
        C=float(c_value),
        gamma=gamma if kernel == "rbf" else "scale",
        class_weight="balanced",
        random_state=random_seed,
    )


def choose_hyperparameters(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    kernel: str,
    c_grid: list[float],
    gamma_grid: list[str | float],
    random_seed: int,
) -> tuple[float, str | float]:
    class_counts = np.bincount(y_train)
    positive_counts = class_counts[class_counts > 0]
    default_gamma: str | float = "scale"

    if len(positive_counts) < 2:
        return 1.0, default_gamma

    n_splits = min(3, int(positive_counts.min()))
    if n_splits < 2:
        return 1.0, default_gamma

    gammas: list[str | float] = gamma_grid if kernel == "rbf" else [default_gamma]
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )
    results: list[dict[str, object]] = []

    for c_value in c_grid:
        for gamma in gammas:
            fold_scores: list[float] = []
            for train_index, validation_index in splitter.split(x_train, y_train):
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

                model = build_model(
                    kernel=kernel,
                    c_value=float(c_value),
                    gamma=gamma,
                    random_seed=random_seed,
                )
                model.fit(train_array, y_fold)
                prediction = model.predict(validation_array)
                fold_scores.append(
                    balanced_accuracy_score(y_validation, prediction)
                )

            gamma_sort = gamma if isinstance(gamma, float) else -1.0
            results.append(
                {
                    "C": float(c_value),
                    "gamma": gamma,
                    "gamma_sort": float(gamma_sort),
                    "score": float(np.mean(fold_scores)),
                }
            )

    # Highest score first; stronger regularisation (smaller C) wins ties.
    results.sort(
        key=lambda result: (
            -float(result["score"]),
            float(result["C"]),
            float(result["gamma_sort"]),
        )
    )
    best = results[0]
    return float(best["C"]), best["gamma"]  # type: ignore[return-value]


def run_nested_lofo(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_b: str,
    correlation_threshold: float,
    max_missing_fraction: float,
    top_k: int,
    kernel: str,
    c_grid: list[float],
    gamma_grid: list[str | float],
    random_seed: int,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    x = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    y = (df[genotype_col].astype(str) == group_b).astype(int).to_numpy()

    splits = list(LeaveOneOut().split(x))
    progress = tqdm(
        splits,
        desc=f"Nested LOFO (svm_{kernel})",
        total=len(splits),
        unit="fish",
        disable=not show_progress,
    )

    ranked_counts: Counter[str] = Counter()
    model_feature_counts: Counter[str] = Counter()
    abs_coefficient_sums: defaultdict[str, float] = defaultdict(float)
    signed_coefficient_sums: defaultdict[str, float] = defaultdict(float)
    prediction_records: list[dict[str, object]] = []

    for fold_number, (train_index, test_index) in enumerate(progress, start=1):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        filtered = helpers.training_fold_filter(
            x_train,
            max_missing_fraction,
            correlation_threshold,
        )
        ranked: list[str] = []
        model_features: list[str] = []
        c_value = np.nan
        gamma: str | float = np.nan

        if not filtered:
            decision_score = float(np.mean(y_train) - 0.5)
            score = float(1.0 / (1.0 + np.exp(-decision_score)))
            prediction = int(score >= 0.5)
        else:
            ranked = helpers.rank_features(x_train[filtered], y_train, top_k)
            for feature in ranked:
                ranked_counts[feature] += 1
                model_feature_counts[feature] += 1
            model_features = ranked

            c_value, gamma = choose_hyperparameters(
                x_train=x_train[ranked],
                y_train=y_train,
                kernel=kernel,
                c_grid=c_grid,
                gamma_grid=gamma_grid,
                random_seed=random_seed + fold_number,
            )

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            train_array = imputer.fit_transform(x_train[ranked])
            test_array = imputer.transform(x_test[ranked])
            train_array = scaler.fit_transform(train_array)
            test_array = scaler.transform(test_array)

            model = build_model(
                kernel=kernel,
                c_value=float(c_value),
                gamma=gamma,
                random_seed=random_seed + fold_number,
            )
            model.fit(train_array, y_train)
            decision_score = float(model.decision_function(test_array)[0])
            score = float(1.0 / (1.0 + np.exp(-decision_score)))
            prediction = int(model.predict(test_array)[0])

            if kernel == "linear" and hasattr(model, "coef_"):
                for feature, coefficient in zip(ranked, model.coef_[0]):
                    coefficient = float(coefficient)
                    if not math.isclose(coefficient, 0.0, abs_tol=1e-10):
                        abs_coefficient_sums[feature] += abs(coefficient)
                        signed_coefficient_sums[feature] += coefficient

        row = df.iloc[test_index[0]]
        prediction_records.append(
            {
                fish_col: row[fish_col],
                genotype_col: row[genotype_col],
                "true_binary": int(y_test[0]),
                "predicted_binary": prediction,
                "score_group_b": score,
                "decision_score_group_b": decision_score,
                "correct": int(prediction == y_test[0]),
                "model_type": f"svm_{kernel}",
                "kernel": kernel,
                "n_filtered_features": len(filtered),
                "n_ranked_features": len(ranked),
                "ranked_features": "|".join(ranked),
                "n_model_features": len(model_features),
                "model_features": "|".join(model_features),
                "inner_selected_C": c_value,
                "inner_selected_gamma": gamma,
            }
        )

        if show_progress and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                held_out=str(row[fish_col]),
                features=len(model_features),
            )

    predictions = pd.DataFrame(prediction_records)
    n_folds = len(predictions)
    stability_records: list[dict[str, object]] = []

    for feature in features:
        model_feature_count = model_feature_counts[feature]
        stability_records.append(
            {
                "feature": feature,
                "outer_folds": n_folds,
                "top_k_frequency": ranked_counts[feature] / n_folds,
                "model_feature_frequency": model_feature_count / n_folds,
                "mean_absolute_linear_coefficient_when_used": (
                    abs_coefficient_sums[feature] / model_feature_count
                    if model_feature_count and kernel == "linear"
                    else np.nan
                ),
                "mean_signed_linear_coefficient_when_used": (
                    signed_coefficient_sums[feature] / model_feature_count
                    if model_feature_count and kernel == "linear"
                    else np.nan
                ),
            }
        )

    stability = (
        pd.DataFrame(stability_records)
        .sort_values(
            [
                "model_feature_frequency",
                "mean_absolute_linear_coefficient_when_used",
            ],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )
    stability["nonzero_selection_frequency"] = stability["model_feature_frequency"]
    stability["mean_absolute_coefficient_when_selected"] = stability[
        "mean_absolute_linear_coefficient_when_used"
    ].fillna(0.0)

    true = predictions["true_binary"].to_numpy()
    predicted = predictions["predicted_binary"].to_numpy()
    decision_scores = predictions["decision_score_group_b"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()

    metrics: dict[str, float | int | str] = {
        "model_type": f"svm_{kernel}",
        "kernel": kernel,
        "n_fish": n_folds,
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "roc_auc": float(roc_auc_score(true, decision_scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return predictions, stability, metrics


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
    kernel: str,
    c_grid: list[float],
    gamma_grid: list[str | float],
    n_permutations: int,
    random_seed: int,
    show_progress: bool,
) -> pd.DataFrame:
    model_type = f"svm_{kernel}"
    if n_permutations <= 0:
        return pd.DataFrame(
            {
                "model_type": [model_type],
                "kernel": [kernel],
                "observed_balanced_accuracy": [observed_accuracy],
                "n_permutations": [0],
                "permutation_mean": [np.nan],
                "permutation_std": [np.nan],
                "permutation_p_value": [np.nan],
            }
        )

    rng = np.random.default_rng(random_seed)
    original_labels = df[genotype_col].to_numpy()
    scores: list[float] = []
    progress = tqdm(
        range(n_permutations),
        desc=f"Permutation test ({model_type})",
        total=n_permutations,
        unit="perm",
        disable=not show_progress,
    )

    for permutation_index in progress:
        permuted = df.copy()
        permuted[genotype_col] = rng.permutation(original_labels)
        _, _, metrics = run_nested_lofo(
            df=permuted,
            fish_col=fish_col,
            genotype_col=genotype_col,
            features=features,
            group_b=group_b,
            correlation_threshold=correlation_threshold,
            max_missing_fraction=max_missing_fraction,
            top_k=top_k,
            kernel=kernel,
            c_grid=c_grid,
            gamma_grid=gamma_grid,
            random_seed=random_seed + 1000 + permutation_index,
            show_progress=False,
        )
        score = float(metrics["balanced_accuracy"])
        scores.append(score)
        if show_progress and hasattr(progress, "set_postfix"):
            progress.set_postfix(latest=f"{score:.3f}")

    scores_array = np.asarray(scores, dtype=float)
    p_value = float(
        (1 + np.sum(scores_array >= observed_accuracy)) / (n_permutations + 1)
    )
    return pd.DataFrame(
        {
            "model_type": [model_type],
            "kernel": [kernel],
            "observed_balanced_accuracy": [observed_accuracy],
            "n_permutations": [n_permutations],
            "permutation_mean": [float(scores_array.mean())],
            "permutation_std": [
                float(scores_array.std(ddof=1)) if len(scores_array) > 1 else 0.0
            ],
            "permutation_p_value": [p_value],
        }
    )


def plot_svm_stability(stability: pd.DataFrame, output_path: Path, kernel: str) -> None:
    table = stability.sort_values("model_feature_frequency")
    fig_height = max(5.5, 0.4 * len(table) + 2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y = np.arange(len(table))
    ax.scatter(table["model_feature_frequency"], y, s=58)
    ax.set_yticks(y)
    ax.set_yticklabels(table["feature"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Feature-use frequency across held-out-fish folds")
    ax.set_title(f"SVM ({kernel}) feature-use stability")
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

    fish_col = helpers.detect_column(df, args.fish_col, FISH_COLUMN_CANDIDATES, "fish")
    genotype_col = helpers.detect_column(
        df,
        args.genotype_col,
        GENOTYPE_COLUMN_CANDIDATES,
        "genotype",
    )

    df = df.copy()
    df[fish_col] = df[fish_col].astype(str).str.strip()
    df[genotype_col] = df[genotype_col].map(helpers.normalise_genotype)
    df = df[df[genotype_col].isin([args.group_a, args.group_b])].copy()
    df = df.sort_values([genotype_col, fish_col]).reset_index(drop=True)

    features = helpers.get_candidate_features(df)
    if len(features) < 2:
        raise ValueError("Fewer than two usable fish_mean__/fish_median__ predictors.")

    print(f"[INFO] Fish count: {len(df)}")
    print(f"[INFO] Genotypes: {df[genotype_col].value_counts().to_dict()}")
    print(f"[INFO] Constrained predictors detected: {len(features)}")
    print(f"[INFO] Model: svm_{args.kernel}")
    print(f"[INFO] C grid: {args.c_grid}")
    if args.kernel == "rbf":
        print(f"[INFO] Gamma grid: {args.gamma_grid}")

    predictions, stability, metrics = run_nested_lofo(
        df=df,
        fish_col=fish_col,
        genotype_col=genotype_col,
        features=features,
        group_b=args.group_b,
        correlation_threshold=args.correlation_threshold,
        max_missing_fraction=args.max_missing_fraction,
        top_k=args.top_k,
        kernel=args.kernel,
        c_grid=args.c_grid,
        gamma_grid=args.gamma_grid,
        random_seed=args.random_seed,
        show_progress=not args.no_progress,
    )

    predictions.to_csv(output_dir / "nested_lofo_predictions.csv", index=False)
    stability.to_csv(output_dir / "feature_selection_stability.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "nested_lofo_metrics.csv", index=False)

    stable_features = helpers.select_stable_features(
        stability,
        args.selection_frequency_threshold,
        args.max_stable_features,
    )
    pd.DataFrame(
        {
            "selected_order": np.arange(1, len(stable_features) + 1),
            "feature": stable_features,
        }
    ).to_csv(output_dir / "stable_selected_features.csv", index=False)

    helpers.make_pca(
        df,
        features,
        fish_col,
        genotype_col,
        f"{args.dataset_name}: all constrained fish features",
        output_dir / "pca_all_constrained_features.png",
    )
    helpers.make_pca(
        df,
        stable_features,
        fish_col,
        genotype_col,
        f"{args.dataset_name}: stable SVM features",
        output_dir / "pca_stable_constrained_features.png",
    )
    plot_svm_stability(stability, output_dir / "feature_selection_stability.png", args.kernel)

    permutation = permutation_test(
        df=df,
        fish_col=fish_col,
        genotype_col=genotype_col,
        features=features,
        group_b=args.group_b,
        observed_accuracy=float(metrics["balanced_accuracy"]),
        correlation_threshold=args.correlation_threshold,
        max_missing_fraction=args.max_missing_fraction,
        top_k=args.top_k,
        kernel=args.kernel,
        c_grid=args.c_grid,
        gamma_grid=args.gamma_grid,
        n_permutations=args.permutations,
        random_seed=args.random_seed,
        show_progress=not args.no_progress,
    )
    permutation.to_csv(output_dir / "permutation_test.csv", index=False)

    run_lines = [
        f"input={input_path}",
        f"dataset_name={args.dataset_name}",
        f"model=svm_{args.kernel}",
        f"kernel={args.kernel}",
        f"fish_count={len(df)}",
        f"candidate_feature_count={len(features)}",
        "c_grid=" + ",".join(str(value) for value in args.c_grid),
        "gamma_grid=" + ",".join(str(value) for value in args.gamma_grid),
        f"balanced_accuracy={metrics['balanced_accuracy']}",
        f"roc_auc={metrics['roc_auc']}",
        "stable_features=" + ",".join(stable_features),
        f"permutation_p_value={permutation.iloc[0]['permutation_p_value']}",
    ]
    (output_dir / "run_information.txt").write_text(
        "\n".join(run_lines) + "\n",
        encoding="utf-8",
    )

    print("[INFO] Stable SVM-used constrained features:")
    for feature in stable_features:
        print(f"       - {feature}")
    print(
        "[RESULT] Nested LOFO balanced accuracy: "
        f"{float(metrics['balanced_accuracy']):.3f}"
    )
    print(f"[RESULT] Nested LOFO ROC AUC: {float(metrics['roc_auc']):.3f}")
    print(
        "[RESULT] Permutation p-value: "
        f"{float(permutation.iloc[0]['permutation_p_value']):.4f}"
    )
    print(f"[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
