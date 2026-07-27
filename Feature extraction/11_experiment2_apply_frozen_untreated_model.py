#!/usr/bin/env python3
"""Experiment 2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of finite numbers."
        )
    return values


def parse_labelled_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=CSV, for example MMP9=fish_table.csv"
        )
    label, path_text = text.split("=", 1)
    label = label.strip()
    path = Path(path_text.strip())
    if not label:
        raise argparse.ArgumentTypeError("Treatment label cannot be empty.")
    if not path_text.strip():
        raise argparse.ArgumentTypeError("CSV path cannot be empty.")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a frozen untreated WT-versus-MUT model and apply it to "
            "treated fish without retraining."
        )
    )
    parser.add_argument(
        "--untreated-input",
        action="append",
        required=True,
        help="Untreated fish-level CSV. Repeat to concatenate files.",
    )
    parser.add_argument(
        "--treated",
        action="append",
        required=True,
        type=parse_labelled_path,
        metavar="LABEL=CSV",
        help=(
            "Treated fish-level CSV with a treatment label. Repeat for each "
            "condition or file."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument(
        "--classifier-script",
        default="06_test_constrained_fish_separation_final.py",
        help="Path to the existing nested-LOFO classifier script.",
    )
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument(
        "--model",
        choices=["l1", "elasticnet"],
        default="l1",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)
    parser.add_argument(
        "--c-grid",
        type=parse_float_list,
        default=parse_float_list("0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100"),
    )
    parser.add_argument(
        "--l1-ratio-grid",
        type=parse_float_list,
        default=parse_float_list("0.10,0.25,0.50,0.75,0.90"),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--show-convergence-warnings", action="store_true")
    return parser.parse_args()


def load_classifier_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"Classifier script was not found: {script_path.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        "existing_fish_classifier",
        script_path.resolve(),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import classifier script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_csvs(paths: list[Path], source_label: str) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path, low_memory=False)
        if table.empty:
            raise ValueError(f"Input table is empty: {path}")
        table = table.copy()
        table["__source_csv"] = str(path)
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True, sort=False)
    print(
        f"[INFO] Loaded {len(combined)} fish rows for {source_label} "
        f"from {len(paths)} file(s)."
    )
    return combined


def prepare_genotype_table(
    df: pd.DataFrame,
    classifier,
    explicit_fish_col: str | None,
    explicit_genotype_col: str | None,
    group_a: str,
    group_b: str,
) -> tuple[pd.DataFrame, str, str]:
    fish_col = classifier.detect_column(
        df,
        explicit_fish_col,
        classifier.FISH_COLUMN_CANDIDATES,
        "fish",
    )
    genotype_col = classifier.detect_column(
        df,
        explicit_genotype_col,
        classifier.GENOTYPE_COLUMN_CANDIDATES,
        "genotype",
    )

    prepared = df.copy()
    prepared[fish_col] = prepared[fish_col].astype(str).str.strip()
    prepared[genotype_col] = prepared[genotype_col].map(
        classifier.normalise_genotype
    )
    prepared = prepared[
        prepared[genotype_col].isin([group_a, group_b])
    ].copy()

    if prepared.empty:
        raise ValueError(
            f"No rows remained after retaining genotypes {group_a!r} and "
            f"{group_b!r}."
        )

    return prepared.reset_index(drop=True), fish_col, genotype_col


def condition_metrics(
    predictions: pd.DataFrame,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for treatment, group in predictions.groupby("treatment", sort=True):
        true = group["true_binary"].to_numpy(int)
        predicted = group["predicted_binary"].to_numpy(int)
        probability = group["probability_MUT"].to_numpy(float)

        record: dict[str, object] = {
            "treatment": treatment,
            "n_fish": len(group),
            f"n_{group_a}": int(np.sum(true == 0)),
            f"n_{group_b}": int(np.sum(true == 1)),
            "balanced_accuracy": np.nan,
            "roc_auc": np.nan,
            "true_negative": np.nan,
            "false_positive": np.nan,
            "false_negative": np.nan,
            "true_positive": np.nan,
            "mean_probability_MUT": float(np.mean(probability)),
            "median_probability_MUT": float(np.median(probability)),
            "mean_absolute_distance_from_boundary": float(
                group["absolute_distance_from_boundary"].mean()
            ),
        }

        if len(np.unique(true)) == 2:
            tn, fp, fn, tp = confusion_matrix(
                true,
                predicted,
                labels=[0, 1],
            ).ravel()
            record.update(
                {
                    "balanced_accuracy": float(
                        balanced_accuracy_score(true, predicted)
                    ),
                    "roc_auc": float(roc_auc_score(true, probability)),
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )

        records.append(record)

    return pd.DataFrame(records)


def make_probability_plot(
    predictions: pd.DataFrame,
    output_path: Path,
    group_a: str,
    group_b: str,
    random_seed: int,
) -> None:
    rng = np.random.default_rng(random_seed)
    ordered_groups: list[tuple[str, str]] = []

    for treatment in predictions["treatment"].drop_duplicates():
        for genotype in [group_a, group_b]:
            if (
                (predictions["treatment"] == treatment)
                & (predictions["genotype"] == genotype)
            ).any():
                ordered_groups.append((str(treatment), genotype))

    fig_width = max(8.5, 1.25 * len(ordered_groups))
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))

    for position, (treatment, genotype) in enumerate(ordered_groups, start=1):
        subset = predictions[
            (predictions["treatment"] == treatment)
            & (predictions["genotype"] == genotype)
        ]
        values = subset["probability_MUT"].to_numpy(float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=70,
            alpha=0.9,
        )

        for x_value, (_, row) in zip(
            np.full(len(values), position) + jitter,
            subset.iterrows(),
        ):
            ax.annotate(
                str(row["fish_id"]),
                (x_value, float(row["probability_MUT"])),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.2,
        label="Decision boundary",
    )
    ax.set_xticks(range(1, len(ordered_groups) + 1))
    ax.set_xticklabels(
        [f"{treatment}\n{genotype}" for treatment, genotype in ordered_groups],
        rotation=25,
        ha="right",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(f"Frozen-model probability of {group_b}")
    ax.set_title("Untreated-trained model applied to treated fish")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = Path(args.classifier_script)
    classifier = load_classifier_module(classifier_path)

    untreated_paths = [Path(path) for path in args.untreated_input]
    untreated_raw = load_csvs(untreated_paths, "untreated")
    untreated, fish_col, genotype_col = prepare_genotype_table(
        untreated_raw,
        classifier,
        args.fish_col,
        args.genotype_col,
        args.group_a,
        args.group_b,
    )

    genotype_counts = untreated[genotype_col].value_counts()
    if args.group_a not in genotype_counts or args.group_b not in genotype_counts:
        raise ValueError(
            "The untreated training data must contain both WT and MUT fish."
        )
    if genotype_counts.min() < 2:
        raise ValueError(
            "At least two untreated fish per genotype are required to fit and "
            "tune the model."
        )

    candidate_features = classifier.get_candidate_features(untreated)
    if len(candidate_features) < 2:
        raise ValueError(
            "Fewer than two usable fish_mean__/fish_median__ features were "
            "found in the untreated table."
        )

    x_untreated = (
        untreated[candidate_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    y_untreated = (
        untreated[genotype_col].astype(str) == args.group_b
    ).astype(int).to_numpy()

    # Every data-dependent decision below uses untreated fish only.
    filtered_features = classifier.training_fold_filter(
        x_untreated,
        args.max_missing_fraction,
        args.correlation_threshold,
    )
    if not filtered_features:
        raise ValueError(
            "No untreated features survived missingness, variance and "
            "correlation filtering."
        )

    ranked_features = classifier.rank_features(
        x_untreated[filtered_features],
        y_untreated,
        args.top_k,
    )
    if not ranked_features:
        raise ValueError("No features were available for the final model.")

    selected_c, selected_l1_ratio = classifier.choose_hyperparameters(
        x_train=x_untreated[ranked_features],
        y_train=y_untreated,
        model_type=args.model,
        c_grid=args.c_grid,
        l1_ratio_grid=args.l1_ratio_grid,
        random_seed=args.random_seed,
        show_convergence_warnings=args.show_convergence_warnings,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    untreated_imputed = imputer.fit_transform(
        x_untreated[ranked_features]
    )
    untreated_scaled = scaler.fit_transform(untreated_imputed)

    model = classifier.build_model(
        model_type=args.model,
        c_value=float(selected_c),
        l1_ratio=selected_l1_ratio,
        random_seed=args.random_seed,
    )
    classifier.fit_without_warning_noise(
        model,
        untreated_scaled,
        y_untreated,
        args.show_convergence_warnings,
    )

    coefficients = model.coef_[0].astype(float)
    nonzero_mask = ~np.isclose(coefficients, 0.0, atol=1e-10)
    nonzero_features = [
        feature
        for feature, is_nonzero in zip(ranked_features, nonzero_mask)
        if is_nonzero
    ]

    coefficient_table = pd.DataFrame(
        {
            "feature": ranked_features,
            "coefficient_standardised": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "nonzero_selected": nonzero_mask.astype(int),
            "direction": np.where(
                coefficients > 0,
                f"toward_{args.group_b}",
                np.where(
                    coefficients < 0,
                    f"toward_{args.group_a}",
                    "zero",
                ),
            ),
            "training_median_used_for_imputation": imputer.statistics_,
            "training_scaler_mean": scaler.mean_,
            "training_scaler_scale": scaler.scale_,
        }
    ).sort_values(
        ["nonzero_selected", "absolute_coefficient"],
        ascending=False,
    )
    coefficient_table.to_csv(
        output_dir / "frozen_model_coefficients.csv",
        index=False,
    )

    treated_groups: dict[str, list[Path]] = {}
    for label, path in args.treated:
        treated_groups.setdefault(label, []).append(path)

    prediction_records: list[dict[str, object]] = []
    contribution_records: list[dict[str, object]] = []
    missing_feature_records: list[dict[str, object]] = []

    for treatment_label, paths in treated_groups.items():
        treated_raw = load_csvs(paths, treatment_label)
        treated, treated_fish_col, treated_genotype_col = prepare_genotype_table(
            treated_raw,
            classifier,
            args.fish_col,
            args.genotype_col,
            args.group_a,
            args.group_b,
        )

        missing_columns = [
            feature
            for feature in ranked_features
            if feature not in treated.columns
        ]
        for feature in missing_columns:
            treated[feature] = np.nan

        missing_feature_records.append(
            {
                "treatment": treatment_label,
                "n_fish": len(treated),
                "required_ranked_feature_count": len(ranked_features),
                "missing_feature_count": len(missing_columns),
                "missing_features": "|".join(missing_columns),
            }
        )

        treated_matrix = (
            treated[ranked_features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        treated_imputed = imputer.transform(treated_matrix)
        treated_scaled = scaler.transform(treated_imputed)

        probabilities = model.predict_proba(treated_scaled)[:, 1]
        predicted_binary = model.predict(treated_scaled).astype(int)
        decision_scores = model.decision_function(treated_scaled).astype(float)
        contributions = treated_scaled * coefficients[None, :]

        for row_position, (_, row) in enumerate(treated.iterrows()):
            true_binary = int(
                str(row[treated_genotype_col]) == args.group_b
            )
            fish_id = str(row[treated_fish_col])
            probability = float(probabilities[row_position])
            decision_score = float(decision_scores[row_position])
            predicted = int(predicted_binary[row_position])

            prediction_records.append(
                {
                    "dataset_name": args.dataset_name,
                    "treatment": treatment_label,
                    "fish_id": fish_id,
                    "genotype": str(row[treated_genotype_col]),
                    "true_binary": true_binary,
                    "predicted_binary": predicted,
                    "predicted_class": (
                        args.group_b if predicted == 1 else args.group_a
                    ),
                    "probability_MUT": probability,
                    "probability_WT": 1.0 - probability,
                    "signed_decision_score": decision_score,
                    "absolute_distance_from_boundary": abs(decision_score),
                    "probability_distance_from_0_5": abs(probability - 0.5),
                    "correct": int(predicted == true_binary),
                }
            )

            for feature_index, feature in enumerate(ranked_features):
                raw_value = pd.to_numeric(
                    pd.Series([row.get(feature, np.nan)]),
                    errors="coerce",
                ).iloc[0]
                contribution_records.append(
                    {
                        "dataset_name": args.dataset_name,
                        "treatment": treatment_label,
                        "fish_id": fish_id,
                        "genotype": str(row[treated_genotype_col]),
                        "feature": feature,
                        "raw_value": raw_value,
                        "imputed_value": float(
                            treated_imputed[row_position, feature_index]
                        ),
                        "scaled_value": float(
                            treated_scaled[row_position, feature_index]
                        ),
                        "coefficient_standardised": float(
                            coefficients[feature_index]
                        ),
                        "contribution_to_MUT_log_odds": float(
                            contributions[row_position, feature_index]
                        ),
                    }
                )

    predictions = pd.DataFrame(prediction_records)
    contributions = pd.DataFrame(contribution_records)
    missing_audit = pd.DataFrame(missing_feature_records)

    predictions = predictions.sort_values(
        ["treatment", "genotype", "fish_id"]
    ).reset_index(drop=True)
    predictions.to_csv(
        output_dir / "treated_fish_predictions.csv",
        index=False,
    )
    contributions.to_csv(
        output_dir / "treated_fish_feature_contributions.csv",
        index=False,
    )
    missing_audit.to_csv(
        output_dir / "treated_feature_compatibility_audit.csv",
        index=False,
    )

    metrics = condition_metrics(
        predictions,
        args.group_a,
        args.group_b,
    )
    metrics.to_csv(
        output_dir / "treated_condition_metrics.csv",
        index=False,
    )

    model_summary = pd.DataFrame(
        [
            {
                "dataset_name": args.dataset_name,
                "model_type": args.model,
                "untreated_fish_count": len(untreated),
                f"untreated_{args.group_a}_count": int(
                    np.sum(y_untreated == 0)
                ),
                f"untreated_{args.group_b}_count": int(
                    np.sum(y_untreated == 1)
                ),
                "candidate_feature_count": len(candidate_features),
                "post_filter_feature_count": len(filtered_features),
                "ranked_feature_count": len(ranked_features),
                "nonzero_selected_feature_count": len(nonzero_features),
                "selected_C": float(selected_c),
                "selected_l1_ratio": (
                    float(selected_l1_ratio)
                    if selected_l1_ratio is not None
                    else np.nan
                ),
                "intercept": float(model.intercept_[0]),
                "ranked_features": "|".join(ranked_features),
                "nonzero_selected_features": "|".join(nonzero_features),
            }
        ]
    )
    model_summary.to_csv(
        output_dir / "frozen_model_summary.csv",
        index=False,
    )

    bundle = {
        "dataset_name": args.dataset_name,
        "model_type": args.model,
        "group_a": args.group_a,
        "group_b": args.group_b,
        "fish_column_in_untreated_table": fish_col,
        "genotype_column_in_untreated_table": genotype_col,
        "candidate_features": candidate_features,
        "filtered_features": filtered_features,
        "ranked_features": ranked_features,
        "nonzero_selected_features": nonzero_features,
        "selected_C": float(selected_c),
        "selected_l1_ratio": selected_l1_ratio,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
    }
    joblib.dump(bundle, output_dir / "frozen_untreated_model.joblib")

    make_probability_plot(
        predictions,
        output_dir / "treated_fish_MUT_probability.png",
        args.group_a,
        args.group_b,
        args.random_seed,
    )

    run_info = {
        "untreated_inputs": [str(path) for path in untreated_paths],
        "treated_inputs": {
            label: [str(path) for path in paths]
            for label, paths in treated_groups.items()
        },
        "classifier_script": str(classifier_path),
        "dataset_name": args.dataset_name,
        "model_type": args.model,
        "group_a": args.group_a,
        "group_b": args.group_b,
        "correlation_threshold": args.correlation_threshold,
        "top_k": args.top_k,
        "max_missing_fraction": args.max_missing_fraction,
        "c_grid": args.c_grid,
        "l1_ratio_grid": args.l1_ratio_grid,
        "random_seed": args.random_seed,
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )

    print("[RESULT] Frozen untreated model")
    print(f"         Model: {args.model}")
    print(f"         Selected C: {selected_c}")
    if selected_l1_ratio is not None:
        print(f"         Selected l1_ratio: {selected_l1_ratio}")
    print(
        "         Nonzero selected features: "
        + (
            ", ".join(nonzero_features)
            if nonzero_features
            else "none"
        )
    )
    print(f"[RESULT] Scored {len(predictions)} treated fish.")
    print(f"[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
