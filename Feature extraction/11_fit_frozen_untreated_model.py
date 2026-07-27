#!/usr/bin/env python3
"""Fit one deployable WT-versus-MUT model after model evaluation is complete."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


logistic = load_module(
    "frozen_logistic_helpers",
    SCRIPT_DIR / "06_test_constrained_fish_separation_final.py",
)
extra = load_module(
    "frozen_extra_classifier_helpers",
    SCRIPT_DIR / "extra_classifier_common.py",
)


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of finite numbers."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and save one frozen untreated WT-versus-MUT model."
    )
    parser.add_argument("--input", required=True, help="Untreated fish-level CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--classifier",
        choices=["l1", "elasticnet", "calibrated_linear_svm"],
        required=True,
    )
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)
    parser.add_argument(
        "--c-grid",
        type=parse_float_list,
        default=parse_float_list(
            "0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100"
        ),
    )
    parser.add_argument(
        "--l1-ratio-grid",
        type=parse_float_list,
        default=parse_float_list("0.10,0.25,0.50,0.75,0.90"),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-results-dir",
        default=None,
        help=(
            "Optional completed nested-LOFO results directory. Its metrics and "
            "run information are copied into the frozen-model provenance."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_evaluation_provenance(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {}
    root = Path(path_text).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation results directory not found: {root}")

    provenance: dict[str, Any] = {"results_dir": str(root)}
    run_info = root / "run_information.txt"
    metrics = root / "nested_lofo_metrics.csv"
    permutation = root / "permutation_test.csv"

    if run_info.exists():
        provenance["run_information"] = run_info.read_text(encoding="utf-8")
    if metrics.exists():
        provenance["nested_lofo_metrics"] = (
            pd.read_csv(metrics).replace({np.nan: None}).to_dict("records")
        )
    if permutation.exists():
        provenance["permutation_test"] = (
            pd.read_csv(permutation).replace({np.nan: None}).to_dict("records")
        )
    return provenance


def extract_linear_parameters(model: Any) -> tuple[np.ndarray | None, float | None]:
    estimator = model
    if not hasattr(estimator, "coef_") and hasattr(
        estimator, "calibrated_classifiers_"
    ):
        calibrated = estimator.calibrated_classifiers_
        if calibrated:
            estimator = calibrated[0].estimator

    coefficients = None
    intercept = None
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_[0], dtype=float)
    if hasattr(estimator, "intercept_"):
        intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    return coefficients, intercept


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    data = pd.read_csv(input_path, low_memory=False)
    if data.empty:
        raise ValueError("Untreated fish-level input is empty.")

    fish_col = logistic.detect_column(
        data, args.fish_col, logistic.FISH_COLUMN_CANDIDATES, "fish"
    )
    genotype_col = logistic.detect_column(
        data,
        args.genotype_col,
        logistic.GENOTYPE_COLUMN_CANDIDATES,
        "genotype",
    )

    data = data.copy()
    data[fish_col] = data[fish_col].astype(str).str.strip()
    data[genotype_col] = data[genotype_col].map(logistic.normalise_genotype)
    data = data[
        data[genotype_col].isin([args.group_a, args.group_b])
    ].copy()
    data = data.sort_values([genotype_col, fish_col]).reset_index(drop=True)

    if data[fish_col].duplicated().any():
        duplicated = data.loc[data[fish_col].duplicated(), fish_col].unique()
        raise ValueError(
            "Expected one row per fish; duplicated IDs: "
            + ", ".join(map(str, duplicated[:20]))
        )

    class_counts = data[genotype_col].value_counts()
    if args.group_a not in class_counts or args.group_b not in class_counts:
        raise ValueError(
            f"Both {args.group_a} and {args.group_b} must be present: "
            f"{class_counts.to_dict()}"
        )

    candidate_features = logistic.get_candidate_features(data)
    if len(candidate_features) < 2:
        raise ValueError("Fewer than two usable fish-level predictors.")

    x_candidates = (
        data[candidate_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    y = (data[genotype_col] == args.group_b).astype(int).to_numpy()

    filtered_features = logistic.training_fold_filter(
        x_candidates,
        args.max_missing_fraction,
        args.correlation_threshold,
    )
    if not filtered_features:
        raise ValueError("No features survived missingness/variance/correlation filtering.")

    selected_features = logistic.rank_features(
        x_candidates[filtered_features],
        y,
        args.top_k,
    )
    if not selected_features:
        raise ValueError("No features survived full-data ranking.")

    x_selected = x_candidates[selected_features]

    if args.classifier in {"l1", "elasticnet"}:
        chosen_c, chosen_l1_ratio = logistic.choose_hyperparameters(
            x_train=x_selected,
            y_train=y,
            model_type=args.classifier,
            c_grid=args.c_grid,
            l1_ratio_grid=args.l1_ratio_grid,
            random_seed=args.random_seed,
            show_convergence_warnings=False,
        )
        model = logistic.build_model(
            args.classifier,
            chosen_c,
            chosen_l1_ratio,
            args.random_seed,
        )
        chosen_shrinkage = None
    else:
        chosen_c, chosen_shrinkage = extra.choose_hyperparameters(
            x_train=x_selected,
            y_train=y,
            classifier_kind="calibrated_linear_svm",
            c_grid=args.c_grid,
            shrinkage_grid=["auto"],
            random_seed=args.random_seed,
        )
        chosen_l1_ratio = None
        model = None

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(x_selected)
    x_scaled = scaler.fit_transform(x_imputed)

    if args.classifier == "calibrated_linear_svm":
        model = extra.fit_final_model(
            classifier_kind="calibrated_linear_svm",
            x_train=x_scaled,
            y_train=y,
            c_value=chosen_c,
            shrinkage=chosen_shrinkage,
            random_seed=args.random_seed,
        )
    else:
        model = logistic.fit_without_warning_noise(
            model,
            x_scaled,
            y,
            show_convergence_warnings=False,
        )

    probabilities = (
        model.predict_proba(x_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else 1.0 / (1.0 + np.exp(-model.decision_function(x_scaled)))
    )
    predictions = model.predict(x_scaled).astype(int)
    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(x_scaled), dtype=float)
    else:
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        decision_scores = np.log(clipped / (1 - clipped))

    coefficients, intercept = extract_linear_parameters(model)

    frozen_at = datetime.now(timezone.utc).isoformat()
    evaluation_provenance = read_evaluation_provenance(
        args.evaluation_results_dir
    )
    bundle = {
        "format_version": 1,
        "created_utc": frozen_at,
        "purpose": "Apply frozen untreated WT-vs-MUT signature to unseen treated fish",
        "dataset_name": args.dataset_name,
        "classifier": args.classifier,
        "class_mapping": {args.group_a: 0, args.group_b: 1},
        "fish_column": fish_col,
        "genotype_column": genotype_col,
        "candidate_features": candidate_features,
        "filtered_features": filtered_features,
        "selected_features": selected_features,
        "correlation_threshold": args.correlation_threshold,
        "max_missing_fraction": args.max_missing_fraction,
        "top_k": args.top_k,
        "chosen_C": float(chosen_c),
        "chosen_l1_ratio": (
            None if chosen_l1_ratio is None else float(chosen_l1_ratio)
        ),
        "chosen_shrinkage": chosen_shrinkage,
        "random_seed": args.random_seed,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "linear_coefficients_standardized_space": coefficients,
        "linear_intercept_standardized_space": intercept,
        "training_input": str(input_path),
        "training_input_sha256": sha256_file(input_path),
        "training_fish_ids": data[fish_col].astype(str).tolist(),
        "training_genotypes": data[genotype_col].astype(str).tolist(),
        "evaluation_provenance": evaluation_provenance,
        "important_note": (
            "Nested-LOFO results, not full-fit training predictions, are the "
            "unbiased estimate of untreated performance."
        ),
    }
    model_path = output_dir / "frozen_untreated_model.joblib"
    joblib.dump(bundle, model_path)

    pd.DataFrame(
        {
            "selected_order": np.arange(1, len(selected_features) + 1),
            "feature": selected_features,
            "imputer_median": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "coefficient_standardized_space": (
                coefficients
                if coefficients is not None
                else np.full(len(selected_features), np.nan)
            ),
        }
    ).to_csv(output_dir / "frozen_model_coefficients.csv", index=False)

    pd.DataFrame(
        {
            fish_col: data[fish_col],
            "known_genotype": data[genotype_col],
            "probability_MUT_resubstitution": probabilities,
            "signed_decision_score_resubstitution": decision_scores,
            "predicted_class_resubstitution": np.where(
                predictions == 1, args.group_b, args.group_a
            ),
        }
    ).to_csv(
        output_dir / "training_predictions_descriptive_only.csv",
        index=False,
    )

    summary = {
        "created_utc": frozen_at,
        "dataset_name": args.dataset_name,
        "classifier": args.classifier,
        "training_input": str(input_path),
        "n_training_fish": int(len(data)),
        f"n_{args.group_a}": int((data[genotype_col] == args.group_a).sum()),
        f"n_{args.group_b}": int((data[genotype_col] == args.group_b).sum()),
        "candidate_feature_count": len(candidate_features),
        "filtered_feature_count": len(filtered_features),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "chosen_C": float(chosen_c),
        "chosen_l1_ratio": chosen_l1_ratio,
        "model_file": str(model_path),
        "performance_warning": (
            "Do not report full-fit training accuracy. Use the linked "
            "nested-LOFO evaluation."
        ),
    }
    (output_dir / "frozen_model_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] Frozen model: {model_path}")
    print(f"[INFO] Training fish: {len(data)} ({class_counts.to_dict()})")
    print(f"[INFO] Selected features: {len(selected_features)}")
    for feature in selected_features:
        print(f"       - {feature}")
    print(f"[INFO] Chosen C: {chosen_c}")
    if chosen_l1_ratio is not None:
        print(f"[INFO] Chosen l1_ratio: {chosen_l1_ratio}")
    print(
        "[NOTE] Full-fit training predictions are descriptive only; "
        "use nested-LOFO metrics for performance."
    )


if __name__ == "__main__":
    main()
