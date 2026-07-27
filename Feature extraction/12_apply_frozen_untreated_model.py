#!/usr/bin/env python3
"""Apply one frozen untreated WT/MUT model to unseen treated fish."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FISH_CANDIDATES = ["fish_id", "block_name", "block", "sample_id"]
GENOTYPE_CANDIDATES = ["genotype", "group", "condition", "class"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a frozen untreated model without refitting."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--treated-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", default="MMP9_inhibited")
    parser.add_argument(
        "--exclude-fish",
        nargs="*",
        default=[],
        help="Fish IDs to exclude from the treated application only.",
    )
    return parser.parse_args()


def detect_column(df: pd.DataFrame, preferred: str, candidates: list[str]) -> str:
    if preferred in df.columns:
        return preferred
    lookup = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Could not detect a required column from {candidates}.")


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def score_table(
    df: pd.DataFrame,
    bundle: dict,
    condition: str,
    source_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = list(bundle["selected_features"])
    missing = [feature for feature in selected if feature not in df.columns]
    if missing:
        raise ValueError(
            "Treated table is missing frozen-model features: " + ", ".join(missing)
        )

    fish_col = detect_column(
        df, str(bundle["fish_column"]), FISH_CANDIDATES
    )
    genotype_col = detect_column(
        df, str(bundle["genotype_column"]), GENOTYPE_CANDIDATES
    )
    if df[fish_col].astype(str).duplicated().any():
        raise ValueError("Expected exactly one row per fish in treated input.")

    numeric = (
        df[selected]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    imputed = bundle["imputer"].transform(numeric)
    scaled = bundle["scaler"].transform(imputed)
    model = bundle["model"]

    probabilities = model.predict_proba(scaled)[:, 1]
    predicted_numeric = model.predict(scaled).astype(int)
    mapping = bundle["class_mapping"]
    inverse_mapping = {int(value): str(key) for key, value in mapping.items()}

    coefficients = bundle.get("linear_coefficients_standardized_space")
    intercept = bundle.get("linear_intercept_standardized_space")
    if coefficients is None or intercept is None:
        raise ValueError(
            "Frozen model lacks linear coefficients required for contributions."
        )
    coefficients = np.asarray(coefficients, dtype=float)
    raw_decision = scaled @ coefficients + float(intercept)
    contributions = scaled * coefficients[None, :]

    known = df[genotype_col].map(normalise_genotype)
    predicted = pd.Series(predicted_numeric).map(inverse_mapping)
    predictions = pd.DataFrame(
        {
            "dataset_name": bundle["dataset_name"],
            "condition": condition,
            "fish_id": df[fish_col].astype(str).str.strip(),
            "known_genotype": known,
            "probability_MUT": probabilities,
            "predicted_class": predicted,
            "correct": predicted.to_numpy() == known.to_numpy(),
            "signed_decision_score": raw_decision,
            "absolute_distance_from_boundary": np.abs(raw_decision),
            "source_table": str(source_path),
        }
    )

    contribution_frames = []
    for feature_index, feature in enumerate(selected):
        contribution_frames.append(
            pd.DataFrame(
                {
                    "dataset_name": bundle["dataset_name"],
                    "condition": condition,
                    "fish_id": predictions["fish_id"],
                    "known_genotype": known,
                    "feature": feature,
                    "raw_value": numeric[feature],
                    "imputed_value": imputed[:, feature_index],
                    "standardized_value": scaled[:, feature_index],
                    "coefficient": coefficients[feature_index],
                    "contribution_to_MUT_log_odds": contributions[:, feature_index],
                }
            )
        )
    contribution_table = pd.concat(contribution_frames, ignore_index=True)

    audit_rows = []
    for index, feature in enumerate(selected):
        values = numeric[feature]
        standardized = scaled[:, index]
        audit_rows.append(
            {
                "dataset_name": bundle["dataset_name"],
                "feature": feature,
                "present": True,
                "treated_missing_count": int(values.isna().sum()),
                "treated_missing_fraction": float(values.isna().mean()),
                "training_imputer_median": float(bundle["imputer"].statistics_[index]),
                "training_scaler_mean": float(bundle["scaler"].mean_[index]),
                "training_scaler_scale": float(bundle["scaler"].scale_[index]),
                "treated_standardized_min": float(np.min(standardized)),
                "treated_standardized_max": float(np.max(standardized)),
                "treated_outside_abs_z3_count": int((np.abs(standardized) > 3).sum()),
            }
        )
    return predictions, contribution_table, pd.DataFrame(audit_rows)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    treated_path = Path(args.treated_input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    treated = pd.read_csv(treated_path, low_memory=False)
    if args.exclude_fish:
        fish_col = detect_column(
            treated, str(bundle["fish_column"]), FISH_CANDIDATES
        )
        excluded = {str(value).strip() for value in args.exclude_fish}
        treated = treated.loc[
            ~treated[fish_col].astype(str).str.strip().isin(excluded)
        ].copy().reset_index(drop=True)
    predictions, contributions, audit = score_table(
        treated, bundle, args.condition, treated_path
    )

    predictions.to_csv(output_dir / "treated_fish_predictions.csv", index=False)
    contributions.to_csv(
        output_dir / "treated_fish_feature_contributions.csv", index=False
    )
    audit.to_csv(
        output_dir / "treated_feature_compatibility_audit.csv", index=False
    )

    training_path = Path(bundle["training_input"])
    if training_path.exists():
        untreated = pd.read_csv(training_path, low_memory=False)
        untreated_predictions, untreated_contributions, _ = score_table(
            untreated,
            bundle,
            "untreated_full_fit_descriptive",
            training_path,
        )
        untreated_predictions.to_csv(
            output_dir / "untreated_reference_predictions_descriptive_only.csv",
            index=False,
        )
        untreated_contributions.to_csv(
            output_dir / "untreated_reference_feature_contributions_descriptive_only.csv",
            index=False,
        )

    summary = {
        "dataset_name": bundle["dataset_name"],
        "condition": args.condition,
        "model_path": str(model_path),
        "treated_input": str(treated_path),
        "n_treated_fish": int(len(predictions)),
        "genotype_counts": predictions["known_genotype"].value_counts().to_dict(),
        "selected_features": list(bundle["selected_features"]),
        "important_note": (
            "The model, imputation, scaling, features, coefficients and decision "
            "boundary were frozen from untreated fish and were not refitted here. "
            "Full-fit untreated reference scores are descriptive, not an unbiased "
            "performance estimate; nested LOFO remains the performance estimate."
        ),
    }
    (output_dir / "application_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] {bundle['dataset_name']}: predicted {len(predictions)} treated fish")
    print(f"[SAVED] {output_dir / 'treated_fish_predictions.csv'}")


if __name__ == "__main__":
    main()
