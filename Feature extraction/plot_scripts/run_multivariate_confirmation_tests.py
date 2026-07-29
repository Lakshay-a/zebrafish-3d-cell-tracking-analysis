#!/usr/bin/env python3
"""Confirm frozen-score and multivariate feature separation without retraining.

Two complementary tests are produced for the final model datasets:

1. Exact label permutations of the fixed frozen-model predictions in the MMP
   and liraglutide cohorts. Scores and predicted classes never change.
2. Euclidean PERMANOVA of the standardized selected-feature matrix using
   genotype, condition and genotype-by-condition terms. Marginal terms use
   Freedman-Lane residual permutations. A six-group dispersion permutation
   test accompanies PERMANOVA because unequal dispersion can affect it.

All rows are independent fish. No cell or track is treated as a replicate.

References
----------
PERMANOVA:
Anderson (2001), Austral Ecology 26, 32-46.
https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x

Multivariate dispersion:
Anderson (2006), Biometrics 62, 245-253.
https://doi.org/10.1111/j.1541-0420.2005.00440.x

ROC AUC and balanced accuracy definitions:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from generate_all_final_model_plots import MODELS


MODEL_MAP = {
    "musc_calibrated_svm_plus_injury": "musc_model_a_plus_injury",
    "macrophage_outside_boundary_calibrated_svm": "macrophage_outside_boundary_model_b",
    "macrophage_all_calibrated_svm_plus_injury": "macrophage_all_model_b_plus_injury",
    "musc_legacy_l1": "musc_model_a_no_injury_drop2",
}
CONDITIONS = ("untreated", "mmp", "liraglutide")
TERM_COLUMNS = {
    "genotype": [1],
    "condition": [2, 3],
    "genotype_x_condition": [4, 5],
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--selected-feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalise_genotype(value: object) -> str:
    text = str(value).upper()
    if "MUT" in text:
        return "MUT"
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", text) or "WILD TYPE" in text:
        return "WT"
    return str(value)


def design_row(genotype: str, condition: str) -> np.ndarray:
    genotype_code = -0.5 if genotype == "WT" else 0.5
    c1, c2 = {
        "untreated": (1.0, 0.0),
        "mmp": (0.0, 1.0),
        "liraglutide": (-1.0, -1.0),
    }[condition]
    return np.array(
        [1.0, genotype_code, c1, c2, genotype_code * c1, genotype_code * c2]
    )


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result.loc[valid.index] = restored
    return result


def exact_fixed_score_test(
    table: pd.DataFrame, model: str, cohort: str
) -> tuple[dict[str, object], pd.DataFrame]:
    """Enumerate every distinct genotype assignment at the observed group sizes."""
    frame = table.copy()
    frame["known_genotype"] = frame["known_genotype"].map(normalise_genotype)
    frame = frame[frame["known_genotype"].isin(["WT", "MUT"])].reset_index(drop=True)
    truth = frame["known_genotype"].eq("MUT").astype(int).to_numpy()
    scores = frame["probability_MUT"].to_numpy(float)
    predictions = frame["predicted_class"].map(normalise_genotype).eq("MUT").astype(int).to_numpy()
    observed_auc = float(roc_auc_score(truth, scores))
    observed_balanced_accuracy = float(balanced_accuracy_score(truth, predictions))
    n_mut = int(truth.sum())
    null_rows = []
    for mut_indexes in itertools.combinations(range(len(frame)), n_mut):
        permuted = np.zeros(len(frame), dtype=int)
        permuted[list(mut_indexes)] = 1
        null_rows.append(
            {
                "roc_auc": float(roc_auc_score(permuted, scores)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(permuted, predictions)
                ),
            }
        )
    null = pd.DataFrame(null_rows)
    directional_auc_p = float(
        (null["roc_auc"].ge(observed_auc).sum() + 1) / (len(null) + 1)
    )
    two_sided_auc_p = float(
        (
            (null["roc_auc"] - 0.5)
            .abs()
            .ge(abs(observed_auc - 0.5))
            .sum()
            + 1
        )
        / (len(null) + 1)
    )
    balanced_accuracy_p = float(
        (
            null["balanced_accuracy"].ge(observed_balanced_accuracy).sum()
            + 1
        )
        / (len(null) + 1)
    )
    result = {
        "model": model,
        "frozen_result_name": MODEL_MAP[model],
        "cohort": cohort,
        "n_fish": len(frame),
        "n_WT": int((truth == 0).sum()),
        "n_MUT": n_mut,
        "observed_roc_auc": observed_auc,
        "observed_balanced_accuracy": observed_balanced_accuracy,
        "exact_label_assignments": len(null),
        "roc_auc_directional_p": directional_auc_p,
        "roc_auc_two_sided_p": two_sided_auc_p,
        "balanced_accuracy_directional_p": balanced_accuracy_p,
    }
    null.insert(0, "assignment", np.arange(1, len(null) + 1))
    null.insert(0, "cohort", cohort)
    null.insert(0, "model", model)
    return result, null


def residual_sum_squares(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    beta = np.linalg.pinv(x) @ y
    fitted = x @ beta
    residuals = y - fitted
    return float(np.sum(residuals**2)), fitted, residuals


def pseudo_f(y: np.ndarray, full_x: np.ndarray, reduced_x: np.ndarray) -> tuple[float, float, int, int]:
    full_sse, _, _ = residual_sum_squares(y, full_x)
    reduced_sse, _, _ = residual_sum_squares(y, reduced_x)
    df_term = int(np.linalg.matrix_rank(full_x) - np.linalg.matrix_rank(reduced_x))
    df_resid = int(len(y) - np.linalg.matrix_rank(full_x))
    extra = max(0.0, reduced_sse - full_sse)
    statistic = (extra / df_term) / (full_sse / df_resid)
    partial_r2 = extra / (extra + full_sse) if extra + full_sse > 0 else np.nan
    return float(statistic), float(partial_r2), df_term, df_resid


def permanova_term(
    y: np.ndarray,
    full_x: np.ndarray,
    columns: list[int],
    permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float | int], np.ndarray]:
    reduced_x = np.delete(full_x, columns, axis=1)
    observed, partial_r2, df_term, df_resid = pseudo_f(y, full_x, reduced_x)
    _, reduced_fitted, reduced_residuals = residual_sum_squares(y, reduced_x)
    null = np.empty(permutations)
    for index in range(permutations):
        permuted_y = reduced_fitted + reduced_residuals[rng.permutation(len(y))]
        null[index] = pseudo_f(permuted_y, full_x, reduced_x)[0]
    p_value = float((np.count_nonzero(null >= observed) + 1) / (permutations + 1))
    return {
        "pseudo_f": observed,
        "partial_r_squared": partial_r2,
        "df_term": df_term,
        "df_resid": df_resid,
        "permutations": permutations,
        "p_value": p_value,
    }, null


def one_way_f(values: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    grand = values.mean()
    between = sum(
        np.count_nonzero(groups == group)
        * (values[groups == group].mean() - grand) ** 2
        for group in unique
    )
    within = sum(
        np.sum((values[groups == group] - values[groups == group].mean()) ** 2)
        for group in unique
    )
    return float(
        (between / (len(unique) - 1)) / (within / (len(values) - len(unique)))
    )


def dispersion_test(
    y: np.ndarray,
    groups: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float | int], np.ndarray]:
    def distances(labels: np.ndarray) -> np.ndarray:
        result = np.empty(len(y))
        for group in np.unique(labels):
            mask = labels == group
            centroid = y[mask].mean(axis=0)
            result[mask] = np.sqrt(np.sum((y[mask] - centroid) ** 2, axis=1))
        return result

    observed_distances = distances(groups)
    observed = one_way_f(observed_distances, groups)
    null = np.empty(permutations)
    for index in range(permutations):
        permuted_groups = groups[rng.permutation(len(groups))]
        null[index] = one_way_f(distances(permuted_groups), permuted_groups)
    p_value = float((np.count_nonzero(null >= observed) + 1) / (permutations + 1))
    return {
        "dispersion_f": observed,
        "df_between": len(np.unique(groups)) - 1,
        "df_within": len(groups) - len(np.unique(groups)),
        "permutations": permutations,
        "p_value": p_value,
    }, null


def selected_matrix(
    feature_dir: Path, selected_root: Path, model: str
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    frames = []
    for condition, relative in MODELS[model]["tables"].items():
        table = pd.read_csv(feature_dir / relative, low_memory=False)
        table = table.copy()
        table["condition"] = condition
        table["genotype"] = table["genotype"].map(normalise_genotype)
        frames.append(table)
    data = pd.concat(frames, ignore_index=True)
    selected_table = pd.read_csv(
        selected_root / model / "cv_union_selected_features.csv"
    )
    column = "feature" if "feature" in selected_table.columns else selected_table.columns[0]
    features = [
        feature
        for feature in selected_table[column].dropna().astype(str)
        if feature in data.columns
    ]
    matrix = (
        data[features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float, copy=True)
    )
    medians = np.nanmedian(matrix, axis=0)
    missing = np.where(np.isnan(matrix))
    matrix[missing] = medians[missing[1]]
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=1)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    matrix = (matrix - means) / scales
    metadata = data[["fish_id", "genotype", "condition"]].reset_index(drop=True)
    return matrix, metadata, features


def plot_score_nulls(
    results: pd.DataFrame, nulls: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, len(MODEL_MAP), figsize=(5 * len(MODEL_MAP), 8), squeeze=False)
    models = list(MODEL_MAP)
    for row, cohort in enumerate(("mmp", "liraglutide")):
        for column, model in enumerate(models):
            ax = axes[row, column]
            values = nulls[
                nulls["model"].eq(model) & nulls["cohort"].eq(cohort)
            ]["roc_auc"]
            observed = results[
                results["model"].eq(model) & results["cohort"].eq(cohort)
            ].iloc[0]
            ax.hist(values, bins=np.linspace(0, 1, 21), color="#9ecae1", edgecolor="white")
            ax.axvline(observed["observed_roc_auc"], color="#d62728", linewidth=2.2)
            ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{model}\n{cohort}: AUC={observed['observed_roc_auc']:.3f}")
            ax.set_xlabel("AUC under exact label assignment")
            ax.set_ylabel("Assignments")
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    feature_dir = args.feature_extraction_dir.resolve()
    selected_root = args.selected_feature_root.resolve()
    output = args.output_dir.resolve()
    score_output = output / "frozen_score_permutation"
    permanova_output = output / "permanova_selected_features"
    score_output.mkdir(parents=True, exist_ok=True)
    permanova_output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prediction_files = {
        "mmp": feature_dir
        / "frozen_mmp_results"
        / "combined_analysis"
        / "all_mmp_fish_predictions.csv",
        "liraglutide": feature_dir
        / "frozen_liraglutide_results"
        / "combined_analysis"
        / "all_liraglutide_fish_predictions.csv",
    }
    score_results = []
    score_nulls = []
    for cohort, path in prediction_files.items():
        predictions = pd.read_csv(path, low_memory=False)
        for model, frozen_name in MODEL_MAP.items():
            treated = predictions[
                predictions["dataset_name"].eq(frozen_name)
                & ~predictions["condition"].str.contains(
                    "untreated", case=False, na=False
                )
            ].copy()
            result, null = exact_fixed_score_test(treated, model, cohort)
            score_results.append(result)
            score_nulls.append(null)
    score_table = pd.DataFrame(score_results)
    for column in (
        "roc_auc_directional_p",
        "roc_auc_two_sided_p",
        "balanced_accuracy_directional_p",
    ):
        score_table[f"{column.removesuffix('_p')}_q"] = bh_adjust(score_table[column])
    null_table = pd.concat(score_nulls, ignore_index=True)
    score_table.to_csv(score_output / "fixed_frozen_score_permutation_results.csv", index=False)
    null_table.to_csv(score_output / "fixed_frozen_score_exact_null_distributions.csv", index=False)
    plot_score_nulls(
        score_table,
        null_table,
        score_output / "fixed_frozen_score_auc_permutation_distributions.png",
    )

    permanova_rows = []
    dispersion_rows = []
    null_rows = []
    for model in MODEL_MAP:
        matrix, metadata, features = selected_matrix(feature_dir, selected_root, model)
        full_x = np.vstack(
            [
                design_row(row.genotype, row.condition)
                for row in metadata.itertuples()
            ]
        )
        for term, columns in TERM_COLUMNS.items():
            result, null = permanova_term(
                matrix, full_x, columns, args.permutations, rng
            )
            permanova_rows.append(
                {
                    "model": model,
                    "term": term,
                    "n_fish": len(metadata),
                    "n_features": len(features),
                    **result,
                }
            )
            null_rows.extend(
                {
                    "model": model,
                    "test": f"permanova__{term}",
                    "permutation": index + 1,
                    "null_statistic": value,
                }
                for index, value in enumerate(null)
            )
        six_groups = (
            metadata["condition"].astype(str)
            + "__"
            + metadata["genotype"].astype(str)
        ).to_numpy()
        dispersion, null = dispersion_test(
            matrix, six_groups, args.permutations, rng
        )
        dispersion_rows.append(
            {
                "model": model,
                "grouping": "condition_x_genotype_six_groups",
                "n_fish": len(metadata),
                "n_features": len(features),
                **dispersion,
            }
        )
        null_rows.extend(
            {
                "model": model,
                "test": "multivariate_dispersion__six_groups",
                "permutation": index + 1,
                "null_statistic": value,
            }
            for index, value in enumerate(null)
        )
        pd.DataFrame({"feature": features}).to_csv(
            permanova_output / f"features_used__{model}.csv", index=False
        )

    permanova = pd.DataFrame(permanova_rows)
    permanova["q_value"] = bh_adjust(permanova["p_value"])
    dispersion = pd.DataFrame(dispersion_rows)
    dispersion["q_value"] = bh_adjust(dispersion["p_value"])
    permanova.to_csv(permanova_output / "permanova_results.csv", index=False)
    dispersion.to_csv(
        permanova_output / "multivariate_dispersion_results.csv", index=False
    )
    pd.DataFrame(null_rows).to_csv(
        permanova_output / "permutation_null_distributions.csv", index=False
    )

    pivot = permanova.pivot(index="model", columns="term", values="q_value").reindex(
        columns=["genotype", "condition", "genotype_x_condition"]
    )
    values = -np.log10(pivot.clip(lower=np.nextafter(0, 1)).to_numpy(float))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    image = ax.imshow(np.clip(values, 0, 4), cmap="viridis", vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(range(3), ["Genotype", "Condition", "Genotype × condition"])
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_title("Selected-feature PERMANOVA")
    fig.colorbar(image, ax=ax, label="−log10(BH q-value), capped at 4")
    fig.tight_layout()
    fig.savefig(permanova_output / "permanova_qvalue_heatmap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
