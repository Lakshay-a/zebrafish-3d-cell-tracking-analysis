#!/usr/bin/env python3
"""Run fish-level genotype-by-condition ANOVA for the final model set.

Each feature is analysed with one row per fish and the fixed-effects model

    value ~ genotype * condition

The design uses sum contrasts, so Type III main effects describe averages over
the other factor. HC3 covariance makes the tests less sensitive to unequal
group variances. The script also reports classical partial eta-squared,
planned genotype and interaction contrasts, residual diagnostics, and
Benjamini-Hochberg q-values. It reads completed feature tables and CV feature
lists only; it never fits or changes a classifier.

Method references
-----------------
Two-way factorial ANOVA:
https://www.itl.nist.gov/div898/handbook/prc/section4/prc437.htm

Type III ANOVA and HC3 covariance conventions:
https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html

Median-centred Levene test and Shapiro-Wilk test:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

Benjamini-Hochberg false-discovery-rate control:
https://www.jstor.org/stable/2346101
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from generate_all_final_model_plots import MODELS


BEST_MODELS = (
    "musc_calibrated_svm_plus_injury",
    "macrophage_outside_boundary_calibrated_svm",
    "macrophage_all_calibrated_svm_plus_injury",
    "musc_legacy_l1",
)
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
    return parser.parse_args()


def normalise_genotype(value: object) -> str:
    text = str(value).upper()
    if "MUT" in text:
        return "MUT"
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", text) or "WILD TYPE" in text:
        return "WT"
    return str(value)


def design_row(genotype: str, condition: str) -> np.ndarray:
    """Return intercept, sum-coded main effects, and interaction columns."""
    genotype_code = -0.5 if genotype == "WT" else 0.5
    condition_code = {
        "untreated": (1.0, 0.0),
        "mmp": (0.0, 1.0),
        "liraglutide": (-1.0, -1.0),
    }[condition]
    c1, c2 = condition_code
    return np.array(
        [1.0, genotype_code, c1, c2, genotype_code * c1, genotype_code * c2]
    )


def bh_adjust(p_values: pd.Series) -> pd.Series:
    """Apply the Benjamini-Hochberg step-up correction, preserving missing data."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
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


def fit_ols_hc3(y: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Fit OLS and calculate the HC3 sandwich covariance matrix."""
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residuals = y - x @ beta
    leverage = np.clip(np.sum((x @ inverse) * x, axis=1), 0, 0.999999)
    scaled_squared = (residuals / (1.0 - leverage)) ** 2
    meat = x.T @ (x * scaled_squared[:, None])
    covariance = inverse @ meat @ inverse
    return {
        "beta": beta,
        "covariance": covariance,
        "residuals": residuals,
        "sse": float(residuals @ residuals),
        "df_resid": int(len(y) - np.linalg.matrix_rank(x)),
    }


def wald_term(
    fit: dict[str, np.ndarray | float | int],
    columns: list[int],
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    beta = np.asarray(fit["beta"])
    covariance = np.asarray(fit["covariance"])
    estimate = beta[columns]
    term_covariance = covariance[np.ix_(columns, columns)]
    df_term = len(columns)
    statistic = float(
        estimate.T @ np.linalg.pinv(term_covariance) @ estimate / df_term
    )
    p_value = float(stats.f.sf(statistic, df_term, int(fit["df_resid"])))

    # Dropping a term from the full design gives the classical Type III
    # extra-sum-of-squares effect used here only for partial eta-squared.
    reduced_x = np.delete(x, columns, axis=1)
    reduced_fit = fit_ols_hc3(y, reduced_x)
    extra_ss = max(0.0, float(reduced_fit["sse"]) - float(fit["sse"]))
    denominator = extra_ss + float(fit["sse"])
    partial_eta_squared = extra_ss / denominator if denominator > 0 else np.nan
    return {
        "df_term": df_term,
        "df_resid": int(fit["df_resid"]),
        "hc3_f": statistic,
        "p_value": p_value,
        "partial_eta_squared": partial_eta_squared,
    }


def contrast(
    fit: dict[str, np.ndarray | float | int],
    vector: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    beta = np.asarray(fit["beta"])
    covariance = np.asarray(fit["covariance"])
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    standard_error = np.sqrt(max(variance, 0))
    df = int(fit["df_resid"])
    t_statistic = estimate / standard_error if standard_error > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_statistic), df)) if np.isfinite(t_statistic) else np.nan
    critical = float(stats.t.ppf(0.975, df))
    return (
        estimate,
        standard_error,
        t_statistic,
        p_value,
        estimate - critical * standard_error,
        estimate + critical * standard_error,
    )


def common_features(tables: dict[str, pd.DataFrame]) -> list[str]:
    shared = None
    for table in tables.values():
        candidates = {
            column
            for column in table.columns
            if column.startswith(("fish_mean__", "fish_median__"))
            and pd.to_numeric(table[column], errors="coerce").notna().sum() >= 2
        }
        shared = candidates if shared is None else shared.intersection(candidates)
    return sorted(shared or set())


def label(feature: str) -> str:
    return feature.replace("fish_mean__", "Mean: ").replace(
        "fish_median__", "Median: "
    ).replace("_", " ")


def plot_qvalue_heatmap(table: pd.DataFrame, output: Path, title: str) -> None:
    if table.empty:
        return
    pivot = table.pivot(index="feature", columns="term", values="q_value")
    columns = ["genotype", "condition", "genotype_x_condition"]
    pivot = pivot.reindex(columns=columns)
    values = -np.log10(pivot.clip(lower=np.nextafter(0, 1)).to_numpy(float))
    values = np.clip(values, 0, 6)
    fig, ax = plt.subplots(
        figsize=(9, max(4.8, 0.33 * len(pivot) + 2.3))
    )
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=6, aspect="auto")
    ax.set_xticks(range(3), ["Genotype", "Condition", "Genotype × condition"])
    ax.set_yticks(range(len(pivot)), [label(item) for item in pivot.index], fontsize=7)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("−log10(BH q-value), capped at 6")
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    feature_dir = args.feature_extraction_dir.resolve()
    selected_root = args.selected_feature_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_anova: list[dict[str, object]] = []
    all_contrasts: list[dict[str, object]] = []
    all_groups: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []

    for model in BEST_MODELS:
        config = MODELS[model]
        tables: dict[str, pd.DataFrame] = {}
        for condition, relative in config["tables"].items():
            table = pd.read_csv(feature_dir / relative, low_memory=False)
            table = table.copy()
            table["condition"] = condition
            table["genotype"] = table["genotype"].map(normalise_genotype)
            tables[condition] = table

        selected_table = pd.read_csv(
            selected_root / model / "cv_union_selected_features.csv"
        )
        selected_column = (
            "feature"
            if "feature" in selected_table.columns
            else selected_table.columns[0]
        )
        selected = set(selected_table[selected_column].dropna().astype(str))
        model_rows = []
        model_contrasts = []

        for feature in common_features(tables):
            pieces = []
            for condition, table in tables.items():
                frame = table[["fish_id", "genotype", feature]].copy()
                frame["condition"] = condition
                frame["value"] = pd.to_numeric(frame[feature], errors="coerce")
                pieces.append(frame.dropna(subset=["value"]))
            data = pd.concat(pieces, ignore_index=True)
            data = data[data["genotype"].isin(["WT", "MUT"])].copy()
            x = np.vstack(
                [
                    design_row(row.genotype, row.condition)
                    for row in data.itertuples()
                ]
            )
            y = data["value"].to_numpy(float)
            fit = fit_ols_hc3(y, x)
            family = "cv_selected" if feature in selected else "nonselected_exploratory"

            for term, columns in TERM_COLUMNS.items():
                result = wald_term(fit, columns, x, y)
                model_rows.append(
                    {
                        "model": model,
                        "feature": feature,
                        "feature_family": family,
                        "term": term,
                        "n_fish": len(data),
                        **result,
                    }
                )

            groups = []
            for condition in CONDITIONS:
                for genotype in ("WT", "MUT"):
                    values = data.loc[
                        data["condition"].eq(condition)
                        & data["genotype"].eq(genotype),
                        "value",
                    ].to_numpy(float)
                    groups.append(values)
                    all_groups.append(
                        {
                            "model": model,
                            "feature": feature,
                            "feature_family": family,
                            "condition": condition,
                            "genotype": genotype,
                            "n_fish": len(values),
                            "mean": np.mean(values),
                            "sd": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                            "median": np.median(values),
                            "q25": np.percentile(values, 25),
                            "q75": np.percentile(values, 75),
                        }
                    )
            residuals = np.asarray(fit["residuals"])
            shapiro = stats.shapiro(residuals) if 3 <= len(residuals) <= 5000 else None
            levene = stats.levene(*groups, center="median")
            all_diagnostics.append(
                {
                    "model": model,
                    "feature": feature,
                    "feature_family": family,
                    "shapiro_w": shapiro.statistic if shapiro else np.nan,
                    "shapiro_p": shapiro.pvalue if shapiro else np.nan,
                    "levene_brown_forsythe_statistic": levene.statistic,
                    "levene_brown_forsythe_p": levene.pvalue,
                }
            )

            genotype_vectors = {
                condition: design_row("MUT", condition)
                - design_row("WT", condition)
                for condition in CONDITIONS
            }
            contrast_vectors = {
                "MUT_minus_WT__untreated": genotype_vectors["untreated"],
                "MUT_minus_WT__mmp": genotype_vectors["mmp"],
                "MUT_minus_WT__liraglutide": genotype_vectors["liraglutide"],
                "interaction__mmp_minus_untreated": (
                    genotype_vectors["mmp"] - genotype_vectors["untreated"]
                ),
                "interaction__liraglutide_minus_untreated": (
                    genotype_vectors["liraglutide"]
                    - genotype_vectors["untreated"]
                ),
            }
            for contrast_name, vector in contrast_vectors.items():
                estimate, se, t_value, p_value, lower, upper = contrast(fit, vector)
                model_contrasts.append(
                    {
                        "model": model,
                        "feature": feature,
                        "feature_family": family,
                        "contrast": contrast_name,
                        "estimate_raw_units": estimate,
                        "hc3_standard_error": se,
                        "t_value": t_value,
                        "df_resid": int(fit["df_resid"]),
                        "p_value": p_value,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                    }
                )

        anova = pd.DataFrame(model_rows)
        contrasts = pd.DataFrame(model_contrasts)
        for (_, family, term), indexes in anova.groupby(
            ["model", "feature_family", "term"]
        ).groups.items():
            anova.loc[indexes, "q_value"] = bh_adjust(
                anova.loc[indexes, "p_value"]
            )
        for (_, family, contrast_name), indexes in contrasts.groupby(
            ["model", "feature_family", "contrast"]
        ).groups.items():
            contrasts.loc[indexes, "q_value"] = bh_adjust(
                contrasts.loc[indexes, "p_value"]
            )

        model_output = output / model
        model_output.mkdir(parents=True, exist_ok=True)
        anova.to_csv(model_output / "two_way_anova_results.csv", index=False)
        contrasts.to_csv(model_output / "planned_contrasts.csv", index=False)
        for family in ("cv_selected", "nonselected_exploratory"):
            subset = anova[anova["feature_family"].eq(family)]
            plot_qvalue_heatmap(
                subset,
                model_output / f"anova_qvalue_heatmap__{family}.png",
                f"{model}: two-way ANOVA ({family.replace('_', ' ')})",
            )
        all_anova.extend(anova.to_dict("records"))
        all_contrasts.extend(contrasts.to_dict("records"))

    pd.DataFrame(all_anova).to_csv(output / "all_models_two_way_anova.csv", index=False)
    pd.DataFrame(all_contrasts).to_csv(
        output / "all_models_planned_contrasts.csv", index=False
    )
    pd.DataFrame(all_groups).to_csv(output / "all_models_group_summaries.csv", index=False)
    pd.DataFrame(all_diagnostics).to_csv(
        output / "all_models_anova_diagnostics.csv", index=False
    )


if __name__ == "__main__":
    main()
