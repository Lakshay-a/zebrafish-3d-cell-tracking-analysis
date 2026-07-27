#!/usr/bin/env python3
"""Summarise and plot frozen untreated-model scores for a treated cohort."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--treated-condition", default="MMP9_inhibited")
    parser.add_argument("--treated-label", default="MMP9 inhibited")
    parser.add_argument("--output-prefix", default="mmp")
    return parser.parse_args()


# Hedges' g correction: https://ideas.repec.org/a/sae/jedbes/v6y1981i2p107-128.html
def hedges_g(wt: np.ndarray, mut: np.ndarray) -> float:
    wt = wt[np.isfinite(wt)]
    mut = mut[np.isfinite(mut)]
    if len(wt) < 2 or len(mut) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(wt) - 1) * np.var(wt, ddof=1) + (len(mut) - 1) * np.var(mut, ddof=1))
        / (len(wt) + len(mut) - 2)
    )
    if pooled == 0:
        return np.nan
    d = (np.mean(mut) - np.mean(wt)) / pooled
    correction = 1 - 3 / (4 * (len(wt) + len(mut)) - 9)
    return float(d * correction)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, condition), group in scores.groupby(["dataset_name", "condition"]):
        wt = group.loc[group.known_genotype == "WT", "probability_MUT"].to_numpy(float)
        mut = group.loc[group.known_genotype == "MUT", "probability_MUT"].to_numpy(float)
        wt_dec = group.loc[group.known_genotype == "WT", "signed_decision_score"].to_numpy(float)
        mut_dec = group.loc[group.known_genotype == "MUT", "signed_decision_score"].to_numpy(float)
        rows.append(
            {
                "dataset_name": dataset,
                "condition": condition,
                "n_WT": len(wt),
                "n_MUT": len(mut),
                "mean_probability_WT": np.mean(wt),
                "mean_probability_MUT": np.mean(mut),
                "probability_separation_MUT_minus_WT": np.mean(mut) - np.mean(wt),
                "decision_separation_MUT_minus_WT": np.mean(mut_dec) - np.mean(wt_dec),
                "hedges_g_probability": hedges_g(wt, mut),
                "mean_boundary_distance": group.absolute_distance_from_boundary.mean(),
                "known_genotype_accuracy": group.correct.mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_scores(
    scores: pd.DataFrame,
    value: str,
    ylabel: str,
    path: Path,
    datasets: list[str],
    figure_title: str,
    treated_condition: str,
    treated_label: str,
) -> None:
    display_names = {
        "macrophage_all_model_b_plus_injury": "Macrophage: all cells + injury",
        "macrophage_outside_boundary_model_b": "Macrophage: outside injury",
        "musc_model_a_plus_injury": "MUSC: SVM + injury",
        "musc_model_a_no_injury_all_fish": "MUSC: L1, no injury (all fish)",
        "musc_model_a_no_injury_drop2": "MUSC: L1, no injury (drop 2)",
    }
    datasets = [name for name in datasets if name in set(scores.dataset_name)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets) + 2.5, 5), squeeze=False)
    rng = np.random.default_rng(42)
    for ax, dataset in zip(axes[0], datasets):
        table = scores[scores.dataset_name == dataset]
        groups = [
            ("untreated_full_fit_descriptive", "WT", "Untreated\nWT"),
            ("untreated_full_fit_descriptive", "MUT", "Untreated\nMUT"),
            (treated_condition, "WT", f"{treated_label}\nWT"),
            (treated_condition, "MUT", f"{treated_label}\nMUT"),
        ]
        for x, (condition, genotype, label) in enumerate(groups):
            values = table.loc[
                (table.known_genotype == genotype) & (table.condition == condition),
                value,
            ].to_numpy(float)
            jitter = rng.uniform(-0.08, 0.08, len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, s=55)
            if len(values):
                ax.hlines(np.mean(values), x - 0.2, x + 0.2, linewidth=3)
        ax.set_xticks(range(len(groups)), [item[2] for item in groups], rotation=20)
        ax.set_title(display_names.get(dataset, dataset.replace("_", " ")))
        ax.set_ylabel(ylabel)
        if value in {"probability_MUT", "probability_WT"}:
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
            ax.set_ylim(-0.03, 1.03)
        else:
            ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.2)
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7,
               label="Each point = one fish"),
        Line2D([0], [0], color="black", linewidth=3, label="Horizontal bar = group mean"),
    ]
    if value in {"probability_MUT", "probability_WT"}:
        legend_items.append(
            Line2D([0], [0], color="grey", linestyle="--", label="0.5 decision threshold")
        )
    else:
        legend_items.append(
            Line2D([0], [0], color="grey", linestyle="--", label="0 decision boundary")
        )
    fig.suptitle(figure_title, fontsize=14)
    fig.legend(
        handles=legend_items, loc="center left", ncol=1,
        bbox_to_anchor=(0.86, 0.5), title="How to read this figure",
    )
    fig.tight_layout(rect=(0, 0, 0.85, 0.90))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_contributions(
    contributions: pd.DataFrame,
    output_dir: Path,
    treated_condition: str,
) -> None:
    treated = contributions[contributions.condition == treated_condition]
    for dataset, table in treated.groupby("dataset_name"):
        pivot = table.pivot(
            index="feature", columns="fish_id", values="contribution_to_MUT_log_odds"
        )
        order = (
            table[["fish_id", "known_genotype"]]
            .drop_duplicates()
            .sort_values(["known_genotype", "fish_id"])
            .fish_id
        )
        pivot = pivot.reindex(columns=order)
        limit = np.nanmax(np.abs(pivot.to_numpy()))
        limit = 1.0 if not np.isfinite(limit) or limit == 0 else limit
        fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(pivot.columns)), max(5, 0.55 * len(pivot.index))))
        image = ax.imshow(pivot, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=55, ha="right")
        ax.set_yticks(range(len(pivot.index)), [x.replace("fish_mean__", "mean: ").replace("fish_median__", "median: ") for x in pivot.index])
        ax.set_title(f"{dataset}: contribution to MUT decision score")
        fig.colorbar(image, ax=ax, label="standardized value × coefficient")
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_feature_contributions.png", dpi=220)
        plt.close(fig)


def plot_feature_shifts(
    shift_table: pd.DataFrame,
    value_column: str,
    title_suffix: str,
    xlabel: str,
    output_path: Path,
) -> None:
    datasets = list(shift_table["dataset_name"].drop_duplicates())
    ncols = min(3, len(datasets))
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5.5 * nrows),
        squeeze=False,
    )
    for ax, dataset in zip(axes.ravel(), datasets):
        table = shift_table[shift_table.dataset_name == dataset].copy()
        pivot = table.pivot(
            index="feature",
            columns="known_genotype",
            values=value_column,
        ).fillna(0)
        order = pivot.abs().max(axis=1).sort_values().index
        pivot = pivot.loc[order]
        y = np.arange(len(pivot))
        width = 0.36
        ax.barh(y - width / 2, pivot.get("WT", 0), height=width, label="WT")
        ax.barh(y + width / 2, pivot.get("MUT", 0), height=width, label="MUT")
        labels = [
            feature.replace("fish_mean__", "mean: ")
            .replace("fish_median__", "median: ")
            .replace("injury_", "injury: ")
            for feature in pivot.index
        ]
        ax.set_yticks(y, labels)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{dataset.replace('_', ' ')}\n{title_suffix}")
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_contribution_gap_arrows(
    contribution_means: pd.DataFrame,
    datasets: list[str],
    output_path: Path,
    figure_title: str,
    treated_condition: str,
    treated_label: str,
) -> None:
    """Show how each feature's MUT-minus-WT contribution gap changes."""
    labels = {
        "macrophage_all_model_b_plus_injury": "Macrophage: all cells + injury",
        "macrophage_outside_boundary_model_b": "Macrophage: outside injury",
        "musc_model_a_plus_injury": "MUSC: SVM + injury",
        "musc_model_a_no_injury_all_fish": "MUSC: L1, no injury (all fish)",
        "musc_model_a_no_injury_drop2": "MUSC: L1, no injury (drop 2)",
    }
    available = [name for name in datasets if name in set(contribution_means.dataset_name)]
    fig, axes = plt.subplots(
        1, len(available), figsize=(7.2 * len(available) + 2.8, 7), squeeze=False
    )
    for ax, dataset in zip(axes[0], available):
        table = contribution_means[contribution_means.dataset_name == dataset]
        pivot = table.pivot_table(
            index="feature",
            columns=["condition", "known_genotype"],
            values="contribution_to_MUT_log_odds",
        )
        untreated = (
            pivot[("untreated_full_fit_descriptive", "MUT")]
            - pivot[("untreated_full_fit_descriptive", "WT")]
        )
        treated = (
            pivot[(treated_condition, "MUT")]
            - pivot[(treated_condition, "WT")]
        )
        values = pd.DataFrame({"Untreated": untreated, "Treated": treated}).fillna(0)
        values["max_magnitude"] = values[["Untreated", "Treated"]].abs().max(axis=1)
        values = values.sort_values("max_magnitude")
        y = np.arange(len(values))
        for yi, (_, row) in zip(y, values.iterrows()):
            start, end = row["Untreated"], row["Treated"]
            color = "#2a9d8f" if abs(end) > abs(start) else "#d95f59"
            ax.annotate(
                "",
                xy=(end, yi),
                xytext=(start, yi),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2),
            )
            ax.scatter(start, yi, s=44, facecolors="white", edgecolors="#333333", zorder=3)
            ax.scatter(end, yi, s=48, color=color, zorder=3)
        feature_labels = [
            feature.replace("fish_mean__", "mean: ")
            .replace("fish_median__", "median: ")
            .replace("injury_", "injury: ")
            for feature in values.index
        ]
        ax.set_yticks(y, feature_labels)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Feature contribution gap (MUT − WT)")
        ax.set_title(labels.get(dataset, dataset.replace("_", " ")))
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(figure_title, fontsize=14)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor="white",
                   markeredgecolor="#333333", linestyle="none", label="Untreated MUT−WT gap"),
            Line2D([0], [0], marker="o", color="#d95f59", linestyle="none",
                   label=f"{treated_label} gap; separation weaker"),
            Line2D([0], [0], marker="o", color="#2a9d8f", linestyle="none",
                   label=f"{treated_label} gap; separation stronger"),
            Line2D([0], [0], color="black", linewidth=1,
                   label="0 = no feature-level separation"),
        ],
        loc="center left", bbox_to_anchor=(0.88, 0.52),
        title="How to read this figure",
    )
    fig.text(
        0.44, 0.94,
        "Positive gap: feature supports MUT > WT   |   Negative gap: reversed association",
        ha="center", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.87, 0.89))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_four_group_feature_contributions(
    group_means: pd.DataFrame,
    datasets: list[str],
    output_path: Path,
    figure_title: str,
    value_column: str = "contribution_to_MUT_log_odds",
    xlabel: str = "Mean contribution to model MUT score",
    treated_condition: str = "MMP9_inhibited",
    treated_label: str = "MMP",
) -> None:
    """Plot four group means so readers can see which genotype moved."""
    display_names = {
        "macrophage_all_model_b_plus_injury": "Macrophage: all cells + injury",
        "macrophage_outside_boundary_model_b": "Macrophage: outside injury",
        "musc_model_a_plus_injury": "MUSC: SVM + injury",
        "musc_model_a_no_injury_all_fish": "MUSC: L1, no injury (all fish)",
        "musc_model_a_no_injury_drop2": "MUSC: L1, no injury (drop 2)",
    }
    available = [name for name in datasets if name in set(group_means.dataset_name)]
    fig, axes = plt.subplots(
        1, len(available), figsize=(7.2 * len(available) + 5.0, 7), squeeze=False
    )
    columns = [
        ("untreated_full_fit_descriptive", "WT"),
        ("untreated_full_fit_descriptive", "MUT"),
        (treated_condition, "WT"),
        (treated_condition, "MUT"),
    ]
    for ax, dataset in zip(axes[0], available):
        table = group_means[group_means.dataset_name == dataset]
        pivot = table.pivot_table(
            index="feature", columns=["condition", "known_genotype"],
            values=value_column,
        ).reindex(columns=columns).fillna(0)
        magnitude = pivot.abs().max(axis=1)
        pivot = pivot.loc[magnitude.sort_values().index]
        y = np.arange(len(pivot))
        for yi, (_, row) in zip(y, pivot.iterrows()):
            uw, um, tw, tm = [row[column] for column in columns]
            # Connect treatment states within each genotype. This directly
            # answers whether WT, MUT, or both groups moved after treatment.
            ax.plot([uw, tw], [yi + 0.14, yi + 0.14], color="#277da1", lw=2)
            ax.plot([um, tm], [yi - 0.14, yi - 0.14], color="#f28e2b", lw=2)
            ax.scatter(uw, yi + 0.14, s=45, facecolors="white", edgecolors="#277da1", zorder=3)
            ax.scatter(tw, yi + 0.14, s=45, color="#277da1", zorder=3)
            ax.scatter(um, yi - 0.14, s=55, marker="^", facecolors="white", edgecolors="#f28e2b", zorder=3)
            ax.scatter(tm, yi - 0.14, s=55, marker="^", color="#f28e2b", zorder=3)
        names = [
            feature.replace("fish_mean__", "mean: ")
            .replace("fish_median__", "median: ")
            .replace("injury_", "injury: ")
            for feature in pivot.index
        ]
        ax.set_yticks(y, names)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(display_names.get(dataset, dataset.replace("_", " ")))
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(figure_title, fontsize=14)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#277da1",
                   linestyle="none", label="Untreated WT"),
            Line2D([0], [0], marker="^", markerfacecolor="white", markeredgecolor="#f28e2b",
                   linestyle="none", label="Untreated MUT"),
            Line2D([0], [0], marker="o", color="#277da1", linestyle="none", label=f"{treated_label} WT"),
            Line2D([0], [0], marker="^", color="#f28e2b", linestyle="none", label=f"{treated_label} MUT"),
            Line2D([0], [0], color="#277da1", label="WT: untreated → treated"),
            Line2D([0], [0], color="#f28e2b", label="MUT: untreated → treated"),
        ],
        loc="center left", bbox_to_anchor=(0.83, 0.5),
        title="Treatment shifts within genotype",
    )
    fig.tight_layout(rect=(0, 0, 0.82, 0.91))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    frozen_root = Path(args.frozen_root)
    output_dir = Path(args.output_dir)
    treated_condition = args.treated_condition
    treated_label = args.treated_label
    output_prefix = args.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    score_frames = []
    contribution_frames = []
    for path in sorted(results_root.glob("*/treated_fish_predictions.csv")):
        score_frames.append(pd.read_csv(path))
    for path in sorted(
        results_root.glob("*/untreated_reference_predictions_descriptive_only.csv")
    ):
        score_frames.append(pd.read_csv(path))
    for path in sorted(results_root.glob("*/treated_fish_feature_contributions.csv")):
        contribution_frames.append(pd.read_csv(path))
    for path in sorted(
        results_root.glob(
            "*/untreated_reference_feature_contributions_descriptive_only.csv"
        )
    ):
        contribution_frames.append(pd.read_csv(path))
    if not score_frames or not contribution_frames:
        raise FileNotFoundError("No complete frozen-model application outputs found.")

    scores = pd.concat(score_frames, ignore_index=True)
    scores["probability_WT"] = 1.0 - scores["probability_MUT"]
    contributions = pd.concat(contribution_frames, ignore_index=True)
    scores.to_csv(output_dir / f"all_{output_prefix}_fish_predictions.csv", index=False)
    contributions.to_csv(output_dir / f"all_{output_prefix}_feature_contributions.csv", index=False)
    summary = summarise(scores)
    summary.to_csv(output_dir / f"{output_prefix}_genotype_signal_summary.csv", index=False)

    separation = summary.pivot(
        index="dataset_name",
        columns="condition",
        values=[
            "probability_separation_MUT_minus_WT",
            "decision_separation_MUT_minus_WT",
            "mean_boundary_distance",
        ],
    )
    comparison_rows = []
    for dataset in separation.index:
        untreated_probability = separation.loc[
            dataset,
            ("probability_separation_MUT_minus_WT", "untreated_full_fit_descriptive"),
        ]
        treated_probability = separation.loc[
            dataset,
            ("probability_separation_MUT_minus_WT", treated_condition),
        ]
        untreated_decision = separation.loc[
            dataset,
            ("decision_separation_MUT_minus_WT", "untreated_full_fit_descriptive"),
        ]
        treated_decision = separation.loc[
            dataset,
            ("decision_separation_MUT_minus_WT", treated_condition),
        ]
        comparison_rows.append(
            {
                "dataset_name": dataset,
                "untreated_probability_separation_descriptive": untreated_probability,
                f"{output_prefix}_probability_separation": treated_probability,
                "change_in_absolute_probability_separation": abs(treated_probability) - abs(untreated_probability),
                "untreated_decision_separation_descriptive": untreated_decision,
                f"{output_prefix}_decision_separation": treated_decision,
                "change_in_absolute_decision_separation": abs(treated_decision) - abs(untreated_decision),
                "signal_interpretation": (
                    "clearer"
                    if abs(treated_decision) > abs(untreated_decision)
                    else "fuzzier"
                    if abs(treated_decision) < abs(untreated_decision)
                    else "unchanged"
                ),
                "reference_warning": "Untreated full-fit scores are descriptive; nested LOFO is the unbiased performance estimate.",
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / f"{output_prefix}_vs_untreated_signal_change.csv", index=False
    )

    contribution_means = (
        contributions.groupby(
            ["dataset_name", "condition", "known_genotype", "feature"],
            as_index=False,
        )["contribution_to_MUT_log_odds"]
        .mean()
    )
    contribution_pivot = contribution_means.pivot_table(
        index=["dataset_name", "known_genotype", "feature"],
        columns="condition",
        values="contribution_to_MUT_log_odds",
    ).reset_index()
    if {
        treated_condition,
        "untreated_full_fit_descriptive",
    }.issubset(contribution_pivot.columns):
        contribution_pivot["change_treated_minus_untreated"] = (
            contribution_pivot[treated_condition]
            - contribution_pivot["untreated_full_fit_descriptive"]
        )
    contribution_pivot.to_csv(
        output_dir / "feature_contribution_change_by_genotype.csv", index=False
    )

    standardized_means = (
        contributions.groupby(
            ["dataset_name", "condition", "known_genotype", "feature"],
            as_index=False,
        )["standardized_value"]
        .mean()
    )
    standardized_pivot = standardized_means.pivot_table(
        index=["dataset_name", "known_genotype", "feature"],
        columns="condition",
        values="standardized_value",
    ).reset_index()
    standardized_pivot["change_treated_minus_untreated"] = (
        standardized_pivot[treated_condition]
        - standardized_pivot["untreated_full_fit_descriptive"]
    )
    standardized_pivot.to_csv(
        output_dir / "standardized_feature_value_change_by_genotype.csv",
        index=False,
    )
    plot_feature_shifts(
        standardized_pivot,
        "change_treated_minus_untreated",
        "change in selected feature values",
        f"{treated_label} − untreated mean (untreated SD units)",
        output_dir / "selected_feature_value_changes.png",
    )
    plot_feature_shifts(
        contribution_pivot,
        "change_treated_minus_untreated",
        "change in contribution to MUT score",
        f"{treated_label} − untreated mean contribution",
        output_dir / "selected_feature_contribution_changes.png",
    )
    plot_contribution_gap_arrows(
        contribution_means,
        [
            "macrophage_all_model_b_plus_injury",
            "macrophage_outside_boundary_model_b",
            "musc_model_a_plus_injury",
        ],
        output_dir / "feature_contribution_gap_primary_models.png",
        f"How {treated_label} changes feature-level genotype separation: primary models",
        treated_condition,
        treated_label,
    )
    plot_contribution_gap_arrows(
        contribution_means,
        [
            "musc_model_a_plus_injury",
            "musc_model_a_no_injury_all_fish",
            "musc_model_a_no_injury_drop2",
        ],
        output_dir / "feature_contribution_gap_musc_models.png",
        f"How {treated_label} changes feature-level genotype separation: MUSC models",
        treated_condition,
        treated_label,
    )
    plot_four_group_feature_contributions(
        contribution_means,
        [
            "macrophage_all_model_b_plus_injury",
            "macrophage_outside_boundary_model_b",
            "musc_model_a_plus_injury",
        ],
        output_dir / "feature_contributions_four_groups_primary_models.png",
        "Which genotype moved? Mean feature contributions in all four groups",
        treated_condition=treated_condition,
        treated_label=treated_label,
    )
    plot_four_group_feature_contributions(
        contribution_means,
        [
            "musc_model_a_plus_injury",
            "musc_model_a_no_injury_all_fish",
            "musc_model_a_no_injury_drop2",
        ],
        output_dir / "feature_contributions_four_groups_musc_models.png",
        "Which genotype moved? Mean feature contributions across MUSC models",
        treated_condition=treated_condition,
        treated_label=treated_label,
    )
    plot_four_group_feature_contributions(
        standardized_means,
        [
            "macrophage_all_model_b_plus_injury",
            "macrophage_outside_boundary_model_b",
            "musc_model_a_plus_injury",
        ],
        output_dir / "feature_values_four_groups_primary_models.png",
        "Which biological measurements moved? Standardized feature values",
        value_column="standardized_value",
        xlabel="Mean standardized feature value (untreated-model SD units)",
        treated_condition=treated_condition,
        treated_label=treated_label,
    )
    plot_four_group_feature_contributions(
        standardized_means,
        [
            "musc_model_a_plus_injury",
            "musc_model_a_no_injury_all_fish",
            "musc_model_a_no_injury_drop2",
        ],
        output_dir / "feature_values_four_groups_musc_models.png",
        "Which biological measurements moved? Standardized MUSC feature values",
        value_column="standardized_value",
        xlabel="Mean standardized feature value (untreated-model SD units)",
        treated_condition=treated_condition,
        treated_label=treated_label,
    )

    compatibility = []
    for path in sorted(results_root.glob("*/treated_feature_compatibility_audit.csv")):
        compatibility.append(pd.read_csv(path))
    if compatibility:
        pd.concat(compatibility, ignore_index=True).to_csv(
            output_dir / "all_feature_compatibility_audit.csv", index=False
        )

    evaluation_rows = []
    for metrics_path in sorted(frozen_root.glob("*/frozen_model_summary.json")):
        evaluation_rows.append({"model_summary": str(metrics_path)})
    pd.DataFrame(evaluation_rows).to_csv(
        output_dir / "frozen_model_provenance_files.csv", index=False
    )

    primary_models = [
        "macrophage_all_model_b_plus_injury",
        "macrophage_outside_boundary_model_b",
        "musc_model_a_plus_injury",
    ]
    musc_models = [
        "musc_model_a_plus_injury",
        "musc_model_a_no_injury_all_fish",
        "musc_model_a_no_injury_drop2",
    ]
    score_plots = [
        ("probability_MUT", "Frozen untreated-model MUT probability", "MUT probability"),
        ("probability_WT", "Frozen untreated-model WT probability", "WT probability"),
        ("signed_decision_score", "Frozen untreated-model decision score", "decision score"),
    ]
    for value, ylabel, title_value in score_plots:
        stem = {
            "probability_MUT": f"{output_prefix}_MUT_probability",
            "probability_WT": f"{output_prefix}_WT_probability",
            "signed_decision_score": f"{output_prefix}_signed_decision_score",
        }[value]
        plot_scores(
            scores, value, ylabel, output_dir / f"{stem}.png", primary_models,
            f"Untreated versus {treated_label} {title_value}: primary models",
            treated_condition,
            treated_label,
        )
        plot_scores(
            scores, value, ylabel, output_dir / f"{stem}_musc_models.png", musc_models,
            f"Untreated versus {treated_label} {title_value}: MUSC models",
            treated_condition,
            treated_label,
        )
    plot_contributions(contributions, output_dir, treated_condition)
    print(f"[DONE] Combined treated-cohort analysis: {output_dir}")


if __name__ == "__main__":
    main()
