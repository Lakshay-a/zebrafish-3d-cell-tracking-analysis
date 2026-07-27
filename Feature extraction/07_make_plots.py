from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_curve


FISH_COLUMN_CANDIDATES = [
    "fish_id",
    "block_name",
    "block",
    "source_block",
    "sample_id",
]
GENOTYPE_COLUMN_CANDIDATES = [
    "genotype",
    "group",
    "condition",
    "class",
    "label",
]

DEFAULT_BIOLOGICAL_FEATURES = {
    "musc": [
        "fish_mean__tortuosity",
        "fish_mean__median_speed_um_per_frame",
        "fish_median__mean_elongation",
        "fish_median__mean_sphericity",
    ],
    "macrophage_all": [
        "fish_mean__tortuosity",
        "fish_mean__mean_squared_displacement_3d_um2",
        "fish_median__mean_elongation",
        "fish_median__mean_sphericity",
    ],
    "macrophage_outside_boundary": [
        "fish_mean__tortuosity",
        "fish_mean__mean_speed_um_per_frame",
        "fish_median__mean_elongation",
        "fish_median__mean_sphericity",
    ],
}

MAIN_CELL_FEATURES = [
    "net_displacement_3d_um",
    "directionality_ratio",
    "tortuosity",
    "mean_squared_displacement_3d_um2",
    "mean_speed_um_per_frame",
    "median_speed_um_per_frame",
    "mean_sphericity",
    "mean_elongation",
    "mean_volume_um3",
]

FEATURE_LABELS = {
    "fish_mean__tortuosity": "Mean cell tortuosity per fish",
    "fish_mean__median_speed_um_per_frame": "Mean cell median 3D speed per fish",
    "fish_mean__mean_speed_um_per_frame": "Mean cell mean 3D speed per fish",
    "fish_mean__mean_squared_displacement_3d_um2": "Mean cell 3D MSD per fish",
    "fish_median__mean_elongation": "Median cell elongation per fish",
    "fish_median__mean_sphericity": "Median cell sphericity per fish",
    "fish_mean__directionality_ratio": "Mean cell directionality ratio per fish",
    "fish_mean__net_displacement_3d_um": "Mean cell net 3D displacement per fish",
    "fish_mean__mean_volume_um3": "Mean cell volume per fish",
    "fish_median__mean_volume_um3": "Median cell volume per fish",
    "net_displacement_3d_um": "Net 3D displacement (um)",
    "directionality_ratio": "Directionality ratio",
    "tortuosity": "Tortuosity",
    "mean_squared_displacement_3d_um2": "Mean squared displacement 3D (um2)",
    "mean_speed_um_per_frame": "Mean speed (um/frame)",
    "median_speed_um_per_frame": "Median speed (um/frame)",
    "mean_sphericity": "Mean sphericity",
    "mean_elongation": "Mean elongation",
    "mean_volume_um3": "Mean volume (um3)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create supervisor-ready plots from fish-level model outputs."
    )
    parser.add_argument(
        "--analysis",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Repeat for each model-result directory.",
    )
    parser.add_argument(
        "--fish-table",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Repeat for each fish-level feature table.",
    )
    parser.add_argument(
        "--cell-table",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Repeat for each cell/track-level table used for per-fish dot plots.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated plots and summary tables.",
    )
    parser.add_argument("--group-a", default="WT")
    parser.add_argument("--group-b", default="MUT")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Bootstrap iterations for Cliff's-delta confidence intervals.",
    )
    parser.add_argument(
        "--label-all-fish",
        action="store_true",
        help="Label every point in probability plots; otherwise label errors only.",
    )
    parser.add_argument(
        "--top-variance-features",
        type=int,
        default=8,
        help="Also plot this many fish-level features with the highest variance.",
    )
    parser.add_argument(
        "--max-cell-points-per-fish",
        type=int,
        default=150,
        help="Maximum cell/track dots plotted per fish for each feature.",
    )
    return parser.parse_args()


def parse_mapping(values: list[str], role: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{role} must use LABEL=PATH format: {value}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not label:
            raise ValueError(f"Empty label in {role}: {value}")
        result[label] = path
    return result


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "unnamed"


def detect_column(
    df: pd.DataFrame,
    candidates: list[str],
    role: str,
) -> str:
    lower_map = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not detect {role} column in {list(df.columns)}")


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def dataset_key(label: str) -> str:
    text = label.lower().replace("-", "_").replace(" ", "_")
    if "outside" in text:
        return "macrophage_outside_boundary"
    if "macrophage" in text:
        return "macrophage_all"
    if "musc" in text or "musc" in text:
        return "musc"
    return text


def model_name(label: str, analysis_dir: Path) -> str:
    text = f"{label} {analysis_dir}".lower()
    return "elastic net" if "elastic" in text else "L1"


def analysis_level(label: str, analysis_dir: Path) -> str:
    text = f"{label} {analysis_dir}".lower()
    return "cell" if "cell" in text else "fish"


def read_single_row(path: Path) -> pd.Series:
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"Empty CSV: {path}")
    return table.iloc[0]


def collect_model_summary(
    analyses: dict[str, Path],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for label, directory in analyses.items():
        metrics_path = directory / "nested_lofo_metrics.csv"
        permutation_path = directory / "permutation_test.csv"
        predictions_path = directory / "nested_lofo_predictions.csv"

        missing = [
            str(path)
            for path in [metrics_path, permutation_path, predictions_path]
            if not path.exists()
        ]
        if missing:
            print(f"[WARN] Skipping {label}; missing: {missing}")
            continue

        metrics = read_single_row(metrics_path)
        permutation = read_single_row(permutation_path)
        predictions = pd.read_csv(predictions_path)

        records.append(
            {
                "analysis": label,
                "dataset": dataset_key(label),
                "model": model_name(label, directory),
                "level": analysis_level(label, directory),
                "fish_count": len(predictions),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "roc_auc": float(metrics["roc_auc"]),
                "permutation_p_value": float(
                    permutation["permutation_p_value"]
                ),
                "analysis_dir": str(directory),
            }
        )

    return pd.DataFrame(records)


def plot_metric_comparison(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    chance_line: float | None = None,
) -> None:
    if summary.empty:
        return

    datasets = list(dict.fromkeys(summary["dataset"].tolist()))
    models = list(dict.fromkeys(summary["model"].tolist()))
    x = np.arange(len(datasets), dtype=float)
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(8.0, 2.1 * len(datasets) + 2.0), 5.8))

    for index, model in enumerate(models):
        offsets = x - 0.4 + width / 2 + index * width
        values = []
        for dataset in datasets:
            match = summary[
                (summary["dataset"] == dataset)
                & (summary["model"] == model)
            ]
            values.append(
                float(match.iloc[0][metric]) if not match.empty else np.nan
            )
        bars = ax.bar(offsets, values, width=width, label=model)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", linewidth=1)

    labels = [
        value.replace("_", " ").replace("macrophage", "Macrophage").replace("musc", "MUSC")
        for value in datasets
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Nested LOFO {ylabel.lower()} by dataset and model")
    ax.legend(title="Classifier")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_evaluation_overview(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if summary.empty:
        return

    table = summary.copy()
    table["label"] = table["analysis"].astype(str)
    table = table.sort_values(["dataset", "level", "model", "analysis"]).reset_index(drop=True)
    y = np.arange(len(table))[::-1]

    dataset_colors = {
        "musc": "#1f77b4",
        "macrophage_all": "#2ca02c",
        "macrophage_outside_boundary": "#d62728",
    }
    level_markers = {
        "fish": "o",
        "cell": "s",
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, max(5.0, 0.48 * len(table) + 2.2)),
        sharey=True,
    )
    metrics = [
        ("balanced_accuracy", "Balanced accuracy", 0.5, "higher"),
        ("roc_auc", "ROC AUC", 0.5, "higher"),
        ("permutation_p_value", "Permutation p", 0.05, "lower"),
    ]

    for ax, (column, title, reference, direction) in zip(axes, metrics):
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        if direction == "higher":
            ax.axvspan(reference, 1, color="#e8f4ea", alpha=0.85, zorder=0)
        else:
            finite_values = values[np.isfinite(values)]
            right_edge = max(0.35, float(np.nanmax(finite_values)) + 0.05) if len(finite_values) else 0.35
            ax.axvspan(0, reference, color="#e8f4ea", alpha=0.85, zorder=0)
            ax.set_xlim(0, right_edge)

        for value, yy, (_, row) in zip(values, y, table.iterrows()):
            if not np.isfinite(value):
                continue
            dataset = str(row["dataset"])
            level = str(row.get("level", "fish"))
            ax.scatter(
                value,
                yy,
                s=92,
                marker=level_markers.get(level, "o"),
                color=dataset_colors.get(dataset, "#666666"),
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            if np.isfinite(value):
                ax.text(value, yy + 0.12, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axvline(reference, linestyle="--", linewidth=1)
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.25)
        if column != "permutation_p_value":
            ax.set_xlim(0, 1)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(table["label"])
    fig.suptitle("Nested LOFO model evaluation", y=0.995)
    dataset_handles = [
        Line2D([0], [0], marker="o", color="w", label="MUSC",
               markerfacecolor=dataset_colors["musc"], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Macrophage all",
               markerfacecolor=dataset_colors["macrophage_all"], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Macrophage outside-boundary",
               markerfacecolor=dataset_colors["macrophage_outside_boundary"], markeredgecolor="black", markersize=8),
    ]
    level_handles = [
        Line2D([0], [0], marker="o", color="black", label="Fish-level",
               markerfacecolor="white", linestyle="None", markersize=8),
        Line2D([0], [0], marker="s", color="black", label="Cell-level",
               markerfacecolor="white", linestyle="None", markersize=8),
    ]
    axes[-1].legend(
        handles=dataset_handles + level_handles,
        loc="lower right",
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def load_predictions(directory: Path) -> pd.DataFrame:
    path = directory / "nested_lofo_predictions.csv"
    df = pd.read_csv(path)
    genotype_col = detect_column(df, GENOTYPE_COLUMN_CANDIDATES, "genotype")
    fish_col = detect_column(df, FISH_COLUMN_CANDIDATES, "fish")
    df = df.copy()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    return df.rename(columns={genotype_col: "_genotype", fish_col: "_fish"})


def plot_predicted_probability(
    label: str,
    directory: Path,
    output_path: Path,
    group_a: str,
    group_b: str,
    random_seed: int,
    label_all_fish: bool,
) -> None:
    df = load_predictions(directory)
    required = {"probability_group_b", "true_binary", "predicted_binary"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Cannot plot probabilities for {label}: {missing}")
        return

    rng = np.random.default_rng(random_seed)
    groups = [group_a, group_b]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    for group_index, group in enumerate(groups):
        subset = df[df["_genotype"] == group].copy()
        values = pd.to_numeric(
            subset["probability_group_b"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(values)
        subset = subset.loc[valid].copy()
        values = values[valid]
        if len(values) == 0:
            continue

        ax.boxplot(
            [values],
            positions=[group_index],
            widths=0.42,
            showfliers=False,
            patch_artist=False,
        )
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            group_index + jitter,
            values,
            s=55,
            alpha=0.9,
            zorder=3,
        )

        for jitter_value, (_, row), probability in zip(
            jitter, subset.iterrows(), values
        ):
            is_error = int(row["true_binary"]) != int(row["predicted_binary"])
            if label_all_fish or is_error:
                ax.annotate(
                    str(row["_fish"]),
                    (group_index + jitter_value, probability),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                )

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(f"Nested-LOFO predicted probability of {group_b}")
    ax.set_title(label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_roc(
    label: str,
    directory: Path,
    output_path: Path,
) -> None:
    df = load_predictions(directory)
    true = pd.to_numeric(df["true_binary"], errors="coerce")
    probability = pd.to_numeric(df["probability_group_b"], errors="coerce")
    valid = true.notna() & probability.notna()
    true = true[valid].to_numpy(int)
    probability = probability[valid].to_numpy(float)
    if len(np.unique(true)) < 2:
        return

    fpr, tpr, _ = roc_curve(true, probability)
    auc = float(np.trapezoid(tpr, fpr))

    fig, ax = plt.subplots(figsize=(6.3, 6.0))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"{label}: nested-LOFO ROC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_confusion(
    label: str,
    directory: Path,
    output_path: Path,
    group_a: str,
    group_b: str,
) -> None:
    df = load_predictions(directory)
    true = pd.to_numeric(df["true_binary"], errors="coerce")
    predicted = pd.to_numeric(df["predicted_binary"], errors="coerce")
    valid = true.notna() & predicted.notna()
    matrix = confusion_matrix(
        true[valid].to_numpy(int),
        predicted[valid].to_numpy(int),
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(5.7, 5.2))
    image = ax.imshow(matrix, aspect="equal")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                fontsize=17,
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([group_a, group_b])
    ax.set_yticklabels([group_a, group_b])
    ax.set_xlabel("Predicted genotype")
    ax.set_ylabel("True genotype")
    ax.set_title(f"{label}: nested-LOFO confusion matrix")
    fig.colorbar(image, ax=ax, label="Fish count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def clean_feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return (
        feature.replace("fish_mean__", "Mean: ")
        .replace("fish_median__", "Median: ")
        .replace("_", " ")
    )


def plot_stability(
    label: str,
    directory: Path,
    output_path: Path,
) -> None:
    path = directory / "feature_selection_stability.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    if "nonzero_selection_frequency" not in table.columns:
        return

    table = table.sort_values(
        ["nonzero_selection_frequency"], ascending=True
    )
    fig_height = max(5.2, 0.42 * len(table) + 1.7)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    y = np.arange(len(table))
    ax.hlines(
        y,
        0,
        table["nonzero_selection_frequency"],
        linewidth=1.2,
    )
    ax.scatter(
        table["nonzero_selection_frequency"],
        y,
        s=55,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([clean_feature_label(v) for v in table["feature"]])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Non-zero selection frequency across outer LOFO folds")
    ax.set_title(f"{label}: feature-selection stability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_coefficient_direction(
    label: str,
    directory: Path,
    output_path: Path,
    group_b: str,
) -> None:
    path = directory / "feature_selection_stability.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    needed = {
        "mean_signed_coefficient_when_selected",
        "nonzero_selection_frequency",
        "feature",
    }
    if not needed.issubset(table.columns):
        print(f"[WARN] Coefficient-direction columns unavailable for {label}.")
        return

    table = table[
        table["nonzero_selection_frequency"] > 0
    ].copy()
    if table.empty:
        return
    table = table.sort_values("mean_signed_coefficient_when_selected")
    y = np.arange(len(table))
    values = table["mean_signed_coefficient_when_selected"].to_numpy(float)

    fig_height = max(5.0, 0.48 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.hlines(y, 0, values, linewidth=1.2)
    ax.scatter(values, y, s=60)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_feature_label(v) for v in table["feature"]])
    ax.set_xlabel(
        f"Mean standardized coefficient when selected "
        f"(positive = greater probability of {group_b})"
    )
    ax.set_title(f"{label}: coefficient direction")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    differences = a[:, None] - b[None, :]
    return float(
        (np.sum(differences > 0) - np.sum(differences < 0))
        / differences.size
    )


def bootstrap_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int,
    random_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    values: list[float] = []
    for _ in range(iterations):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        values.append(cliffs_delta(sample_a, sample_b))
    lower, upper = np.nanpercentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def plot_fish_feature(
    dataset_label: str,
    fish_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    feature: str,
    output_path: Path,
    group_a: str,
    group_b: str,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, object] | None:
    if feature not in fish_table.columns:
        print(f"[WARN] Missing {feature} in {dataset_label}.")
        return None

    values = pd.to_numeric(
        fish_table[feature], errors="coerce"
    )
    table = fish_table[[fish_col, genotype_col]].copy()
    table["_value"] = values
    table = table.dropna(subset=["_value"])

    groups = [group_a, group_b]
    arrays = [
        table.loc[table[genotype_col] == group, "_value"].to_numpy(float)
        for group in groups
    ]
    if any(len(array) == 0 for array in arrays):
        return None

    delta = cliffs_delta(arrays[0], arrays[1])
    lower, upper = bootstrap_delta_ci(
        arrays[0],
        arrays[1],
        bootstrap_iterations,
        random_seed,
    )

    rng = np.random.default_rng(random_seed)
    fig, ax = plt.subplots(figsize=(7.1, 6.0))
    ax.boxplot(
        arrays,
        positions=[0, 1],
        widths=0.42,
        showfliers=False,
        patch_artist=False,
    )

    for group_index, group in enumerate(groups):
        subset = table[table[genotype_col] == group]
        jitter = rng.normal(0, 0.045, size=len(subset))
        ax.scatter(
            group_index + jitter,
            subset["_value"],
            s=55,
            alpha=0.9,
            zorder=3,
        )
        for jitter_value, (_, row) in zip(jitter, subset.iterrows()):
            ax.annotate(
                str(row[fish_col]),
                (group_index + jitter_value, row["_value"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups)
    ax.set_ylabel(clean_feature_label(feature))
    ax.set_title(
        f"{dataset_label}\n"
        f"Cliff's delta ({group_a} − {group_b}) = {delta:.2f} "
        f"[95% bootstrap CI {lower:.2f}, {upper:.2f}]"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)

    return {
        "dataset": dataset_label,
        "feature": feature,
        f"n_{safe_name(group_a)}": len(arrays[0]),
        f"n_{safe_name(group_b)}": len(arrays[1]),
        f"median_{safe_name(group_a)}": float(np.median(arrays[0])),
        f"median_{safe_name(group_b)}": float(np.median(arrays[1])),
        "cliffs_delta_positive_means_group_a_higher": delta,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
    }


def top_variance_features(
    fish_table: pd.DataFrame,
    n_features: int,
    already_included: list[str],
) -> list[str]:
    columns = [
        str(column)
        for column in fish_table.columns
        if str(column).startswith("fish_mean__")
        or str(column).startswith("fish_median__")
    ]
    if n_features <= 0:
        return []

    scored: list[tuple[str, float]] = []
    for column in columns:
        values = pd.to_numeric(fish_table[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if values.notna().sum() < 3 or values.nunique(dropna=True) < 2:
            continue
        variance = float(values.var(skipna=True))
        if np.isfinite(variance):
            scored.append((column, variance))

    included = set(already_included)
    ranked = [
        feature
        for feature, _ in sorted(scored, key=lambda item: (-item[1], item[0]))
        if feature not in included
    ]
    return ranked[:n_features]


def plot_correlation_heatmap(
    dataset_label: str,
    fish_table: pd.DataFrame,
    output_path: Path,
) -> None:
    columns = [
        str(column)
        for column in fish_table.columns
        if str(column).startswith("fish_mean__")
        or str(column).startswith("fish_median__")
    ]
    if len(columns) < 2:
        return

    matrix = fish_table[columns].apply(
        pd.to_numeric, errors="coerce"
    )
    correlation = matrix.corr(method="spearman")
    labels = [clean_feature_label(column) for column in columns]
    size = max(10.5, 0.46 * len(columns) + 4.0)

    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(
        correlation.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"{dataset_label}: fish-level Spearman correlations")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_cell_feature_by_fish(
    dataset_label: str,
    cell_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    feature: str,
    output_path: Path,
    summary_records: list[dict[str, object]],
    random_seed: int,
    max_points_per_fish: int,
) -> None:
    if feature not in cell_table.columns:
        print(f"[WARN] Missing {feature} in {dataset_label} cell table.")
        return

    table = cell_table[[fish_col, genotype_col, feature]].copy()
    table[feature] = pd.to_numeric(table[feature], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    table = table.dropna(subset=[feature])
    if table.empty:
        return

    fish_order = (
        table[[fish_col, genotype_col]]
        .drop_duplicates()
        .sort_values([genotype_col, fish_col])
        .reset_index(drop=True)
    )
    fish_to_x = {
        str(row[fish_col]): index
        for index, row in fish_order.iterrows()
    }
    rng = np.random.default_rng(random_seed)
    genotype_colors = {
        "WT": "#1f77b4",
        "MUT": "#d62728",
    }

    width = max(8.5, 0.48 * len(fish_order) + 3.0)
    fig, ax = plt.subplots(figsize=(width, 5.8))

    for index, row in fish_order.iterrows():
        fish = str(row[fish_col])
        genotype = str(row[genotype_col])
        values = table.loc[
            table[fish_col].astype(str) == fish,
            feature,
        ].to_numpy(float)
        if len(values) == 0:
            continue
        plotted = values
        if max_points_per_fish > 0 and len(values) > max_points_per_fish:
            plotted = rng.choice(values, size=max_points_per_fish, replace=False)
        jitter = rng.normal(0, 0.055, size=len(plotted))
        ax.scatter(
            index + jitter,
            plotted,
            s=14,
            color=genotype_colors.get(genotype, "#4d4d4d"),
            alpha=0.55,
            linewidths=0,
        )
        median = float(np.nanmedian(values))
        ax.hlines(
            median,
            index - 0.22,
            index + 0.22,
            color="black",
            linewidth=2.6,
            zorder=4,
        )
        summary_records.append(
            {
                "dataset": dataset_label,
                "fish_id": fish,
                "genotype": genotype,
                "feature": feature,
                "n_cell_track_values": int(len(values)),
                "median": median,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )

    labels = [
        f"{row[fish_col]}\n{row[genotype_col]}"
        for _, row in fish_order.iterrows()
    ]
    ax.set_xticks(np.arange(len(fish_order)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(clean_feature_label(feature))
    ax.set_title(f"{dataset_label}: {clean_feature_label(feature)} by fish")
    ax.grid(axis="y", alpha=0.22)
    legend_handles = [
        Line2D([0], [0], marker="o", color="#1f77b4", lw=0,
               markerfacecolor="#1f77b4", markersize=5, label="WT cell/track"),
        Line2D([0], [0], marker="o", color="#d62728", lw=0,
               markerfacecolor="#d62728", markersize=5, label="MUT cell/track"),
        Line2D([0], [0], color="black", lw=2.6, label="Fish median"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses = parse_mapping(args.analysis, "analysis")
    fish_tables = parse_mapping(args.fish_table, "fish-table")
    cell_tables = parse_mapping(args.cell_table, "cell-table")

    if not analyses and not fish_tables and not cell_tables:
        raise ValueError(
            "Provide at least one --analysis, --fish-table, or --cell-table."
        )

    summary = collect_model_summary(analyses)
    if not summary.empty:
        summary.to_csv(output_dir / "model_summary.csv", index=False)
        plot_metric_comparison(
            summary,
            "balanced_accuracy",
            "Balanced accuracy",
            output_dir / "model_comparison_balanced_accuracy.png",
            chance_line=0.5,
        )
        plot_metric_comparison(
            summary,
            "roc_auc",
            "ROC AUC",
            output_dir / "model_comparison_roc_auc.png",
            chance_line=0.5,
        )
        plot_metric_comparison(
            summary,
            "permutation_p_value",
            "Permutation p-value",
            output_dir / "model_comparison_permutation_p.png",
            chance_line=0.05,
        )
        plot_evaluation_overview(
            summary,
            output_dir / "evaluation_overview__nested_lofo.png",
        )

    for index, (label, directory) in enumerate(analyses.items()):
        if not directory.exists():
            print(f"[WARN] Analysis directory not found: {directory}")
            continue

        suffix = safe_name(label)
        plot_predicted_probability(
            label,
            directory,
            output_dir / f"predicted_probability__{suffix}.png",
            args.group_a,
            args.group_b,
            args.random_seed + index,
            args.label_all_fish,
        )
        plot_roc(
            label,
            directory,
            output_dir / f"roc_curve__{suffix}.png",
        )
        plot_confusion(
            label,
            directory,
            output_dir / f"confusion_matrix__{suffix}.png",
            args.group_a,
            args.group_b,
        )
        plot_stability(
            label,
            directory,
            output_dir / f"feature_stability__{suffix}.png",
        )
        plot_coefficient_direction(
            label,
            directory,
            output_dir / f"coefficient_direction__{suffix}.png",
            args.group_b,
        )

    effect_records: list[dict[str, object]] = []
    for table_index, (label, table_path) in enumerate(fish_tables.items()):
        if not table_path.exists():
            print(f"[WARN] Fish table not found: {table_path}")
            continue
        fish_table = pd.read_csv(table_path, low_memory=False)
        fish_col = detect_column(
            fish_table, FISH_COLUMN_CANDIDATES, "fish"
        )
        genotype_col = detect_column(
            fish_table, GENOTYPE_COLUMN_CANDIDATES, "genotype"
        )
        fish_table = fish_table.copy()
        fish_table[genotype_col] = fish_table[genotype_col].map(
            normalise_genotype
        )
        fish_table = fish_table[
            fish_table[genotype_col].isin([args.group_a, args.group_b])
        ].copy()

        key = dataset_key(label)
        features = DEFAULT_BIOLOGICAL_FEATURES.get(key, []).copy()
        features.extend(
            top_variance_features(
                fish_table,
                args.top_variance_features,
                features,
            )
        )
        for feature_index, feature in enumerate(features):
            record = plot_fish_feature(
                label,
                fish_table,
                fish_col,
                genotype_col,
                feature,
                output_dir
                / f"fish_feature__{safe_name(label)}__{safe_name(feature)}.png",
                args.group_a,
                args.group_b,
                args.bootstrap_iterations,
                args.random_seed + table_index * 100 + feature_index,
            )
            if record is not None:
                effect_records.append(record)

        plot_correlation_heatmap(
            label,
            fish_table,
            output_dir / f"correlation_heatmap__{safe_name(label)}.png",
        )

    if effect_records:
        pd.DataFrame(effect_records).to_csv(
            output_dir / "fish_feature_effects.csv",
            index=False,
        )

    cell_summary_records: list[dict[str, object]] = []
    for table_index, (label, table_path) in enumerate(cell_tables.items()):
        if not table_path.exists():
            print(f"[WARN] Cell table not found: {table_path}")
            continue
        cell_table = pd.read_csv(table_path, low_memory=False)
        fish_col = detect_column(
            cell_table, FISH_COLUMN_CANDIDATES, "fish"
        )
        genotype_col = detect_column(
            cell_table, GENOTYPE_COLUMN_CANDIDATES, "genotype"
        )
        cell_table = cell_table.copy()
        cell_table[fish_col] = cell_table[fish_col].astype(str).str.strip()
        cell_table[genotype_col] = cell_table[genotype_col].map(
            normalise_genotype
        )
        cell_table = cell_table[
            cell_table[genotype_col].isin([args.group_a, args.group_b])
        ].copy()

        for feature_index, feature in enumerate(MAIN_CELL_FEATURES):
            plot_cell_feature_by_fish(
                label,
                cell_table,
                fish_col,
                genotype_col,
                feature,
                output_dir
                / f"cell_feature_by_fish__{safe_name(label)}__{safe_name(feature)}.png",
                cell_summary_records,
                args.random_seed + table_index * 100 + feature_index,
                args.max_cell_points_per_fish,
            )

    if cell_summary_records:
        pd.DataFrame(cell_summary_records).to_csv(
            output_dir / "cell_feature_by_fish_summary.csv",
            index=False,
        )

    print(f"[DONE] Supervisor plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
