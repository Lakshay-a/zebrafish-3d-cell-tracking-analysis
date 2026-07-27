from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = {
    "musc": [
        "directionality_ratio",
        "mean_speed_um_per_frame",
        "total_path_length_3d_um",
        "net_displacement_3d_um",
        "z_range_um",
        "mean_sphericity",
        "mean_elongation",
        "mean_volume_um3",
    ],
    "macrophage_all": [
        "mean_speed_um_per_frame",
        "mean_step_distance_3d_um",
        "total_path_length_3d_um",
        "net_displacement_3d_um",
        "moving_step_fraction",
        "directionality_ratio",
        "z_range_um",
        "mean_sphericity",
        "mean_elongation",
        "median_elongation",
        "mean_volume_um3",
    ],
    "macrophage_outside_boundary": [
        "mean_speed_um_per_frame",
        "mean_step_distance_3d_um",
        "total_path_length_3d_um",
        "net_displacement_3d_um",
        "moving_step_fraction",
        "directionality_ratio",
        "z_range_um",
        "mean_sphericity",
        "mean_elongation",
        "median_elongation",
        "mean_volume_um3",
        "near_cluster_boundary_fraction",
        "mean_distance_to_cluster_boundary_um",
        "min_distance_to_cluster_boundary_um",
        "mean_distance_to_cluster_boundary_px",
        "min_distance_to_cluster_boundary_px",
    ],
}

FISH_COLUMN_CANDIDATES = [
    "fish_id",
    "block_name",
    "block",
    "source_block",
    "sample_id",
    "dataset_id",
    "czi_name",
    "file",
]

GENOTYPE_COLUMN_CANDIDATES = [
    "genotype",
    "group",
    "condition",
    "class",
    "label",
]

TRACK_COLUMN_CANDIDATES = [
    "track_id",
    "global_track_id",
    "cell_track_id",
    "cell_id",
    "object_track_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fish-level features and WT/MUT comparison plots."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Primary/normal-QC cell-level feature CSV.",
    )
    parser.add_argument(
        "--strict-input",
        default=None,
        help="Optional strict-QC cell-level feature CSV.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DEFAULT_FEATURES),
        help="Dataset-specific default feature set.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Optional comma-separated feature list. "
            "Overrides the dataset default features."
        ),
    )
    parser.add_argument(
        "--fish-col",
        default=None,
        help="Fish identifier column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--genotype-col",
        default=None,
        help="Genotype column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--track-col",
        default=None,
        help="Track/cell identifier column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--group-a",
        default="WT",
        help="Reference genotype for signed effects. Default: WT.",
    )
    parser.add_argument(
        "--group-b",
        default="MUT",
        help="Comparison genotype. Default: MUT.",
    )
    parser.add_argument(
        "--min-cell-tracks-per-fish",
        type=int,
        default=5,
        help="Minimum number of valid cell/track rows required per fish.",
    )
    parser.add_argument(
        "--no-fish-labels",
        action="store_true",
        help="Do not annotate individual fish IDs on genotype plots.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "unnamed"


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    role: str,
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested {role} column '{explicit}' was not found.\n"
                f"Available columns include: {list(df.columns)}"
            )
        return explicit

    lower_to_original = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not auto-detect the {role} column. "
            f"Use --{role.replace('_', '-')}-col explicitly.\n"
            f"Available columns include: {list(df.columns)}"
        )
    return None


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()

    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"

    return text


def load_cell_table(
    path: str | Path,
    fish_col: str | None,
    genotype_col: str | None,
    track_col: str | None,
) -> tuple[pd.DataFrame, str, str, str | None]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    resolved_fish_col = detect_column(
        df, fish_col, FISH_COLUMN_CANDIDATES, "fish"
    )
    resolved_genotype_col = detect_column(
        df, genotype_col, GENOTYPE_COLUMN_CANDIDATES, "genotype"
    )
    resolved_track_col = detect_column(
        df,
        track_col,
        TRACK_COLUMN_CANDIDATES,
        "track",
        required=False,
    )

    df = df.copy()
    df[resolved_fish_col] = df[resolved_fish_col].astype(str).str.strip()
    df[resolved_genotype_col] = df[resolved_genotype_col].map(normalise_genotype)

    missing_identity = (
        df[resolved_fish_col].eq("")
        | df[resolved_genotype_col].eq("")
        | df[resolved_fish_col].isna()
        | df[resolved_genotype_col].isna()
    )
    if missing_identity.any():
        print(
            f"[WARN] Dropping {int(missing_identity.sum())} rows with missing "
            "fish/genotype identity."
        )
        df = df.loc[~missing_identity].copy()

    return (
        df,
        resolved_fish_col,
        resolved_genotype_col,
        resolved_track_col,
    )


def choose_features(
    df: pd.DataFrame,
    dataset: str,
    feature_argument: str | None,
) -> list[str]:
    if feature_argument:
        requested = [
            item.strip()
            for item in feature_argument.split(",")
            if item.strip()
        ]
    else:
        requested = list(DEFAULT_FEATURES[dataset])

    available: list[str] = []
    unavailable: list[str] = []

    for feature in requested:
        if feature not in df.columns:
            unavailable.append(feature)
            continue

        converted = pd.to_numeric(df[feature], errors="coerce")
        if converted.notna().sum() < 2:
            unavailable.append(feature)
            continue

        df[feature] = converted
        available.append(feature)

    if unavailable:
        print("[WARN] Requested features not available or not numeric:")
        for feature in unavailable:
            print(f"       - {feature}")

    if not available:
        raise ValueError(
            "None of the requested features are available as numeric columns."
        )

    print("[INFO] Features included:")
    for feature in available:
        print(f"       - {feature}")

    return available


def coefficient_of_variation(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return np.nan

    mean_value = float(values.mean())
    if math.isclose(mean_value, 0.0, abs_tol=1e-12):
        return np.nan

    return float(values.std(ddof=1) / abs(mean_value))


def build_fish_level_table(
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    track_col: str | None,
    features: list[str],
    min_cell_tracks_per_fish: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [fish_col, genotype_col]
    grouped = df.groupby(group_columns, dropna=False, sort=True)

    counts = grouped.size().reset_index(name="n_cell_track_rows")

    if track_col and track_col in df.columns:
        unique_tracks = (
            grouped[track_col]
            .nunique(dropna=True)
            .reset_index(name="n_unique_tracks")
        )
        counts = counts.merge(unique_tracks, on=group_columns, how="left")
        eligibility_column = "n_unique_tracks"
    else:
        counts["n_unique_tracks"] = counts["n_cell_track_rows"]
        eligibility_column = "n_cell_track_rows"

    tables = [counts]

    for feature in features:
        summary = (
            grouped[feature]
            .agg(
                median="median",
                mean="mean",
                std="std",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )
        summary[f"iqr_{feature}"] = summary["q75"] - summary["q25"]

        cv_table = (
            grouped[feature]
            .apply(coefficient_of_variation)
            .reset_index(name=f"cv_{feature}")
        )

        summary = summary.rename(
            columns={
                "median": f"median_{feature}",
                "mean": f"mean_{feature}",
                "std": f"std_{feature}",
                "q25": f"q25_{feature}",
                "q75": f"q75_{feature}",
            }
        )
        summary = summary.merge(cv_table, on=group_columns, how="left")
        tables.append(summary)

    fish_table = tables[0]
    for table in tables[1:]:
        fish_table = fish_table.merge(table, on=group_columns, how="outer")

    fish_table = fish_table.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)

    keep_mask = (
        fish_table[eligibility_column].fillna(0)
        >= min_cell_tracks_per_fish
    )
    excluded = fish_table.loc[~keep_mask].copy()
    included = fish_table.loc[keep_mask].copy()

    return included, excluded


def ordered_groups(
    values: pd.Series,
    group_a: str,
    group_b: str,
) -> list[str]:
    observed = [str(v) for v in values.dropna().unique()]
    ordered: list[str] = []

    for preferred in [group_a, group_b]:
        if preferred in observed and preferred not in ordered:
            ordered.append(preferred)

    ordered.extend(sorted(v for v in observed if v not in ordered))
    return ordered


def add_fish_jitter(
    base_x: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 1:
        return np.array([base_x], dtype=float)
    return base_x + rng.normal(0.0, 0.055, size=n)


def plot_fish_level_genotype_comparisons(
    fish_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    output_dir: Path,
    group_a: str,
    group_b: str,
    annotate_fish: bool,
    prefix: str,
) -> None:
    plot_dir = output_dir / "fish_level_genotype_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    groups = ordered_groups(fish_table[genotype_col], group_a, group_b)
    rng = np.random.default_rng(42)

    for feature in features:
        value_col = f"median_{feature}"
        if value_col not in fish_table.columns:
            continue

        data = fish_table[[fish_col, genotype_col, value_col]].dropna()
        if data.empty:
            continue

        arrays = [
            data.loc[data[genotype_col] == group, value_col].to_numpy()
            for group in groups
        ]

        fig, ax = plt.subplots(figsize=(7.4, 5.8))
        valid_arrays = [
            arr if len(arr) else np.array([np.nan])
            for arr in arrays
        ]
        ax.boxplot(
            valid_arrays,
            positions=np.arange(1, len(groups) + 1),
            widths=0.5,
            showfliers=False,
        )

        for position, group in enumerate(groups, start=1):
            group_data = data.loc[data[genotype_col] == group]
            x_values = add_fish_jitter(position, len(group_data), rng)

            ax.scatter(
                x_values,
                group_data[value_col],
                s=48,
                alpha=0.9,
                label=group,
            )

            if annotate_fish:
                for x_value, (_, row) in zip(
                    x_values, group_data.iterrows()
                ):
                    ax.annotate(
                        str(row[fish_col]),
                        (x_value, row[value_col]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.85,
                    )

        ax.set_xticks(np.arange(1, len(groups) + 1))
        ax.set_xticklabels(groups)
        ax.set_xlabel("Genotype")
        ax.set_ylabel(f"Fish median: {feature}")
        ax.set_title(f"{prefix}: fish-level {feature}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        fig.savefig(
            plot_dir / f"{safe_name(feature)}_fish_median_by_genotype.png",
            dpi=220,
        )
        plt.close(fig)


# Cliff's delta reference: https://revistas.javeriana.edu.co/index.php/revPsycho/article/view/643
def cliffs_delta(group_a_values: np.ndarray, group_b_values: np.ndarray) -> float:
    a = np.asarray(group_a_values, dtype=float)
    b = np.asarray(group_b_values, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    differences = a[:, None] - b[None, :]
    wins = np.sum(differences > 0)
    losses = np.sum(differences < 0)
    return float((wins - losses) / (len(a) * len(b)))


# Benjamini-Hochberg FDR: https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()

    if valid.empty:
        return result

    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result.loc[order] = adjusted
    return result


def calculate_effect_sizes(
    fish_table: pd.DataFrame,
    genotype_col: str,
    features: list[str],
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for feature in features:
        value_col = f"median_{feature}"
        if value_col not in fish_table.columns:
            continue

        a = (
            fish_table.loc[
                fish_table[genotype_col] == group_a, value_col
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        b = (
            fish_table.loc[
                fish_table[genotype_col] == group_b, value_col
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(a) and len(b):
            try:
                test = mannwhitneyu(a, b, alternative="two-sided")
                u_value = float(test.statistic)
                p_value = float(test.pvalue)
            except ValueError:
                u_value = np.nan
                p_value = np.nan

            median_a = float(np.median(a))
            median_b = float(np.median(b))
            median_difference = median_a - median_b
            delta = cliffs_delta(a, b)
        else:
            u_value = np.nan
            p_value = np.nan
            median_a = np.nan
            median_b = np.nan
            median_difference = np.nan
            delta = np.nan

        records.append(
            {
                "feature": feature,
                f"n_{safe_name(group_a)}": len(a),
                f"n_{safe_name(group_b)}": len(b),
                f"median_{safe_name(group_a)}": median_a,
                f"median_{safe_name(group_b)}": median_b,
                f"median_difference_{safe_name(group_a)}_minus_{safe_name(group_b)}": median_difference,
                "cliffs_delta_positive_means_group_a_higher": delta,
                "mann_whitney_u": u_value,
                "mann_whitney_p": p_value,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result["mann_whitney_fdr_bh"] = benjamini_hochberg(
            result["mann_whitney_p"]
        )
        result["absolute_cliffs_delta"] = result[
            "cliffs_delta_positive_means_group_a_higher"
        ].abs()
        result = result.sort_values(
            "absolute_cliffs_delta", ascending=False
        ).reset_index(drop=True)

    return result


def plot_effect_sizes(
    effect_table: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    group_a: str,
    group_b: str,
) -> None:
    if effect_table.empty:
        return

    table = effect_table.dropna(
        subset=["cliffs_delta_positive_means_group_a_higher"]
    ).copy()
    if table.empty:
        return

    table = table.sort_values(
        "cliffs_delta_positive_means_group_a_higher"
    )

    fig_height = max(4.8, 0.48 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))

    y = np.arange(len(table))
    x = table["cliffs_delta_positive_means_group_a_higher"]

    ax.scatter(x, y, s=58)
    ax.hlines(y, 0, x, linewidth=1.2)
    ax.axvline(0, linewidth=1, linestyle="--")

    ax.set_yticks(y)
    ax.set_yticklabels(table["feature"])
    ax.set_xlabel(
        f"Cliff's delta (positive = {group_a} higher; negative = {group_b} higher)"
    )
    ax.set_title(f"{prefix}: fish-level genotype effect sizes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    fig.savefig(output_dir / "fish_level_effect_sizes.png", dpi=220)
    plt.close(fig)


def prepare_pca_matrix(
    fish_table: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, list[str]]:
    columns = [
        f"median_{feature}"
        for feature in features
        if f"median_{feature}" in fish_table.columns
    ]

    usable_columns: list[str] = []
    for column in columns:
        values = pd.to_numeric(fish_table[column], errors="coerce")
        if values.notna().sum() < 2:
            continue
        if values.nunique(dropna=True) < 2:
            continue
        usable_columns.append(column)

    if len(usable_columns) < 2:
        return np.empty((len(fish_table), 0)), []

    matrix = fish_table[usable_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    matrix_imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    matrix_scaled = StandardScaler().fit_transform(matrix_imputed)

    return matrix_scaled, usable_columns


# PCA API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def plot_fish_level_pca(
    fish_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    output_dir: Path,
    group_a: str,
    group_b: str,
    prefix: str,
) -> None:
    matrix, usable_columns = prepare_pca_matrix(fish_table, features)

    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        print(f"[WARN] Not enough usable data for {prefix} PCA.")
        return

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(matrix)

    pca_table = fish_table[[fish_col, genotype_col]].copy()
    pca_table["PC1"] = coordinates[:, 0]
    pca_table["PC2"] = coordinates[:, 1]
    pca_table.to_csv(output_dir / "fish_level_pca_coordinates.csv", index=False)

    loadings = pd.DataFrame(
        {
            "feature": usable_columns,
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
        }
    )
    loadings.to_csv(output_dir / "fish_level_pca_loadings.csv", index=False)

    groups = ordered_groups(
        fish_table[genotype_col], group_a, group_b
    )

    fig, ax = plt.subplots(figsize=(7.4, 6.2))

    for group in groups:
        subset = pca_table[pca_table[genotype_col] == group]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=70,
            alpha=0.9,
            label=group,
        )

        for _, row in subset.iterrows():
            ax.annotate(
                str(row[fish_col]),
                (row["PC1"], row["PC2"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title(f"{prefix}: selected-feature fish-level PCA")
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.axvline(0, linewidth=0.8, alpha=0.4)
    ax.legend(title="Genotype")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(output_dir / "fish_level_pca.png", dpi=220)
    plt.close(fig)


def plot_fish_feature_heatmap(
    fish_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    output_dir: Path,
    group_a: str,
    group_b: str,
    prefix: str,
) -> None:
    columns = [
        f"median_{feature}"
        for feature in features
        if f"median_{feature}" in fish_table.columns
    ]
    if not columns:
        return

    ordered = fish_table.copy()
    group_order = {
        group: index
        for index, group in enumerate(
            ordered_groups(
                fish_table[genotype_col], group_a, group_b
            )
        )
    }
    ordered["_group_order"] = ordered[genotype_col].map(group_order)
    ordered = ordered.sort_values(
        ["_group_order", fish_col]
    ).reset_index(drop=True)

    matrix = ordered[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(axis=0))

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0).replace(0, np.nan)
    z = (matrix - means) / stds
    z = z.fillna(0.0)

    row_labels = [
        f"{row[fish_col]} ({row[genotype_col]})"
        for _, row in ordered.iterrows()
    ]
    column_labels = [
        column.removeprefix("median_") for column in columns
    ]

    fig_width = max(9.0, 0.65 * len(columns) + 4.5)
    fig_height = max(5.0, 0.45 * len(ordered) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(z.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(f"{prefix}: standardised fish-level feature profiles")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Fish")
    fig.colorbar(image, ax=ax, label="Z-score across fish")
    fig.tight_layout()

    fig.savefig(output_dir / "fish_by_feature_heatmap.png", dpi=220)
    plt.close(fig)


def plot_cell_track_counts(
    fish_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    output_dir: Path,
    prefix: str,
) -> None:
    count_col = (
        "n_unique_tracks"
        if "n_unique_tracks" in fish_table.columns
        else "n_cell_track_rows"
    )

    table = fish_table.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)

    labels = [
        f"{row[fish_col]}\n{row[genotype_col]}"
        for _, row in table.iterrows()
    ]

    fig_width = max(8.5, 0.72 * len(table) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.3))

    x = np.arange(len(table))
    ax.bar(x, table[count_col])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Valid tracked cells/tracks")
    ax.set_xlabel("Fish")
    ax.set_title(f"{prefix}: valid tracked cells per fish")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(output_dir / "valid_cell_tracks_per_fish.png", dpi=220)
    plt.close(fig)


def run_single_qc_analysis(
    cell_df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    track_col: str | None,
    features: list[str],
    output_dir: Path,
    group_a: str,
    group_b: str,
    min_cell_tracks_per_fish: int,
    annotate_fish: bool,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fish_table, excluded = build_fish_level_table(
        cell_df,
        fish_col,
        genotype_col,
        track_col,
        features,
        min_cell_tracks_per_fish,
    )

    fish_table.to_csv(
        output_dir / "fish_level_features.csv", index=False
    )
    excluded.to_csv(
        output_dir / "excluded_fish_too_few_cell_tracks.csv",
        index=False,
    )

    plot_fish_level_genotype_comparisons(
        fish_table,
        fish_col,
        genotype_col,
        features,
        output_dir,
        group_a,
        group_b,
        annotate_fish,
        prefix,
    )

    effect_table = calculate_effect_sizes(
        fish_table,
        genotype_col,
        features,
        group_a,
        group_b,
    )
    effect_table.to_csv(
        output_dir / "fish_level_effect_size_summary.csv",
        index=False,
    )
    plot_effect_sizes(
        effect_table,
        output_dir,
        prefix,
        group_a,
        group_b,
    )

    plot_fish_level_pca(
        fish_table,
        fish_col,
        genotype_col,
        features,
        output_dir,
        group_a,
        group_b,
        prefix,
    )

    plot_fish_feature_heatmap(
        fish_table,
        fish_col,
        genotype_col,
        features,
        output_dir,
        group_a,
        group_b,
        prefix,
    )

    plot_cell_track_counts(
        fish_table,
        fish_col,
        genotype_col,
        output_dir,
        prefix,
    )

    return fish_table, effect_table


def robust_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(valid["x"], valid["y"])

    return float(result.statistic), float(result.pvalue)


def get_effect_for_feature(
    table: pd.DataFrame,
    feature: str,
) -> float:
    match = table.loc[table["feature"] == feature]
    if match.empty:
        return np.nan
    return float(
        match.iloc[0][
            "cliffs_delta_positive_means_group_a_higher"
        ]
    )


def make_qc_sensitivity_analysis(
    normal_table: pd.DataFrame,
    strict_table: pd.DataFrame,
    normal_effects: pd.DataFrame,
    strict_effects: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    output_dir: Path,
    group_a: str,
    group_b: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_columns = [fish_col, genotype_col]
    merged = normal_table.merge(
        strict_table,
        on=merge_columns,
        how="outer",
        suffixes=("_normal", "_strict"),
        indicator=True,
    )
    merged.to_csv(
        output_dir / "normal_vs_strict_fish_table_merged.csv",
        index=False,
    )

    shared = merged[merged["_merge"] == "both"].copy()

    records: list[dict[str, object]] = []
    for feature in features:
        normal_col = f"median_{feature}_normal"
        strict_col = f"median_{feature}_strict"

        if normal_col not in shared.columns or strict_col not in shared.columns:
            continue

        valid = shared[
            [fish_col, genotype_col, normal_col, strict_col]
        ].dropna()
        if valid.empty:
            continue

        abs_change = (valid[strict_col] - valid[normal_col]).abs()
        denominator = valid[normal_col].abs().replace(0, np.nan)
        relative_change = abs_change / denominator * 100

        correlation, correlation_p = robust_spearman(
            valid[normal_col], valid[strict_col]
        )

        normal_delta = get_effect_for_feature(normal_effects, feature)
        strict_delta = get_effect_for_feature(strict_effects, feature)

        records.append(
            {
                "feature": feature,
                "n_shared_fish": len(valid),
                "spearman_normal_vs_strict": correlation,
                "spearman_p": correlation_p,
                "median_absolute_change": float(abs_change.median()),
                "median_relative_absolute_change_pct": float(
                    relative_change.median()
                )
                if relative_change.notna().any()
                else np.nan,
                "normal_cliffs_delta": normal_delta,
                "strict_cliffs_delta": strict_delta,
                "absolute_effect_size_change": abs(
                    strict_delta - normal_delta
                )
                if np.isfinite(normal_delta)
                and np.isfinite(strict_delta)
                else np.nan,
                "effect_direction_same": (
                    np.sign(normal_delta) == np.sign(strict_delta)
                )
                if np.isfinite(normal_delta)
                and np.isfinite(strict_delta)
                else np.nan,
            }
        )

        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        for _, row in valid.iterrows():
            ax.plot(
                [0, 1],
                [row[normal_col], row[strict_col]],
                marker="o",
                linewidth=1,
                alpha=0.75,
            )
            ax.annotate(
                str(row[fish_col]),
                (1, row[strict_col]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                va="center",
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal QC", "Strict QC"])
        ax.set_ylabel(f"Fish median: {feature}")
        ax.set_title(f"QC sensitivity by fish: {feature}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"{safe_name(feature)}_normal_vs_strict_by_fish.png",
            dpi=220,
        )
        plt.close(fig)

    sensitivity = pd.DataFrame(records)
    sensitivity.to_csv(
        output_dir / "qc_sensitivity_summary.csv", index=False
    )

    if not sensitivity.empty:
        plot_table = sensitivity.dropna(
            subset=["normal_cliffs_delta", "strict_cliffs_delta"]
        ).copy()
        if not plot_table.empty:
            plot_table = plot_table.sort_values("normal_cliffs_delta")
            y = np.arange(len(plot_table))

            fig_height = max(5.0, 0.52 * len(plot_table) + 1.8)
            fig, ax = plt.subplots(figsize=(9.2, fig_height))

            for index, (_, row) in enumerate(plot_table.iterrows()):
                ax.plot(
                    [row["normal_cliffs_delta"], row["strict_cliffs_delta"]],
                    [index, index],
                    linewidth=1.4,
                )

            ax.scatter(
                plot_table["normal_cliffs_delta"],
                y,
                s=55,
                label="Normal QC",
            )
            ax.scatter(
                plot_table["strict_cliffs_delta"],
                y,
                s=55,
                marker="D",
                label="Strict QC",
            )
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(plot_table["feature"])
            ax.set_xlabel(
                f"Cliff's delta (positive = {group_a} higher; negative = {group_b} higher)"
            )
            ax.set_title("Fish-level genotype effects: normal versus strict QC")
            ax.legend()
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                output_dir / "normal_vs_strict_effect_sizes.png",
                dpi=220,
            )
            plt.close(fig)

    count_col = (
        "n_unique_tracks"
        if "n_unique_tracks_normal" in shared.columns
        else "n_cell_track_rows"
    )
    normal_count_col = f"{count_col}_normal"
    strict_count_col = f"{count_col}_strict"

    if (
        normal_count_col in shared.columns
        and strict_count_col in shared.columns
    ):
        count_table = shared[
            [
                fish_col,
                genotype_col,
                normal_count_col,
                strict_count_col,
            ]
        ].copy()
        count_table["removed_count"] = (
            count_table[normal_count_col] - count_table[strict_count_col]
        )
        count_table["removed_fraction"] = (
            count_table["removed_count"]
            / count_table[normal_count_col].replace(0, np.nan)
        )
        count_table.to_csv(
            output_dir / "normal_vs_strict_cell_track_counts.csv",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        for _, row in count_table.iterrows():
            ax.plot(
                [0, 1],
                [row[normal_count_col], row[strict_count_col]],
                marker="o",
                linewidth=1,
                alpha=0.75,
            )
            ax.annotate(
                str(row[fish_col]),
                (1, row[strict_count_col]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                va="center",
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal QC", "Strict QC"])
        ax.set_ylabel("Valid tracked cells/tracks")
        ax.set_title("Valid tracked cells per fish: normal versus strict QC")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / "normal_vs_strict_cell_tracks_by_fish.png",
            dpi=220,
        )
        plt.close(fig)

        removal_summary = (
            count_table.groupby(genotype_col, dropna=False)
            .agg(
                n_fish=(fish_col, "nunique"),
                total_normal=(normal_count_col, "sum"),
                total_strict=(strict_count_col, "sum"),
                median_removed_fraction=("removed_fraction", "median"),
                mean_removed_fraction=("removed_fraction", "mean"),
                max_removed_fraction=("removed_fraction", "max"),
            )
            .reset_index()
        )
        removal_summary["overall_removed_fraction"] = (
            1
            - removal_summary["total_strict"]
            / removal_summary["total_normal"].replace(0, np.nan)
        )
        removal_summary.to_csv(
            output_dir / "qc_removal_summary_by_genotype.csv",
            index=False,
        )

        ordered_counts = count_table.sort_values(
            [genotype_col, fish_col]
        ).reset_index(drop=True)
        labels = [
            f"{row[fish_col]}\n{row[genotype_col]}"
            for _, row in ordered_counts.iterrows()
        ]

        fig_width = max(8.5, 0.72 * len(ordered_counts) + 3.5)
        fig, ax = plt.subplots(figsize=(fig_width, 5.3))
        x = np.arange(len(ordered_counts))
        ax.bar(x, ordered_counts["removed_fraction"] * 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right")
        ax.set_ylabel("Cell/tracks removed by strict QC (%)")
        ax.set_title("Strict-QC removal fraction by fish")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / "strict_qc_removed_fraction_by_fish.png",
            dpi=220,
        )
        plt.close(fig)


def save_run_information(
    output_dir: Path,
    input_path: str,
    strict_input_path: str | None,
    fish_col: str,
    genotype_col: str,
    track_col: str | None,
    features: list[str],
    min_cell_tracks_per_fish: int,
) -> None:
    lines = [
        f"input={input_path}",
        f"strict_input={strict_input_path}",
        f"fish_col={fish_col}",
        f"genotype_col={genotype_col}",
        f"track_col={track_col}",
        f"min_cell_tracks_per_fish={min_cell_tracks_per_fish}",
        "features=" + ",".join(features),
    ]
    (output_dir / "run_information.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_df, fish_col, genotype_col, track_col = load_cell_table(
        args.input,
        args.fish_col,
        args.genotype_col,
        args.track_col,
    )
    features = choose_features(
        normal_df,
        args.dataset,
        args.features,
    )

    print()
    print(f"[INFO] Normal-QC input rows: {len(normal_df)}")
    print(f"[INFO] Fish column: {fish_col}")
    print(f"[INFO] Genotype column: {genotype_col}")
    print(f"[INFO] Track column: {track_col}")

    normal_dir = output_dir / "normal_qc"
    normal_table, normal_effects = run_single_qc_analysis(
        normal_df,
        fish_col,
        genotype_col,
        track_col,
        features,
        normal_dir,
        args.group_a,
        args.group_b,
        args.min_cell_tracks_per_fish,
        not args.no_fish_labels,
        prefix=f"{args.dataset} normal QC",
    )

    print(
        f"[SAVED] Normal-QC fish-level analysis: {normal_dir}"
    )
    print(
        f"[INFO] Included normal-QC fish: {len(normal_table)}"
    )

    if args.strict_input:
        strict_df, strict_fish_col, strict_genotype_col, strict_track_col = (
            load_cell_table(
                args.strict_input,
                args.fish_col or fish_col,
                args.genotype_col or genotype_col,
                args.track_col or track_col,
            )
        )

        if strict_fish_col != fish_col:
            strict_df = strict_df.rename(
                columns={strict_fish_col: fish_col}
            )
        if strict_genotype_col != genotype_col:
            strict_df = strict_df.rename(
                columns={strict_genotype_col: genotype_col}
            )
        if (
            track_col
            and strict_track_col
            and strict_track_col != track_col
        ):
            strict_df = strict_df.rename(
                columns={strict_track_col: track_col}
            )

        for feature in features:
            if feature in strict_df.columns:
                strict_df[feature] = pd.to_numeric(
                    strict_df[feature], errors="coerce"
                )

        strict_features = [
            feature for feature in features
            if feature in strict_df.columns
            and strict_df[feature].notna().sum() >= 2
        ]

        missing_in_strict = [
            feature for feature in features
            if feature not in strict_features
        ]
        if missing_in_strict:
            print("[WARN] Features unavailable in strict-QC input:")
            for feature in missing_in_strict:
                print(f"       - {feature}")

        print()
        print(f"[INFO] Strict-QC input rows: {len(strict_df)}")

        strict_dir = output_dir / "strict_qc"
        strict_table, strict_effects = run_single_qc_analysis(
            strict_df,
            fish_col,
            genotype_col,
            track_col if track_col in strict_df.columns else None,
            strict_features,
            strict_dir,
            args.group_a,
            args.group_b,
            args.min_cell_tracks_per_fish,
            not args.no_fish_labels,
            prefix=f"{args.dataset} strict QC",
        )

        print(
            f"[SAVED] Strict-QC fish-level analysis: {strict_dir}"
        )
        print(
            f"[INFO] Included strict-QC fish: {len(strict_table)}"
        )

        shared_features = [
            feature for feature in features
            if feature in strict_features
        ]

        sensitivity_dir = output_dir / "normal_vs_strict_qc"
        make_qc_sensitivity_analysis(
            normal_table,
            strict_table,
            normal_effects,
            strict_effects,
            fish_col,
            genotype_col,
            shared_features,
            sensitivity_dir,
            args.group_a,
            args.group_b,
        )
        print(
            f"[SAVED] Normal-versus-strict QC analysis: "
            f"{sensitivity_dir}"
        )

    save_run_information(
        output_dir,
        args.input,
        args.strict_input,
        fish_col,
        genotype_col,
        track_col,
        features,
        args.min_cell_tracks_per_fish,
    )

    print()
    print("[DONE] Fish-level feature extraction and plotting complete.")


if __name__ == "__main__":
    main()
