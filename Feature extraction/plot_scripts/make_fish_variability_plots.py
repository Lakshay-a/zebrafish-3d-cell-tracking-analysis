#!/usr/bin/env python3
"""Describe within- and between-fish variability in saved cell-track features."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FISH_CANDIDATES = ["fish_id", "block_name", "block", "source_block", "sample_id"]
GENOTYPE_CANDIDATES = ["genotype", "group", "condition", "class", "label"]

GENOTYPE_COLORS = {
    "WT": "#1f77b4",
    "MUT": "#d62728",
}

DEFAULT_FEATURES = [
    "net_displacement_3d_um",
    "mean_squared_displacement_3d_um2_per_min",
    "mean_speed_um_per_min",
    "median_speed_um_per_min",
    "tortuosity",
    "directionality_ratio",
    "mean_sphericity",
    "mean_elongation",
    "mean_volume_um3",
    "z_range_um",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cell-feature variability between fish within genotype."
    )
    parser.add_argument("--dataset", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--group-order", nargs="*", default=["WT", "MUT"])
    parser.add_argument("--example-genotype", default="WT")
    parser.add_argument("--max-cell-points-per-fish", type=int, default=140)
    parser.add_argument("--top-variance-features", type=int, default=8)
    parser.add_argument(
        "--violin-y-quantile",
        type=float,
        default=0.98,
        help="Upper display quantile for violin plots; use 1.0 to show full range.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Use LABEL=CSV format: {value}")
        label, raw_path = value.split("=", 1)
        result[label.strip()] = Path(raw_path.strip())
    return result


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} column '{explicit}' not found.")
        return explicit
    lower_map = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not detect {role} column.")


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def feature_label(feature: str) -> str:
    replacements = {
        "net_displacement_3d_um": "Net 3D displacement (um)",
        "mean_squared_displacement_3d_um2_per_min": "Mean squared displacement rate 3D (um2/min)",
        "mean_speed_um_per_min": "Mean speed (um/min)",
        "median_speed_um_per_min": "Median speed (um/min)",
        "tortuosity": "Tortuosity",
        "directionality_ratio": "Directionality ratio",
        "mean_sphericity": "Mean sphericity",
        "mean_elongation": "Mean elongation",
        "mean_volume_um3": "Mean volume (um3)",
        "z_range_um": "Z range (um)",
    }
    return replacements.get(feature, feature.replace("_", " "))


def choose_features(df: pd.DataFrame, requested: list[str], top_n: int) -> list[str]:
    if requested:
        features = [feature for feature in requested if feature in df.columns]
    else:
        features = [feature for feature in DEFAULT_FEATURES if feature in df.columns]
    if top_n <= 0 or len(features) <= top_n:
        return features
    variances = []
    for feature in features:
        values = numeric(df[feature])
        if values.notna().sum() >= 3:
            variances.append((feature, float(values.var(skipna=True))))
    return [feature for feature, _ in sorted(variances, key=lambda item: -item[1])[:top_n]]


def fish_order(df: pd.DataFrame, fish_col: str, genotype_col: str, group_order: list[str]) -> pd.DataFrame:
    order = df[[fish_col, genotype_col]].drop_duplicates().copy()
    group_rank = {group: idx for idx, group in enumerate(group_order)}
    order["_rank"] = order[genotype_col].map(group_rank).fillna(len(group_rank))
    return order.sort_values(["_rank", fish_col]).drop(columns="_rank").reset_index(drop=True)


def plot_variability_strips(
    label: str,
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_order: list[str],
    output_path: Path,
    title_suffix: str,
    max_points_per_fish: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = fish_order(df, fish_col, genotype_col, group_order)
    if order.empty:
        return pd.DataFrame()

    n_cols = 2
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(10.5, 0.38 * len(order) + 5.0), 3.4 * n_rows),
        squeeze=False,
    )
    records: list[dict[str, object]] = []
    for ax, feature in zip(axes.ravel(), features):
        table = df[[fish_col, genotype_col, feature]].copy()
        table[feature] = numeric(table[feature])
        table = table.dropna(subset=[feature])
        for x_pos, row in order.iterrows():
            fish = str(row[fish_col])
            genotype = str(row[genotype_col])
            values = table.loc[table[fish_col].astype(str).eq(fish), feature].to_numpy(float)
            if len(values) == 0:
                continue
            plotted = values
            if max_points_per_fish > 0 and len(values) > max_points_per_fish:
                plotted = rng.choice(values, size=max_points_per_fish, replace=False)
            jitter = rng.normal(0, 0.055, size=len(plotted))
            color = GENOTYPE_COLORS.get(genotype, "#666666")
            ax.scatter(x_pos + jitter, plotted, s=10, alpha=0.33, color=color, linewidths=0)
            ax.hlines(np.nanmedian(values), x_pos - 0.24, x_pos + 0.24, color="black", linewidth=2.0)
            records.append(
                {
                    "dataset": label,
                    "fish_id": fish,
                    "genotype": genotype,
                    "feature": feature,
                    "n_tracks": int(len(values)),
                    "median": float(np.nanmedian(values)),
                    "q25": float(np.nanpercentile(values, 25)),
                    "q75": float(np.nanpercentile(values, 75)),
                    "iqr": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)),
                }
            )
        ax.set_title(feature_label(feature), fontsize=10)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([f"{r[fish_col]}\n{r[genotype_col]}" for _, r in order.iterrows()], rotation=65, ha="right", fontsize=6.5)
        ax.grid(axis="y", alpha=0.22)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="o", color="w", label=f"{group} cell/track",
               markerfacecolor=GENOTYPE_COLORS.get(group, "#666666"), markersize=6)
        for group in group_order
    ]
    handles.append(Line2D([0], [0], color="black", linewidth=2, label="Fish median"))
    fig.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)
    fig.suptitle(f"{label}: cell-feature variability between fish {title_suffix}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(records)


def plot_single_feature_side_panel_pngs(
    label: str,
    df: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    features: list[str],
    group_order: list[str],
    output_dir: Path,
    max_points_per_fish: int,
    violin_y_quantile: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    points_out = ensure_dir(output_dir / "points_plus_summary")
    summary_out = ensure_dir(output_dir / "median_iqr_summary")
    violin_out = ensure_dir(output_dir / "violin_summary")
    order = fish_order(df, fish_col, genotype_col, group_order)
    if order.empty:
        return

    for feature_index, feature in enumerate(features):
        if feature not in df.columns:
            continue
        table = df[[fish_col, genotype_col, feature]].copy()
        table[feature] = numeric(table[feature])
        table = table.dropna(subset=[feature])
        if table.empty:
            continue
        visible_upper = np.nan
        all_values = table[feature].to_numpy(float)
        all_values = all_values[np.isfinite(all_values)]
        if len(all_values):
            quantile = min(max(float(violin_y_quantile), 0.50), 1.0)
            visible_upper = float(np.nanquantile(all_values, quantile))
            visible_lower = float(np.nanquantile(all_values, max(0.0, 1.0 - quantile)))
            if visible_lower >= 0 and np.nanmin(all_values) >= 0:
                visible_lower = 0.0
            visible_range = visible_upper - visible_lower
            if not np.isfinite(visible_range) or visible_range <= 0:
                visible_lower = float(np.nanmin(all_values))
                visible_upper = float(np.nanmax(all_values))
                visible_range = visible_upper - visible_lower
            visible_pad = visible_range * 0.08 if np.isfinite(visible_range) and visible_range > 0 else 1.0

        summary_records: list[dict[str, object]] = []
        fig, ax = plt.subplots(figsize=(max(9.0, 0.48 * len(order) + 3.4), 5.8))
        for x_pos, row in order.iterrows():
            fish = str(row[fish_col])
            genotype = str(row[genotype_col])
            values = table.loc[table[fish_col].astype(str).eq(fish), feature].to_numpy(float)
            if len(values) == 0:
                continue
            plotted = values
            if max_points_per_fish > 0 and len(values) > max_points_per_fish:
                plotted = rng.choice(values, size=max_points_per_fish, replace=False)
            jitter = rng.normal(0, 0.06, size=len(plotted))
            color = GENOTYPE_COLORS.get(genotype, "#666666")
            q25, median, q75 = np.nanpercentile(values, [25, 50, 75])
            summary_records.append(
                {
                    "x_pos": x_pos,
                    "fish": fish,
                    "genotype": genotype,
                    "q25": q25,
                    "median": median,
                    "q75": q75,
                    "n": len(values),
                    "color": color,
                }
            )
            ax.vlines(x_pos, q25, q75, color="black", linewidth=3.4, alpha=0.58, zorder=2)
            ax.scatter(
                x_pos + jitter,
                plotted,
                s=11,
                alpha=0.22,
                color=color,
                edgecolors="none",
                zorder=3,
            )
            ax.scatter(
                x_pos,
                median,
                s=56,
                color=color,
                edgecolor="black",
                linewidth=0.75,
                zorder=5,
            )
            ax.hlines(median, x_pos - 0.24, x_pos + 0.24, color="black", linewidth=2.7, zorder=4)

        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [f"{row[fish_col]}\n{row[genotype_col]}" for _, row in order.iterrows()],
            rotation=65,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel(feature_label(feature))
        ax.set_title(f"{label}: {feature_label(feature)} by fish")
        ax.grid(axis="y", alpha=0.22)
        handles = [
            Line2D([0], [0], marker="o", color="w", label=f"{group} cell/track",
                   markerfacecolor=GENOTYPE_COLORS.get(group, "#666666"), markersize=6)
            for group in group_order
        ]
        handles.extend(
            [
                Line2D([0], [0], color="black", linewidth=3.4, alpha=0.58, label="Fish IQR"),
                Line2D([0], [0], color="black", linewidth=2.7, label="Fish median"),
            ]
        )
        ax.legend(handles=handles, frameon=True, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(points_out / f"feature__{feature_index + 1:02d}__{safe_name(feature)}.png", dpi=260, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(9.0, 0.48 * len(order) + 3.4), 5.8))
        for record in summary_records:
            x_pos = record["x_pos"]
            color = str(record["color"])
            ax.vlines(
                x_pos,
                float(record["q25"]),
                float(record["q75"]),
                color=color,
                linewidth=8.0,
                alpha=0.42,
                zorder=2,
            )
            ax.scatter(
                x_pos,
                float(record["median"]),
                s=115,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=4,
            )
            ax.hlines(
                float(record["median"]),
                x_pos - 0.26,
                x_pos + 0.26,
                color="black",
                linewidth=2.4,
                zorder=5,
            )
            ax.text(
                x_pos,
                float(record["q75"]),
                f"n={int(record['n'])}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#333333",
                rotation=90,
            )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [f"{row[fish_col]}\n{row[genotype_col]}" for _, row in order.iterrows()],
            rotation=65,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel(feature_label(feature))
        ax.set_title(f"{label}: {feature_label(feature)} fish medians and IQR")
        ax.grid(axis="y", alpha=0.22)
        summary_handles = [
            Line2D([0], [0], marker="o", color="w", label=f"{group} fish median",
                   markerfacecolor=GENOTYPE_COLORS.get(group, "#666666"),
                   markeredgecolor="black", markersize=8)
            for group in group_order
        ]
        summary_handles.append(Line2D([0], [0], color="black", linewidth=2.4, label="Median tick; vertical band = IQR"))
        ax.legend(handles=summary_handles, frameon=True, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(summary_out / f"feature__{feature_index + 1:02d}__{safe_name(feature)}.png", dpi=260, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(9.0, 0.48 * len(order) + 3.4), 5.8))
        violin_values: list[np.ndarray] = []
        violin_positions: list[int] = []
        violin_colors: list[str] = []
        for record in summary_records:
            fish = str(record["fish"])
            values = table.loc[table[fish_col].astype(str).eq(fish), feature].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                continue
            plot_values = values
            if np.isfinite(visible_upper) and violin_y_quantile < 1.0:
                plot_values = values[(values >= visible_lower) & (values <= visible_upper)]
            if len(plot_values) < 2:
                continue
            violin_values.append(plot_values)
            violin_positions.append(int(record["x_pos"]))
            violin_colors.append(str(record["color"]))
        if violin_values:
            parts = ax.violinplot(
                violin_values,
                positions=violin_positions,
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, color in zip(parts["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.28)
                body.set_linewidth(0.7)
        for record in summary_records:
            x_pos = int(record["x_pos"])
            color = str(record["color"])
            fish = str(record["fish"])
            values = table.loc[table[fish_col].astype(str).eq(fish), feature].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            plotted = values
            if np.isfinite(visible_upper) and violin_y_quantile < 1.0:
                plotted = plotted[(plotted >= visible_lower) & (plotted <= visible_upper)]
            if len(plotted) == 0:
                continue
            point_limit = max(25, min(max_points_per_fish, 65))
            if len(values) > point_limit:
                plotted = rng.choice(values, size=point_limit, replace=False)
            jitter = rng.normal(0, 0.045, size=len(plotted))
            ax.scatter(
                x_pos + jitter,
                plotted,
                s=8,
                alpha=0.20,
                color=color,
                edgecolors="none",
                zorder=2,
            )
            ax.scatter(
                x_pos,
                float(record["median"]),
                s=95,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
            ax.hlines(
                float(record["median"]),
                x_pos - 0.26,
                x_pos + 0.26,
                color="black",
                linewidth=2.4,
                zorder=6,
            )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [f"{row[fish_col]}\n{row[genotype_col]}" for _, row in order.iterrows()],
            rotation=65,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel(feature_label(feature))
        title = f"{label}: {feature_label(feature)} cell distributions by fish"
        if np.isfinite(visible_upper) and violin_y_quantile < 1.0:
            title += f" (y-axis zoomed to {violin_y_quantile:.0%})"
            ax.set_ylim(visible_lower - visible_pad, visible_upper + visible_pad)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        violin_handles = [
            Line2D([0], [0], marker="o", color="w", label=f"{group} fish median",
                   markerfacecolor=GENOTYPE_COLORS.get(group, "#666666"),
                   markeredgecolor="black", markersize=8)
            for group in group_order
        ]
        violin_handles.append(Line2D([0], [0], color="black", linewidth=2.4, label="Black tick = fish median; violin = cell distribution"))
        ax.legend(handles=violin_handles, frameon=True, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(violin_out / f"feature__{feature_index + 1:02d}__{safe_name(feature)}.png", dpi=260, bbox_inches="tight")
        plt.close(fig)


def fish_summary(df: pd.DataFrame, fish_col: str, genotype_col: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for (genotype, fish), sub in df.groupby([genotype_col, fish_col], dropna=False):
        row: dict[str, object] = {
            "genotype": genotype,
            "fish_id": fish,
            "n_tracks": int(len(sub)),
        }
        for feature in features:
            values = numeric(sub[feature]).dropna().to_numpy(float)
            row[f"{feature}__median"] = float(np.nanmedian(values)) if len(values) else np.nan
            row[f"{feature}__iqr"] = (
                float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
                if len(values)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["genotype", "fish_id"]).reset_index(drop=True)


def plot_fingerprint_heatmap(
    label: str,
    summary: pd.DataFrame,
    features: list[str],
    output_path: Path,
    genotype: str | None = None,
) -> None:
    table = summary.copy()
    title_suffix = "all fish"
    if genotype is not None:
        table = table[table["genotype"].eq(genotype)].copy()
        title_suffix = f"{genotype} fish"
    if table.empty:
        return
    columns = [f"{feature}__median" for feature in features]
    matrix = table[columns].to_numpy(float)
    scaled = np.zeros_like(matrix)
    for idx in range(matrix.shape[1]):
        col = matrix[:, idx]
        center = np.nanmedian(col)
        spread = np.nanpercentile(col, 75) - np.nanpercentile(col, 25)
        if not np.isfinite(spread) or spread <= 0:
            spread = np.nanstd(col)
        if not np.isfinite(spread) or spread <= 0:
            spread = 1.0
        scaled[:, idx] = (col - center) / spread

    fig, ax = plt.subplots(figsize=(max(9.0, 0.5 * len(features) + 5), max(5.2, 0.42 * len(table) + 2.2)))
    image = ax.imshow(scaled, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([feature_label(feature) for feature in features], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels([f"{row.fish_id} ({row.genotype}, n={row.n_tracks})" for row in table.itertuples()], fontsize=7)
    ax.set_title(f"{label}: fish fingerprint heatmap ({title_suffix})")
    fig.colorbar(image, ax=ax, label="Robust z-score of fish median")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_fish_pca_and_distance(
    label: str,
    summary: pd.DataFrame,
    features: list[str],
    output_dir: Path,
) -> None:
    columns = [f"{feature}__median" for feature in features]
    matrix = summary[columns].to_numpy(float)
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return
    scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(matrix))
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for genotype in summary["genotype"].dropna().unique():
        mask = summary["genotype"].eq(genotype).to_numpy()
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=82,
            color=GENOTYPE_COLORS.get(str(genotype), "#666666"),
            edgecolor="black",
            linewidth=0.6,
            label=str(genotype),
        )
        for x, y, fish in zip(coords[mask, 0], coords[mask, 1], summary.loc[mask, "fish_id"]):
            ax.annotate(str(fish), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"{label}: fish-to-fish feature map")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "fish_to_fish_pca.png", dpi=260)
    plt.close(fig)

    distances = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(coords[:, 0])
    fig, ax = plt.subplots(figsize=(max(6.2, 0.45 * len(summary) + 3), max(5.8, 0.45 * len(summary) + 2.5)))
    image = ax.imshow(distances[order][:, order], cmap="viridis", aspect="equal")
    labels = [f"{summary.iloc[i]['fish_id']}\n{summary.iloc[i]['genotype']}" for i in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"{label}: fish-to-fish distance from cell-feature medians")
    fig.colorbar(image, ax=ax, label="Euclidean distance in scaled feature space")
    fig.tight_layout()
    fig.savefig(output_dir / "fish_to_fish_distance_heatmap.png", dpi=260)
    plt.close(fig)


def plot_variance_partition(label: str, df: pd.DataFrame, fish_col: str, genotype_col: str, features: list[str], output_path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for feature in features:
        table = df[[fish_col, genotype_col, feature]].copy()
        table[feature] = numeric(table[feature])
        table = table.dropna(subset=[feature])
        if table[feature].nunique() < 2:
            continue
        total_var = float(table[feature].var(ddof=1))
        if not np.isfinite(total_var) or total_var <= 0:
            continue
        genotype_means = table.groupby(genotype_col)[feature].mean()
        fish_means = table.groupby([genotype_col, fish_col])[feature].mean()
        genotype_component = float(genotype_means.var(ddof=1)) if len(genotype_means) > 1 else 0.0
        fish_component = float(fish_means.var(ddof=1)) if len(fish_means) > 1 else 0.0
        residual = table[feature] - table.groupby(fish_col)[feature].transform("mean")
        residual_component = float(residual.var(ddof=1))
        components = np.array([genotype_component, max(0.0, fish_component - genotype_component), residual_component])
        if components.sum() <= 0:
            continue
        proportions = components / components.sum()
        records.append(
            {
                "dataset": label,
                "feature": feature,
                "genotype_fraction": float(proportions[0]),
                "fish_within_genotype_fraction": float(proportions[1]),
                "cell_within_fish_fraction": float(proportions[2]),
            }
        )
    result = pd.DataFrame(records)
    if result.empty:
        return result

    result = result.sort_values("fish_within_genotype_fraction", ascending=True)
    y = np.arange(len(result))
    fig, ax = plt.subplots(figsize=(9.0, max(5.4, 0.44 * len(result) + 2.0)))
    left = np.zeros(len(result))
    pieces = [
        ("genotype_fraction", "Genotype", "#9467bd"),
        ("fish_within_genotype_fraction", "Fish within genotype", "#ff7f0e"),
        ("cell_within_fish_fraction", "Cells/tracks within fish", "#7f7f7f"),
    ]
    for column, display, color in pieces:
        values = result[column].to_numpy(float)
        ax.barh(y, values, left=left, color=color, label=display)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(feature) for feature in result["feature"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Approximate fraction of observed variance")
    ax.set_title(f"{label}: where feature variability sits")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return result



def run_dataset(label: str, path: Path, output_root: Path, args: argparse.Namespace, dataset_index: int) -> None:
    df = pd.read_csv(path, low_memory=False)
    fish_col = detect_column(df, args.fish_col, FISH_CANDIDATES, "fish")
    genotype_col = detect_column(df, args.genotype_col, GENOTYPE_CANDIDATES, "genotype")
    df = df.copy()
    df[fish_col] = df[fish_col].astype(str).str.strip()
    df[genotype_col] = df[genotype_col].map(normalise_genotype)
    df = df[df[genotype_col].isin(args.group_order)].copy()
    features = choose_features(df, args.feature, args.top_variance_features)

    out = ensure_dir(output_root / safe_name(label))
    summary_dir = ensure_dir(out / "summary_tables")
    strip_all = plot_variability_strips(
        label,
        df,
        fish_col,
        genotype_col,
        features,
        args.group_order,
        out / "01_all_genotypes_cell_feature_variability_by_fish.png",
        "(all genotypes)",
        args.max_cell_points_per_fish,
        args.random_seed + dataset_index,
    )
    example = df[df[genotype_col].eq(args.example_genotype)].copy()
    strip_example = plot_variability_strips(
        label,
        example,
        fish_col,
        genotype_col,
        features,
        [args.example_genotype],
        out / f"02_{safe_name(args.example_genotype)}_only_cell_feature_variability_by_fish.png",
        f"({args.example_genotype} only)",
        args.max_cell_points_per_fish,
        args.random_seed + 100 + dataset_index,
    )
    if not strip_all.empty:
        strip_all.to_csv(summary_dir / "cell_feature_variability_by_fish.csv", index=False)
    if not strip_example.empty:
        strip_example.to_csv(summary_dir / f"{safe_name(args.example_genotype)}_only_cell_feature_variability_by_fish.csv", index=False)
    plot_single_feature_side_panel_pngs(
        label,
        df,
        fish_col,
        genotype_col,
        features,
        args.group_order,
        out / "interactive_side_panel_pngs" / "all_genotypes",
        args.max_cell_points_per_fish,
        args.violin_y_quantile,
        args.random_seed + 200 + dataset_index,
    )
    plot_single_feature_side_panel_pngs(
        label,
        example,
        fish_col,
        genotype_col,
        features,
        [args.example_genotype],
        out / "interactive_side_panel_pngs" / f"{safe_name(args.example_genotype)}_only",
        args.max_cell_points_per_fish,
        args.violin_y_quantile,
        args.random_seed + 300 + dataset_index,
    )

    fish_table = fish_summary(df, fish_col, genotype_col, features)
    fish_table.to_csv(summary_dir / "fish_feature_median_iqr_summary.csv", index=False)
    plot_fingerprint_heatmap(label, fish_table, features, out / "03_fish_fingerprint_heatmap_all_genotypes.png")
    for genotype in args.group_order:
        plot_fingerprint_heatmap(label, fish_table, features, out / f"03_fish_fingerprint_heatmap_{safe_name(genotype)}.png", genotype)
    plot_fish_pca_and_distance(label, fish_table, features, out)
    variance = plot_variance_partition(label, df, fish_col, genotype_col, features, out / "06_variance_partition_by_feature.png")
    if not variance.empty:
        variance.to_csv(summary_dir / "variance_partition_by_feature.csv", index=False)
    print(f"[DONE] {label}: fish variability plots saved to {out}")


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_dir))
    for index, (label, path) in enumerate(parse_mapping(args.dataset).items()):
        run_dataset(label, path, output_root, args, index)
    print(f"[DONE] All fish variability plots saved under: {output_root}")


if __name__ == "__main__":
    main()
