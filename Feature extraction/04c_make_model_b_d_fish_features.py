#!/usr/bin/env python3
"""Build hypothesis-driven fish-level feature tables that are directly."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_FEATURES = [
    "net_displacement_3d_um",
    "directionality_ratio",
    "tortuosity",
    "mean_squared_displacement_3d_um2_per_min",
    "mean_speed_um_per_min",
    "median_speed_um_per_min",
    "mean_sphericity",
    "mean_elongation",
    "mean_volume_um3",
]

# Canonical output name -> accepted input aliases.
MODEL_B_ALIASES = {
    "mean_surface_area_um2": [
        "mean_surface_area_um2",
        "mean_surface_area",
        "surface_area_um2_mean",
        "track_mean_surface_area_um2",
    ],
    "median_surface_area_um2": [
        "median_surface_area_um2",
        "median_surface_area",
        "surface_area_um2_median",
    ],
    "mean_surface_area_to_volume": [
        "mean_surface_area_to_volume",
        "mean_surface_area_to_volume_ratio",
        "mean_surface_to_volume_ratio",
        "mean_sa_to_volume",
    ],
    "mean_solidity_3d": [
        "mean_solidity_3d",
        "mean_solidity",
        "solidity_3d_mean",
    ],
    "mean_extent_3d": [
        "mean_extent_3d",
        "mean_extent",
        "extent_3d_mean",
    ],
    "mean_major_axis_length_um": [
        "mean_major_axis_length_um",
        "mean_major_axis_length",
        "mean_long_axis_length_um",
        "mean_longest_length_um",
    ],
    "mean_minor_axis_length_um": [
        "mean_minor_axis_length_um",
        "mean_minor_axis_length",
        "mean_short_axis_length_um",
    ],
    "mean_aspect_ratio_3d": [
        "mean_aspect_ratio_3d",
        "mean_aspect_ratio",
        "mean_major_minor_axis_ratio",
    ],
    "mean_roughness_3d": [
        "mean_roughness_3d",
        "mean_roughness",
        "roughness_3d_mean",
    ],
    "mean_convex_hull_volume_ratio": [
        "mean_convex_hull_volume_ratio",
        "mean_volume_to_convex_hull_ratio",
        "mean_convexity_3d",
        "mean_convexity",
    ],
    "mean_prolate_ellipticity": [
        "mean_prolate_ellipticity",
        "prolate_ellipticity_mean",
    ],
    "mean_oblate_ellipticity": [
        "mean_oblate_ellipticity",
        "oblate_ellipticity_mean",
    ],
}

MODEL_D_ALIASES = {
    "sphericity_cv": [
        "sphericity_cv",
        "track_sphericity_cv",
        "coefficient_variation_sphericity",
    ],
    "elongation_cv": [
        "elongation_cv",
        "track_elongation_cv",
        "coefficient_variation_elongation",
    ],
    "volume_um3_cv": [
        "volume_um3_cv",
        "volume_cv",
        "track_volume_cv",
        "coefficient_variation_volume",
    ],
    "surface_area_to_volume_cv": [
        "surface_area_to_volume_cv",
        "surface_to_volume_cv",
        "sa_to_volume_cv",
    ],
    "median_absolute_sphericity_change": [
        "median_absolute_sphericity_change",
        "median_abs_sphericity_change",
        "median_sphericity_step",
    ],
    "median_absolute_elongation_change": [
        "median_absolute_elongation_change",
        "median_abs_elongation_change",
        "median_elongation_step",
    ],
    "median_relative_volume_change": [
        "median_relative_volume_change",
        "median_abs_relative_volume_change",
        "median_volume_fold_change",
    ],
    "maximum_relative_volume_change": [
        "maximum_relative_volume_change",
        "max_relative_volume_change",
        "max_observed_volume_fold_change",
    ],
    "sphericity_slope_per_min": [
        "sphericity_slope_per_min",
        "sphericity_slope_per_minute",
        "sphericity_slope_per_frame",
        "sphericity_slope",
    ],
    "elongation_slope_per_min": [
        "elongation_slope_per_min",
        "elongation_slope_per_minute",
        "elongation_slope_per_frame",
        "elongation_slope",
    ],
    "volume_slope_um3_per_min": [
        "volume_um3_slope_per_min",
        "volume_slope_um3_per_min",
        "volume_slope_um3_per_minute",
        "volume_slope_um3_per_frame",
        "volume_slope_per_frame",
        "volume_slope",
    ],
    "median_shape_step": [
        "median_shape_step",
        "median_shape_space_step",
        "shape_median_step",
    ],
    "shape_path_length": [
        "shape_path_length",
        "shape_space_path_length",
    ],
    "shape_persistence": [
        "shape_persistence",
        "shape_space_persistence",
    ],
}

MODEL_C_ALIASES = {
    "near_cluster_boundary_fraction": [
        "near_cluster_boundary_fraction",
        "fraction_near_cluster_boundary",
        "boundary_near_fraction",
    ],
    "mean_distance_to_cluster_boundary_px": [
        "mean_distance_to_cluster_boundary_px",
        "mean_distance_to_cluster_boundary",
        "mean_cluster_boundary_distance_px",
        "mean_distance_to_wound_boundary_px",
    ],
    "min_distance_to_cluster_boundary_px": [
        "min_distance_to_cluster_boundary_px",
        "minimum_distance_to_cluster_boundary_px",
        "min_cluster_boundary_distance_px",
        "min_distance_to_wound_boundary_px",
    ],
    "inside_cluster_fraction": [
        "inside_cluster_fraction",
        "fraction_inside_cluster",
    ],
    "ever_inside_cluster": [
        "ever_inside_cluster",
        "track_ever_inside_cluster",
    ],
    "overlap_cluster_mask_fraction": [
        "overlap_cluster_mask_fraction",
        "cluster_mask_overlap_fraction",
    ],
    "mean_cluster_overlap_fraction": [
        "mean_cluster_overlap_fraction",
        "mean_overlap_cluster_fraction",
    ],
    "max_cluster_overlap_fraction": [
        "max_cluster_overlap_fraction",
        "maximum_cluster_overlap_fraction",
    ],
    "total_mac_segmented_volume_near_wound": [
        "total_mac_segmented_volume_near_wound",
        "total_macrophage_segmented_volume_near_wound",
        "total_mac_segmented_vol_near_wound",
        "macrophage_volume_near_wound",
    ],
    "cluster_area_over_time": [
        "cluster_area_over_time",
        "cluster_area_time",
        "cluster_area",
        "wound_cluster_area",
    ],
    "cluster_volume_over_time": [
        "cluster_volume_over_time",
        "cluster_volume_time",
        "cluster_volume",
        "wound_cluster_volume",
    ],
    "mac_accumulation_rate": [
        "mac_accumulation_rate",
        "macrophage_accumulation_rate",
        "rate_of_mac_accumulation",
    ],
    "mac_dispersal_rate": [
        "mac_dispersal_rate",
        "macrophage_dispersal_rate",
        "rate_of_mac_dispersal",
    ],
    "cluster_expansion_rate": [
        "cluster_expansion_rate",
        "cluster_area_expansion_rate",
        "cluster_volume_expansion_rate",
    ],
    "cluster_contraction_rate": [
        "cluster_contraction_rate",
        "cluster_area_contraction_rate",
        "cluster_volume_contraction_rate",
    ],
    "n_cells_entering_cluster_per_window": [
        "n_cells_entering_cluster_per_window",
        "num_cells_entering_cluster_per_window",
        "cells_entering_cluster_per_time_window",
        "macrophages_entering_cluster_per_window",
    ],
    "n_cells_leaving_cluster_per_window": [
        "n_cells_leaving_cluster_per_window",
        "num_cells_leaving_cluster_per_window",
        "cells_leaving_cluster_per_time_window",
        "macrophages_leaving_cluster_per_window",
    ],
    "n_mac_objects_near_cluster_boundary": [
        "n_mac_objects_near_cluster_boundary",
        "num_mac_objects_near_cluster_boundary",
        "n_macrophage_objects_near_cluster_boundary",
        "macrophage_objects_near_cluster_boundary",
    ],
}

BASE_ALIASES = {
    "net_displacement_3d_um": [
        "net_displacement_3d_um",
    ],
    "directionality_ratio": [
        "directionality_ratio",
    ],
    "tortuosity": [
        "tortuosity",
    ],
    "mean_squared_displacement_3d_um2_per_min": [
        "mean_squared_displacement_3d_um2_per_min",
        "mean_squared_displacement_3d_um2_per_second",
    ],
    "mean_speed_um_per_min": [
        "mean_speed_um_per_min",
        "mean_speed_um_per_second",
    ],
    "median_speed_um_per_min": [
        "median_speed_um_per_min",
        "median_speed_um_per_second",
    ],
    "mean_sphericity": [
        "mean_sphericity",
    ],
    "mean_elongation": [
        "mean_elongation",
    ],
    "mean_volume_um3": [
        "mean_volume_um3",
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
        description="Build Model B, C, or D fish-level feature tables."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", choices=["b", "c", "d"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--fish-col", default=None)
    parser.add_argument("--genotype-col", default=None)
    parser.add_argument("--track-col", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--metadata-fish-col", default=None)
    parser.add_argument(
        "--extra-feature",
        action="append",
        default=[],
        help="Additional exact input column to include; may be repeated.",
    )
    parser.add_argument(
        "--minimum-new-features",
        type=int,
        default=2,
        help=(
            "Fail when fewer than this number of Model B/C/D additions are "
            "available. Set to 0 to allow a diagnostic-only run."
        ),
    )
    parser.add_argument(
        "--require-all-base",
        action="store_true",
        help="Fail if any of the nine current constrained features are unavailable.",
    )
    parser.add_argument(
        "--no-correlation-plot",
        action="store_true",
    )
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def detect_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested {role} column '{explicit}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    lower_map = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not detect {role} column. "
            f"Provide --{role.replace('_', '-')}-col."
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


# Directionality ratio: https://pmc.ncbi.nlm.nih.gov/articles/PMC4439174/
def derive_directionality_ratio(df: pd.DataFrame) -> pd.Series | None:
    needed = {"net_displacement_3d_um", "total_path_length_3d_um"}
    if not needed.issubset(df.columns):
        return None
    net = safe_numeric(df["net_displacement_3d_um"])
    path = safe_numeric(df["total_path_length_3d_um"])
    return (net / path.replace(0, np.nan)).clip(lower=0)


# Tortuosity reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC4439174/
def derive_tortuosity(df: pd.DataFrame) -> pd.Series | None:
    needed = {"net_displacement_3d_um", "total_path_length_3d_um"}
    if not needed.issubset(df.columns):
        return None
    net = safe_numeric(df["net_displacement_3d_um"])
    path = safe_numeric(df["total_path_length_3d_um"])
    return path / net.replace(0, np.nan)


def resolve_alias(
    df: pd.DataFrame,
    canonical: str,
    aliases: list[str],
) -> tuple[pd.Series | None, str, str]:
    lower_map = {str(column).lower(): str(column) for column in df.columns}

    for alias in aliases:
        original = lower_map.get(alias.lower())
        if original is not None:
            values = safe_numeric(df[original])
            if original.lower().endswith("_per_second") and canonical.endswith("_per_min"):
                return values * 60.0, f"{original} * 60", "converted_from_seconds"
            return values, original, "present_in_input"

    if canonical == "directionality_ratio":
        values = derive_directionality_ratio(df)
        if values is not None:
            return (
                values,
                "net_displacement_3d_um / total_path_length_3d_um",
                "derived",
            )

    if canonical == "tortuosity":
        values = derive_tortuosity(df)
        if values is not None:
            return (
                values,
                "total_path_length_3d_um / net_displacement_3d_um",
                "derived",
            )

    return None, "", "missing"


def apply_metadata_filter(
    raw: pd.DataFrame,
    fish_col: str,
    metadata_file: str | None,
    metadata_fish_col: str | None,
    output_dir: Path,
) -> pd.DataFrame:
    if not metadata_file:
        return raw

    path = Path(metadata_file)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    metadata = pd.read_csv(path, low_memory=False)
    resolved_metadata_col = detect_column(
        metadata,
        metadata_fish_col,
        [fish_col] + FISH_COLUMN_CANDIDATES,
        "metadata fish",
    )
    allowed = set(
        metadata[resolved_metadata_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    present = set(raw[fish_col].dropna().astype(str).str.strip())
    retained = present & allowed

    report_rows = []
    for fish in sorted(present | allowed):
        report_rows.append(
            {
                "fish": fish,
                "listed_in_metadata": fish in allowed,
                "found_in_feature_table": fish in present,
                "retained": fish in retained,
            }
        )
    pd.DataFrame(report_rows).to_csv(
        output_dir / "metadata_fish_filter_report.csv",
        index=False,
    )

    filtered = raw[raw[fish_col].astype(str).str.strip().isin(allowed)].copy()
    print(
        f"[INFO] Metadata filter retained {filtered[fish_col].nunique()} "
        f"of {raw[fish_col].nunique()} fish."
    )
    return filtered


def prepare_features(
    raw: pd.DataFrame,
    model: str,
    extra_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if model == "b":
        model_aliases = MODEL_B_ALIASES
    elif model == "c":
        model_aliases = MODEL_C_ALIASES
    else:
        model_aliases = MODEL_D_ALIASES
    requested: list[tuple[str, list[str], str]] = []

    for feature in BASE_FEATURES:
        requested.append((feature, BASE_ALIASES[feature], "base"))

    for canonical, aliases in model_aliases.items():
        requested.append((canonical, aliases, f"model_{model}"))

    for feature in extra_features:
        requested.append((feature, [feature], "user_extra"))

    feature_data: dict[str, pd.Series] = {}
    report_records: list[dict[str, object]] = []
    available_base: list[str] = []
    available_new: list[str] = []

    seen: set[str] = set()
    for canonical, aliases, family in requested:
        if canonical in seen:
            continue
        seen.add(canonical)

        values, source, status = resolve_alias(raw, canonical, aliases)
        if values is None:
            report_records.append(
                {
                    "feature": canonical,
                    "family": family,
                    "available": False,
                    "status": "missing",
                    "source": "",
                    "nonmissing_rows": 0,
                    "missing_rows": len(raw),
                    "unique_numeric_values": 0,
                }
            )
            continue

        nonmissing = int(values.notna().sum())
        unique = int(values.nunique(dropna=True))
        available = nonmissing >= 3 and unique >= 2

        report_records.append(
            {
                "feature": canonical,
                "family": family,
                "available": available,
                "status": status if available else "insufficient_numeric_data",
                "source": source,
                "nonmissing_rows": nonmissing,
                "missing_rows": int(values.isna().sum()),
                "unique_numeric_values": unique,
            }
        )

        if not available:
            continue

        feature_data[canonical] = values
        if family == "base":
            available_base.append(canonical)
        else:
            available_new.append(canonical)

    return (
        pd.DataFrame(feature_data),
        pd.DataFrame(report_records),
        available_base,
        available_new,
    )


def build_fish_table(
    cell_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    track_col: str | None,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [fish_col, genotype_col]
    grouped = cell_table.groupby(group_cols, dropna=False, sort=True)

    fish_table = grouped.size().reset_index(name="n_cell_track_rows")
    if track_col and track_col in cell_table.columns:
        unique_tracks = (
            grouped[track_col]
            .nunique(dropna=True)
            .reset_index(name="n_unique_tracks")
        )
        fish_table = fish_table.merge(unique_tracks, on=group_cols, how="left")
    else:
        fish_table["n_unique_tracks"] = fish_table["n_cell_track_rows"]

    counts_table = fish_table.copy()

    for feature in features:
        summary = (
            grouped[feature]
            .agg(["mean", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"fish_mean__{feature}",
                    "median": f"fish_median__{feature}",
                    "count": f"n_nonmissing__{feature}",
                }
            )
        )
        fish_table = fish_table.merge(
            summary.drop(columns=[f"n_nonmissing__{feature}"]),
            on=group_cols,
            how="left",
        )
        counts_table = counts_table.merge(
            summary[group_cols + [f"n_nonmissing__{feature}"]],
            on=group_cols,
            how="left",
        )

    fish_table = fish_table.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)
    counts_table = counts_table.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)
    return fish_table, counts_table


def save_correlation_outputs(
    fish_table: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    make_plot: bool,
) -> None:
    columns = [
        str(column)
        for column in fish_table.columns
        if str(column).startswith("fish_mean__")
        or str(column).startswith("fish_median__")
    ]
    if len(columns) < 2:
        return

    matrix = fish_table[columns].apply(pd.to_numeric, errors="coerce")
    correlation = matrix.corr(method="spearman")
    correlation.to_csv(
        output_dir / "fish_level_spearman_correlations.csv"
    )

    if not make_plot:
        return

    labels = [
        column.replace("fish_mean__", "mean: ")
        .replace("fish_median__", "median: ")
        .replace("_", " ")
        for column in columns
    ]
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
    ax.set_title(f"{dataset_name}: fish-level feature correlations")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(
        output_dir / "fish_level_spearman_correlations.png",
        dpi=230,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw = pd.read_csv(input_path, low_memory=False)
    if raw.empty:
        raise ValueError(f"Input CSV is empty: {input_path}")

    fish_col = detect_column(
        raw, args.fish_col, FISH_COLUMN_CANDIDATES, "fish"
    )
    genotype_col = detect_column(
        raw, args.genotype_col, GENOTYPE_COLUMN_CANDIDATES, "genotype"
    )
    track_col = detect_column(
        raw,
        args.track_col,
        TRACK_COLUMN_CANDIDATES,
        "track",
        required=False,
    )

    raw = raw.copy()
    raw[fish_col] = raw[fish_col].astype(str).str.strip()
    raw[genotype_col] = raw[genotype_col].map(normalise_genotype)

    invalid_identity = (
        raw[fish_col].isna()
        | raw[genotype_col].isna()
        | raw[fish_col].eq("")
        | raw[genotype_col].eq("")
    )
    if invalid_identity.any():
        raw = raw.loc[~invalid_identity].copy()

    raw = apply_metadata_filter(
        raw,
        fish_col,
        args.metadata_file,
        args.metadata_fish_col,
        output_dir,
    )

    (
        feature_frame,
        availability,
        available_base,
        available_new,
    ) = prepare_features(raw, args.model, args.extra_feature)

    availability.to_csv(
        output_dir / "feature_availability_report.csv",
        index=False,
    )

    missing_base = [
        feature for feature in BASE_FEATURES if feature not in available_base
    ]
    if missing_base:
        print("[WARN] Missing current constrained features:")
        for feature in missing_base:
            print(f"       - {feature}")
        if args.require_all_base:
            raise ValueError(
                "Base features are missing. See feature_availability_report.csv."
            )

    print(
        f"[INFO] Model {args.model.upper()} new features available: "
        f"{len(available_new)}"
    )
    for feature in available_new:
        print(f"       - {feature}")

    if len(available_new) < args.minimum_new_features:
        raise ValueError(
            f"Only {len(available_new)} Model {args.model.upper()} additions "
            f"were available; minimum required is {args.minimum_new_features}. "
            "Inspect feature_availability_report.csv or rerun with a lower "
            "--minimum-new-features value for diagnostics."
        )

    available_features = available_base + available_new
    if len(available_features) < 2:
        raise ValueError("Fewer than two usable features are available.")

    identity_columns = [fish_col, genotype_col]
    if track_col:
        identity_columns.append(track_col)

    cell_table = pd.concat(
        [
            raw[identity_columns].reset_index(drop=True),
            feature_frame[available_features].reset_index(drop=True),
        ],
        axis=1,
    )
    cell_table.to_csv(
        output_dir / "constrained_cell_track_features.csv",
        index=False,
    )

    fish_table, counts_table = build_fish_table(
        cell_table,
        fish_col,
        genotype_col,
        track_col,
        available_features,
    )
    fish_table.to_csv(
        output_dir / "constrained_fish_level_mean_median.csv",
        index=False,
    )
    counts_table.to_csv(
        output_dir / "fish_data_counts.csv",
        index=False,
    )

    save_correlation_outputs(
        fish_table,
        output_dir,
        args.dataset_name,
        make_plot=not args.no_correlation_plot,
    )

    definition_lines = [
        f"dataset_name={args.dataset_name}",
        f"model={args.model.upper()}",
        f"input={input_path}",
        f"fish_col={fish_col}",
        f"genotype_col={genotype_col}",
        f"track_col={track_col}",
        f"fish_count={fish_table[fish_col].nunique()}",
        "base_features=" + ",".join(available_base),
        "new_features=" + ",".join(available_new),
        "all_features=" + ",".join(available_features),
        "fish_level_aggregation=mean,median",
    ]
    (output_dir / "model_definition.txt").write_text(
        "\n".join(definition_lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"[SAVED] Fish-level table: "
        f"{output_dir / 'constrained_fish_level_mean_median.csv'}"
    )
    print(f"[DONE] Model {args.model.upper()} feature construction complete.")


if __name__ == "__main__":
    main()
