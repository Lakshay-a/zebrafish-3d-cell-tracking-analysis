from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_FEATURES = [
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

FEATURE_ALIASES = {
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
        description=(
            "Extract constrained 3D track features and aggregate them to "
            "fish-level mean and median values."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="QC-filtered cell/track-level feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for constrained feature outputs.",
    )
    parser.add_argument(
        "--dataset-name",
        default="dataset",
        help="Name used in plot titles and run metadata.",
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
        help="Track identifier column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--extra-feature",
        action="append",
        default=[],
        help=(
            "Add an optional secondary feature. May be supplied multiple "
            "times, e.g. --extra-feature z_range_um."
        ),
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested feature is unavailable.",
    )
    parser.add_argument(
        "--no-correlation-plot",
        action="store_true",
        help="Skip the fish-level correlation heatmap.",
    )

    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Optional metadata CSV containing only fish to retain.",
    )

    parser.add_argument(
        "--metadata-fish-col",
        default=None,
        help="Fish ID column in metadata. Auto-detected when omitted.",
    )
    return parser.parse_args()


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
                f"Requested {role} column '{explicit}' was not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    lower_to_original = {
        str(column).lower(): str(column)
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not auto-detect the {role} column. "
            f"Provide --{role.replace('_', '-')}-col explicitly."
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


def safe_numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


# Directionality ratio: https://pmc.ncbi.nlm.nih.gov/articles/PMC4439174/
def derive_directionality_ratio(df: pd.DataFrame) -> pd.Series | None:
    required = {
        "net_displacement_3d_um",
        "total_path_length_3d_um",
    }
    if not required.issubset(df.columns):
        return None

    net = safe_numeric(df["net_displacement_3d_um"])
    path = safe_numeric(df["total_path_length_3d_um"])
    result = net / path.replace(0, np.nan)
    return result.clip(lower=0)


# Tortuosity reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC4439174/
def derive_tortuosity(df: pd.DataFrame) -> pd.Series | None:
    required = {
        "net_displacement_3d_um",
        "total_path_length_3d_um",
    }
    if not required.issubset(df.columns):
        return None

    net = safe_numeric(df["net_displacement_3d_um"])
    path = safe_numeric(df["total_path_length_3d_um"])
    result = path / net.replace(0, np.nan)

    # Tortuosity should normally be >= 1. Small numerical violations are
    # retained rather than silently altering data.
    return result


def prepare_requested_features(
    df: pd.DataFrame,
    requested_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_data: dict[str, pd.Series] = {}
    report_records: list[dict[str, object]] = []

    for feature in requested_features:
        source = ""
        status = ""
        values: pd.Series | None = None

        aliases = FEATURE_ALIASES.get(feature, [feature])
        for alias in aliases:
            if alias in df.columns:
                values = safe_numeric(df[alias])
                if alias.endswith("_per_second") and feature.endswith("_per_min"):
                    values = values * 60.0
                    source = f"{alias} * 60"
                    status = "converted_from_seconds"
                else:
                    source = alias
                    status = "present_in_input"
                break

        if values is None and feature == "directionality_ratio":
            values = derive_directionality_ratio(df)
            if values is not None:
                source = (
                    "derived: net_displacement_3d_um / "
                    "total_path_length_3d_um"
                )
                status = "derived"

        elif values is None and feature == "tortuosity":
            values = derive_tortuosity(df)
            if values is not None:
                source = (
                    "derived: total_path_length_3d_um / "
                    "net_displacement_3d_um"
                )
                status = "derived"

        if values is None:
            report_records.append(
                {
                    "feature": feature,
                    "available": False,
                    "status": "missing",
                    "source": "",
                    "nonmissing_rows": 0,
                    "missing_rows": len(df),
                    "nonmissing_fraction": 0.0,
                    "unique_numeric_values": 0,
                }
            )
            continue

        nonmissing = int(values.notna().sum())
        unique_count = int(values.nunique(dropna=True))
        feature_data[feature] = values

        report_records.append(
            {
                "feature": feature,
                "available": True,
                "status": status,
                "source": source,
                "nonmissing_rows": nonmissing,
                "missing_rows": int(values.isna().sum()),
                "nonmissing_fraction": (
                    float(nonmissing / len(values))
                    if len(values)
                    else np.nan
                ),
                "unique_numeric_values": unique_count,
            }
        )

    return pd.DataFrame(feature_data), pd.DataFrame(report_records)


def build_fish_level_table(
    cell_table: pd.DataFrame,
    fish_col: str,
    genotype_col: str,
    track_col: str | None,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [fish_col, genotype_col]
    grouped = cell_table.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    fish_table = grouped.size().reset_index(
        name="n_cell_track_rows"
    )

    if track_col and track_col in cell_table.columns:
        unique_tracks = (
            grouped[track_col]
            .nunique(dropna=True)
            .reset_index(name="n_unique_tracks")
        )
        fish_table = fish_table.merge(
            unique_tracks,
            on=group_columns,
            how="left",
        )
    else:
        fish_table["n_unique_tracks"] = fish_table[
            "n_cell_track_rows"
        ]

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
            summary,
            on=group_columns,
            how="left",
        )

        counts_table = counts_table.merge(
            summary[
                group_columns + [f"n_nonmissing__{feature}"]
            ],
            on=group_columns,
            how="left",
        )

    fish_table = fish_table.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)

    counts_table = counts_table.sort_values(
        [genotype_col, fish_col]
    ).reset_index(drop=True)

    return fish_table, counts_table


def correlation_columns(
    fish_table: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in fish_table.columns
        if column.startswith("fish_mean__")
        or column.startswith("fish_median__")
    ]


def save_correlation_outputs(
    fish_table: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    make_plot: bool,
) -> None:
    columns = correlation_columns(fish_table)
    if len(columns) < 2:
        return

    matrix = fish_table[columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    correlation = matrix.corr(method="spearman")
    correlation.to_csv(
        output_dir / "fish_level_spearman_correlations.csv"
    )

    if not make_plot:
        return

    short_labels = [
        column.replace("fish_mean__", "mean: ")
        .replace("fish_median__", "median: ")
        for column in columns
    ]

    size = max(10.0, 0.54 * len(columns) + 4.0)
    fig, ax = plt.subplots(figsize=(size, size))

    image = ax.imshow(
        correlation.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_xticklabels(
        short_labels,
        rotation=60,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title(
        f"{dataset_name}: constrained fish-level feature correlations"
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Spearman correlation",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "fish_level_spearman_correlations.png",
        dpi=220,
    )
    plt.close(fig)

def filter_to_metadata_fish(
    raw: pd.DataFrame,
    fish_col: str,
    metadata_file: str | None,
    metadata_fish_col: str | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Keep only fish whose IDs are present in the metadata CSV."""

    if metadata_file is None:
        return raw

    metadata_path = Path(metadata_file)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path, low_memory=False)

    if metadata.empty:
        raise ValueError(
            f"Metadata file is empty: {metadata_path}"
        )

    resolved_metadata_fish_col = detect_column(
        metadata,
        metadata_fish_col,
        FISH_COLUMN_CANDIDATES,
        "metadata fish",
    )

    desired_fish = set(
        metadata[resolved_metadata_fish_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    desired_fish.discard("")

    if not desired_fish:
        raise ValueError(
            "No valid fish IDs were found in the metadata file."
        )

    raw = raw.copy()
    raw[fish_col] = raw[fish_col].astype(str).str.strip()

    available_fish = set(raw[fish_col].dropna().unique())

    missing_from_features = sorted(
        desired_fish - available_fish
    )
    excluded_from_analysis = sorted(
        available_fish - desired_fish
    )
    retained_fish = sorted(
        available_fish & desired_fish
    )

    report_rows = []

    for fish in sorted(desired_fish | available_fish):
        report_rows.append(
            {
                "fish_id": fish,
                "listed_in_metadata": fish in desired_fish,
                "present_in_feature_table": fish in available_fish,
                "retained": (
                    fish in desired_fish
                    and fish in available_fish
                ),
            }
        )

    pd.DataFrame(report_rows).to_csv(
        output_dir / "metadata_fish_filter_report.csv",
        index=False,
    )

    filtered = raw[
        raw[fish_col].isin(desired_fish)
    ].copy()

    print()
    print("[INFO] Metadata fish filtering")
    print(f"       Metadata file: {metadata_path}")
    print(
        f"       Metadata fish column: "
        f"{resolved_metadata_fish_col}"
    )
    print(f"       Fish listed in metadata: {len(desired_fish)}")
    print(f"       Fish retained: {len(retained_fish)}")
    print(
        f"       Fish excluded from feature table: "
        f"{len(excluded_from_analysis)}"
    )

    if missing_from_features:
        print(
            "[WARN] Fish listed in metadata but absent from "
            "the feature table:"
        )
        for fish in missing_from_features:
            print(f"       - {fish}")

    if filtered.empty:
        raise ValueError(
            "No feature rows remained after metadata filtering. "
            "Check whether fish IDs match exactly."
        )

    return filtered

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path, low_memory=False)
    if raw.empty:
        raise ValueError(f"Input table is empty: {input_path}")

    fish_col = detect_column(
        raw,
        args.fish_col,
        FISH_COLUMN_CANDIDATES,
        "fish",
    )
    genotype_col = detect_column(
        raw,
        args.genotype_col,
        GENOTYPE_COLUMN_CANDIDATES,
        "genotype",
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
    raw[genotype_col] = raw[genotype_col].map(
        normalise_genotype
    )

    raw = filter_to_metadata_fish(
        raw=raw,
        fish_col=fish_col,
        metadata_file=args.metadata_file,
        metadata_fish_col=args.metadata_fish_col,
        output_dir=output_dir,
    )

    invalid_identity = (
        raw[fish_col].isna()
        | raw[genotype_col].isna()
        | raw[fish_col].eq("")
        | raw[genotype_col].eq("")
    )
    if invalid_identity.any():
        print(
            f"[WARN] Dropping {int(invalid_identity.sum())} rows with "
            "missing fish/genotype identity."
        )
        raw = raw.loc[~invalid_identity].copy()

    requested_features = list(
        dict.fromkeys(CORE_FEATURES + args.extra_feature)
    )

    feature_frame, availability = prepare_requested_features(
        raw,
        requested_features,
    )
    availability.to_csv(
        output_dir / "feature_availability_report.csv",
        index=False,
    )

    missing_features = availability.loc[
        ~availability["available"],
        "feature",
    ].tolist()

    if missing_features:
        print("[WARN] Requested features unavailable:")
        for feature in missing_features:
            print(f"       - {feature}")

        if args.require_all:
            raise ValueError(
                "One or more requested features are unavailable. "
                "See feature_availability_report.csv."
            )

    available_features = availability.loc[
        availability["available"],
        "feature",
    ].tolist()

    if len(available_features) < 2:
        raise ValueError(
            "Fewer than two constrained features are available."
        )

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

    fish_table, counts_table = build_fish_level_table(
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

    run_lines = [
        f"input={input_path}",
        f"dataset_name={args.dataset_name}",
        f"fish_col={fish_col}",
        f"genotype_col={genotype_col}",
        f"track_col={track_col}",
        f"input_rows={len(raw)}",
        f"fish_count={fish_table[fish_col].nunique()}",
        "requested_features=" + ",".join(requested_features),
        "available_features=" + ",".join(available_features),
        "missing_features=" + ",".join(missing_features),
        (
            "fish_level_summaries="
            "mean,median"
        ),
    ]
    (output_dir / "run_information.txt").write_text(
        "\n".join(run_lines) + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] Input rows: {len(raw)}")
    print(
        f"[INFO] Fish counts: "
        f"{fish_table[genotype_col].value_counts().to_dict()}"
    )
    print("[INFO] Available constrained features:")
    for feature in available_features:
        print(f"       - {feature}")

    print(
        f"[SAVED] Cell/track constrained table: "
        f"{output_dir / 'constrained_cell_track_features.csv'}"
    )
    print(
        f"[SAVED] Fish-level mean/median table: "
        f"{output_dir / 'constrained_fish_level_mean_median.csv'}"
    )
    print("[DONE] Constrained feature extraction complete.")


if __name__ == "__main__":
    main()
