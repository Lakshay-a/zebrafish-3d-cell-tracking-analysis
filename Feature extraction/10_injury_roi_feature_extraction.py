from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import distance_transform_edt



# User settings


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

BLOCKS_ROOT = Path(os.environ.get(
    "INJURY_BLOCKS_ROOT",
    PROJECT_ROOT / "overnight_batch_outputs" / "MMP",
))

ANNOTATION_ROOT = Path(os.environ.get(
    "INJURY_ANNOTATION_ROOT",
    HERE / "manual_injury_annotations_mmp",
))

OUTPUT_ROOT = Path(os.environ.get(
    "INJURY_FEATURE_OUTPUT_ROOT",
    HERE / "manual_injury_feature_outputs_time_corrected_mmp",
))

MUSC_MODEL_A_TABLE = Path(os.environ.get(
    "INJURY_MUSC_TABLE",
    HERE / "constrained_fish_features_time_corrected_mmp" / "musc"
    / "constrained_fish_level_mean_median.csv",
))

MACROPHAGE_ALL_MODEL_A_TABLE = Path(os.environ.get(
    "INJURY_MACROPHAGE_ALL_TABLE",
    HERE / "constrained_fish_features_time_corrected_mmp" / "model_b"
    / "macrophage_all" / "constrained_fish_level_mean_median.csv",
))

MACROPHAGE_OUTSIDE_MODEL_A_TABLE = Path(os.environ.get(
    "INJURY_MACROPHAGE_OUTSIDE_TABLE",
    HERE / "constrained_fish_features_time_corrected_mmp" / "model_b"
    / "macrophage_outside_boundary" / "constrained_fish_level_mean_median.csv",
))

FRAME_INTERVAL_METADATA = Path(os.environ.get(
    "INJURY_FRAME_INTERVAL_METADATA",
    HERE / "MMP_metadata.csv",
))

DEFAULT_XY_UM = float(os.environ.get("INJURY_DEFAULT_XY_UM", "0.2959437779017855"))
DEFAULT_Z_UM = float(os.environ.get("INJURY_DEFAULT_Z_UM", "1.0"))

XY_OVERRIDES_UM = {}

Z_OVERRIDES_UM = {}

EXCLUDED_FISH = {
    value.strip()
    for value in os.environ.get("INJURY_EXCLUDED_FISH", "20240604_block06").split(",")
    if value.strip()
}

# Use only a compact, pre-specified feature set in the classifier input.
# Every extracted feature is still saved in injury_features_fish_level_all.csv.
USE_COMPACT_MODEL_FEATURES = True

# Direction is determined from the sign of physical-time approach velocity.
# Set a non-zero threshold only after estimating localisation noise in um/min.
DIRECTION_TOLERANCE_UM_PER_MIN = 0.0




FISH_COLUMN_CANDIDATES = [
    "block_name",
    "fish_id",
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

TIME_COLUMN_CANDIDATES = [
    "time",
    "frame",
    "t",
    "timepoint",
]

TRACK_COLUMN_CANDIDATES = [
    "track_id",
    "global_track_id",
    "cell_track_id",
]

Z_COLUMN_CANDIDATES = [
    "centroid_z",
    "z",
    "center_z",
]

Y_COLUMN_CANDIDATES = [
    "centroid_y",
    "y",
    "center_y",
]

X_COLUMN_CANDIDATES = [
    "centroid_x",
    "x",
    "center_x",
]

MUSC_TRACK_NAMES = [
    "musc_tracks_nearest_good_filtered.csv",
    "musc_tracks_nearest_clean_for_features.csv",
    "musc_tracks_nearest.csv",
    "musc_tracks_lap_good_filtered.csv",
    "musc_tracks_lap.csv",
]

MACROPHAGE_ALL_TRACK_NAMES = [
    "macrophage_tracks_lap_good_filtered.csv",
    "macrophage_tracks_lap_clean_for_features.csv",
    "macrophage_tracks_lap.csv",
]

# Compact fish-level features added to Model A.
COMPACT_TRACK_AGGREGATES = [
    "median_net_approach_um",
    "median_min_abs_distance_to_injury_um",
    "mean_fraction_detections_inside",
    "mean_fraction_steps_toward",
    "mean_toward_velocity_um_per_min",
    "mean_away_velocity_um_per_min",
]

COMPACT_FISH_COUNTS = [
    "entries_per_track",
    "exits_per_track",
]


def detect_column(
    df: pd.DataFrame,
    candidates: list[str],
    role: str,
    required: bool = True,
) -> str | None:
    lookup = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match

    if required:
        raise ValueError(
            f"Could not find {role} column. Available columns: "
            f"{list(df.columns)}"
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


def load_frame_interval_metadata(
    metadata_path: Path,
) -> dict[str, float]:
    """Load one fixed frame interval, in seconds, for each block/fish."""
    metadata = pd.read_csv(metadata_path, low_memory=False)

    required = {"block_name", "time_interval_seconds"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Frame-interval metadata is missing columns {sorted(missing)}. "
            f"Available columns: {list(metadata.columns)}"
        )

    table = metadata[["block_name", "time_interval_seconds"]].copy()
    table["block_name"] = (
        table["block_name"].astype(str).str.strip()
    )
    table["time_interval_seconds"] = pd.to_numeric(
        table["time_interval_seconds"], errors="coerce"
    )

    duplicate_mask = table["block_name"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(
            table.loc[duplicate_mask, "block_name"].unique().tolist()
        )
        raise ValueError(
            "Duplicate block_name values in frame-interval metadata: "
            + ", ".join(duplicates)
        )

    invalid = table[
        table["time_interval_seconds"].isna()
        | (table["time_interval_seconds"] <= 0)
    ]
    if not invalid.empty:
        invalid_names = invalid["block_name"].tolist()
        raise ValueError(
            "Missing or non-positive time_interval_seconds for: "
            + ", ".join(invalid_names)
        )

    return dict(
        zip(
            table["block_name"],
            table["time_interval_seconds"].astype(float),
        )
    )


def longest_true_duration(
    values: np.ndarray,
    elapsed_seconds: np.ndarray,
) -> float:
    """Longest continuously observed true run, measured in seconds."""
    values = np.asarray(values, dtype=bool)
    elapsed_seconds = np.asarray(elapsed_seconds, dtype=float)

    best = 0.0
    run_start: float | None = None

    for value, elapsed in zip(values, elapsed_seconds):
        if not np.isfinite(elapsed):
            run_start = None
            continue

        if value:
            if run_start is None:
                run_start = float(elapsed)
            best = max(best, float(elapsed) - run_start)
        else:
            run_start = None

    return float(best)


def find_track_file(
    block_dir: Path,
    candidate_names: list[str],
    channel: str,
) -> Path | None:
    matches: list[Path] = []

    for name in candidate_names:
        matches.extend(block_dir.rglob(name))

    if channel == "macrophage":
        matches = [
            path
            for path in matches
            if "outside_boundary" not in str(path).lower()
            and "outside-boundary" not in str(path).lower()
        ]

    if not matches:
        return None

    rank = {name: index for index, name in enumerate(candidate_names)}
    return sorted(
        set(matches),
        key=lambda path: (
            rank.get(path.name, 999),
            len(path.parts),
            str(path),
        ),
    )[0]


def load_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(tiff.imread(mask_path)).astype(bool)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a YX injury mask, got shape {mask.shape}: {mask_path}"
        )
    if not mask.any():
        raise ValueError(f"Injury mask is empty: {mask_path}")

    return mask


def signed_distance_map_um(mask: np.ndarray, xy_um: float) -> np.ndarray:
    """Positive outside the injury, negative inside the injury."""
    outside_distance = distance_transform_edt(~mask) * xy_um
    inside_distance = distance_transform_edt(mask) * xy_um

    signed = outside_distance.astype(float)
    signed[mask] = -inside_distance[mask]
    return signed


def sample_map(
    image: np.ndarray,
    y_values: pd.Series,
    x_values: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(float)
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(float)

    yi = np.rint(y).astype(int, copy=False)
    xi = np.rint(x).astype(int, copy=False)

    valid = (
        np.isfinite(y)
        & np.isfinite(x)
        & (yi >= 0)
        & (yi < image.shape[0])
        & (xi >= 0)
        & (xi < image.shape[1])
    )

    sampled = np.full(len(y), np.nan, dtype=float)
    sampled[valid] = image[yi[valid], xi[valid]]
    return sampled, valid


def longest_true_run(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool)
    best = 0
    current = 0

    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return int(best)


def safe_mean(values) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else np.nan


def safe_median(values) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else np.nan


def enrich_detections(
    tracks: pd.DataFrame,
    mask: np.ndarray,
    xy_um: float,
    z_um: float,
    time_interval_seconds: float,
) -> tuple[pd.DataFrame, dict[str, str]]:
    time_col = detect_column(tracks, TIME_COLUMN_CANDIDATES, "time")
    track_col = detect_column(tracks, TRACK_COLUMN_CANDIDATES, "track")
    y_col = detect_column(tracks, Y_COLUMN_CANDIDATES, "centroid Y")
    x_col = detect_column(tracks, X_COLUMN_CANDIDATES, "centroid X")
    z_col = detect_column(
        tracks,
        Z_COLUMN_CANDIDATES,
        "centroid Z",
        required=False,
    )

    if not np.isfinite(time_interval_seconds) or time_interval_seconds <= 0:
        raise ValueError(
            f"Invalid time_interval_seconds={time_interval_seconds}"
        )

    table = tracks.copy()
    table[time_col] = pd.to_numeric(table[time_col], errors="coerce")
    table[y_col] = pd.to_numeric(table[y_col], errors="coerce")
    table[x_col] = pd.to_numeric(table[x_col], errors="coerce")

    if z_col is not None:
        table[z_col] = pd.to_numeric(table[z_col], errors="coerce")
    else:
        table["_synthetic_z"] = 0.0
        z_col = "_synthetic_z"

    table = table.dropna(
        subset=[time_col, track_col, y_col, x_col]
    ).copy()

    distance_map = signed_distance_map_um(mask, xy_um)
    signed_distance, valid = sample_map(
        distance_map,
        table[y_col],
        table[x_col],
    )

    table["_inside_injury"] = False
    yi = np.rint(table[y_col].to_numpy(float)).astype(int)
    xi = np.rint(table[x_col].to_numpy(float)).astype(int)
    table.loc[valid, "_inside_injury"] = mask[
        yi[valid],
        xi[valid],
    ]

    table["_signed_distance_to_injury_um"] = signed_distance
    table["_abs_distance_to_injury_boundary_um"] = np.abs(
        signed_distance
    )

    table = table.sort_values([track_col, time_col]).copy()

    table["_delta_frame"] = table.groupby(track_col)[time_col].diff()
    non_initial = table["_delta_frame"].notna()
    invalid_delta = non_initial & (table["_delta_frame"] <= 0)
    if invalid_delta.any():
        bad_tracks = sorted(
            table.loc[invalid_delta, track_col].astype(str).unique().tolist()
        )
        raise ValueError(
            "Non-positive frame differences found for tracks: "
            + ", ".join(bad_tracks[:20])
        )

    table["_delta_time_seconds"] = (
        table["_delta_frame"] * float(time_interval_seconds)
    )
    table["_delta_time_minutes"] = (
        table["_delta_time_seconds"] / 60.0
    )

    start_frame = table.groupby(track_col)[time_col].transform("min")
    table["_elapsed_time_seconds"] = (
        table[time_col] - start_frame
    ) * float(time_interval_seconds)
    table["_elapsed_time_minutes"] = (
        table["_elapsed_time_seconds"] / 60.0
    )

    table["_previous_inside"] = (
        table.groupby(track_col)["_inside_injury"].shift(1)
    )
    table["_entry_event"] = (
        table["_previous_inside"].eq(False)
        & table["_inside_injury"].eq(True)
    )
    table["_exit_event"] = (
        table["_previous_inside"].eq(True)
        & table["_inside_injury"].eq(False)
    )

    previous_signed = table.groupby(track_col)[
        "_signed_distance_to_injury_um"
    ].shift(1)
    table["_injury_approach_displacement_um"] = (
        previous_signed - table["_signed_distance_to_injury_um"]
    )

    # Legacy frame-normalised values are retained for auditing only.
    table["_injury_approach_velocity_um_per_frame"] = (
        table["_injury_approach_displacement_um"]
        / table["_delta_frame"]
    )
    table["_injury_approach_velocity_um_per_min"] = (
        table["_injury_approach_displacement_um"]
        / table["_delta_time_minutes"]
    )

    table["_toward_step"] = (
        table["_injury_approach_velocity_um_per_min"]
        > DIRECTION_TOLERANCE_UM_PER_MIN
    )
    table["_away_step"] = (
        table["_injury_approach_velocity_um_per_min"]
        < -DIRECTION_TOLERANCE_UM_PER_MIN
    )

    for axis_column, scale, output_name in [
        (x_col, xy_um, "_dx_um"),
        (y_col, xy_um, "_dy_um"),
        (z_col, z_um, "_dz_um"),
    ]:
        table[output_name] = (
            table[axis_column]
            - table.groupby(track_col)[axis_column].shift(1)
        ) * scale

    table["_step_distance_3d_um"] = np.sqrt(
        table["_dx_um"] ** 2
        + table["_dy_um"] ** 2
        + table["_dz_um"] ** 2
    )
    table["_speed_3d_um_per_frame"] = (
        table["_step_distance_3d_um"] / table["_delta_frame"]
    )
    table["_speed_3d_um_per_min"] = (
        table["_step_distance_3d_um"]
        / table["_delta_time_minutes"]
    )

    aliases = {
        "time": time_col,
        "track": track_col,
        "z": z_col,
        "y": y_col,
        "x": x_col,
    }
    return table, aliases

def summarise_tracks(
    detections: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    time_col = aliases["time"]
    track_col = aliases["track"]

    records: list[dict[str, object]] = []

    for track_id, group in detections.groupby(track_col):
        group = group.sort_values(time_col).copy()

        signed = group["_signed_distance_to_injury_um"].to_numpy(float)
        valid_signed = signed[np.isfinite(signed)]

        approach_per_frame = group[
            "_injury_approach_velocity_um_per_frame"
        ].to_numpy(float)
        approach_per_min = group[
            "_injury_approach_velocity_um_per_min"
        ].to_numpy(float)

        toward_values_per_frame = approach_per_frame[
            np.isfinite(approach_per_min)
            & (approach_per_min > DIRECTION_TOLERANCE_UM_PER_MIN)
        ]
        away_values_per_frame = -approach_per_frame[
            np.isfinite(approach_per_min)
            & (approach_per_min < -DIRECTION_TOLERANCE_UM_PER_MIN)
        ]
        toward_values_per_min = approach_per_min[
            np.isfinite(approach_per_min)
            & (approach_per_min > DIRECTION_TOLERANCE_UM_PER_MIN)
        ]
        away_values_per_min = -approach_per_min[
            np.isfinite(approach_per_min)
            & (approach_per_min < -DIRECTION_TOLERANCE_UM_PER_MIN)
        ]

        step_valid = np.isfinite(approach_per_min)
        inside = group["_inside_injury"].to_numpy(bool)
        speeds_per_frame = group["_speed_3d_um_per_frame"].to_numpy(float)
        speeds_per_min = group["_speed_3d_um_per_min"].to_numpy(float)

        inside_speed_per_frame = speeds_per_frame[
            inside & np.isfinite(speeds_per_frame)
        ]
        outside_speed_per_frame = speeds_per_frame[
            (~inside) & np.isfinite(speeds_per_frame)
        ]
        inside_speed_per_min = speeds_per_min[
            inside & np.isfinite(speeds_per_min)
        ]
        outside_speed_per_min = speeds_per_min[
            (~inside) & np.isfinite(speeds_per_min)
        ]

        entries = int(group["_entry_event"].sum())
        exits = int(group["_exit_event"].sum())

        entry_rows = group.loc[group["_entry_event"]]
        exit_rows = group.loc[group["_exit_event"]]

        start_frame = float(group[time_col].iloc[0])
        end_frame = float(group[time_col].iloc[-1])
        duration_frames = end_frame - start_frame
        duration_seconds = float(group["_elapsed_time_seconds"].iloc[-1])
        duration_minutes = duration_seconds / 60.0

        longest_inside_seconds = longest_true_duration(
            inside,
            group["_elapsed_time_seconds"].to_numpy(float),
        )

        record = {
            "track_id": track_id,
            "n_detections": len(group),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "track_duration_frames": duration_frames,
            "track_duration_seconds": duration_seconds,
            "track_duration_minutes": duration_minutes,
            "start_signed_distance_um": (
                float(valid_signed[0]) if len(valid_signed) else np.nan
            ),
            "end_signed_distance_um": (
                float(valid_signed[-1]) if len(valid_signed) else np.nan
            ),
            "net_approach_um": (
                float(valid_signed[0] - valid_signed[-1])
                if len(valid_signed)
                else np.nan
            ),
            "min_abs_distance_to_injury_um": (
                float(np.min(np.abs(valid_signed)))
                if len(valid_signed)
                else np.nan
            ),
            "mean_signed_distance_um": (
                float(np.mean(valid_signed))
                if len(valid_signed)
                else np.nan
            ),
            "fraction_detections_inside": float(np.mean(inside)),
            "ever_inside": bool(np.any(inside)),
            "starts_inside": bool(inside[0]),
            "ends_inside": bool(inside[-1]),
            "entry_count": entries,
            "exit_count": exits,
            "first_entry_frame": (
                float(entry_rows[time_col].iloc[0])
                if len(entry_rows)
                else np.nan
            ),
            "first_exit_frame": (
                float(exit_rows[time_col].iloc[0])
                if len(exit_rows)
                else np.nan
            ),
            "first_entry_time_seconds": (
                float(entry_rows["_elapsed_time_seconds"].iloc[0])
                if len(entry_rows)
                else np.nan
            ),
            "first_exit_time_seconds": (
                float(exit_rows["_elapsed_time_seconds"].iloc[0])
                if len(exit_rows)
                else np.nan
            ),
            "first_entry_time_minutes": (
                float(entry_rows["_elapsed_time_minutes"].iloc[0])
                if len(entry_rows)
                else np.nan
            ),
            "first_exit_time_minutes": (
                float(exit_rows["_elapsed_time_minutes"].iloc[0])
                if len(exit_rows)
                else np.nan
            ),
            "longest_inside_run_frames": longest_true_run(inside),
            "longest_inside_duration_seconds": longest_inside_seconds,
            "longest_inside_duration_minutes": (
                longest_inside_seconds / 60.0
            ),
            # Legacy frame-normalised values retained for audit/comparison.
            "mean_approach_velocity_um_per_frame": safe_mean(
                approach_per_frame
            ),
            "mean_toward_velocity_um_per_frame": safe_mean(
                toward_values_per_frame
            ),
            "mean_away_velocity_um_per_frame": safe_mean(
                away_values_per_frame
            ),
            # Physical-time values used for corrected biological comparison.
            "mean_approach_velocity_um_per_min": safe_mean(
                approach_per_min
            ),
            "mean_toward_velocity_um_per_min": safe_mean(
                toward_values_per_min
            ),
            "mean_away_velocity_um_per_min": safe_mean(
                away_values_per_min
            ),
            "fraction_steps_toward": (
                float(np.mean(group.loc[step_valid, "_toward_step"]))
                if step_valid.sum()
                else np.nan
            ),
            "fraction_steps_away": (
                float(np.mean(group.loc[step_valid, "_away_step"]))
                if step_valid.sum()
                else np.nan
            ),
            "mean_speed_inside_um_per_frame": safe_mean(
                inside_speed_per_frame
            ),
            "mean_speed_outside_um_per_frame": safe_mean(
                outside_speed_per_frame
            ),
            "mean_speed_inside_um_per_min": safe_mean(
                inside_speed_per_min
            ),
            "mean_speed_outside_um_per_min": safe_mean(
                outside_speed_per_min
            ),
            "path_length_inside_um": float(
                np.nansum(
                    group.loc[
                        group["_inside_injury"],
                        "_step_distance_3d_um",
                    ]
                )
            ),
            "path_length_outside_um": float(
                np.nansum(
                    group.loc[
                        ~group["_inside_injury"],
                        "_step_distance_3d_um",
                    ]
                )
            ),
        }
        records.append(record)

    return pd.DataFrame(records)

def summarise_fish(
    fish_name: str,
    track_summary: pd.DataFrame,
) -> dict[str, object]:
    result: dict[str, object] = {
        "block_name": fish_name,
        "n_tracks_injury_analysis": len(track_summary),
        "fraction_tracks_ever_inside": float(
            track_summary["ever_inside"].mean()
        ),
        "fraction_tracks_starting_inside": float(
            track_summary["starts_inside"].mean()
        ),
        "fraction_tracks_ending_inside": float(
            track_summary["ends_inside"].mean()
        ),
        "fraction_tracks_entering": float(
            track_summary["entry_count"].gt(0).mean()
        ),
        "fraction_tracks_exiting": float(
            track_summary["exit_count"].gt(0).mean()
        ),
        "total_entries": int(track_summary["entry_count"].sum()),
        "total_exits": int(track_summary["exit_count"].sum()),
        "entries_per_track": float(
            track_summary["entry_count"].mean()
        ),
        "exits_per_track": float(
            track_summary["exit_count"].mean()
        ),
        "net_entry_exit_flux_per_track": float(
            (
                track_summary["entry_count"]
                - track_summary["exit_count"]
            ).mean()
        ),
        "mean_longest_inside_run_frames": float(
            track_summary["longest_inside_run_frames"].mean()
        ),
        "median_longest_inside_run_frames": float(
            track_summary["longest_inside_run_frames"].median()
        ),
    }

    numeric_track_features = [
        column
        for column in track_summary.columns
        if column
        not in {
            "track_id",
            "ever_inside",
            "starts_inside",
            "ends_inside",
            "entry_count",
            "exit_count",
        }
        and pd.api.types.is_numeric_dtype(track_summary[column])
    ]

    for feature in numeric_track_features:
        values = pd.to_numeric(
            track_summary[feature], errors="coerce"
        )
        result[f"mean_{feature}"] = float(values.mean())
        result[f"median_{feature}"] = float(values.median())

    return result


def model_a_fish_column(df: pd.DataFrame) -> str:
    return detect_column(df, FISH_COLUMN_CANDIDATES, "Model A fish")


def merge_with_model_a(
    model_a_path: Path,
    fish_summary: pd.DataFrame,
    output_path: Path,
    same_cohort_model_a_path: Path,
    cohort_report_path: Path,
) -> tuple[list[str], list[str]]:
    """Merge injury features only for fish with a completed manual ROI."""
    model_a = pd.read_csv(model_a_path, low_memory=False)
    fish_col = model_a_fish_column(model_a)

    model_a = model_a.copy()
    model_a[fish_col] = model_a[fish_col].astype(str).str.strip()

    fish_summary = fish_summary.copy()
    fish_summary["block_name"] = (
        fish_summary["block_name"].astype(str).str.strip()
    )

    if model_a[fish_col].duplicated().any():
        duplicates = (
            model_a.loc[model_a[fish_col].duplicated(), fish_col]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Duplicate fish rows in Model A table {model_a_path}: "
            f"{duplicates}"
        )

    if fish_summary["block_name"].duplicated().any():
        duplicates = (
            fish_summary.loc[
                fish_summary["block_name"].duplicated(),
                "block_name",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Duplicate fish rows in injury summary: {duplicates}"
        )

    if USE_COMPACT_MODEL_FEATURES:
        requested = COMPACT_TRACK_AGGREGATES + COMPACT_FISH_COUNTS
    else:
        requested = [
            column
            for column in fish_summary.columns
            if column != "block_name"
            and pd.api.types.is_numeric_dtype(fish_summary[column])
        ]

    available = [
        feature
        for feature in requested
        if feature in fish_summary.columns
    ]
    if not available:
        raise ValueError(
            "No requested injury predictors were available in the "
            "fish-level injury summary."
        )

    rename_map = {
        feature: f"fish_mean__injury_{feature}"
        for feature in available
    }

    injury_subset = fish_summary[
        ["block_name"] + available
    ].rename(columns=rename_map)

    injury_fish = set(injury_subset["block_name"])
    model_fish = set(model_a[fish_col])

    included_fish = [
        fish
        for fish in model_a[fish_col].tolist()
        if fish in injury_fish
    ]
    excluded_fish = [
        fish
        for fish in model_a[fish_col].tolist()
        if fish not in injury_fish
    ]
    extra_injury_fish = sorted(injury_fish - model_fish)

    if not included_fish:
        raise ValueError(
            f"No fish overlap between Model A table {model_a_path} "
            "and the injury-feature table."
        )

    same_cohort_model_a = model_a[
        model_a[fish_col].isin(included_fish)
    ].copy()

    merged = same_cohort_model_a.merge(
        injury_subset,
        left_on=fish_col,
        right_on="block_name",
        how="inner",
        validate="one_to_one",
    )

    if fish_col != "block_name":
        merged = merged.drop(columns=["block_name"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    same_cohort_model_a_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cohort_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    same_cohort_model_a.to_csv(
        same_cohort_model_a_path,
        index=False,
    )
    merged.to_csv(output_path, index=False)

    report_rows: list[dict[str, object]] = []
    for fish in model_a[fish_col].tolist():
        included = fish in injury_fish
        report_rows.append(
            {
                "fish_name": fish,
                "present_in_model_a": True,
                "injury_roi_features_available": included,
                "included_in_same_cohort_comparison": included,
                "reason": (
                    "included"
                    if included
                    else "no completed injury ROI/features"
                ),
            }
        )

    for fish in extra_injury_fish:
        report_rows.append(
            {
                "fish_name": fish,
                "present_in_model_a": False,
                "injury_roi_features_available": True,
                "included_in_same_cohort_comparison": False,
                "reason": "injury features exist but fish is absent from this Model A table",
            }
        )

    pd.DataFrame(report_rows).to_csv(
        cohort_report_path,
        index=False,
    )

    print(
        f"[COHORT] {model_a_path.name}: "
        f"included={len(included_fish)}, "
        f"excluded={len(excluded_fish)}",
        flush=True,
    )
    if excluded_fish:
        print(
            "[COHORT] Excluded because no completed ROI/features: "
            + ", ".join(excluded_fish),
            flush=True,
        )

    return list(rename_map.values()), excluded_fish


def process_channel(
    channel: str,
    track_names: list[str],
    model_a_targets: list[tuple[str, Path]],
    fish_names: list[str],
    frame_intervals_seconds: dict[str, float],
    audit_rows: list[dict[str, object]],
) -> None:
    channel_dir = OUTPUT_ROOT / channel
    detection_dir = channel_dir / "per_detection"
    track_dir = channel_dir / "per_track"

    detection_dir.mkdir(parents=True, exist_ok=True)
    track_dir.mkdir(parents=True, exist_ok=True)

    fish_records: list[dict[str, object]] = []

    for fish_name in fish_names:
        block_dir = BLOCKS_ROOT / fish_name
        mask_path = (
            ANNOTATION_ROOT
            / fish_name
            / "manual_injury_roi_mask_YX.tif"
        )

        try:
            if not block_dir.exists():
                raise FileNotFoundError(
                    f"Block folder does not exist: {block_dir}"
                )
            if not mask_path.exists():
                audit_rows.append(
                    {
                        "fish_name": fish_name,
                        "channel": channel,
                        "status": "skipped_no_manual_roi",
                        "mask_path": str(mask_path),
                        "reason": (
                            "No completed manual injury ROI; fish is excluded "
                            "from both same-cohort Model A and Model A + injury."
                        ),
                    }
                )
                print(
                    f"[SKIP] {channel}: {fish_name}: no manual injury ROI",
                    flush=True,
                )
                continue

            track_path = find_track_file(
                block_dir,
                candidate_names=track_names,
                channel=channel,
            )
            if track_path is None:
                raise FileNotFoundError(
                    f"No {channel} track CSV found under {block_dir}"
                )

            if fish_name not in frame_intervals_seconds:
                raise ValueError(
                    "No valid time_interval_seconds in metadata for "
                    f"{fish_name}"
                )
            time_interval_seconds = frame_intervals_seconds[fish_name]

            xy_um = XY_OVERRIDES_UM.get(
                fish_name,
                DEFAULT_XY_UM,
            )
            z_um = Z_OVERRIDES_UM.get(
                fish_name,
                DEFAULT_Z_UM,
            )

            mask = load_mask(mask_path)
            tracks = pd.read_csv(track_path, low_memory=False)

            detections, aliases = enrich_detections(
                tracks=tracks,
                mask=mask,
                xy_um=xy_um,
                z_um=z_um,
                time_interval_seconds=time_interval_seconds,
            )
            track_summary = summarise_tracks(
                detections,
                aliases,
            )
            fish_summary = summarise_fish(
                fish_name,
                track_summary,
            )
            fish_records.append(fish_summary)

            detections.to_csv(
                detection_dir
                / f"{fish_name}__{channel}_injury_detections.csv",
                index=False,
            )
            track_summary.to_csv(
                track_dir
                / f"{fish_name}__{channel}_injury_tracks.csv",
                index=False,
            )

            audit_rows.append(
                {
                    "fish_name": fish_name,
                    "channel": channel,
                    "status": "ok",
                    "track_path": str(track_path),
                    "mask_path": str(mask_path),
                    "time_interval_seconds": time_interval_seconds,
                    "minutes_per_frame": time_interval_seconds / 60.0,
                    "n_detection_rows": len(detections),
                    "n_tracks": len(track_summary),
                }
            )

            print(
                f"[DONE] {channel}: {fish_name}, "
                f"tracks={len(track_summary)}",
                flush=True,
            )

        except Exception as exc:
            audit_rows.append(
                {
                    "fish_name": fish_name,
                    "channel": channel,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(
                f"[WARN] {channel}: {fish_name}: {exc}",
                flush=True,
            )

    if not fish_records:
        raise RuntimeError(
            f"No {channel} fish completed successfully."
        )

    fish_summary_table = pd.DataFrame(fish_records).sort_values(
        "block_name"
    )
    fish_summary_path = (
        channel_dir / "injury_features_fish_level_all.csv"
    )
    fish_summary_table.to_csv(
        fish_summary_path,
        index=False,
    )

    merged_outputs: dict[str, str] = {}
    same_cohort_baseline_outputs: dict[str, str] = {}
    cohort_report_outputs: dict[str, str] = {}
    added_columns_by_target: dict[str, list[str]] = {}
    excluded_fish_by_target: dict[str, list[str]] = {}

    for target_name, model_a_path in model_a_targets:
        merged_path = (
            channel_dir
            / f"constrained_fish_level_{target_name}_plus_injury.csv"
        )
        same_cohort_path = (
            channel_dir
            / f"constrained_fish_level_{target_name}_same_injury_cohort.csv"
        )
        cohort_report_path = (
            channel_dir
            / f"injury_cohort_report_{target_name}.csv"
        )

        added_columns, excluded_fish = merge_with_model_a(
            model_a_path=model_a_path,
            fish_summary=fish_summary_table,
            output_path=merged_path,
            same_cohort_model_a_path=same_cohort_path,
            cohort_report_path=cohort_report_path,
        )

        merged_outputs[target_name] = str(merged_path)
        same_cohort_baseline_outputs[target_name] = str(
            same_cohort_path
        )
        cohort_report_outputs[target_name] = str(
            cohort_report_path
        )
        added_columns_by_target[target_name] = added_columns
        excluded_fish_by_target[target_name] = excluded_fish

        # Preserve the previous generic augmented filename for compatibility.
        if (
            target_name == "musc_model_a"
            or target_name == "macrophage_outside_boundary_model_b"
        ):
            legacy_path = (
                channel_dir
                / "constrained_fish_level_model_a_plus_injury.csv"
            )
            pd.read_csv(merged_path, low_memory=False).to_csv(
                legacy_path,
                index=False,
            )

        print(
            f"[SAVED] Same-cohort Model A: {same_cohort_path}",
            flush=True,
        )
        print(
            f"[SAVED] Model A + injury: {merged_path}",
            flush=True,
        )

    definition = {
        "channel": channel,
        "source_model_a_tables": {
            target_name: str(model_a_path)
            for target_name, model_a_path in model_a_targets
        },
        "same_cohort_model_a_outputs": same_cohort_baseline_outputs,
        "merged_classifier_outputs": merged_outputs,
        "cohort_report_outputs": cohort_report_outputs,
        "excluded_fish_by_target": excluded_fish_by_target,
        "manual_annotation_root": str(ANNOTATION_ROOT),
        "macrophage_track_mode": (
            "all" if channel == "macrophage" else None
        ),
        "frame_interval_metadata": str(FRAME_INTERVAL_METADATA),
        "direction_tolerance_um_per_min": (
            DIRECTION_TOLERANCE_UM_PER_MIN
        ),
        "physical_time_unit": "minutes",
        "legacy_per_frame_columns_retained_for_audit": True,
        "compact_feature_set": USE_COMPACT_MODEL_FEATURES,
        "added_predictor_columns": added_columns_by_target,
    }
    (
        channel_dir / "injury_model_definition.json"
    ).write_text(
        json.dumps(definition, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    if not BLOCKS_ROOT.exists():
        raise FileNotFoundError(BLOCKS_ROOT)
    if not ANNOTATION_ROOT.exists():
        raise FileNotFoundError(ANNOTATION_ROOT)
    if not FRAME_INTERVAL_METADATA.exists():
        raise FileNotFoundError(FRAME_INTERVAL_METADATA)

    frame_intervals_seconds = load_frame_interval_metadata(
        FRAME_INTERVAL_METADATA
    )

    required_tables = [
        MUSC_MODEL_A_TABLE,
        MACROPHAGE_ALL_MODEL_A_TABLE,
        MACROPHAGE_OUTSIDE_MODEL_A_TABLE,
    ]
    for table_path in required_tables:
        if not table_path.exists():
            raise FileNotFoundError(table_path)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    musc_model = pd.read_csv(MUSC_MODEL_A_TABLE, low_memory=False)
    mac_all_model = pd.read_csv(
        MACROPHAGE_ALL_MODEL_A_TABLE,
        low_memory=False,
    )
    mac_outside_model = pd.read_csv(
        MACROPHAGE_OUTSIDE_MODEL_A_TABLE,
        low_memory=False,
    )

    musc_fish_col = model_a_fish_column(musc_model)
    mac_all_fish_col = model_a_fish_column(mac_all_model)
    mac_outside_fish_col = model_a_fish_column(mac_outside_model)

    musc_fish_names = sorted(
        name for name in musc_model[musc_fish_col]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
        if name not in EXCLUDED_FISH
    )

    macrophage_candidates = (
        set(
            mac_all_model[mac_all_fish_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        | set(
            mac_outside_model[mac_outside_fish_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
    )
    macrophage_fish_names = sorted(
        name
        for name in macrophage_candidates
        if name not in EXCLUDED_FISH
    )

    audit_rows: list[dict[str, object]] = []

    process_channel(
        channel="musc",
        track_names=MUSC_TRACK_NAMES,
        model_a_targets=[
            ("musc_model_a", MUSC_MODEL_A_TABLE),
        ],
        fish_names=musc_fish_names,
        frame_intervals_seconds=frame_intervals_seconds,
        audit_rows=audit_rows,
    )

    process_channel(
        channel="macrophage",
        track_names=MACROPHAGE_ALL_TRACK_NAMES,
        model_a_targets=[
            ("macrophage_all_model_b", MACROPHAGE_ALL_MODEL_A_TABLE),
            (
                "macrophage_outside_boundary_model_b",
                MACROPHAGE_OUTSIDE_MODEL_A_TABLE,
            ),
        ],
        fish_names=macrophage_fish_names,
        frame_intervals_seconds=frame_intervals_seconds,
        audit_rows=audit_rows,
    )

    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_ROOT / "injury_feature_audit.csv",
        index=False,
    )

    print()
    print("============================================================")
    print("MANUAL-INJURY FEATURE EXTRACTION COMPLETE")
    print("============================================================")
    print(
        OUTPUT_ROOT
        / "musc"
        / "constrained_fish_level_musc_model_a_plus_injury.csv"
    )
    print(
        OUTPUT_ROOT
        / "macrophage"
        / "constrained_fish_level_macrophage_all_model_b_plus_injury.csv"
    )
    print(
        OUTPUT_ROOT
        / "macrophage"
        / (
            "constrained_fish_level_macrophage_outside_boundary_"
            "model_b_plus_injury.csv"
        )
    )


if __name__ == "__main__":
    main()
