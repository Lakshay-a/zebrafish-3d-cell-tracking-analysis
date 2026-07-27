#!/usr/bin/env python3
"""Model C = macrophage outside-boundary Model A."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


DEFAULT_XY_UM = 0.7533114346590908
DEFAULT_Z_UM = 1.0

KNOWN_XY_UM = {
    "260420_block03": 0.828642578125,
    "260427_block02": 0.7533114346590908,
    "260427_block03": 0.7533114346590908,
    "260427_block06": 0.7533114346590908,
    "260427_block07": 0.7533114346590908,
    "260427_block08": 0.7533114346590908,
    "260428_block02": 0.7533114346590908,
    "260428_block03": 0.7533114346590908,
    "260428_block04": 0.7533114346590908,
    "260511_block01": 0.7533114346590908,
    "260511_block02": 0.7533114346590908,
    "260511_block03": 0.7533114346590908,
    "260511_block05": 0.7533114346590908,
    "260511_block06": 0.7533114346590908,
    "260511_block07": 0.7533114346590908,
    "20240325_block02": 0.3452677408854165,
    "20240422_block01": 0.4143212890625,
    "20240422_block06": 0.4143212890625,
    "20240422_block08": 0.4143212890625,
    "20240422_block11": 0.4143212890625,
}

MASK_NAMES = [
    "macrophage_cluster_core_TYX.tif",
    "cluster_core_TYX.tif",
    "macrophage_cluster_mask_TYX.tif",
    "cluster_mask_TYX.tif",
    "macrophage_cluster_tracking_exclusion_mask_TYX.tif",
    "cluster_tracking_exclusion_mask_TYX.tif",
    "macrophage_cluster_core_TZYX.tif",
    "cluster_core_TZYX.tif",
]

TRACK_RELATIVE_PATHS = [
    Path("macrophage_tracking_outputs_all/macrophage_tracks_lap.csv"),
    Path("macrophage_tracking_outputs_all/macrophage_tracks_lap_good_filtered.csv"),
]

TRACK_NAMES = [
    "macrophage_tracks_lap.csv",
    "macrophage_tracks_lap_good_filtered.csv",
]

FISH_COLUMNS = ["block_name", "fish_id", "block", "source_block", "sample_id"]
TIME_COLUMNS = ["time", "frame", "t", "timepoint"]
TRACK_COLUMNS = ["track_id", "global_track_id", "cell_track_id"]
REGION_COLUMNS = ["cluster_region_class", "region_class"]
INSIDE_COLUMNS = ["inside_cluster", "overlap_cluster_mask"]
BOUNDARY_COLUMNS = ["near_cluster_boundary", "on_cluster_boundary"]
VOLUME_UM3_COLUMNS = ["volume_um3", "mean_volume_um3"]
VOLUME_VOXEL_COLUMNS = ["volume_voxels", "voxel_count", "size_voxels"]
DISTANCE_UM_COLUMNS = ["distance_to_cluster_boundary_um"]
DISTANCE_PX_COLUMNS = ["distance_to_cluster_boundary_px"]

COMPACT_FEATURES = [
    "cluster_area_auc_um2_minute",
    "max_cluster_expansion_rate_um2_per_minute",
    "max_cluster_contraction_rate_um2_per_minute",
    "near_cluster_volume_auc_um3_minute",
    "max_macrophage_accumulation_rate_um3_per_minute",
    "max_macrophage_dispersal_rate_um3_per_minute",
    "mean_boundary_object_count",
    "net_cluster_flux",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks-root", required=True)
    p.add_argument("--model-a-table", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--feature-set", choices=["compact", "full"], default="compact")
    p.add_argument("--rolling-window", type=int, default=5)
    p.add_argument("--phase-fraction", type=float, default=0.33)
    p.add_argument("--frame-interval-minutes", type=float, default=None)
    p.add_argument("--metadata-file", default="block_metadata.csv")
    p.add_argument("--time-interval-col", default="time_interval_seconds")
    p.add_argument("--near-distance-um", type=float, default=None)
    p.add_argument("--pixel-size-csv", default=None)
    p.add_argument("--model-a-fish-col", default=None)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def detect_col(
    df: pd.DataFrame,
    candidates: list[str],
    role: str,
    explicit: str | None = None,
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} column '{explicit}' not found")
        return explicit

    lower = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise ValueError(f"Could not detect {role} column in {list(df.columns)}")
    return None


def load_time_intervals(path: Path, interval_col: str) -> dict[str, float]:
    if not path.exists():
        return {}
    metadata = pd.read_csv(path, low_memory=False)
    if "block_name" not in metadata.columns or interval_col not in metadata.columns:
        return {}
    table = metadata[["block_name", interval_col]].copy()
    table["block_name"] = table["block_name"].astype(str).str.strip()
    table[interval_col] = pd.to_numeric(table[interval_col], errors="coerce")
    table = table.dropna(subset=[interval_col])
    table = table[table[interval_col] > 0]
    return dict(zip(table["block_name"], table[interval_col] / 60.0))


def bool_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(False, index=series.index)
    numeric_rows = numeric.notna()
    result.loc[numeric_rows] = numeric.loc[numeric_rows].ne(0)
    text = series.astype(str).str.strip().str.lower()
    result.loc[~numeric_rows] = text.loc[~numeric_rows].isin(
        {"true", "t", "yes", "y", "1"}
    )
    return result


def find_mask(block_dir: Path) -> Path | None:
    matches = [p for p in block_dir.rglob("*") if p.is_file() and p.name in MASK_NAMES]
    if not matches:
        return None

    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if "core" in name:
            rank = 0
        elif "exclusion" not in name:
            rank = 1
        else:
            rank = 2
        return rank, len(path.parts), str(path)

    return sorted(matches, key=priority)[0]


def find_tracks(block_dir: Path) -> Path | None:
    for relative in TRACK_RELATIVE_PATHS:
        candidate = block_dir / relative
        if candidate.exists():
            return candidate

    matches = [
        p
        for p in block_dir.rglob("*")
        if p.is_file()
        and p.name in TRACK_NAMES
        and "outside_boundary" not in str(p).lower()
        and "outside-boundary" not in str(p).lower()
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: (p.name != "macrophage_tracks_lap.csv", len(p.parts)))
    return matches[0]


def load_pixel_sizes(csv_path: str | None) -> dict[str, tuple[float, float]]:
    mapping = {block: (xy, DEFAULT_Z_UM) for block, xy in KNOWN_XY_UM.items()}
    if csv_path is None:
        return mapping

    table = pd.read_csv(csv_path)
    block_col = detect_col(table, ["block_name", "fish_id", "block"], "block")
    xy_col = detect_col(
        table,
        ["xy_pixel_size_um", "xy_um", "pixel_size_xy_um"],
        "XY pixel size",
    )
    z_col = detect_col(
        table,
        ["z_step_size_um", "z_um", "pixel_size_z_um"],
        "Z step",
        required=False,
    )
    for _, row in table.iterrows():
        block = str(row[block_col]).strip()
        xy = float(row[xy_col])
        z = float(row[z_col]) if z_col and pd.notna(row[z_col]) else DEFAULT_Z_UM
        mapping[block] = (xy, z)
    return mapping


def region_state(
    tracks: pd.DataFrame,
    xy_um: float,
    near_distance_um: float | None,
) -> pd.Series:
    inside = pd.Series(False, index=tracks.index)
    boundary = pd.Series(False, index=tracks.index)

    region_col = detect_col(
        tracks, REGION_COLUMNS, "region", required=False
    )
    if region_col:
        text = tracks[region_col].astype(str).str.lower()
        inside |= text.str.contains("inside", na=False)
        boundary |= text.str.contains("boundary|near", regex=True, na=False)

    for col in INSIDE_COLUMNS:
        if col in tracks.columns:
            inside |= bool_series(tracks[col])

    for col in BOUNDARY_COLUMNS:
        if col in tracks.columns:
            boundary |= bool_series(tracks[col])

    if near_distance_um is not None:
        distance_um_col = detect_col(
            tracks, DISTANCE_UM_COLUMNS, "boundary distance", required=False
        )
        distance_px_col = detect_col(
            tracks, DISTANCE_PX_COLUMNS, "boundary distance", required=False
        )
        if distance_um_col:
            distance = pd.to_numeric(tracks[distance_um_col], errors="coerce").abs()
            boundary |= distance.le(near_distance_um)
        elif distance_px_col:
            distance = (
                pd.to_numeric(tracks[distance_px_col], errors="coerce").abs() * xy_um
            )
            boundary |= distance.le(near_distance_um)

    boundary &= ~inside
    state = pd.Series(0, index=tracks.index, dtype=int)
    state.loc[boundary] = 1
    state.loc[inside] = 2
    return state


def volume_um3(tracks: pd.DataFrame, xy_um: float, z_um: float) -> pd.Series:
    for col in VOLUME_UM3_COLUMNS:
        if col in tracks.columns:
            values = pd.to_numeric(tracks[col], errors="coerce")
            if values.notna().any():
                return values

    for col in VOLUME_VOXEL_COLUMNS:
        if col in tracks.columns:
            return (
                pd.to_numeric(tracks[col], errors="coerce")
                * xy_um
                * xy_um
                * z_um
            )

    raise ValueError(
        "No macrophage volume column found. Expected one of "
        f"{VOLUME_UM3_COLUMNS + VOLUME_VOXEL_COLUMNS}"
    )


def mask_series(
    mask_path: Path,
    xy_um: float,
    z_um: float,
    track_times: pd.Series,
) -> tuple[pd.DataFrame, str]:
    mask = np.asarray(tiff.imread(mask_path)).astype(bool)
    if mask.ndim == 2:
        mask = mask[None, ...]

    numeric_times = pd.to_numeric(track_times, errors="coerce").dropna()
    one_based = (
        not numeric_times.empty
        and int(numeric_times.min()) == 1
        and int(numeric_times.max()) <= mask.shape[0]
    )
    time = (
        np.arange(1, mask.shape[0] + 1)
        if one_based
        else np.arange(mask.shape[0])
    )

    if mask.ndim == 3:  # T,Y,X
        pixels = mask.reshape(mask.shape[0], -1).sum(axis=1)
        return (
            pd.DataFrame(
                {
                    "time": time,
                    "cluster_mask_pixel_count": pixels,
                    "cluster_projected_area_um2": pixels * xy_um * xy_um,
                }
            ),
            "TYX_projected_area",
        )

    if mask.ndim == 4:  # T,Z,Y,X
        voxels = mask.reshape(mask.shape[0], -1).sum(axis=1)
        projected = mask.any(axis=1)
        pixels = projected.reshape(projected.shape[0], -1).sum(axis=1)
        return (
            pd.DataFrame(
                {
                    "time": time,
                    "cluster_mask_voxel_count": voxels,
                    "cluster_volume_um3": voxels * xy_um * xy_um * z_um,
                    "cluster_mask_pixel_count": pixels,
                    "cluster_projected_area_um2": pixels * xy_um * xy_um,
                }
            ),
            "TZYX_true_volume_and_projected_area",
        )

    raise ValueError(f"Unsupported mask shape {mask.shape}")


def event_table(
    tracks: pd.DataFrame,
    time_col: str,
    track_col: str,
) -> pd.DataFrame:
    events = []
    for _, group in tracks.groupby(track_col):
        group = group.sort_values(time_col)
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy()
        states = group["_cluster_state"].to_numpy(int)
        valid = np.isfinite(times)
        times = times[valid].astype(int)
        states = states[valid]

        for i in range(1, len(times)):
            previous, current = states[i - 1], states[i]
            entry = int(previous == 0 and current >= 1)
            exit_ = int(previous >= 1 and current == 0)
            core_entry = int(previous < 2 and current == 2)
            core_exit = int(previous == 2 and current < 2)
            if entry or exit_ or core_entry or core_exit:
                events.append(
                    {
                        "time": int(times[i]),
                        "cluster_entries": entry,
                        "cluster_exits": exit_,
                        "cluster_core_entries": core_entry,
                        "cluster_core_exits": core_exit,
                    }
                )

    columns = [
        "time",
        "cluster_entries",
        "cluster_exits",
        "cluster_core_entries",
        "cluster_core_exits",
    ]
    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(events).groupby("time", as_index=False).sum()


def rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) < window:
        return result
    x = np.arange(window, dtype=float)
    for end in range(window - 1, len(values)):
        y = values[end - window + 1 : end + 1]
        valid = np.isfinite(y)
        if valid.sum() >= 2:
            result[end] = np.polyfit(x[valid], y[valid], 1)[0]
    return result


def linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x[valid], values[valid], 1)[0])


def auc(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[valid]
    y = values[valid]
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def finite_max(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else default


def finite_min(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.min()) if len(values) else default


def summarise(
    block_name: str,
    ts: pd.DataFrame,
    window: int,
    phase_fraction: float,
    xy_um: float,
    z_um: float,
    mask_mode: str,
    mask_path: Path,
    tracks_path: Path,
    frame_interval_minutes: float | None,
) -> dict[str, object]:
    ts = ts.sort_values("time").reset_index(drop=True)
    n = len(ts)
    phase_n = min(n, max(2, math.ceil(n * phase_fraction)))

    area = ts["cluster_projected_area_um2"].to_numpy(float)
    volume = ts["near_cluster_macrophage_volume_um3"].to_numpy(float)
    boundary_count = ts["boundary_object_count"].to_numpy(float)
    entries = ts["cluster_entries"].to_numpy(float)
    exits = ts["cluster_exits"].to_numpy(float)

    area_rate = rolling_slope(area, window)
    volume_rate = rolling_slope(volume, window)
    entry_window = pd.Series(entries).rolling(window, min_periods=1).sum().to_numpy()
    exit_window = pd.Series(exits).rolling(window, min_periods=1).sum().to_numpy()

    result: dict[str, object] = {
        "block_name": block_name,
        "n_frames": n,
        "xy_pixel_size_um": xy_um,
        "z_step_size_um": z_um,
        "mask_mode": mask_mode,
        "mask_path": str(mask_path),
        "tracks_path": str(tracks_path),

        "mean_cluster_projected_area_um2": float(np.nanmean(area)),
        "max_cluster_projected_area_um2": finite_max(area),
        "cluster_area_auc_um2_frame": auc(area),
        "net_cluster_area_change_um2": float(area[-1] - area[0]),
        "early_cluster_area_slope_um2_per_frame": linear_slope(area[:phase_n]),
        "late_cluster_area_slope_um2_per_frame": linear_slope(area[-phase_n:]),
        "max_cluster_expansion_rate_um2_per_frame": max(0.0, finite_max(area_rate)),
        "max_cluster_contraction_rate_um2_per_frame": max(
            0.0, -finite_min(area_rate)
        ),

        "mean_near_cluster_macrophage_volume_um3": float(np.nanmean(volume)),
        "max_near_cluster_macrophage_volume_um3": finite_max(volume),
        "near_cluster_volume_auc_um3_frame": auc(volume),
        "net_near_cluster_volume_change_um3": float(volume[-1] - volume[0]),
        "early_accumulation_slope_um3_per_frame": linear_slope(volume[:phase_n]),
        "late_accumulation_slope_um3_per_frame": linear_slope(volume[-phase_n:]),
        "max_macrophage_accumulation_rate_um3_per_frame": max(
            0.0, finite_max(volume_rate)
        ),
        "max_macrophage_dispersal_rate_um3_per_frame": max(
            0.0, -finite_min(volume_rate)
        ),

        "mean_boundary_object_count": float(np.nanmean(boundary_count)),
        "max_boundary_object_count": finite_max(boundary_count),
        "boundary_object_count_auc_frame": auc(boundary_count),

        "total_cluster_entries": int(np.nansum(entries)),
        "total_cluster_exits": int(np.nansum(exits)),
        "total_cluster_core_entries": int(ts["cluster_core_entries"].sum()),
        "total_cluster_core_exits": int(ts["cluster_core_exits"].sum()),
        "peak_entries_per_window": finite_max(entry_window),
        "peak_exits_per_window": finite_max(exit_window),
        "net_cluster_flux": int(np.nansum(entries) - np.nansum(exits)),
    }

    if "cluster_volume_um3" in ts.columns:
        cluster_volume = ts["cluster_volume_um3"].to_numpy(float)
        result.update(
            {
                "mean_cluster_volume_um3": float(np.nanmean(cluster_volume)),
                "max_cluster_volume_um3": finite_max(cluster_volume),
                "cluster_volume_auc_um3_frame": auc(cluster_volume),
            }
        )

    if frame_interval_minutes is not None:
        if frame_interval_minutes <= 0:
            raise ValueError("Frame interval must be positive")
        result["frame_interval_minutes"] = float(frame_interval_minutes)
        result["cluster_area_auc_um2_minute"] = result["cluster_area_auc_um2_frame"] * frame_interval_minutes
        result["near_cluster_volume_auc_um3_minute"] = result["near_cluster_volume_auc_um3_frame"] * frame_interval_minutes
        result["boundary_object_count_auc_minute"] = result["boundary_object_count_auc_frame"] * frame_interval_minutes
        for name, value in list(result.items()):
            if name.endswith("_per_frame") and isinstance(value, (int, float)):
                result[name.replace("_per_frame", "_per_minute")] = (
                    value / frame_interval_minutes
                )

    return result


def process_block(
    block_dir: Path,
    per_block_dir: Path,
    xy_um: float,
    z_um: float,
    window: int,
    phase_fraction: float,
    frame_interval_minutes: float | None,
    near_distance_um: float | None,
) -> tuple[dict[str, object], dict[str, object]]:
    mask_path = find_mask(block_dir)
    tracks_path = find_tracks(block_dir)
    if mask_path is None:
        raise FileNotFoundError("Cluster mask not found")
    if tracks_path is None:
        raise FileNotFoundError("Macrophage-all LAP tracks not found")

    tracks = pd.read_csv(tracks_path, low_memory=False)
    time_col = detect_col(tracks, TIME_COLUMNS, "time")
    track_col = detect_col(tracks, TRACK_COLUMNS, "track")

    tracks = tracks.copy()
    tracks[time_col] = pd.to_numeric(tracks[time_col], errors="coerce")
    tracks = tracks.dropna(subset=[time_col, track_col])
    tracks[time_col] = tracks[time_col].astype(int)
    tracks["_cluster_state"] = region_state(
        tracks, xy_um=xy_um, near_distance_um=near_distance_um
    )
    tracks["_volume_um3"] = volume_um3(tracks, xy_um=xy_um, z_um=z_um)

    mask_ts, mask_mode = mask_series(
        mask_path, xy_um=xy_um, z_um=z_um, track_times=tracks[time_col]
    )

    rows = []
    for time_value, group in tracks.groupby(time_col):
        near = group["_cluster_state"].ge(1)
        rows.append(
            {
                "time": int(time_value),
                "near_cluster_macrophage_volume_um3": float(
                    group.loc[near, "_volume_um3"].sum(min_count=1)
                ),
                "near_cluster_object_count": int(near.sum()),
                "boundary_object_count": int(group["_cluster_state"].eq(1).sum()),
                "inside_cluster_object_count": int(group["_cluster_state"].eq(2).sum()),
            }
        )
    object_ts = pd.DataFrame(rows)
    events = event_table(tracks, time_col=time_col, track_col=track_col)

    ts = mask_ts.merge(object_ts, on="time", how="left")
    ts = ts.merge(events, on="time", how="left")

    zero_columns = [
        "near_cluster_macrophage_volume_um3",
        "near_cluster_object_count",
        "boundary_object_count",
        "inside_cluster_object_count",
        "cluster_entries",
        "cluster_exits",
        "cluster_core_entries",
        "cluster_core_exits",
    ]
    for col in zero_columns:
        if col not in ts.columns:
            ts[col] = 0
        ts[col] = pd.to_numeric(ts[col], errors="coerce").fillna(0)

    ts["cluster_area_change_um2_per_frame"] = (
        ts["cluster_projected_area_um2"].diff()
    )
    ts["near_cluster_volume_change_um3_per_frame"] = (
        ts["near_cluster_macrophage_volume_um3"].diff()
    )
    ts["cluster_area_rolling_slope_um2_per_frame"] = rolling_slope(
        ts["cluster_projected_area_um2"].to_numpy(float), window
    )
    ts["near_cluster_volume_rolling_slope_um3_per_frame"] = rolling_slope(
        ts["near_cluster_macrophage_volume_um3"].to_numpy(float), window
    )
    ts["entries_per_window"] = (
        ts["cluster_entries"].rolling(window, min_periods=1).sum()
    )
    ts["exits_per_window"] = (
        ts["cluster_exits"].rolling(window, min_periods=1).sum()
    )
    ts["net_cluster_flux_per_frame"] = ts["cluster_entries"] - ts["cluster_exits"]

    per_block_path = (
        per_block_dir / f"{block_dir.name}__cluster_dynamics_time_series.csv"
    )
    ts.to_csv(per_block_path, index=False)

    summary = summarise(
        block_name=block_dir.name,
        ts=ts,
        window=window,
        phase_fraction=phase_fraction,
        xy_um=xy_um,
        z_um=z_um,
        mask_mode=mask_mode,
        mask_path=mask_path,
        tracks_path=tracks_path,
        frame_interval_minutes=frame_interval_minutes,
    )
    audit = {
        "block_name": block_dir.name,
        "status": "ok",
        "mask_path": str(mask_path),
        "tracks_path": str(tracks_path),
        "mask_mode": mask_mode,
        "n_track_rows": len(tracks),
        "n_tracks": tracks[track_col].nunique(),
        "n_frames": len(ts),
        "time_series_output": str(per_block_path),
    }
    return summary, audit


def merge_model_a(
    model_a_path: Path,
    summaries: pd.DataFrame,
    feature_set: str,
    explicit_fish_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    model_a = pd.read_csv(model_a_path, low_memory=False)
    fish_col = detect_col(
        model_a,
        FISH_COLUMNS,
        "Model A fish",
        explicit=explicit_fish_col,
    )

    model_a = model_a.copy()
    model_a[fish_col] = model_a[fish_col].astype(str).str.strip()
    summaries = summaries.copy()
    summaries["block_name"] = summaries["block_name"].astype(str).str.strip()

    if feature_set == "compact":
        selected = [f for f in COMPACT_FEATURES if f in summaries.columns]
        if len(selected) < len(COMPACT_FEATURES):
            fallback = [
                "cluster_area_auc_um2_frame",
                "max_cluster_expansion_rate_um2_per_frame",
                "max_cluster_contraction_rate_um2_per_frame",
                "near_cluster_volume_auc_um3_frame",
                "max_macrophage_accumulation_rate_um3_per_frame",
                "max_macrophage_dispersal_rate_um3_per_frame",
                "mean_boundary_object_count",
                "net_cluster_flux",
            ]
            selected.extend(f for f in fallback if f in summaries.columns and f not in selected)
    else:
        excluded = {
            "block_name",
            "n_frames",
            "xy_pixel_size_um",
            "z_step_size_um",
            "mask_mode",
            "mask_path",
            "tracks_path",
        }
        selected = [
            c
            for c in summaries.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(summaries[c])
        ]

    renamed = {
        feature: f"fish_mean__cluster_{feature}"
        for feature in selected
    }
    dynamics = summaries[["block_name"] + selected].rename(columns=renamed)

    merged = model_a.merge(
        dynamics,
        left_on=fish_col,
        right_on="block_name",
        how="left",
        suffixes=("", "_cluster"),
    )
    if fish_col != "block_name" and "block_name" in merged.columns:
        merged = merged.drop(columns=["block_name"])

    new_columns = list(renamed.values())
    missing = merged[new_columns].isna().all(axis=1)
    report = pd.DataFrame(
        {
            "fish": model_a[fish_col],
            "cluster_dynamics_found": ~missing.to_numpy(),
        }
    )
    if missing.any():
        raise ValueError(
            "Missing cluster dynamics for Model A fish: "
            f"{merged.loc[missing, fish_col].tolist()}"
        )

    return merged, report, new_columns


def main() -> None:
    args = parse_args()
    blocks_root = Path(args.blocks_root)
    model_a_path = Path(args.model_a_table)
    output_dir = Path(args.output_dir)
    per_block_dir = output_dir / "per_block"

    output_dir.mkdir(parents=True, exist_ok=True)
    per_block_dir.mkdir(parents=True, exist_ok=True)

    if args.rolling_window < 2:
        raise ValueError("--rolling-window must be >= 2")
    if not 0 < args.phase_fraction <= 0.5:
        raise ValueError("--phase-fraction must be in (0, 0.5]")

    pixel_sizes = load_pixel_sizes(args.pixel_size_csv)
    metadata_intervals = load_time_intervals(Path(args.metadata_file), args.time_interval_col)
    block_dirs = sorted(
        p for p in blocks_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    summaries = []
    audits = []
    for block_dir in block_dirs:
        xy_um, z_um = pixel_sizes.get(
            block_dir.name, (DEFAULT_XY_UM, DEFAULT_Z_UM)
        )
        try:
            frame_interval_minutes = args.frame_interval_minutes
            if frame_interval_minutes is None:
                frame_interval_minutes = metadata_intervals.get(block_dir.name)
            summary, audit = process_block(
                block_dir=block_dir,
                per_block_dir=per_block_dir,
                xy_um=xy_um,
                z_um=z_um,
                window=args.rolling_window,
                phase_fraction=args.phase_fraction,
                frame_interval_minutes=frame_interval_minutes,
                near_distance_um=args.near_distance_um,
            )
            summaries.append(summary)
            audits.append(audit)
            print(
                f"[DONE] {block_dir.name}: "
                f"entries={summary['total_cluster_entries']}, "
                f"exits={summary['total_cluster_exits']}"
            )
        except Exception as exc:
            audits.append(
                {
                    "block_name": block_dir.name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[WARN] {block_dir.name}: {exc}")
            if args.strict:
                pd.DataFrame(audits).to_csv(
                    output_dir / "cluster_dynamics_audit.csv", index=False
                )
                raise

    pd.DataFrame(audits).to_csv(
        output_dir / "cluster_dynamics_audit.csv", index=False
    )
    if not summaries:
        raise RuntimeError("No block completed successfully")

    summary_table = pd.DataFrame(summaries).sort_values("block_name")
    summary_table.to_csv(
        output_dir / "cluster_dynamics_fish_level_all_features.csv",
        index=False,
    )

    merged, merge_report, new_columns = merge_model_a(
        model_a_path=model_a_path,
        summaries=summary_table,
        feature_set=args.feature_set,
        explicit_fish_col=args.model_a_fish_col,
    )
    merge_report.to_csv(output_dir / "model_c_merge_report.csv", index=False)
    merged.to_csv(
        output_dir / "constrained_fish_level_model_c.csv", index=False
    )

    definition = {
        "model": "Model C: Model A + cluster and accumulation dynamics",
        "feature_set": args.feature_set,
        "rolling_window_frames": args.rolling_window,
        "phase_fraction": args.phase_fraction,
        "frame_interval_minutes": args.frame_interval_minutes,
        "metadata_file": args.metadata_file,
        "time_interval_col": args.time_interval_col,
        "near_distance_um": args.near_distance_um,
        "new_predictor_columns": new_columns,
        "fish_count": len(merged),
    }
    (output_dir / "model_c_definition.json").write_text(
        json.dumps(definition, indent=2), encoding="utf-8"
    )

    print()
    print("============================================================")
    print("MODEL C FEATURE EXTRACTION COMPLETE")
    print("============================================================")
    print(f"Fish count: {len(merged)}")
    print(f"New predictors added: {len(new_columns)}")
    for col in new_columns:
        print(f"  - {col}")
    print(
        "Classifier input: "
        f"{output_dir / 'constrained_fish_level_model_c.csv'}"
    )


if __name__ == "__main__":
    main()
