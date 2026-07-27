#!/usr/bin/env python3
"""Plot MSD, directionality and related movement summaries from retained tracks."""

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


DATASET_PATTERNS = {
    "musc": "musc/*object_timepoint_features.csv",
    "macrophage_all": "macrophage_all/*object_timepoint_features.csv",
    "macrophage_outside_boundary": "macrophage_outside_boundary/*object_timepoint_features.csv",
}

DISPLAY_NAMES = {
    "musc": "MUSC",
    "macrophage_all": "Macrophage all",
    "macrophage_outside_boundary": "Macrophage outside-boundary",
}

GENOTYPE_COLORS = {
    "WT": "#1f77b4",
    "MUT": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make Fig. 1-style trajectory, directionality, and MSD plots from raw per-timepoint tracks."
    )
    parser.add_argument("--root", default="final_feature_outputs")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_PATTERNS), required=True)
    parser.add_argument("--qc-track-table", action="append", default=[], metavar="DATASET=CSV")
    parser.add_argument("--metadata-file", default="block_metadata.csv")
    parser.add_argument("--time-interval-col", default="time_interval_seconds")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-track-points", type=int, default=5)
    parser.add_argument("--max-example-tracks-per-genotype", type=int, default=18)
    parser.add_argument("--max-directionality-lines-per-genotype", type=int, default=80)
    parser.add_argument("--max-msd-lag", type=int, default=20)
    parser.add_argument("--msd-time-bin-minutes", type=float, default=30.0)
    parser.add_argument("--rose-bins", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Use DATASET=CSV format: {value}")
        dataset, raw_path = value.split("=", 1)
        result[dataset.strip()] = Path(raw_path.strip())
    return result


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


def load_qc_keys(path: Path) -> set[tuple[str, str]]:
    table = pd.read_csv(path, low_memory=False)
    if "block_name" not in table.columns or "track_id" not in table.columns:
        raise ValueError(f"QC table needs block_name and track_id columns: {path}")
    return set(zip(table["block_name"].astype(str), table["track_id"].astype(str)))


def load_time_metadata(path: Path, interval_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = pd.read_csv(path, low_memory=False)
    if "block_name" not in metadata.columns or interval_col not in metadata.columns:
        raise ValueError(f"{path} must contain block_name and {interval_col}")
    metadata = metadata[["block_name", interval_col]].copy()
    metadata["block_name"] = metadata["block_name"].astype(str).str.strip()
    metadata[interval_col] = numeric(metadata[interval_col])
    bad = metadata[metadata[interval_col].isna() | (metadata[interval_col] <= 0)]
    if not bad.empty:
        raise ValueError(
            f"{path} has missing/non-positive {interval_col} for: "
            + ", ".join(bad["block_name"].astype(str).head(20))
        )
    return metadata.rename(columns={interval_col: "time_interval_seconds"})


def load_timepoint_dataset(root: Path, dataset: str, qc_path: Path | None, metadata: pd.DataFrame) -> pd.DataFrame:
    files = sorted(root.glob(f"*/{DATASET_PATTERNS[dataset]}"))
    if not files:
        raise FileNotFoundError(f"No object timepoint files found for {dataset} under {root}")

    needed = {
        "time",
        "track_id",
        "block_name",
        "fish_id",
        "genotype",
        "centroid_x_um",
        "centroid_y_um",
        "centroid_z_um",
    }
    frames: list[pd.DataFrame] = []
    for path in files:
        head = pd.read_csv(path, nrows=0)
        usecols = [column for column in head.columns if column in needed]
        missing = needed - set(usecols)
        if missing:
            print(f"[WARN] Skipping {path}; missing {sorted(missing)}")
            continue
        frames.append(pd.read_csv(path, usecols=usecols, low_memory=False))
    if not frames:
        raise ValueError(f"No usable timepoint files found for {dataset}")

    data = pd.concat(frames, ignore_index=True)
    data["block_name"] = data["block_name"].astype(str)
    data["track_id"] = data["track_id"].astype(str)
    data["genotype"] = data["genotype"].map(normalise_genotype)
    data = data[data["genotype"].isin(["WT", "MUT"])].copy()
    data["global_track_id"] = data["block_name"] + "::" + data["track_id"]
    for column in ["time", "centroid_x_um", "centroid_y_um", "centroid_z_um"]:
        data[column] = numeric(data[column])
    data = data.dropna(subset=["time", "centroid_x_um", "centroid_y_um", "centroid_z_um"])
    data = data.merge(metadata, on="block_name", how="left", validate="many_to_one")
    missing_intervals = sorted(data.loc[data["time_interval_seconds"].isna(), "block_name"].unique())
    if missing_intervals:
        raise ValueError(
            "No time_interval_seconds found in metadata for blocks: "
            + ", ".join(map(str, missing_intervals[:50]))
        )

    if qc_path is not None:
        qc_keys = load_qc_keys(qc_path)
        before = len(data)
        data = data[
            [
                (block, track) in qc_keys
                for block, track in zip(data["block_name"], data["track_id"])
            ]
        ].copy()
        print(f"[INFO] {dataset}: QC filter kept {len(data):,}/{before:,} object timepoints.")
    return data.sort_values(["global_track_id", "time"]).reset_index(drop=True)


def add_track_metrics(data: pd.DataFrame, min_track_points: int) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for track_id, track in data.groupby("global_track_id", sort=False):
        track = track.sort_values("time").copy()
        if len(track) < min_track_points:
            continue
        coords = track[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy(float)
        deltas = np.diff(coords, axis=0)
        step_lengths = np.sqrt((deltas**2).sum(axis=1))
        cumulative_path = np.concatenate([[0.0], np.cumsum(step_lengths)])
        displacement = np.sqrt(((coords - coords[0]) ** 2).sum(axis=1))
        directionality = np.divide(
            displacement,
            cumulative_path,
            out=np.zeros_like(displacement),
            where=cumulative_path > 0,
        )
        interval_seconds = float(track["time_interval_seconds"].iloc[0])
        elapsed = (track["time"].to_numpy(float) - float(track["time"].iloc[0])) * interval_seconds / 60.0
        duration = elapsed[-1]
        if duration <= 0:
            continue
        track["elapsed_time"] = elapsed
        track["relative_time"] = elapsed / duration
        track["directionality_ratio_over_time"] = directionality
        track["x_rel_um"] = coords[:, 0] - coords[0, 0]
        track["y_rel_um"] = coords[:, 1] - coords[0, 1]
        track["z_rel_um"] = coords[:, 2] - coords[0, 2]
        records.append(track)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def wrap_angle_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def compute_relative_turning_angles(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (genotype, track_id), track in data.groupby(["genotype", "global_track_id"], sort=False):
        track = track.sort_values("elapsed_time")
        coords = track[["centroid_x_um", "centroid_y_um"]].to_numpy(float)
        if len(coords) < 3:
            continue
        deltas = np.diff(coords, axis=0)
        step_lengths = np.sqrt((deltas**2).sum(axis=1))
        valid = step_lengths > 1e-9
        if valid.sum() < 2:
            continue
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        first_angle = angles[np.flatnonzero(valid)[0]]
        relative_angles = wrap_angle_radians(angles - first_angle)
        times = track["elapsed_time"].to_numpy(float)
        for step_index, (angle, length, is_valid) in enumerate(zip(relative_angles, step_lengths, valid), start=1):
            if not is_valid:
                continue
            rows.append(
                {
                    "genotype": genotype,
                    "global_track_id": track_id,
                    "step_index": step_index,
                    "is_reference_step": bool(step_index == int(np.flatnonzero(valid)[0]) + 1),
                    "step_start_elapsed_minutes": float(times[step_index - 1]),
                    "step_end_elapsed_minutes": float(times[step_index]),
                    "relative_angle_deg": float(np.degrees(angle)),
                    "relative_angle_rad": float(angle),
                    "step_length_um": float(length),
                }
            )
    return pd.DataFrame(rows)


def summarize_relative_turning_angles(angles: pd.DataFrame, bins: int) -> pd.DataFrame:
    if angles.empty:
        return pd.DataFrame()
    angles = angles[~angles["is_reference_step"]].copy()
    if angles.empty:
        return pd.DataFrame()
    edges = np.linspace(-180, 180, bins + 1)
    rows: list[dict[str, object]] = []
    for genotype, sub in angles.groupby("genotype"):
        values = sub["relative_angle_deg"].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        percentages = counts / counts.sum() * 100 if counts.sum() else counts
        for start, end, count, percentage in zip(edges[:-1], edges[1:], counts, percentages):
            rows.append(
                {
                    "genotype": genotype,
                    "bin_start_deg": float(start),
                    "bin_end_deg": float(end),
                    "bin_center_deg": float((start + end) / 2),
                    "n_steps": int(count),
                    "percent_steps": float(percentage),
                    "n_tracks": int(sub["global_track_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def summarize_directionality(data: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0, 1, 22)
    labels = np.linspace(0, 1, 21)
    table = data.copy()
    table["relative_time_bin"] = pd.cut(
        table["relative_time"], bins=bins, labels=labels, include_lowest=True
    ).astype(float)
    rows: list[dict[str, object]] = []
    for (genotype, time_bin), sub in table.groupby(["genotype", "relative_time_bin"], dropna=True):
        values = sub["directionality_ratio_over_time"].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        rows.append(
            {
                "genotype": genotype,
                "relative_time": float(time_bin),
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
                "n_timepoints": int(len(values)),
                "n_tracks": int(sub["global_track_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def compute_msd(data: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (genotype, track_id), track in data.groupby(["genotype", "global_track_id"], sort=False):
        track = track.sort_values("time")
        coords = track[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy(float)
        interval_seconds = float(track["time_interval_seconds"].iloc[0])
        times = track["time"].to_numpy(float) * interval_seconds / 60.0
        if len(coords) < 3:
            continue
        for lag in range(1, min(max_lag, len(coords) - 1) + 1):
            interval = np.nanmedian(times[lag:] - times[:-lag])
            if not np.isfinite(interval) or interval <= 0:
                interval = float(lag)
            diffs = coords[lag:] - coords[:-lag]
            squared = (diffs**2).sum(axis=1)
            if len(squared) == 0:
                continue
            rows.append(
                {
                    "genotype": genotype,
                    "global_track_id": track_id,
                    "lag_frames": lag,
                    "time_interval_minutes": interval,
                    "msd_um2": float(np.nanmean(squared)),
                }
            )
    track_msd = pd.DataFrame(rows)
    if track_msd.empty:
        return track_msd, pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for (genotype, lag), sub in track_msd.groupby(["genotype", "lag_frames"], dropna=True):
        values = sub["msd_um2"].dropna().to_numpy(float)
        intervals = sub["time_interval_minutes"].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        summary_rows.append(
            {
                "genotype": genotype,
                "lag_frames": int(lag),
                "time_interval_minutes": float(np.nanmedian(intervals)),
                "median_msd_um2": float(np.nanmedian(values)),
                "q25_msd_um2": float(np.nanpercentile(values, 25)),
                "q75_msd_um2": float(np.nanpercentile(values, 75)),
                "mean_msd_um2": float(np.nanmean(values)),
                "n_tracks": int(sub["global_track_id"].nunique()),
            }
        )
    return track_msd, pd.DataFrame(summary_rows)


def compute_msd_timecourse(data: pd.DataFrame, bin_minutes: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bin_minutes <= 0:
        raise ValueError("--msd-time-bin-minutes must be positive")
    rows: list[dict[str, object]] = []
    for (genotype, track_id), track in data.groupby(["genotype", "global_track_id"], sort=False):
        track = track.sort_values("elapsed_time")
        if len(track) < 2:
            continue
        coords = track[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy(float)
        elapsed = track["elapsed_time"].to_numpy(float)
        squared = ((coords - coords[0]) ** 2).sum(axis=1)
        for time_value, msd_value in zip(elapsed, squared):
            if not np.isfinite(time_value) or not np.isfinite(msd_value):
                continue
            rows.append(
                {
                    "genotype": genotype,
                    "global_track_id": track_id,
                    "elapsed_minutes": float(time_value),
                    "time_bin_minutes": float(round(time_value / bin_minutes) * bin_minutes),
                    "squared_displacement_um2": float(msd_value),
                }
            )
    point_table = pd.DataFrame(rows)
    if point_table.empty:
        return point_table, pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for (genotype, time_bin), sub in point_table.groupby(["genotype", "time_bin_minutes"], dropna=True):
        values = sub["squared_displacement_um2"].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        summary_rows.append(
            {
                "genotype": genotype,
                "elapsed_minutes": float(time_bin),
                "mean_msd_um2": float(np.nanmean(values)),
                "median_msd_um2": float(np.nanmedian(values)),
                "sem_msd_um2": float(np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "q25_msd_um2": float(np.nanpercentile(values, 25)),
                "q75_msd_um2": float(np.nanpercentile(values, 75)),
                "n_observations": int(len(values)),
                "n_tracks": int(sub["global_track_id"].nunique()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary = summary[summary["elapsed_minutes"] >= 0].sort_values(["genotype", "elapsed_minutes"])
    return point_table, summary


def fit_log_msd(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for genotype, sub in summary.groupby("genotype"):
        sub = sub[(sub["time_interval_minutes"] > 0) & (sub["median_msd_um2"] > 0)].sort_values("time_interval_minutes")
        if len(sub) < 3:
            continue
        x = np.log10(sub["time_interval_minutes"].to_numpy(float))
        y = np.log10(sub["median_msd_um2"].to_numpy(float))
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        ss_res = float(((y - predicted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(
            {
                "genotype": genotype,
                "alpha_slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
            }
        )
    return pd.DataFrame(rows)


def plot_trajectories(label: str, data: pd.DataFrame, output_path: Path, max_tracks: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), sharex=True, sharey=True)
    for ax, genotype in zip(axes, ["WT", "MUT"]):
        sub = data[data["genotype"].eq(genotype)]
        tracks = np.array(sorted(sub["global_track_id"].unique()))
        if len(tracks) > max_tracks:
            tracks = rng.choice(tracks, size=max_tracks, replace=False)
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        for track_id in tracks:
            track = sub[sub["global_track_id"].eq(track_id)].sort_values("elapsed_time")
            ax.plot(track["x_rel_um"], track["y_rel_um"], color=color, alpha=0.55, linewidth=1.0)
            ax.scatter(track["x_rel_um"].iloc[0], track["y_rel_um"].iloc[0], color="black", s=12, zorder=3)
            ax.scatter(track["x_rel_um"].iloc[-1], track["y_rel_um"].iloc[-1], color=color, edgecolor="black", s=24, zorder=3)
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
        ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{genotype} example tracks")
        ax.set_xlabel("Relative x displacement (um)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Relative y displacement (um)")
    fig.suptitle(f"{label}: Fig. 1-style cell trajectories", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_directionality(label: str, data: pd.DataFrame, summary: pd.DataFrame, output_path: Path, max_lines: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for genotype in ["WT", "MUT"]:
        sub = data[data["genotype"].eq(genotype)]
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        tracks = np.array(sorted(sub["global_track_id"].unique()))
        if len(tracks) > max_lines:
            tracks = rng.choice(tracks, size=max_lines, replace=False)
        for track_id in tracks:
            track = sub[sub["global_track_id"].eq(track_id)].sort_values("relative_time")
            ax.plot(track["relative_time"], track["directionality_ratio_over_time"], color=color, alpha=0.08, linewidth=0.8)
        sm = summary[summary["genotype"].eq(genotype)].sort_values("relative_time")
        if not sm.empty:
            ax.fill_between(
                sm["relative_time"].to_numpy(float),
                sm["q25"].to_numpy(float),
                sm["q75"].to_numpy(float),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(sm["relative_time"], sm["median"], color=color, linewidth=2.7, label=f"{genotype} median")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Relative track time (0=start, 1=end)")
    ax.set_ylabel("Directionality ratio")
    ax.set_title(f"{label}: directionality ratio over track lifetime")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_msd(label: str, summary: pd.DataFrame, fit: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for genotype in ["WT", "MUT"]:
        sm = summary[summary["genotype"].eq(genotype)].sort_values("time_interval_minutes")
        if sm.empty:
            continue
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        x = sm["time_interval_minutes"].to_numpy(float)
        y = sm["median_msd_um2"].to_numpy(float)
        ax.scatter(x, y, color=color, edgecolor="black", linewidth=0.45, s=55, label=f"{genotype} median MSD")
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        fit_row = fit[fit["genotype"].eq(genotype)]
        if not fit_row.empty:
            slope = float(fit_row.iloc[0]["alpha_slope"])
            intercept = float(fit_row.iloc[0]["intercept"])
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ys = 10 ** (slope * np.log10(xs) + intercept)
            ax.plot(xs, ys, color=color, linestyle="--", linewidth=2.0)
            ax.text(
                xs[-1],
                ys[-1],
                f"{genotype} alpha={slope:.2f}",
                color=color,
                fontsize=9,
                ha="right",
                va="bottom",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time interval (min)")
    ax.set_ylabel("MSD (um2)")
    ax.set_title(f"{label}: log-log mean squared displacement")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_msd_timecourse(label: str, summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    max_y = 0.0
    for genotype in ["WT", "MUT"]:
        sm = summary[summary["genotype"].eq(genotype)].sort_values("elapsed_minutes")
        sm = sm[sm["n_tracks"] >= 3]
        if sm.empty:
            continue
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        x = sm["elapsed_minutes"].to_numpy(float)
        y = sm["mean_msd_um2"].to_numpy(float)
        sem = sm["sem_msd_um2"].to_numpy(float)
        max_y = max(max_y, float(np.nanmax(y + sem)))
        ax.errorbar(
            x,
            y,
            yerr=sem,
            color=color,
            marker="o",
            markersize=3.4,
            linewidth=2.2,
            elinewidth=1.1,
            capsize=2.2,
            alpha=0.88,
            label=f"{genotype} mean +/- SEM",
        )
    ax.set_xlabel("Elapsed time from track start (min)")
    ax.set_ylabel("MSD (um2)")
    ax.set_title(f"{label}: MSD over elapsed track time")
    ax.set_ylim(bottom=0, top=max_y * 1.12 if max_y > 0 else None)
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_relative_turning_rose(label: str, summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.0),
        subplot_kw={"projection": "polar"},
    )
    max_percent = float(summary["percent_steps"].max()) if not summary.empty else 1.0
    radial_max = max(5.0, math.ceil(max_percent / 5.0) * 5.0)
    for ax, genotype in zip(axes, ["WT", "MUT"]):
        sub = summary[summary["genotype"].eq(genotype)].sort_values("bin_center_deg")
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        if sub.empty:
            ax.set_title(f"{genotype}: no usable steps")
            continue
        theta = np.radians(sub["bin_center_deg"].to_numpy(float))
        width = np.radians((sub["bin_end_deg"] - sub["bin_start_deg"]).median())
        radii = sub["percent_steps"].to_numpy(float)
        ax.bar(
            theta,
            radii,
            width=width * 0.92,
            bottom=0,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetagrids(
            [0, 45, 90, 135, 180, 225, 270, 315],
            ["0 deg", "+45", "+90", "+135", "180", "-135", "-90", "-45"],
        )
        ax.set_ylim(0, radial_max)
        ax.set_rlabel_position(135)
        ax.grid(alpha=0.25)
        tracks = int(sub["n_tracks"].max())
        steps = int(sub["n_steps"].sum())
        ax.set_title(f"{genotype}: relative XY step direction\n{steps:,} steps from {tracks:,} tracks")
    fig.suptitle(
        f"{label}: normalized rose plot of movement direction after each track's first step",
        y=1.03,
    )
    fig.text(
        0.5,
        0.02,
        "Each cell track is rotated so its first valid XY movement points at 0 deg. Radius = percent of later steps for that genotype.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_combined(label: str, data: pd.DataFrame, direction_summary: pd.DataFrame, msd_summary: pd.DataFrame, fit: pd.DataFrame, output_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.7))
    ax = axes[0]
    for genotype in ["WT", "MUT"]:
        sub = data[data["genotype"].eq(genotype)]
        tracks = np.array(sorted(sub["global_track_id"].unique()))
        if len(tracks) > 15:
            tracks = rng.choice(tracks, size=15, replace=False)
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        for track_id in tracks:
            track = sub[sub["global_track_id"].eq(track_id)].sort_values("elapsed_time")
            ax.plot(track["x_rel_um"], track["y_rel_um"], color=color, alpha=0.5, linewidth=0.9)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Relative x (um)")
    ax.set_ylabel("Relative y (um)")
    ax.set_title("Example tracks")
    ax.grid(alpha=0.22)

    ax = axes[1]
    for genotype in ["WT", "MUT"]:
        sm = direction_summary[direction_summary["genotype"].eq(genotype)].sort_values("relative_time")
        if sm.empty:
            continue
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        ax.fill_between(sm["relative_time"], sm["q25"], sm["q75"], color=color, alpha=0.16)
        ax.plot(sm["relative_time"], sm["median"], color=color, linewidth=2.7, label=genotype)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Relative track time")
    ax.set_ylabel("Directionality ratio")
    ax.set_title("Directionality over time")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)

    ax = axes[2]
    for genotype in ["WT", "MUT"]:
        sm = msd_summary[msd_summary["genotype"].eq(genotype)].sort_values("time_interval_minutes")
        if sm.empty:
            continue
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        x = sm["time_interval_minutes"].to_numpy(float)
        y = sm["median_msd_um2"].to_numpy(float)
        ax.scatter(x, y, color=color, edgecolor="black", linewidth=0.45, s=45, label=genotype)
        fit_row = fit[fit["genotype"].eq(genotype)]
        if not fit_row.empty:
            slope = float(fit_row.iloc[0]["alpha_slope"])
            intercept = float(fit_row.iloc[0]["intercept"])
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ys = 10 ** (slope * np.log10(xs) + intercept)
            ax.plot(xs, ys, color=color, linestyle="--", linewidth=2.0, label=f"{genotype} alpha={slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time interval (min)")
    ax.set_ylabel("MSD (um2)")
    ax.set_title("MSD log-log")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=True, fontsize=8)

    fig.suptitle(f"{label}: Fig. 1-style movement summary", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def run_dataset(dataset: str, root: Path, qc_paths: dict[str, Path], output_root: Path, args: argparse.Namespace, dataset_index: int) -> None:
    label = DISPLAY_NAMES.get(dataset, dataset)
    out = output_root / safe_name(label)
    out.mkdir(parents=True, exist_ok=True)
    metadata = load_time_metadata(Path(args.metadata_file), args.time_interval_col)
    data = load_timepoint_dataset(root, dataset, qc_paths.get(dataset), metadata)
    data = add_track_metrics(data, args.min_track_points)
    if data.empty:
        print(f"[WARN] No usable tracks after filtering for {dataset}.")
        return
    direction_summary = summarize_directionality(data)
    track_msd, msd_summary = compute_msd(data, args.max_msd_lag)
    msd_timecourse_points, msd_timecourse_summary = compute_msd_timecourse(data, args.msd_time_bin_minutes)
    fit = fit_log_msd(msd_summary)
    turning_angles = compute_relative_turning_angles(data)
    turning_summary = summarize_relative_turning_angles(turning_angles, args.rose_bins)

    plot_trajectories(label, data, out / "01_trajectory_examples.png", args.max_example_tracks_per_genotype, args.random_seed + dataset_index)
    plot_directionality(label, data, direction_summary, out / "02_directionality_ratio_over_time.png", args.max_directionality_lines_per_genotype, args.random_seed + 100 + dataset_index)
    plot_msd(label, msd_summary, fit, out / "03_msd_loglog.png")
    plot_combined(label, data, direction_summary, msd_summary, fit, out / "04_fig1_style_movement_summary.png", args.random_seed + 200 + dataset_index)
    if not turning_summary.empty:
        plot_relative_turning_rose(label, turning_summary, out / "05_normalized_relative_direction_rose.png")
    if not msd_timecourse_summary.empty:
        plot_msd_timecourse(label, msd_timecourse_summary, out / "06_msd_timecourse_linear.png")

    direction_summary.to_csv(out / "directionality_over_time_summary.csv", index=False)
    track_msd.to_csv(out / "track_level_msd_by_lag.csv", index=False)
    msd_summary.to_csv(out / "msd_by_lag_summary.csv", index=False)
    msd_timecourse_points.to_csv(out / "msd_timecourse_points.csv", index=False)
    msd_timecourse_summary.to_csv(out / "msd_timecourse_summary.csv", index=False)
    fit.to_csv(out / "msd_loglog_fit_summary.csv", index=False)
    turning_angles.to_csv(out / "relative_direction_angles_by_step.csv", index=False)
    turning_summary.to_csv(out / "relative_direction_rose_summary.csv", index=False)
    print(f"[DONE] {label}: MSD/directionality plots saved to {out}")


# Track-derived MSD and directionality definitions follow the cell-migration
# framework described by Gorelik and Gautreau (2014):
# https://pmc.ncbi.nlm.nih.gov/articles/PMC4439174/
def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    qc_paths = parse_mapping(args.qc_track_table)
    for index, dataset in enumerate(args.dataset):
        run_dataset(dataset, Path(args.root), qc_paths, output_root, args, index)
    print(f"[DONE] All MSD/directionality plots saved under: {output_root}")


if __name__ == "__main__":
    main()
