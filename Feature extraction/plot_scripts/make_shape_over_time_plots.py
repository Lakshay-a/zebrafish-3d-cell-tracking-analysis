#!/usr/bin/env python3
"""Plot cell-shape trajectories over time for each retained analysis dataset."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


GENOTYPE_COLORS = {
    "WT": "#1f77b4",
    "MUT": "#d62728",
}

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

SHAPE_FEATURES = [
    ("sphericity", "Sphericity"),
    ("elongation", "Elongation"),
    ("aspect_ratio_3d", "3D aspect ratio"),
    ("volume_um3", "Volume (um3)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true per-timepoint shape change for tracked cells."
    )
    parser.add_argument("--root", default="final_feature_outputs")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_PATTERNS), required=True)
    parser.add_argument("--qc-track-table", action="append", default=[], metavar="DATASET=CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-track-points", type=int, default=4)
    parser.add_argument("--time-bins", type=int, default=21)
    parser.add_argument("--max-lines-per-genotype", type=int, default=90)
    parser.add_argument("--max-morphospace-tracks-per-genotype", type=int, default=70)
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


def load_dataset(root: Path, dataset: str, qc_path: Path | None) -> pd.DataFrame:
    files = sorted(root.glob(f"*/{DATASET_PATTERNS[dataset]}"))
    if not files:
        raise FileNotFoundError(f"No object timepoint files found for {dataset} under {root}")

    frames: list[pd.DataFrame] = []
    needed = {
        "time",
        "track_id",
        "block_name",
        "fish_id",
        "genotype",
        "sphericity",
        "elongation",
        "aspect_ratio_3d",
        "volume_um3",
    }
    for path in files:
        head = pd.read_csv(path, nrows=0)
        usecols = [column for column in head.columns if column in needed]
        missing = {"time", "track_id", "block_name", "genotype", "sphericity", "elongation"} - set(usecols)
        if missing:
            print(f"[WARN] Skipping {path}; missing {sorted(missing)}")
            continue
        frames.append(pd.read_csv(path, usecols=usecols, low_memory=False))

    if not frames:
        raise ValueError(f"No usable object timepoint files found for {dataset}")

    data = pd.concat(frames, ignore_index=True)
    data["block_name"] = data["block_name"].astype(str)
    data["track_id"] = data["track_id"].astype(str)
    data["genotype"] = data["genotype"].map(normalise_genotype)
    data = data[data["genotype"].isin(["WT", "MUT"])].copy()
    data["global_track_id"] = data["block_name"] + "::" + data["track_id"]
    data["time"] = numeric(data["time"])
    for feature, _ in SHAPE_FEATURES:
        if feature in data.columns:
            data[feature] = numeric(data[feature])

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

    counts = data.groupby("global_track_id")["time"].nunique()
    keep_tracks = set(counts[counts >= 1].index.astype(str))
    return data[data["global_track_id"].isin(keep_tracks)].copy()


def add_relative_time(data: pd.DataFrame, min_track_points: int, time_bins: int) -> pd.DataFrame:
    data = data.sort_values(["global_track_id", "time"]).copy()
    grouped = data.groupby("global_track_id", sort=False)
    duration = grouped["time"].transform(lambda values: values.max() - values.min())
    n_points = grouped["time"].transform("count")
    start = grouped["time"].transform("min")
    data["relative_time"] = np.where(duration > 0, (data["time"] - start) / duration, np.nan)
    data = data[(n_points >= min_track_points) & data["relative_time"].notna()].copy()
    data["time_bin"] = pd.cut(
        data["relative_time"],
        bins=np.linspace(0, 1, time_bins + 1),
        labels=np.linspace(0, 1, time_bins),
        include_lowest=True,
    ).astype(float)
    return data


def summarize_time_bins(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (genotype, time_bin), sub in data.groupby(["genotype", "time_bin"], dropna=True):
        values = sub[feature].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        rows.append(
            {
                "genotype": genotype,
                "relative_time": float(time_bin),
                "feature": feature,
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
                "n_object_timepoints": int(len(values)),
                "n_tracks": int(sub["global_track_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_timecourses(
    label: str,
    data: pd.DataFrame,
    out: Path,
    max_lines_per_genotype: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    available = [(feature, display) for feature, display in SHAPE_FEATURES if feature in data.columns]
    summary_frames: list[pd.DataFrame] = []

    for feature, display in available:
        feature_data = data.dropna(subset=[feature, "relative_time"]).copy()
        if feature_data.empty:
            continue
        summary = summarize_time_bins(feature_data, feature)
        summary_frames.append(summary)

        fig, ax = plt.subplots(figsize=(8.4, 5.8))
        for genotype in ["WT", "MUT"]:
            sub = feature_data[feature_data["genotype"] == genotype]
            if sub.empty:
                continue
            color = GENOTYPE_COLORS.get(genotype, "#666666")
            tracks = np.array(sorted(sub["global_track_id"].unique()))
            if len(tracks) > max_lines_per_genotype:
                tracks = rng.choice(tracks, size=max_lines_per_genotype, replace=False)
            for track_id in tracks:
                track = sub[sub["global_track_id"] == track_id].sort_values("relative_time")
                ax.plot(
                    track["relative_time"],
                    track[feature],
                    color=color,
                    alpha=0.055,
                    linewidth=0.8,
                )

            sm = summary[summary["genotype"] == genotype].sort_values("relative_time")
            if not sm.empty:
                ax.fill_between(
                    sm["relative_time"].to_numpy(float),
                    sm["q25"].to_numpy(float),
                    sm["q75"].to_numpy(float),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
                ax.plot(
                    sm["relative_time"],
                    sm["median"],
                    color=color,
                    linewidth=2.8,
                    label=f"{genotype} median with IQR",
                )

        ax.set_xlabel("Relative track time (0=start, 1=end)")
        ax.set_ylabel(display)
        ax.set_title(f"{label}: {display} over tracked-cell lifetime")
        ax.grid(alpha=0.22)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(out / f"shape_over_time__{safe_name(feature)}.png", dpi=260)
        plt.close(fig)

    return pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()


def plot_shape_score_timecourse(label: str, data: pd.DataFrame, out: Path) -> pd.DataFrame:
    if not {"sphericity", "elongation"}.issubset(data.columns):
        return pd.DataFrame()
    shape_data = data.dropna(subset=["sphericity", "elongation", "relative_time"]).copy()
    if shape_data.empty:
        return pd.DataFrame()

    s_center = shape_data["sphericity"].median()
    s_spread = shape_data["sphericity"].quantile(0.75) - shape_data["sphericity"].quantile(0.25)
    e_center = shape_data["elongation"].median()
    e_spread = shape_data["elongation"].quantile(0.75) - shape_data["elongation"].quantile(0.25)
    s_spread = s_spread if np.isfinite(s_spread) and s_spread > 0 else shape_data["sphericity"].std()
    e_spread = e_spread if np.isfinite(e_spread) and e_spread > 0 else shape_data["elongation"].std()
    s_spread = s_spread if np.isfinite(s_spread) and s_spread > 0 else 1.0
    e_spread = e_spread if np.isfinite(e_spread) and e_spread > 0 else 1.0
    shape_data["elongation_score"] = (
        (shape_data["elongation"] - e_center) / e_spread
        - (shape_data["sphericity"] - s_center) / s_spread
    )
    summary = summarize_time_bins(shape_data, "elongation_score")

    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    for genotype in ["WT", "MUT"]:
        sm = summary[summary["genotype"] == genotype].sort_values("relative_time")
        if sm.empty:
            continue
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        ax.fill_between(
            sm["relative_time"].to_numpy(float),
            sm["q25"].to_numpy(float),
            sm["q75"].to_numpy(float),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(sm["relative_time"], sm["median"], color=color, linewidth=2.8, label=genotype)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.7)
    ax.text(0.01, 0.96, "more elongated", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.text(0.01, 0.04, "more round/spherical", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    ax.set_xlabel("Relative track time (0=start, 1=end)")
    ax.set_ylabel("Shape score: elongation high, sphericity low")
    ax.set_title(f"{label}: round-to-elongated shape score over time")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "shape_over_time__round_to_elongated_score.png", dpi=260)
    plt.close(fig)
    return summary


def plot_morphospace_trajectories(
    label: str,
    data: pd.DataFrame,
    out: Path,
    max_tracks_per_genotype: int,
    seed: int,
) -> None:
    if not {"sphericity", "elongation"}.issubset(data.columns):
        return
    rng = np.random.default_rng(seed)
    shape_data = data.dropna(subset=["sphericity", "elongation", "relative_time"]).copy()
    if shape_data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.8))
    ax = axes[0]
    median_points: list[tuple[float, float]] = []
    genotype_medians: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for genotype in ["WT", "MUT"]:
        sub = shape_data[shape_data["genotype"] == genotype]
        if sub.empty:
            continue
        starts = sub.sort_values("relative_time").groupby("global_track_id").first()
        ends = sub.sort_values("relative_time").groupby("global_track_id").last()
        start_xy = (float(starts["sphericity"].median()), float(starts["elongation"].median()))
        end_xy = (float(ends["sphericity"].median()), float(ends["elongation"].median()))
        genotype_medians[genotype] = (start_xy, end_xy)
        median_points.extend([start_xy, end_xy])

    label_offsets = {
        "WT": (-54, 26),
        "MUT": (24, -34),
    }
    for genotype in ["WT", "MUT"]:
        sub = shape_data[shape_data["genotype"] == genotype]
        if sub.empty:
            continue
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        tracks = np.array(sorted(sub["global_track_id"].unique()))
        if len(tracks) > max_tracks_per_genotype:
            tracks = rng.choice(tracks, size=max_tracks_per_genotype, replace=False)
        for track_id in tracks:
            track = sub[sub["global_track_id"] == track_id].sort_values("relative_time")
            ax.plot(track["sphericity"], track["elongation"], color=color, alpha=0.14, linewidth=0.9)
            if len(track) >= 2:
                start = track.iloc[0]
                end = track.iloc[-1]
                ax.annotate(
                    "",
                    xy=(end["sphericity"], end["elongation"]),
                    xytext=(start["sphericity"], start["elongation"]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "alpha": 0.16,
                        "linewidth": 0.7,
                    },
                )

        start_xy, end_xy = genotype_medians[genotype]
        ax.annotate(
            "",
            xy=end_xy,
            xytext=start_xy,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 4.2,
                "mutation_scale": 22,
                "shrinkA": 8,
                "shrinkB": 8,
            },
            zorder=6,
        )
        ax.scatter(
            [start_xy[0]],
            [start_xy[1]],
            s=150,
            marker="s",
            color=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=7,
        )
        ax.scatter(
            [end_xy[0]],
            [end_xy[1]],
            s=210,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=8,
        )
        offset = label_offsets.get(genotype, (12, 12))
        ax.annotate(
            f"{genotype} start",
            xy=start_xy,
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0, "alpha": 0.8},
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "alpha": 0.86},
            zorder=9,
        )
        ax.annotate(
            f"{genotype} end",
            xy=end_xy,
            xytext=(offset[0], -offset[1]),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0, "alpha": 0.8},
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "alpha": 0.86},
            zorder=9,
        )

    ax.set_xlabel("Sphericity")
    ax.set_ylabel("Elongation")
    ax.set_title("Median position in shape space")
    ax.grid(alpha=0.22)
    if median_points:
        xs = np.array([point[0] for point in median_points], dtype=float)
        ys = np.array([point[1] for point in median_points], dtype=float)
        x_span = max(float(xs.max() - xs.min()), 0.018)
        y_span = max(float(ys.max() - ys.min()), 0.018)
        ax.set_xlim(xs.min() - x_span * 2.4, xs.max() + x_span * 2.4)
        ax.set_ylim(ys.min() - y_span * 2.4, ys.max() + y_span * 2.4)
    handles = [
        Line2D([0], [0], color=GENOTYPE_COLORS["WT"], linewidth=4, label="WT median direction"),
        Line2D([0], [0], color=GENOTYPE_COLORS["MUT"], linewidth=4, label="MUT median direction"),
        Line2D([0], [0], marker="s", color="black", markerfacecolor="white",
               linestyle="None", markersize=8, label="Start point"),
        Line2D([0], [0], marker="*", color="black", markerfacecolor="white",
               linestyle="None", markersize=12, label="End point"),
    ]
    ax.legend(handles=handles, frameon=True, fontsize=8)
    delta_ax = axes[1]
    deltas: list[tuple[float, float]] = []
    for genotype, (start_xy, end_xy) in genotype_medians.items():
        color = GENOTYPE_COLORS.get(genotype, "#666666")
        dx = float(end_xy[0] - start_xy[0])
        dy = float(end_xy[1] - start_xy[1])
        deltas.append((dx, dy))
        delta_ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 4.5,
                "mutation_scale": 24,
            },
            zorder=5,
        )
        delta_ax.scatter(
            [dx],
            [dy],
            s=220,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
        )
        delta_ax.annotate(
            f"{genotype} end\nΔS={dx:+.4f}\nΔE={dy:+.4f}",
            xy=(dx, dy),
            xytext=(16 if dx >= 0 else -80, 18 if dy >= 0 else -54),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0},
        )

    delta_ax.scatter(
        [0],
        [0],
        s=150,
        marker="s",
        color="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=6,
    )
    delta_ax.annotate(
        "common start",
        xy=(0, 0),
        xytext=(10, -32),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )
    delta_ax.axhline(0, color="black", linewidth=1.0, alpha=0.55)
    delta_ax.axvline(0, color="black", linewidth=1.0, alpha=0.55)
    if deltas:
        dxs = np.array([value[0] for value in deltas] + [0.0], dtype=float)
        dys = np.array([value[1] for value in deltas] + [0.0], dtype=float)
        limit = max(float(np.nanmax(np.abs(dxs))), float(np.nanmax(np.abs(dys))), 0.001)
        delta_ax.set_xlim(-limit * 2.4, limit * 2.4)
        delta_ax.set_ylim(-limit * 2.4, limit * 2.4)
    delta_ax.set_aspect("equal", adjustable="box")
    delta_ax.set_xlabel("Change in sphericity from start to end")
    delta_ax.set_ylabel("Change in elongation from start to end")
    delta_ax.set_title("Median change vector")
    delta_ax.grid(alpha=0.22)
    fig.suptitle(f"{label}: tracked-cell shape-space change", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "shape_over_time__morphospace_trajectories.png", dpi=260)
    ax.autoscale(enable=True, axis="both", tight=False)
    ax.relim()
    ax.autoscale_view()
    ax.set_title("Full shape-space field")
    fig.tight_layout()
    fig.savefig(out / "shape_over_time__morphospace_trajectories_full_field.png", dpi=260)
    plt.close(fig)


def run_dataset(
    dataset: str,
    root: Path,
    qc_paths: dict[str, Path],
    output_root: Path,
    args: argparse.Namespace,
    dataset_index: int,
) -> None:
    label = DISPLAY_NAMES.get(dataset, dataset)
    out = output_root / safe_name(label)
    out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(root, dataset, qc_paths.get(dataset))
    data = add_relative_time(data, args.min_track_points, args.time_bins)
    if data.empty:
        print(f"[WARN] No usable timepoint rows after filtering for {dataset}.")
        return

    feature_summary = plot_feature_timecourses(
        label,
        data,
        out,
        args.max_lines_per_genotype,
        args.random_seed + dataset_index,
    )
    if not feature_summary.empty:
        feature_summary.to_csv(out / "shape_over_time_feature_summary.csv", index=False)

    score_summary = plot_shape_score_timecourse(label, data, out)
    if not score_summary.empty:
        score_summary.to_csv(out / "shape_over_time_score_summary.csv", index=False)

    plot_morphospace_trajectories(
        label,
        data,
        out,
        args.max_morphospace_tracks_per_genotype,
        args.random_seed + 100 + dataset_index,
    )

    track_counts = (
        data.groupby(["genotype", "block_name", "global_track_id"], dropna=False)
        .size()
        .reset_index(name="n_timepoints")
    )
    track_counts.to_csv(out / "shape_over_time_track_counts.csv", index=False)
    print(f"[DONE] {label}: shape-over-time plots saved to {out}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    qc_paths = parse_mapping(args.qc_track_table)
    for index, dataset in enumerate(args.dataset):
        run_dataset(dataset, root, qc_paths, output_root, args, index)
    print(f"[DONE] All shape-over-time plots saved under: {output_root}")


if __name__ == "__main__":
    main()
