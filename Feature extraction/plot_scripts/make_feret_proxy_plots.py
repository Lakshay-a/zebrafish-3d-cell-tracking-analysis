#!/usr/bin/env python3
"""Plot time-resolved cell-length proxies derived from three-dimensional shape."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


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

GENOTYPE_COLORS = {"WT": "#1f77b4", "MUT": "#d62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Feret-distance proxy / major-axis length from raw per-timepoint cell shapes."
    )
    parser.add_argument("--root", default="final_feature_outputs")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_PATTERNS), required=True)
    parser.add_argument("--qc-track-table", action="append", default=[], metavar="DATASET=CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-track-points", type=int, default=4)
    parser.add_argument("--time-bins", type=int, default=21)
    parser.add_argument("--max-cell-points-per-fish", type=int, default=140)
    parser.add_argument("--max-scatter-points", type=int, default=3500)
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


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalise_genotype(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", upper) or "WILD TYPE" in upper:
        return "WT"
    if "MUT" in upper:
        return "MUT"
    return text


def load_qc_keys(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    table = pd.read_csv(path, usecols=["block_name", "track_id"], low_memory=False)
    return set(zip(table["block_name"].astype(str), table["track_id"].astype(str)))


def load_dataset(root: Path, dataset: str, qc_path: Path | None) -> pd.DataFrame:
    files = sorted(root.glob(f"*/{DATASET_PATTERNS[dataset]}"))
    if not files:
        raise FileNotFoundError(f"No object timepoint files found for {dataset} under {root}")

    needed = {
        "time",
        "track_id",
        "block_name",
        "fish_id",
        "genotype",
        "principal_axis_length_1_um",
        "principal_axis_length_2_um",
        "principal_axis_length_3_um",
        "bbox_x_length_um",
        "bbox_y_length_um",
        "bbox_z_length_um",
        "elongation",
        "aspect_ratio_3d",
        "sphericity",
        "volume_um3",
    }
    frames: list[pd.DataFrame] = []
    for path in files:
        header = pd.read_csv(path, nrows=0)
        usecols = [column for column in header.columns if column in needed]
        missing_base = {"time", "track_id", "block_name", "fish_id", "genotype"} - set(usecols)
        if missing_base:
            print(f"[WARN] Skipping {path}; missing {sorted(missing_base)}")
            continue
        frames.append(pd.read_csv(path, usecols=usecols, low_memory=False))
    if not frames:
        raise ValueError(f"No usable object timepoint files found for {dataset}")

    data = pd.concat(frames, ignore_index=True)
    data["block_name"] = data["block_name"].astype(str)
    data["track_id"] = data["track_id"].astype(str)
    data["fish_id"] = data["fish_id"].astype(str)
    data["genotype"] = data["genotype"].map(normalise_genotype)
    data = data[data["genotype"].isin(["WT", "MUT"])].copy()

    for column in [
        "time",
        "principal_axis_length_1_um",
        "principal_axis_length_2_um",
        "principal_axis_length_3_um",
        "bbox_x_length_um",
        "bbox_y_length_um",
        "bbox_z_length_um",
        "elongation",
        "aspect_ratio_3d",
        "sphericity",
        "volume_um3",
    ]:
        if column in data.columns:
            data[column] = numeric(data[column])

    axis_cols = [
        column
        for column in [
            "principal_axis_length_1_um",
            "principal_axis_length_2_um",
            "principal_axis_length_3_um",
        ]
        if column in data.columns
    ]
    if axis_cols and data[axis_cols].notna().any().any():
        data["max_feret_proxy_um"] = data[axis_cols].max(axis=1)
        data["min_feret_proxy_um"] = data[axis_cols].min(axis=1)
        data["mid_feret_proxy_um"] = data[axis_cols].median(axis=1) if len(axis_cols) >= 3 else np.nan
        data["feret_proxy_um"] = data["max_feret_proxy_um"]
        data["feret_proxy_source"] = "principal_axis_lengths_1_2_3_um"
    else:
        bbox_cols = [c for c in ["bbox_x_length_um", "bbox_y_length_um", "bbox_z_length_um"] if c in data.columns]
        if not bbox_cols:
            raise ValueError(f"No principal-axis or bbox length columns found for {dataset}")
        data["max_feret_proxy_um"] = data[bbox_cols].max(axis=1)
        data["min_feret_proxy_um"] = data[bbox_cols].min(axis=1)
        data["mid_feret_proxy_um"] = data[bbox_cols].median(axis=1) if len(bbox_cols) >= 3 else np.nan
        data["feret_proxy_um"] = data["max_feret_proxy_um"]
        data["feret_proxy_source"] = "max_bbox_dimension_um"

    data["feret_ratio_3d"] = data["max_feret_proxy_um"] / data["min_feret_proxy_um"].replace(0, np.nan)

    data = data.dropna(subset=["time", "feret_proxy_um"])
    qc_keys = load_qc_keys(qc_path)
    if qc_keys is not None:
        before = len(data)
        data = data[[(b, t) in qc_keys for b, t in zip(data["block_name"], data["track_id"])]].copy()
        print(f"[INFO] {dataset}: QC filter kept {len(data):,}/{before:,} object timepoints.")

    data["global_track_id"] = data["block_name"] + "::" + data["track_id"]
    counts = data.groupby("global_track_id")["time"].nunique()
    keep_tracks = set(counts[counts >= 2].index)
    data = data[data["global_track_id"].isin(keep_tracks)].copy()
    return data.sort_values(["global_track_id", "time"]).reset_index(drop=True)


def build_track_summary(data: pd.DataFrame, min_track_points: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for track_id, track in data.groupby("global_track_id", sort=False):
        track = track.sort_values("time")
        if len(track) < min_track_points:
            continue
        max_feret = track["max_feret_proxy_um"].dropna().to_numpy(float)
        min_feret = track["min_feret_proxy_um"].dropna().to_numpy(float)
        ratio = track["feret_ratio_3d"].dropna().to_numpy(float)
        if len(max_feret) == 0 or len(min_feret) == 0:
            continue
        row: dict[str, object] = {
            "block_name": str(track["block_name"].iloc[0]),
            "fish_id": str(track["fish_id"].iloc[0]),
            "genotype": str(track["genotype"].iloc[0]),
            "track_id": str(track["track_id"].iloc[0]),
            "global_track_id": track_id,
            "n_timepoints": int(len(track)),
            "mean_max_feret_proxy_um": float(np.nanmean(max_feret)),
            "median_max_feret_proxy_um": float(np.nanmedian(max_feret)),
            "max_observed_max_feret_proxy_um": float(np.nanmax(max_feret)),
            "max_feret_proxy_change_start_to_end_um": float(max_feret[-1] - max_feret[0]) if len(max_feret) > 1 else np.nan,
            "mean_min_feret_proxy_um": float(np.nanmean(min_feret)),
            "median_min_feret_proxy_um": float(np.nanmedian(min_feret)),
            "min_observed_min_feret_proxy_um": float(np.nanmin(min_feret)),
            "min_feret_proxy_change_start_to_end_um": float(min_feret[-1] - min_feret[0]) if len(min_feret) > 1 else np.nan,
            "mean_feret_ratio_3d": float(np.nanmean(ratio)) if len(ratio) else np.nan,
            "median_feret_ratio_3d": float(np.nanmedian(ratio)) if len(ratio) else np.nan,
            "max_feret_ratio_3d": float(np.nanmax(ratio)) if len(ratio) else np.nan,
            "mean_elongation": float(np.nanmean(track["elongation"])) if "elongation" in track.columns else np.nan,
            "mean_aspect_ratio_3d": float(np.nanmean(track["aspect_ratio_3d"])) if "aspect_ratio_3d" in track.columns else np.nan,
            "mean_sphericity": float(np.nanmean(track["sphericity"])) if "sphericity" in track.columns else np.nan,
            "mean_volume_um3": float(np.nanmean(track["volume_um3"])) if "volume_um3" in track.columns else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fish_order(table: pd.DataFrame) -> pd.DataFrame:
    return table[["fish_id", "genotype"]].drop_duplicates().sort_values(["genotype", "fish_id"]).reset_index(drop=True)


def plot_metric_by_fish(
    label: str,
    tracks: pd.DataFrame,
    out: Path,
    metric: str,
    ylabel: str,
    title_suffix: str,
    filename: str,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    order = fish_order(tracks)
    fig, ax = plt.subplots(figsize=(max(11.0, 0.52 * len(order) + 3), 5.8))
    for x_pos, row in order.iterrows():
        fish = str(row["fish_id"])
        genotype = str(row["genotype"])
        values = tracks.loc[tracks["fish_id"].astype(str).eq(fish), metric].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        plotted = values
        if max_points > 0 and len(plotted) > max_points:
            plotted = rng.choice(plotted, size=max_points, replace=False)
        parts = ax.violinplot(values, positions=[x_pos], widths=0.72, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(GENOTYPE_COLORS.get(genotype, "#666666"))
            body.set_alpha(0.23)
            body.set_edgecolor("black")
            body.set_linewidth(0.7)
        ax.scatter(
            x_pos + rng.normal(0, 0.06, len(plotted)),
            plotted,
            s=12,
            alpha=0.45,
            color=GENOTYPE_COLORS.get(genotype, "#666666"),
            linewidths=0,
        )
        ax.hlines(np.nanmedian(values), x_pos - 0.24, x_pos + 0.24, color="black", linewidth=2.2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{r.fish_id}\n{r.genotype}" for r in order.itertuples()], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{label}: {title_suffix} by fish")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GENOTYPE_COLORS["WT"], label="WT", markersize=8),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GENOTYPE_COLORS["MUT"], label="MUT", markersize=8),
            Line2D([0], [0], color="black", linewidth=2.2, label="Fish median"),
        ],
        frameon=True,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_feret_by_fish(label: str, tracks: pd.DataFrame, out: Path, max_points: int, seed: int) -> None:
    plot_metric_by_fish(
        label,
        tracks,
        out,
        "mean_max_feret_proxy_um",
        "Max Feret proxy / major axis length (um)",
        "cell max Feret-distance proxy",
        "01_max_feret_proxy_by_fish.png",
        max_points,
        seed,
    )
    plot_metric_by_fish(
        label,
        tracks,
        out,
        "mean_min_feret_proxy_um",
        "Min Feret proxy / minor axis length (um)",
        "cell min Feret-distance proxy",
        "02_min_feret_proxy_by_fish.png",
        max_points,
        seed + 1000,
    )
    plot_metric_by_fish(
        label,
        tracks,
        out,
        "mean_feret_ratio_3d",
        "Max/min Feret proxy ratio",
        "3D Feret proxy ratio",
        "03_feret_ratio_by_fish.png",
        max_points,
        seed + 2000,
    )


def plot_feret_scatter(label: str, tracks: pd.DataFrame, out: Path, max_points: int, seed: int) -> None:
    table = tracks.dropna(subset=["mean_max_feret_proxy_um", "mean_min_feret_proxy_um"]).copy()
    if table.empty:
        return
    rng = np.random.default_rng(seed)
    if max_points > 0 and len(table) > max_points:
        table = table.sample(max_points, random_state=seed)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    for genotype in ["WT", "MUT"]:
        sub = table[table["genotype"].eq(genotype)]
        if sub.empty:
            continue
        ax.scatter(
            sub["mean_max_feret_proxy_um"],
            sub["mean_min_feret_proxy_um"],
            s=14,
            alpha=0.34,
            color=GENOTYPE_COLORS.get(genotype, "#666666"),
            linewidths=0,
            label=genotype,
        )
        median_x = float(np.nanmedian(sub["mean_max_feret_proxy_um"]))
        median_y = float(np.nanmedian(sub["mean_min_feret_proxy_um"]))
        ax.scatter(median_x, median_y, s=95, color=GENOTYPE_COLORS.get(genotype, "#666666"), edgecolor="black", zorder=3)
        ax.text(median_x, median_y, f" {genotype} median", va="center", fontsize=9)
    ax.set_xlabel("Max Feret proxy / major axis length (um)")
    ax.set_ylabel("Min Feret proxy / minor axis length (um)")
    ax.set_title(f"{label}: 3D max vs min Feret-distance proxy")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "04_max_vs_min_feret_proxy.png", dpi=260)
    plt.close(fig)

    ratio_table = tracks.dropna(subset=["mean_feret_ratio_3d", "mean_sphericity"]).copy()
    if ratio_table.empty:
        return
    if max_points > 0 and len(ratio_table) > max_points:
        ratio_table = ratio_table.sample(max_points, random_state=seed + 1)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    for genotype in ["WT", "MUT"]:
        sub = ratio_table[ratio_table["genotype"].eq(genotype)]
        if sub.empty:
            continue
        ax.scatter(
            sub["mean_feret_ratio_3d"],
            sub["mean_sphericity"],
            s=14,
            alpha=0.34,
            color=GENOTYPE_COLORS.get(genotype, "#666666"),
            linewidths=0,
            label=genotype,
        )
        median_x = float(np.nanmedian(sub["mean_feret_ratio_3d"]))
        median_y = float(np.nanmedian(sub["mean_sphericity"]))
        ax.scatter(median_x, median_y, s=95, color=GENOTYPE_COLORS.get(genotype, "#666666"), edgecolor="black", zorder=3)
        ax.text(median_x, median_y, f" {genotype} median", va="center", fontsize=9)
    ax.set_xlabel("Max/min Feret proxy ratio")
    ax.set_ylabel("Mean sphericity")
    ax.set_title(f"{label}: Feret ratio vs roundness")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "05_feret_ratio_vs_sphericity.png", dpi=260)
    plt.close(fig)


def plot_feret_over_time(label: str, data: pd.DataFrame, out: Path, time_bins: int) -> pd.DataFrame:
    table = data.copy()
    grouped = table.groupby("global_track_id")
    start = grouped["time"].transform("min")
    duration = grouped["time"].transform(lambda s: s.max() - s.min())
    table["relative_time"] = np.where(duration > 0, (table["time"] - start) / duration, np.nan)
    table = table.dropna(subset=["relative_time"])
    table["time_bin"] = pd.cut(
        table["relative_time"],
        bins=np.linspace(0, 1, time_bins + 1),
        labels=np.linspace(0, 1, time_bins),
        include_lowest=True,
    ).astype(float)
    rows: list[dict[str, object]] = []
    metrics = [
        ("max_feret_proxy_um", "max_feret_proxy_um"),
        ("min_feret_proxy_um", "min_feret_proxy_um"),
        ("feret_ratio_3d", "feret_ratio_3d"),
    ]
    for (genotype, time_bin), sub in table.groupby(["genotype", "time_bin"], dropna=True):
        for column, metric in metrics:
            values = sub[column].dropna().to_numpy(float)
            if len(values) == 0:
                continue
            rows.append(
                {
                    "genotype": genotype,
                    "relative_time": float(time_bin),
                    "metric": metric,
                    "median": float(np.nanmedian(values)),
                    "q25": float(np.nanpercentile(values, 25)),
                    "q75": float(np.nanpercentile(values, 75)),
                    "n_object_timepoints": int(len(values)),
                    "n_tracks": int(sub["global_track_id"].nunique()),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    plot_specs = [
        ("max_feret_proxy_um", "Max Feret proxy / major axis length (um)", "06_max_feret_proxy_over_time.png"),
        ("min_feret_proxy_um", "Min Feret proxy / minor axis length (um)", "07_min_feret_proxy_over_time.png"),
        ("feret_ratio_3d", "Max/min Feret proxy ratio", "08_feret_ratio_over_time.png"),
    ]
    for metric, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for genotype in ["WT", "MUT"]:
            sm = summary[(summary["genotype"].eq(genotype)) & (summary["metric"].eq(metric))].sort_values("relative_time")
            if sm.empty:
                continue
            color = GENOTYPE_COLORS.get(genotype, "#666666")
            ax.fill_between(sm["relative_time"], sm["q25"], sm["q75"], color=color, alpha=0.17, linewidth=0)
            ax.plot(sm["relative_time"], sm["median"], color=color, linewidth=2.8, marker="o", markersize=4, label=genotype)
        ax.set_xlabel("Relative track time (0=start, 1=end)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label}: {ylabel} over tracked-cell lifetime")
        ax.grid(alpha=0.22)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=260)
        plt.close(fig)
    return summary


def run_dataset(dataset: str, root: Path, qc_paths: dict[str, Path], output_root: Path, args: argparse.Namespace, index: int) -> None:
    label = DISPLAY_NAMES.get(dataset, dataset)
    out = output_root / safe_name(label)
    out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(root, dataset, qc_paths.get(dataset))
    tracks = build_track_summary(data, args.min_track_points)
    if tracks.empty:
        print(f"[WARN] No usable tracks for {dataset}.")
        return
    tracks.to_csv(out / "feret_3d_proxy_track_summary.csv", index=False)
    plot_feret_by_fish(label, tracks, out, args.max_cell_points_per_fish, args.random_seed + index)
    plot_feret_scatter(label, tracks, out, args.max_scatter_points, args.random_seed + 100 + index)
    time_summary = plot_feret_over_time(label, data, out, args.time_bins)
    if not time_summary.empty:
        time_summary.to_csv(out / "feret_3d_proxy_over_time_summary.csv", index=False)
    print(f"[DONE] {label}: Feret proxy plots saved to {out}")


# The plotted major-axis length is an explicitly labelled 3D Feret proxy, not
# an exact caliper diameter. Region-property definitions:
# https://scikit-image.org/docs/stable/api/skimage.measure.html
def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    qc_paths = parse_mapping(args.qc_track_table)
    for index, dataset in enumerate(args.dataset):
        run_dataset(dataset, Path(args.root), qc_paths, output_root, args, index)
    print(f"[DONE] All Feret proxy plots saved under: {output_root}")


if __name__ == "__main__":
    main()
