#!/usr/bin/env python3
"""Replot the MUSC MSD time course after excluding one prespecified fish."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "WT": "#1f77b4",
    "MUT": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot MUSC MSD elapsed-time course after excluding selected fish/block IDs."
    )
    parser.add_argument(
        "--points",
        default="plots_time_corrected/msd_directionality_fig1_style/MUSC/msd_timecourse_points.csv",
        help="Input msd_timecourse_points.csv from make_msd_directionality_plots.py.",
    )
    parser.add_argument("--exclude-fish", action="append", required=True)
    parser.add_argument(
        "--output-dir",
        default="plots_time_corrected/msd_directionality_fig1_style/MUSC/exclude_20240422_block06_20240422_block08",
    )
    parser.add_argument("--title", default="MUSC: MSD over elapsed track time")
    return parser.parse_args()


def block_from_global_track_id(value: object) -> str:
    return str(value).split("::", 1)[0]


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (genotype, time_bin), sub in points.groupby(["genotype", "time_bin_minutes"], dropna=True):
        values = pd.to_numeric(sub["squared_displacement_um2"], errors="coerce").dropna().to_numpy(float)
        if len(values) == 0:
            continue
        rows.append(
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
    return pd.DataFrame(rows).sort_values(["genotype", "elapsed_minutes"])


def summarize_by_fish(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fish_time = (
        points.groupby(["genotype", "block_name", "time_bin_minutes"], dropna=True)
        .agg(
            fish_mean_msd_um2=("squared_displacement_um2", "mean"),
            fish_median_msd_um2=("squared_displacement_um2", "median"),
            n_observations=("squared_displacement_um2", "size"),
            n_tracks=("global_track_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"time_bin_minutes": "elapsed_minutes"})
    )

    rows: list[dict[str, object]] = []
    for (genotype, time_bin), sub in fish_time.groupby(["genotype", "elapsed_minutes"], dropna=True):
        values = pd.to_numeric(sub["fish_mean_msd_um2"], errors="coerce").dropna().to_numpy(float)
        if len(values) == 0:
            continue
        rows.append(
            {
                "genotype": genotype,
                "elapsed_minutes": float(time_bin),
                "mean_msd_um2": float(np.nanmean(values)),
                "median_msd_um2": float(np.nanmedian(values)),
                "sem_msd_um2": float(np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "q25_msd_um2": float(np.nanpercentile(values, 25)),
                "q75_msd_um2": float(np.nanpercentile(values, 75)),
                "n_fish": int(sub["block_name"].nunique()),
                "n_tracks": int(sub["n_tracks"].sum()),
                "n_observations": int(sub["n_observations"].sum()),
            }
        )
    return fish_time, pd.DataFrame(rows).sort_values(["genotype", "elapsed_minutes"])


def plot_summary(summary: pd.DataFrame, output_path: Path, title: str, excluded: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for genotype in ["WT", "MUT"]:
        group = summary[summary["genotype"].eq(genotype)].sort_values("elapsed_minutes")
        if group.empty:
            continue
        ax.errorbar(
            group["elapsed_minutes"],
            group["mean_msd_um2"],
            yerr=group["sem_msd_um2"],
            color=COLORS.get(genotype, "black"),
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            capsize=3,
            label=f"{genotype} mean +/- SEM",
        )
    ax.set_title(title, fontsize=20, pad=12)
    ax.set_xlabel("Elapsed time from track start (min)", fontsize=15)
    ax.set_ylabel("MSD (um2)", fontsize=15)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Excluded: " + ", ".join(excluded),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_median_summary(summary: pd.DataFrame, output_path: Path, title: str, excluded: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for genotype in ["WT", "MUT"]:
        group = summary[summary["genotype"].eq(genotype)].sort_values("elapsed_minutes")
        if group.empty:
            continue
        y = group["median_msd_um2"].to_numpy(float)
        lower = y - group["q25_msd_um2"].to_numpy(float)
        upper = group["q75_msd_um2"].to_numpy(float) - y
        ax.errorbar(
            group["elapsed_minutes"],
            y,
            yerr=np.vstack([lower, upper]),
            color=COLORS.get(genotype, "black"),
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            capsize=3,
            label=f"{genotype} median + IQR",
        )
    ax.set_title(title, fontsize=20, pad=12)
    ax.set_xlabel("Elapsed time from track start (min)", fontsize=15)
    ax.set_ylabel("Median MSD (um2)", fontsize=15)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Excluded: " + ", ".join(excluded),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    points_path = Path(args.points)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(points_path)
    points["block_name"] = points["global_track_id"].map(block_from_global_track_id)
    excluded = set(args.exclude_fish)
    filtered = points[~points["block_name"].isin(excluded)].copy()

    summary = summarize(filtered)
    fish_time, fish_normalized_summary = summarize_by_fish(filtered)
    filtered.to_csv(output_dir / "msd_timecourse_points_excluding_fish.csv", index=False)
    summary.to_csv(output_dir / "msd_timecourse_summary_excluding_fish.csv", index=False)
    fish_time.to_csv(output_dir / "msd_timecourse_fish_level_points_excluding_fish.csv", index=False)
    fish_normalized_summary.to_csv(output_dir / "msd_timecourse_fish_normalized_summary_excluding_fish.csv", index=False)
    plot_summary(summary, output_dir / "musc_msd_timecourse_excluding_fish.png", args.title, args.exclude_fish)
    plot_median_summary(
        summary,
        output_dir / "musc_msd_timecourse_median_iqr_excluding_fish.png",
        args.title + " (median)",
        args.exclude_fish,
    )
    plot_summary(
        fish_normalized_summary,
        output_dir / "musc_msd_timecourse_fish_normalized_excluding_fish.png",
        args.title + " (fish-level normalized)",
        args.exclude_fish,
    )
    plot_median_summary(
        fish_normalized_summary,
        output_dir / "musc_msd_timecourse_fish_normalized_median_iqr_excluding_fish.png",
        args.title + " (fish-level median)",
        args.exclude_fish,
    )

    counts = (
        points.assign(is_excluded=points["block_name"].isin(excluded))
        .groupby(["block_name", "genotype", "is_excluded"], dropna=False)
        .agg(n_observations=("squared_displacement_um2", "size"), n_tracks=("global_track_id", "nunique"))
        .reset_index()
        .sort_values(["is_excluded", "genotype", "block_name"], ascending=[False, True, True])
    )
    counts.to_csv(output_dir / "included_excluded_point_counts_by_fish.csv", index=False)
    print(f"Saved: {output_dir / 'musc_msd_timecourse_excluding_fish.png'}")
    print(f"Saved: {output_dir / 'musc_msd_timecourse_median_iqr_excluding_fish.png'}")
    print(f"Saved: {output_dir / 'musc_msd_timecourse_fish_normalized_excluding_fish.png'}")
    print(f"Saved: {output_dir / 'musc_msd_timecourse_fish_normalized_median_iqr_excluding_fish.png'}")
    print(f"Saved: {output_dir / 'msd_timecourse_summary_excluding_fish.csv'}")


if __name__ == "__main__":
    main()
