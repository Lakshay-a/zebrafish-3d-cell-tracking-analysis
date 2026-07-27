#!/usr/bin/env python3
"""Inspect the MUSC mean-squared-displacement drop using saved feature tables."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import pandas as pd


MUSC_DIR = Path("plots_time_corrected/msd_directionality_fig1_style/MUSC")
OUTPUT_DIR = MUSC_DIR / "drop_investigation"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(MUSC_DIR / "msd_timecourse_summary.csv")
    points = pd.read_csv(MUSC_DIR / "msd_timecourse_points.csv")
    metadata = pd.read_csv("block_metadata.csv", low_memory=False)

    points["block_name"] = points["global_track_id"].str.split("::").str[0]
    metadata_cols = [
        col
        for col in ["block_name", "genotype", "time_interval_seconds"]
        if col in metadata.columns
    ]
    block_metadata = metadata[metadata_cols].drop_duplicates("block_name")

    late_summary = summary[summary["elapsed_minutes"] >= 600].copy()
    late_summary.to_csv(OUTPUT_DIR / "musc_msd_summary_after_600_min.csv", index=False)

    by_block = (
        points[points["time_bin_minutes"] >= 690]
        .groupby(["genotype", "time_bin_minutes", "block_name"])
        .agg(
            n_observations=("squared_displacement_um2", "size"),
            n_tracks=("global_track_id", "nunique"),
            mean_msd_um2=("squared_displacement_um2", "mean"),
            median_msd_um2=("squared_displacement_um2", "median"),
        )
        .reset_index()
        .merge(block_metadata, on="block_name", how="left", suffixes=("", "_metadata"))
        .sort_values(["time_bin_minutes", "genotype", "block_name"])
    )
    by_block.to_csv(OUTPUT_DIR / "musc_late_msd_by_block_after_690_min.csv", index=False)

    track_duration = (
        points.groupby(["genotype", "block_name", "global_track_id"])
        .agg(
            max_time_bin_minutes=("time_bin_minutes", "max"),
            n_observations=("time_bin_minutes", "size"),
            mean_track_msd_um2=("squared_displacement_um2", "mean"),
            median_track_msd_um2=("squared_displacement_um2", "median"),
        )
        .reset_index()
    )
    track_by_block = (
        track_duration.groupby(["genotype", "block_name"])
        .agg(
            n_tracks=("global_track_id", "nunique"),
            median_max_time_bin_minutes=("max_time_bin_minutes", "median"),
            max_time_bin_minutes=("max_time_bin_minutes", "max"),
            n_tracks_ge_690=("max_time_bin_minutes", lambda s: int((s >= 690).sum())),
            n_tracks_ge_750=("max_time_bin_minutes", lambda s: int((s >= 750).sum())),
            n_tracks_ge_900=("max_time_bin_minutes", lambda s: int((s >= 900).sum())),
            median_track_mean_msd_um2=("mean_track_msd_um2", "median"),
        )
        .reset_index()
        .merge(block_metadata, on="block_name", how="left", suffixes=("", "_metadata"))
        .sort_values(["genotype", "max_time_bin_minutes"], ascending=[True, False])
    )
    track_by_block.to_csv(OUTPUT_DIR / "musc_track_duration_and_late_contribution_by_block.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    for genotype, group in summary.groupby("genotype"):
        ax.errorbar(
            group["elapsed_minutes"],
            group["mean_msd_um2"],
            yerr=group["sem_msd_um2"],
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=f"{genotype} mean +/- SEM",
        )
    ax.axvline(720, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.axvline(750, color="black", linestyle=":", linewidth=1.1, alpha=0.7)
    ax.text(724, ax.get_ylim()[1] * 0.92, "720 min", fontsize=9)
    ax.text(754, ax.get_ylim()[1] * 0.84, "750 min", fontsize=9)
    ax.set_title("MUSC MSD: drop occurs when contributing blocks change")
    ax.set_xlabel("Elapsed time from track start (min)")
    ax.set_ylabel("MSD (um2)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_musc_msd_drop_marked.png", dpi=260)
    plt.close(fig)

    for genotype in ["MUT", "WT"]:
        subset = by_block[by_block["genotype"] == genotype]
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index="time_bin_minutes",
            columns="block_name",
            values="n_tracks",
            aggfunc="sum",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        pivot.plot(kind="bar", stacked=True, ax=ax, width=0.88)
        ax.set_title(f"MUSC {genotype}: late contributing tracks by block")
        ax.set_xlabel("Elapsed time bin (min)")
        ax.set_ylabel("Number of contributing tracks")
        ax.legend(title="Block", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"02_{genotype}_late_track_counts_by_block.png", dpi=260)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for (genotype, block_name), group in by_block.groupby(["genotype", "block_name"]):
        linestyle = "-" if genotype == "WT" else "--"
        ax.plot(
            group["time_bin_minutes"],
            group["mean_msd_um2"],
            marker="o",
            linewidth=1.8,
            linestyle=linestyle,
            label=f"{genotype} {block_name}",
        )
    ax.set_title("MUSC late MSD by contributing block")
    ax.set_xlabel("Elapsed time bin (min)")
    ax.set_ylabel("Mean MSD (um2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_late_msd_by_block.png", dpi=260)
    plt.close(fig)

    print(f"Saved diagnostics to: {OUTPUT_DIR}")
    print()
    print("Late MUT blocks after 750 min:")
    print(
        track_by_block[
            (track_by_block["genotype"] == "MUT")
            & (track_by_block["n_tracks_ge_750"] > 0)
        ][
            [
                "block_name",
                "time_interval_seconds",
                "n_tracks",
                "max_time_bin_minutes",
                "n_tracks_ge_750",
                "n_tracks_ge_900",
                "median_track_mean_msd_um2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
