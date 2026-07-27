#!/usr/bin/env python3
"""Compare untreated and treated manual injury ROI position and size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treated-condition", default="MMP9_inhibited")
    parser.add_argument("--treated-label", default="MMP9 inhibited")
    parser.add_argument(
        "--treated-annotation-root",
        default=str(ROOT / "manual_injury_annotations_mmp"),
    )
    parser.add_argument(
        "--treated-metadata",
        default=str(ROOT / "MMP_analysis_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "frozen_mmp_results" / "combined_analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [
        ("untreated", ROOT / "manual_injury_annotations", ROOT / "block_metadata.csv"),
        (
            args.treated_condition,
            Path(args.treated_annotation_root),
            Path(args.treated_metadata),
        ),
    ]
    output_dir = Path(args.output_dir)
    rows = []
    for condition, annotation_root, metadata_path in sources:
        metadata = pd.read_csv(metadata_path)
        genotype = dict(zip(metadata.block_name.astype(str), metadata.genotype.astype(str)))
        allowed = set(metadata.block_name.astype(str))
        for mask_path in sorted(annotation_root.glob("*/manual_injury_roi_mask_YX.tif")):
            fish = mask_path.parent.name
            if fish not in allowed:
                continue
            mask = np.asarray(tiff.imread(mask_path)).astype(bool)
            y, x = np.nonzero(mask)
            if not len(x):
                continue
            metadata_json = mask_path.with_name("manual_injury_roi_metadata.json")
            details = json.loads(metadata_json.read_text()) if metadata_json.exists() else {}
            xy_um = float(details.get("xy_pixel_size_um", np.nan))
            rows.append(
                {
                    "condition": condition,
                    "fish_id": fish,
                    "genotype": genotype.get(fish, ""),
                    "image_height_px": mask.shape[0],
                    "image_width_px": mask.shape[1],
                    "roi_area_pixels": int(mask.sum()),
                    "roi_area_fraction": float(mask.mean()),
                    "roi_area_um2": float(mask.sum() * xy_um**2),
                    "centroid_x_fraction": float(x.mean() / (mask.shape[1] - 1)),
                    "centroid_y_fraction": float(y.mean() / (mask.shape[0] - 1)),
                }
            )

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "injury_roi_position_area_audit.csv", index=False)
    summary = (
        table.groupby(["condition", "genotype"])[
            ["centroid_x_fraction", "centroid_y_fraction", "roi_area_fraction", "roi_area_um2"]
        ]
        .agg(["count", "mean", "std", "min", "max"])
    )
    summary.to_csv(output_dir / "injury_roi_position_area_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("centroid_x_fraction", "ROI centroid X / image width"),
        ("centroid_y_fraction", "ROI centroid Y / image height"),
        ("roi_area_fraction", "ROI area / image area"),
    ]
    groups = [
        ("untreated", "WT"), ("untreated", "MUT"),
        (args.treated_condition, "WT"), (args.treated_condition, "MUT"),
    ]
    labels = [
        "Untreated\nWT",
        "Untreated\nMUT",
        f"{args.treated_label}\nWT",
        f"{args.treated_label}\nMUT",
    ]
    rng = np.random.default_rng(42)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for index, (condition, genotype) in enumerate(groups):
            values = table.loc[
                (table.condition == condition) & (table.genotype == genotype), metric
            ].to_numpy(float)
            ax.scatter(index + rng.uniform(-0.08, 0.08, len(values)), values, s=45)
            if len(values):
                ax.hlines(np.mean(values), index - 0.2, index + 0.2, linewidth=3)
        ax.set_xticks(range(4), labels, rotation=20)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Manual injury ROI position and size audit")
    fig.tight_layout()
    fig.savefig(output_dir / "injury_roi_position_area_audit.png", dpi=220)
    plt.close(fig)
    print(f"[DONE] ROI audit: {output_dir}")


if __name__ == "__main__":
    main()
