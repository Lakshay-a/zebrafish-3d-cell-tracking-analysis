#!/usr/bin/env python3
"""Test fish-level imaging proxies for injury retention and recruitment.

The analysis addresses two imaging-level hypotheses in untreated fish:

* tert-mutant macrophages show greater persistence at the injury;
* tert-mutant muSCs show poorer recruitment to the injury.

NAMPT secretion and CCR5 activation are not measured and therefore cannot be
tested here. Track measurements are first aggregated within fish so that the
fish, rather than each tracked cell, remains the independent replicate.

Two-sided exact genotype-label permutation tests compare fish means. Cliff's
delta and fish-level bootstrap confidence intervals describe effect size.
Benjamini-Hochberg correction is applied separately to macrophage and muSC
proxy families.

References
----------
SciPy permutation-test principles:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html

Benjamini and Hochberg (1995), false-discovery-rate control:
https://www.jstor.org/stable/2346101

Lazic et al. (2018), identifying the experimental unit:
https://doi.org/10.1371/journal.pbio.2005282
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "mean_longest_inside_duration_minutes": (
        "longest_inside_duration_minutes",
        "mean",
        "Mean longest continuous time inside injury ROI (min)",
    ),
    "median_longest_inside_duration_minutes": (
        "longest_inside_duration_minutes",
        "median",
        "Median longest continuous time inside injury ROI (min)",
    ),
    "fraction_tracks_ever_inside": (
        "ever_inside",
        "mean",
        "Fraction of tracks that ever entered injury ROI",
    ),
    "fraction_tracks_end_inside": (
        "ends_inside",
        "mean",
        "Fraction of tracks ending inside injury ROI",
    ),
    "mean_fraction_detections_inside": (
        "fraction_detections_inside",
        "mean",
        "Mean fraction of detections inside injury ROI",
    ),
    "mean_entries_per_track": (
        "entry_count",
        "mean",
        "Mean injury-ROI entries per track",
    ),
    "mean_exits_per_track": (
        "exit_count",
        "mean",
        "Mean injury-ROI exits per track",
    ),
    "median_min_distance_to_injury_um": (
        "min_abs_distance_to_injury_um",
        "median",
        "Median minimum distance to injury (µm)",
    ),
    "median_first_entry_time_minutes": (
        "first_entry_time_minutes",
        "median",
        "Median first-entry time among entering tracks (min)",
    ),
    "median_net_approach_um": (
        "net_approach_um",
        "median",
        "Median net approach to injury (µm)",
    ),
}

# Directions are fixed from the biological hypotheses before looking at the
# statistical result. Ambiguous movement summaries remain two-sided only.
DIRECTIONAL_HYPOTHESES = {
    "macrophage": {
        "mean_longest_inside_duration_minutes": "MUT_higher",
        "fraction_tracks_end_inside": "MUT_higher",
        "mean_fraction_detections_inside": "MUT_higher",
    },
    "musc": {
        "fraction_tracks_ever_inside": "MUT_lower",
        "fraction_tracks_end_inside": "MUT_lower",
        "mean_fraction_detections_inside": "MUT_lower",
        "mean_entries_per_track": "MUT_lower",
        "median_min_distance_to_injury_um": "MUT_higher",
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_genotype(value: object) -> str:
    text = str(value).upper()
    if "MUT" in text:
        return "MUT"
    if re.search(r"(^|[^A-Z])WT([^A-Z]|$)", text):
        return "WT"
    return str(value)


def aggregate_fish(per_track_dir: Path, metadata: pd.DataFrame, cell_type: str) -> pd.DataFrame:
    rows = []
    genotype_map = metadata.set_index("block_name")["genotype"].map(safe_genotype)
    for path in sorted(per_track_dir.glob("*.csv")):
        suffix = f"__{cell_type}_injury_tracks"
        fish_id = path.stem.removesuffix(suffix)
        if fish_id not in genotype_map.index:
            continue
        tracks = pd.read_csv(path, low_memory=False)
        row: dict[str, object] = {
            "cell_type": cell_type,
            "fish_id": fish_id,
            "genotype": genotype_map.loc[fish_id],
            "n_tracks": len(tracks),
        }
        for output_name, (column, operation, _) in METRICS.items():
            values = pd.to_numeric(tracks[column], errors="coerce").dropna()
            if values.empty:
                row[output_name] = np.nan
            elif operation == "mean":
                row[output_name] = float(values.mean())
            else:
                row[output_name] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(mutant: np.ndarray, wild_type: np.ndarray) -> float:
    differences = mutant[:, None] - wild_type[None, :]
    return float((np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size)


def exact_difference_test(
    mutant: np.ndarray, wild_type: np.ndarray
) -> tuple[float, float, float, float, int]:
    combined = np.concatenate([mutant, wild_type])
    observed = float(mutant.mean() - wild_type.mean())
    n_mutant = len(mutant)
    exceedances = 0
    greater = 0
    lower = 0
    assignments = 0
    for indexes in itertools.combinations(range(len(combined)), n_mutant):
        mask = np.zeros(len(combined), dtype=bool)
        mask[list(indexes)] = True
        statistic = combined[mask].mean() - combined[~mask].mean()
        exceedances += abs(statistic) >= abs(observed) - 1e-15
        greater += statistic >= observed - 1e-15
        lower += statistic <= observed + 1e-15
        assignments += 1
    return (
        observed,
        float(exceedances / assignments),
        float(greater / assignments),
        float(lower / assignments),
        assignments,
    )


def bootstrap_delta_interval(
    mutant: np.ndarray,
    wild_type: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.empty(iterations)
    for index in range(iterations):
        sampled_mutant = rng.choice(mutant, len(mutant), replace=True)
        sampled_wild_type = rng.choice(wild_type, len(wild_type), replace=True)
        values[index] = cliffs_delta(sampled_mutant, sampled_wild_type)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def bh_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(float))
    ranked = values.to_numpy(float)[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    return pd.Series(restored, index=values.index)


def plot_effects(results: pd.DataFrame, cell_type: str, output: Path) -> None:
    table = results[results["cell_type"].eq(cell_type)].copy()
    table = table.sort_values("cliffs_delta_mut_higher")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.errorbar(
        table["cliffs_delta_mut_higher"],
        y,
        xerr=[
            table["cliffs_delta_mut_higher"] - table["cliffs_delta_ci95_lower"],
            table["cliffs_delta_ci95_upper"] - table["cliffs_delta_mut_higher"],
        ],
        fmt="o",
        color="#4c78a8" if cell_type == "macrophage" else "#e45756",
        ecolor="#555555",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, table["display_label"], fontsize=8)
    ax.set_xlabel("Cliff's delta (positive = tert mutant higher)")
    ax.set_title(f"{cell_type}: untreated WT–tert mutant injury-response proxies")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    feature_dir = args.feature_extraction_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(feature_dir / "block_metadata.csv")
    root = feature_dir / "manual_injury_feature_outputs_time_corrected"
    rng = np.random.default_rng(args.seed)
    fish_tables = []
    results = []

    for cell_type in ("macrophage", "musc"):
        fish = aggregate_fish(root / cell_type / "per_track", metadata, cell_type)
        fish_tables.append(fish)
        for metric, (_, _, display_label) in METRICS.items():
            mutant = fish.loc[fish["genotype"].eq("MUT"), metric].dropna().to_numpy(float)
            wild_type = fish.loc[fish["genotype"].eq("WT"), metric].dropna().to_numpy(float)
            difference, p_value, greater_p, lower_p, assignments = exact_difference_test(
                mutant, wild_type
            )
            direction = DIRECTIONAL_HYPOTHESES.get(cell_type, {}).get(metric)
            directional_p = (
                greater_p
                if direction == "MUT_higher"
                else lower_p
                if direction == "MUT_lower"
                else np.nan
            )
            delta = cliffs_delta(mutant, wild_type)
            lower, upper = bootstrap_delta_interval(
                mutant, wild_type, args.bootstrap_iterations, rng
            )
            results.append(
                {
                    "cell_type": cell_type,
                    "metric": metric,
                    "display_label": display_label,
                    "n_WT": len(wild_type),
                    "n_MUT": len(mutant),
                    "mean_WT": wild_type.mean(),
                    "mean_MUT": mutant.mean(),
                    "median_WT": np.median(wild_type),
                    "median_MUT": np.median(mutant),
                    "mean_difference_MUT_minus_WT": difference,
                    "cliffs_delta_mut_higher": delta,
                    "cliffs_delta_ci95_lower": lower,
                    "cliffs_delta_ci95_upper": upper,
                    "exact_assignments": assignments,
                    "p_value_two_sided": p_value,
                    "prespecified_direction": direction,
                    "p_value_directional": directional_p,
                }
            )

    result_table = pd.DataFrame(results)
    for _, indexes in result_table.groupby("cell_type").groups.items():
        result_table.loc[indexes, "q_value_BH_within_cell_type"] = bh_adjust(
            result_table.loc[indexes, "p_value_two_sided"]
        )
    for cell_type, indexes in result_table.dropna(
        subset=["p_value_directional"]
    ).groupby("cell_type").groups.items():
        result_table.loc[indexes, "q_value_directional_BH"] = bh_adjust(
            result_table.loc[indexes, "p_value_directional"]
        )
    pd.concat(fish_tables, ignore_index=True).to_csv(
        output / "fish_level_injury_proxy_values.csv", index=False
    )
    result_table.to_csv(output / "injury_proxy_statistical_results.csv", index=False)
    for cell_type in ("macrophage", "musc"):
        plot_effects(
            result_table,
            cell_type,
            output / f"effect_sizes__{cell_type}.png",
        )


if __name__ == "__main__":
    main()
