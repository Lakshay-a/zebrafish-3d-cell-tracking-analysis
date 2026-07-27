from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FISH_COLUMNS = ["fish_id", "block_name", "block", "sample_id"]
GENOTYPE_COLUMNS = ["genotype", "group", "condition", "class"]
COMPACT_INJURY_FEATURES = [
    "median_net_approach_um",
    "median_min_abs_distance_to_injury_um",
    "mean_fraction_detections_inside",
    "mean_fraction_steps_toward",
    "mean_toward_velocity_um_per_min",
    "mean_away_velocity_um_per_min",
    "entries_per_track",
    "exits_per_track",
]


def detect_column(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    lookup = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Could not detect {role} column in {list(df.columns)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge compact manual injury predictors into a fish-level model table."
    )
    parser.add_argument("--base-table", required=True)
    parser.add_argument("--injury-table", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_table, low_memory=False)
    injury = pd.read_csv(args.injury_table, low_memory=False)

    base_fish_col = detect_column(base, FISH_COLUMNS, "base fish")
    injury_fish_col = detect_column(injury, FISH_COLUMNS, "injury fish")
    base_genotype_col = detect_column(base, GENOTYPE_COLUMNS, "base genotype")

    prefixed_injury_features = [
        column
        for column in injury.columns
        if str(column).startswith("fish_mean__injury_")
    ]
    if prefixed_injury_features:
        injury_features = prefixed_injury_features
    else:
        raw_injury_features = [
            column
            for column in COMPACT_INJURY_FEATURES
            if column in injury.columns
        ]
        if not raw_injury_features:
            raise ValueError(
                "No compact injury features or fish_mean__injury_ columns "
                "found in injury table."
            )
        rename_map = {
            column: f"fish_mean__injury_{column}"
            for column in raw_injury_features
        }
        injury = injury.rename(columns=rename_map)
        injury_features = list(rename_map.values())

    base = base.copy()
    injury = injury.copy()
    base[base_fish_col] = base[base_fish_col].astype(str).str.strip()
    injury[injury_fish_col] = injury[injury_fish_col].astype(str).str.strip()

    injury_subset = injury[[injury_fish_col] + injury_features].drop_duplicates(
        subset=[injury_fish_col]
    )
    merged = base.merge(
        injury_subset,
        left_on=base_fish_col,
        right_on=injury_fish_col,
        how="inner",
    )
    if injury_fish_col != base_fish_col and injury_fish_col in merged.columns:
        merged = merged.drop(columns=[injury_fish_col])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Fish count: {len(merged)}")
    print(f"Genotypes: {merged[base_genotype_col].value_counts().to_dict()}")
    print(f"Injury features added: {len(injury_features)}")


if __name__ == "__main__":
    main()
