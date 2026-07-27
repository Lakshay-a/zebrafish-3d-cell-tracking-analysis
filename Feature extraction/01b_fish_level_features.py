import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("fish_level_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "musc": {
        "input": "qc_outlier_outputs/musc/cell_track_features_qc_filtered.csv",
        "prefix": "musc",
        "output": "musc_fish_level_features.csv",
    },
    "macrophage_all": {
        "input": "qc_outlier_outputs/macrophage_all/cell_track_features_qc_filtered.csv",
        "prefix": "mac_all",
        "output": "macrophage_all_fish_level_features.csv",
    },
    "macrophage_outside_boundary": {
        "input": "qc_outlier_outputs/macrophage_outside_boundary/cell_track_features_qc_filtered.csv",
        "prefix": "mac_outside",
        "output": "macrophage_outside_boundary_fish_level_features.csv",
    },
}

CORE_FEATURES = [
    # movement
    "track_length",
    "duration_frames",
    "track_completeness",
    "mean_speed_um_per_frame",
    "median_speed_um_per_frame",
    "max_speed_um_per_frame",
    "speed_std_um_per_frame",
    "speed_cv_um_per_frame",
    "total_path_length_3d_um",
    "total_path_length_xy_um",
    "net_displacement_3d_um",
    "net_displacement_xy_um",
    "mean_squared_displacement_3d_um2",
    "mean_step_distance_3d_um",
    "median_step_distance_3d_um",
    "max_step_distance_3d_um",
    "directionality_ratio",
    "tortuosity",
    "moving_step_fraction",

    # z movement
    "z_range_um",
    "z_path_length_um",
    "z_displacement_um",
    "fraction_steps_positive_z",

    # morphology
    "mean_volume_um3",
    "median_volume_um3",
    "volume_um3_cv",
    "mean_sphericity",
    "median_sphericity",
    "sphericity_cv",
    "mean_elongation",
    "median_elongation",
    "elongation_cv",
    "mean_surface_area_to_volume",
    "median_surface_area_to_volume",
    "surface_area_to_volume_cv",

    # intensity
    "mean_mean_intensity",
    "median_mean_intensity",
    "mean_max_intensity",
    "median_max_intensity",

    # cluster features, if present
    "inside_cluster_fraction",
    "near_cluster_boundary_fraction",
    "overlap_cluster_mask_fraction",
    "mean_cluster_overlap_fraction",
    "max_cluster_overlap_fraction",
    "mean_distance_to_cluster_boundary_um",
    "min_distance_to_cluster_boundary_um",
]

def q25(x):
    return np.nanpercentile(x, 25)

def q75(x):
    return np.nanpercentile(x, 75)

def iqr(x):
    return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

def cv(x):
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if mean == 0 or np.isnan(mean):
        return np.nan
    return std / mean

def make_fish_level(input_path, prefix, output_name):
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"[WARN] Missing input: {input_path}")
        return None

    df = pd.read_csv(input_path)

    required = {"fish_id", "genotype"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    available = [f for f in CORE_FEATURES if f in df.columns]

    print("\n" + "=" * 70)
    print(prefix)
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Rows/cells: {len(df)}")
    print(f"Fish: {df['fish_id'].nunique()}")
    print(f"Available core features: {len(available)}")

    grouped = df.groupby(["genotype", "fish_id"], dropna=False)

    out = grouped.size().reset_index(name=f"{prefix}_n_cells")

    # useful track-count summaries
    if "track_length" in df.columns:
        temp = grouped["track_length"].agg(
            **{
                f"{prefix}_median_track_length": "median",
                f"{prefix}_mean_track_length": "mean",
                f"{prefix}_long_tracks_ge20_fraction": lambda x: np.mean(x >= 20),
                f"{prefix}_long_tracks_ge50_fraction": lambda x: np.mean(x >= 50),
            }
        ).reset_index()
        out = out.merge(temp, on=["genotype", "fish_id"], how="left")

    # aggregate selected biological features
    for feature in available:
        temp = grouped[feature].agg(
            median="median",
            mean="mean",
            std="std",
            q25=q25,
            q75=q75,
            iqr=iqr,
            cv=cv,
        ).reset_index()

        rename = {
            "median": f"{prefix}_{feature}_median",
            "mean": f"{prefix}_{feature}_mean",
            "std": f"{prefix}_{feature}_std",
            "q25": f"{prefix}_{feature}_q25",
            "q75": f"{prefix}_{feature}_q75",
            "iqr": f"{prefix}_{feature}_iqr",
            "cv": f"{prefix}_{feature}_cv",
        }

        temp = temp.rename(columns=rename)
        out = out.merge(temp, on=["genotype", "fish_id"], how="left")

    output_path = OUT_DIR / output_name
    out.to_csv(output_path, index=False)

    print(f"[SAVED] {output_path}")
    print(f"Fish-level rows: {len(out)}")
    print(f"Fish-level columns: {len(out.columns)}")

    return out

fish_tables = {}

for name, info in DATASETS.items():
    table = make_fish_level(
        input_path=info["input"],
        prefix=info["prefix"],
        output_name=info["output"],
    )
    if table is not None:
        fish_tables[name] = table

# Combined fish-level table across MUSC + macrophage modes
combined = None

for name, table in fish_tables.items():
    if combined is None:
        combined = table.copy()
    else:
        table_no_genotype = table.drop(columns=["genotype"], errors="ignore")
        combined = combined.merge(
            table_no_genotype,
            on="fish_id",
            how="outer"
        )

if combined is not None:
    # keep genotype as first column if present
    cols = list(combined.columns)
    first_cols = [c for c in ["genotype", "fish_id"] if c in cols]
    other_cols = [c for c in cols if c not in first_cols]
    combined = combined[first_cols + other_cols]

    combined_out = OUT_DIR / "combined_fish_level_features.csv"
    combined.to_csv(combined_out, index=False)

    print("\n" + "=" * 70)
    print("[SAVED] Combined fish-level table")
    print("=" * 70)
    print(combined_out)
    print(f"Rows/fish: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")

print("\n[DONE]")