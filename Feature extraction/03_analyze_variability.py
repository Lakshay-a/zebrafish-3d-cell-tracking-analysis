from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# Optional imports


try:
    from scipy.stats import mannwhitneyu, kruskal
except Exception:
    mannwhitneyu = None
    kruskal = None

try:
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False



# Default feature priority


DEFAULT_FEATURES = [
    # Movement
    "track_length",
    "duration_frames",
    "track_completeness",
    "total_path_length_3d_um",
    "total_path_length_xy_um",
    "net_displacement_3d_um",
    "net_displacement_xy_um",
    "mean_squared_displacement_3d_um2",
    "mean_step_distance_3d_um",
    "max_step_distance_3d_um",
    "mean_speed_um_per_frame",
    "median_speed_um_per_frame",
    "max_speed_um_per_frame",
    "speed_std_um_per_frame",
    "speed_cv_um_per_frame",
    "directionality_ratio",
    "tortuosity",
    "moving_step_fraction",
    "z_range_um",
    "z_path_length_um",
    "z_displacement_um",

    # Shape / morphology
    "mean_volume_um3",
    "volume_um3_cv",
    "volume_um3_change_start_to_end",
    "volume_um3_slope",
    "mean_surface_area_um2",
    "surface_area_um2_cv",
    "mean_surface_area_to_volume",
    "mean_projected_area_xy_um2",
    "mean_max_cross_section_area_um2",
    "mean_sphericity",
    "sphericity_cv",
    "sphericity_change_start_to_end",
    "sphericity_slope",
    "mean_elongation",
    "elongation_cv",
    "elongation_change_start_to_end",
    "elongation_slope",
    "mean_flatness",
    "mean_aspect_ratio_3d",
    "mean_prolate_ellipticity",
    "mean_oblate_ellipticity",
    "mean_compactness_3d",
    "mean_solidity_3d",
    "mean_extent_3d",

    # Intensity
    "mean_intensity",
    "intensity_cv",
    "intensity_change_start_to_end",
    "mean_max_intensity",
    "mean_sum_intensity",

    # Macrophage cluster / static flags if present
    "inside_cluster_fraction",
    "near_cluster_boundary_fraction",
    "mean_cluster_overlap_fraction",
    "mean_distance_to_cluster_boundary_um",
    "min_distance_to_cluster_boundary_um",
]

REQUIRED_FEATURE_BY_FISH_PLOTS = [
    # Core movement
    "mean_speed_um_per_frame",
    "median_speed_um_per_frame",
    "max_speed_um_per_frame",
    "mean_step_distance_3d_um",
    "median_step_distance_3d_um",
    "max_step_distance_3d_um",
    "total_path_length_3d_um",
    "net_displacement_3d_um",
    "directionality_ratio",
    "tortuosity",
    "moving_step_fraction",

    # Z movement
    "z_range_um",
    "z_path_length_um",
    "z_displacement_um",

    # Morphology
    "mean_sphericity",
    "median_sphericity",
    "sphericity_cv",
    "mean_elongation",
    "median_elongation",
    "elongation_cv",
    "mean_volume_um3",
    "median_volume_um3",
    "volume_um3_cv",
    "mean_surface_area_to_volume",
    "median_surface_area_to_volume",
    "surface_area_to_volume_cv",

    # Intensity
    "mean_mean_intensity",
    "median_mean_intensity",
    "mean_max_intensity",
    "median_max_intensity",

    # Macrophage / cluster features, generated only if columns exist
    "inside_cluster_fraction",
    "near_cluster_boundary_fraction",
    "overlap_cluster_mask_fraction",
    "mean_cluster_overlap_fraction",
    "max_cluster_overlap_fraction",
    "mean_distance_to_cluster_boundary_px",
    "min_distance_to_cluster_boundary_px",
    "mean_distance_to_cluster_boundary_um",
    "min_distance_to_cluster_boundary_um",
]


ID_COLUMNS = {
    "file",
    "source_file",
    "fish_id",
    "genotype",
    "cell_type",
    "track_id",
    "object_label",
    "time",
    "start_time",
    "end_time",
    "track_region_type",
    "macrophage_motility_class",
    "macrophage_static_reason",
}



# Utility functions


def safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def split_csv_paths(text: str) -> list[Path]:
    return [Path(p.strip()) for p in text.split(",") if p.strip()]


def normalise_genotype(value) -> str:
    s = str(value).strip()

    if s.lower() in {"wt", "wildtype", "wild_type", "control"}:
        return "WT"

    if s.lower() in {"mut", "mutant", "tert", "tert_mutant", "tert-/ -", "tert-/-"}:
        return "MUT"

    return s


def infer_genotype_from_text(text: str) -> str | None:
    s = str(text).lower()

    if re.search(r"\bwt\b|wildtype|wild_type", s):
        return "WT"

    if re.search(r"\bmut\b|mutant|tert", s):
        return "MUT"

    return None


def infer_fish_id_from_file(value: str) -> str:
    s = str(value)
    s = Path(s).stem if "/" in s else s
    s = re.sub(r"\.csv$", "", s)
    return safe_name(s)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        return numeric.fillna(0) > 0

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def choose_feature_columns(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            print("[WARN] Requested features missing and will be skipped:")
            for c in missing:
                print("   ", c)
        return [c for c in requested if c in df.columns]

    priority = [c for c in DEFAULT_FEATURES if c in df.columns]

    numeric_cols = []
    for c in df.columns:
        if c in ID_COLUMNS:
            continue
        if c.endswith("_exclude") or c.endswith("_flag"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    final = []
    for c in priority + numeric_cols:
        if c not in final:
            final.append(c)

    return final


def clean_numeric_feature(df: pd.DataFrame, feature: str) -> pd.Series:
    y = pd.to_numeric(df[feature], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    return y



# Loading / metadata


def load_feature_files(input_paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df["source_feature_csv"] = str(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    return out


def attach_metadata(
    df: pd.DataFrame,
    metadata_path: Path | None,
    fish_col: str,
    genotype_col: str,
    file_col: str,
) -> pd.DataFrame:
    df = df.copy()

    if metadata_path is not None:
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        meta = pd.read_csv(metadata_path)

        if file_col not in df.columns:
            raise ValueError(
                f"Feature table does not contain file column '{file_col}'. "
                "Either add a file column or use --file-col with the correct name."
            )

        if file_col not in meta.columns:
            raise ValueError(
                f"Metadata table must contain column '{file_col}' for merging."
            )

        df[file_col] = df[file_col].astype(str)
        meta[file_col] = meta[file_col].astype(str)

        df = df.merge(meta, on=file_col, how="left", suffixes=("", "_meta"))

        if fish_col not in df.columns and f"{fish_col}_meta" in df.columns:
            df[fish_col] = df[f"{fish_col}_meta"]

        if genotype_col not in df.columns and f"{genotype_col}_meta" in df.columns:
            df[genotype_col] = df[f"{genotype_col}_meta"]

    if fish_col not in df.columns:
        if file_col in df.columns:
            print("[WARN] fish_id column not found. Inferring fish_id from file column.")
            df[fish_col] = df[file_col].apply(infer_fish_id_from_file)
        else:
            raise ValueError(
                f"No '{fish_col}' column and no '{file_col}' column available. "
                "Add fish_id to the feature CSV or provide --metadata."
            )

    if genotype_col not in df.columns:
        if file_col in df.columns:
            print("[WARN] genotype column not found. Trying to infer genotype from file names.")
            df[genotype_col] = df[file_col].apply(infer_genotype_from_text)
        else:
            raise ValueError(
                f"No '{genotype_col}' column available. "
                "Add genotype to the feature CSV or provide --metadata."
            )

    df[fish_col] = df[fish_col].astype(str).map(safe_name)
    df[genotype_col] = df[genotype_col].map(normalise_genotype)

    missing_meta = df[fish_col].isna() | df[genotype_col].isna() | (df[genotype_col].astype(str) == "None")

    if missing_meta.any():
        bad = df.loc[missing_meta, [file_col, fish_col, genotype_col]].drop_duplicates()
        print("[WARN] Some rows are missing fish/genotype metadata:")
        print(bad.head(20).to_string(index=False))
        df = df.loc[~missing_meta].copy()

    return df



# Variance / ICC calculations


# ICC reference: https://pubmed.ncbi.nlm.nih.gov/18839484/
def one_way_icc(y: pd.Series, groups: pd.Series) -> float:
    temp = pd.DataFrame({"y": y, "group": groups}).dropna()

    if temp["group"].nunique() < 2:
        return np.nan

    group_stats = temp.groupby("group")["y"].agg(["count", "mean"])
    k = len(group_stats)
    n_total = len(temp)

    if n_total <= k:
        return np.nan

    grand_mean = temp["y"].mean()

    ss_between = float((group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum())

    temp = temp.join(group_stats["mean"], on="group", rsuffix="_group")
    ss_within = float(((temp["y"] - temp["mean"]) ** 2).sum())

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)

    n_i = group_stats["count"].to_numpy(dtype=float)

    # Unbalanced average group size.
    n_bar = (n_total - (np.sum(n_i ** 2) / n_total)) / (k - 1)

    denom = ms_between + (n_bar - 1) * ms_within

    if denom == 0:
        return np.nan

    return float((ms_between - ms_within) / denom)


def eta_squared(y: pd.Series, groups: pd.Series) -> float:
    temp = pd.DataFrame({"y": y, "group": groups}).dropna()

    if temp["group"].nunique() < 2:
        return np.nan

    grand = temp["y"].mean()
    ss_total = float(((temp["y"] - grand) ** 2).sum())

    if ss_total == 0:
        return np.nan

    group_means = temp.groupby("group")["y"].mean()
    group_counts = temp.groupby("group")["y"].count()

    ss_between = float((group_counts * (group_means - grand) ** 2).sum())

    return ss_between / ss_total


def genotype_p_value_on_fish_medians(
    per_fish: pd.DataFrame,
    feature: str,
    genotype_col: str,
) -> float:
    if mannwhitneyu is None:
        return np.nan

    temp = per_fish[[genotype_col, feature]].dropna()
    groups = list(temp[genotype_col].dropna().unique())

    if len(groups) == 2:
        a = temp.loc[temp[genotype_col] == groups[0], feature]
        b = temp.loc[temp[genotype_col] == groups[1], feature]

        if len(a) < 2 or len(b) < 2:
            return np.nan

        try:
            return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            return np.nan

    if len(groups) > 2 and kruskal is not None:
        vals = [temp.loc[temp[genotype_col] == g, feature] for g in groups]
        vals = [v for v in vals if len(v) >= 2]

        if len(vals) < 2:
            return np.nan

        try:
            return float(kruskal(*vals).pvalue)
        except Exception:
            return np.nan

    return np.nan


def build_fish_effect_summary(
    df: pd.DataFrame,
    per_fish_medians: pd.DataFrame,
    features: list[str],
    fish_col: str,
    genotype_col: str,
) -> pd.DataFrame:
    rows = []

    for feature in features:
        y = clean_numeric_feature(df, feature)

        usable = pd.DataFrame({
            "y": y,
            "fish_id": df[fish_col],
            "genotype": df[genotype_col],
        }).dropna()

        if usable.empty:
            continue

        fish_icc_all_raw = one_way_icc(usable["y"], usable["fish_id"])
        fish_icc_all = max(0.0, fish_icc_all_raw) if pd.notna(fish_icc_all_raw) else np.nan

        within_iccs = []
        for genotype, sub in usable.groupby("genotype"):
            if sub["fish_id"].nunique() >= 2:
                v = one_way_icc(sub["y"], sub["fish_id"])
                if pd.notna(v):
                    within_iccs.append(max(0.0, v))

        fish_icc_within_mean = float(np.mean(within_iccs)) if within_iccs else np.nan
        fish_icc_within_max = float(np.max(within_iccs)) if within_iccs else np.nan

        fish_eta = eta_squared(usable["y"], usable["fish_id"])
        genotype_eta = eta_squared(usable["y"], usable["genotype"])

        p_fish_median = genotype_p_value_on_fish_medians(
            per_fish=per_fish_medians,
            feature=feature,
            genotype_col=genotype_col,
        )

        n_cells = len(usable)
        n_fish = usable["fish_id"].nunique()
        n_genotypes = usable["genotype"].nunique()

        if n_fish < 4 or n_genotypes < 2:
            decision = "insufficient_fish_count"
        elif pd.notna(fish_icc_within_mean) and fish_icc_within_mean >= 0.30:
            decision = "fish_variable_high_use_fish_level_or_mixed"
        elif pd.notna(fish_icc_within_mean) and fish_icc_within_mean >= 0.10:
            decision = "fish_variable_moderate_use_cell_level_with_group_split"
        elif pd.notna(genotype_eta) and pd.notna(fish_eta) and genotype_eta > fish_eta:
            decision = "genotype_signal_stronger_cell_level_possible"
        else:
            decision = "fish_effect_low_or_unclear_use_group_split"

        rows.append({
            "feature": feature,
            "n_cells": n_cells,
            "n_fish": n_fish,
            "n_genotypes": n_genotypes,
            "fish_icc_all": fish_icc_all,
            "fish_icc_within_genotype_mean": fish_icc_within_mean,
            "fish_icc_within_genotype_max": fish_icc_within_max,
            "fish_eta_squared_cell_level": fish_eta,
            "genotype_eta_squared_cell_level": genotype_eta,
            "genotype_p_value_on_fish_medians": p_fish_median,
            "decision": decision,
        })

    return pd.DataFrame(rows)



# Plotting

def plot_required_feature_by_fish(df, output_dir):
    """Always generate key biological feature-by-fish plots,."""

    output_dir = Path(output_dir)
    out_dir = output_dir / "feature_by_fish"
    out_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"fish_id", "genotype"}
    missing = required_cols - set(df.columns)

    if missing:
        print(f"[WARN] Cannot generate required feature-by-fish plots. Missing columns: {missing}")
        return

    available_features = [
        feature for feature in REQUIRED_FEATURE_BY_FISH_PLOTS
        if feature in df.columns
    ]

    if not available_features:
        print("[WARN] None of the required biological plot features were found in the dataframe.")
        return

    print()
    print("[INFO] Generating required biological feature-by-fish plots:")
    for feature in available_features:
        print(f"       - {feature}")

    fish_order = (
        df[["fish_id", "genotype"]]
        .drop_duplicates()
        .sort_values(["genotype", "fish_id"])
    )

    fish_ids = fish_order["fish_id"].tolist()
    x_map = {fish: i for i, fish in enumerate(fish_ids)}

    rng = np.random.default_rng(42)

    for feature in available_features:
        plot_df = df[["fish_id", "genotype", feature]].dropna().copy()

        if plot_df.empty:
            continue

        plt.figure(figsize=(max(12, len(fish_ids) * 0.75), 5.5))

        for genotype, g in plot_df.groupby("genotype"):
            x = g["fish_id"].map(x_map).astype(float).to_numpy()
            jitter = rng.normal(0, 0.08, size=len(g))

            plt.scatter(
                x + jitter,
                g[feature],
                alpha=0.28,
                s=9,
                label=str(genotype),
            )

        fish_medians = plot_df.groupby("fish_id")[feature].median()

        plt.scatter(
            [x_map[f] for f in fish_medians.index],
            fish_medians.values,
            s=70,
            marker="D",
            label="Fish median",
        )

        plt.xticks(range(len(fish_ids)), fish_ids, rotation=90)
        plt.xlabel("Fish ID")
        plt.ylabel(feature)
        plt.title(f"{feature} cell values by fish")
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"{feature}_cell_values_by_fish.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"[SAVED] Required biological feature-by-fish plots: {out_dir}")

def sorted_fish_order(df: pd.DataFrame, fish_col: str, genotype_col: str) -> list[str]:
    temp = (
        df[[fish_col, genotype_col]]
        .drop_duplicates()
        .sort_values([genotype_col, fish_col])
    )
    return temp[fish_col].tolist()


def plot_feature_by_fish(
    df: pd.DataFrame,
    feature: str,
    fish_col: str,
    genotype_col: str,
    output_dir: Path,
    max_points: int = 4000,
):
    temp = df[[fish_col, genotype_col, feature]].copy()
    temp[feature] = clean_numeric_feature(temp, feature)
    temp = temp.dropna()

    if temp.empty or temp[fish_col].nunique() < 2:
        return

    if len(temp) > max_points:
        temp = temp.sample(max_points, random_state=42)

    fish_order = sorted_fish_order(temp, fish_col, genotype_col)
    fish_to_x = {fish: i for i, fish in enumerate(fish_order)}

    fig_w = max(10, min(24, len(fish_order) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    genotypes = list(temp[genotype_col].dropna().unique())

    for genotype in genotypes:
        sub = temp[temp[genotype_col] == genotype]
        x = sub[fish_col].map(fish_to_x).astype(float).to_numpy()
        jitter = np.random.default_rng(42).normal(0, 0.08, size=len(sub))
        ax.scatter(x + jitter, sub[feature], alpha=0.25, s=8, label=str(genotype))

    med = temp.groupby(fish_col)[feature].median()
    ax.plot(
        [fish_to_x[f] for f in fish_order],
        [med.loc[f] for f in fish_order],
        marker="o",
        linewidth=1,
        label="fish median",
    )

    ax.set_title(f"{feature} by fish")
    ax.set_xlabel("Fish ID")
    ax.set_ylabel(feature)
    ax.set_xticks(range(len(fish_order)))
    ax.set_xticklabels(fish_order, rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = output_dir / f"{safe_name(feature)}_cell_values_by_fish.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)


def plot_fish_medians_by_genotype(
    per_fish: pd.DataFrame,
    feature: str,
    fish_col: str,
    genotype_col: str,
    output_dir: Path,
):
    temp = per_fish[[fish_col, genotype_col, feature]].copy()
    temp[feature] = clean_numeric_feature(temp, feature)
    temp = temp.dropna()

    if temp.empty or temp[genotype_col].nunique() < 2:
        return

    genotypes = sorted(temp[genotype_col].unique().tolist())
    x_map = {g: i for i, g in enumerate(genotypes)}

    fig, ax = plt.subplots(figsize=(7, 5))

    values_for_box = [temp.loc[temp[genotype_col] == g, feature].values for g in genotypes]
    ax.boxplot(values_for_box, labels=genotypes, showfliers=False)

    rng = np.random.default_rng(42)
    for g in genotypes:
        sub = temp[temp[genotype_col] == g]
        x = np.full(len(sub), x_map[g] + 1, dtype=float)
        jitter = rng.normal(0, 0.04, size=len(sub))
        ax.scatter(x + jitter, sub[feature], s=35, alpha=0.85)

        for _, row in sub.iterrows():
            ax.annotate(
                str(row[fish_col]),
                (x_map[g] + 1 + 0.06, row[feature]),
                fontsize=7,
                alpha=0.75,
            )

    ax.set_title(f"Per-fish median {feature} by genotype")
    ax.set_xlabel("Genotype")
    ax.set_ylabel(f"Median {feature}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = output_dir / f"{safe_name(feature)}_fish_medians_by_genotype.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)


def plot_icc_summary(summary: pd.DataFrame, output_dir: Path, top_n: int = 25):
    if summary.empty:
        return

    temp = summary.copy()
    temp["icc"] = pd.to_numeric(temp["fish_icc_within_genotype_mean"], errors="coerce")
    temp = temp.dropna(subset=["icc"])
    temp = temp.sort_values("icc", ascending=False).head(top_n)

    if temp.empty:
        return

    fig_h = max(6, len(temp) * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_pos = np.arange(len(temp))
    ax.barh(y_pos, temp["icc"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(temp["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Fish ICC within genotype")
    ax.set_title("Features most affected by fish-to-fish variation")
    ax.axvline(0.10, linestyle="--", linewidth=1)
    ax.axvline(0.30, linestyle="--", linewidth=1)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    fig.savefig(output_dir / "icc_fish_effect_summary.png", dpi=250)
    plt.close(fig)


def plot_decision_counts(summary: pd.DataFrame, output_dir: Path):
    if summary.empty or "decision" not in summary.columns:
        return

    counts = summary["decision"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Feature-level decision summary")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Number of features")
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(output_dir / "feature_decision_counts.png", dpi=250)
    plt.close(fig)


def plot_ml_summary(ml_summary: pd.DataFrame, output_dir: Path):
    if ml_summary.empty:
        return

    temp = ml_summary.copy()

    labels = temp["model"].astype(str).tolist()
    x = np.arange(len(temp))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, temp["random_cell_split_balanced_accuracy"], width, label="random cell split")
    ax.bar(x + width / 2, temp["groupkfold_by_fish_balanced_accuracy"], width, label="GroupKFold by fish")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Random cell split vs fish-aware split")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(output_dir / "ml_random_vs_groupkfold.png", dpi=250)
    plt.close(fig)


# PCA API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def plot_pca(
    df: pd.DataFrame,
    features: list[str],
    fish_col: str,
    genotype_col: str,
    output_path: Path,
    title: str,
    max_points: int = 3000,
):
    if not SKLEARN_AVAILABLE:
        return

    temp = df[[fish_col, genotype_col] + features].copy()

    for f in features:
        temp[f] = clean_numeric_feature(temp, f)

    temp = temp.dropna(subset=[fish_col, genotype_col])

    if len(temp) < 5 or temp[genotype_col].nunique() < 2:
        return

    if len(temp) > max_points:
        temp = temp.sample(max_points, random_state=42)

    X = temp[features].copy()

    # Keep only usable features.
    usable = []
    for f in features:
        vals = pd.to_numeric(X[f], errors="coerce")
        if vals.notna().mean() >= 0.70 and vals.nunique(dropna=True) > 1:
            usable.append(f)

    if len(usable) < 2:
        return

    X = X[usable]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
    ])

    coords = pipe.fit_transform(X)
    pca = pipe.named_steps["pca"]

    fig, ax = plt.subplots(figsize=(7, 6))

    for genotype in sorted(temp[genotype_col].unique()):
        mask = temp[genotype_col] == genotype
        ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.55, label=str(genotype))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(output_path, dpi=250)
    plt.close(fig)



# ML fish-aware validation


# Random forest API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
def evaluate_ml(
    df: pd.DataFrame,
    features: list[str],
    fish_col: str,
    genotype_col: str,
) -> pd.DataFrame:
    if not SKLEARN_AVAILABLE:
        print("[WARN] scikit-learn not available. Skipping ML split comparison.")
        return pd.DataFrame()

    temp = df[[fish_col, genotype_col] + features].copy()
    temp = temp.dropna(subset=[fish_col, genotype_col])

    if temp[genotype_col].nunique() < 2:
        print("[WARN] Less than two genotypes. Skipping ML.")
        return pd.DataFrame()

    if temp[fish_col].nunique() < 4:
        print("[WARN] Fewer than 4 fish. GroupKFold not reliable. Skipping ML.")
        return pd.DataFrame()

    usable_features = []
    for f in features:
        vals = clean_numeric_feature(temp, f)
        if vals.notna().mean() >= 0.70 and vals.nunique(dropna=True) > 1:
            temp[f] = vals
            usable_features.append(f)

    if len(usable_features) < 2:
        print("[WARN] Not enough usable numeric features for ML.")
        return pd.DataFrame()

    X = temp[usable_features]
    y_raw = temp[genotype_col].astype(str)
    groups = temp[fish_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    min_class_count = pd.Series(y).value_counts().min()
    n_random_splits = min(5, int(min_class_count))

    if n_random_splits < 2:
        print("[WARN] Not enough samples per genotype for stratified CV.")
        return pd.DataFrame()

    n_group_splits = min(5, groups.nunique())

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
    }

    rows = []

    for name, model in models.items():
        random_scores = []
        group_scores = []

        skf = StratifiedKFold(n_splits=n_random_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in skf.split(X, y):
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            random_scores.append(balanced_accuracy_score(y[test_idx], pred))

        gkf = GroupKFold(n_splits=n_group_splits)

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            # Skip impossible folds where training has only one genotype.
            if len(np.unique(y[train_idx])) < 2:
                continue

            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            group_scores.append(balanced_accuracy_score(y[test_idx], pred))

        random_mean = float(np.mean(random_scores)) if random_scores else np.nan
        group_mean = float(np.mean(group_scores)) if group_scores else np.nan

        rows.append({
            "model": name,
            "n_features": len(usable_features),
            "n_cells": len(temp),
            "n_fish": groups.nunique(),
            "random_cell_split_balanced_accuracy": random_mean,
            "groupkfold_by_fish_balanced_accuracy": group_mean,
            "accuracy_drop_random_minus_group": random_mean - group_mean,
            "interpretation": (
                "possible_fish_leakage" if pd.notna(group_mean) and random_mean - group_mean > 0.15
                else "genotype_signal_may_generalise"
            ),
        })

    return pd.DataFrame(rows)



# Main


def main():
    parser = argparse.ArgumentParser(
        description="Analyse whether fish_id is a major variable in cell-level feature tables."
    )

    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated cell-level feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for summaries and plots.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata CSV containing file, fish_id and genotype.",
    )
    parser.add_argument("--fish-col", default="fish_id")
    parser.add_argument("--genotype-col", default="genotype")
    parser.add_argument("--file-col", default="file")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated feature list. If omitted, features are auto-selected.",
    )
    parser.add_argument(
        "--max-plot-features",
        type=int,
        default=20,
        help="Maximum number of individual feature plots to generate.",
    )
    parser.add_argument(
        "--exclude-static-macrophages",
        action="store_true",
        help="Remove rows where macrophage_static_exclude is True, if present.",
    )

    args = parser.parse_args()

    input_paths = split_csv_paths(args.inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_by_fish_dir = output_dir / "feature_by_fish"
    feature_by_fish_dir.mkdir(exist_ok=True)

    fish_median_plot_dir = output_dir / "feature_fish_medians"
    fish_median_plot_dir.mkdir(exist_ok=True)

    metadata_path = Path(args.metadata) if args.metadata else None

    requested_features = (
        [x.strip() for x in args.features.split(",") if x.strip()]
        if args.features
        else None
    )

    print("============================================================")
    print("[INFO] Fish variability analysis")
    print("============================================================")
    print("[INFO] Inputs:")
    for p in input_paths:
        print("  ", p)
    print("[INFO] Output:", output_dir)

    df = load_feature_files(input_paths)

    df = attach_metadata(
        df=df,
        metadata_path=metadata_path,
        fish_col=args.fish_col,
        genotype_col=args.genotype_col,
        file_col=args.file_col,
    )

    if args.exclude_static_macrophages and "macrophage_static_exclude" in df.columns:
        before = len(df)
        df = df.loc[~bool_series(df["macrophage_static_exclude"])].copy()
        print(f"[INFO] Removed static macrophage rows: {before - len(df)}")

    if args.cell_type_col not in df.columns:
        df[args.cell_type_col] = "unknown"

    features = choose_feature_columns(df, requested_features)

    # Remove features that are mostly missing or constant.
    usable_features = []
    for f in features:
        vals = clean_numeric_feature(df, f)
        non_missing = vals.notna().mean()
        nunique = vals.nunique(dropna=True)

        if non_missing >= 0.50 and nunique > 1:
            df[f] = vals
            usable_features.append(f)

    features = usable_features

    if not features:
        raise ValueError("No usable numeric features found.")

    print(f"[INFO] Usable feature count: {len(features)}")

    # Save cleaned cell table used for analysis.
    df.to_csv(output_dir / "analysed_cell_features_used.csv", index=False)

    # Basic counts.
    counts = (
        df.groupby([args.genotype_col, args.fish_col])
        .size()
        .reset_index(name="n_cell_tracks")
        .sort_values([args.genotype_col, args.fish_col])
    )
    counts.to_csv(output_dir / "cell_track_counts_by_fish.csv", index=False)

    # Per-fish medians.
    per_fish_medians = (
        df.groupby([args.genotype_col, args.fish_col], as_index=False)[features]
        .median(numeric_only=True)
    )
    per_fish_medians.to_csv(output_dir / "per_fish_feature_medians.csv", index=False)

    # Fish effect summary.
    fish_summary = build_fish_effect_summary(
        df=df,
        per_fish_medians=per_fish_medians,
        features=features,
        fish_col=args.fish_col,
        genotype_col=args.genotype_col,
    )

    fish_summary = fish_summary.sort_values(
        ["fish_icc_within_genotype_mean", "fish_eta_squared_cell_level"],
        ascending=False,
        na_position="last",
    )

    fish_summary.to_csv(output_dir / "fish_effect_summary.csv", index=False)

    # ML split comparison.
    ml_summary = evaluate_ml(
        df=df,
        features=features,
        fish_col=args.fish_col,
        genotype_col=args.genotype_col,
    )
    ml_summary.to_csv(output_dir / "ml_group_split_summary.csv", index=False)

    # Feature plots.
    features_to_plot = fish_summary["feature"].head(args.max_plot_features).tolist()

    # Also force key biological features if present.
    important = [
        "mean_speed_um_per_frame",
        "total_path_length_3d_um",
        "net_displacement_3d_um",
        "directionality_ratio",
        "mean_sphericity",
        "mean_elongation",
        "mean_volume_um3",
        "z_range_um",
    ]

    for f in important:
        if f in features and f not in features_to_plot:
            features_to_plot.append(f)

    features_to_plot = features_to_plot[: args.max_plot_features]

    print("[INFO] Generating feature plots:")
    for f in features_to_plot:
        print("   ", f)
        plot_feature_by_fish(
            df=df,
            feature=f,
            fish_col=args.fish_col,
            genotype_col=args.genotype_col,
            output_dir=feature_by_fish_dir,
        )
        plot_fish_medians_by_genotype(
            per_fish=per_fish_medians,
            feature=f,
            fish_col=args.fish_col,
            genotype_col=args.genotype_col,
            output_dir=fish_median_plot_dir,
        )

    plot_icc_summary(fish_summary, output_dir=output_dir)
    plot_decision_counts(fish_summary, output_dir=output_dir)
    plot_ml_summary(ml_summary, output_dir=output_dir)

    # PCA plots.
    plot_pca(
        df=df,
        features=features,
        fish_col=args.fish_col,
        genotype_col=args.genotype_col,
        output_path=output_dir / "pca_cell_level.png",
        title="Cell-level PCA of tracked-cell features",
    )

    plot_pca(
        df=per_fish_medians,
        features=features,
        fish_col=args.fish_col,
        genotype_col=args.genotype_col,
        output_path=output_dir / "pca_fish_median_level.png",
        title="Fish-level PCA using per-fish median features",
        max_points=10000,
    )
    plot_required_feature_by_fish(df, output_dir)
    print("\n============================================================")
    print("[DONE] Fish variability analysis complete")
    print("============================================================")
    print(f"[SAVED] {output_dir / 'fish_effect_summary.csv'}")
    print(f"[SAVED] {output_dir / 'per_fish_feature_medians.csv'}")
    print(f"[SAVED] {output_dir / 'ml_group_split_summary.csv'}")
    print(f"[PLOTS] {feature_by_fish_dir}")
    print(f"[PLOTS] {fish_median_plot_dir}")
    print("============================================================")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
