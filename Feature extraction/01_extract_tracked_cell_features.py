from __future__ import annotations

import os
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops, marching_cubes, mesh_surface_area



# Load project config


try:
    import config as cfg
except Exception:
    cfg = None


def get_setting(env_name: str, config_name: str | None = None, default=None):
    value = os.getenv(env_name)
    if value not in (None, ""):
        return value

    if cfg is not None and config_name is not None and hasattr(cfg, config_name):
        return getattr(cfg, config_name)

    return default


def get_path(env_name: str, config_name: str | None = None, default=None) -> Path | None:
    value = get_setting(env_name, config_name, default)
    if value in (None, ""):
        return None
    return Path(value)



# User / env settings


CELL_TYPE = str(get_setting("BATCH_CELL_TYPE", "CELL_TYPE", "musc")).lower()

BLOCK_NAME = str(get_setting("FEATURE_BLOCK_NAME", None, ""))
FISH_ID = str(get_setting("FEATURE_FISH_ID", None, ""))
GENOTYPE = str(get_setting("FEATURE_GENOTYPE", None, ""))

RAW_TCZYX_TIF = get_path("FEATURE_RAW_TIF", "RAW_TCZYX_TIF")
LABELS_3D_TZYX_TIF = get_path("FEATURE_LABELS_TIF", "LABELS_3D_TZYX_TIF")
TRACKING_OUTPUT_DIR = get_path("FEATURE_OUTPUT_BASE_DIR", "TRACKING_OUTPUT_DIR", Path.cwd())

FEATURE_TRACKS_CSV = get_path("FEATURE_TRACKS_CSV", None)

CHANNEL_INDEX = int(get_setting("FEATURE_CHANNEL_INDEX", "CHANNEL_INDEX", 0))

XY_PIXEL_SIZE_UM = float(
    get_setting("FEATURE_XY_PIXEL_SIZE_UM", "XY_PIXEL_SIZE_UM", 0.7533114346590908)
)

Z_STEP_SIZE_UM = float(
    get_setting("FEATURE_Z_STEP_SIZE_UM", "Z_STEP_SIZE_UM", 1.0)
)

# Set this in config.py if you know it.
# If unknown, speed_um_per_frame is still valid.
FRAME_INTERVAL_MIN = get_setting("FEATURE_FRAME_INTERVAL_MIN", "FRAME_INTERVAL_MIN", None)
FRAME_INTERVAL_MIN = None if FRAME_INTERVAL_MIN in (None, "") else float(FRAME_INTERVAL_MIN)

REGION_MODE = str(get_setting("MACROPHAGE_REGION_MODE", "MACROPHAGE_REGION_MODE", "all")).lower()

FEATURE_METHOD = str(get_setting("FEATURE_METHOD", None, "")).lower()

# For macrophage:
# outside_boundary = include cluster-region features
# all = do not include cluster-region features
INCLUDE_CLUSTER_FEATURES = get_setting("FEATURE_INCLUDE_CLUSTER", None, None)
if INCLUDE_CLUSTER_FEATURES is None:
    INCLUDE_CLUSTER_FEATURES = CELL_TYPE == "macrophage" and REGION_MODE == "outside_boundary"
else:
    INCLUDE_CLUSTER_FEATURES = str(INCLUDE_CLUSTER_FEATURES).lower() in {"1", "true", "yes", "y"}

COMPUTE_SURFACE_AREA = str(get_setting("FEATURE_COMPUTE_SURFACE_AREA", None, "1")).lower() in {
    "1",
    "true",
    "yes",
    "y",
}

# Static macrophage filtering happens after feature extraction.
STATIC_MIN_TRACK_LENGTH = int(get_setting("MAC_STATIC_MIN_TRACK_LENGTH", None, 5))
STATIC_TOTAL_PATH_UM = float(get_setting("MAC_STATIC_TOTAL_PATH_UM", None, 6.0))
STATIC_NET_DISPLACEMENT_UM = float(get_setting("MAC_STATIC_NET_DISPLACEMENT_UM", None, 4.0))
STATIC_MOVING_STEP_FRACTION = float(get_setting("MAC_STATIC_MOVING_STEP_FRACTION", None, 0.20))
MOVING_STEP_THRESHOLD_UM = float(get_setting("MAC_MOVING_STEP_THRESHOLD_UM", None, 1.0))


CLUSTER_COLS = [
    "cluster_id",
    "inside_cluster",
    "near_cluster_boundary",
    "overlap_cluster_mask",
    "cluster_overlap_pixels",
    "cluster_overlap_fraction",
    "distance_to_cluster_boundary_px",
    "distance_to_cluster_boundary_um",
    "cluster_region_class",
]



# Helpers


def load_raw_as_tzyx(raw_path: Path | None, channel_index: int) -> np.ndarray | None:
    if raw_path is None or not raw_path.exists():
        print("[WARN] Raw image not found. Intensity features will be NaN.")
        return None

    raw = tifffile.imread(raw_path)
    print(f"[INFO] Loaded raw shape: {raw.shape}")

    if raw.ndim == 5:
        # Tczyx -> tzyx
        return raw[:, channel_index]

    if raw.ndim == 4:
        # Already TZYX
        return raw

    raise ValueError(f"Unsupported raw shape: {raw.shape}")


def load_labels_tzyx(labels_path: Path | None) -> np.ndarray:
    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(
            "3D label image not found. Set FEATURE_LABELS_TIF or LABELS_3D_TZYX_TIF."
        )

    labels = tifffile.imread(labels_path)
    print(f"[INFO] Loaded labels shape: {labels.shape}")

    if labels.ndim != 4:
        raise ValueError(f"Expected labels as TZYX, got shape: {labels.shape}")

    return labels


def normalise_track_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}

    if "z" in df.columns and "centroid_z" not in df.columns:
        rename_map["z"] = "centroid_z"
    if "y" in df.columns and "centroid_y" not in df.columns:
        rename_map["y"] = "centroid_y"
    if "x" in df.columns and "centroid_x" not in df.columns:
        rename_map["x"] = "centroid_x"

    df = df.rename(columns=rename_map)

    required = ["time", "object_label", "track_id", "centroid_z", "centroid_y", "centroid_x"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Track CSV is missing required columns: {missing}")

    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
    df["object_label"] = pd.to_numeric(df["object_label"], errors="coerce").astype("Int64")

    for c in ["centroid_z", "centroid_y", "centroid_x"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time", "object_label", "track_id", "centroid_z", "centroid_y", "centroid_x"])
    df["time"] = df["time"].astype(int)
    df["object_label"] = df["object_label"].astype(int)

    # Keep tracked objects only.
    df = df[df["track_id"].notna()].copy()
    df = df[~df["track_id"].astype(str).str.lower().isin(["nan", "none", "-1"])].copy()

    if "cell_type" not in df.columns:
        df["cell_type"] = CELL_TYPE

    if "file" not in df.columns:
        df["file"] = ""

    return df


def find_tracks_csv() -> Path:
    if FEATURE_TRACKS_CSV is not None:
        if not FEATURE_TRACKS_CSV.exists():
            raise FileNotFoundError(f"FEATURE_TRACKS_CSV does not exist: {FEATURE_TRACKS_CSV}")
        return FEATURE_TRACKS_CSV

    search_dir = TRACKING_OUTPUT_DIR
    patterns = [
        f"{CELL_TYPE}_tracks_*_good_filtered.csv",
        f"{CELL_TYPE}_tracks_*.csv",
        f"{CELL_TYPE}_clean_track_ids_*.csv",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(search_dir.glob(pattern))

    valid = []
    required = {"time", "object_label", "track_id"}

    for path in candidates:
        try:
            cols = set(pd.read_csv(path, nrows=5).columns)
            if required.issubset(cols):
                valid.append(path)
        except Exception:
            pass

    if not valid:
        raise FileNotFoundError(
            f"No valid tracked-cell CSV found in {search_dir}. "
            f"Set FEATURE_TRACKS_CSV explicitly."
        )

    valid = sorted(valid, key=lambda p: p.stat().st_mtime, reverse=True)
    return valid[0]


def safe_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    mean = np.nanmean(values)
    if mean == 0 or not np.isfinite(mean):
        return np.nan

    return float(np.nanstd(values) / mean)


def safe_slope(times: np.ndarray, values: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]

    if len(values) < 2:
        return np.nan

    if np.ptp(times) == 0:
        return np.nan

    return float(np.polyfit(times, values, 1)[0])


def safe_fold_change(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    min_v = np.nanmin(values)
    max_v = np.nanmax(values)

    if min_v <= 0:
        return np.nan

    return float(max_v / min_v)


# Circular statistics: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html
def circular_mean_deg(angles_rad: np.ndarray) -> float:
    angles_rad = np.asarray(angles_rad, dtype=float)
    angles_rad = angles_rad[np.isfinite(angles_rad)]

    if len(angles_rad) == 0:
        return np.nan

    return float(np.degrees(np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))))


# Circular statistics: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circstd.html
def circular_std_deg(angles_rad: np.ndarray) -> float:
    angles_rad = np.asarray(angles_rad, dtype=float)
    angles_rad = angles_rad[np.isfinite(angles_rad)]

    if len(angles_rad) == 0:
        return np.nan

    r = np.sqrt(np.mean(np.sin(angles_rad)) ** 2 + np.mean(np.cos(angles_rad)) ** 2)

    if r <= 0:
        return np.nan

    return float(np.degrees(np.sqrt(-2 * np.log(r))))


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        return numeric.fillna(0) > 0

    lowered = series.astype(str).str.lower()
    return lowered.isin(["true", "1", "yes", "y"])


def estimate_surface_area_um2(mask_zyx: np.ndarray) -> float:
    if not COMPUTE_SURFACE_AREA:
        return np.nan

    if mask_zyx.sum() == 0:
        return np.nan

    padded = np.pad(mask_zyx.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)

    try:
        verts, faces, _, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=(Z_STEP_SIZE_UM, XY_PIXEL_SIZE_UM, XY_PIXEL_SIZE_UM),
        )
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return np.nan


# Principal-axis method: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def principal_axis_features_um(coords_zyx: np.ndarray) -> dict:
    if coords_zyx.shape[0] < 4:
        return {
            "principal_axis_length_1_um": np.nan,
            "principal_axis_length_2_um": np.nan,
            "principal_axis_length_3_um": np.nan,
            "aspect_ratio_3d": np.nan,
            "elongation": np.nan,
            "flatness": np.nan,
            "prolate_ellipticity": np.nan,
            "oblate_ellipticity": np.nan,
        }

    pts = np.column_stack(
        [
            coords_zyx[:, 0] * Z_STEP_SIZE_UM,
            coords_zyx[:, 1] * XY_PIXEL_SIZE_UM,
            coords_zyx[:, 2] * XY_PIXEL_SIZE_UM,
        ]
    )

    try:
        cov = np.cov(pts, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0))[::-1]

        # Approximate physical object lengths from coordinate spread.
        lengths = 4.0 * np.sqrt(eigvals)
        l1, l2, l3 = lengths

        eps = 1e-9

        return {
            "principal_axis_length_1_um": float(l1),
            "principal_axis_length_2_um": float(l2),
            "principal_axis_length_3_um": float(l3),
            "aspect_ratio_3d": float(l1 / (l3 + eps)),
            "elongation": float(1.0 - (l2 / (l1 + eps))),
            "flatness": float(1.0 - (l3 / (l2 + eps))),
            "prolate_ellipticity": float((l1 - l2) / (l1 + eps)),
            "oblate_ellipticity": float((l2 - l3) / (l2 + eps)),
        }

    except Exception:
        return {
            "principal_axis_length_1_um": np.nan,
            "principal_axis_length_2_um": np.nan,
            "principal_axis_length_3_um": np.nan,
            "aspect_ratio_3d": np.nan,
            "elongation": np.nan,
            "flatness": np.nan,
            "prolate_ellipticity": np.nan,
            "oblate_ellipticity": np.nan,
        }



# Object-level features


def extract_object_features_for_tracked_labels(
    tracks: pd.DataFrame,
    labels_tzyx: np.ndarray,
    raw_tzyx: np.ndarray | None,
) -> pd.DataFrame:
    needed = tracks[["time", "object_label"]].drop_duplicates().copy()
    features = []

    times = sorted(needed["time"].unique())

    for t in times:
        if t < 0 or t >= labels_tzyx.shape[0]:
            print(f"[WARN] Skipping time {t}; outside label stack.")
            continue

        wanted_labels = set(needed.loc[needed["time"] == t, "object_label"].astype(int).tolist())

        label_img = labels_tzyx[t]
        raw_img = None

        if raw_tzyx is not None and t < raw_tzyx.shape[0] and raw_tzyx[t].shape == label_img.shape:
            raw_img = raw_tzyx[t]

        print(f"[INFO] Extracting object features at T={t}, labels={len(wanted_labels)}")

        for prop in regionprops(label_img, intensity_image=raw_img):
            lab = int(prop.label)

            if lab not in wanted_labels:
                continue

            coords = prop.coords
            volume_voxels = int(prop.area)
            volume_um3 = float(volume_voxels * Z_STEP_SIZE_UM * XY_PIXEL_SIZE_UM * XY_PIXEL_SIZE_UM)

            z0, y0, x0, z1, y1, x1 = prop.bbox

            bbox_z_length_vox = z1 - z0
            bbox_y_length_vox = y1 - y0
            bbox_x_length_vox = x1 - x0

            bbox_z_length_um = bbox_z_length_vox * Z_STEP_SIZE_UM
            bbox_y_length_um = bbox_y_length_vox * XY_PIXEL_SIZE_UM
            bbox_x_length_um = bbox_x_length_vox * XY_PIXEL_SIZE_UM

            bbox_volume_um3 = bbox_z_length_um * bbox_y_length_um * bbox_x_length_um
            extent_3d = volume_um3 / bbox_volume_um3 if bbox_volume_um3 > 0 else np.nan

            projected_area_xy_um2 = (
                pd.DataFrame(coords[:, [1, 2]]).drop_duplicates().shape[0]
                * XY_PIXEL_SIZE_UM
                * XY_PIXEL_SIZE_UM
            )

            z_counts = pd.Series(coords[:, 0]).value_counts()
            max_cross_section_area_um2 = (
                float(z_counts.max()) * XY_PIXEL_SIZE_UM * XY_PIXEL_SIZE_UM
                if len(z_counts) > 0
                else np.nan
            )

            surface_area_um2 = estimate_surface_area_um2(prop.image)

            if surface_area_um2 and np.isfinite(surface_area_um2) and surface_area_um2 > 0:
                sphericity = (
                    (math.pi ** (1.0 / 3.0))
                    * ((6.0 * volume_um3) ** (2.0 / 3.0))
                    / surface_area_um2
                )
                compactness_3d = (
                    (surface_area_um2 ** 3.0)
                    / (36.0 * math.pi * (volume_um3 ** 2.0))
                    if volume_um3 > 0
                    else np.nan
                )
                surface_area_to_volume = surface_area_um2 / volume_um3 if volume_um3 > 0 else np.nan
            else:
                sphericity = np.nan
                compactness_3d = np.nan
                surface_area_to_volume = np.nan

            axis_features = principal_axis_features_um(coords)

            coord_span = np.ptp(coords, axis=0)

            if coords.shape[0] >= 4 and np.count_nonzero(coord_span > 0) >= 3:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        solidity_3d = float(prop.solidity)
                except Exception:
                    solidity_3d = np.nan
            else:
                solidity_3d = np.nan

            if raw_img is not None:
                intensity_img = getattr(prop, "image_intensity", None)

                if intensity_img is None:
                    intensity_img = prop.intensity_image

                vals = intensity_img[prop.image].astype(float)
                mean_intensity = float(np.nanmean(vals)) if vals.size else np.nan
                median_intensity = float(np.nanmedian(vals)) if vals.size else np.nan
                max_intensity = float(np.nanmax(vals)) if vals.size else np.nan
                sum_intensity = float(np.nansum(vals)) if vals.size else np.nan
                intensity_std = float(np.nanstd(vals)) if vals.size else np.nan
                intensity_cv = intensity_std / mean_intensity if mean_intensity not in (0, np.nan) else np.nan
            else:
                mean_intensity = np.nan
                median_intensity = np.nan
                max_intensity = np.nan
                sum_intensity = np.nan
                intensity_std = np.nan
                intensity_cv = np.nan

            features.append(
                {
                    "time": int(t),
                    "object_label": lab,

                    "volume_voxels": volume_voxels,
                    "volume_um3": volume_um3,

                    "surface_area_um2": surface_area_um2,
                    "surface_area_to_volume": surface_area_to_volume,

                    "projected_area_xy_um2": projected_area_xy_um2,
                    "max_cross_section_area_um2": max_cross_section_area_um2,

                    "bbox_z_length_vox": bbox_z_length_vox,
                    "bbox_y_length_vox": bbox_y_length_vox,
                    "bbox_x_length_vox": bbox_x_length_vox,
                    "bbox_z_length_um": bbox_z_length_um,
                    "bbox_y_length_um": bbox_y_length_um,
                    "bbox_x_length_um": bbox_x_length_um,

                    "z_span_vox": bbox_z_length_vox,
                    "extent_3d": extent_3d,
                    "solidity_3d": solidity_3d,
                    "sphericity": sphericity,
                    "compactness_3d": compactness_3d,

                    **axis_features,

                    "mean_intensity": mean_intensity,
                    "median_intensity": median_intensity,
                    "max_intensity": max_intensity,
                    "sum_intensity": sum_intensity,
                    "intensity_std": intensity_std,
                    "intensity_cv": intensity_cv,
                }
            )

    return pd.DataFrame(features)



# Track-level features


def summarise_time_feature(out: dict, group: pd.DataFrame, source_col: str, name: str):
    if source_col not in group.columns:
        out[f"mean_{name}"] = np.nan
        out[f"median_{name}"] = np.nan
        out[f"{name}_std"] = np.nan
        out[f"{name}_cv"] = np.nan
        out[f"{name}_change_start_to_end"] = np.nan
        out[f"{name}_slope"] = np.nan
        return

    values = pd.to_numeric(group[source_col], errors="coerce").to_numpy(dtype=float)
    times = group["time"].to_numpy(dtype=float)

    finite = values[np.isfinite(values)]

    out[f"mean_{name}"] = float(np.nanmean(finite)) if len(finite) else np.nan
    out[f"median_{name}"] = float(np.nanmedian(finite)) if len(finite) else np.nan
    out[f"{name}_std"] = float(np.nanstd(finite)) if len(finite) else np.nan
    out[f"{name}_cv"] = safe_cv(values)

    if len(values) >= 2:
        out[f"{name}_change_start_to_end"] = float(values[-1] - values[0])
    else:
        out[f"{name}_change_start_to_end"] = np.nan

    out[f"{name}_slope"] = safe_slope(times, values)


def add_cluster_track_features(out: dict, group: pd.DataFrame):
    if not INCLUDE_CLUSTER_FEATURES:
        return

    if "inside_cluster" in group.columns:
        inside = to_bool_series(group["inside_cluster"])
        out["inside_cluster_fraction"] = float(inside.mean())
        out["ever_inside_cluster"] = bool(inside.any())
    else:
        out["inside_cluster_fraction"] = np.nan
        out["ever_inside_cluster"] = False

    if "near_cluster_boundary" in group.columns:
        near = to_bool_series(group["near_cluster_boundary"])
        out["near_cluster_boundary_fraction"] = float(near.mean())
        ever_near = bool(near.any())
    else:
        out["near_cluster_boundary_fraction"] = np.nan
        ever_near = False

    if "overlap_cluster_mask" in group.columns:
        overlap_bool = to_bool_series(group["overlap_cluster_mask"])
        out["overlap_cluster_mask_fraction"] = float(overlap_bool.mean())
        ever_overlap = bool(overlap_bool.any())
    else:
        out["overlap_cluster_mask_fraction"] = np.nan
        ever_overlap = False

    if "cluster_overlap_fraction" in group.columns:
        vals = pd.to_numeric(group["cluster_overlap_fraction"], errors="coerce")
        out["mean_cluster_overlap_fraction"] = float(vals.mean())
        out["max_cluster_overlap_fraction"] = float(vals.max())
    else:
        out["mean_cluster_overlap_fraction"] = np.nan
        out["max_cluster_overlap_fraction"] = np.nan

    dist_col = None
    if "distance_to_cluster_boundary_um" in group.columns:
        dist_col = "distance_to_cluster_boundary_um"
    elif "distance_to_cluster_boundary_px" in group.columns:
        dist_col = "distance_to_cluster_boundary_px"

    if dist_col is not None:
        vals = pd.to_numeric(group[dist_col], errors="coerce")
        out[f"mean_{dist_col}"] = float(vals.mean())
        out[f"min_{dist_col}"] = float(vals.min())
    else:
        out["mean_distance_to_cluster_boundary"] = np.nan
        out["min_distance_to_cluster_boundary"] = np.nan

    if out.get("inside_cluster_fraction", 0) > 0 or ever_overlap:
        out["track_region_type"] = "cluster_overlap"
    elif ever_near:
        out["track_region_type"] = "near_cluster_boundary"
    else:
        out["track_region_type"] = "outside_only"


# Cell-trajectory metrics: https://pubmed.ncbi.nlm.nih.gov/27713081/
def extract_track_features(tracked_objects: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for track_id, group in tracked_objects.groupby("track_id", sort=False):
        group = group.sort_values("time").copy()

        times = group["time"].to_numpy(dtype=float)

        z_um = group["centroid_z"].to_numpy(dtype=float) * Z_STEP_SIZE_UM
        y_um = group["centroid_y"].to_numpy(dtype=float) * XY_PIXEL_SIZE_UM
        x_um = group["centroid_x"].to_numpy(dtype=float) * XY_PIXEL_SIZE_UM

        n = len(group)
        start_time = int(np.nanmin(times))
        end_time = int(np.nanmax(times))
        duration_frames = int(end_time - start_time + 1)
        gap_count = int(duration_frames - n)
        track_completeness = n / duration_frames if duration_frames > 0 else np.nan

        out = {
            "file": group["file"].iloc[0] if "file" in group.columns else "",
            "cell_type": group["cell_type"].iloc[0] if "cell_type" in group.columns else CELL_TYPE,
            "track_id": track_id,
            "track_length": int(n),
            "start_time": start_time,
            "end_time": end_time,
            "duration_frames": duration_frames,
            "gap_count": gap_count,
            "track_completeness": track_completeness,

            "start_x_um": float(x_um[0]),
            "start_y_um": float(y_um[0]),
            "start_z_um": float(z_um[0]),
            "end_x_um": float(x_um[-1]),
            "end_y_um": float(y_um[-1]),
            "end_z_um": float(z_um[-1]),

            "x_range_um": float(np.nanmax(x_um) - np.nanmin(x_um)),
            "y_range_um": float(np.nanmax(y_um) - np.nanmin(y_um)),
            "z_range_um": float(np.nanmax(z_um) - np.nanmin(z_um)),
            "z_displacement_um": float(z_um[-1] - z_um[0]),
        }

        if n >= 2:
            dt = np.diff(times)
            dx = np.diff(x_um)
            dy = np.diff(y_um)
            dz = np.diff(z_um)

            valid = dt > 0

            step_dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            step_dist_xy = np.sqrt(dx**2 + dy**2)

            step_dist_3d_valid = step_dist_3d[valid]
            step_dist_xy_valid = step_dist_xy[valid]
            dt_valid = dt[valid]

            total_path_length_3d = float(np.nansum(step_dist_3d_valid))
            total_path_length_xy = float(np.nansum(step_dist_xy_valid))
            z_path_length = float(np.nansum(np.abs(dz[valid])))

            net_displacement_3d = float(
                np.sqrt((x_um[-1] - x_um[0]) ** 2 + (y_um[-1] - y_um[0]) ** 2 + (z_um[-1] - z_um[0]) ** 2)
            )
            net_displacement_xy = float(
                np.sqrt((x_um[-1] - x_um[0]) ** 2 + (y_um[-1] - y_um[0]) ** 2)
            )

            speed_um_per_frame = step_dist_3d_valid / dt_valid
            speed_xy_um_per_frame = step_dist_xy_valid / dt_valid

            if FRAME_INTERVAL_MIN is not None and FRAME_INTERVAL_MIN > 0:
                speed_um_per_min = step_dist_3d_valid / (dt_valid * FRAME_INTERVAL_MIN)
            else:
                speed_um_per_min = np.full_like(speed_um_per_frame, np.nan)

            dist_from_start_3d = np.sqrt(
                (x_um - x_um[0]) ** 2 + (y_um - y_um[0]) ** 2 + (z_um - z_um[0]) ** 2
            )
            msd_3d = float(np.nanmean(dist_from_start_3d**2))

            directionality_ratio = (
                net_displacement_3d / total_path_length_3d
                if total_path_length_3d > 0
                else np.nan
            )

            tortuosity = (
                total_path_length_3d / net_displacement_3d
                if net_displacement_3d > 0
                else np.nan
            )

            moving_step_fraction = (
                float(np.mean(step_dist_3d_valid > MOVING_STEP_THRESHOLD_UM))
                if len(step_dist_3d_valid) > 0
                else np.nan
            )

            acceleration = np.diff(speed_um_per_frame)

            vx = dx[valid] / dt_valid
            vy = dy[valid] / dt_valid
            vz = dz[valid] / dt_valid

            moving_mask = step_dist_3d_valid > 1e-9

            angle_xy = np.arctan2(dy[valid][moving_mask], dx[valid][moving_mask])
            angle_xz = np.arctan2(dz[valid][moving_mask], dx[valid][moving_mask])
            angle_yz = np.arctan2(dz[valid][moving_mask], dy[valid][moving_mask])

            out.update(
                {
                    "total_path_length_3d_um": total_path_length_3d,
                    "total_path_length_xy_um": total_path_length_xy,
                    "z_path_length_um": z_path_length,

                    "net_displacement_3d_um": net_displacement_3d,
                    "net_displacement_xy_um": net_displacement_xy,
                    "mean_squared_displacement_3d_um2": msd_3d,

                    "mean_step_distance_3d_um": float(np.nanmean(step_dist_3d_valid)),
                    "median_step_distance_3d_um": float(np.nanmedian(step_dist_3d_valid)),
                    "max_step_distance_3d_um": float(np.nanmax(step_dist_3d_valid)),

                    "mean_speed_um_per_frame": float(np.nanmean(speed_um_per_frame)),
                    "median_speed_um_per_frame": float(np.nanmedian(speed_um_per_frame)),
                    "max_speed_um_per_frame": float(np.nanmax(speed_um_per_frame)),
                    "speed_std_um_per_frame": float(np.nanstd(speed_um_per_frame)),
                    "speed_cv_um_per_frame": safe_cv(speed_um_per_frame),

                    "mean_speed_xy_um_per_frame": float(np.nanmean(speed_xy_um_per_frame)),

                    "mean_speed_um_per_min": float(np.nanmean(speed_um_per_min)) if np.isfinite(speed_um_per_min).any() else np.nan,
                    "median_speed_um_per_min": float(np.nanmedian(speed_um_per_min)) if np.isfinite(speed_um_per_min).any() else np.nan,
                    "max_speed_um_per_min": float(np.nanmax(speed_um_per_min)) if np.isfinite(speed_um_per_min).any() else np.nan,

                    "mean_acceleration_um_per_frame2": float(np.nanmean(acceleration)) if len(acceleration) else np.nan,
                    "acceleration_std_um_per_frame2": float(np.nanstd(acceleration)) if len(acceleration) else np.nan,

                    "directionality_ratio": directionality_ratio,
                    "tortuosity": tortuosity,
                    "moving_step_fraction": moving_step_fraction,

                    "mean_velocity_x_um_per_frame": float(np.nanmean(vx)),
                    "mean_velocity_y_um_per_frame": float(np.nanmean(vy)),
                    "mean_velocity_z_um_per_frame": float(np.nanmean(vz)),

                    "fraction_steps_positive_x": float(np.mean(vx > 0)),
                    "fraction_steps_positive_y": float(np.mean(vy > 0)),
                    "fraction_steps_positive_z": float(np.mean(vz > 0)),

                    "velocity_angle_xy_mean_deg": circular_mean_deg(angle_xy),
                    "velocity_angle_xz_mean_deg": circular_mean_deg(angle_xz),
                    "velocity_angle_yz_mean_deg": circular_mean_deg(angle_yz),

                    "velocity_angle_xy_std_deg": circular_std_deg(angle_xy),
                    "velocity_angle_xz_std_deg": circular_std_deg(angle_xz),
                    "velocity_angle_yz_std_deg": circular_std_deg(angle_yz),
                }
            )

        else:
            movement_nan_cols = [
                "total_path_length_3d_um",
                "total_path_length_xy_um",
                "z_path_length_um",
                "net_displacement_3d_um",
                "net_displacement_xy_um",
                "mean_squared_displacement_3d_um2",
                "mean_step_distance_3d_um",
                "median_step_distance_3d_um",
                "max_step_distance_3d_um",
                "mean_speed_um_per_frame",
                "median_speed_um_per_frame",
                "max_speed_um_per_frame",
                "speed_std_um_per_frame",
                "speed_cv_um_per_frame",
                "mean_speed_xy_um_per_frame",
                "mean_speed_um_per_min",
                "median_speed_um_per_min",
                "max_speed_um_per_min",
                "mean_acceleration_um_per_frame2",
                "acceleration_std_um_per_frame2",
                "directionality_ratio",
                "tortuosity",
                "moving_step_fraction",
                "mean_velocity_x_um_per_frame",
                "mean_velocity_y_um_per_frame",
                "mean_velocity_z_um_per_frame",
                "fraction_steps_positive_x",
                "fraction_steps_positive_y",
                "fraction_steps_positive_z",
                "velocity_angle_xy_mean_deg",
                "velocity_angle_xz_mean_deg",
                "velocity_angle_yz_mean_deg",
                "velocity_angle_xy_std_deg",
                "velocity_angle_xz_std_deg",
                "velocity_angle_yz_std_deg",
            ]
            for c in movement_nan_cols:
                out[c] = np.nan

        # Track-level summaries of morphology/intensity over time.
        summary_map = {
            "volume_um3": "volume_um3",
            "surface_area_um2": "surface_area_um2",
            "surface_area_to_volume": "surface_area_to_volume",
            "projected_area_xy_um2": "projected_area_xy_um2",
            "max_cross_section_area_um2": "max_cross_section_area_um2",
            "sphericity": "sphericity",
            "elongation": "elongation",
            "flatness": "flatness",
            "aspect_ratio_3d": "aspect_ratio_3d",
            "prolate_ellipticity": "prolate_ellipticity",
            "oblate_ellipticity": "oblate_ellipticity",
            "compactness_3d": "compactness_3d",
            "solidity_3d": "solidity_3d",
            "extent_3d": "extent_3d",
            "bbox_x_length_um": "bbox_x_length_um",
            "bbox_y_length_um": "bbox_y_length_um",
            "bbox_z_length_um": "bbox_z_length_um",
            "z_span_vox": "z_span_vox",
            "mean_intensity": "intensity",
            "max_intensity": "max_intensity",
            "sum_intensity": "sum_intensity",
            "intensity_cv": "object_intensity_cv",
        }

        for source_col, name in summary_map.items():
            summarise_time_feature(out, group, source_col, name)

        out["max_volume_fold_change"] = (
            safe_fold_change(group["volume_um3"].to_numpy(dtype=float))
            if "volume_um3" in group.columns
            else np.nan
        )

        out["max_intensity_fold_change"] = (
            safe_fold_change(group["mean_intensity"].to_numpy(dtype=float))
            if "mean_intensity" in group.columns
            else np.nan
        )

        add_cluster_track_features(out, group)

        rows.append(out)

    return pd.DataFrame(rows)



# Macrophage static filtering


def add_macrophage_static_flags(track_features: pd.DataFrame) -> pd.DataFrame:
    df = track_features.copy()

    if CELL_TYPE != "macrophage":
        return df

    motility_class = []
    static_exclude = []
    static_reason = []

    for _, row in df.iterrows():
        track_length = row.get("track_length", np.nan)
        path = row.get("total_path_length_3d_um", np.nan)
        net = row.get("net_displacement_3d_um", np.nan)
        moving_fraction = row.get("moving_step_fraction", np.nan)

        if pd.isna(track_length) or track_length < STATIC_MIN_TRACK_LENGTH:
            motility_class.append("too_short_to_classify")
            static_exclude.append(False)
            static_reason.append("")
            continue

        is_static = (
            pd.notna(path)
            and pd.notna(net)
            and pd.notna(moving_fraction)
            and path < STATIC_TOTAL_PATH_UM
            and net < STATIC_NET_DISPLACEMENT_UM
            and moving_fraction < STATIC_MOVING_STEP_FRACTION
        )

        if is_static:
            motility_class.append("static_candidate")
            static_exclude.append(True)
            static_reason.append(
                f"path<{STATIC_TOTAL_PATH_UM}um; net<{STATIC_NET_DISPLACEMENT_UM}um; "
                f"moving_fraction<{STATIC_MOVING_STEP_FRACTION}"
            )
        elif pd.notna(path) and pd.notna(moving_fraction) and (path >= 10.0 or moving_fraction >= 0.30):
            motility_class.append("motile")
            static_exclude.append(False)
            static_reason.append("")
        else:
            motility_class.append("low_motility")
            static_exclude.append(False)
            static_reason.append("")

    df["macrophage_motility_class"] = motility_class
    df["macrophage_static_exclude"] = static_exclude
    df["macrophage_static_reason"] = static_reason

    return df



# Main


def main():
    tracks_csv = find_tracks_csv()

    print("============================================================")
    print("[INFO] Extracting tracked-cell features")
    print("============================================================")
    print(f"[INFO] CELL_TYPE: {CELL_TYPE}")
    print(f"[INFO] REGION_MODE: {REGION_MODE}")
    print(f"[INFO] INCLUDE_CLUSTER_FEATURES: {INCLUDE_CLUSTER_FEATURES}")
    print(f"[INFO] Tracks CSV: {tracks_csv}")
    print(f"[INFO] Labels TIF: {LABELS_3D_TZYX_TIF}")
    print(f"[INFO] Raw TIF: {RAW_TCZYX_TIF}")
    print(f"[INFO] XY_PIXEL_SIZE_UM: {XY_PIXEL_SIZE_UM}")
    print(f"[INFO] Z_STEP_SIZE_UM: {Z_STEP_SIZE_UM}")
    print(f"[INFO] FRAME_INTERVAL_MIN: {FRAME_INTERVAL_MIN}")

    tracks = pd.read_csv(tracks_csv)
    tracks = normalise_track_columns(tracks)

    print(f"[INFO] Tracked rows loaded: {len(tracks)}")
    print(f"[INFO] Unique tracks: {tracks['track_id'].nunique()}")

    labels_tzyx = load_labels_tzyx(LABELS_3D_TZYX_TIF)
    raw_tzyx = load_raw_as_tzyx(RAW_TCZYX_TIF, CHANNEL_INDEX)

    object_features = extract_object_features_for_tracked_labels(
        tracks=tracks,
        labels_tzyx=labels_tzyx,
        raw_tzyx=raw_tzyx,
    )

    print(f"[INFO] Object feature rows: {len(object_features)}")

    # Remove columns from tracks that we recomputed from the actual 3D label image.
    recomputed_cols = [c for c in object_features.columns if c not in {"time", "object_label"}]
    tracks_for_merge = tracks.drop(columns=[c for c in recomputed_cols if c in tracks.columns], errors="ignore")

    tracked_objects = tracks_for_merge.merge(
        object_features,
        on=["time", "object_label"],
        how="left",
    )

    tracked_objects["centroid_x_um"] = tracked_objects["centroid_x"] * XY_PIXEL_SIZE_UM
    tracked_objects["centroid_y_um"] = tracked_objects["centroid_y"] * XY_PIXEL_SIZE_UM
    tracked_objects["centroid_z_um"] = tracked_objects["centroid_z"] * Z_STEP_SIZE_UM

    if not INCLUDE_CLUSTER_FEATURES:
        tracked_objects = tracked_objects.drop(columns=[c for c in CLUSTER_COLS if c in tracked_objects.columns], errors="ignore")

    track_features = extract_track_features(tracked_objects)
    track_features = add_macrophage_static_flags(track_features)

    block_name_value = BLOCK_NAME
    fish_id_value = FISH_ID if FISH_ID else BLOCK_NAME
    genotype_value = GENOTYPE

    for df_out in [tracked_objects, track_features]:
        df_out["block_name"] = block_name_value
        df_out["fish_id"] = fish_id_value
        df_out["genotype"] = genotype_value

    if CELL_TYPE == "macrophage":
        region_suffix = REGION_MODE
        cluster_suffix = "with_cluster_features" if INCLUDE_CLUSTER_FEATURES else "no_cluster_features"
        output_prefix = f"{CELL_TYPE}_{region_suffix}_{cluster_suffix}"
    else:
        output_prefix = f"{CELL_TYPE}_tracked_features"

    if FEATURE_METHOD:
        output_prefix = f"{output_prefix}_{FEATURE_METHOD}"

    output_dir = Path(get_setting("FEATURE_OUTPUT_DIR", None, TRACKING_OUTPUT_DIR / "feature_extraction"))
    output_dir.mkdir(parents=True, exist_ok=True)

    object_out = output_dir / f"{output_prefix}_object_timepoint_features.csv"
    track_out = output_dir / f"{output_prefix}_cell_track_features.csv"

    tracked_objects.to_csv(object_out, index=False)
    track_features.to_csv(track_out, index=False)

    print(f"[SAVED] Object-timepoint features: {object_out}")
    print(f"[SAVED] Cell/track-level features: {track_out}")

    if CELL_TYPE == "macrophage":
        filtered = track_features.copy()

        if "macrophage_static_exclude" in filtered.columns:
            filtered = filtered[~filtered["macrophage_static_exclude"]].copy()

        filtered_out = output_dir / f"{output_prefix}_cell_track_features_motile_filtered.csv"
        filtered.to_csv(filtered_out, index=False)

        print(f"[SAVED] Motile-filtered macrophage features: {filtered_out}")
        print("[INFO] Macrophage motility class counts:")
        print(track_features["macrophage_motility_class"].value_counts(dropna=False))

    print("[DONE]")


if __name__ == "__main__":
    main()
