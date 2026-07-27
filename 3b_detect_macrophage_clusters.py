from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.filters import threshold_otsu

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    RAW_TCZYX_TIF,
    LABELS_3D_TZYX_TIF,
    OBJECT_FEATURES_CSV,
    PROJECT_DIR,
)



# User settings



# Cluster detection settings


# Use local macrophage density rather than simple dilation.
# Larger sigma = smoother/larger cluster core.
CLUSTER_DENSITY_SIGMA = 8.0

# Keep only pixels whose local macrophage density is high.
# Increase this if isolated cells are included.
# Decrease if the real cluster becomes too small.
CLUSTER_DENSITY_PERCENTILE = 70

# Minimum density threshold after gaussian smoothing.
# Increase if weak outside regions are included.
MIN_CLUSTER_DENSITY = 0.08

# Final small expansion after dense core is selected.
# Keep this much smaller than before.
CLUSTER_FINAL_DILATION_RADIUS = 4

# Smooth/close small gaps in selected cluster.
CLUSTER_CLOSING_RADIUS = 4

# Minimum XY-projection area for a dense component to count as a cluster.
CLUSTER_MIN_AREA_XY = 1500

# Usually the wound macrophage cluster is the largest dense component.
KEEP_ONLY_LARGEST_CLUSTER = True

# Objects within this many XY pixels of the cluster edge are marked as boundary.
CLUSTER_BOUNDARY_DISTANCE_PX = 30

OBJECT_CLUSTER_OVERLAP_MIN_PIXELS = 1

# Fraction of object XY projection overlapping the cluster.
# Use 0.0 for strict "any overlap removes it".
OBJECT_CLUSTER_OVERLAP_MIN_FRACTION = 0.0

# Larger mask used only for excluding objects from single-cell tracking.
# This should cover the visually crowded wound/cluster area in the Z-projection.
CLUSTER_TRACKING_EXCLUSION_DILATION_RADIUS = 25


# Cluster mask temporal persistence


# If the cluster detector fails at a timepoint, reuse the most recent
# valid cluster mask. This prevents the exclusion region from disappearing
# for one/few bad frames.
PERSIST_CLUSTER_MASK_WHEN_MISSING = True

# Treat masks smaller than this as failed/missing detections.
# This prevents tiny false fragments from replacing a good previous mask.
MIN_VALID_CLUSTER_AREA_FOR_PERSISTENCE = CLUSTER_MIN_AREA_XY



# Optional fish/body mask


USE_FISH_BODY_MASK = True

# Minimum area for fish/body mask component.
FISH_BODY_MIN_AREA_XY = 15000

# Fill small holes in body mask.
FISH_BODY_HOLE_AREA = 5000

# Expand body mask slightly so the true edge is not cut off.
FISH_BODY_DILATION_RADIUS = 3

# Output files
CLUSTER_MASK_TYX_TIF = PROJECT_DIR / f"{CELL_TYPE}_cluster_mask_TYX.tif"

CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF = (
    PROJECT_DIR / f"{CELL_TYPE}_cluster_tracking_exclusion_mask_TYX.tif"
)

OBJECT_FEATURES_WITH_CLUSTER_CSV = (
    PROJECT_DIR / f"{CELL_TYPE}_3d_object_features_with_cluster_flags.csv"
)

OBJECT_FEATURES_OUTSIDE_BOUNDARY_CSV = (
    PROJECT_DIR / f"{CELL_TYPE}_3d_object_features_outside_boundary.csv"
)

OBJECT_FEATURES_OUTSIDE_ONLY_CSV = (
    PROJECT_DIR / f"{CELL_TYPE}_3d_object_features_outside_only.csv"
)

CLUSTER_FEATURES_CSV = (
    PROJECT_DIR / f"{CELL_TYPE}_cluster_level_features.csv"
)



# Helpers


def load_selected_raw_channel():
    raw = tiff.imread(RAW_TCZYX_TIF)

    if raw.ndim == 5:
        return raw[:, CHANNEL_INDEX, :, :, :]
    elif raw.ndim == 4:
        return raw
    else:
        raise ValueError(f"Unexpected raw shape: {raw.shape}")

# Morphology reference: https://scikit-image.org/docs/stable/api/skimage.morphology.html
def make_fish_body_mask_tyx():
    """Create a rough fish/body mask from the raw image."""

    raw = tiff.imread(RAW_TCZYX_TIF)

    if raw.ndim == 5:
        # raw is T,C,Z,Y,X. Use all channels and all Z to estimate body.
        raw_proj_tyx = raw.max(axis=(1, 2))
    elif raw.ndim == 4:
        # raw is T,Z,Y,X
        raw_proj_tyx = raw.max(axis=1)
    else:
        raise ValueError(f"Unexpected raw shape for fish mask: {raw.shape}")

    T, Y, X = raw_proj_tyx.shape
    fish_masks = np.zeros((T, Y, X), dtype=bool)

    for t in range(T):
        img = raw_proj_tyx[t].astype(np.float32)

        # Robust normalisation
        lo, hi = np.percentile(img, [1, 99.8])
        if hi <= lo:
            continue

        img = np.clip((img - lo) / (hi - lo), 0, 1)
        img = gaussian_filter(img, sigma=2.0)

        try:
            thr = threshold_otsu(img)
        except ValueError:
            continue

        # Use a slightly relaxed Otsu threshold to keep dim fish tissue.
        mask = img > max(thr * 0.6, 0.03)

        mask = binary_closing(mask, disk(10))
        mask = remove_small_holes(mask, area_threshold=FISH_BODY_HOLE_AREA)
        mask = remove_small_objects(mask, min_size=FISH_BODY_MIN_AREA_XY)

        cc = label(mask)
        props = regionprops(cc)

        if len(props) == 0:
            continue

        # Keep largest connected tissue/body component.
        largest = max(props, key=lambda p: p.area)
        mask = cc == largest.label

        mask = binary_dilation(mask, disk(FISH_BODY_DILATION_RADIUS))

        fish_masks[t] = mask

    return fish_masks

# Otsu thresholding: https://scikit-image.org/docs/stable/api/skimage.filters.html
# Connected regions: https://scikit-image.org/docs/stable/api/skimage.measure.html
def detect_cluster_mask_for_timepoint(label_3d, fish_mask_xy=None):
    """Detect dense macrophage cluster in one timepoint."""

    # XY projection of all macrophage labels
    binary_xy = (label_3d > 0).max(axis=0)

    # Do not allow cluster outside the fish/body mask
    if fish_mask_xy is not None:
        binary_xy = binary_xy & fish_mask_xy

    if binary_xy.sum() == 0:
        return np.zeros(binary_xy.shape, dtype=np.uint16)

    # Local macrophage density map
    density = gaussian_filter(
        binary_xy.astype(np.float32),
        sigma=CLUSTER_DENSITY_SIGMA,
    )

    density_values = density[binary_xy]

    if len(density_values) == 0:
        return np.zeros(binary_xy.shape, dtype=np.uint16)

    density_thr = max(
        np.percentile(density_values, CLUSTER_DENSITY_PERCENTILE),
        MIN_CLUSTER_DENSITY,
    )

    # Dense cluster core only
    cluster_core = density >= density_thr

    # Restrict again to fish/body if available
    if fish_mask_xy is not None:
        cluster_core = cluster_core & fish_mask_xy

    # Close small holes, then remove tiny components
    cluster_core = binary_closing(cluster_core, disk(CLUSTER_CLOSING_RADIUS))
    cluster_core = remove_small_objects(
        cluster_core,
        min_size=CLUSTER_MIN_AREA_XY,
    )

    cc = label(cluster_core)
    props = regionprops(cc)

    valid = [p for p in props if p.area >= CLUSTER_MIN_AREA_XY]

    cluster_mask = np.zeros_like(cc, dtype=np.uint16)

    if len(valid) == 0:
        return cluster_mask

    if KEEP_ONLY_LARGEST_CLUSTER:
        valid = [max(valid, key=lambda p: p.area)]

    for new_id, p in enumerate(valid, start=1):
        cluster_mask[cc == p.label] = new_id

    # Final small dilation to include immediate cluster boundary
    if CLUSTER_FINAL_DILATION_RADIUS > 0:
        final_binary = binary_dilation(
            cluster_mask > 0,
            disk(CLUSTER_FINAL_DILATION_RADIUS),
        )

        if fish_mask_xy is not None:
            final_binary = final_binary & fish_mask_xy

        cluster_mask = final_binary.astype(np.uint16)

    return cluster_mask

def make_tracking_exclusion_mask(cluster_mask_xy, fish_mask_xy=None):
    """Create a larger exclusion mask for single-cell tracking."""

    exclusion = cluster_mask_xy > 0

    if CLUSTER_TRACKING_EXCLUSION_DILATION_RADIUS > 0:
        exclusion = binary_dilation(
            exclusion,
            disk(CLUSTER_TRACKING_EXCLUSION_DILATION_RADIUS),
        )

    if fish_mask_xy is not None:
        exclusion = exclusion & fish_mask_xy

    return exclusion.astype(np.uint16)

def persist_masks_forward_tyx(
    masks_tyx,
    name="mask",
    min_valid_area=1,
):
    """Forward-fill a TYX mask through time."""

    if not PERSIST_CLUSTER_MASK_WHEN_MISSING:
        return masks_tyx

    out = np.zeros_like(masks_tyx)

    last_valid_mask = None
    n_original_missing = 0
    n_persisted = 0

    T = masks_tyx.shape[0]

    for t in range(T):
        current = masks_tyx[t]

        current_area = int((current > 0).sum())
        current_is_valid = current_area >= int(min_valid_area)

        if current_is_valid:
            out[t] = current
            last_valid_mask = current.copy()

        else:
            n_original_missing += 1

            if last_valid_mask is not None:
                out[t] = last_valid_mask
                n_persisted += 1
                print(
                    f"[PERSIST] {name}: T={t} had no valid mask "
                    f"(area={current_area}); reused previous valid mask."
                )
            else:
                out[t] = current
                print(
                    f"[PERSIST] {name}: T={t} had no valid mask "
                    "and no previous valid mask exists yet."
                )

    print()
    print(f"[PERSIST SUMMARY] {name}")
    print(f"    Missing/invalid original masks: {n_original_missing}")
    print(f"    Timepoints forward-filled:      {n_persisted}")
    print()

    return out 

def _normalise_time_index(time_value, n_timepoints: int, time_offset: int) -> int:
    """Convert object table time values to the 0-based index used by labels_tzyx."""
    return int(time_value) - int(time_offset)


def infer_time_offset(object_df: pd.DataFrame, n_timepoints: int) -> int:
    """Infer whether object_df['time'] is 0-based or 1-based."""
    if "time" not in object_df.columns or len(object_df) == 0:
        return 0

    times = pd.to_numeric(object_df["time"], errors="coerce").dropna().astype(int)
    if len(times) == 0:
        return 0

    min_t = int(times.min())
    max_t = int(times.max())

    if min_t >= 0 and max_t <= n_timepoints - 1:
        return 0

    if min_t >= 1 and max_t <= n_timepoints:
        print("[WARNING] Object table time appears to be 1-based. Converting to 0-based internally.")
        return 1

    raise ValueError(
        "Object feature table has time values outside the label image range.\n"
        f"time min/max in table: {min_t}/{max_t}\n"
        f"labels_tzyx has valid 0-based time indices: 0..{n_timepoints - 1}"
    )


def find_object_label_column(
    object_df: pd.DataFrame,
    labels_tzyx: np.ndarray | None = None,
    time_offset: int = 0,
) -> str:
    """Find the object feature column that matches the label values in LABELS_3D_TZYX_TIF."""

    # Prefer true label columns over generic IDs. A generic object_id may be globally
    # unique and may not match the per-timepoint label values in the TIFF.
    candidate_cols = [
        "label_3d",
        "label_id",
        "label",
        "object_label",
        "object_id_3d",
        "reconstruction_id",
        "object_id",
        "id",
    ]

    present_cols = [c for c in candidate_cols if c in object_df.columns]

    if not present_cols:
        raise ValueError(
            "Could not find an object label column in the object feature table.\n"
            f"Available columns:\n{list(object_df.columns)}\n\n"
            "Expected one of these columns:\n"
            f"{candidate_cols}\n\n"
            "The chosen column must contain the numeric label value used in "
            "LABELS_3D_TZYX_TIF for each timepoint."
        )

    if labels_tzyx is None or "time" not in object_df.columns:
        return present_cols[0]

    n_timepoints = labels_tzyx.shape[0]

    # Cache valid labels for each timepoint.
    valid_labels_by_t = {}
    for t in range(n_timepoints):
        labs = np.unique(labels_tzyx[t])
        valid_labels_by_t[t] = set(int(v) for v in labs if int(v) > 0)

    scores = []
    for col in present_cols:
        values = pd.to_numeric(object_df[col], errors="coerce")
        times = pd.to_numeric(object_df["time"], errors="coerce")

        valid_rows = values.notna() & times.notna()
        if valid_rows.sum() == 0:
            scores.append((col, 0.0, 0, int(len(object_df))))
            continue

        checked = 0
        matched = 0

        for value, time_value in zip(values[valid_rows], times[valid_rows]):
            t = _normalise_time_index(time_value, n_timepoints, time_offset)
            if t < 0 or t >= n_timepoints:
                continue

            checked += 1
            if int(value) in valid_labels_by_t[t]:
                matched += 1

        score = matched / max(checked, 1)
        scores.append((col, score, matched, checked))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    best_col, best_score, best_matched, best_checked = scores[0]

    print("[INFO] Candidate object-label column match scores:")
    for col, score, matched, checked in scores:
        print(f"    {col}: {matched}/{checked} matched ({score:.3f})")

    if best_score < 0.50:
        raise ValueError(
            "Could not confidently match any object feature column to labels_tzyx.\n"
            "This means the table may not contain the same label IDs as the 3D label TIFF, "
            "or time indexing may be wrong.\n"
            f"Best candidate: {best_col} ({best_matched}/{best_checked} matched).\n"
            f"Available columns: {list(object_df.columns)}"
        )

    if best_score < 0.98:
        print(
            f"[WARNING] Best label column '{best_col}' only matched "
            f"{best_matched}/{best_checked} rows. Check this if many objects become outside_cluster."
        )

    return best_col


def _empty_cluster_info(region_class: str = "outside_cluster") -> dict:
    return {
        "cluster_id": 0,
        "inside_cluster": False,
        "near_cluster_boundary": False,
        "overlap_cluster_mask": False,
        "cluster_overlap_pixels": 0,
        "cluster_overlap_fraction": 0.0,
        "distance_to_cluster_boundary_px": np.nan,
        "cluster_region_class": region_class,
    }


def classify_objects_by_cluster_overlap(
    object_df: pd.DataFrame,
    labels_tzyx: np.ndarray,
    cluster_masks_tyx: np.ndarray,
) -> pd.DataFrame:
    """Classify each 3D object as:."""

    if labels_tzyx.ndim != 4:
        raise ValueError(f"Expected labels_tzyx with shape T,Z,Y,X. Got {labels_tzyx.shape}")

    if cluster_masks_tyx.ndim != 3:
        raise ValueError(
            f"Expected cluster_masks_tyx with shape T,Y,X. Got {cluster_masks_tyx.shape}"
        )

    if labels_tzyx.shape[0] != cluster_masks_tyx.shape[0]:
        raise ValueError(
            "Time dimension mismatch between labels_tzyx and cluster_masks_tyx:\n"
            f"labels_tzyx: {labels_tzyx.shape}\n"
            f"cluster_masks_tyx: {cluster_masks_tyx.shape}"
        )

    if labels_tzyx.shape[2:] != cluster_masks_tyx.shape[1:]:
        raise ValueError(
            "Y/X shape mismatch between labels_tzyx and cluster_masks_tyx:\n"
            f"labels_tzyx: {labels_tzyx.shape}\n"
            f"cluster_masks_tyx: {cluster_masks_tyx.shape}"
        )

    if "time" not in object_df.columns:
        raise ValueError(
            "Object feature table must contain a 'time' column.\n"
            f"Available columns: {list(object_df.columns)}"
        )

    object_df = object_df.copy()
    n_timepoints = labels_tzyx.shape[0]
    time_offset = infer_time_offset(object_df, n_timepoints)
    label_col = find_object_label_column(object_df, labels_tzyx, time_offset=time_offset)

    print(f"[INFO] Using object label column for overlap classification: {label_col}")

    # Build lookup:
    # key = (0-based time, object_label_in_labels_tzyx)
    # value = classification info
    classification_lookup = {}

    for t in range(n_timepoints):
        print(f"[INFO] Object-overlap cluster classification T={t + 1}/{n_timepoints}")

        label_3d = labels_tzyx[t]
        cluster_label_xy = cluster_masks_tyx[t].astype(np.uint16)
        cluster_binary_xy = cluster_label_xy > 0

        if label_3d.max() == 0:
            continue

        if cluster_binary_xy.sum() == 0:
            # No detected cluster at this timepoint, so every object is outside.
            for p in regionprops(label_3d):
                classification_lookup[(t, int(p.label))] = _empty_cluster_info()
            continue

        # Distance from outside pixels to nearest cluster pixel.
        # For pixels outside cluster: > 0. For cluster pixels: 0.
        outside_distance = distance_transform_edt(~cluster_binary_xy)

        # Distance from inside pixels to nearest outside pixel.
        # Used only for negative distance reporting for overlapping objects.
        inside_distance = distance_transform_edt(cluster_binary_xy)

        # Nearest cluster-pixel coordinates for every XY pixel. This lets us assign
        # a cluster_id to boundary objects, even if KEEP_ONLY_LARGEST_CLUSTER=False.
        _, nearest_cluster_indices = distance_transform_edt(
            ~cluster_binary_xy,
            return_indices=True,
        )

        for p in regionprops(label_3d):
            object_label = int(p.label)

            # p.bbox for a 3D object is:
            # min_z, min_y, min_x, max_z, max_y, max_x
            z0, y0, x0, z1, y1, x1 = p.bbox

            object_zyx = p.image.astype(bool)
            object_xy = object_zyx.max(axis=0)
            object_xy_pixels = int(object_xy.sum())

            if object_xy_pixels == 0:
                classification_lookup[(t, object_label)] = _empty_cluster_info()
                continue

            cluster_crop = cluster_binary_xy[y0:y1, x0:x1]
            cluster_label_crop = cluster_label_xy[y0:y1, x0:x1]

            overlap_xy = object_xy & cluster_crop
            overlap_pixels = int(overlap_xy.sum())
            overlap_fraction = overlap_pixels / max(object_xy_pixels, 1)

            overlaps_cluster = (
                overlap_pixels >= OBJECT_CLUSTER_OVERLAP_MIN_PIXELS
                and overlap_fraction >= OBJECT_CLUSTER_OVERLAP_MIN_FRACTION
            )

            if overlaps_cluster:
                overlapping_cluster_ids = cluster_label_crop[overlap_xy]
                overlapping_cluster_ids = overlapping_cluster_ids[overlapping_cluster_ids > 0]

                if len(overlapping_cluster_ids) > 0:
                    ids, counts = np.unique(overlapping_cluster_ids, return_counts=True)
                    cluster_id = int(ids[np.argmax(counts)])
                else:
                    cluster_id = 1

                inside_crop = inside_distance[y0:y1, x0:x1]
                distance_to_boundary = -float(np.max(inside_crop[overlap_xy]))

                classification_lookup[(t, object_label)] = {
                    "cluster_id": cluster_id,
                    "inside_cluster": True,
                    "near_cluster_boundary": False,
                    "overlap_cluster_mask": True,
                    "cluster_overlap_pixels": overlap_pixels,
                    "cluster_overlap_fraction": float(overlap_fraction),
                    "distance_to_cluster_boundary_px": distance_to_boundary,
                    "cluster_region_class": "inside_cluster",
                }
                continue

            # Only objects with zero overlap can reach this part.
            outside_crop = outside_distance[y0:y1, x0:x1]
            object_distances = outside_crop[object_xy]
            min_distance = float(np.min(object_distances)) if len(object_distances) else np.nan

            near_boundary = (
                np.isfinite(min_distance)
                and min_distance <= CLUSTER_BOUNDARY_DISTANCE_PX
            )

            if near_boundary:
                # Use the closest object pixel to find the nearest cluster ID.
                local_object_coords = np.argwhere(object_xy)
                closest_local_idx = int(np.argmin(object_distances))
                local_y, local_x = local_object_coords[closest_local_idx]
                global_y = int(y0 + local_y)
                global_x = int(x0 + local_x)

                nearest_y = int(nearest_cluster_indices[0, global_y, global_x])
                nearest_x = int(nearest_cluster_indices[1, global_y, global_x])
                cluster_id = int(cluster_label_xy[nearest_y, nearest_x])

                classification_lookup[(t, object_label)] = {
                    "cluster_id": cluster_id,
                    "inside_cluster": False,
                    "near_cluster_boundary": True,
                    "overlap_cluster_mask": False,
                    "cluster_overlap_pixels": overlap_pixels,
                    "cluster_overlap_fraction": float(overlap_fraction),
                    "distance_to_cluster_boundary_px": min_distance,
                    "cluster_region_class": "cluster_boundary",
                }
            else:
                classification_lookup[(t, object_label)] = {
                    "cluster_id": 0,
                    "inside_cluster": False,
                    "near_cluster_boundary": False,
                    "overlap_cluster_mask": False,
                    "cluster_overlap_pixels": overlap_pixels,
                    "cluster_overlap_fraction": float(overlap_fraction),
                    "distance_to_cluster_boundary_px": min_distance,
                    "cluster_region_class": "outside_cluster",
                }

    # Apply lookup back to object_df.
    output_cols = {
        "cluster_id": [],
        "inside_cluster": [],
        "near_cluster_boundary": [],
        "overlap_cluster_mask": [],
        "cluster_overlap_pixels": [],
        "cluster_overlap_fraction": [],
        "distance_to_cluster_boundary_px": [],
        "cluster_region_class": [],
    }

    missing_lookup_count = 0

    for _, row in object_df.iterrows():
        t = _normalise_time_index(row["time"], n_timepoints, time_offset)
        object_label = int(row[label_col])

        info = classification_lookup.get((t, object_label))
        if info is None:
            missing_lookup_count += 1
            info = _empty_cluster_info()

        for col in output_cols:
            output_cols[col].append(info[col])

    for col, values in output_cols.items():
        object_df[col] = values

    if missing_lookup_count > 0:
        raise RuntimeError(
            f"{missing_lookup_count} objects were not found in the label image lookup. "
            "Stopping because treating unmatched objects as outside_cluster could allow "
            "cluster-overlapping cells into outside_boundary/outside_only. Check the object "
            "label column and time indexing."
        )

    print("\nCluster region class counts:")
    print(object_df["cluster_region_class"].value_counts())

    print("\nObjects overlapping cluster mask:")
    print(int(object_df["overlap_cluster_mask"].sum()))

    # Strict sanity check: no outside/boundary object should overlap the cluster mask.
    bad = object_df[
        object_df["cluster_region_class"].isin(["outside_cluster", "cluster_boundary"])
        & object_df["overlap_cluster_mask"]
    ]
    if len(bad) > 0:
        raise RuntimeError(
            "Overlap classification failed: some outside/boundary objects still overlap the cluster mask."
        )

    return object_df


# Backwards-compatible alias, but use the explicit name in main() to avoid confusion.
classify_objects_by_cluster = classify_objects_by_cluster_overlap

def extract_cluster_level_features(object_df_with_flags, cluster_masks_tyx):
    """Per-timepoint aggregate cluster features."""

    rows = []

    T = cluster_masks_tyx.shape[0]

    for t in range(T):
        mask_xy = cluster_masks_tyx[t]
        cluster_area_xy = int((mask_xy > 0).sum())
        n_clusters = int(mask_xy.max())

        objects_t = object_df_with_flags[object_df_with_flags["time"] == t].copy()
        inside = objects_t[objects_t["inside_cluster"]].copy()
        boundary = objects_t[objects_t["near_cluster_boundary"]].copy()
        outside = objects_t[objects_t["cluster_region_class"] == "outside_cluster"].copy()

        row = {
            "cell_type": CELL_TYPE,
            "time": t,

            # cluster mask geometry
            "cluster_area_xy_px": cluster_area_xy,
            "n_detected_clusters": n_clusters,

            # object counts
            "n_objects_total": int(len(objects_t)),
            "n_objects_inside_cluster": int(len(inside)),
            "n_objects_boundary": int(len(boundary)),
            "n_objects_outside_cluster": int(len(outside)),
        }

        if len(inside) > 0:
            row.update(
                {
                    "cluster_total_volume_voxels": float(inside["volume_voxels"].sum()),
                    "cluster_mean_object_volume": float(inside["volume_voxels"].mean()),
                    "cluster_max_object_volume": float(inside["volume_voxels"].max()),
                    "cluster_mean_intensity": float(inside["mean_intensity"].mean())
                    if "mean_intensity" in inside.columns
                    else np.nan,
                    "cluster_max_intensity": float(inside["max_intensity"].max())
                    if "max_intensity" in inside.columns
                    else np.nan,
                    "cluster_mean_z": float(inside["z"].mean()),
                    "cluster_mean_y": float(inside["y"].mean()),
                    "cluster_mean_x": float(inside["x"].mean()),
                    "cluster_z_min": float(inside["z"].min()),
                    "cluster_z_max": float(inside["z"].max()),
                    "cluster_z_range": float(inside["z"].max() - inside["z"].min()),
                }
            )
        else:
            row.update(
                {
                    "cluster_total_volume_voxels": 0.0,
                    "cluster_mean_object_volume": np.nan,
                    "cluster_max_object_volume": np.nan,
                    "cluster_mean_intensity": np.nan,
                    "cluster_max_intensity": np.nan,
                    "cluster_mean_z": np.nan,
                    "cluster_mean_y": np.nan,
                    "cluster_mean_x": np.nan,
                    "cluster_z_min": np.nan,
                    "cluster_z_max": np.nan,
                    "cluster_z_range": np.nan,
                }
            )

        rows.append(row)

    cluster_df = pd.DataFrame(rows)

    # Add simple temporal derivatives
    for col in [
        "cluster_area_xy_px",
        "cluster_total_volume_voxels",
        "n_objects_inside_cluster",
        "n_objects_boundary",
        "n_objects_outside_cluster",
    ]:
        cluster_df[f"{col}_delta"] = cluster_df[col].diff()

    return cluster_df



# Main


def main():
    print("=" * 70)
    print("MACROPHAGE CLUSTER DETECTION / CENSORING")
    print("=" * 70)

    if CELL_TYPE != "macrophage":
        print("[WARNING] This script is intended mainly for macrophages.")
        return

    print(f"Labels:          {LABELS_3D_TZYX_TIF}")
    print(f"Object features: {OBJECT_FEATURES_CSV}")

    labels_tzyx = tiff.imread(LABELS_3D_TZYX_TIF)
    object_df = pd.read_csv(OBJECT_FEATURES_CSV)

    print("Labels shape:", labels_tzyx.shape)
    print("Object features:", object_df.shape)

    if labels_tzyx.ndim != 4:
        raise ValueError(f"Expected labels T,Z,Y,X. Got {labels_tzyx.shape}")

    T, Z, Y, X = labels_tzyx.shape

    cluster_masks = np.zeros((T, Y, X), dtype=np.uint16)
    cluster_tracking_exclusion_masks = np.zeros((T, Y, X), dtype=np.uint16)

    if USE_FISH_BODY_MASK:
        print("[INFO] Creating fish/body mask...")
        fish_masks_tyx = make_fish_body_mask_tyx()
        print("Fish/body mask shape:", fish_masks_tyx.shape)
    else:
        fish_masks_tyx = None


    # 1) Detect raw cluster masks per timepoint


    for t in range(T):
        print(f"[INFO] Detecting cluster T={t+1}/{T}")

        fish_mask_xy = None
        if fish_masks_tyx is not None:
            fish_mask_xy = fish_masks_tyx[t]

        cluster_masks[t] = detect_cluster_mask_for_timepoint(
            labels_tzyx[t],
            fish_mask_xy=fish_mask_xy,
        )


    # 2) Persist cluster mask through missing timepoints


    cluster_masks = persist_masks_forward_tyx(
        cluster_masks,
        name="dense cluster mask",
        min_valid_area=MIN_VALID_CLUSTER_AREA_FOR_PERSISTENCE,
    )


    # 3) Build tracking exclusion mask from the persisted cluster mask


    for t in range(T):
        fish_mask_xy = None
        if fish_masks_tyx is not None:
            fish_mask_xy = fish_masks_tyx[t]

        cluster_tracking_exclusion_masks[t] = make_tracking_exclusion_mask(
            cluster_masks[t],
            fish_mask_xy=fish_mask_xy,
        )

    # Extra safety: also persist the final tracking exclusion mask.
    # This protects against rare fish/body-mask flicker.
    cluster_tracking_exclusion_masks = persist_masks_forward_tyx(
        cluster_tracking_exclusion_masks,
        name="tracking exclusion mask",
        min_valid_area=MIN_VALID_CLUSTER_AREA_FOR_PERSISTENCE,
    )

    print("[INFO] Classifying objects by cluster region using object-mask overlap...")
    object_with_flags = classify_objects_by_cluster_overlap(
        object_df,
        labels_tzyx,
        cluster_tracking_exclusion_masks,
    )

    print("[INFO] Extracting cluster-level features...")
    # cluster_features = extract_cluster_level_features(object_with_flags, cluster_masks)
    cluster_features = extract_cluster_level_features(object_with_flags, cluster_tracking_exclusion_masks)

    outside_boundary_objects = object_with_flags[
        object_with_flags["cluster_region_class"].isin(
            ["outside_cluster", "cluster_boundary"]
        )
    ].copy()

    outside_only_objects = object_with_flags[
        object_with_flags["cluster_region_class"] == "outside_cluster"
    ].copy()

    bad_overlap = outside_boundary_objects[
        (outside_boundary_objects["overlap_cluster_mask"] == True)
        | (outside_boundary_objects["cluster_overlap_pixels"] > 0)
    ]

    if len(bad_overlap) > 0:
        raise RuntimeError(
            f"ERROR: {len(bad_overlap)} outside_boundary objects still overlap "
            "the tracking exclusion mask."
        )

    print("[CHECK PASSED] outside_boundary contains no objects overlapping the tracking exclusion mask.")

    print()
    print("Object counts:")
    print(object_with_flags["cluster_region_class"].value_counts())

    print()
    print("Tracking input table sizes:")
    print(f"All objects:              {len(object_with_flags)}")
    print(f"Outside + boundary:       {len(outside_boundary_objects)}")
    print(f"Outside only:             {len(outside_only_objects)}")
    print(f"Inside cluster excluded:  {len(object_with_flags) - len(outside_boundary_objects)}")

    print()
    print("Saving outputs...")

    tiff.imwrite(
        CLUSTER_MASK_TYX_TIF,
        cluster_masks.astype(np.uint16),
        imagej=True,
        metadata={"axes": "TYX"},
    )

    tiff.imwrite(
        CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF,
        cluster_tracking_exclusion_masks.astype(np.uint16),
        imagej=True,
        metadata={"axes": "TYX"},
    )
    print("[SAVED]", CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF)
    object_with_flags.to_csv(OBJECT_FEATURES_WITH_CLUSTER_CSV, index=False)

    outside_boundary_objects.to_csv(
        OBJECT_FEATURES_OUTSIDE_BOUNDARY_CSV,
        index=False,
    )

    outside_only_objects.to_csv(
        OBJECT_FEATURES_OUTSIDE_ONLY_CSV,
        index=False,
    )

    cluster_features.to_csv(CLUSTER_FEATURES_CSV, index=False)

    print("[SAVED]", CLUSTER_MASK_TYX_TIF)
    print("[SAVED]", OBJECT_FEATURES_WITH_CLUSTER_CSV)
    print("[SAVED]", OBJECT_FEATURES_OUTSIDE_BOUNDARY_CSV)
    print("[SAVED]", OBJECT_FEATURES_OUTSIDE_ONLY_CSV)
    print("[SAVED]", CLUSTER_FEATURES_CSV)

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()
