"""Common utilities for 2D/3D tracking-by-detection of segmented cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackingParams:
    """Parameters shared by the tracking algorithms."""

    # Physical/voxel scaling. Use z_um / xy_um if known.
    z_scale: float = 3.0

    # Maximum allowed 3D scaled centroid displacement per saved time step.
    max_link_distance: float = 35.0
    max_z_step: Optional[float] = None  # in raw z-slices; None disables this gate
    max_gap: int = 2

    # Cost weights. Keep volume/intensity weights modest because cells deform.
    volume_weight: float = 0.10
    intensity_weight: float = 0.03

    # Keyhole-specific parameters.
    keyhole_forward_distance: float = 45.0
    keyhole_back_radius: float = 12.0
    keyhole_angle_degrees: float = 80.0
    keyhole_min_speed: float = 1e-3

    # Output filtering/QC only. Does not affect linking.
    min_track_length: int = 3

    # Upper cost cap used after Hungarian assignment.
    assignment_cost_cutoff: Optional[float] = None


def default_params(cell_type: str = "unknown") -> TrackingParams:
    """Return neutral defaults only."""
    return TrackingParams()


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    return None


def canonicalize_features(features):
    """Standardise column names from the 3D reconstruction output so all trackers."""

    features = features.copy()

    aliases = {
        # time
        "t": "time",
        "time_index": "time",
        "frame": "time",

        # object label / ID
        "object_id": "object_label",
        "object_id_3d": "object_label",
        "label": "object_label",
        "label_id": "object_label",
        "mask_label": "object_label",
        "track_object_id": "object_label",

        # centroids
        "z": "centroid_z",
        "y": "centroid_y",
        "x": "centroid_x",
        "cz": "centroid_z",
        "cy": "centroid_y",
        "cx": "centroid_x",

        # volume / size
        "volume": "volume_voxels",
        "area": "volume_voxels",
        "size": "volume_voxels",
        "n_voxels": "volume_voxels",

        # intensity
        "intensity_mean": "mean_intensity",
        "mean_raw_intensity": "mean_intensity",
        "intensity_max": "max_intensity",
        "max_raw_intensity": "max_intensity",
    }

    rename_map = {}

    for col in features.columns:
        if col in aliases:
            canonical = aliases[col]

            # Avoid overwriting if canonical column already exists.
            if canonical not in features.columns:
                rename_map[col] = canonical

    features = features.rename(columns=rename_map)

    required = [
        "time",
        "object_label",
        "centroid_z",
        "centroid_y",
        "centroid_x",
        "volume_voxels",
    ]

    missing = [col for col in required if col not in features.columns]

    if missing:
        raise ValueError(
            "Missing required object feature columns after alias matching: "
            f"{missing}. Available columns: {list(features.columns)}"
        )

    # Ensure correct numeric types
    features["time"] = features["time"].astype(int)
    features["object_label"] = features["object_label"].astype(int)

    for col in ["centroid_z", "centroid_y", "centroid_x", "volume_voxels"]:
        features[col] = features[col].astype(float)

    if "mean_intensity" not in features.columns:
        features["mean_intensity"] = 0.0

    if "max_intensity" not in features.columns:
        features["max_intensity"] = 0.0

    features["mean_intensity"] = features["mean_intensity"].astype(float)
    features["max_intensity"] = features["max_intensity"].astype(float)

    # Sort for stable tracking
    features = features.sort_values(
        ["time", "object_label"]
    ).reset_index(drop=True)

    return features


def scaled_xyz_from_row(row: pd.Series, params: TrackingParams) -> np.ndarray:
    """Return centroid as scaled z,y,x."""
    return np.array(
        [row["centroid_z"] * params.z_scale, row["centroid_y"], row["centroid_x"]],
        dtype=float,
    )


def scaled_xyz_from_active_row(row: pd.Series, params: TrackingParams, prefix: str = "last") -> np.ndarray:
    """Return active-track position as scaled z,y,x."""
    return np.array(
        [row[f"{prefix}_z"] * params.z_scale, row[f"{prefix}_y"], row[f"{prefix}_x"]],
        dtype=float,
    )


def safe_relative_difference(a: float, b: float, eps: float = 1e-6) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return float(abs(a - b) / max(abs(a), abs(b), eps))


def finite_assignments(cost: np.ndarray, cutoff: Optional[float] = None) -> List[Tuple[int, int]]:
    """Solve Hungarian assignment and return finite accepted pairs."""
    if cost.size == 0:
        return []

    finite = np.isfinite(cost)
    if not finite.any():
        return []

    # scipy cannot solve matrices containing inf only in some rows/cols robustly,
    # so replace inf with a large impossible value.
    max_finite = float(np.nanmax(cost[finite]))
    impossible = max(max_finite * 1000.0, 1e9)
    safe_cost = np.where(finite, cost, impossible)

    row_idx, col_idx = linear_sum_assignment(safe_cost)
    pairs: List[Tuple[int, int]] = []
    for r, c in zip(row_idx, col_idx):
        val = cost[r, c]
        if not np.isfinite(val):
            continue
        if cutoff is not None and val > cutoff:
            continue
        pairs.append((int(r), int(c)))
    return pairs


def _active_tracks_to_df(active_tracks: List[dict]) -> pd.DataFrame:
    rows = []
    for tr in active_tracks:
        last = tr["last_row"]
        prev = tr.get("prev_row")
        row = {
            "track_id": tr["track_id"],
            "gap": tr.get("gap", 0),
            "last_time": int(last["time"]),
            "last_object_label": int(last["object_label"]),
            "last_z": float(last["centroid_z"]),
            "last_y": float(last["centroid_y"]),
            "last_x": float(last["centroid_x"]),
            "last_volume_voxels": float(last.get("volume_voxels", 1.0)),
            "last_mean_intensity": float(last.get("mean_intensity", np.nan)),
        }
        if prev is None:
            row.update({"prev_z": np.nan, "prev_y": np.nan, "prev_x": np.nan, "prev_time": np.nan})
        else:
            row.update(
                {
                    "prev_z": float(prev["centroid_z"]),
                    "prev_y": float(prev["centroid_y"]),
                    "prev_x": float(prev["centroid_x"]),
                    "prev_time": int(prev["time"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _output_row(row: pd.Series, track_id: int, method_name: str) -> dict:
    out = row.to_dict()
    out["track_id"] = int(track_id)
    out["tracking_method"] = method_name
    return out


def track_by_cost_builder(
    features: pd.DataFrame,
    params: TrackingParams,
    cost_builder: Callable[[pd.DataFrame, pd.DataFrame, TrackingParams], np.ndarray],
    method_name: str,
) -> pd.DataFrame:
    """Generic online tracker with gap closing."""
    features = canonicalize_features(features)
    times = sorted(features["time"].unique())

    next_track_id = 1
    active_tracks: List[dict] = []
    output_rows: List[dict] = []

    for t in times:
        detections = features[features["time"] == t].reset_index(drop=True)

        if not active_tracks:
            for _, det in detections.iterrows():
                track_id = next_track_id
                next_track_id += 1
                active_tracks.append({"track_id": track_id, "last_row": det.copy(), "prev_row": None, "gap": 0})
                output_rows.append(_output_row(det, track_id, method_name))
            continue

        active_df = _active_tracks_to_df(active_tracks)
        cost = cost_builder(active_df, detections, params)
        pairs = finite_assignments(cost, cutoff=params.assignment_cost_cutoff)

        assigned_active = {r for r, _ in pairs}
        assigned_det = {c for _, c in pairs}

        # Update matched active tracks.
        for active_i, det_j in pairs:
            det = detections.iloc[det_j].copy()
            tr = active_tracks[active_i]
            output_rows.append(_output_row(det, tr["track_id"], method_name))
            tr["prev_row"] = tr["last_row"]
            tr["last_row"] = det
            tr["gap"] = 0

        # Age unmatched tracks.
        new_active: List[dict] = []
        for idx, tr in enumerate(active_tracks):
            if idx not in assigned_active:
                tr["gap"] = int(tr.get("gap", 0)) + 1
            if tr["gap"] <= params.max_gap:
                new_active.append(tr)
        active_tracks = new_active

        # Start new tracks for unmatched detections.
        for det_j, det in detections.iterrows():
            if det_j in assigned_det:
                continue
            track_id = next_track_id
            next_track_id += 1
            active_tracks.append({"track_id": track_id, "last_row": det.copy(), "prev_row": None, "gap": 0})
            output_rows.append(_output_row(det, track_id, method_name))

    out = pd.DataFrame(output_rows)
    if out.empty:
        return out

    return out.sort_values(["track_id", "time"]).reset_index(drop=True)


def track_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """Small QC table per track."""
    if tracks.empty:
        return pd.DataFrame()
    rows = []
    for tid, g in tracks.groupby("track_id"):
        g = g.sort_values("time")
        duration = int(g["time"].max() - g["time"].min() + 1)
        rows.append(
            {
                "track_id": tid,
                "start_time": int(g["time"].min()),
                "end_time": int(g["time"].max()),
                "n_detections": int(len(g)),
                "duration": duration,
                "completeness": float(len(g) / max(duration, 1)),
                "mean_volume_voxels": float(g["volume_voxels"].mean()) if "volume_voxels" in g else np.nan,
                "z_range": float(g["centroid_z"].max() - g["centroid_z"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_detections", "duration"], ascending=False)
