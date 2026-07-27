"""Keyhole-style tracker."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    TrackingParams,
    safe_relative_difference,
    scaled_xyz_from_active_row,
    scaled_xyz_from_row,
    track_by_cost_builder,
)


def build_keyhole_cost(active_df: pd.DataFrame, next_df: pd.DataFrame, params: TrackingParams) -> np.ndarray:
    cost = np.full((len(active_df), len(next_df)), np.inf, dtype=float)
    if len(active_df) == 0 or len(next_df) == 0:
        return cost

    next_pos = np.vstack([scaled_xyz_from_row(row, params) for _, row in next_df.iterrows()])
    angle_limit = np.deg2rad(params.keyhole_angle_degrees)

    for i, (_, tr) in enumerate(active_df.iterrows()):
        last_pos = scaled_xyz_from_active_row(tr, params, prefix="last")
        gap_factor = float(tr.get("gap", 0) + 1)

        has_prev = np.isfinite(tr.get("prev_z", np.nan))
        if has_prev:
            prev_pos = scaled_xyz_from_active_row(tr, params, prefix="prev")
            velocity = last_pos - prev_pos
            speed = float(np.linalg.norm(velocity))
        else:
            velocity = np.zeros(3, dtype=float)
            speed = 0.0

        # If track has no history or is almost static, fall back to a normal sphere.
        use_keyhole = has_prev and speed > params.keyhole_min_speed
        if use_keyhole:
            direction = velocity / speed
            predicted_pos = last_pos + velocity * gap_factor
            forward_limit = params.keyhole_forward_distance * gap_factor
            back_radius = params.keyhole_back_radius
        else:
            direction = None
            predicted_pos = last_pos
            forward_limit = params.max_link_distance * gap_factor
            back_radius = params.max_link_distance * gap_factor

        for j, (_, cand) in enumerate(next_df.iterrows()):
            if params.max_z_step is not None:
                dz = abs(float(cand["centroid_z"]) - float(tr["last_z"]))
                if dz > params.max_z_step * gap_factor:
                    continue

            cand_pos = next_pos[j]
            vec = cand_pos - last_pos
            total_dist = float(np.linalg.norm(vec))

            if not use_keyhole:
                if total_dist > params.max_link_distance * gap_factor:
                    continue
                motion_cost = total_dist
            else:
                forward_distance = float(np.dot(vec, direction))
                perpendicular_vec = vec - forward_distance * direction
                perpendicular_distance = float(np.linalg.norm(perpendicular_vec))

                in_back_sphere = total_dist <= back_radius
                in_forward_cone = (
                    forward_distance >= 0.0
                    and forward_distance <= forward_limit
                    and np.arctan2(perpendicular_distance, forward_distance + 1e-6) <= angle_limit
                )
                if not (in_back_sphere or in_forward_cone):
                    continue
                motion_cost = float(np.linalg.norm(cand_pos - predicted_pos))

            volume_penalty = safe_relative_difference(tr["last_volume_voxels"], cand.get("volume_voxels", 1.0))
            if "mean_intensity" in next_df.columns and np.isfinite(tr.get("last_mean_intensity", np.nan)):
                intensity_penalty = safe_relative_difference(tr["last_mean_intensity"], cand.get("mean_intensity", np.nan))
            else:
                intensity_penalty = 0.0

            cost[i, j] = (
                motion_cost
                + params.volume_weight * volume_penalty * params.max_link_distance
                + params.intensity_weight * intensity_penalty * params.max_link_distance
            )

    return cost


def run_keyhole_tracker(features: pd.DataFrame, params: TrackingParams) -> pd.DataFrame:
    return track_by_cost_builder(features, params, build_keyhole_cost, method_name="keyhole")
