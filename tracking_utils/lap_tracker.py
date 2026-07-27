"""LAP/Hungarian tracker using distance + volume/intensity costs."""

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


def build_lap_cost(active_df: pd.DataFrame, next_df: pd.DataFrame, params: TrackingParams) -> np.ndarray:
    cost = np.full((len(active_df), len(next_df)), np.inf, dtype=float)
    if len(active_df) == 0 or len(next_df) == 0:
        return cost

    next_pos = np.vstack([scaled_xyz_from_row(row, params) for _, row in next_df.iterrows()])

    for i, (_, tr) in enumerate(active_df.iterrows()):
        last_pos = scaled_xyz_from_active_row(tr, params, prefix="last")
        gap_factor = float(tr.get("gap", 0) + 1)
        max_dist = params.max_link_distance * gap_factor

        for j, (_, cand) in enumerate(next_df.iterrows()):
            if params.max_z_step is not None:
                dz = abs(float(cand["centroid_z"]) - float(tr["last_z"]))
                if dz > params.max_z_step * gap_factor:
                    continue

            dist = float(np.linalg.norm(next_pos[j] - last_pos))
            if dist > max_dist:
                continue

            volume_penalty = safe_relative_difference(tr["last_volume_voxels"], cand.get("volume_voxels", 1.0))

            if "mean_intensity" in next_df.columns and np.isfinite(tr.get("last_mean_intensity", np.nan)):
                intensity_penalty = safe_relative_difference(tr["last_mean_intensity"], cand.get("mean_intensity", np.nan))
            else:
                intensity_penalty = 0.0

            cost[i, j] = (
                dist
                + params.volume_weight * volume_penalty * params.max_link_distance
                + params.intensity_weight * intensity_penalty * params.max_link_distance
            )

    return cost


# Linear assignment: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
def run_lap_tracker(features: pd.DataFrame, params: TrackingParams) -> pd.DataFrame:
    return track_by_cost_builder(features, params, build_lap_cost, method_name="lap")
