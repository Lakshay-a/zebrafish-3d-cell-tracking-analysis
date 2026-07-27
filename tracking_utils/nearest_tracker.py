"""Nearest-neighbour / distance-only Hungarian tracker."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import TrackingParams, scaled_xyz_from_active_row, scaled_xyz_from_row, track_by_cost_builder


def build_nearest_cost(active_df: pd.DataFrame, next_df: pd.DataFrame, params: TrackingParams) -> np.ndarray:
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
            if dist <= max_dist:
                cost[i, j] = dist

    return cost


def run_nearest_tracker(features: pd.DataFrame, params: TrackingParams) -> pd.DataFrame:
    return track_by_cost_builder(features, params, build_nearest_cost, method_name="nearest")
