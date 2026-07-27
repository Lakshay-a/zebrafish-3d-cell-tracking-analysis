#!/usr/bin/env python3
"""Extract time-corrected injury-relative features for Liraglutide fish."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
FISH_ROOT = HERE / "constrained_fish_features_time_corrected_liraglutide"

os.environ["INJURY_BLOCKS_ROOT"] = str(
    PROJECT_ROOT / "overnight_batch_outputs" / "Liraglutide"
)
os.environ["INJURY_ANNOTATION_ROOT"] = str(
    HERE / "manual_injury_annotations_liraglutide"
)
os.environ["INJURY_FEATURE_OUTPUT_ROOT"] = str(
    HERE / "manual_injury_feature_outputs_time_corrected_liraglutide"
)
os.environ["INJURY_MUSC_TABLE"] = str(
    FISH_ROOT / "musc" / "constrained_fish_level_mean_median.csv"
)
os.environ["INJURY_MACROPHAGE_ALL_TABLE"] = str(
    FISH_ROOT / "model_b" / "macrophage_all" / "constrained_fish_level_mean_median.csv"
)
os.environ["INJURY_MACROPHAGE_OUTSIDE_TABLE"] = str(
    FISH_ROOT / "model_b" / "macrophage_outside_boundary" / "constrained_fish_level_mean_median.csv"
)
os.environ["INJURY_FRAME_INTERVAL_METADATA"] = str(
    HERE / "Liraglutide_metadata.csv"
)
os.environ["INJURY_EXCLUDED_FISH"] = ""
os.environ["INJURY_DEFAULT_XY_UM"] = "0.7533114346590908"
os.environ["INJURY_DEFAULT_Z_UM"] = "1.0"

runpy.run_path(
    str(HERE / "10_injury_roi_feature_extraction.py"),
    run_name="__main__",
)
