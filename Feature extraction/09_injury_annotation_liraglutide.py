#!/usr/bin/env python3
"""Launch the unchanged injury-annotation logic with Liraglutide paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

spec = importlib.util.spec_from_file_location(
    "injury_annotation_shared",
    HERE / "09_injury_annotation.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load 09_injury_annotation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Configuration only. Fish selection still comes from these completed
# constrained tables, exactly as in the original annotation logic.
module.BLOCKS_ROOT = PROJECT_ROOT / "overnight_batch_outputs" / "Liraglutide"
module.OUTPUT_ROOT = HERE / "manual_injury_annotations_liraglutide"
module.MUSC_MODEL_A_TABLE = (
    HERE / "constrained_fish_features_time_corrected_liraglutide"
    / "musc" / "constrained_fish_level_mean_median.csv"
)
module.MACROPHAGE_MODEL_A_TABLE = (
    HERE / "constrained_fish_features_time_corrected_liraglutide"
    / "macrophage_all" / "constrained_fish_level_mean_median.csv"
)
module.EXCLUDED_FISH = set()
module.DEFAULT_XY_UM = 0.7533114346590908

module.main()
