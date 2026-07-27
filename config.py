from pathlib import Path
import os



# Paths


# Example override: BATCH_PROJECT_DIR=/path/to/output_directory

# CELL_TYPE = "musc"  # "musc" or "macrophage"
# CELL_TYPE="macrophage"  # "musc" or "macrophage"


# Batch overrides


REPOSITORY_ROOT = Path(__file__).resolve().parent

# Override these defaults when the raw data are stored elsewhere:
# BATCH_PROJECT_DIR=/path/to/output BATCH_CZI_PATH=/path/to/input.czi
DEFAULT_PROJECT_DIR = REPOSITORY_ROOT
DEFAULT_CZI_PATH = REPOSITORY_ROOT / "data" / "input.czi"

PROJECT_DIR = Path(os.environ.get("BATCH_PROJECT_DIR", DEFAULT_PROJECT_DIR)).expanduser().resolve()
CZI_PATH = Path(os.environ.get("BATCH_CZI_PATH", DEFAULT_CZI_PATH)).expanduser().resolve()

CELL_TYPE = os.environ.get("BATCH_CELL_TYPE", "macrophage").lower().strip()
# CELL_TYPE = os.environ.get("BATCH_CELL_TYPE", "musc").lower().strip()

if CELL_TYPE not in {"musc", "macrophage"}:
    raise ValueError("CELL_TYPE must be either 'musc' or 'macrophage'.")


# Input and output paths

# Converted raw stack
RAW_TCZYX_TIF = PROJECT_DIR / "raw_TCZYX_drift_corrected.tif"
RAW_TCZYX_UNCORRECTED_TIF = PROJECT_DIR / "raw_TCZYX_uncorrected_subset.tif"

# Keep this alias so older scripts do not break immediately.
# New scripts should use RAW_TCZYX_TIF.
RAW_TZYX_TIF = RAW_TCZYX_TIF
RAW_TZYX_UNCORRECTED_TIF = RAW_TCZYX_UNCORRECTED_TIF

DRIFT_QC_DIR = PROJECT_DIR / "drift_qc"

CELLPOSE_MASKS_TZYX_TIF = PROJECT_DIR / f"{CELL_TYPE}_cellpose_masks_TZYX.tif"
LABELS_3D_TZYX_TIF = PROJECT_DIR / f"{CELL_TYPE}_3d_labels_TZYX.tif"

RECONSTRUCTION_CSV = PROJECT_DIR / f"{CELL_TYPE}_3d_reconstruction_info.csv"
OBJECT_FEATURES_CSV = PROJECT_DIR / f"{CELL_TYPE}_3d_object_features.csv"

TRACKS_CSV = PROJECT_DIR / f"{CELL_TYPE}_tracks.csv"
FILTERED_TRACKS_CSV = PROJECT_DIR / f"{CELL_TYPE}_tracks_good_filtered.csv"

TRACKING_OUTPUT_DIR = PROJECT_DIR / f"{CELL_TYPE}_tracking_outputs"
TRACKING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# TRACKS_CSV = PROJECT_DIR / "musc_tracks.csv"



# Czi reading settings


SCENE_INDEX = 0
# CHANNEL_INDEX = 0   # green / MUSC channel

USE_EXPLICIT_TIMEPOINTS = False

EXPLICIT_TIMEPOINTS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 207]

TIME_START = int(os.environ.get("BATCH_TIME_START", 0))
TIME_END_ENV = os.environ.get("BATCH_TIME_END", "120")
TIME_END = None if TIME_END_ENV.lower() == "none" else int(TIME_END_ENV)
TIME_STRIDE = 1

# Use all Z slices
Z_START = 0
Z_END = None




# Cellpose settings



# Cell-type-specific channels and models


MUSC_CHANNEL_INDEX = 0          # green / MUSC channel
MACROPHAGE_CHANNEL_INDEX = 1    # usually red / macrophage channel; change if your CZI differs

# Model weights are not bundled; set these variables or place models under models/.
MUSC_CELLPOSE_MODEL = Path(os.environ.get(
    "MUSC_CELLPOSE_MODEL",
    REPOSITORY_ROOT / "models" / "musc_cellpose_model",
)).expanduser().resolve()

MACROPHAGE_CELLPOSE_MODEL = Path(os.environ.get(
    "MACROPHAGE_CELLPOSE_MODEL",
    REPOSITORY_ROOT / "models" / "macrophage_cellpose_model",
)).expanduser().resolve()

if CELL_TYPE == "musc":
    CHANNEL_INDEX = MUSC_CHANNEL_INDEX
    CUSTOM_CELLPOSE_MODEL = MUSC_CELLPOSE_MODEL

elif CELL_TYPE == "macrophage":
    CHANNEL_INDEX = MACROPHAGE_CHANNEL_INDEX
    CUSTOM_CELLPOSE_MODEL = MACROPHAGE_CELLPOSE_MODEL

else:
    raise ValueError("CELL_TYPE must be 'musc' or 'macrophage'.")

USE_GPU = True
CELLPOSE_DIAMETER = None
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
NORMALIZE = True
CELLPOSE_BATCH_SIZE = 8
RESAMPLE = True



# 3d reconstruction settings


MODE = "loose"   # "loose" or "strict"

if CELL_TYPE == "musc":

    if MODE == "loose":
        MAX_Z_SPAN = 18
        MAX_Z_GAP = 3
        MIN_OVERLAP_FRACTION = 0.15
        MIN_IOU = 0.03
        MAX_CENTROID_XY_DISTANCE = 35.0

    elif MODE == "strict":
        MAX_Z_SPAN = 12
        MAX_Z_GAP = 2
        MIN_OVERLAP_FRACTION = 0.20
        MIN_IOU = 0.04
        MAX_CENTROID_XY_DISTANCE = 25.0

    MIN_2D_MASK_AREA = 20
    MIN_3D_VOXELS = 30
    MIN_DETECTED_Z_SLICES = 1

elif CELL_TYPE == "macrophage":

    if MODE == "loose":
        MAX_Z_SPAN = 10
        MAX_Z_GAP = 1
        MIN_OVERLAP_FRACTION = 0.03
        MIN_IOU = 0.001
        MAX_CENTROID_XY_DISTANCE = 21.0

    elif MODE == "strict":
        MAX_Z_SPAN = 8
        MAX_Z_GAP = 1
        MIN_OVERLAP_FRACTION = 0.06
        MIN_IOU = 0.005
        MAX_CENTROID_XY_DISTANCE = 16.0

    MIN_2D_MASK_AREA = 5
    MIN_3D_VOXELS = 5
    MIN_DETECTED_Z_SLICES = 1

else:
    raise ValueError("CELL_TYPE must be 'musc' or 'macrophage'.")


# Tracking settings


DEFAULT_TRACKING_METHOD = "nearest" if CELL_TYPE == "musc" else "lap"

TRACKING_METHOD = os.environ.get(
    "TRACKING_METHOD",
    DEFAULT_TRACKING_METHOD,
).lower().strip()

if TRACKING_METHOD not in {
    "nearest",
    "lap",
    "keyhole",
}:
    raise ValueError(f"Invalid TRACKING_METHOD: {TRACKING_METHOD}")

TRACKING_METHODS = [TRACKING_METHOD]

MAX_TIME_GAP = 2

# Converts Z movement into XY-equivalent distance.
# Use physical voxel spacing if known.
# Z_DISTANCE_WEIGHT = 4.0

if CELL_TYPE == "musc":
    ASSIGNMENT_COST_CUTOFF = 32.0
    MAX_TRACK_XY_DISTANCE = 28.0
    MAX_TRACK_Z_DISTANCE = 4.0
    MAX_VOLUME_RATIO = 4.0

    VOLUME_COST_WEIGHT = 10.0
    INTENSITY_COST_WEIGHT = 2.0

    KEYHOLE_FORWARD_DISTANCE = 25.0
    KEYHOLE_BACK_RADIUS = 6.0
    KEYHOLE_ANGLE_DEGREES = 60.0

    MIN_TRACK_LENGTH = 5

elif CELL_TYPE == "macrophage":
    ASSIGNMENT_COST_CUTOFF = 32.0
    MAX_TRACK_XY_DISTANCE = 35.0
    MAX_TRACK_Z_DISTANCE = 5.0
    MAX_VOLUME_RATIO = 5.0

    VOLUME_COST_WEIGHT = 0.5
    INTENSITY_COST_WEIGHT = 0.2

    KEYHOLE_FORWARD_DISTANCE = 45.0
    KEYHOLE_BACK_RADIUS = 25.0
    KEYHOLE_ANGLE_DEGREES = 120.0

    MIN_TRACK_LENGTH = 5

else:
    raise ValueError("CELL_TYPE must be 'musc' or 'macrophage'.")


# Voxel size / 3d distance scaling


XY_PIXEL_SIZE_UM = float(os.environ["BATCH_XY_PIXEL_SIZE_UM"]) if "BATCH_XY_PIXEL_SIZE_UM" in os.environ else 0.7533114346590908
Z_STEP_SIZE_UM = float(os.environ["BATCH_Z_STEP_SIZE_UM"]) if "BATCH_Z_STEP_SIZE_UM" in os.environ else 1.0

if XY_PIXEL_SIZE_UM is not None and Z_STEP_SIZE_UM is not None:
    Z_DISTANCE_WEIGHT = Z_STEP_SIZE_UM / XY_PIXEL_SIZE_UM
else:
    raise ValueError("XY_PIXEL_SIZE_UM and Z_STEP_SIZE_UM must be set to real values.")





# Drift correction settings


DRIFT_UPSAMPLE_FACTOR = 5
DRIFT_CROP_FRACTION = 0.75

# For consecutive frames, use 10–25.
# Your old code comment said 10–25 for TIMEPOINT_STEP=1.
DRIFT_MAX_SHIFT_PER_STEP = 20

SAVE_DRIFT_QC = True
SAVE_UNCORRECTED_SUBSET_TIF = True

# Keep original intensity scale for Cellpose.
# Set True only if you specifically want old-style per-stack uint16 normalisation.
NORMALIZE_DRIFT_OUTPUT_TO_UINT16 = False


# Track qc settings


# QC_TRACKING_METHOD = "keyhole"
# QC_TRACKING_METHOD = "nearest"
# QC_TRACKING_METHOD = "lap"
QC_TRACKING_METHOD = TRACKING_METHOD
# Options: "nearest", "lap", "keyhole"

MIN_GOOD_TRACK_LENGTH = MIN_TRACK_LENGTH

LARGE_JUMP_THRESHOLD = 80.0


# Track feature extraction settings


# FEATURE_TRACKING_METHOD = "keyhole"
# FEATURE_TRACKING_METHOD = "nearest" 
# FEATURE_TRACKING_METHOD = "lap"
FEATURE_TRACKING_METHOD = TRACKING_METHOD
# Options: "nearest", "lap", "keyhole"

STATIC_DISPLACEMENT_THRESHOLD = 5.0
LONG_TRACK_LENGTH = 10
VERY_LONG_TRACK_LENGTH = 20
SUSPICIOUS_JUMP_THRESHOLD = 50.0

# TRACK_FEATURES_CSV = (
# Tracking_output_dir
#     / f"{CELL_TYPE}_track_quality_features_{FEATURE_TRACKING_METHOD}.csv"
# )


# Good track filter settings


if CELL_TYPE == "musc":
    MIN_GOOD_FILTER_TRACK_LENGTH = 10
    MAX_GOOD_FILTER_STEP_DISTANCE_3D = 50.0
    MAX_GOOD_FILTER_Z_RANGE = 12.0
    REMOVE_STATIC_TRACKS = True

elif CELL_TYPE == "macrophage":
    MIN_GOOD_FILTER_TRACK_LENGTH = 5
    MAX_GOOD_FILTER_STEP_DISTANCE_3D = 60.0
    MAX_GOOD_FILTER_Z_RANGE = 15.0
    REMOVE_STATIC_TRACKS = False


# Napari view settings


# VIEW_TRACKING_METHOD = "keyhole"
# VIEW_TRACKING_METHOD = "nearest"
# VIEW_TRACKING_METHOD = "lap"
VIEW_TRACKING_METHOD = TRACKING_METHOD
# Options: "nearest", "lap", "keyhole"


# Split / merge warning settings


# These do not change tracking.
# They only flag tracks whose morphology/intensity changes look suspicious.

ENABLE_SPLIT_MERGE_FLAGS = True

# Sudden volume change between consecutive detections.
# Example: 4.0 means volume suddenly becomes >4x or <1/4x.
MAX_VOLUME_FOLD_CHANGE_FOR_SPLIT_MERGE = 4.0

# Sudden mean-intensity change between consecutive detections.
MAX_INTENSITY_FOLD_CHANGE_FOR_SPLIT_MERGE = 4.0

# High variation of volume over the whole track.
MAX_VOLUME_CV_FOR_SPLIT_MERGE = 1.0

# Step spike relative to that track's normal step size.
MAX_STEP_TO_MEDIAN_RATIO_FOR_SPLIT_MERGE = 8.0

# Do not flag tiny numerical changes as step spikes.
MIN_ABSOLUTE_STEP_FOR_SPLIT_MERGE = 10.0

# For final clean tracks, remove tracks with split/merge warnings.
REMOVE_SPLIT_MERGE_WARNINGS = True


# Region-aware tracking feature selection
# Only affects macrophages.
# MUSC remains unchanged.


MACROPHAGE_REGION_MODE = os.environ.get(
    "MACROPHAGE_REGION_MODE",
    "outside_boundary",
).lower().strip()

if MACROPHAGE_REGION_MODE not in {
    "all",
    "outside_boundary",
    "outside_only",
}:
    raise ValueError(
        "MACROPHAGE_REGION_MODE must be one of: "
        "'all', 'outside_boundary', 'outside_only'."
    )


# Cluster-detection outputs
CLUSTER_MASK_TYX_TIF = PROJECT_DIR / f"{CELL_TYPE}_cluster_mask_TYX.tif"

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


# Default: use normal object features
TRACKING_FEATURES_CSV = OBJECT_FEATURES_CSV


# Output directory and tracking input
if CELL_TYPE == "macrophage":

    if MACROPHAGE_REGION_MODE == "all":
        TRACKING_FEATURES_CSV = OBJECT_FEATURES_CSV

    elif MACROPHAGE_REGION_MODE == "outside_boundary":
        TRACKING_FEATURES_CSV = OBJECT_FEATURES_OUTSIDE_BOUNDARY_CSV

    elif MACROPHAGE_REGION_MODE == "outside_only":
        TRACKING_FEATURES_CSV = OBJECT_FEATURES_OUTSIDE_ONLY_CSV

    TRACKING_OUTPUT_DIR = (
        PROJECT_DIR / f"{CELL_TYPE}_tracking_outputs_{MACROPHAGE_REGION_MODE}"
    )

else:
    # IMPORTANT: MUSC should stay exactly as before
    MACROPHAGE_REGION_MODE = "not_applicable"
    TRACKING_FEATURES_CSV = OBJECT_FEATURES_CSV
    TRACKING_OUTPUT_DIR = PROJECT_DIR / f"{CELL_TYPE}_tracking_outputs"


TRACKING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKS_CSV = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks.csv"
FILTERED_TRACKS_CSV = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_good_filtered.csv"

TRACK_FEATURES_CSV = (
    TRACKING_OUTPUT_DIR
    / f"{CELL_TYPE}_track_quality_features_{FEATURE_TRACKING_METHOD}.csv"
)

# Macrophage cluster-crossing track split
# Only used for macrophage region-aware tracking.
# MUSC is unaffected.


CLUSTER_TRACKING_EXCLUSION_MASK_TYX_TIF = (
    PROJECT_DIR / f"{CELL_TYPE}_cluster_tracking_exclusion_mask_TYX.tif"
)

SPLIT_TRACKS_CROSSING_CLUSTER_MASK = True

# Number of interpolated XY points checked along each track link.
# Higher = stricter but slightly slower.
CLUSTER_CROSSING_LINE_SAMPLES = 80