import os
import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from aicspylibczi import CziFile
from scipy.ndimage import shift as ndi_shift
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.registration import phase_cross_correlation
from skimage.io import imsave


# ============================================================
# MACROPHAGE DATASET SETTINGS
# ============================================================

PROJECT_ROOT = Path(os.environ.get("CELLPOSE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATASET_DIR = PROJECT_ROOT / "macrophage_cellpose_dataset_initial"
METADATA_DIR = DATASET_DIR / "metadata"

TRAIN_IMAGES_DIR = DATASET_DIR / "train" / "images"
VAL_IMAGES_DIR = DATASET_DIR / "val" / "images"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"

TRAIN_MASKS_DIR = DATASET_DIR / "train" / "masks"
VAL_MASKS_DIR = DATASET_DIR / "val" / "masks"
TEST_MASKS_DIR = DATASET_DIR / "test" / "masks"

QC_TRAIN_DIR = DATASET_DIR / "qc_png" / "train"
QC_VAL_DIR = DATASET_DIR / "qc_png" / "val"
QC_TEST_DIR = DATASET_DIR / "qc_png" / "test"

DRIFT_QC_DIR = DATASET_DIR / "drift_qc"

# These can be CZI files, CZI folders, or parent folders.
# The script searches recursively where needed.
SOURCE_PATHS = [
    path for path in os.environ.get("CELLPOSE_SOURCE_PATHS", "").split(os.pathsep)
    if path
]

# Channel setup.
# Macrophages/fms-mCherry are usually channel 1 in two-channel files.
# Drift registration is usually more stable on channel 0, like the muSC workflow,
# because macrophages themselves can migrate and may be less reliable for phase correlation.
MACROPHAGE_CHANNEL_INDEX = 1
DRIFT_REGISTRATION_CHANNEL_INDEX = 0
ALLOW_SINGLE_CHANNEL_FALLBACK = True
APPEND_RUN_NAME = "append_new_macrophage_datasets_02"

# CZI/acquisition-block level split.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42

# Final generous extraction:
# 2 start + 2 middle + 2 end timepoints = 6 timepoints per CZI.
TIMEPOINTS_PER_TIME_REGION = 2

# 4 spread-out Z-slices from start/middle/end Z regions = 12 Z-slices per timepoint.
Z_SLICES_PER_Z_REGION = 4

# Normal datasets.
DEFAULT_IGNORE_FIRST_Z = 3
DEFAULT_IGNORE_LAST_Z = 5

# Keep the macrophage extraction consistent for this first dataset:
# ignore the first 3 Z-slices and last 5 Z-slices for every file.
# Add date-specific overrides here later if QC shows a specific acquisition needs more trimming.
SPECIAL_EARLY_Z_IGNORE_BY_DATE = {}
SPECIAL_LAST_Z_IGNORE_BY_DATE = {}

# Exclude bad acquisition blocks if needed after QC.
# Format example: {"20240702": {1, 8}}
EXCLUDE_BLOCKS_BY_DATE = {}

# Percentile normalisation for saved Cellpose images.
P_LOW = 1
P_HIGH = 99.8

# Drift correction: final approach = Option B.
# Estimate consecutive-frame drift across ALL timepoints.
DRIFT_TIMEPOINT_STEP = 1
DRIFT_UPSAMPLE_FACTOR = 5
DRIFT_CROP_FRACTION = 0.75
DRIFT_MAX_SHIFT_PER_STEP = 80

# QC output for final report.
SAVE_DRIFT_QC = True
NUM_FILES_FOR_DRIFT_QC = 3
DRIFT_QC_VIDEO_FPS = 5

# Output safety.
OVERWRITE_EXISTING_DATASET = False


def get_effective_channel_index(reader, requested_index, channel_name):
    """
    Returns a valid channel index for either macrophage extraction or drift registration.

    Default behaviour:
    - macrophage extraction uses MACROPHAGE_CHANNEL_INDEX=1 for fms/mCherry
    - drift registration uses DRIFT_REGISTRATION_CHANNEL_INDEX=0 for stability
    - single-channel files fall back to channel 0
    """
    if reader.size_c > requested_index:
        return requested_index

    if ALLOW_SINGLE_CHANNEL_FALLBACK and reader.size_c == 1:
        print(
            f"[WARNING] Requested {channel_name} channel {requested_index}, "
            "but this file has only one channel. Falling back to channel 0."
        )
        return 0

    raise ValueError(
        f"Requested {channel_name} channel index {requested_index}, "
        f"but this file has only C={reader.size_c}. "
        "Change the channel setting after checking the CZI channels."
    )


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_dirs():
    for d in [
        DATASET_DIR,
        METADATA_DIR,
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TEST_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        VAL_MASKS_DIR,
        TEST_MASKS_DIR,
        QC_TRAIN_DIR,
        QC_VAL_DIR,
        QC_TEST_DIR,
        DRIFT_QC_DIR,
    ]:
        ensure_dir(d)


# ============================================================
# PATH / NAMING HELPERS
# ============================================================

def safe_token(text):
    text = str(text)
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def extract_date_token(path):
    """
    Extracts date-like token from path.

    Examples:
    20240604
    2024-06-04 -> 20240604
    260427
    """
    text = str(path)

    m = re.search(r"(20\d{6})", text)
    if m:
        return m.group(1)

    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", text)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    m = re.search(r"(\d{6})", text)
    if m:
        return m.group(1)

    return "unknown_date"


def extract_acquisition_block(path):
    """
    Extracts acquisition block number from paths like:
    New-01_AcquisitionBlock7.czi
    acquisition block 7
    Aquisition Block 7
    block_07
    """
    text = str(path).lower()

    patterns = [
        r"acquisition\s*block\s*[_\- ]*(\d+)",
        r"aquisition\s*block\s*[_\- ]*(\d+)",
        r"acquisition[_\- ]*block[_\- ]*(\d+)",
        r"aquisition[_\- ]*block[_\- ]*(\d+)",
        r"acquisitionblock\s*(\d+)",
        r"aquisitionblock\s*(\d+)",
        r"block\s*[_\- ]*(\d+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))

    return None


def should_exclude_czi(czi_path):
    date_token = extract_date_token(czi_path)
    block_num = extract_acquisition_block(czi_path)

    excluded_blocks = EXCLUDE_BLOCKS_BY_DATE.get(date_token, set())

    if block_num in excluded_blocks:
        return True, f"Excluded {date_token} acquisition block {block_num}"

    return False, ""


def find_czi_files():
    """
    Searches all SOURCE_PATHS.

    Handles:
    - direct .czi file paths
    - .czi package folders on macOS
    - parent folders containing acquisition-block CZI files
    """
    found = []

    for source in SOURCE_PATHS:
        p = Path(source)

        if p.exists() and p.is_file() and p.suffix.lower() == ".czi":
            found.append(p)
            continue

        if p.exists() and p.is_dir():
            # On macOS, .czi can sometimes appear as a folder/package,
            # and acquisition blocks can be inside it.
            nested = sorted(p.rglob("*.czi"))

            if len(nested) > 0:
                found.extend(nested)
            elif p.suffix.lower() == ".czi":
                found.append(p)

            continue

        if p.suffix.lower() == ".czi" and p.parent.exists():
            print(f"[INFO] Path not found directly. Searching parent: {p.parent}")
            found.extend(sorted(p.parent.rglob("*.czi")))
            continue

        print(f"[WARNING] Source path not found and skipped: {p}")

    unique = []
    seen = set()

    for p in found:
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)

    usable = []
    excluded = []

    for czi in unique:
        exclude, reason = should_exclude_czi(czi)
        if exclude:
            print(f"[EXCLUDE] {czi} | {reason}")
            excluded.append({
                "file_path": str(czi),
                "date_token": extract_date_token(czi),
                "acquisition_block": extract_acquisition_block(czi),
                "reason": reason,
            })
        else:
            usable.append(czi)

    pd.DataFrame(excluded).to_csv(
        METADATA_DIR / "excluded_czi_files.csv",
        index=False,
    )

    return sorted(usable)


# ============================================================
# CZI LAZY READER
# ============================================================

def get_dim_size(dim_shape, dim_name):
    """
    Handles aicspylibczi get_dims_shape output.

    Example:
    [{'X': (0, 1024), 'Y': (0, 1024), 'Z': (0, 60), 'C': (0, 2), 'T': (0, 120)}]
    """
    if isinstance(dim_shape, list):
        dim_shape = dim_shape[0]

    if dim_name not in dim_shape:
        return 1

    start, size = dim_shape[dim_name]
    return int(size)


def squeeze_to_zyx(img):
    """
    Converts aicspylibczi output after reading one T and one C into Z,Y,X.
    """
    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    if img.ndim != 3:
        raise ValueError(f"Expected Z,Y,X or Y,X after squeeze. Got shape: {img.shape}")

    return img.astype(np.float32)


class LazyCziReader:
    def __init__(self, czi_path):
        self.czi_path = Path(czi_path)
        self.czi = CziFile(self.czi_path)

        self.dim_shape = self.czi.get_dims_shape()

        self.size_t = get_dim_size(self.dim_shape, "T")
        self.size_z = get_dim_size(self.dim_shape, "Z")
        self.size_c = get_dim_size(self.dim_shape, "C")
        self.size_y = get_dim_size(self.dim_shape, "Y")
        self.size_x = get_dim_size(self.dim_shape, "X")

    def read_timepoint_zyx(self, t_index, c_index=0):
        img, shape = self.czi.read_image(T=t_index, C=c_index)
        return squeeze_to_zyx(img)

    def read_slice_yx(self, t_index, z_index, c_index=0):
        """
        Reads one timepoint/channel, then returns one Z plane.
        This is not as memory-minimal as true plane reading, but is safer with CZI dimension quirks.
        """
        zyx = self.read_timepoint_zyx(t_index=t_index, c_index=c_index)
        return zyx[z_index].astype(np.float32)

    def read_timepoint_max_projection(self, t_index, c_index=0):
        zyx = self.read_timepoint_zyx(t_index=t_index, c_index=c_index)
        return np.max(zyx, axis=0).astype(np.float32)


# ============================================================
# NORMALISATION AND QC
# ============================================================

def normalize_to_uint16(img):
    img = np.asarray(img, dtype=np.float32)

    p1, p99 = np.percentile(img, (P_LOW, P_HIGH))

    if p99 <= p1:
        return np.zeros_like(img, dtype=np.uint16)

    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    img = img * 65535

    return img.astype(np.uint16)


def normalize_for_video(img):
    img = np.asarray(img, dtype=np.float32)

    p1, p99 = np.percentile(img, (1, 99))

    if p99 <= p1:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip(img, p1, p99)
    img = rescale_intensity(img, in_range=(p1, p99), out_range=(0, 255))

    return img.astype(np.uint8)


# ============================================================
# FINAL DRIFT CORRECTION: OPTION B
# ============================================================

def preprocess_for_registration(img, crop_fraction=0.75):
    """
    Preprocess max-Z projection before drift estimation.

    Steps:
    1. Central crop to avoid unstable edges.
    2. Percentile normalisation.
    3. Smooth background subtraction.
    """
    img = np.asarray(img, dtype=np.float32)

    h, w = img.shape

    crop_h = int(h * crop_fraction)
    crop_w = int(w * crop_fraction)

    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]

    p1, p99 = np.percentile(cropped, (1, 99.5))

    if p99 > p1:
        cropped = np.clip(cropped, p1, p99)
        cropped = rescale_intensity(cropped, in_range=(p1, p99), out_range=(0, 1))
    else:
        cropped = np.zeros_like(cropped, dtype=np.float32)

    background = gaussian(cropped, sigma=8)
    enhanced = cropped - background
    enhanced = np.clip(enhanced, 0, None)

    return enhanced.astype(np.float32)


def limit_shift(shift_yx, max_shift_per_step=80):
    shift_y, shift_x = float(shift_yx[0]), float(shift_yx[1])

    if abs(shift_y) > max_shift_per_step or abs(shift_x) > max_shift_per_step:
        print(
            f"[WARNING] Rejecting suspicious step shift: "
            f"Y={shift_y:.2f}, X={shift_x:.2f}"
        )
        return np.array([0.0, 0.0], dtype=np.float32)

    return np.array([shift_y, shift_x], dtype=np.float32)


# Adapted from: https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation
def estimate_drift_all_timepoints_option_b(reader, channel_index):
    """
    Final drift correction approach.

    Registers consecutive max projections:
        T1 to T0
        T2 to T1
        T3 to T2
        ...

    Then accumulates the correction shifts.

    Returns:
        time_indices: list of actual timepoints
        shifts_yx: array with correction shift for each timepoint
        original_projections: raw max projections for QC video
    """
    time_indices = list(range(0, reader.size_t, DRIFT_TIMEPOINT_STEP))

    if len(time_indices) == 0:
        raise ValueError("No timepoints found for drift correction.")

    cumulative_shifts = []
    original_projections = []

    cumulative_shift = np.array([0.0, 0.0], dtype=np.float32)

    print("[DRIFT] Reading reference timepoint...")

    previous_raw = reader.read_timepoint_max_projection(
        t_index=time_indices[0],
        c_index=channel_index,
    )

    previous_processed = preprocess_for_registration(
        previous_raw,
        crop_fraction=DRIFT_CROP_FRACTION,
    )

    cumulative_shifts.append(cumulative_shift.copy())
    original_projections.append(previous_raw)

    print(
        f"[DRIFT] T={time_indices[0]:04d} | "
        f"cumulative_y=0.000, cumulative_x=0.000"
    )

    for i in range(1, len(time_indices)):
        t_idx = time_indices[i]

        current_raw = reader.read_timepoint_max_projection(
            t_index=t_idx,
            c_index=channel_index,
        )

        current_processed = preprocess_for_registration(
            current_raw,
            crop_fraction=DRIFT_CROP_FRACTION,
        )

        step_shift_yx, error, diffphase = phase_cross_correlation(
            previous_processed,
            current_processed,
            upsample_factor=DRIFT_UPSAMPLE_FACTOR,
        )

        step_shift_yx = limit_shift(
            step_shift_yx,
            max_shift_per_step=DRIFT_MAX_SHIFT_PER_STEP,
        )

        cumulative_shift = cumulative_shift + step_shift_yx

        cumulative_shifts.append(cumulative_shift.copy())
        original_projections.append(current_raw)

        print(
            f"[DRIFT] T={t_idx:04d} | "
            f"step_y={step_shift_yx[0]:.3f}, "
            f"step_x={step_shift_yx[1]:.3f} | "
            f"cumulative_y={cumulative_shift[0]:.3f}, "
            f"cumulative_x={cumulative_shift[1]:.3f} | "
            f"error={error:.5f}"
        )

        previous_processed = current_processed

    return (
        time_indices,
        np.asarray(cumulative_shifts, dtype=np.float32),
        np.asarray(original_projections, dtype=np.float32),
    )


def apply_shift_black_fill(img2d, shift_y, shift_x):
    """
    Applies correction shift to one 2D slice.
    Empty shifted-in regions are filled with black.
    """
    corrected = ndi_shift(
        img2d.astype(np.float32),
        shift=(float(shift_y), float(shift_x)),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    return corrected.astype(np.float32)


def apply_shift_to_projection(proj, shift_y, shift_x):
    return apply_shift_black_fill(proj, shift_y, shift_x)


# ============================================================
# DRIFT QC SAVING
# ============================================================

def save_qc_video(max_proj_tyx, output_path, fps=5):
    frames = [normalize_for_video(frame) for frame in max_proj_tyx]
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"[QC] Saved video: {output_path}")


def save_shift_csv(time_indices, shifts_yx, output_path):
    rows = []

    for i, t_idx in enumerate(time_indices):
        rows.append({
            "processed_index": i,
            "actual_timepoint": t_idx,
            "shift_y": float(shifts_yx[i, 0]),
            "shift_x": float(shifts_yx[i, 1]),
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[QC] Saved drift CSV: {output_path}")


def save_shift_plot(time_indices, shifts_yx, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, shifts_yx[:, 0], label="Y correction shift")
    plt.plot(time_indices, shifts_yx[:, 1], label="X correction shift")
    plt.xlabel("Timepoint")
    plt.ylabel("Correction shift in pixels")
    plt.title("Estimated global specimen drift")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[QC] Saved drift plot: {output_path}")


def save_before_after_images(original_tyx, corrected_tyx, time_indices, output_dir):
    if len(time_indices) == 0:
        return

    selected = sorted(set([
        0,
        len(time_indices) // 2,
        len(time_indices) - 1,
    ]))

    for i in selected:
        original = normalize_for_video(original_tyx[i])
        corrected = normalize_for_video(corrected_tyx[i])

        comparison = np.concatenate([original, corrected], axis=1)

        actual_t = time_indices[i]
        out_path = output_dir / f"before_after_T{actual_t:04d}.png"

        imageio.imwrite(out_path, comparison)
        print(f"[QC] Saved before/after image: {out_path}")


def save_drift_qc_outputs(
    reader,
    czi_path,
    file_index,
    time_indices,
    shifts_yx,
    original_projections,
):
    """
    Saves final report-ready drift QC for selected files:
    - original max projection timelapse
    - corrected max projection timelapse
    - before/after PNGs
    - shift CSV
    - shift plot
    """
    date_token = extract_date_token(czi_path)
    block_num = extract_acquisition_block(czi_path)

    if block_num is None:
        block_token = "block_unknown"
    else:
        block_token = f"block{block_num:02d}"

    qc_dir = DRIFT_QC_DIR / f"file{file_index:03d}_{date_token}_{block_token}"
    ensure_dir(qc_dir)

    corrected_projections = []

    for i, proj in enumerate(original_projections):
        shift_y, shift_x = shifts_yx[i]
        corrected = apply_shift_to_projection(proj, shift_y, shift_x)
        corrected_projections.append(corrected)

    corrected_projections = np.asarray(corrected_projections, dtype=np.float32)

    save_shift_csv(
        time_indices=time_indices,
        shifts_yx=shifts_yx,
        output_path=qc_dir / "estimated_drift_shifts.csv",
    )

    save_shift_plot(
        time_indices=time_indices,
        shifts_yx=shifts_yx,
        output_path=qc_dir / "drift_shift_plot.png",
    )

    save_qc_video(
        original_projections,
        qc_dir / "original_lazy_max_projection_timelapse.mp4",
        fps=DRIFT_QC_VIDEO_FPS,
    )

    save_qc_video(
        corrected_projections,
        qc_dir / "corrected_lazy_max_projection_timelapse.mp4",
        fps=DRIFT_QC_VIDEO_FPS,
    )

    save_before_after_images(
        original_tyx=original_projections,
        corrected_tyx=corrected_projections,
        time_indices=time_indices,
        output_dir=qc_dir,
    )


# ============================================================
# TIMEPOINT AND Z-SELECTION
# ============================================================

def choose_two_from_region(indices):
    indices = list(indices)

    if len(indices) == 0:
        return []

    if len(indices) <= TIMEPOINTS_PER_TIME_REGION:
        return [int(x) for x in indices]

    positions = np.linspace(
        0,
        len(indices) - 1,
        TIMEPOINTS_PER_TIME_REGION + 2,
    )

    positions = np.round(positions[1:-1]).astype(int)
    return sorted(set(int(indices[p]) for p in positions))


def choose_timepoints(num_t):
    """
    Extraction:
    2 from start, 2 from middle, 2 from end.
    """
    if num_t <= 0:
        return []

    if num_t <= 6:
        return list(range(num_t))

    all_t = np.arange(num_t)

    start_region = all_t[: max(1, num_t // 3)]
    middle_region = all_t[max(0, num_t // 3): min(num_t, 2 * num_t // 3)]
    end_region = all_t[max(0, 2 * num_t // 3):]

    selected = []
    selected += choose_two_from_region(start_region)
    selected += choose_two_from_region(middle_region)
    selected += choose_two_from_region(end_region)

    return sorted(set(int(t) for t in selected))


def get_z_ignore_values(date_token):
    ignore_first = SPECIAL_EARLY_Z_IGNORE_BY_DATE.get(date_token, DEFAULT_IGNORE_FIRST_Z)
    ignore_last = SPECIAL_LAST_Z_IGNORE_BY_DATE.get(date_token, DEFAULT_IGNORE_LAST_Z)

    return int(ignore_first), int(ignore_last)


def choose_spread_indices(indices, n):
    """
    Selects n spread-out non-consecutive-ish indices from a region.
    """
    indices = list(indices)

    if len(indices) == 0:
        return []

    if len(indices) <= n:
        return [int(x) for x in indices]

    positions = np.linspace(0, len(indices) - 1, n)
    positions = np.round(positions).astype(int)

    selected = []
    seen = set()

    for p in positions:
        value = int(indices[p])
        if value not in seen:
            selected.append(value)
            seen.add(value)

    return selected


def choose_z_slices(num_z, date_token):
    """
    Extraction:
    ignore early/late bad slices,
    then take 4 spread-out slices from start, middle, and end Z regions.
    """
    ignore_first, ignore_last = get_z_ignore_values(date_token)

    z_start = min(ignore_first, max(0, num_z - 1))
    z_end_exclusive = max(z_start + 1, num_z - ignore_last)

    valid_z = np.arange(z_start, z_end_exclusive)

    if len(valid_z) == 0:
        return []

    if len(valid_z) <= Z_SLICES_PER_Z_REGION * 3:
        return sorted(set(int(z) for z in valid_z))

    one_third = len(valid_z) // 3
    two_third = 2 * len(valid_z) // 3

    start_region = valid_z[:one_third]
    middle_region = valid_z[one_third:two_third]
    end_region = valid_z[two_third:]

    selected = []
    selected += choose_spread_indices(start_region, Z_SLICES_PER_Z_REGION)
    selected += choose_spread_indices(middle_region, Z_SLICES_PER_Z_REGION)
    selected += choose_spread_indices(end_region, Z_SLICES_PER_Z_REGION)

    return sorted(set(int(z) for z in selected))


def classify_time_region(t, num_t):
    if num_t <= 1:
        return "tsingle"

    frac = t / max(1, num_t - 1)

    if frac < 1 / 3:
        return "tstart"
    if frac < 2 / 3:
        return "tmiddle"
    return "tend"


def classify_z_region(z, num_z, date_token):
    ignore_first, ignore_last = get_z_ignore_values(date_token)

    z_start = min(ignore_first, max(0, num_z - 1))
    z_end = max(z_start + 1, num_z - ignore_last - 1)

    frac = (z - z_start) / max(1, z_end - z_start)

    if frac < 1 / 3:
        return "zstart"
    if frac < 2 / 3:
        return "zmiddle"
    return "zend"


# ============================================================
# SPLITTING AND METADATA
# ============================================================

def inspect_czi(czi_path):
    reader = LazyCziReader(czi_path)

    return {
        "date_token": extract_date_token(czi_path),
        "acquisition_block": extract_acquisition_block(czi_path),
        "file_name": Path(czi_path).name,
        "file_path": str(czi_path),
        "num_t": reader.size_t,
        "num_c": reader.size_c,
        "num_z": reader.size_z,
        "num_y": reader.size_y,
        "num_x": reader.size_x,
        "dim_shape": str(reader.dim_shape),
    }


def create_inventory(czi_files):
    records = []

    for czi in czi_files:
        print(f"[INVENTORY] {czi}")

        try:
            rec = inspect_czi(czi)
            rec["error"] = ""
        except Exception as e:
            rec = {
                "date_token": extract_date_token(czi),
                "acquisition_block": extract_acquisition_block(czi),
                "file_name": Path(czi).name,
                "file_path": str(czi),
                "error": str(e),
            }

        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(METADATA_DIR / "czi_inventory.csv", index=False)

    return df


def split_files(czi_files):
    random.seed(RANDOM_SEED)

    files = list(czi_files)
    random.shuffle(files)

    n = len(files)
    n_train = int(round(n * TRAIN_FRACTION))
    n_val = int(round(n * VAL_FRACTION))

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    records = []

    for split, split_files_list in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        for p in split_files_list:
            records.append({
                "file_path": str(p),
                "split": split,
                "date_token": extract_date_token(p),
                "acquisition_block": extract_acquisition_block(p),
            })

    df = pd.DataFrame(records)
    df.to_csv(METADATA_DIR / "czi_file_split.csv", index=False)

    return df


def get_output_dirs(split):
    if split == "train":
        return TRAIN_IMAGES_DIR, QC_TRAIN_DIR
    if split == "val":
        return VAL_IMAGES_DIR, QC_VAL_DIR
    if split == "test":
        return TEST_IMAGES_DIR, QC_TEST_DIR

    raise ValueError(f"Unknown split: {split}")


def make_slice_filename(
    date_token,
    block_num,
    split,
    file_index,
    t_idx,
    z_idx,
    time_region,
    z_region,
):
    if block_num is None:
        block_token = "block_unknown"
    else:
        block_token = f"block{int(block_num):02d}"

    return (
        f"macrophage_{safe_token(date_token)}_"
        f"{block_token}_"
        f"{split}_"
        f"file{file_index:03d}_"
        f"T{t_idx:04d}_"
        f"Z{z_idx:04d}_"
        f"{time_region}_"
        f"{z_region}.tif"
    )


# ============================================================
# SLICE EXTRACTION AFTER DRIFT CORRECTION
# ============================================================

def get_shift_for_timepoint(time_indices, shifts_yx, t_idx):
    """
    Returns correction shift for actual timepoint t_idx.
    Since DRIFT_TIMEPOINT_STEP is normally 1, this is direct.
    """
    time_to_i = {int(t): i for i, t in enumerate(time_indices)}

    if int(t_idx) not in time_to_i:
        raise ValueError(
            f"Selected timepoint {t_idx} was not included in drift estimation. "
            f"Use DRIFT_TIMEPOINT_STEP=1 for macrophage dataset creation."
        )

    i = time_to_i[int(t_idx)]
    return float(shifts_yx[i, 0]), float(shifts_yx[i, 1])


def save_corrected_training_slice(
    reader,
    czi_path,
    split,
    file_index,
    t_idx,
    z_idx,
    shift_y,
    shift_x,
    channel_index,
):
    img_dir, qc_dir = get_output_dirs(split)

    date_token = extract_date_token(czi_path)
    block_num = extract_acquisition_block(czi_path)

    time_region = classify_time_region(t_idx, reader.size_t)
    z_region = classify_z_region(z_idx, reader.size_z, date_token)

    filename = make_slice_filename(
        date_token=date_token,
        block_num=block_num,
        split=split,
        file_index=file_index,
        t_idx=t_idx,
        z_idx=z_idx,
        time_region=time_region,
        z_region=z_region,
    )

    out_tif = img_dir / filename
    out_qc = qc_dir / filename.replace(".tif", "_qc.png")

    if out_tif.exists() and not OVERWRITE_EXISTING_DATASET:
        raise FileExistsError(
            f"Output already exists: {out_tif}\n"
            f"Set OVERWRITE_EXISTING_DATASET=True to overwrite."
        )

    raw_slice = reader.read_slice_yx(
        t_index=t_idx,
        z_index=z_idx,
        c_index=channel_index,
    )

    corrected_slice = apply_shift_black_fill(raw_slice, shift_y, shift_x)

    corrected_uint16 = normalize_to_uint16(corrected_slice)

    tiff.imwrite(
        out_tif,
        corrected_uint16,
        photometric="minisblack",
    )

    qc_uint8 = rescale_intensity(
        corrected_uint16,
        in_range="image",
        out_range=np.uint8,
    ).astype(np.uint8)

    imsave(out_qc, qc_uint8, check_contrast=False)

    return {
        "split": split,
        "date_token": date_token,
        "acquisition_block": block_num,
        "source_czi": str(czi_path),
        "file_index": file_index,
        "time_index": int(t_idx),
        "z_index": int(z_idx),
        "time_region": time_region,
        "z_region": z_region,
        "shift_y": float(shift_y),
        "shift_x": float(shift_x),
        "channel_index": channel_index,
        "output_tif": str(out_tif),
        "output_qc_png": str(out_qc),
        "normalisation_p_low": P_LOW,
        "normalisation_p_high": P_HIGH,
    }


def process_one_czi(czi_path, split, file_index, save_drift_qc_for_this_file=False):
    print("\n" + "=" * 90)
    print(f"[PROCESS] file_index={file_index:03d} | split={split}")
    print(f"[CZI] {czi_path}")

    reader = LazyCziReader(czi_path)

    date_token = extract_date_token(czi_path)
    block_num = extract_acquisition_block(czi_path)

    print(
        f"[INFO] date={date_token}, block={block_num}, "
        f"T={reader.size_t}, C={reader.size_c}, Z={reader.size_z}, "
        f"Y={reader.size_y}, X={reader.size_x}"
    )

    macrophage_channel_index = get_effective_channel_index(
        reader,
        requested_index=MACROPHAGE_CHANNEL_INDEX,
        channel_name="macrophage extraction",
    )
    drift_channel_index = get_effective_channel_index(
        reader,
        requested_index=DRIFT_REGISTRATION_CHANNEL_INDEX,
        channel_name="drift registration",
    )

    print(f"[INFO] selected macrophage extraction channel index: {macrophage_channel_index}")
    print(f"[INFO] selected drift registration channel index: {drift_channel_index}")

    selected_t = choose_timepoints(reader.size_t)
    selected_z = choose_z_slices(reader.size_z, date_token)

    ignore_first, ignore_last = get_z_ignore_values(date_token)

    print(f"[INFO] ignore_first_z={ignore_first}, ignore_last_z={ignore_last}")
    print(f"[INFO] selected timepoints: {selected_t}")
    print(f"[INFO] selected z-slices: {selected_z}")
    print("[DRIFT] Estimating Option-B drift across ALL timepoints...")

    time_indices, shifts_yx, original_projections = estimate_drift_all_timepoints_option_b(
        reader=reader,
        channel_index=drift_channel_index,
    )

    drift_csv_name = (
        f"drift_shifts_file{file_index:03d}_"
        f"{safe_token(date_token)}_"
        f"block{block_num if block_num is not None else 'unknown'}.csv"
    )

    save_shift_csv(
        time_indices=time_indices,
        shifts_yx=shifts_yx,
        output_path=METADATA_DIR / drift_csv_name,
    )

    if SAVE_DRIFT_QC and save_drift_qc_for_this_file:
        print("[QC] Saving report-ready drift QC outputs for this file...")
        save_drift_qc_outputs(
            reader=reader,
            czi_path=czi_path,
            file_index=file_index,
            time_indices=time_indices,
            shifts_yx=shifts_yx,
            original_projections=original_projections,
        )

    records = []

    print("[EXTRACT] Saving drift-corrected selected slices...")

    for t_idx in selected_t:
        shift_y, shift_x = get_shift_for_timepoint(
            time_indices=time_indices,
            shifts_yx=shifts_yx,
            t_idx=t_idx,
        )

        for z_idx in selected_z:
            try:
                rec = save_corrected_training_slice(
                    reader=reader,
                    czi_path=czi_path,
                    split=split,
                    file_index=file_index,
                    t_idx=t_idx,
                    z_idx=z_idx,
                    shift_y=shift_y,
                    shift_x=shift_x,
                    channel_index=macrophage_channel_index,
                )
                rec["drift_registration_channel_index"] = drift_channel_index
                records.append(rec)

            except Exception as e:
                print(
                    f"[WARNING] Failed saving slice "
                    f"T={t_idx}, Z={z_idx}: {e}"
                )

    print(f"[DONE] Saved {len(records)} corrected candidate slices from this file.")

    return records

def resolved_path_string(path):
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(Path(path).absolute())


def append_rows_to_csv(new_df, csv_path, drop_duplicates_by=None):
    csv_path = Path(csv_path)

    if new_df is None or len(new_df) == 0:
        return

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    if drop_duplicates_by is not None:
        existing_cols = [c for c in drop_duplicates_by if c in combined.columns]
        if len(existing_cols) > 0:
            combined = combined.drop_duplicates(subset=existing_cols, keep="last")

    combined.to_csv(csv_path, index=False)


def get_existing_czi_paths():
    existing_paths = set()

    for csv_name in ["czi_file_split.csv", "czi_inventory.csv"]:
        csv_path = METADATA_DIR / csv_name

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        if "file_path" not in df.columns:
            continue

        for p in df["file_path"].dropna().tolist():
            existing_paths.add(resolved_path_string(p))

    return existing_paths


def get_next_file_index():
    metadata_path = METADATA_DIR / "extracted_slices_metadata.csv"

    if not metadata_path.exists():
        return 0

    df = pd.read_csv(metadata_path)

    if "file_index" not in df.columns or len(df) == 0:
        return 0

    return int(df["file_index"].max()) + 1


def create_inventory_append(czi_files):
    records = []

    for czi in czi_files:
        print(f"[INVENTORY] {czi}")

        try:
            rec = inspect_czi(czi)
            rec["error"] = ""
        except Exception as e:
            rec = {
                "date_token": extract_date_token(czi),
                "acquisition_block": extract_acquisition_block(czi),
                "file_name": Path(czi).name,
                "file_path": str(czi),
                "error": str(e),
            }

        rec["append_run"] = APPEND_RUN_NAME
        records.append(rec)

    df_new = pd.DataFrame(records)

    # save this append inventory separately
    df_new.to_csv(
        METADATA_DIR / f"czi_inventory_{APPEND_RUN_NAME}.csv",
        index=False,
    )

    # append into main inventory
    append_rows_to_csv(
        df_new,
        METADATA_DIR / "czi_inventory.csv",
        drop_duplicates_by=["file_path"],
    )

    return df_new


def split_files_append(czi_files, next_file_index):
    random.seed(RANDOM_SEED)

    files = list(czi_files)
    random.shuffle(files)

    n = len(files)
    n_train = int(round(n * TRAIN_FRACTION))
    n_val = int(round(n * VAL_FRACTION))

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    records = []
    file_index = next_file_index

    for split, split_files_list in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        for p in split_files_list:
            records.append({
                "file_index": file_index,
                "file_path": str(p),
                "split": split,
                "date_token": extract_date_token(p),
                "acquisition_block": extract_acquisition_block(p),
                "append_run": APPEND_RUN_NAME,
            })
            file_index += 1

    df_new = pd.DataFrame(records)

    # save this append split separately
    df_new.to_csv(
        METADATA_DIR / f"czi_file_split_{APPEND_RUN_NAME}.csv",
        index=False,
    )

    # append into main split metadata
    append_rows_to_csv(
        df_new,
        METADATA_DIR / "czi_file_split.csv",
        drop_duplicates_by=["file_path"],
    )

    return df_new


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    make_dirs()

    print("\n[STEP 1] Finding NEW CZI/acquisition-block files...")
    czi_files = find_czi_files()

    if len(czi_files) == 0:
        raise RuntimeError("No CZI files found. Check SOURCE_PATHS.")

    print(f"[INFO] Found {len(czi_files)} CZI files from new source paths.")

    existing_paths = get_existing_czi_paths()

    new_czi_files = []
    skipped_existing = []

    for p in czi_files:
        rp = resolved_path_string(p)

        if rp in existing_paths:
            skipped_existing.append(p)
        else:
            new_czi_files.append(p)

    print(f"[INFO] New CZI files to process: {len(new_czi_files)}")
    print(f"[INFO] Already-existing CZI files skipped: {len(skipped_existing)}")

    if len(new_czi_files) == 0:
        print("[DONE] Nothing new to append.")
        return

    print("\n[STEP 2] Creating append inventory...")
    inventory_df = create_inventory_append(new_czi_files)

    usable_df = inventory_df[inventory_df["error"].fillna("") == ""].copy()
    usable_paths = [Path(p) for p in usable_df["file_path"].tolist()]

    if len(usable_paths) == 0:
        raise RuntimeError("No readable new CZI files found after inventory.")

    print(f"[INFO] Readable new CZI files: {len(usable_paths)}")

    print("\n[STEP 3] Creating train/val/test split for NEW files only...")
    next_file_index = get_next_file_index()
    print(f"[INFO] Starting new file_index from: {next_file_index}")

    split_df = split_files_append(
        czi_files=usable_paths,
        next_file_index=next_file_index,
    )

    print("\n[INFO] New split counts:")
    print(split_df["split"].value_counts())

    print("\n[STEP 4] Running drift correction and extracting NEW corrected slices...")

    all_records = []
    successful_qc_files = 0
    processing_errors = []

    for _, row in split_df.reset_index(drop=True).iterrows():
        czi_path = Path(row["file_path"])
        split = row["split"]
        file_index = int(row["file_index"])

        save_qc_for_this = successful_qc_files < NUM_FILES_FOR_DRIFT_QC

        try:
            records = process_one_czi(
                czi_path=czi_path,
                split=split,
                file_index=file_index,
                save_drift_qc_for_this_file=save_qc_for_this,
            )

            for rec in records:
                rec["append_run"] = APPEND_RUN_NAME

            all_records.extend(records)

            if save_qc_for_this:
                successful_qc_files += 1

        except Exception as e:
            print(f"[ERROR] Failed processing file: {czi_path}")
            print(f"[ERROR] {e}")

            processing_errors.append({
                "file_index": file_index,
                "split": split,
                "file_path": str(czi_path),
                "date_token": extract_date_token(czi_path),
                "acquisition_block": extract_acquisition_block(czi_path),
                "append_run": APPEND_RUN_NAME,
                "error": str(e),
            })

    print("\n[STEP 5] Appending metadata...")

    extracted_df = pd.DataFrame(all_records)

    if len(extracted_df) > 0:
        extracted_df.to_csv(
            METADATA_DIR / f"extracted_slices_metadata_{APPEND_RUN_NAME}.csv",
            index=False,
        )

        append_rows_to_csv(
            extracted_df,
            METADATA_DIR / "extracted_slices_metadata.csv",
            drop_duplicates_by=["output_tif"],
        )

    if len(processing_errors) > 0:
        error_df = pd.DataFrame(processing_errors)

        error_df.to_csv(
            METADATA_DIR / f"processing_errors_{APPEND_RUN_NAME}.csv",
            index=False,
        )

        append_rows_to_csv(
            error_df,
            METADATA_DIR / "processing_errors.csv",
        )

    print("\n" + "=" * 90)
    print("[APPEND COMPLETE]")
    print(f"New readable CZI/acquisition-block files: {len(usable_paths)}")
    print(f"New extracted candidate slices: {len(extracted_df)}")
    print(f"Dataset folder updated: {DATASET_DIR}")
    print(f"Metadata folder: {METADATA_DIR}")
    print(f"Drift QC folder: {DRIFT_QC_DIR}")

    if len(extracted_df) > 0:
        print("\nNew slices per split:")
        print(extracted_df["split"].value_counts())

        print("\nNew slices per date:")
        print(extracted_df["date_token"].value_counts())

        print("\nNew slices per Z region:")
        print(extracted_df["z_region"].value_counts())

        print("\nNew slices per time region:")
        print(extracted_df["time_region"].value_counts())

    print("\nNext step:")
    print("Run the preprocessing append script to add these new slices to macrophage_cellpose_dataset_preprocessed.")


if __name__ == "__main__":
    main()
