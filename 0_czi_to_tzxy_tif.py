import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from aicspylibczi import CziFile
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian

from config import (
    CZI_PATH,
    PROJECT_DIR,
    RAW_TCZYX_TIF,
    RAW_TCZYX_UNCORRECTED_TIF,
    RAW_TZYX_TIF,
    RAW_TZYX_UNCORRECTED_TIF,
    DRIFT_QC_DIR,
    CHANNEL_INDEX,
    SCENE_INDEX,
    TIME_START,
    TIME_END,
    TIME_STRIDE,
    Z_START,
    Z_END,
    DRIFT_UPSAMPLE_FACTOR,
    DRIFT_CROP_FRACTION,
    DRIFT_MAX_SHIFT_PER_STEP,
    SAVE_DRIFT_QC,
    SAVE_UNCORRECTED_SUBSET_TIF,
    NORMALIZE_DRIFT_OUTPUT_TO_UINT16,
)



# Basic helpers


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_for_video(img):
    img = np.asarray(img, dtype=np.float32)

    p1, p99 = np.percentile(img, (1, 99))

    if p99 <= p1:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip(img, p1, p99)
    img = rescale_intensity(img, in_range=(p1, p99), out_range=(0, 255))

    return img.astype(np.uint8)


def normalize_to_uint16(img):
    """Optional old-style clean uint16 conversion."""
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99.8))

    if p99 <= p1:
        return np.zeros_like(img, dtype=np.uint16)

    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1)
    img = img * 65535

    return img.astype(np.uint16)


def cast_like_original(img_float, original_dtype):
    """Preserve original intensity scale where possible."""
    if NORMALIZE_DRIFT_OUTPUT_TO_UINT16:
        return normalize_to_uint16(img_float)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        img_float = np.clip(img_float, info.min, info.max)
        return np.rint(img_float).astype(original_dtype)

    return img_float.astype(original_dtype)


def squeeze_to_zyx(img):
    """Converts aicspylibczi output into Z, Y, X."""

    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    if img.ndim != 3:
        raise ValueError(
            f"Expected squeezed image to be Z,Y,X or Y,X. Got shape: {img.shape}"
        )

    return img


def get_dim_size(dim_shape, dim_name):
    """Handles aicspylibczi get_dims_shape output."""

    if isinstance(dim_shape, list):
        dim_shape = dim_shape[0]

    if dim_name not in dim_shape:
        return 1

    start, size = dim_shape[dim_name]
    return size


def choose_indices(total_size, start, end, stride):
    if end is None:
        end = total_size

    end = min(end, total_size)

    return list(range(start, end, stride))



# Czi lazy reader


class LazyCziReader:
    def __init__(self, czi_path, scene_index=None):
        self.czi_path = Path(czi_path)
        self.scene_index = scene_index

        print("[INFO] Opening CZI file...")
        self.czi = CziFile(self.czi_path)

        print("[INFO] CZI dims:", self.czi.dims)

        self.dim_shape = self.czi.get_dims_shape()
        print("[INFO] CZI dim shape:", self.dim_shape)

        self.size_t = get_dim_size(self.dim_shape, "T")
        self.size_z = get_dim_size(self.dim_shape, "Z")
        self.size_c = get_dim_size(self.dim_shape, "C")
        self.size_y = get_dim_size(self.dim_shape, "Y")
        self.size_x = get_dim_size(self.dim_shape, "X")

        print("\n[INFO] Detected sizes:")
        print("T:", self.size_t)
        print("Z:", self.size_z)
        print("C:", self.size_c)
        print("Y:", self.size_y)
        print("X:", self.size_x)

    def read_timepoint_zyx(self, t_index, c_index=0):

        kwargs = {
            "T": int(t_index),
            "C": int(c_index),
        }

        if self.scene_index is not None and "S" in self.czi.dims:
            kwargs["S"] = int(self.scene_index)

        img, shape = self.czi.read_image(**kwargs)
        zyx = squeeze_to_zyx(img)

        if zyx.shape[1:] != (self.size_y, self.size_x):
            raise ValueError(
                f"Unexpected YX shape at T={t_index}, C={c_index}: "
                f"{zyx.shape[1:]}, expected {(self.size_y, self.size_x)}"
            )

        z_start = max(0, int(Z_START))

        requested_z_end = (
            int(Z_END)
            if Z_END is not None
            else int(self.size_z)
        )

        requested_z_end = min(
            requested_z_end,
            int(self.size_z),
        )

        expected_z = requested_z_end - z_start

        available_z_end = min(
            requested_z_end,
            zyx.shape[0],
        )

        selected = zyx[z_start:available_z_end]

        if selected.shape[0] != expected_z:
            fixed = np.zeros(
                (
                    expected_z,
                    self.size_y,
                    self.size_x,
                ),
                dtype=zyx.dtype,
            )

            copy_z = min(
                selected.shape[0],
                expected_z,
            )

            fixed[:copy_z] = selected[:copy_z]

            print(
                f"[WARNING] Z-depth mismatch at "
                f"T={t_index}, C={c_index}: "
                f"found {selected.shape[0]}, "
                f"expected {expected_z}. "
                f"Zero-padding missing Z slices."
            )

            selected = fixed

        return selected

    def read_timepoint_max_projection(self, t_index, c_index=0):
        zyx = self.read_timepoint_zyx(
            t_index=t_index,
            c_index=c_index
        )

        max_proj = np.max(zyx, axis=0)

        return max_proj.astype(np.float32)



# Drift preprocessing


def preprocess_for_registration(img, crop_fraction=0.75):
    """Robust preprocessing before phase cross-correlation."""

    img = img.astype(np.float32)

    h, w = img.shape

    crop_h = int(h * crop_fraction)
    crop_w = int(w * crop_fraction)

    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]

    p1, p99 = np.percentile(cropped, (1, 99.5))

    if p99 > p1:
        cropped = np.clip(cropped, p1, p99)
        cropped = rescale_intensity(
            cropped,
            in_range=(p1, p99),
            out_range=(0, 1)
        )
    else:
        cropped = np.zeros_like(cropped, dtype=np.float32)

    background = gaussian(cropped, sigma=8)
    enhanced = cropped - background
    enhanced = np.clip(enhanced, 0, None)

    return enhanced.astype(np.float32)


def limit_shift(shift_yx, max_shift_per_step=20):
    """Reject unrealistic registration jumps."""

    shift_y, shift_x = shift_yx

    if abs(shift_y) > max_shift_per_step or abs(shift_x) > max_shift_per_step:
        print(
            f"[WARNING] Rejecting suspicious shift: "
            f"Y={shift_y:.2f}, X={shift_x:.2f}. Using 0,0 instead."
        )
        return np.array([0.0, 0.0], dtype=np.float32)

    return np.array([shift_y, shift_x], dtype=np.float32)



# Drift estimation


# Phase-correlation algorithm: https://github.com/scikit-image/scikit-image/blob/main/skimage/registration/_phase_cross_correlation.py
# API example: https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html
def estimate_drift_lazy_robust(
    reader,
    time_indices,
    channel_index,
    upsample_factor=5,
    crop_fraction=0.75,
    max_shift_per_step=20,
):
    """Consecutive-frame drift estimation."""

    cumulative_shifts = []
    original_projections_for_qc = []

    cumulative_shift = np.array([0.0, 0.0], dtype=np.float32)

    print("\n[INFO] Reading first timepoint for drift estimation...")

    previous_raw = reader.read_timepoint_max_projection(
        t_index=time_indices[0],
        c_index=channel_index
    )

    previous_processed = preprocess_for_registration(
        previous_raw,
        crop_fraction=crop_fraction
    )

    cumulative_shifts.append(cumulative_shift.copy())
    original_projections_for_qc.append(previous_raw)

    print(
        f"[DRIFT] T={time_indices[0]:04d} | "
        f"step_y=0.000, step_x=0.000 | "
        f"cumulative_y=0.000, cumulative_x=0.000"
    )

    for i in range(1, len(time_indices)):
        t = time_indices[i]

        current_raw = reader.read_timepoint_max_projection(
            t_index=t,
            c_index=channel_index
        )

        current_processed = preprocess_for_registration(
            current_raw,
            crop_fraction=crop_fraction
        )

        step_shift_yx, error, diffphase = phase_cross_correlation(
            previous_processed,
            current_processed,
            upsample_factor=upsample_factor
        )

        step_shift_yx = limit_shift(
            step_shift_yx,
            max_shift_per_step=max_shift_per_step
        )

        cumulative_shift = cumulative_shift + step_shift_yx

        cumulative_shifts.append(cumulative_shift.copy())
        original_projections_for_qc.append(current_raw)

        print(
            f"[DRIFT] T={t:04d} | "
            f"step_y={step_shift_yx[0]:.3f}, "
            f"step_x={step_shift_yx[1]:.3f} | "
            f"cumulative_y={cumulative_shift[0]:.3f}, "
            f"cumulative_x={cumulative_shift[1]:.3f} | "
            f"error={error:.5f}"
        )

        # update reference to current processed frame.
        previous_processed = current_processed

    cumulative_shifts = np.asarray(cumulative_shifts, dtype=np.float32)
    original_projections_for_qc = np.asarray(original_projections_for_qc, dtype=np.float32)

    return cumulative_shifts, original_projections_for_qc



# Apply drift correction and build tzyx


# Shift interpolation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
def apply_drift_and_build_tczyx(reader, time_indices, shifts_yx):
    """Applies the same drift correction shifts to all channels."""
    corrected_timepoints = []
    uncorrected_timepoints = []
    corrected_max_projections = []

    output_dtype = None

    print("\n[INFO] Applying drift correction to all selected timepoints and all channels...")

    for i, t in enumerate(time_indices):
        corrected_channels = []
        uncorrected_channels = []

        shift_y, shift_x = shifts_yx[i]

        for c in range(reader.size_c):
            zyx = reader.read_timepoint_zyx(
                t_index=t,
                c_index=c
            )

            if output_dtype is None:
                output_dtype = zyx.dtype

            zyx_float = zyx.astype(np.float32)

            corrected_zyx_float = np.zeros_like(zyx_float, dtype=np.float32)

            for z in range(zyx_float.shape[0]):
                corrected_zyx_float[z] = ndi_shift(
                    zyx_float[z],
                    shift=(float(shift_y), float(shift_x)),
                    order=1,
                    mode="constant",
                    cval=0
                )

            corrected_zyx = cast_like_original(
                corrected_zyx_float,
                original_dtype=output_dtype
            )

            corrected_channels.append(corrected_zyx)
            uncorrected_channels.append(zyx.astype(output_dtype))

            # QC projection only from the drift-estimation channel
            if c == CHANNEL_INDEX:
                corrected_max_projections.append(np.max(corrected_zyx_float, axis=0))

        corrected_czyx = np.stack(corrected_channels, axis=0)
        uncorrected_czyx = np.stack(uncorrected_channels, axis=0)

        corrected_timepoints.append(corrected_czyx)
        uncorrected_timepoints.append(uncorrected_czyx)

        print(
            f"[INFO] Corrected T={t:04d} | "
            f"shift_y={shift_y:.3f}, shift_x={shift_x:.3f} | "
            f"shape={corrected_czyx.shape}"
        )

    corrected_tczyx = np.stack(corrected_timepoints, axis=0)
    uncorrected_tczyx = np.stack(uncorrected_timepoints, axis=0)
    corrected_max_projections = np.asarray(corrected_max_projections, dtype=np.float32)

    return corrected_tczyx, uncorrected_tczyx, corrected_max_projections



# Qc saving


def save_shift_csv(shifts_yx, time_indices, output_dir):
    output_path = Path(output_dir) / "estimated_drift_shifts.csv"

    df = pd.DataFrame({
        "processed_index": np.arange(len(time_indices)),
        "actual_timepoint": time_indices,
        "shift_y": shifts_yx[:, 0],
        "shift_x": shifts_yx[:, 1],
    })

    df.to_csv(output_path, index=False)
    print("[INFO] Saved:", output_path)


def save_shift_plot(shifts_yx, time_indices, output_dir):
    output_path = Path(output_dir) / "drift_shift_plot.png"

    y_shift = shifts_yx[:, 0]
    x_shift = shifts_yx[:, 1]

    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, y_shift, label="Y shift")
    plt.plot(time_indices, x_shift, label="X shift")
    plt.xlabel("Actual timepoint")
    plt.ylabel("Correction shift in pixels")
    plt.title("Estimated global specimen drift")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("[INFO] Saved:", output_path)


def save_qc_video(max_proj_tyx, output_path, fps=5):
    frames = []

    for t in range(max_proj_tyx.shape[0]):
        frames.append(normalize_for_video(max_proj_tyx[t]))

    imageio.mimsave(output_path, frames, fps=fps)
    print("[INFO] Saved:", output_path)


def save_before_after_images(original_tyx, corrected_tyx, time_indices, output_dir):
    output_dir = Path(output_dir)

    selected_indices = sorted(set([
        0,
        len(time_indices) // 2,
        len(time_indices) - 1
    ]))

    for i in selected_indices:
        original = normalize_for_video(original_tyx[i])
        corrected = normalize_for_video(corrected_tyx[i])

        comparison = np.concatenate([original, corrected], axis=1)

        actual_t = time_indices[i]

        output_path = output_dir / f"before_after_T{actual_t:04d}.png"
        imageio.imwrite(output_path, comparison)

        print("[INFO] Saved:", output_path)



# Main


def main():
    ensure_dir(PROJECT_DIR)
    ensure_dir(DRIFT_QC_DIR)

    print("[INFO] Starting CZI → drift-corrected TCZYX conversion")
    print("[INFO] CZI:", CZI_PATH)
    print("[INFO] Output corrected TZYX:", RAW_TZYX_TIF)

    reader = LazyCziReader(
        czi_path=CZI_PATH,
        scene_index=SCENE_INDEX
    )

    time_end = TIME_END if TIME_END is not None else reader.size_t

    time_indices = choose_indices(
        total_size=reader.size_t,
        start=TIME_START,
        end=time_end,
        stride=TIME_STRIDE
    )

    if len(time_indices) == 0:
        raise ValueError("No timepoints selected. Check TIME_START, TIME_END, TIME_STRIDE.")

    print("\n[INFO] Timepoints to process:")
    print(time_indices)

    print("\n[INFO] Z selection:")
    print(f"Z_START={Z_START}, Z_END={Z_END}")

    shifts_yx, original_max_projections = estimate_drift_lazy_robust(
        reader=reader,
        time_indices=time_indices,
        channel_index=CHANNEL_INDEX,
        upsample_factor=DRIFT_UPSAMPLE_FACTOR,
        crop_fraction=DRIFT_CROP_FRACTION,
        max_shift_per_step=DRIFT_MAX_SHIFT_PER_STEP
    )

    corrected_tczyx, uncorrected_tczyx, corrected_max_projections = apply_drift_and_build_tczyx(
        reader=reader,
        time_indices=time_indices,
        shifts_yx=shifts_yx,
    )

    print("\n[INFO] Corrected TCZYX shape:", corrected_tczyx.shape)

    print("[INFO] Saving drift-corrected TCZYX:")
    print(RAW_TCZYX_TIF)

    tiff.imwrite(
        RAW_TCZYX_TIF,
        corrected_tczyx,
        bigtiff=True,
        metadata={"axes": "TCZYX"}
    )

    if SAVE_UNCORRECTED_SUBSET_TIF:
        print("[INFO] Saving uncorrected subset TZYX:")
        print(RAW_TZYX_UNCORRECTED_TIF)

        tiff.imwrite(
            RAW_TCZYX_UNCORRECTED_TIF,
            uncorrected_tczyx,
            bigtiff=True,
            metadata={"axes": "TCZYX"}
        )

    if SAVE_DRIFT_QC:
        save_shift_csv(
            shifts_yx=shifts_yx,
            time_indices=time_indices,
            output_dir=DRIFT_QC_DIR
        )

        save_shift_plot(
            shifts_yx=shifts_yx,
            time_indices=time_indices,
            output_dir=DRIFT_QC_DIR
        )

        save_qc_video(
            original_max_projections,
            Path(DRIFT_QC_DIR) / "original_max_projection_timelapse.mp4",
            fps=5
        )

        save_qc_video(
            corrected_max_projections,
            Path(DRIFT_QC_DIR) / "corrected_max_projection_timelapse.mp4",
            fps=5
        )

        save_before_after_images(
            original_tyx=original_max_projections,
            corrected_tyx=corrected_max_projections,
            time_indices=time_indices,
            output_dir=DRIFT_QC_DIR
        )

    print("\n[DONE] Drift-corrected TCZYX stack created.")
    print("Use this file for Cellpose:")
    print(RAW_TZYX_TIF)


if __name__ == "__main__":
    main()
