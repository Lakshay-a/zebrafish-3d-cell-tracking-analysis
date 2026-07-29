import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.exposure import rescale_intensity
from skimage.io import imsave


# ============================================================
# SETTINGS
# ============================================================

RAW_FRAMES_DIR = Path(os.environ.get(
    "CELLPOSE_RAW_FRAMES_DIR",
    Path(__file__).resolve().parents[1] / "datasets" / "manual_fiji_macrophages",
))

OUTPUT_DATASET_DIR = Path(os.environ.get(
    "CELLPOSE_OUTPUT_DATASET_DIR",
    Path(__file__).resolve().parents[1] / "datasets" / "macrophage_preprocessed",
))

TARGET_SPLIT = "train"   # "train", "val", or "test"

OVERWRITE = False

# If Fiji saved an RGB image, red/mCherry is usually RGB channel 0.
RGB_CHANNEL_TO_KEEP = 1

# If OpenCV is used as fallback, it reads RGB as BGR.
# So red/mCherry becomes BGR channel 2.
CV2_RED_CHANNEL = 3

# If Fiji saved C,Y,X multichannel image, macrophage/mCherry is usually channel 1.
CYX_CHANNEL_TO_KEEP = 1

P_LOW = 0.5
P_HIGH = 99.6
GAMMA = 0.75


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_IMAGES_DIR = OUTPUT_DATASET_DIR / TARGET_SPLIT / "images"
OUTPUT_MASKS_DIR = OUTPUT_DATASET_DIR / TARGET_SPLIT / "masks"
OUTPUT_QC_DIR = OUTPUT_DATASET_DIR / "qc_png" / TARGET_SPLIT / "manual_fiji_frames"
OUTPUT_METADATA_DIR = OUTPUT_DATASET_DIR / "metadata"


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_token(text):
    text = str(text)
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def preprocess_macrophage_slice(img):
    """
    Same safe macrophage preprocessing:
    percentile clipping + gamma brightening.
    """
    img = np.asarray(img, dtype=np.float32)

    p_low, p_high = np.percentile(img, (P_LOW, P_HIGH))

    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.uint16)

    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-8)

    img = np.power(img, GAMMA)

    img = img * 65535
    img = img.astype(np.uint16)
    img = np.ascontiguousarray(img)

    return img


def make_qc_png(img_uint16):
    qc = rescale_intensity(
        img_uint16,
        in_range="image",
        out_range=np.uint8,
    ).astype(np.uint8)

    return qc


def read_tif_with_pil(path):
    """
    Reads Fiji TIFF using PIL.
    This avoids the tifffile/newbyteorder issue for many Fiji exports.
    """
    from PIL import Image, ImageSequence

    frames = []

    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            frames.append(np.array(frame))

    if len(frames) == 0:
        raise RuntimeError("PIL found no frames.")

    if len(frames) == 1:
        arr = frames[0]
    else:
        arr = np.stack(frames, axis=0)

    return arr, "pil"


def read_tif_with_cv2(path):
    """
    Reads TIFF using OpenCV fallback.
    """
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if arr is None:
        raise RuntimeError("cv2.imread returned None.")

    return arr, "cv2"


def read_tif_robust(path):
    """
    Tries multiple readers.
    Important: tifffile is tried last because your file triggers newbyteorder error.
    """
    errors = []

    try:
        arr, reader_name = read_tif_with_pil(path)
        print(f"[INFO] Read using PIL")
        return arr, reader_name
    except Exception as e:
        errors.append(f"PIL failed: {e}")

    try:
        arr, reader_name = read_tif_with_cv2(path)
        print(f"[INFO] Read using OpenCV")
        return arr, reader_name
    except Exception as e:
        errors.append(f"OpenCV failed: {e}")

    try:
        arr = tiff.imread(path)
        print(f"[INFO] Read using tifffile")
        return arr, "tifffile"
    except Exception as e:
        errors.append(f"tifffile failed: {e}")

    raise RuntimeError(
        "All TIFF readers failed:\n" + "\n".join(errors)
    )


def convert_to_2d_single_channel(img, reader_name, file_path):
    """
    Converts Fiji/ImageJ TIFF output into a simple 2D single-channel image.
    """
    img = np.asarray(img)
    original_shape = img.shape
    original_dtype = img.dtype

    img = np.squeeze(img)

    print(f"[INFO] {file_path.name}")
    print(f"       reader: {reader_name}")
    print(f"       original shape: {original_shape}, dtype: {original_dtype}")
    print(f"       squeezed shape: {img.shape}, dtype: {img.dtype}")

    # Already 2D
    if img.ndim == 2:
        img_2d = img

    # RGB/RGBA image: Y, X, 3 or Y, X, 4
    elif img.ndim == 3 and img.shape[-1] in [3, 4]:
        if reader_name == "cv2":
            channel = CV2_RED_CHANNEL
            print(f"       detected OpenCV BGR/BGRA image, keeping red channel {channel}")
        else:
            channel = RGB_CHANNEL_TO_KEEP
            print(f"       detected RGB/RGBA image, keeping red channel {channel}")

        img_2d = img[..., channel]

    # C, Y, X multichannel image
    elif img.ndim == 3 and img.shape[0] in [2, 3, 4]:
        channel = min(CYX_CHANNEL_TO_KEEP, img.shape[0] - 1)
        print(f"       detected C,Y,X image, keeping channel {channel}")
        img_2d = img[channel]

    # Multi-page TIFF / stack
    elif img.ndim == 3:
        raise ValueError(
            f"{file_path.name} still looks like a stack: shape={img.shape}\n"
            "This means Fiji probably saved multiple Z/T planes. "
            "Export only one 2D plane, or tell me which axis/plane to keep."
        )

    else:
        raise ValueError(
            f"{file_path.name} has unsupported shape after squeeze: {img.shape}"
        )

    if img_2d.ndim != 2:
        raise ValueError(f"Final image is not 2D. Final shape={img_2d.shape}")

    img_2d = np.asarray(img_2d, dtype=np.float32)
    img_2d = np.ascontiguousarray(img_2d)

    return img_2d, original_shape, original_dtype


def make_output_filename(input_path):
    stem = safe_token(input_path.stem)

    if not stem.startswith("manual_"):
        stem = "manual_" + stem

    return stem + ".tif"


def find_raw_tifs():
    tif_paths = []
    tif_paths.extend(sorted(RAW_FRAMES_DIR.glob("*.tif")))
    tif_paths.extend(sorted(RAW_FRAMES_DIR.glob("*.tiff")))
    return tif_paths


# ============================================================
# MAIN
# ============================================================

def main():
    if TARGET_SPLIT not in ["train", "val", "test"]:
        raise ValueError("TARGET_SPLIT must be 'train', 'val', or 'test'.")

    if not RAW_FRAMES_DIR.exists():
        raise RuntimeError(
            f"Raw Fiji frames folder not found:\n{RAW_FRAMES_DIR}"
        )

    ensure_dir(OUTPUT_IMAGES_DIR)
    ensure_dir(OUTPUT_MASKS_DIR)
    ensure_dir(OUTPUT_QC_DIR)
    ensure_dir(OUTPUT_METADATA_DIR)

    raw_tifs = find_raw_tifs()

    if len(raw_tifs) == 0:
        raise RuntimeError(f"No .tif/.tiff files found in:\n{RAW_FRAMES_DIR}")

    print(f"[INFO] Found {len(raw_tifs)} manual Fiji TIFF files.")
    print(f"[INFO] Target split: {TARGET_SPLIT}")
    print(f"[INFO] Output folder: {OUTPUT_IMAGES_DIR}")

    records = []
    failed = []

    for i, raw_path in enumerate(raw_tifs, start=1):
        print("\n" + "=" * 80)
        print(f"[PROCESS] {i}/{len(raw_tifs)}: {raw_path.name}")

        try:
            output_name = make_output_filename(raw_path)
            out_tif = OUTPUT_IMAGES_DIR / output_name
            out_qc = OUTPUT_QC_DIR / output_name.replace(".tif", "_qc.png")

            if out_tif.exists() and not OVERWRITE:
                print(f"[SKIP] Output already exists: {out_tif.name}")
                continue

            img, reader_name = read_tif_robust(raw_path)

            img_2d, original_shape, original_dtype = convert_to_2d_single_channel(
                img=img,
                reader_name=reader_name,
                file_path=raw_path,
            )

            processed = preprocess_macrophage_slice(img_2d)

            tiff.imwrite(
                out_tif,
                processed,
                photometric="minisblack",
                metadata=None,
            )

            qc_png = make_qc_png(processed)
            imsave(out_qc, qc_png, check_contrast=False)

            records.append({
                "raw_fiji_tif": str(raw_path),
                "reader_used": reader_name,
                "target_split": TARGET_SPLIT,
                "output_tif": str(out_tif),
                "output_qc_png": str(out_qc),
                "original_shape": str(original_shape),
                "original_dtype": str(original_dtype),
                "processed_shape": str(processed.shape),
                "processed_dtype": str(processed.dtype),
                "p_low": P_LOW,
                "p_high": P_HIGH,
                "gamma": GAMMA,
            })

            print(f"[SAVED] {out_tif}")
            print(f"[QC]    {out_qc}")

        except Exception as e:
            print(f"[ERROR] Failed: {raw_path.name}")
            print(f"[ERROR] {e}")

            failed.append({
                "raw_fiji_tif": str(raw_path),
                "error": str(e),
            })

    if len(records) > 0:
        log_path = OUTPUT_METADATA_DIR / f"manual_fiji_frames_added_{TARGET_SPLIT}.csv"
        new_log = pd.DataFrame(records)

        if log_path.exists():
            old_log = pd.read_csv(log_path)
            combined = pd.concat([old_log, new_log], ignore_index=True)
            combined = combined.drop_duplicates(subset=["output_tif"], keep="last")
            combined.to_csv(log_path, index=False)
        else:
            new_log.to_csv(log_path, index=False)

        print("\n[INFO] Saved addition log:")
        print(log_path)

    if len(failed) > 0:
        fail_path = OUTPUT_METADATA_DIR / f"manual_fiji_frames_failed_{TARGET_SPLIT}.csv"
        pd.DataFrame(failed).to_csv(fail_path, index=False)

        print("\n[WARNING] Some files failed. See:")
        print(fail_path)

    print("\n" + "=" * 80)
    print("[DONE] Manual Fiji frames prepared for Cellpose.")
    print("\nLoad fixed images from:")
    print(OUTPUT_IMAGES_DIR)
    print("\nCheck QC PNGs here:")
    print(OUTPUT_QC_DIR)


if __name__ == "__main__":
    main()
