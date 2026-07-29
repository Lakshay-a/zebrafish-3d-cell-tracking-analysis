import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.exposure import rescale_intensity
from skimage.io import imsave


# ============================================================
# PATHS
# ============================================================

INPUT_DATASET_DIR = Path(os.environ.get(
    "CELLPOSE_INPUT_DATASET_DIR",
    Path(__file__).resolve().parents[1] / "datasets" / "macrophage_initial",
))

OUTPUT_DATASET_DIR = Path(os.environ.get(
    "CELLPOSE_OUTPUT_DATASET_DIR",
    Path(__file__).resolve().parents[1] / "datasets" / "macrophage_preprocessed",
))

# Do not overwrite unless you are sure
OVERWRITE = False


# ============================================================
# PREPROCESSING SETTINGS
# ============================================================

P_LOW = 0.5
P_HIGH = 99.6
GAMMA = 0.75


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def preprocess_macrophage_slice(img):
    """
    Safe macrophage preprocessing.

    It preserves macrophage interiors and only improves visibility:
    1. percentile clipping
    2. rescaling
    3. mild gamma brightening
    """
    img = np.asarray(img, dtype=np.float32)

    p_low, p_high = np.percentile(img, (P_LOW, P_HIGH))

    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.uint16)

    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-8)

    img = np.power(img, GAMMA)

    return (img * 65535).astype(np.uint16)


def make_qc_png(img_uint16):
    """
    Creates an 8-bit PNG preview from the preprocessed TIFF.
    """
    img_uint16 = np.asarray(img_uint16)

    qc = rescale_intensity(
        img_uint16,
        in_range="image",
        out_range=np.uint8,
    ).astype(np.uint8)

    return qc


def copy_metadata():
    input_metadata = INPUT_DATASET_DIR / "metadata"
    output_metadata = OUTPUT_DATASET_DIR / "metadata"

    if input_metadata.exists():
        shutil.copytree(
            input_metadata,
            output_metadata,
            dirs_exist_ok=True,
        )
        print(f"[INFO] Copied metadata to: {output_metadata}")


def copy_masks_if_present(split):
    """
    Copies existing masks, if you already have any.
    If masks folders are empty, this will just create empty folders.
    """
    input_masks_dir = INPUT_DATASET_DIR / split / "masks"
    output_masks_dir = OUTPUT_DATASET_DIR / split / "masks"

    ensure_dir(output_masks_dir)

    if input_masks_dir.exists():
        for mask_path in input_masks_dir.iterdir():
            if mask_path.is_file():
                shutil.copy2(mask_path, output_masks_dir / mask_path.name)


def process_split(split):
    input_images_dir = INPUT_DATASET_DIR / split / "images"
    output_images_dir = OUTPUT_DATASET_DIR / split / "images"
    output_qc_dir = OUTPUT_DATASET_DIR / "qc_png" / split

    ensure_dir(output_images_dir)
    ensure_dir(output_qc_dir)

    copy_masks_if_present(split)

    image_paths = sorted(input_images_dir.glob("*.tif"))

    print(f"\n[PROCESS] {split}")
    print(f"[INFO] Found {len(image_paths)} images")

    records = []

    for i, img_path in enumerate(image_paths, start=1):
        out_tif = output_images_dir / img_path.name
        out_qc = output_qc_dir / img_path.name.replace(".tif", "_qc.png")

        if out_tif.exists() and not OVERWRITE:
            print(f"[SKIP] Already preprocessed: {out_tif.name}")
            continue

        img = tiff.imread(img_path).astype(np.float32)

        processed = preprocess_macrophage_slice(img)

        tiff.imwrite(
            out_tif,
            processed,
            photometric="minisblack",
        )

        qc_png = make_qc_png(processed)
        imsave(out_qc, qc_png, check_contrast=False)

        records.append({
            "split": split,
            "input_tif": str(img_path),
            "output_tif": str(out_tif),
            "output_qc_png": str(out_qc),
            "p_low": P_LOW,
            "p_high": P_HIGH,
            "gamma": GAMMA,
        })

        if i % 100 == 0 or i == len(image_paths):
            print(f"[INFO] {split}: processed {i}/{len(image_paths)}")

    return records


def main():
    if not INPUT_DATASET_DIR.exists():
        raise RuntimeError(f"Input dataset folder not found:\n{INPUT_DATASET_DIR}")

    if OUTPUT_DATASET_DIR.exists():
        print(f"[INFO] Output dataset folder already exists:\n{OUTPUT_DATASET_DIR}")
        print("[INFO] Existing preprocessed images will be skipped.")
    else:
        ensure_dir(OUTPUT_DATASET_DIR)

    all_records = []

    for split in ["train", "val", "test"]:
        records = process_split(split)
        all_records.extend(records)

    copy_metadata()

    preprocessing_log = pd.DataFrame(all_records)
    ensure_dir(OUTPUT_DATASET_DIR / "metadata")

    log_path = OUTPUT_DATASET_DIR / "metadata" / "preprocessing_log.csv"

    if log_path.exists() and len(preprocessing_log) > 0:
        old_log = pd.read_csv(log_path)
        combined_log = pd.concat([old_log, preprocessing_log], ignore_index=True)

        if "output_tif" in combined_log.columns:
            combined_log = combined_log.drop_duplicates(
                subset=["output_tif"],
                keep="last",
            )

        combined_log.to_csv(log_path, index=False)

    elif len(preprocessing_log) > 0:
        preprocessing_log.to_csv(log_path, index=False)

    else:
        print("[INFO] No new images were preprocessed.")

    print("\n" + "=" * 80)
    print("[DONE] Created preprocessed macrophage dataset")
    print(f"Input dataset:  {INPUT_DATASET_DIR}")
    print(f"Output dataset: {OUTPUT_DATASET_DIR}")

    print("\nImages saved in:")
    print(OUTPUT_DATASET_DIR / "train" / "images")
    print(OUTPUT_DATASET_DIR / "val" / "images")
    print(OUTPUT_DATASET_DIR / "test" / "images")

    print("\nQC PNGs saved in:")
    print(OUTPUT_DATASET_DIR / "qc_png")

    print("\nPreprocessing log:")
    print(OUTPUT_DATASET_DIR / "metadata" / "preprocessing_log.csv")


if __name__ == "__main__":
    main()
