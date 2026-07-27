from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    PROJECT_DIR,
    CELLPOSE_MASKS_TZYX_TIF,
    RAW_TCZYX_TIF,
    RAW_TCZYX_UNCORRECTED_TIF,
)

TIME_INDEX = 23
Z_INDEX = 16

OUT_DIR = PROJECT_DIR / "alignment_debug"
OUT_DIR.mkdir(exist_ok=True)


def load_channel(path):
    raw = tiff.imread(path)
    print(f"{path.name}: {raw.shape}")

    if raw.ndim == 5:
        return raw[:, CHANNEL_INDEX, :, :, :]
    elif raw.ndim == 4:
        return raw
    else:
        raise ValueError(f"Unexpected raw shape: {raw.shape}")


def norm(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99.8])
    return np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)


def save_overlay(raw_2d, mask_2d, title, out_path):
    mask_bin = mask_2d > 0
    boundary = find_boundaries(mask_bin, mode="outer")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(norm(raw_2d), cmap="gray")

    fill = np.ma.masked_where(~mask_bin, mask_bin)
    ax.imshow(fill, cmap="spring", alpha=0.25)

    edge = np.ma.masked_where(~boundary, boundary)
    ax.imshow(edge, cmap="autumn", alpha=1.0)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("[SAVED]", out_path)


def main():
    print("CELL_TYPE:", CELL_TYPE)
    print("CHANNEL_INDEX:", CHANNEL_INDEX)
    print("TIME_INDEX:", TIME_INDEX)
    print("Z_INDEX:", Z_INDEX)

    mask = tiff.imread(CELLPOSE_MASKS_TZYX_TIF)

    raw_corrected = load_channel(RAW_TCZYX_TIF)
    raw_uncorrected = load_channel(RAW_TCZYX_UNCORRECTED_TIF)

    print("mask:", mask.shape)
    print("corrected:", raw_corrected.shape)
    print("uncorrected:", raw_uncorrected.shape)

    if mask.shape != raw_corrected.shape:
        print("[WARNING] mask and corrected raw shapes differ")

    if mask.shape != raw_uncorrected.shape:
        print("[WARNING] mask and uncorrected raw shapes differ")

    mask_2d = mask[TIME_INDEX, Z_INDEX]

    save_overlay(
        raw_corrected[TIME_INDEX, Z_INDEX],
        mask_2d,
        f"Mask on DRIFT-CORRECTED raw | T={TIME_INDEX}, Z={Z_INDEX}",
        OUT_DIR / f"{CELL_TYPE}_mask_on_corrected_T{TIME_INDEX:04d}_Z{Z_INDEX:04d}.png",
    )

    save_overlay(
        raw_uncorrected[TIME_INDEX, Z_INDEX],
        mask_2d,
        f"Mask on UNCORRECTED raw | T={TIME_INDEX}, Z={Z_INDEX}",
        OUT_DIR / f"{CELL_TYPE}_mask_on_uncorrected_T{TIME_INDEX:04d}_Z{Z_INDEX:04d}.png",
    )


if __name__ == "__main__":
    main()