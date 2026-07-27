import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndi_shift
from skimage.segmentation import find_boundaries

from config import (
    CELL_TYPE,
    CHANNEL_INDEX,
    RAW_TCZYX_TIF,
    CELLPOSE_MASKS_TZYX_TIF,
    PROJECT_DIR,
)

# Test a few representative slices
TIMEPOINTS = [10, 23, 40, 60, 80]
Z_SLICES = [12, 16, 20, 24, 28, 32]

MAX_SHIFT = 5  # test shifts from -5 to +5 pixels
OUT_DIR = PROJECT_DIR / "mask_shift_debug"
OUT_DIR.mkdir(exist_ok=True)


def load_raw_selected_channel():
    raw = tiff.imread(RAW_TCZYX_TIF)

    if raw.ndim == 5:
        return raw[:, CHANNEL_INDEX, :, :, :]
    elif raw.ndim == 4:
        return raw
    else:
        raise ValueError(f"Unexpected raw shape: {raw.shape}")


def normalize(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99.8])
    return np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)


def score_mask_alignment(raw_2d, mask_2d):
    """Score = mean raw intensity inside mask minus mean intensity just outside mask."""
    mask_bin = mask_2d > 0

    if mask_bin.sum() == 0:
        return np.nan

    inside = raw_2d[mask_bin].mean()

    # Simple background estimate from non-mask pixels
    outside = raw_2d[~mask_bin].mean()

    return float(inside - outside)


def shift_mask(mask_2d, dy, dx):
    shifted = ndi_shift(
        mask_2d.astype(np.uint16),
        shift=(dy, dx),
        order=0,          # nearest neighbour, preserves labels
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return shifted.astype(mask_2d.dtype)


def save_overlay(raw_2d, mask_2d, out_path, title):
    raw_disp = normalize(raw_2d)
    mask_bin = mask_2d > 0
    boundaries = find_boundaries(mask_bin, mode="outer")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(raw_disp, cmap="gray")

    fill = np.ma.masked_where(~mask_bin, mask_bin)
    ax.imshow(fill, cmap="spring", alpha=0.25)

    edge = np.ma.masked_where(~boundaries, boundaries)
    ax.imshow(edge, cmap="autumn", alpha=1.0)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    print("=" * 70)
    print("MASK XY SHIFT CHECK")
    print("=" * 70)

    raw = load_raw_selected_channel()
    masks = tiff.imread(CELLPOSE_MASKS_TZYX_TIF)

    print("Raw shape:", raw.shape)
    print("Mask shape:", masks.shape)

    if raw.shape != masks.shape:
        raise ValueError(f"Shape mismatch: raw={raw.shape}, masks={masks.shape}")

    rows = []

    for t in TIMEPOINTS:
        for z in Z_SLICES:
            if t >= raw.shape[0] or z >= raw.shape[1]:
                continue

            raw_2d = raw[t, z]
            mask_2d = masks[t, z]

            if (mask_2d > 0).sum() == 0:
                continue

            best_score = -np.inf
            best_dy = 0
            best_dx = 0

            original_score = score_mask_alignment(raw_2d, mask_2d)

            for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
                for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
                    shifted = shift_mask(mask_2d, dy, dx)
                    score = score_mask_alignment(raw_2d, shifted)

                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_dy = dy
                        best_dx = dx

            rows.append(
                {
                    "time": t,
                    "z": z,
                    "original_score": original_score,
                    "best_score": best_score,
                    "best_dy": best_dy,
                    "best_dx": best_dx,
                    "score_gain": best_score - original_score,
                    "mask_pixels": int((mask_2d > 0).sum()),
                    "n_objects": int(len(np.unique(mask_2d)) - 1),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"{CELL_TYPE}_mask_xy_shift_scores.csv"
    df.to_csv(out_csv, index=False)

    print(df)
    print()
    print("Median best dy:", df["best_dy"].median())
    print("Median best dx:", df["best_dx"].median())
    print("Mean score gain:", df["score_gain"].mean())
    print("[SAVED]", out_csv)

    # Save example overlay for the first tested slice
    example = df.sort_values("score_gain", ascending=False).iloc[0]
    t = int(example["time"])
    z = int(example["z"])
    dy = int(example["best_dy"])
    dx = int(example["best_dx"])

    raw_2d = raw[t, z]
    mask_2d = masks[t, z]
    shifted = shift_mask(mask_2d, dy, dx)

    save_overlay(
        raw_2d,
        mask_2d,
        OUT_DIR / f"{CELL_TYPE}_original_overlay_T{t:04d}_Z{z:04d}.png",
        f"Original mask | T={t}, Z={z}",
    )

    save_overlay(
        raw_2d,
        shifted,
        OUT_DIR / f"{CELL_TYPE}_best_shift_overlay_T{t:04d}_Z{z:04d}_dy{dy}_dx{dx}.png",
        f"Best shifted mask | T={t}, Z={z}, dy={dy}, dx={dx}",
    )


if __name__ == "__main__":
    main()