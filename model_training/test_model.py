import os
from pathlib import Path
import inspect

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

import torch
from cellpose import models


# ============================================================
# USER SETTINGS
# ============================================================

PROJECT_ROOT = Path(os.environ.get("CELLPOSE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATASET_DIR = PROJECT_ROOT / "cellpose_dataset_final"

VAL_DIR = DATASET_DIR / "test" / "images"

BEST_MODEL_PATH = Path(os.environ.get(
    "CELLPOSE_MODEL_PATH",
    Path(__file__).resolve().parents[1] / "models" / "cellpose_model",
))

OUTPUT_DIR = DATASET_DIR / "model_test_outputs"
PRED_MASK_DIR = OUTPUT_DIR / "predicted_masks"
OVERLAY_DIR = OUTPUT_DIR / "prediction_overlays"

PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# For first test, keep these conservative
DIAMETER = 30
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

# Use MPS if available, otherwise CPU
USE_GPU = torch.backends.mps.is_available() or torch.cuda.is_available()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_real_image(path: Path):
    name = path.name

    if not name.endswith(".tif"):
        return False

    excluded = [
        "_masks.tif",
        "_flows.tif",
        "_outlines.tif",
        "_cp_masks.tif",
    ]

    return not any(name.endswith(suffix) for suffix in excluded)


def collect_val_images():
    image_paths = []

    for path in sorted(VAL_DIR.glob("*.tif")):
        if is_real_image(path):
            image_paths.append(path)

    if not image_paths:
        raise RuntimeError(f"No validation images found in: {VAL_DIR}")

    return image_paths


def normalize_for_display(image, p_low=1, p_high=99.8):
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (p_low, p_high))
    image = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)
    return image


def filter_eval_kwargs(model, kwargs):
    """
    Cellpose versions differ. This keeps only eval arguments supported
    by the installed model.eval() function.
    """
    sig = inspect.signature(model.eval)
    accepted = set(sig.parameters.keys())

    filtered = {}

    for key, value in kwargs.items():
        if key in accepted:
            filtered[key] = value
        else:
            print(f"[INFO] model.eval does not accept argument: {key}")

    return filtered


def run_model_on_image(model, image):
    """
    Run Cellpose model on one 2D grayscale image.
    """
    raw_kwargs = {
        "x": image,
        "channels": [0, 0],
        "diameter": DIAMETER,
        "flow_threshold": FLOW_THRESHOLD,
        "cellprob_threshold": CELLPROB_THRESHOLD,
    }

    eval_kwargs = filter_eval_kwargs(model, raw_kwargs)

    result = model.eval(**eval_kwargs)

    # Depending on Cellpose version, output can be:
    # masks, flows, styles
    # or masks, flows, styles, diams
    masks = result[0]

    return masks


def create_overlay(image, gt_mask, pred_mask):
    """
    Overlay:
    - grayscale image
    - green boundaries = manual/ground truth mask
    - red boundaries = predicted mask
    - yellow-ish overlap where both boundaries coincide
    """
    img = normalize_for_display(image)

    rgb = np.dstack([img, img, img])

    if gt_mask is not None:
        gt_boundary = find_boundaries(gt_mask > 0, mode="outer")
        rgb[gt_boundary, 0] = 0.0
        rgb[gt_boundary, 1] = 1.0
        rgb[gt_boundary, 2] = 0.0

    pred_boundary = find_boundaries(pred_mask > 0, mode="outer")
    rgb[pred_boundary, 0] = 1.0
    rgb[pred_boundary, 1] = 0.0
    rgb[pred_boundary, 2] = 0.0

    return rgb


def save_overlay(image_path, image, gt_mask, pred_mask):
    overlay = create_overlay(image, gt_mask, pred_mask)

    out_path = OVERLAY_DIR / f"{image_path.stem}_prediction_overlay.png"

    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.title(image_path.stem)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Testing trained custom Cellpose model")
    print("=" * 80)
    print(f"Best model path: {BEST_MODEL_PATH}")
    print(f"Validation folder: {VAL_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Using GPU flag: {USE_GPU}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")

    image_paths = collect_val_images()
    print(f"Found validation images: {len(image_paths)}")

    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model=str(BEST_MODEL_PATH),
    )

    for image_path in image_paths:
        print(f"\nProcessing: {image_path.name}")

        image = tifffile.imread(image_path)

        gt_mask_path = image_path.with_name(image_path.stem + "_masks.tif")

        if gt_mask_path.exists():
            gt_mask = tifffile.imread(gt_mask_path)
        else:
            gt_mask = None

        pred_mask = run_model_on_image(model, image)

        pred_mask_path = PRED_MASK_DIR / f"{image_path.stem}_pred_masks.tif"
        tifffile.imwrite(pred_mask_path, pred_mask.astype(np.uint16))

        overlay_path = save_overlay(
            image_path=image_path,
            image=image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
        )

        print(f"Saved predicted mask: {pred_mask_path.name}")
        print(f"Saved overlay: {overlay_path.name}")

    print("\nDone.")
    print(f"Check overlays here:\n{OVERLAY_DIR}")


if __name__ == "__main__":
    main()
