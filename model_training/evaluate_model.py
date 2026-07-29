import os
from pathlib import Path
import inspect
import json

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from skimage.measure import label
from skimage.segmentation import find_boundaries
from scipy.optimize import linear_sum_assignment

import torch
from cellpose import models


# ============================================================
# USER SETTINGS
# ============================================================

PROJECT_ROOT = Path(os.environ.get("CELLPOSE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATASET_DIR = PROJECT_ROOT / "macrophage_cellpose_dataset_preprocessed"
# DATASET_DIR = PROJECT_ROOT / "cellpose_dataset_final"

TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
TEST_MASKS_DIR = DATASET_DIR / "test" / "masks"

BEST_MODEL_PATH = Path(os.environ.get(
    "CELLPOSE_MODEL_PATH",
    Path(__file__).resolve().parents[1] / "models" / "cellpose_model",
))

EVAL_OUTPUT_DIR = DATASET_DIR / "evaluation_outputs"

PRED_MASK_DIR = EVAL_OUTPUT_DIR / "predicted_masks"
OVERLAY_DIR = EVAL_OUTPUT_DIR / "overlays"
METRICS_DIR = EVAL_OUTPUT_DIR / "metrics"

PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INFERENCE SETTINGS
# ============================================================

INFERENCE_SETTINGS = [
    {
        "name": "auto_cp0_flow04",
        "diameter": None,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
    },
    {
        "name": "d30_cp0_flow04",
        "diameter": 30,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
    },
    # {
    #     "name": "d50_cp0_flow04",
    #     "diameter": 50,
    #     "cellprob_threshold": 0.0,
    #     "flow_threshold": 0.4,
    # },
    {
        "name": "d30_cpm05_flow04",
        "diameter": 30,
        "cellprob_threshold": -0.5,
        "flow_threshold": 0.4,
    },
    # {
    #     "name": "d50_cpm05_flow04",
    #     "diameter": 50,
    #     "cellprob_threshold": -0.5,
    #     "flow_threshold": 0.4,
    # },
    {
        "name": "d30_cpm05_flow08",
        "diameter": 30,
        "cellprob_threshold": -0.5,
        "flow_threshold": 0.8,
    },
]

# A predicted object is counted as a true positive if IoU with a GT object >= this.
OBJECT_IOU_THRESHOLD = 0.35

USE_GPU = torch.backends.mps.is_available() or torch.cuda.is_available()


# ============================================================
# DATA COLLECTION
# ============================================================

def is_real_image(path: Path):
    """
    Returns True only for real input images.

    Excludes:
    - manual masks
    - Cellpose flows
    - outlines
    - predicted masks
    """
    name = path.name

    if path.suffix.lower() not in [".tif", ".tiff"]:
        return False

    excluded_suffixes = [
        "_masks.tif",
        "_masks.tiff",
        "_flows.tif",
        "_flows.tiff",
        "_outlines.tif",
        "_outlines.tiff",
        "_cp_masks.tif",
        "_cp_masks.tiff",
        "_pred_masks.tif",
        "_pred_masks.tiff",
    ]

    return not any(name.endswith(suffix) for suffix in excluded_suffixes)


def find_test_mask_for_image(image_path: Path):
    """
    Finds the matching manual mask for a test image.

    Supports both structures:

    1. Mask saved beside image:
       test/images/sample.tif
       test/images/sample_masks.tif

    2. Mask saved in separate masks folder:
       test/images/sample.tif
       test/masks/sample_masks.tif
    """
    mask_name = image_path.stem + "_masks.tif"

    possible_paths = [
        image_path.with_name(mask_name),
        TEST_MASKS_DIR / mask_name,
    ]

    for mask_path in possible_paths:
        if mask_path.exists():
            return mask_path

    return None


def collect_test_pairs():
    """
    Collects only test images that have a matching manual test mask.

    Images without masks are skipped completely and are not included
    in prediction, overlays, or metric calculation.
    """
    image_paths = []

    for path in sorted(TEST_IMAGES_DIR.glob("*.tif")):
        if is_real_image(path):
            image_paths.append(path)

    pairs = []
    missing_mask_images = []

    for image_path in image_paths:
        mask_path = find_test_mask_for_image(image_path)

        if mask_path is None:
            missing_mask_images.append(image_path.name)
            continue

        pairs.append((image_path, mask_path))

    print("\n" + "=" * 80)
    print("[TEST DATA]")
    print("=" * 80)
    print(f"Test images folder: {TEST_IMAGES_DIR}")
    print(f"Test masks folder:  {TEST_MASKS_DIR}")
    print(f"Total real test images found: {len(image_paths)}")
    print(f"Images with matching masks:   {len(pairs)}")
    print(f"Images skipped, no mask:      {len(missing_mask_images)}")

    if missing_mask_images:
        skipped_csv = METRICS_DIR / "skipped_test_images_no_mask.csv"

        pd.DataFrame(
            {
                "skipped_image_name": missing_mask_images,
            }
        ).to_csv(skipped_csv, index=False)

        print(f"Skipped-image list saved to: {skipped_csv}")

    if not pairs:
        raise RuntimeError(
            "No image-mask pairs found.\n"
            f"Checked image folder: {TEST_IMAGES_DIR}\n"
            f"Checked mask folder:  {TEST_MASKS_DIR}"
        )

    return pairs


# ============================================================
# MODEL EVALUATION
# ============================================================

def filter_eval_kwargs(model, kwargs):
    """
    Cellpose versions differ slightly.
    This keeps only arguments accepted by the installed model.eval().
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


def run_model(model, image, setting):
    raw_kwargs = {
        "x": image,
        "channels": [0, 0],
        "diameter": setting["diameter"],
        "cellprob_threshold": setting["cellprob_threshold"],
        "flow_threshold": setting["flow_threshold"],
    }

    eval_kwargs = filter_eval_kwargs(model, raw_kwargs)
    result = model.eval(**eval_kwargs)

    pred_mask = result[0]

    return pred_mask


# ============================================================
# METRICS
# ============================================================

def binary_iou_and_dice(gt_mask, pred_mask):
    """
    Pixel-level binary IoU and Dice.

    Any labelled object > 0 is treated as foreground.
    """
    gt = gt_mask > 0
    pred = pred_mask > 0

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    gt_sum = gt.sum()
    pred_sum = pred.sum()

    iou = intersection / union if union > 0 else np.nan

    if (gt_sum + pred_sum) > 0:
        dice = (2 * intersection) / (gt_sum + pred_sum)
    else:
        dice = np.nan

    return float(iou), float(dice)


def compute_object_iou_matrix(gt_labels, pred_labels):
    """
    Computes object-level IoU matrix between GT and predicted objects.
    """
    gt_ids = np.array([x for x in np.unique(gt_labels) if x != 0])
    pred_ids = np.array([x for x in np.unique(pred_labels) if x != 0])

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return gt_ids, pred_ids, iou_matrix

    for i, gt_id in enumerate(gt_ids):
        gt_obj = gt_labels == gt_id
        gt_area = gt_obj.sum()

        for j, pred_id in enumerate(pred_ids):
            pred_obj = pred_labels == pred_id
            pred_area = pred_obj.sum()

            intersection = np.logical_and(gt_obj, pred_obj).sum()
            union = gt_area + pred_area - intersection

            if union > 0:
                iou_matrix[i, j] = intersection / union

    return gt_ids, pred_ids, iou_matrix


# Assignment solver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
def object_precision_recall_f1(gt_mask, pred_mask, iou_threshold=0.35):
    """
    Object-level precision, recall and F1 using Hungarian matching.
    """
    gt_labels = label(gt_mask > 0)
    pred_labels = label(pred_mask > 0)

    gt_ids, pred_ids, iou_matrix = compute_object_iou_matrix(
        gt_labels,
        pred_labels,
    )

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    if n_gt == 0 and n_pred == 0:
        return {
            "n_gt_objects": 0,
            "n_pred_objects": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mean_matched_object_iou": np.nan,
        }

    if n_gt == 0:
        return {
            "n_gt_objects": 0,
            "n_pred_objects": n_pred,
            "true_positives": 0,
            "false_positives": n_pred,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": np.nan,
            "f1": 0.0,
            "mean_matched_object_iou": np.nan,
        }

    if n_pred == 0:
        return {
            "n_gt_objects": n_gt,
            "n_pred_objects": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": n_gt,
            "precision": np.nan,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_object_iou": np.nan,
        }

    # Hungarian matching maximises IoU by minimising negative IoU.
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_ious = []

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched_ious.append(float(iou_matrix[r, c]))

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / n_pred if n_pred > 0 else np.nan
    recall = tp / n_gt if n_gt > 0 else np.nan

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else np.nan

    return {
        "n_gt_objects": int(n_gt),
        "n_pred_objects": int(n_pred),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "recall": float(recall) if not np.isnan(recall) else np.nan,
        "f1": float(f1),
        "mean_matched_object_iou": mean_iou,
    }


# ============================================================
# OVERLAY CREATION
# ============================================================

def normalize_for_display(image, p_low=1, p_high=99.8):
    image = image.astype(np.float32)

    lo, hi = np.percentile(image, (p_low, p_high))

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)

    return image.astype(np.float32)


def make_overlay(image, gt_mask, pred_mask):
    """
    Creates RGB overlay:
    - GT only = green
    - prediction only = red
    - overlap between GT and prediction = yellow
    """
    img = normalize_for_display(image)
    rgb = np.dstack([img, img, img])

    gt_boundary = find_boundaries(gt_mask > 0, mode="outer")
    pred_boundary = find_boundaries(pred_mask > 0, mode="outer")

    overlap = gt_boundary & pred_boundary
    gt_only = gt_boundary & ~pred_boundary
    pred_only = pred_boundary & ~gt_boundary

    # GT only = green
    rgb[gt_only, 0] = 0.0
    rgb[gt_only, 1] = 1.0
    rgb[gt_only, 2] = 0.0

    # Prediction only = red
    rgb[pred_only, 0] = 1.0
    rgb[pred_only, 1] = 0.0
    rgb[pred_only, 2] = 0.0

    # Overlap = yellow
    rgb[overlap, 0] = 1.0
    rgb[overlap, 1] = 1.0
    rgb[overlap, 2] = 0.0

    return rgb


def save_overlay(image_path, image, gt_mask, pred_mask, setting_name):
    overlay = make_overlay(image, gt_mask, pred_mask)

    setting_overlay_dir = OVERLAY_DIR / setting_name
    setting_overlay_dir.mkdir(parents=True, exist_ok=True)

    out_path = setting_overlay_dir / f"{image_path.stem}_{setting_name}_overlay.png"

    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.title(f"{image_path.stem} | {setting_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# SAVING PREDICTIONS
# ============================================================

def save_predicted_mask(image_path, pred_mask, setting_name):
    setting_pred_dir = PRED_MASK_DIR / setting_name
    setting_pred_dir.mkdir(parents=True, exist_ok=True)

    out_path = setting_pred_dir / f"{image_path.stem}_{setting_name}_pred_masks.tif"

    if pred_mask.max() <= 65535:
        pred_to_save = pred_mask.astype(np.uint16)
    else:
        pred_to_save = pred_mask.astype(np.uint32)

    tifffile.imwrite(out_path, pred_to_save)

    return out_path


# ============================================================
# SUMMARY HELPERS
# ============================================================

def make_summary(df):
    """
    Creates summary metrics by inference setting.
    """
    summary = df.groupby("setting").agg(
        {
            "pixel_iou": "mean",
            "dice": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "mean_matched_object_iou": "mean",
            "n_gt_objects": "sum",
            "n_pred_objects": "sum",
            "true_positives": "sum",
            "false_positives": "sum",
            "false_negatives": "sum",
        }
    ).reset_index()

    return summary


def save_evaluation_config(pairs):
    config = {
        "project_root": str(PROJECT_ROOT),
        "dataset_dir": str(DATASET_DIR),
        "test_images_dir": str(TEST_IMAGES_DIR),
        "test_masks_dir": str(TEST_MASKS_DIR),
        "best_model_path": str(BEST_MODEL_PATH),
        "eval_output_dir": str(EVAL_OUTPUT_DIR),
        "object_iou_threshold": OBJECT_IOU_THRESHOLD,
        "use_gpu": USE_GPU,
        "n_test_pairs_evaluated": len(pairs),
        "inference_settings": INFERENCE_SETTINGS,
    }

    config_path = METRICS_DIR / "evaluation_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Evaluation config saved to: {config_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Evaluating trained Cellpose model on manually annotated test set")
    print("=" * 80)
    print(f"Test images folder: {TEST_IMAGES_DIR}")
    print(f"Test masks folder:  {TEST_MASKS_DIR}")
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"GPU flag: {USE_GPU}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")

    pairs = collect_test_pairs()
    save_evaluation_config(pairs)

    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model=str(BEST_MODEL_PATH),
    )

    all_records = []

    for setting in INFERENCE_SETTINGS:
        setting_name = setting["name"]

        print("\n" + "=" * 80)
        print(f"Evaluating setting: {setting_name}")
        print(setting)
        print("=" * 80)

        for image_path, gt_mask_path in pairs:
            image = tifffile.imread(image_path)
            gt_mask = tifffile.imread(gt_mask_path)

            if image.shape != gt_mask.shape:
                print(f"[SKIP] Shape mismatch for {image_path.name}")
                print(f"       image: {image.shape}")
                print(f"       mask:  {gt_mask.shape}")
                continue

            pred_mask = run_model(model, image, setting)

            pred_mask_path = save_predicted_mask(
                image_path=image_path,
                pred_mask=pred_mask,
                setting_name=setting_name,
            )

            pixel_iou, dice = binary_iou_and_dice(gt_mask, pred_mask)

            obj_metrics = object_precision_recall_f1(
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                iou_threshold=OBJECT_IOU_THRESHOLD,
            )

            save_overlay(
                image_path=image_path,
                image=image,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                setting_name=setting_name,
            )

            record = {
                "setting": setting_name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "gt_mask_path": str(gt_mask_path),
                "pred_mask_path": str(pred_mask_path),
                "pixel_iou": pixel_iou,
                "dice": dice,
                **obj_metrics,
            }

            all_records.append(record)

            precision = obj_metrics["precision"]
            recall = obj_metrics["recall"]
            f1 = obj_metrics["f1"]

            print(
                f"{image_path.name} | "
                f"IoU={pixel_iou:.3f} | "
                f"Dice={dice:.3f} | "
                f"Precision={precision:.3f} | "
                f"Recall={recall:.3f} | "
                f"F1={f1:.3f}"
            )

    df = pd.DataFrame(all_records)

    detailed_csv = METRICS_DIR / "detailed_metrics.csv"
    summary_csv = METRICS_DIR / "summary_by_setting.csv"

    df.to_csv(detailed_csv, index=False)

    summary = make_summary(df)
    summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print("=" * 80)
    print(f"Detailed metrics: {detailed_csv}")
    print(f"Summary metrics:  {summary_csv}")
    print(f"Predicted masks:  {PRED_MASK_DIR}")
    print(f"Overlays:         {OVERLAY_DIR}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
