import os
from pathlib import Path
import shutil
import json
import inspect
import time

import numpy as np
import pandas as pd
import tifffile
import torch

from cellpose import models, train


# ============================================================
# USER SETTINGS
# ============================================================

DATASET_DIR = Path(os.environ.get(
    "CELLPOSE_DATASET_DIR",
    Path(__file__).resolve().parents[1] / "datasets" / "cellpose_dataset",
))

TRAIN_DIR = DATASET_DIR / "train" / "images"
VAL_DIR = DATASET_DIR / "val" / "images"

RUNS_DIR = DATASET_DIR / "macrophage_annotation_helper_training_runs"

EXPERIMENT_NAME = "macrophage_annotation_helper_v1"

# Existing pilot model trained on 38 slices
# PRETRAINED_MODEL = Path(
# )

PRETRAINED_MODEL = 'cpsam'

# For grayscale GFP data
CHANNEL_AXIS = None
USE_GPU = torch.backends.mps.is_available()

# Fine-tuning hyperparameters
TOTAL_EPOCHS = 250
EPOCHS_PER_BLOCK = 10 # 5

BATCH_SIZE = 1
LEARNING_RATE = 1e-5 # 5e-5
WEIGHT_DECAY = 1e-3

BSIZE = 256
MIN_TRAIN_MASKS = 1

EARLY_STOPPING_PATIENCE = 4 # 6
MIN_DELTA = 1e-4

SAVE_EVERY = EPOCHS_PER_BLOCK
NORMALIZE = True
COMPUTE_FLOWS = False

MASK_SUFFIX = "_masks.tif"


# ============================================================
# FILE HELPERS
# ============================================================

def is_real_training_image(path: Path):
    name = path.name

    if not name.endswith(".tif"):
        return False

    excluded_suffixes = [
        "_masks.tif",
        "_flows.tif",
        "_outlines.tif",
        "_cp_masks.tif",
        "_pred_masks.tif",
    ]

    return not any(name.endswith(suffix) for suffix in excluded_suffixes)


def image_to_mask_path(image_path: Path):
    return image_path.with_name(image_path.stem + MASK_SUFFIX)


def convert_seg_npy_to_masks(folder: Path):
    """
    Converts Cellpose GUI *_seg.npy files to *_masks.tif.
    Does not move, delete, or rename *_seg.npy files.
    Only creates *_masks.tif next to the image.
    """
    seg_files = sorted(folder.glob("*_seg.npy"))

    print(f"\n[CONVERT] Folder: {folder}")
    print(f"[CONVERT] Found *_seg.npy files: {len(seg_files)}")

    converted = 0
    skipped = 0
    failed = 0

    for seg_file in seg_files:
        try:
            base_stem = seg_file.name.replace("_seg.npy", "")
            mask_file = folder / f"{base_stem}_masks.tif"

            if mask_file.exists():
                skipped += 1
                continue

            seg = np.load(seg_file, allow_pickle=True).item()

            if "masks" not in seg:
                print(f"[WARNING] No 'masks' key in: {seg_file.name}")
                failed += 1
                continue

            masks = np.asarray(seg["masks"])

            if masks.ndim != 2:
                print(f"[WARNING] Expected 2D masks, got {masks.shape}: {seg_file.name}")
                failed += 1
                continue

            if masks.max() <= 65535:
                masks_to_save = masks.astype(np.uint16)
            else:
                masks_to_save = masks.astype(np.uint32)

            tifffile.imwrite(mask_file, masks_to_save)
            converted += 1

        except Exception as e:
            print(f"[ERROR] Failed converting {seg_file.name}: {e}")
            failed += 1

    print(f"[CONVERT] Converted: {converted}")
    print(f"[CONVERT] Skipped existing: {skipped}")
    print(f"[CONVERT] Failed: {failed}")


def collect_image_mask_pairs(folder: Path):
    image_files = []

    for path in sorted(folder.glob("*.tif")):
        if is_real_training_image(path):
            image_files.append(path)

    image_paths = []
    mask_paths = []

    for image_path in image_files:
        mask_path = image_to_mask_path(image_path)

        if not mask_path.exists():
            continue

        image_paths.append(str(image_path))
        mask_paths.append(str(mask_path))

    return image_paths, mask_paths


def check_pairs_are_valid(image_paths, mask_paths, name="dataset"):
    print(f"\n[CHECK] Checking {name} pairs...")
    print("-" * 60)

    valid_images = []
    valid_masks = []
    records = []

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)

        if image.shape != mask.shape:
            print(f"[SKIP] Shape mismatch: {Path(image_path).name}")
            print(f"       image: {image.shape}, mask: {mask.shape}")
            continue

        max_label = int(mask.max())

        if max_label < 1:
            print(f"[SKIP] Empty mask: {Path(mask_path).name}")
            continue

        valid_images.append(image_path)
        valid_masks.append(mask_path)

        records.append({
            "image": image_path,
            "mask": mask_path,
            "n_objects": max_label,
            "shape_y": int(image.shape[0]),
            "shape_x": int(image.shape[1]),
        })

    print(f"[CHECK] Valid {name} pairs: {len(valid_images)}")

    if len(valid_images) == 0:
        raise RuntimeError(f"No valid image-mask pairs found for {name}.")

    return valid_images, valid_masks, pd.DataFrame(records)


# ============================================================
# TRAINING HELPERS
# ============================================================

def get_last_nonzero_loss(losses):
    """
    Cellpose validation loss arrays can contain zero entries for epochs
    where validation was not evaluated. Use the last non-zero finite value.
    """
    if losses is None:
        return None

    arr = np.asarray(losses).reshape(-1)

    if arr.size == 0:
        return None

    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return None

    nonzero = arr[arr > 0]

    if nonzero.size == 0:
        return None

    return float(nonzero[-1])


def filter_kwargs_for_train_seg(kwargs):
    """
    Cellpose versions differ slightly. This keeps only arguments accepted
    by the installed train.train_seg.
    """
    sig = inspect.signature(train.train_seg)
    accepted = set(sig.parameters.keys())

    filtered = {}

    for key, value in kwargs.items():
        if key in accepted:
            filtered[key] = value
        else:
            print(f"[INFO] train_seg in this Cellpose version does not accept: {key}")

    return filtered


def make_model(pretrained_model):
    print(f"\n[MODEL] Loading model from: {pretrained_model}")

    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model=str(pretrained_model),
    )

    return model


def copy_best_model(model_path, best_model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"[WARNING] Model path does not exist, cannot copy: {model_path}")
        return False

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, best_model_path)

    print(f"[BEST] Copied best model to: {best_model_path}")
    return True


def save_training_log(log_path, log_data):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)


# ============================================================
# MAIN TRAINING
# ============================================================

# Adapted from: https://cellpose.readthedocs.io/en/latest/train.html
def train_annotation_helper_with_safe_reloading():
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    run_dir = RUNS_DIR / f"{EXPERIMENT_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model" / EXPERIMENT_NAME
    log_path = run_dir / "training_log.json"

    print("\n" + "=" * 80)
    print("Annotation-helper Cellpose fine-tuning")
    print("=" * 80)
    print(f"Dataset dir: {DATASET_DIR}")
    print(f"Train dir:   {TRAIN_DIR}")
    print(f"Val dir:     {VAL_DIR}")
    print(f"Run dir:     {run_dir}")
    print(f"Starting model: {PRETRAINED_MODEL}")
    print(f"GPU/MPS enabled: {USE_GPU}")
    print("Torch version:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    # if not Path(PRETRAINED_MODEL).exists():
    #     raise FileNotFoundError(f"Pilot model not found:\n{PRETRAINED_MODEL}")

    def is_builtin_cellpose_model(model_source):
        return str(model_source).lower() in [
            "cpsam",
            "cyto",
            "cyto2",
            "cyto3",
            "nuclei",
        ]


    if not is_builtin_cellpose_model(PRETRAINED_MODEL):
        if not Path(PRETRAINED_MODEL).exists():
            raise FileNotFoundError(f"Pretrained model not found:\n{PRETRAINED_MODEL}")

    # This only creates *_masks.tif.
    # It does not move/delete/change *_seg.npy files.
    convert_seg_npy_to_masks(TRAIN_DIR)
    convert_seg_npy_to_masks(VAL_DIR)

    train_images, train_masks = collect_image_mask_pairs(TRAIN_DIR)
    val_images, val_masks = collect_image_mask_pairs(VAL_DIR)

    train_images, train_masks, train_df = check_pairs_are_valid(
        train_images,
        train_masks,
        name="training",
    )

    val_images, val_masks, val_df = check_pairs_are_valid(
        val_images,
        val_masks,
        name="validation",
    )

    train_df.to_csv(run_dir / "train_pairs.csv", index=False)
    val_df.to_csv(run_dir / "val_pairs.csv", index=False)

    print("\n[DATA]")
    print(f"Training images:   {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    if len(train_images) < 30:
        raise RuntimeError(
            "Too few training annotations. For a useful helper model, "
            "aim for at least 80–120 total train slices."
        )

    if len(val_images) < 10:
        raise RuntimeError(
            "Too few validation annotations. Add at least 15–25 validation slices."
        )

    current_model_source = str(PRETRAINED_MODEL)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    epochs_done = 0
    block_index = 0

    training_log = {
        "experiment_name": EXPERIMENT_NAME,
        "pretrained_model": str(PRETRAINED_MODEL),
        "train_dir": str(TRAIN_DIR),
        "val_dir": str(VAL_DIR),
        "use_gpu": USE_GPU,
        "total_epochs_requested": TOTAL_EPOCHS,
        "epochs_per_block": EPOCHS_PER_BLOCK,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "bsize": BSIZE,
        "min_train_masks": MIN_TRAIN_MASKS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_delta": MIN_DELTA,
        "normalize": NORMALIZE,
        "compute_flows": COMPUTE_FLOWS,
        "n_train": len(train_images),
        "n_val": len(val_images),
        "blocks": [],
    }

    save_training_log(log_path, training_log)

    while epochs_done < TOTAL_EPOCHS:
        block_index += 1

        remaining_epochs = TOTAL_EPOCHS - epochs_done
        n_epochs_this_block = min(EPOCHS_PER_BLOCK, remaining_epochs)

        print("\n" + "=" * 80)
        print(f"[BLOCK {block_index}]")
        print(f"Epochs this block: {n_epochs_this_block}")
        print(f"Epochs completed before block: {epochs_done}")
        print(f"Current model source: {current_model_source}")
        print("=" * 80)

        # Important: reload latest checkpoint at every block.
        model = make_model(current_model_source)

        block_model_name = f"{EXPERIMENT_NAME}_block{block_index:03d}"

        raw_kwargs = {
            "net": model.net,

            "train_files": train_images,
            "train_labels_files": train_masks,

            "test_files": val_images,
            "test_labels_files": val_masks,

            "load_files": True,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "n_epochs": n_epochs_this_block,
            "weight_decay": WEIGHT_DECAY,

            "channel_axis": CHANNEL_AXIS,
            "normalize": NORMALIZE,
            "compute_flows": COMPUTE_FLOWS,

            "save_path": str(run_dir),
            "save_every": SAVE_EVERY,
            "save_each": False,

            "bsize": BSIZE,
            "min_train_masks": MIN_TRAIN_MASKS,
            "model_name": block_model_name,

            # Kept for version compatibility.
            # filter_kwargs_for_train_seg removes unsupported arguments.
            "channels": [0, 0],
            "rgb": False,
            "rescale": False,
        }

        train_kwargs = filter_kwargs_for_train_seg(raw_kwargs)

        result = train.train_seg(**train_kwargs)

        if not isinstance(result, tuple):
            raise RuntimeError(
                f"Unexpected train_seg return type: {type(result)}. "
                "Expected tuple: model_path, train_losses, val_losses"
            )

        model_path = result[0]
        train_losses = result[1] if len(result) > 1 else None
        val_losses = result[2] if len(result) > 2 else None

        final_train_loss = get_last_nonzero_loss(train_losses)
        final_val_loss = get_last_nonzero_loss(val_losses)

        epochs_done += n_epochs_this_block

        print("\n[BLOCK RESULT]")
        print("-" * 60)
        print(f"Saved model path: {model_path}")
        print(f"Final train loss: {final_train_loss}")
        print(f"Final val loss:   {final_val_loss}")
        print(f"Total epochs done: {epochs_done}")
        print("Raw train losses:", train_losses)
        print("Raw val losses:", val_losses)

        block_record = {
            "block_index": block_index,
            "epochs_this_block": n_epochs_this_block,
            "epochs_done": epochs_done,
            "model_path": str(model_path),
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "raw_train_losses": np.asarray(train_losses).tolist() if train_losses is not None else None,
            "raw_val_losses": np.asarray(val_losses).tolist() if val_losses is not None else None,
        }

        training_log["blocks"].append(block_record)

        if final_val_loss is None:
            print("[WARNING] No validation loss returned. Early stopping cannot be applied for this block.")
            current_model_source = str(model_path)
            save_training_log(log_path, training_log)
            continue

        improved = final_val_loss < (best_val_loss - MIN_DELTA)

        if improved:
            print(f"[IMPROVED] Validation loss improved: {best_val_loss} -> {final_val_loss}")

            best_val_loss = final_val_loss
            best_epoch = epochs_done
            patience_counter = 0

            copied = copy_best_model(model_path, best_model_path)

            if copied:
                training_log["best_val_loss"] = best_val_loss
                training_log["best_epoch"] = best_epoch
                training_log["best_model_path"] = str(best_model_path)

        else:
            patience_counter += 1
            print(f"[NO IMPROVEMENT] Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        training_log["latest_model_path"] = str(model_path)
        training_log["patience_counter"] = patience_counter
        save_training_log(log_path, training_log)

        # This is the important safe-reloading step.
        # The next training block starts from the latest saved checkpoint.
        current_model_source = str(model_path)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\n" + "=" * 80)
            print("[EARLY STOPPING TRIGGERED]")
            print("=" * 80)
            print(f"Best validation loss: {best_val_loss}")
            print(f"Best epoch: {best_epoch}")
            print(f"Best model saved at: {best_model_path}")
            break

    print("\n" + "=" * 80)
    print("[TRAINING FINISHED]")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Training log: {log_path}")

    if best_model_path.exists():
        print(f"Best model: {best_model_path}")
        print("\nAdd this model to Cellpose GUI:")
        print(best_model_path)
    else:
        print("No best model was saved. Check whether validation loss was returned.")


if __name__ == "__main__":
    train_annotation_helper_with_safe_reloading()
