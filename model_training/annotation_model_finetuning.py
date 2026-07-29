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
VAL_DIR   = DATASET_DIR / "val"   / "images"

RUNS_DIR = DATASET_DIR / 'macrophage_annotation_helper_training_runs'
# RUNS_DIR = DATASET_DIR / 'musc_training_runs'

EXPERIMENT_NAME = "macrophage_annotation_helper_v31"
# EXPERIMENT_NAME = "musc_final"

# PRETRAINED_MODEL = 'cpsam'
PRETRAINED_MODEL = os.environ.get("CELLPOSE_PRETRAINED_MODEL", "cpsam")
# For grayscale GFP data
CHANNEL_AXIS = None
USE_GPU = torch.backends.mps.is_available()

# ── Fine-tuning hyperparameters ─────────────────────────────
TOTAL_EPOCHS      = 250
EPOCHS_PER_BLOCK  = 5

BATCH_SIZE    = 1
LEARNING_RATE = 1e-5   # initial LR; may be reduced during training
WEIGHT_DECAY  = 2e-3   # reduced from 1e-3 — less regularisation needed when
                        # starting from an already-adapted pilot model

BSIZE           = 256
MIN_TRAIN_MASKS = 1

# Early stopping (based on val loss)
EARLY_STOPPING_PATIENCE = 12   # blocks before stopping
MIN_DELTA                = 1e-4

# ── Block-level ReduceLROnPlateau ───────────────────────────
# LR is halved after LR_PATIENCE consecutive non-improving blocks.
# This counter is INDEPENDENT of early stopping patience:
#   - LR reduction resets its own counter but not the ES counter.
#   - If the LR reduction helps, ES patience gets more time to benefit.
# Once LR falls to LR_MIN it is never reduced further.
LR_PATIENCE        = 4
LR_REDUCTION_FACTOR = 0.2
LR_MIN             = 1e-7

# Guard: if val loss returns None this many consecutive blocks, raise.
MAX_CONSECUTIVE_NONE = 5

# ───────────────────────────────────────────────────────────
SAVE_EVERY    = EPOCHS_PER_BLOCK
NORMALIZE     = True
COMPUTE_FLOWS = True

MASK_SUFFIX = "_masks.tif"


# ============================================================
# FILE HELPERS
# ============================================================

def is_real_training_image(path: Path) -> bool:
    name = path.name
    if not name.endswith(".tif"):
        return False
    excluded_suffixes = [
        "_masks.tif", "_flows.tif", "_outlines.tif",
        "_cp_masks.tif", "_pred_masks.tif",
    ]
    return not any(name.endswith(s) for s in excluded_suffixes)


def image_to_mask_path(image_path: Path) -> Path:
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

    converted = skipped = failed = 0
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
            masks_to_save = masks.astype(np.uint16 if masks.max() <= 65535 else np.uint32)
            tifffile.imwrite(mask_file, masks_to_save)
            converted += 1
        except Exception as e:
            print(f"[ERROR] Failed converting {seg_file.name}: {e}")
            failed += 1

    print(f"[CONVERT] Converted: {converted}  Skipped: {skipped}  Failed: {failed}")


def collect_image_mask_pairs(folder: Path):
    image_files = [p for p in sorted(folder.glob("*.tif")) if is_real_training_image(p)]
    image_paths, mask_paths = [], []
    for image_path in image_files:
        mask_path = image_to_mask_path(image_path)
        if mask_path.exists():
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
    return image_paths, mask_paths


def check_pairs_are_valid(image_paths, mask_paths, name="dataset"):
    print(f"\n[CHECK] Checking {name} pairs...")
    print("-" * 60)
    valid_images, valid_masks, records = [], [], []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = tifffile.imread(image_path)
        mask  = tifffile.imread(mask_path)
        if image.shape != mask.shape:
            print(f"[SKIP] Shape mismatch: {Path(image_path).name}  "
                  f"image={image.shape}, mask={mask.shape}")
            continue
        max_label = int(mask.max())
        if max_label < 1:
            print(f"[SKIP] Empty mask: {Path(mask_path).name}")
            continue
        valid_images.append(image_path)
        valid_masks.append(mask_path)
        records.append({
            "image": image_path, "mask": mask_path,
            "n_objects": max_label,
            "shape_y": int(image.shape[0]), "shape_x": int(image.shape[1]),
        })
    print(f"[CHECK] Valid {name} pairs: {len(valid_images)}")
    if not valid_images:
        raise RuntimeError(f"No valid image-mask pairs found for {name}.")
    return valid_images, valid_masks, pd.DataFrame(records)


# ============================================================
# TRAINING HELPERS
# ============================================================

def get_last_nonzero_loss(losses):
    """
    Cellpose validation loss arrays can contain zero entries for epochs
    where validation was not evaluated. Returns the last non-zero finite value.
    """
    if losses is None:
        return None
    arr = np.asarray(losses).reshape(-1)
    arr = arr[np.isfinite(arr)]
    nonzero = arr[arr > 0]
    return float(nonzero[-1]) if nonzero.size > 0 else None


def filter_kwargs_for_train_seg(kwargs):
    """Removes kwargs not accepted by the installed Cellpose version."""
    sig      = inspect.signature(train.train_seg)
    accepted = set(sig.parameters.keys())
    filtered = {}
    for key, value in kwargs.items():
        if key in accepted:
            filtered[key] = value
        else:
            print(f"[INFO] train_seg does not accept: {key}")
    return filtered


def make_model(pretrained_model):
    print(f"\n[MODEL] Loading model from: {pretrained_model}")
    return models.CellposeModel(gpu=USE_GPU, pretrained_model=str(pretrained_model))


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

def purge_cached_flows(folder: Path):
    removed = 0
    for flow_file in folder.glob("*_flows.npy"):
        flow_file.unlink()
        removed += 1
    print(f"[CACHE] Purged {removed} cached flow files from {folder}")


# ============================================================
# MAIN TRAINING
# ============================================================

# Adapted from: https://cellpose.readthedocs.io/en/latest/train.html
def train_annotation_helper_with_safe_reloading():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir         = RUNS_DIR / f"{EXPERIMENT_NAME}_{timestamp}"
    best_model_path = run_dir / "best_model" / EXPERIMENT_NAME
    log_path        = run_dir / "training_log.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Annotation-helper Cellpose fine-tuning")
    print("=" * 80)
    print(f"Dataset dir:    {DATASET_DIR}")
    print(f"Train dir:      {TRAIN_DIR}")
    print(f"Val dir:        {VAL_DIR}")
    print(f"Run dir:        {run_dir}")
    print(f"Starting model: {PRETRAINED_MODEL}")
    print(f"GPU/MPS:        {USE_GPU}")
    print(f"Torch version:  {torch.__version__}")
    print(f"MPS built:      {torch.backends.mps.is_built()}")
    print(f"MPS available:  {torch.backends.mps.is_available()}")

    def is_builtin_cellpose_model(src):
        return str(src).lower() in {"cpsam", "cyto", "cyto2", "cyto3", "nuclei"}

    if not is_builtin_cellpose_model(PRETRAINED_MODEL):
        if not Path(PRETRAINED_MODEL).exists():
            raise FileNotFoundError(f"Pretrained model not found:\n{PRETRAINED_MODEL}")

    convert_seg_npy_to_masks(TRAIN_DIR)
    convert_seg_npy_to_masks(VAL_DIR)

    train_images, train_masks = collect_image_mask_pairs(TRAIN_DIR)
    val_images,   val_masks   = collect_image_mask_pairs(VAL_DIR)

    train_images, train_masks, train_df = check_pairs_are_valid(train_images, train_masks, "training")
    val_images,   val_masks,   val_df   = check_pairs_are_valid(val_images,   val_masks,   "validation")

    train_df.to_csv(run_dir / "train_pairs.csv", index=False)
    val_df.to_csv(  run_dir / "val_pairs.csv",   index=False)

    print(f"\n[DATA]  Train: {len(train_images)}  Val: {len(val_images)}")

    if len(train_images) < 30:
        raise RuntimeError("Too few training annotations (need ≥ 30).")
    if len(val_images) < 10:
        raise RuntimeError("Too few validation annotations (need ≥ 10).")

    # ── State ────────────────────────────────────────────────
    current_model_source = str(PRETRAINED_MODEL)
    current_lr           = LEARNING_RATE

    best_val_loss  = float("inf")
    best_epoch     = 0

    es_patience    = 0   # early-stopping counter
    lr_patience    = 0   # LR-reduction counter (independent)
    none_streak    = 0   # consecutive blocks with None val loss

    epochs_done  = 0
    block_index  = 0

    training_log = {
        "experiment_name":       EXPERIMENT_NAME,
        "pretrained_model":      str(PRETRAINED_MODEL),
        "train_dir":             str(TRAIN_DIR),
        "val_dir":               str(VAL_DIR),
        "use_gpu":               USE_GPU,
        "total_epochs_requested": TOTAL_EPOCHS,
        "epochs_per_block":      EPOCHS_PER_BLOCK,
        "batch_size":            BATCH_SIZE,
        "initial_learning_rate": LEARNING_RATE,
        "weight_decay":          WEIGHT_DECAY,
        "lr_patience":           LR_PATIENCE,
        "lr_reduction_factor":   LR_REDUCTION_FACTOR,
        "lr_min":                LR_MIN,
        "bsize":                 BSIZE,
        "min_train_masks":       MIN_TRAIN_MASKS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_delta":             MIN_DELTA,
        "normalize":             NORMALIZE,
        "compute_flows":         COMPUTE_FLOWS,
        "n_train":               len(train_images),
        "n_val":                 len(val_images),
        "blocks":                [],
    }
    save_training_log(log_path, training_log)

    # ── Training loop ─────────────────────────────────────────
    while epochs_done < TOTAL_EPOCHS:
        block_index += 1
        n_epochs_this_block = min(EPOCHS_PER_BLOCK, TOTAL_EPOCHS - epochs_done)

        # ← ADD THIS
        purge_cached_flows(VAL_DIR)
        purge_cached_flows(TRAIN_DIR)

        print("\n" + "=" * 80)
        print(f"[BLOCK {block_index}]  epochs this block: {n_epochs_this_block}  "
              f"done so far: {epochs_done}  LR: {current_lr:.2e}")
        print(f"Current model: {current_model_source}")
        print("=" * 80)

        model = make_model(current_model_source)
        # block_model_name = f"{EXPERIMENT_NAME}_block{block_index:03d}"
        block_model_name = f"{EXPERIMENT_NAME}_epoch{epochs_done + 1:03d}"

        raw_kwargs = {
            "net":               model.net,
            "train_files":       train_images,
            "train_labels_files": train_masks,
            "test_files":        val_images,
            "test_labels_files": val_masks,
            "load_files":        True,
            "batch_size":        BATCH_SIZE,
            "learning_rate":     current_lr,   # ← uses live LR, not constant
            "n_epochs":          n_epochs_this_block,
            "weight_decay":      WEIGHT_DECAY,
            "channel_axis":      CHANNEL_AXIS,
            "normalize":         NORMALIZE,
            "compute_flows":     COMPUTE_FLOWS,
            "save_path":         str(run_dir),
            "save_every":        SAVE_EVERY,
            "save_each":         False,
            "bsize":             BSIZE,
            "min_train_masks":   MIN_TRAIN_MASKS,
            "model_name":        block_model_name,
            "channels":          [0, 0],
            "rgb":               False,
            "rescale":           False,
        }

        train_kwargs = filter_kwargs_for_train_seg(raw_kwargs)
        result       = train.train_seg(**train_kwargs)

        if not isinstance(result, tuple):
            raise RuntimeError(
                f"Unexpected train_seg return type: {type(result)}. "
                "Expected (model_path, train_losses, val_losses)."
            )

        model_path   = result[0]
        train_losses = result[1] if len(result) > 1 else None
        val_losses   = result[2] if len(result) > 2 else None

        final_train_loss = get_last_nonzero_loss(train_losses)
        final_val_loss   = get_last_nonzero_loss(val_losses)

        epochs_done += n_epochs_this_block

        print("\n[BLOCK RESULT]")
        print("-" * 60)
        print(f"Saved model path:  {model_path}")
        print(f"Learning rate:     {current_lr:.2e}")
        print(f"Final train loss:  {final_train_loss}")
        print(f"Final val loss:    {final_val_loss}")
        print(f"Total epochs done: {epochs_done}")
        print(f"Raw val losses:    {val_losses}")

        block_record = {
            "block_index":       block_index,
            "epochs_this_block": n_epochs_this_block,
            "epochs_done":       epochs_done,
            "learning_rate":     current_lr,
            "model_path":        str(model_path),
            "final_train_loss":  final_train_loss,
            "final_val_loss":    final_val_loss,
            "raw_train_losses":  np.asarray(train_losses).tolist() if train_losses is not None else None,
            "raw_val_losses":    np.asarray(val_losses).tolist()   if val_losses   is not None else None,
        }
        training_log["blocks"].append(block_record)

        # ── Guard: too many consecutive None val losses ───────
        if final_val_loss is None:
            none_streak += 1
            print(f"[WARNING] No val loss returned. "
                  f"Consecutive None streak: {none_streak}/{MAX_CONSECUTIVE_NONE}")
            if none_streak >= MAX_CONSECUTIVE_NONE:
                print(f"[ABORT] Val loss has been None for {none_streak} consecutive blocks. "
                      "Check that test_files are being evaluated correctly.")
                save_training_log(log_path, training_log)
                break
            current_model_source = str(model_path)
            save_training_log(log_path, training_log)
            continue

        none_streak = 0   # reset on a valid loss

        # ── Improvement check ─────────────────────────────────
        improved = final_val_loss < (best_val_loss - MIN_DELTA)

        if improved:
            print(f"[IMPROVED] {best_val_loss:.6f} → {final_val_loss:.6f}")
            best_val_loss = final_val_loss
            best_epoch    = epochs_done
            es_patience   = 0
            lr_patience   = 0   # reset LR patience on genuine improvement

            copied = copy_best_model(model_path, best_model_path)
            if copied:
                training_log["best_val_loss"]   = best_val_loss
                training_log["best_epoch"]      = best_epoch
                training_log["best_model_path"] = str(best_model_path)

        else:
            es_patience += 1
            lr_patience += 1
            print(f"[NO IMPROVEMENT]  "
                  f"ES patience: {es_patience}/{EARLY_STOPPING_PATIENCE}  "
                  f"LR patience: {lr_patience}/{LR_PATIENCE}")

            # ── Block-level ReduceLROnPlateau ─────────────────
            if lr_patience >= LR_PATIENCE:
                candidate_lr = current_lr * LR_REDUCTION_FACTOR
                if candidate_lr >= LR_MIN:
                    current_lr  = candidate_lr
                    lr_patience = 0   # reset ONLY the LR counter
                    print(f"[LR REDUCE] Learning rate → {current_lr:.2e}")
                    training_log["lr_reductions"] = training_log.get("lr_reductions", [])
                    training_log["lr_reductions"].append({
                        "block": block_index,
                        "epoch": epochs_done,
                        "new_lr": current_lr,
                    })
                else:
                    print(f"[LR FLOOR] Already at minimum LR ({LR_MIN:.2e}), not reducing further.")
                    lr_patience = 0   # reset so the message doesn't repeat every block

        training_log["latest_model_path"] = str(model_path)
        training_log["es_patience"]        = es_patience
        training_log["lr_patience"]        = lr_patience
        training_log["current_lr"]         = current_lr
        save_training_log(log_path, training_log)

        current_model_source = str(model_path)

        if es_patience >= EARLY_STOPPING_PATIENCE:
            print("\n" + "=" * 80)
            print("[EARLY STOPPING TRIGGERED]")
            print(f"Best val loss: {best_val_loss:.6f}  at epoch {best_epoch}")
            print(f"Best model:    {best_model_path}")
            print("=" * 80)
            break

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[TRAINING FINISHED]")
    print(f"Run dir:      {run_dir}")
    print(f"Training log: {log_path}")
    if best_model_path.exists():
        print(f"Best model:   {best_model_path}")
        print("\nAdd this path to Cellpose GUI:")
        print(best_model_path)
    else:
        print("No best model saved — check whether val loss was ever returned.")


if __name__ == "__main__":
    train_annotation_helper_with_safe_reloading()
