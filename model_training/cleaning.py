import os
from pathlib import Path
import shutil


# ============================================================
# USER SETTINGS
# ============================================================

DATASET_DIR = Path(os.environ.get(
    "CELLPOSE_DATASET_DIR",
    Path(__file__).resolve().parents[1] / "datasets" / "cellpose_dataset",
))

FOLDERS_TO_CLEAN = [
    DATASET_DIR / "train" / "images",
    DATASET_DIR / "val" / "images",
    DATASET_DIR / "test" / "images",
]

BACKUP_DIR = DATASET_DIR / "cellpose_extra_outputs_backup"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# Files created by Cellpose GUI/inference that should not be treated as training images.
# IMPORTANT:
# We are NOT moving *_seg.npy because those are your GUI annotations.
# We are NOT moving *_masks.tif because those are needed for training.
EXTRA_PATTERNS = [
    "*_flows.tif",
    "*_flows.png",
    "*_outlines.tif",
    "*_outlines.png",
    "*_cp_masks.tif",
    "*_cp_masks.png",
    "*_pred_masks.tif",
    "*_pred_masks.png",
]


# If True, only prints what would be moved.
# Set to False when you are confident.
DRY_RUN = False


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def make_unique_target_path(target_path: Path):
    """
    If a file with the same name already exists in backup,
    create a unique filename instead of overwriting it.
    """
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent

    counter = 1

    while True:
        new_path = parent / f"{stem}_copy{counter}{suffix}"

        if not new_path.exists():
            return new_path

        counter += 1


def clean_folder(folder: Path):
    if not folder.exists():
        print(f"[SKIP] Folder does not exist: {folder}")
        return 0

    folder_backup = BACKUP_DIR / folder.relative_to(DATASET_DIR)
    folder_backup.mkdir(parents=True, exist_ok=True)

    moved = 0

    print("\n" + "=" * 80)
    print(f"[CLEANING] {folder}")
    print(f"[BACKUP]   {folder_backup}")
    print("=" * 80)

    for pattern in EXTRA_PATTERNS:
        files = sorted(folder.glob(pattern))

        for file_path in files:
            target_path = folder_backup / file_path.name
            target_path = make_unique_target_path(target_path)

            if DRY_RUN:
                print(f"[DRY RUN] Would move: {file_path.name}")
                print(f"          To: {target_path}")
            else:
                print(f"[MOVE] {file_path.name}")
                shutil.move(str(file_path), str(target_path))

            moved += 1

    print(f"[DONE] Moved {moved} extra files from: {folder.name}")

    return moved


def main():
    print("Cleaning Cellpose extra output files...")
    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Backup folder:  {BACKUP_DIR}")
    print(f"Dry run:        {DRY_RUN}")

    total_moved = 0

    for folder in FOLDERS_TO_CLEAN:
        total_moved += clean_folder(folder)

    print("\n" + "=" * 80)
    print("Cleanup complete.")
    print(f"Total extra files moved: {total_moved}")
    print(f"Extra files moved to: {BACKUP_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
