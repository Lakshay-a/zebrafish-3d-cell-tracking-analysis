import os
from pathlib import Path
import numpy as np
import tifffile


PROJECT_ROOT = Path(os.environ.get("CELLPOSE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATASET_DIR = PROJECT_ROOT / "macrophage_cellpose_dataset_preprocessed"
TEST_DIR = DATASET_DIR / "test/images"


def load_cellpose_masks(seg_npy_path: Path):
    data = np.load(seg_npy_path, allow_pickle=True).item()

    if "masks" not in data:
        raise KeyError(
            f"No 'masks' key found in {seg_npy_path.name}. "
            f"Available keys: {list(data.keys())}"
        )

    masks = data["masks"]

    if masks is None:
        raise ValueError(f"'masks' is None in {seg_npy_path.name}")

    return masks


def is_auxiliary_file(path: Path):
    name = path.name

    bad_suffixes = [
        "_masks.tif",
        "_flows.tif",
        "_outlines.tif",
        "_cp_masks.tif",
        "_pred_masks.tif",
    ]

    return any(name.endswith(s) for s in bad_suffixes)


def main():
    seg_files = sorted(TEST_DIR.glob("*_seg.npy"))

    if not seg_files:
        raise FileNotFoundError(f"No *_seg.npy files found in: {TEST_DIR}")

    print("=" * 80)
    print("Converting Cellpose test _seg.npy files to _masks.tif")
    print("=" * 80)
    print(f"Test folder: {TEST_DIR}")
    print(f"Found _seg.npy files: {len(seg_files)}")

    converted = 0
    skipped = 0

    for seg_path in seg_files:
        image_name = seg_path.name.replace("_seg.npy", ".tif")
        image_path = TEST_DIR / image_name

        if not image_path.exists() or is_auxiliary_file(image_path):
            print(f"[SKIP] Matching image not found for: {seg_path.name}")
            skipped += 1
            continue

        image = tifffile.imread(image_path)
        masks = load_cellpose_masks(seg_path)

        if image.shape != masks.shape:
            print(f"[SKIP] Shape mismatch for {seg_path.name}")
            print(f"       image shape: {image.shape}")
            print(f"       mask shape : {masks.shape}")
            skipped += 1
            continue

        max_label = int(masks.max())

        if max_label <= 65535:
            masks_to_save = masks.astype(np.uint16)
        else:
            masks_to_save = masks.astype(np.uint32)

        output_name = seg_path.name.replace("_seg.npy", "_masks.tif")
        output_path = TEST_DIR / output_name

        tifffile.imwrite(output_path, masks_to_save)

        print(f"[OK] {seg_path.name} -> {output_name} | objects: {max_label}")
        converted += 1

    print("\nConversion complete.")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
