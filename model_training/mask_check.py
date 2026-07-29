import os
from pathlib import Path
import numpy as np
import tifffile

image_path = Path(os.environ.get(
    "CELLPOSE_MASK_CHECK_IMAGE",
    Path(__file__).resolve().parents[1] / "datasets" / "cellpose_dataset" / "test" / "images" / "example.tif",
))

seg_path = image_path.with_name(image_path.stem + "_seg.npy")
mask_path = image_path.with_name(image_path.stem + "_masks.tif")

seg = np.load(seg_path, allow_pickle=True).item()
seg_mask = seg["masks"]
tif_mask = tifffile.imread(mask_path)

print("Objects in _seg.npy:", int(seg_mask.max()))
print("Objects in _masks.tif:", int(tif_mask.max()))
print("Same shape:", seg_mask.shape == tif_mask.shape)
print("Same mask content:", np.array_equal(seg_mask, tif_mask))
