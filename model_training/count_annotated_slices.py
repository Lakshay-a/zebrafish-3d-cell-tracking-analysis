import os
from pathlib import Path
import pandas as pd

TRAIN_IMAGES_DIR = Path(os.environ.get(
    "CELLPOSE_ANNOTATION_DIR",
    Path(__file__).resolve().parents[1] / "datasets" / "cellpose_dataset" / "test" / "images",
))

# Cellpose GUI usually saves annotations as *_seg.npy
npy_files = sorted(TRAIN_IMAGES_DIR.glob("*_seg.npy"))

print("=" * 60)
print("TRAIN ANNOTATION COUNT")
print("=" * 60)
print(f"Folder: {TRAIN_IMAGES_DIR}")
print(f"Number of annotated *_seg.npy files: {len(npy_files)}")
print(f"The time right now is {pd.Timestamp.now()}")

# print("\nFirst 10 annotated files:")
# for f in npy_files[:10]:
#     print(f.name)

# print("\nLast annotated file:")
# for f in npy_files[-3:]:
#     print(f.name)
