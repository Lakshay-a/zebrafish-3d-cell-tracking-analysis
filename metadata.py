import os
from pathlib import Path
from aicsimageio import AICSImage

PROJECT_ROOT = Path(__file__).resolve().parent
# Set CZI_PATH to inspect a file stored outside the repository.
CZI_PATH = Path(os.environ.get(
    "CZI_PATH",
    PROJECT_ROOT / "data" / "input.czi",
)).expanduser().resolve()

img = AICSImage(CZI_PATH)

print("Dims:", img.dims)
print("Shape:", img.shape)
print("Physical pixel sizes:")
print(img.physical_pixel_sizes)

px = img.physical_pixel_sizes

print("\nSuggested config values:")
print(f"XY_PIXEL_SIZE_UM = {px.X}")
print(f"Z_STEP_SIZE_UM = {px.Z}")
print(f"Z_DISTANCE_WEIGHT = {px.Z / px.X if px.X and px.Z else None}")