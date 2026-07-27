"""Print the software environment used for an analysis run as JSON."""

from importlib import metadata
import json
import os
import platform
import sys


PACKAGES = [
    "aicsimageio",
    "aicspylibczi",
    "cellpose",
    "imageio",
    "joblib",
    "matplotlib",
    "napari",
    "numpy",
    "pandas",
    "qtpy",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "tifffile",
    "torch",
    "tqdm",
    "zarr",
]

REPRODUCIBILITY_VARIABLES = [
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MPLBACKEND",
]


def installed_version(package):
    """Return an installed package version or null."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main():
    """Print a portable environment report."""
    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: installed_version(name) for name in PACKAGES},
        "environment": {
            name: os.environ.get(name) for name in REPRODUCIBILITY_VARIABLES
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
