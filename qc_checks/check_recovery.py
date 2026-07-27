from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import tifffile



# User setting — change only this path

FOLDER_TO_CHECK = Path(
    "/Users/lakshayarora/Desktop/recovery"
)


# Files to test inside the folder above.
TIFF_FILES = [
    "musc_3d_labels_TZYX.tif",
    "macrophage_3d_labels_TZYX.tif",
    "musc_cellpose_masks_TZYX.tif",
    "macrophage_cellpose_masks_TZYX.tif",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check TIFF structure and read sample pages without loading full stacks."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=FOLDER_TO_CHECK,
        help="Folder containing the TIFF files. Overrides FOLDER_TO_CHECK.",
    )
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def expected_page_count(shape: tuple[int, ...]) -> int | None:
    """Assume the final two dimensions are Y and X."""
    if len(shape) < 2:
        return None
    return int(math.prod(shape[:-2])) if len(shape) > 2 else 1


def read_sample_page(tif: tifffile.TiffFile, page_index: int) -> str:
    try:
        image = tif.pages[page_index].asarray()
        return (
            f"READ page {page_index}: shape={image.shape}, "
            f"dtype={image.dtype}, min={np.min(image)}, max={np.max(image)}"
        )
    except Exception as exc:  # Recovery diagnostics should continue after failures.
        return f"FAILED page {page_index}: {type(exc).__name__}: {exc}"


def check_tiff(path: Path) -> bool:
    print("\n" + "=" * 78)
    print(f"FILE: {path.name}")
    print(f"PATH: {path}")

    if not path.exists():
        print("RESULT: MISSING")
        return False

    if not path.is_file():
        print("RESULT: NOT A REGULAR FILE")
        return False

    print(f"File size: {human_size(path.stat().st_size)}")

    # A TIFF normally starts with II or MM. BigTIFF also uses the same byte order marker.
    try:
        with path.open("rb") as handle:
            signature = handle.read(8)
        print(f"First 8 bytes: {signature!r}")
    except OSError as exc:
        print(f"RESULT: FAILED TO READ FILE HEADER: {exc}")
        return False

    warning_messages: list[str] = []

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            with tifffile.TiffFile(path) as tif:
                if not tif.series:
                    print("RESULT: TIFF OPENED, BUT NO IMAGE SERIES WERE FOUND")
                    return False

                series = tif.series[0]
                shape = tuple(int(value) for value in series.shape)
                dtype = np.dtype(series.dtype)
                actual_pages = len(tif.pages)
                expected_pages = expected_page_count(shape)

                print(f"Shape: {shape}")
                print(f"Dtype: {dtype}")
                print(f"TIFF pages found: {actual_pages}")

                if expected_pages is not None:
                    print(f"Expected 2D pages from shape: {expected_pages}")
                    approximate_bytes = int(math.prod(shape)) * dtype.itemsize
                    print(
                        "Approximate uncompressed pixel data: "
                        f"{human_size(approximate_bytes)}"
                    )

                # Read the first, middle and last available pages. This does not
                # load the complete TIFF stack into memory.
                if actual_pages > 0:
                    sample_indices = sorted({0, actual_pages // 2, actual_pages - 1})
                    for index in sample_indices:
                        print(read_sample_page(tif, index))

            warning_messages = [str(item.message) for item in caught]

    except Exception as exc:
        print(f"RESULT: FAILED TO OPEN AS TIFF: {type(exc).__name__}: {exc}")
        return False

    if warning_messages:
        print("Warnings raised by tifffile:")
        for message in warning_messages:
            print(f"  - {message}")

    page_count_ok = (
        expected_pages is None
        or actual_pages == expected_pages
        or expected_pages == 1
    )

    if warning_messages or not page_count_ok:
        print("RESULT: SUSPICIOUS / POSSIBLY CORRUPT")
        if not page_count_ok:
            print(
                f"Reason: found {actual_pages} page(s), but shape suggests "
                f"approximately {expected_pages}."
            )
        return False

    print("RESULT: PASSED BASIC STRUCTURE AND SAMPLE-READ CHECKS")
    return True


def main() -> None:
    args = parse_args()
    root = args.folder.expanduser().resolve()

    print(f"Checking folder: {root}")
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {root}")

    results = {name: check_tiff(root / name) for name in TIFF_FILES}

    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'CHECK'}  {name}")

    passed_count = sum(results.values())
    print(f"\nPassed: {passed_count}/{len(results)}")
    print(
        "Note: PASS means the TIFF opened cleanly and sampled pages were readable. "
        "It is still not a complete pixel-by-pixel validation of every page."
    )


if __name__ == "__main__":
    main()