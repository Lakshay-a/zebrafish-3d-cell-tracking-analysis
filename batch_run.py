# caffeinate -dimsu python3 -u batch_run.py 2>&1 | tee overnight_run_$(date +%Y%m%d_%H%M%S).log
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path



# User settings


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get(
    "CELL_TRACKING_DATA_ROOT",
    SCRIPT_DIR / "data",
)).expanduser().resolve()

SCRIPT_00 = SCRIPT_DIR / "0_czi_to_tzxy_tif.py"
SCRIPT_01 = SCRIPT_DIR / "1_run_cellpose_2d.py"

BASE_OUTPUT_DIR = Path(os.environ.get(
    "BATCH_OUTPUT_ROOT",
    SCRIPT_DIR / "overnight_batch_outputs" / "Liraglutide",
)).expanduser().resolve()


# Enter CZI files here.
#
# cell_types can be:
#     ["musc"]
#     ["macrophage"]
#     ["musc", "macrophage"]
#
# 00 will run once per CZI.
# 01 will run once per listed cell type.


JOBS = [
    # Add jobs with paths relative to CELL_TRACKING_DATA_ROOT:
    # {
    #     "name": "example_block",
    #     "czi_path": DATA_ROOT / "experiment/input.czi",
    #     "cell_types": ["musc", "macrophage"],
    #     "drift_cell_type": "musc",
    #     "time_start": 0,
    #     "time_end": 50,
    #     "flip_xy": False,
    # },
    {
        "name": "260519_block01",
        "czi_path": DATA_ROOT / "260519/New-01.czi/New-01_AcquisitionBlock1.czi/New-01_AcquisitionBlock1.czi",
        "cell_types": ["musc", "macrophage"],
        "drift_cell_type": "musc",
        "time_start": 0,
        "time_end": 130,
        "flip_xy": False,
    },
    {
        "name": "260519_block05",
        "czi_path": DATA_ROOT / "260519/New-01.czi/New-01_AcquisitionBlock5.czi/New-01_AcquisitionBlock5.czi",
        "cell_types": ["musc", "macrophage"],
        "drift_cell_type": "musc",
        "time_start": 0,
        "time_end": 45,
        "flip_xy": False,
    },
    {
        "name": "260519_block07",
        "czi_path": DATA_ROOT / "260519/New-01.czi/New-01_AcquisitionBlock7.czi/New-01_AcquisitionBlock7.czi",
        "cell_types": ["musc", "macrophage"],
        "drift_cell_type": "musc",
        "time_start": 0,
        "time_end": 130,
        "flip_xy": False,
    },
    {
        "name": "260519_block08",
        "czi_path": DATA_ROOT / "260519/New-01.czi/New-01_AcquisitionBlock8.czi/New-01_AcquisitionBlock8.czi",
        "cell_types": ["musc", "macrophage"],
        "drift_cell_type": "musc",
        "time_start": 0,
        "time_end": 130,
        "flip_xy": False,
    },
    {
        "name": "260601_block01",
        "czi_path": DATA_ROOT / "260601/New-01.czi/New-01_AcquisitionBlock1.czi/New-01_AcquisitionBlock1.czi",
        "cell_types": ["musc", "macrophage"],
        "drift_cell_type": "musc",
        "time_start": 0,
        "time_end": 130,
        "flip_xy": False,
    },
    {
        "name": "260601_block03",
        "czi_path": DATA_ROOT / "260601/New-01.czi/New-01_AcquisitionBlock3.czi/New-01_AcquisitionBlock3.czi",
        "cell_types": ["musc", "macrophage"],
        "drift_cell_type": "musc",
        "time_start": 0,
        "time_end": 40,
        "flip_xy": False,
    },
    {
        "name": "260601_block05",
        "czi_path": DATA_ROOT / "260601/New-01.czi/New-01_AcquisitionBlock5.czi/New-01_AcquisitionBlock5.czi",
        "cell_types": ["musc", "macrophage"],
        "drift_cell_type": "musc",
        "time_start": 0,
        "time_end": 130,
        "flip_xy": False,
    },
 
  

]

# If False, skips 00 when raw_TCZYX_drift_corrected.tif already exists.
FORCE_RERUN_00 = False

# If False, skips 01 when the cell-type mask already exists.
FORCE_RERUN_01 = False
PAUSE_FOR_QC = False


# Helpers


def safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def run_script(script_path: Path, env: dict, log_path: Path) -> int:
    """Run one Python script and save stdout/stderr to a log file."""
    print("\n" + "=" * 80)
    print(f"[RUNNING] {script_path.name}")
    print(f"[LOG]     {log_path}")
    print("=" * 80)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", buffering=1) as log_file:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        process.wait()

    print(f"\n[EXIT CODE] {process.returncode}")

    return int(process.returncode)


def check_scripts_exist():
    missing = []

    for p in [SCRIPT_00, SCRIPT_01]:
        if not p.exists():
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "Missing required script(s):\n"
            + "\n".join(str(p) for p in missing)
        )

def valid_raw_tczyx_tif(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        import tifffile as tiff
        with tiff.TiffFile(path) as tif:
            if len(tif.pages) == 0:
                return False

        arr = tiff.imread(path)
        if arr.ndim not in (4, 5):
            return False

        if 0 in arr.shape:
            return False

        return True

    except Exception as e:
        print(f"[WARNING] Existing raw TIFF is invalid: {path}")
        print(f"[WARNING] Reason: {e}")
        return False

def flip_tiff_xy_once(
    tiff_path: Path,
    marker_path: Path,
) -> bool:
    """Flip the last two TIFF axes (Y and X) exactly once."""
    if marker_path.exists():
        print(
            f"[SKIP FLIP] XY flip already recorded: {marker_path}"
        )
        return False

    if not tiff_path.exists():
        raise FileNotFoundError(
            f"Cannot flip missing TIFF: {tiff_path}"
        )

    import numpy as np
    import tifffile as tiff

    temporary_path = tiff_path.with_name(
        f"{tiff_path.stem}__xy_flip_tmp{tiff_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    print("[FLIP XY] Rotating TIFF by 180 degrees.")
    print(f"[FLIP XY] Input:  {tiff_path}")
    print(f"[FLIP XY] Temp:   {temporary_path}")

    try:
        with tiff.TiffFile(tiff_path) as tif:
            series = tif.series[0]
            axes = str(series.axes).upper()
            shape = tuple(int(value) for value in series.shape)
            dtype = np.dtype(series.dtype)

        if len(shape) not in (4, 5):
            raise ValueError(
                f"Expected a 4D or 5D TIFF, got shape {shape}"
            )

        if not axes.endswith("YX"):
            raise ValueError(
                f"Expected Y and X as the last TIFF axes, got {axes}"
            )

        # Use memory mapping so the complete time-lapse does not need to be
        # loaded into RAM. tifffile writes the negative-stride flipped view
        # page by page.
        source = tiff.memmap(tiff_path)

        tiff.imwrite(
            temporary_path,
            source[..., ::-1, ::-1],
            bigtiff=True,
            photometric="minisblack",
            metadata={"axes": axes},
        )

        del source

        # Verify the temporary TIFF before replacing the original.
        with tiff.TiffFile(temporary_path) as tif:
            flipped_series = tif.series[0]
            flipped_shape = tuple(
                int(value) for value in flipped_series.shape
            )
            flipped_dtype = np.dtype(flipped_series.dtype)

        if flipped_shape != shape:
            raise ValueError(
                "Flipped TIFF shape changed: "
                f"{shape} -> {flipped_shape}"
            )

        if flipped_dtype != dtype:
            raise ValueError(
                "Flipped TIFF dtype changed: "
                f"{dtype} -> {flipped_dtype}"
            )

        temporary_path.replace(tiff_path)

        marker_path.write_text(
            "\n".join(
                [
                    "flip_xy=True",
                    f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"axes={axes}",
                    f"shape={shape}",
                    f"dtype={dtype}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print(f"[FLIP XY] Completed: {tiff_path}")
        return True

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def remove_stale_cellpose_masks(
    job_output_dir: Path,
) -> None:
    """Delete masks generated from an older TIFF orientation/version."""
    stale_masks = sorted(
        job_output_dir.glob("*_cellpose_masks_TZYX.tif")
    )

    for mask_path in stale_masks:
        print(f"[REMOVE STALE MASK] {mask_path}")
        mask_path.unlink()
def save_pre_cellpose_qc(
    tiff_path: Path,
    output_path: Path,
    job_name: str,
) -> None:
    """Save MUSC, macrophage and merged max-Z views from the middle timepoint."""
    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile as tiff

    def normalise(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)

        low, high = np.percentile(image, [1, 99.8])

        if high <= low:
            return np.zeros_like(image, dtype=np.float32)

        return np.clip(
            (image - low) / (high - low),
            0,
            1,
        )

    array = tiff.memmap(tiff_path)

    if array.ndim == 5:
        # Tczyx
        time_index = array.shape[0] // 2

        musc_projection = np.max(
            array[time_index, 0],
            axis=0,
        )

        if array.shape[1] > 1:
            macrophage_projection = np.max(
                array[time_index, 1],
                axis=0,
            )
        else:
            macrophage_projection = np.zeros_like(
                musc_projection
            )

    elif array.ndim == 4:
        # TZYX, MUSC-only
        time_index = array.shape[0] // 2

        musc_projection = np.max(
            array[time_index],
            axis=0,
        )

        macrophage_projection = np.zeros_like(
            musc_projection
        )

    else:
        raise ValueError(
            f"Unsupported TIFF shape for QC: {array.shape}"
        )

    musc_normalised = normalise(musc_projection)
    macrophage_normalised = normalise(
        macrophage_projection
    )

    merged = np.zeros(
        (*musc_normalised.shape, 3),
        dtype=np.float32,
    )

    merged[..., 0] = macrophage_normalised
    merged[..., 1] = musc_normalised
    merged[..., 2] = macrophage_normalised

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        musc_normalised,
        cmap="Greens",
    )
    axes[0].set_title("MUSC")

    axes[1].imshow(
        macrophage_normalised,
        cmap="magma",
    )
    axes[1].set_title("Macrophage")

    axes[2].imshow(merged)
    axes[2].set_title("Merged")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{job_name} — timepoint {time_index}"
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    del array

    print(f"[QC SAVED] {output_path}")

def main():
    check_scripts_exist()

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_start = time.strftime("%Y-%m-%d_%H-%M-%S")
    print("=" * 80)
    print("OVERNIGHT 00 + 01 BATCH RUN")
    print("=" * 80)
    print(f"Batch start:      {batch_start}")
    print(f"Script folder:    {SCRIPT_DIR}")
    print(f"Output base dir:  {BASE_OUTPUT_DIR}")
    print(f"Number of jobs:   {len(JOBS)}")
    print("=" * 80)

    failed_jobs = []

    for job_index, job in enumerate(JOBS, start=1):
        czi_path = Path(job["czi_path"])
        cell_types = job.get("cell_types", ["musc"])
        drift_cell_type = job.get("drift_cell_type", cell_types[0])
        time_start = job.get("time_start", 0)
        time_end = job.get("time_end", "120")
        flip_xy = bool(job.get("flip_xy", False))

        if time_end is None:
            time_end = "none"

        if not czi_path.exists():
            print(f"\n[ERROR] CZI path does not exist: {czi_path}")
            failed_jobs.append((job.get("name", str(job_index)), "missing_czi"))
            continue

        job_name = job.get("name")
        if not job_name:
            job_name = czi_path.stem

        job_name = safe_name(job_name)

        job_output_dir = BASE_OUTPUT_DIR / job_name
        logs_dir = job_output_dir / "logs"
        job_output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        raw_tczyx_path = job_output_dir / "raw_TCZYX_drift_corrected.tif"
        flip_marker_path = job_output_dir / ".raw_tczyx_xy_flipped"

        print("\n" + "#" * 80)
        print(f"[JOB {job_index}/{len(JOBS)}] {job_name}")
        print("#" * 80)
        print(f"CZI:              {czi_path}")
        print(f"Output folder:    {job_output_dir}")
        print(f"Cell types:       {cell_types}")
        print(f"Drift cell type:  {drift_cell_type}")
        print(f"Flip Y and X:     {flip_xy}")


        # Run 00 once per CZI

        raw_regenerated = False

        if valid_raw_tczyx_tif(raw_tczyx_path) and not FORCE_RERUN_00:
            print(f"[SKIP 00] Existing valid raw TCZYX found: {raw_tczyx_path}")
        else:
            # Script 00 will create a fresh, unflipped TIFF, so any old
            # orientation marker is no longer valid.
            if flip_marker_path.exists():
                flip_marker_path.unlink()

            if raw_tczyx_path.exists():
                print(f"[INFO] Existing raw TCZYX is invalid or FORCE_RERUN_00=True, regenerating:")
                print(raw_tczyx_path)
            env_00 = os.environ.copy()
            env_00["BATCH_PROJECT_DIR"] = str(job_output_dir)
            env_00["BATCH_CZI_PATH"] = str(czi_path)
            env_00["BATCH_CELL_TYPE"] = str(drift_cell_type)
            env_00["BATCH_TIME_START"] = str(time_start)
            env_00["BATCH_TIME_END"] = str(time_end).lower()

            log_00 = logs_dir / "00_czi_to_TCZYX.log"

            code = run_script(SCRIPT_00, env=env_00, log_path=log_00)

            if code != 0:
                print(f"[ERROR] 00 failed for job: {job_name}")
                failed_jobs.append((job_name, "00_failed"))
                continue

            raw_regenerated = True


        # Optional orientation correction after 00 and before 01

        flipped_now = False

        if flip_xy:
            try:
                flipped_now = flip_tiff_xy_once(
                    tiff_path=raw_tczyx_path,
                    marker_path=flip_marker_path,
                )
            except Exception as error:
                print(
                    f"[ERROR] XY flip failed for job={job_name}: "
                    f"{error}"
                )
                failed_jobs.append((job_name, "xy_flip_failed"))
                continue
        else:
            print("[FLIP XY] Not requested for this job.")

        # Any fresh TIFF or newly changed orientation invalidates old masks.
        if raw_regenerated or flipped_now:
            remove_stale_cellpose_masks(job_output_dir)
        
        qc_path = (
            job_output_dir
            / "qc_before_cellpose_mask_generation.png"
        )

        save_pre_cellpose_qc(
            tiff_path=raw_tczyx_path,
            output_path=qc_path,
            job_name=job_name,
        )

        if PAUSE_FOR_QC:
            subprocess.run(
                ["open", str(qc_path)],
                check=False,
            )

            response = input(
                "\nInspect the QC image.\n"
                "Press Enter to begin Cellpose, "
                "or type 'skip' to skip this fish: "
            ).strip().lower()

            if response == "skip":
                print(
                    f"[SKIP JOB AFTER QC] {job_name}"
                )
                failed_jobs.append(
                    (job_name, "manually_skipped_after_qc")
                )
                continue

        # Run 01 once per requested cell type

        for cell_type in cell_types:
            cell_type = cell_type.lower().strip()

            mask_path = job_output_dir / f"{cell_type}_cellpose_masks_TZYX.tif"

            if mask_path.exists() and not FORCE_RERUN_01:
                print(f"[SKIP 01] Existing {cell_type} mask found: {mask_path}")
                continue

            env_01 = os.environ.copy()
            env_01["BATCH_PROJECT_DIR"] = str(job_output_dir)
            env_01["BATCH_CZI_PATH"] = str(czi_path)
            env_01["BATCH_CELL_TYPE"] = str(cell_type)
            env_01["BATCH_TIME_START"] = str(time_start)
            env_01["BATCH_TIME_END"] = str(time_end).lower()

            log_01 = logs_dir / f"01_cellpose_{cell_type}.log"

            code = run_script(SCRIPT_01, env=env_01, log_path=log_01)

            if code != 0:
                print(f"[ERROR] 01 failed for job={job_name}, cell_type={cell_type}")
                failed_jobs.append((job_name, f"01_failed_{cell_type}"))
                continue

        print(f"\n[DONE JOB] {job_name}")

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)

    if failed_jobs:
        print("[WARNING] Some jobs failed:")
        for name, reason in failed_jobs:
            print(f"  - {name}: {reason}")
    else:
        print("[SUCCESS] All jobs completed.")

    print(f"\nOutputs saved under:\n{BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()