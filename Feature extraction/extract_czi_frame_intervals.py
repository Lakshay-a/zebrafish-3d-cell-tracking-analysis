#!/usr/bin/env python3
"""Extract CZI frame intervals from per-timepoint acquisition timestamps."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from aicspylibczi import CziFile


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        default=str(HERE / "Liraglutide_metadata.csv"),
    )
    parser.add_argument(
        "--blocks-root",
        default=str(PROJECT_ROOT / "overnight_batch_outputs" / "Liraglutide"),
    )
    parser.add_argument(
        "--czi-search-root",
        default=str(Path(os.environ.get(
            "CZI_SEARCH_ROOT",
            PROJECT_ROOT / "data",
        )).expanduser()),
        help="Fallback CZI root; may also be set with CZI_SEARCH_ROOT.",
    )
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optionally save the printed results as an audit CSV.",
    )
    return parser.parse_args()


def original_czi_from_log(block_dir: Path) -> Path:
    log_path = block_dir / "logs" / "00_czi_to_TCZYX.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Conversion log not found: {log_path}")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\[INFO\]\s+CZI:\s+(.+?)\s*$", line)
        if match:
            path = Path(match.group(1))
            if not path.exists():
                raise FileNotFoundError(f"CZI recorded in log is unavailable: {path}")
            return path
    raise ValueError(f"No '[INFO] CZI:' entry found in {log_path}")


def discover_original_czi(
    block_name: str,
    all_czi_paths: list[Path],
) -> Path:
    match = re.fullmatch(r"(\d{6}|\d{8})_block0*(\d+)", block_name)
    if not match:
        raise ValueError(f"Cannot derive date/block number from {block_name}")
    date_token, block_number = match.groups()
    acquisition_pattern = re.compile(
        rf"_AcquisitionBlock0*{int(block_number)}\.czi$",
        re.IGNORECASE,
    )
    candidates = []
    for path in all_czi_paths:
        digits_only = re.sub(r"\D", "", str(path))
        if date_token not in digits_only:
            continue
        if acquisition_pattern.search(path.name):
            candidates.append(path)
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No original CZI found for {block_name} under the CZI search root"
        )
    raise ValueError(
        f"Multiple possible CZIs found for {block_name}: "
        + " | ".join(str(path) for path in candidates)
    )


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def extract_interval(czi_path: Path) -> dict[str, float | int]:
    czi = CziFile(czi_path)
    records = czi.read_subblock_metadata(
        unified_xml=False,
        S=0,
        C=0,
        Z=0,
    )
    timestamps_by_t: dict[int, datetime] = {}
    for dimensions, xml_text in records:
        if "T" not in dimensions:
            continue
        acquisition_time = ET.fromstring(xml_text).findtext(".//AcquisitionTime")
        if acquisition_time:
            timestamps_by_t[int(dimensions["T"])] = parse_timestamp(acquisition_time)

    ordered = sorted(timestamps_by_t.items())
    if len(ordered) < 2:
        raise ValueError(
            f"Fewer than two timepoint timestamps found in {czi_path}"
        )

    seconds = np.array([item[1].timestamp() for item in ordered], dtype=float)
    differences = np.diff(seconds)
    if np.any(~np.isfinite(differences)) or np.any(differences <= 0):
        raise ValueError(f"Non-increasing acquisition timestamps in {czi_path}")

    return {
        "n_czi_timepoints": int(len(ordered)),
        "frame_interval_seconds": float(
            (seconds[-1] - seconds[0]) / (len(seconds) - 1)
        ),
        "median_consecutive_interval_seconds": float(np.median(differences)),
        "interval_sd_seconds": float(np.std(differences, ddof=1)),
        "interval_min_seconds": float(np.min(differences)),
        "interval_max_seconds": float(np.max(differences)),
    }


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata).resolve()
    blocks_root = Path(args.blocks_root).resolve()
    czi_search_root = Path(args.czi_search_root)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    required = {"block_name", "time_interval_seconds"}
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"{metadata_path} is missing columns: {missing_columns}")

    audit_rows: list[dict[str, object]] = []
    all_czi_paths: list[Path] | None = None

    for index, row in metadata.iterrows():
        block_name = str(row["block_name"]).strip()
        try:
            try:
                czi_path = original_czi_from_log(blocks_root / block_name)
            except (FileNotFoundError, ValueError):
                if all_czi_paths is None:
                    if not czi_search_root.exists():
                        raise FileNotFoundError(
                            f"CZI search root is unavailable: {czi_search_root}"
                        )
                    all_czi_paths = sorted(
                        path
                        for path in czi_search_root.rglob("*.czi")
                        if path.is_file()
                    )
                czi_path = discover_original_czi(block_name, all_czi_paths)
            result = extract_interval(czi_path)
            value = float(result["frame_interval_seconds"])
            audit_rows.append(
                {
                    "block_name": block_name,
                    "status": "extracted",
                    "original_czi": str(czi_path),
                    **result,
                }
            )
            print(
                f"[OK] {block_name}: {value:.6f} s "
                f"(T={result['n_czi_timepoints']}, "
                f"SD={result['interval_sd_seconds']:.3f} s)"
            )
        except Exception as error:
            audit_rows.append(
                {
                    "block_name": block_name,
                    "status": "error",
                    "error": str(error),
                }
            )
            print(f"[ERROR] {block_name}: {error}")

    if args.audit_output and audit_rows:
        audit_path = Path(args.audit_output).resolve()
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        print(f"[AUDIT] {audit_path}")

    failures = [row for row in audit_rows if row["status"] == "error"]
    if failures:
        raise SystemExit(f"{len(failures)} block(s) failed.")


if __name__ == "__main__":
    main()
