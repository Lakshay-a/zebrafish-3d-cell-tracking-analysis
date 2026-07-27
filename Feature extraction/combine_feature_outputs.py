import argparse
from pathlib import Path
import pandas as pd


FEATURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_DIR.parent


PATTERNS = {
    "musc_all_blocks_cell_track_features.csv":
        "*/musc/*cell_track_features.csv",

    "macrophage_all_blocks_cell_track_features.csv":
        "*/macrophage_all/*cell_track_features.csv",

    "macrophage_all_blocks_cell_track_features_motile_filtered.csv":
        "*/macrophage_all/*motile_filtered.csv",

    "macrophage_outside_boundary_all_blocks_cell_track_features.csv":
        "*/macrophage_outside_boundary/*cell_track_features.csv",

    "macrophage_outside_boundary_all_blocks_cell_track_features_motile_filtered.csv":
        "*/macrophage_outside_boundary/*motile_filtered.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine per-block feature-extraction CSVs by dataset."
    )
    parser.add_argument(
        "--feature-root",
        default=str(FEATURE_DIR / "final_feature_outputs"),
        help="Root containing block/dataset feature output folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FEATURE_DIR / "final_combined_features"),
        help="Directory for combined CSV files.",
    )
    return parser.parse_args()


def combine(
    feature_root: Path,
    output_dir: Path,
    pattern: str,
    output_name: str,
):
    files = sorted(feature_root.glob(pattern))

    if not files:
        print(f"[WARN] No files found for {output_name}")
        print(f"       Pattern: {feature_root / pattern}")
        return

    frames = []

    for file in files:
        df = pd.read_csv(file)
        df["source_feature_file"] = str(file)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    out = output_dir / output_name
    combined.to_csv(out, index=False)

    print(f"[SAVED] {out}")
    print(f"        rows: {len(combined)}")
    print(f"        files combined: {len(files)}")


def main():
    args = parse_args()
    feature_root = Path(args.feature_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("COMBINING FEATURE OUTPUTS")
    print("============================================================")
    print(f"Feature root: {feature_root}")
    print(f"Output dir:   {output_dir}")
    print()

    for output_name, pattern in PATTERNS.items():
        combine(feature_root, output_dir, pattern, output_name)

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()
