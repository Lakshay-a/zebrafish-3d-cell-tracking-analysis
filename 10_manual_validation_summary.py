from pathlib import Path
import pandas as pd

from config import CELL_TYPE, TRACKING_OUTPUT_DIR



# Paths


VALIDATION_DIR = TRACKING_OUTPUT_DIR / "manual_tracking_validation"

CURRENT_SUMMARY_CSV = (
    VALIDATION_DIR
    / f"{CELL_TYPE}_manual_tracking_score_summary_all.csv"
)

MASTER_SUMMARY_CSV = (
    VALIDATION_DIR
    / f"{CELL_TYPE}_manual_tracking_score_summary_master.csv"
)

OUTPUT_TABLE_CSV = (
    VALIDATION_DIR
    / f"{CELL_TYPE}_tracking_accuracy_summary_table.csv"
)



# Settings


METHOD_ORDER = {
    "nearest": 0,
    "keyhole": 1,
    "lap": 2,
    "LAP": 2,
}

VARIANT_ORDER = {
    "raw": 0,
    "good_filtered": 1,
    "clean_for_features": 2,
}


def pct(x):
    if pd.isna(x):
        return ""
    return f"{100 * float(x):.2f}%"


def main():
    print("=" * 70)
    print("TRACKING ACCURACY SUMMARY TABLE")
    print("=" * 70)

    if not CURRENT_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Could not find current summary CSV:\n{CURRENT_SUMMARY_CSV}\n"
            "Run 9_score_manual_tracking_ground_truth.py first."
        )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    current = pd.read_csv(CURRENT_SUMMARY_CSV)

    if MASTER_SUMMARY_CSV.exists():
        master = pd.read_csv(MASTER_SUMMARY_CSV)
        combined = pd.concat([master, current], ignore_index=True)
    else:
        combined = current.copy()

    # Keep latest result for each method + variant.
    combined = combined.drop_duplicates(
        subset=["tracking_method", "track_variant"],
        keep="last",
    )

    combined["_method_order"] = (
        combined["tracking_method"]
        .map(METHOD_ORDER)
        .fillna(99)
    )

    combined["_variant_order"] = (
        combined["track_variant"]
        .map(VARIANT_ORDER)
        .fillna(99)
    )

    combined = combined.sort_values(
        ["_method_order", "_variant_order", "tracking_method", "track_variant"]
    ).drop(columns=["_method_order", "_variant_order"])

    combined.to_csv(MASTER_SUMMARY_CSV, index=False)


    # Display table like screenshot

    table = pd.DataFrame()

    table["Tracking method"] = combined["tracking_method"].replace(
        {"lap": "LAP"}
    )

    table["Variant"] = combined["track_variant"]

    table["Matched point rate"] = combined["matched_point_rate"].apply(pct)

    table["Correct links"] = combined.apply(
        lambda r: f"{int(r['correct_links'])} / {int(r['total_scored_links'])}",
        axis=1,
    )

    table["Broken links"] = combined["broken_links"].astype(int)
    table["Wrong links"] = combined["wrong_inappropriate_links"].astype(int)
    table["Correct link rate"] = combined["correct_link_rate"].apply(pct)

    table["Tracking score"] = combined[
        "tracking_score_plus1_minus1_minus2"
    ].astype(float).map(lambda x: f"{x:.3f}")

    table.to_csv(OUTPUT_TABLE_CSV, index=False)


if __name__ == "__main__":
    main()