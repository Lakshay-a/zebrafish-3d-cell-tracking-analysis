from pathlib import Path
import pandas as pd



# Settings


PROJECT_DIR = Path(
    "/Users/lakshayarora/Desktop/Lakshay Dissertation/Custom cellpose/Final_approach/3d tracking"
)

OVERNIGHT_DIR = PROJECT_DIR / "overnight_batch_outputs"

REPORT_CSV = OVERNIGHT_DIR / "overnight_tracking_completion_report.csv"



# Expected file helpers


def expected_tracking_files(output_dir: Path, cell_type: str, method: str):
    return [
        output_dir / f"{cell_type}_tracks_{method}.csv",
        output_dir / f"{cell_type}_track_summary_{method}.csv",

        output_dir / f"{cell_type}_track_lengths_summary_{method}.csv",
        output_dir / f"{cell_type}_tracks_{method}_with_step_qc.csv",

        output_dir / f"{cell_type}_track_quality_features_{method}.csv",

        output_dir / f"{cell_type}_tracks_{method}_good_filtered.csv",
        output_dir / f"{cell_type}_good_track_ids_{method}.csv",
        output_dir / f"{cell_type}_good_track_quality_{method}.csv",

        output_dir / f"{cell_type}_tracks_{method}_clean_for_features.csv",
        output_dir / f"{cell_type}_clean_track_ids_{method}.csv",
        output_dir / f"{cell_type}_clean_track_quality_{method}.csv",

        output_dir / f"{cell_type}_track_filter_report_{method}.csv",
    ]


def check_files(paths):
    missing = [str(p.name) for p in paths if not p.exists()]
    return missing


def status_from_missing(missing):
    return "DONE" if len(missing) == 0 else "INCOMPLETE"



# Main check


rows = []

for block_dir in sorted(OVERNIGHT_DIR.iterdir()):
    if not block_dir.is_dir():
        continue

    block = block_dir.name


    # MUSC nearest


    musc_mask = block_dir / "musc_cellpose_masks_TZYX.tif"
    musc_labels = block_dir / "musc_3d_labels_TZYX.tif"
    musc_out = block_dir / "musc_tracking_outputs"

    if musc_mask.exists():
        expected = [musc_labels] + expected_tracking_files(
            musc_out,
            "musc",
            "nearest",
        )

        missing = check_files(expected)

        rows.append({
            "block": block,
            "cell_type": "musc",
            "region_mode": "not_applicable",
            "tracking_method": "nearest",
            "input_mask_exists": True,
            "cluster_mask_exists": "not_applicable",
            "status": status_from_missing(missing),
            "missing_count": len(missing),
            "missing_files": "; ".join(missing),
        })

    else:
        rows.append({
            "block": block,
            "cell_type": "musc",
            "region_mode": "not_applicable",
            "tracking_method": "nearest",
            "input_mask_exists": False,
            "cluster_mask_exists": "not_applicable",
            "status": "SKIPPED_NO_MASK",
            "missing_count": 0,
            "missing_files": "",
        })


    # MACROPHAGE LAP all


    mac_mask = block_dir / "macrophage_cellpose_masks_TZYX.tif"
    mac_labels = block_dir / "macrophage_3d_labels_TZYX.tif"

    cluster_exclusion = (
        block_dir / "macrophage_cluster_tracking_exclusion_mask_TYX.tif"
    )

    outside_boundary_csv = (
        block_dir / "macrophage_3d_object_features_outside_boundary.csv"
    )

    if mac_mask.exists():
        # all mode
        mac_all_out = block_dir / "macrophage_tracking_outputs_all"

        expected_all = [mac_labels] + expected_tracking_files(
            mac_all_out,
            "macrophage",
            "lap",
        )

        missing_all = check_files(expected_all)

        rows.append({
            "block": block,
            "cell_type": "macrophage",
            "region_mode": "all",
            "tracking_method": "lap",
            "input_mask_exists": True,
            "cluster_mask_exists": cluster_exclusion.exists(),
            "status": status_from_missing(missing_all),
            "missing_count": len(missing_all),
            "missing_files": "; ".join(missing_all),
        })

        # outside_boundary mode
        mac_ob_out = block_dir / "macrophage_tracking_outputs_outside_boundary"

        if cluster_exclusion.exists() and outside_boundary_csv.exists():
            expected_ob = [
                mac_labels,
                cluster_exclusion,
                outside_boundary_csv,
            ] + expected_tracking_files(
                mac_ob_out,
                "macrophage",
                "lap",
            )

            missing_ob = check_files(expected_ob)

            rows.append({
                "block": block,
                "cell_type": "macrophage",
                "region_mode": "outside_boundary",
                "tracking_method": "lap",
                "input_mask_exists": True,
                "cluster_mask_exists": True,
                "status": status_from_missing(missing_ob),
                "missing_count": len(missing_ob),
                "missing_files": "; ".join(missing_ob),
            })

        else:
            reason = []
            if not cluster_exclusion.exists():
                reason.append("macrophage_cluster_tracking_exclusion_mask_TYX.tif")
            if not outside_boundary_csv.exists():
                reason.append("macrophage_3d_object_features_outside_boundary.csv")

            rows.append({
                "block": block,
                "cell_type": "macrophage",
                "region_mode": "outside_boundary",
                "tracking_method": "lap",
                "input_mask_exists": True,
                "cluster_mask_exists": cluster_exclusion.exists(),
                "status": "SKIPPED_NO_CLUSTER_OUTPUT",
                "missing_count": len(reason),
                "missing_files": "; ".join(reason),
            })

    else:
        rows.append({
            "block": block,
            "cell_type": "macrophage",
            "region_mode": "all",
            "tracking_method": "lap",
            "input_mask_exists": False,
            "cluster_mask_exists": False,
            "status": "SKIPPED_NO_MASK",
            "missing_count": 0,
            "missing_files": "",
        })

        rows.append({
            "block": block,
            "cell_type": "macrophage",
            "region_mode": "outside_boundary",
            "tracking_method": "lap",
            "input_mask_exists": False,
            "cluster_mask_exists": False,
            "status": "SKIPPED_NO_MASK",
            "missing_count": 0,
            "missing_files": "",
        })


df = pd.DataFrame(rows)

df.to_csv(REPORT_CSV, index=False)

print("\n============================================================")
print("OVERNIGHT TRACKING COMPLETION CHECK")
print("============================================================")
print(f"Blocks checked: {df['block'].nunique()}")
print()
print(df.groupby(["cell_type", "region_mode", "status"]).size())
print()
print("Incomplete / skipped rows:")
print(
    df[df["status"] != "DONE"][
        [
            "block",
            "cell_type",
            "region_mode",
            "status",
            "missing_count",
            "missing_files",
        ]
    ].to_string(index=False)
)
print()
print(f"[SAVED] {REPORT_CSV}")