from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CELL_TYPE,
    TRACKING_OUTPUT_DIR,
    FEATURE_TRACKING_METHOD,
    Z_DISTANCE_WEIGHT,
)



# User settings


# Manual GT file created from Napari GT001, GT002, ... points.
MANUAL_GT_CSV = TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_manual_ground_truth_tracks.csv"

# Keep this as [FEATURE_TRACKING_METHOD] for the current run.
# Later you can change it to ["nearest", "lap", "keyhole"] if those files exist.
TRACKING_METHODS_TO_SCORE = [
    # Feature_tracking_method,
    "nearest",
    "lap",
    "keyhole",
]

# raw = all automated tracks
# good_filtered = after length/step/Z/static filtering
# clean_for_features = final classifier-ready tracks
TRACK_VARIANTS_TO_SCORE = [
    "raw",
    "good_filtered",
    "clean_for_features",
]

# Maximum 3D distance allowed between manual GT point and automated object centroid.
# Units are XY-pixel-equivalent units. Z is scaled by Z_DISTANCE_WEIGHT.
MAX_GT_MATCH_DISTANCE_3D = 15.0

# If True, score only links where manual labels are on consecutive timepoints,
# e.g. T10->T11. If a GT track has T10 then T12 with T11 missing, that link is skipped.
REQUIRE_CONSECUTIVE_GT_TIMES = True

VALIDATION_OUTPUT_DIR = TRACKING_OUTPUT_DIR / "manual_tracking_validation"



# Helper functions


def normalise_track_id(x) -> str:
    """Normalise track IDs so 91, 91.0 and '91' compare as the same ID."""
    if pd.isna(x):
        return ""
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
    except Exception:
        pass
    return str(x).strip()


def get_track_file(method: str, variant: str):
    if variant == "raw":
        return TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{method}.csv"
    if variant == "good_filtered":
        return TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{method}_good_filtered.csv"
    if variant == "clean_for_features":
        return TRACKING_OUTPUT_DIR / f"{CELL_TYPE}_tracks_{method}_clean_for_features.csv"
    raise ValueError(f"Unknown track variant: {variant}")


def check_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def distance_3d_to_point(
    dets: pd.DataFrame,
    z: float,
    y: float,
    x: float,
    z_distance_weight: float,
) -> np.ndarray:
    """3D distance from automated detections to one manual GT point."""
    dz = (dets["centroid_z"].to_numpy(dtype=float) - float(z)) * float(z_distance_weight)
    dy = dets["centroid_y"].to_numpy(dtype=float) - float(y)
    dx = dets["centroid_x"].to_numpy(dtype=float) - float(x)
    return np.sqrt(dx**2 + dy**2 + dz**2)


def load_manual_gt(gt_csv) -> pd.DataFrame:
    if not gt_csv.exists():
        raise FileNotFoundError(
            f"Manual ground-truth CSV not found:\n{gt_csv}\n\n"
            "Create this first using the Napari GT script."
        )

    gt = pd.read_csv(gt_csv)

    required = ["gt_track_id", "time", "centroid_z", "centroid_y", "centroid_x"]
    check_required_columns(gt, required, "manual GT CSV")

    gt = gt.copy()
    gt["gt_track_id"] = gt["gt_track_id"].astype(str)
    gt["time"] = gt["time"].astype(int)
    gt = gt.dropna(subset=["centroid_z", "centroid_y", "centroid_x"]).copy()
    gt = gt.sort_values(["gt_track_id", "time"]).reset_index(drop=True)

    dup_mask = gt.duplicated(subset=["gt_track_id", "time"], keep=False)
    if dup_mask.any():
        print("\n[WARNING] Duplicate manual GT points found for the same gt_track_id/time.")
        print("Keeping the first point for each duplicate.")
        print(gt.loc[dup_mask, ["gt_track_id", "time", "centroid_z", "centroid_y", "centroid_x"]])
        gt = gt.drop_duplicates(subset=["gt_track_id", "time"], keep="first").copy()

    return gt


def load_auto_tracks(tracks_csv) -> pd.DataFrame:
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Automated tracks CSV not found:\n{tracks_csv}")

    tracks = pd.read_csv(tracks_csv)

    required = ["track_id", "time", "centroid_z", "centroid_y", "centroid_x"]
    check_required_columns(tracks, required, "automated tracks CSV")

    tracks = tracks.copy()
    tracks["time"] = tracks["time"].astype(int)
    tracks["_track_id_key"] = tracks["track_id"].apply(normalise_track_id)
    tracks["_auto_row_id"] = np.arange(len(tracks))

    return tracks


def match_gt_points_to_auto(
    gt: pd.DataFrame,
    tracks: pd.DataFrame,
    max_match_distance_3d: float,
    z_distance_weight: float,
) -> pd.DataFrame:
    """Match each manual GT point to the nearest automated detection at the same timepoint."""
    tracks_by_time = {int(t): g.copy() for t, g in tracks.groupby("time", sort=False)}
    rows = []

    for _, gt_row in gt.iterrows():
        gt_id = str(gt_row["gt_track_id"])
        t = int(gt_row["time"])

        gt_z = float(gt_row["centroid_z"])
        gt_y = float(gt_row["centroid_y"])
        gt_x = float(gt_row["centroid_x"])

        dets_t = tracks_by_time.get(t)

        base = {
            "gt_track_id": gt_id,
            "time": t,
            "gt_centroid_z": gt_z,
            "gt_centroid_y": gt_y,
            "gt_centroid_x": gt_x,
            "matched": False,
            "auto_track_id": np.nan,
            "auto_track_id_key": "",
            "auto_row_id": np.nan,
            "auto_centroid_z": np.nan,
            "auto_centroid_y": np.nan,
            "auto_centroid_x": np.nan,
            "match_distance_3d": np.nan,
            "second_best_distance_3d": np.nan,
            "match_margin_3d": np.nan,
            "ambiguous_point_match": False,
            "point_match_note": "",
        }

        if dets_t is None or dets_t.empty:
            base["point_match_note"] = "no_automated_detections_at_time"
            rows.append(base)
            continue

        dists = distance_3d_to_point(dets_t, gt_z, gt_y, gt_x, z_distance_weight)
        order = np.argsort(dists)

        best_pos = int(order[0])
        best_dist = float(dists[best_pos])
        second_best_dist = float(dists[int(order[1])]) if len(order) >= 2 else np.nan
        best = dets_t.iloc[best_pos]

        base["match_distance_3d"] = best_dist
        base["second_best_distance_3d"] = second_best_dist
        base["match_margin_3d"] = second_best_dist - best_dist if np.isfinite(second_best_dist) else np.nan

        if best_dist <= max_match_distance_3d:
            base["matched"] = True
            base["auto_track_id"] = best["track_id"]
            base["auto_track_id_key"] = best["_track_id_key"]
            base["auto_row_id"] = int(best["_auto_row_id"])
            base["auto_centroid_z"] = float(best["centroid_z"])
            base["auto_centroid_y"] = float(best["centroid_y"])
            base["auto_centroid_x"] = float(best["centroid_x"])
            base["point_match_note"] = "matched"
        else:
            base["point_match_note"] = "nearest_auto_detection_too_far"

        rows.append(base)

    matches = pd.DataFrame(rows)

    # Flag if the same automated detection is matched to multiple GT points at the same time.
    if not matches.empty and "auto_row_id" in matches.columns:
        matched = matches[matches["matched"]].copy()
        if not matched.empty:
            dup_keys = matched.groupby(["time", "auto_row_id"])["gt_track_id"].transform("nunique") > 1
            ambiguous_index = matched.loc[dup_keys].index
            matches.loc[ambiguous_index, "ambiguous_point_match"] = True
            matches.loc[ambiguous_index, "point_match_note"] = (
                matches.loc[ambiguous_index, "point_match_note"].astype(str)
                + ";same_auto_detection_matched_to_multiple_gt_points"
            )

    return matches


def nearest_gt_to_auto_detection(
    gt_at_time: pd.DataFrame,
    auto_z: float,
    auto_y: float,
    auto_x: float,
    current_gt_track_id: str,
    z_distance_weight: float,
) -> tuple[str, float]:
    """Find nearest other manual GT point to an automated detection at the same time."""
    other = gt_at_time[gt_at_time["gt_track_id"] != current_gt_track_id].copy()

    if other.empty:
        return "", np.nan

    dz = (other["centroid_z"].to_numpy(dtype=float) - float(auto_z)) * float(z_distance_weight)
    dy = other["centroid_y"].to_numpy(dtype=float) - float(auto_y)
    dx = other["centroid_x"].to_numpy(dtype=float) - float(auto_x)

    dists = np.sqrt(dx**2 + dy**2 + dz**2)

    idx = int(np.argmin(dists))
    nearest_row = other.iloc[idx]

    return str(nearest_row["gt_track_id"]), float(dists[idx])


# Tracking-evaluation reference: https://doaj.org/article/e123b0ee2a7b41089c2404fdec50f84e
def score_manual_links(
    gt: pd.DataFrame,
    tracks: pd.DataFrame,
    matches: pd.DataFrame,
    max_match_distance_3d: float,
    z_distance_weight: float,
    require_consecutive_gt_times: bool = True,
) -> pd.DataFrame:
    """Score links between consecutive manual GT points."""

    tracks_by_time_and_id = {
        (int(t), str(track_key)): g.copy()
        for (t, track_key), g in tracks.groupby(["time", "_track_id_key"], sort=False)
    }

    gt_by_time = {int(t): g.copy() for t, g in gt.groupby("time", sort=False)}

    match_lookup = {
        (str(row["gt_track_id"]), int(row["time"])): row
        for _, row in matches.iterrows()
    }

    link_rows = []

    for gt_track_id, g in gt.groupby("gt_track_id"):
        g = g.sort_values("time").copy()

        if len(g) < 2:
            continue

        times = g["time"].astype(int).to_list()

        for i in range(len(times) - 1):
            t0 = int(times[i])
            t1 = int(times[i + 1])
            manual_time_gap = t1 - t0

            base = {
                "gt_track_id": gt_track_id,
                "t0": t0,
                "t1": t1,
                "manual_time_gap": manual_time_gap,
                "auto_track_t0": np.nan,
                "auto_track_t1": np.nan,
                "distance_t0": np.nan,
                "distance_t1": np.nan,
                "auto_track_t0_continues_to_t1": False,
                "continuation_distance_to_gt_t1": np.nan,
                "nearest_other_gt_to_continuation": "",
                "nearest_other_gt_distance": np.nan,
                "link_type": "not_scored",
                "score": np.nan,
                "reason": "",
                "ambiguous_point_match_t0": False,
                "ambiguous_point_match_t1": False,
            }

            if require_consecutive_gt_times and manual_time_gap != 1:
                base["reason"] = "skipped_non_consecutive_manual_gt_timepoints"
                link_rows.append(base)
                continue

            m0 = match_lookup.get((str(gt_track_id), t0))
            m1 = match_lookup.get((str(gt_track_id), t1))

            if m0 is None or m1 is None:
                base["link_type"] = "broken_or_lost"
                base["score"] = -1
                base["reason"] = "missing_manual_match_record"
                link_rows.append(base)
                continue

            matched0 = bool(m0["matched"])
            matched1 = bool(m1["matched"])

            base["distance_t0"] = m0["match_distance_3d"]
            base["distance_t1"] = m1["match_distance_3d"]
            base["ambiguous_point_match_t0"] = bool(m0.get("ambiguous_point_match", False))
            base["ambiguous_point_match_t1"] = bool(m1.get("ambiguous_point_match", False))

            if matched0:
                base["auto_track_t0"] = m0["auto_track_id"]

            if matched1:
                base["auto_track_t1"] = m1["auto_track_id"]

            if not matched0 and not matched1:
                base["link_type"] = "broken_or_lost"
                base["score"] = -1
                base["reason"] = "no_auto_detection_at_t0_or_t1"
                link_rows.append(base)
                continue

            if not matched0:
                base["link_type"] = "broken_or_lost"
                base["score"] = -1
                base["reason"] = "no_auto_detection_at_t0"
                link_rows.append(base)
                continue

            if not matched1:
                base["link_type"] = "broken_or_lost"
                base["score"] = -1
                base["reason"] = "no_auto_detection_at_t1"
                link_rows.append(base)
                continue

            track0_key = str(m0["auto_track_id_key"])
            track1_key = str(m1["auto_track_id_key"])

            # Same automated track ID at both manual timepoints = correct.
            if track0_key == track1_key:
                base["link_type"] = "correct"
                base["score"] = 1
                base["reason"] = "same_auto_track_id"
                link_rows.append(base)
                continue

            # Track ID changed. Decide broken vs inappropriate.
            # If the original automated track continues at t1 somewhere else,
            # then it linked the manual cell at t0 to another object: wrong/inappropriate.
            cont = tracks_by_time_and_id.get((t1, track0_key))

            if cont is None or cont.empty:
                base["link_type"] = "broken_or_lost"
                base["score"] = -1
                base["reason"] = "auto_track_id_changed_and_original_track_not_present_at_t1"
                link_rows.append(base)
                continue

            gt_t1_row = gt[(gt["gt_track_id"] == gt_track_id) & (gt["time"] == t1)].iloc[0]

            cont_dists_to_gt1 = distance_3d_to_point(
                dets=cont,
                z=float(gt_t1_row["centroid_z"]),
                y=float(gt_t1_row["centroid_y"]),
                x=float(gt_t1_row["centroid_x"]),
                z_distance_weight=z_distance_weight,
            )

            best_cont_pos = int(np.argmin(cont_dists_to_gt1))
            best_cont = cont.iloc[best_cont_pos]
            best_cont_dist = float(cont_dists_to_gt1[best_cont_pos])

            base["auto_track_t0_continues_to_t1"] = True
            base["continuation_distance_to_gt_t1"] = best_cont_dist

            # If original track is still close to the manual cell, count correct but flag.
            # This can occur if two detections compete near one manual cell.
            if best_cont_dist <= max_match_distance_3d:
                base["link_type"] = "correct"
                base["score"] = 1
                base["reason"] = (
                    "original_auto_track_continues_near_gt_t1;"
                    "nearest_detection_at_t1_had_different_track_id"
                )
                link_rows.append(base)
                continue

            gt_at_t1 = gt_by_time.get(t1, pd.DataFrame())

            nearest_other_gt, nearest_other_dist = nearest_gt_to_auto_detection(
                gt_at_time=gt_at_t1,
                auto_z=float(best_cont["centroid_z"]),
                auto_y=float(best_cont["centroid_y"]),
                auto_x=float(best_cont["centroid_x"]),
                current_gt_track_id=str(gt_track_id),
                z_distance_weight=z_distance_weight,
            )

            base["nearest_other_gt_to_continuation"] = nearest_other_gt
            base["nearest_other_gt_distance"] = nearest_other_dist

            base["link_type"] = "wrong_or_inappropriate"
            base["score"] = -2

            if nearest_other_gt and np.isfinite(nearest_other_dist) and nearest_other_dist <= max_match_distance_3d:
                base["reason"] = f"original_auto_track_continues_to_other_manual_gt:{nearest_other_gt}"
            else:
                base["reason"] = "original_auto_track_continues_away_from_manual_gt"

            link_rows.append(base)

    return pd.DataFrame(link_rows)


def summarise_scores(link_scores, matches, method, variant, tracks_csv):
    scored = link_scores[link_scores["score"].notna()].copy()

    total_scored_links = len(scored)

    if total_scored_links == 0:
        correct_links = broken_links = wrong_links = 0
        tracking_score = np.nan
        correct_link_rate = broken_link_rate = wrong_link_rate = np.nan
        link_jaccard_like = np.nan
    else:
        correct_links = int((scored["link_type"] == "correct").sum())
        broken_links = int((scored["link_type"] == "broken_or_lost").sum())
        wrong_links = int((scored["link_type"] == "wrong_or_inappropriate").sum())

        tracking_score = (correct_links - broken_links - 2 * wrong_links) / total_scored_links

        correct_link_rate = correct_links / total_scored_links
        broken_link_rate = broken_links / total_scored_links
        wrong_link_rate = wrong_links / total_scored_links

        # Simple link-overlap score: TP / all manual links.
        # This is not the full Track Performance Tool alpha/beta/Jaccard,
        # but it gives a transparent link-fit value.
        link_jaccard_like = correct_links / max(correct_links + broken_links + wrong_links, 1)

    total_manual_points = len(matches)
    matched_manual_points = int(matches["matched"].sum()) if not matches.empty else 0
    matched_point_rate = matched_manual_points / total_manual_points if total_manual_points > 0 else np.nan

    skipped_nonconsecutive = int(
        (link_scores["reason"] == "skipped_non_consecutive_manual_gt_timepoints").sum()
    ) if not link_scores.empty else 0

    ambiguous_point_matches = int(matches["ambiguous_point_match"].sum()) if "ambiguous_point_match" in matches.columns else 0

    summary = pd.DataFrame(
        [
            {
                "cell_type": CELL_TYPE,
                "tracking_method": method,
                "track_variant": variant,
                "tracks_csv": str(tracks_csv),
                "manual_gt_csv": str(MANUAL_GT_CSV),
                "max_gt_match_distance_3d": MAX_GT_MATCH_DISTANCE_3D,
                "require_consecutive_gt_times": REQUIRE_CONSECUTIVE_GT_TIMES,
                "total_manual_points": total_manual_points,
                "matched_manual_points": matched_manual_points,
                "matched_point_rate": matched_point_rate,
                "ambiguous_point_matches": ambiguous_point_matches,
                "total_scored_links": total_scored_links,
                "skipped_nonconsecutive_links": skipped_nonconsecutive,
                "correct_links": correct_links,
                "broken_links": broken_links,
                "wrong_inappropriate_links": wrong_links,
                "correct_link_rate": correct_link_rate,
                "broken_link_rate": broken_link_rate,
                "wrong_link_rate": wrong_link_rate,
                "tracking_score_plus1_minus1_minus2": tracking_score,
                "link_jaccard_like": link_jaccard_like,
            }
        ]
    )

    per_gt_rows = []

    if not scored.empty:
        for gt_track_id, g in scored.groupby("gt_track_id"):
            total = len(g)
            c = int((g["link_type"] == "correct").sum())
            b = int((g["link_type"] == "broken_or_lost").sum())
            w = int((g["link_type"] == "wrong_or_inappropriate").sum())

            per_gt_rows.append(
                {
                    "cell_type": CELL_TYPE,
                    "tracking_method": method,
                    "track_variant": variant,
                    "gt_track_id": gt_track_id,
                    "total_scored_links": total,
                    "correct_links": c,
                    "broken_links": b,
                    "wrong_inappropriate_links": w,
                    "correct_link_rate": c / total if total else np.nan,
                    "broken_link_rate": b / total if total else np.nan,
                    "wrong_link_rate": w / total if total else np.nan,
                    "tracking_score_plus1_minus1_minus2": (c - b - 2 * w) / total if total else np.nan,
                }
            )

    per_gt = pd.DataFrame(per_gt_rows)

    return summary, per_gt


def score_one_track_file(gt, method, variant, tracks_csv):
    print("\n------------------------------------------------------------")
    print(f"Scoring method={method}, variant={variant}")
    print("Tracks:", tracks_csv)
    print("------------------------------------------------------------")

    tracks = load_auto_tracks(tracks_csv)

    matches = match_gt_points_to_auto(
        gt=gt,
        tracks=tracks,
        max_match_distance_3d=MAX_GT_MATCH_DISTANCE_3D,
        z_distance_weight=Z_DISTANCE_WEIGHT,
    )

    link_scores = score_manual_links(
        gt=gt,
        tracks=tracks,
        matches=matches,
        max_match_distance_3d=MAX_GT_MATCH_DISTANCE_3D,
        z_distance_weight=Z_DISTANCE_WEIGHT,
        require_consecutive_gt_times=REQUIRE_CONSECUTIVE_GT_TIMES,
    )

    summary, per_gt = summarise_scores(
        link_scores=link_scores,
        matches=matches,
        method=method,
        variant=variant,
        tracks_csv=tracks_csv,
    )

    return {
        "matches": matches,
        "link_scores": link_scores,
        "summary": summary,
        "per_gt": per_gt,
    }


def main():
    print("\n============================================================")
    print("MANUAL TRACKING ACCURACY SCORING")
    print("============================================================")
    print(f"Cell type:        {CELL_TYPE}")
    print(f"Manual GT CSV:    {MANUAL_GT_CSV}")
    print(f"Output directory: {VALIDATION_OUTPUT_DIR}")
    print(f"Z distance weight:{Z_DISTANCE_WEIGHT}")
    print(f"Max match dist:   {MAX_GT_MATCH_DISTANCE_3D}")
    print(f"Consecutive only: {REQUIRE_CONSECUTIVE_GT_TIMES}")

    VALIDATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_manual_gt(MANUAL_GT_CSV)

    print("\nManual GT summary:")
    print(f"Manual points: {len(gt)}")
    print(f"GT tracks:     {gt['gt_track_id'].nunique()}")
    print("\nPoints per GT track:")
    print(gt.groupby("gt_track_id")["time"].count())

    all_summaries = []
    all_per_gt = []

    for method in TRACKING_METHODS_TO_SCORE:
        for variant in TRACK_VARIANTS_TO_SCORE:
            tracks_csv = get_track_file(method, variant)

            if not tracks_csv.exists():
                print(f"\n[SKIP] Missing track file for method={method}, variant={variant}:")
                print(tracks_csv)
                continue

            result = score_one_track_file(gt=gt, method=method, variant=variant, tracks_csv=tracks_csv)

            prefix = f"{CELL_TYPE}_manual_tracking_{method}_{variant}"

            matches_csv = VALIDATION_OUTPUT_DIR / f"{prefix}_point_matches.csv"
            links_csv = VALIDATION_OUTPUT_DIR / f"{prefix}_link_scores.csv"
            per_gt_csv = VALIDATION_OUTPUT_DIR / f"{prefix}_per_gt_summary.csv"

            result["matches"].to_csv(matches_csv, index=False)
            result["link_scores"].to_csv(links_csv, index=False)
            result["per_gt"].to_csv(per_gt_csv, index=False)

            all_summaries.append(result["summary"])
            if not result["per_gt"].empty:
                all_per_gt.append(result["per_gt"])

            print("\nSummary:")
            print(result["summary"].T)

            print("\nSaved:")
            print(matches_csv)
            print(links_csv)
            print(per_gt_csv)

    if all_summaries:
        summary_all = pd.concat(all_summaries, ignore_index=True)
        summary_csv = VALIDATION_OUTPUT_DIR / f"{CELL_TYPE}_manual_tracking_score_summary_all.csv"
        summary_all.to_csv(summary_csv, index=False)

        print("\n============================================================")
        print("ALL TRACKING SCORE SUMMARY")
        print("============================================================")
        display_cols = [
            "tracking_method",
            "track_variant",
            "total_manual_points",
            "matched_point_rate",
            "total_scored_links",
            "correct_links",
            "broken_links",
            "wrong_inappropriate_links",
            "correct_link_rate",
            "broken_link_rate",
            "wrong_link_rate",
            "tracking_score_plus1_minus1_minus2",
            "link_jaccard_like",
        ]
        display_cols = [c for c in display_cols if c in summary_all.columns]
        print(summary_all[display_cols])
        print("\nSaved overall summary:")
        print(summary_csv)

    if all_per_gt:
        per_gt_all = pd.concat(all_per_gt, ignore_index=True)
        per_gt_all_csv = VALIDATION_OUTPUT_DIR / f"{CELL_TYPE}_manual_tracking_per_gt_summary_all.csv"
        per_gt_all.to_csv(per_gt_all_csv, index=False)
        print("\nSaved per-GT summary:")
        print(per_gt_all_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
