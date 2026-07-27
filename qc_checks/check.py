import pandas as pd
import numpy as np

tracks = pd.read_csv(
    "/Users/lakshayarora/Desktop/Lakshay Dissertation/Custom cellpose/Final_approach/3d tracking/macrophage_tracking_outputs_outside_boundary/macrophage_tracks_lap.csv"
)

rows = []

for track_id, g in tracks.groupby("track_id"):
    g = g.sort_values("time")

    prev = None
    for _, row in g.iterrows():
        if prev is not None:
            dt = int(row["time"]) - int(prev["time"])
            dz = abs(float(row["centroid_z"]) - float(prev["centroid_z"]))
            dy = float(row["centroid_y"]) - float(prev["centroid_y"])
            dx = float(row["centroid_x"]) - float(prev["centroid_x"])
            dxy = np.sqrt(dx**2 + dy**2)

            rows.append({
                "track_id": track_id,
                "time0": int(prev["time"]),
                "time1": int(row["time"]),
                "dt": dt,
                "z0": float(prev["centroid_z"]),
                "z1": float(row["centroid_z"]),
                "dz": dz,
                "dxy": dxy,
                "allowed_dz_current_logic": 5.0 * dt,
            })

        prev = row

jumps = pd.DataFrame(rows)

print("\n[ALL JUMPS SUMMARY]")
print(jumps[["dt", "dz", "dxy"]].describe())

print("\n[CONSECUTIVE FRAME Z JUMPS > 5]")
bad_dt1 = jumps[(jumps["dt"] == 1) & (jumps["dz"] > 5)]
print("count:", len(bad_dt1))
print(bad_dt1.sort_values("dz", ascending=False).head(30))

print("\n[GAP-CLOSING Z JUMPS > 5]")
gap_jumps = jumps[(jumps["dt"] > 1) & (jumps["dz"] > 5)]
print("count:", len(gap_jumps))
print(gap_jumps.sort_values("dz", ascending=False).head(30))

print("\n[JUMPS EXCEEDING CURRENT ALLOWED DZ]")
too_large = jumps[jumps["dz"] > jumps["allowed_dz_current_logic"]]
print("count:", len(too_large))
print(too_large.sort_values("dz", ascending=False).head(30))