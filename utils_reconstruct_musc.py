import numpy as np
import pandas as pd
from skimage.measure import regionprops

from config import (
    MAX_Z_GAP,
    MAX_Z_SPAN,
    MIN_2D_MASK_AREA,
    MIN_3D_VOXELS,
    MIN_DETECTED_Z_SLICES,
    MIN_OVERLAP_FRACTION,
    MIN_IOU,
    MAX_CENTROID_XY_DISTANCE,
)


# Disjoint-set algorithm: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a

    def groups(self):
        grouped = {}

        for item in self.parent:
            root = self.find(item)
            grouped.setdefault(root, []).append(item)

        return list(grouped.values())


def extract_2d_nodes(mask_3d):
    nodes = {}
    nodes_by_z = {}

    Z = mask_3d.shape[0]

    for z in range(Z):
        slice_mask = mask_3d[z]
        labels = np.unique(slice_mask)
        labels = labels[labels != 0]

        for label in labels:
            coords = np.argwhere(slice_mask == label)
            area = len(coords)

            if area < MIN_2D_MASK_AREA:
                continue

            y_centroid = coords[:, 0].mean()
            x_centroid = coords[:, 1].mean()

            node_id = (z, int(label))

            nodes[node_id] = {
                "z": z,
                "label": int(label),
                "area": int(area),
                "centroid_y": float(y_centroid),
                "centroid_x": float(x_centroid),
            }

            nodes_by_z.setdefault(z, []).append(node_id)

    return nodes, nodes_by_z


def mask_overlap_metrics(mask_a, label_a, mask_b, label_b):
    a = mask_a == label_a
    b = mask_b == label_b

    intersection = np.logical_and(a, b).sum()

    if intersection == 0:
        return 0.0, 0.0

    area_a = a.sum()
    area_b = b.sum()
    union = area_a + area_b - intersection

    iou = intersection / union
    overlap_fraction = intersection / min(area_a, area_b)

    return float(iou), float(overlap_fraction)


def centroid_xy_distance(node_a, node_b):
    dy = node_a["centroid_y"] - node_b["centroid_y"]
    dx = node_a["centroid_x"] - node_b["centroid_x"]

    return float(np.sqrt(dx * dx + dy * dy))


# 3D image-processing reference: https://scikit-image.org/skimage-tutorials/lectures/three_dimensional_image_processing.html
def reconstruct_3d_objects_for_timepoint(mask_3d):
    """mask_3d: Z, Y, X Cellpose 2D masks."""

    mask_3d = mask_3d.astype(np.uint16)

    nodes, nodes_by_z = extract_2d_nodes(mask_3d)

    if len(nodes) == 0:
        return np.zeros_like(mask_3d, dtype=np.uint16), pd.DataFrame()

    uf = UnionFind(list(nodes.keys()))

    z_values = sorted(nodes_by_z.keys())

    for z in z_values:
        for gap in range(1, MAX_Z_GAP + 1):
            z_next = z + gap

            if z_next not in nodes_by_z:
                continue

            for node_a_id in nodes_by_z[z]:
                for node_b_id in nodes_by_z[z_next]:
                    node_a = nodes[node_a_id]
                    node_b = nodes[node_b_id]

                    xy_dist = centroid_xy_distance(node_a, node_b)

                    if xy_dist > MAX_CENTROID_XY_DISTANCE:
                        continue

                    label_a = node_a["label"]
                    label_b = node_b["label"]

                    iou, overlap_fraction = mask_overlap_metrics(
                        mask_3d[z],
                        label_a,
                        mask_3d[z_next],
                        label_b
                    )

                    should_join = (
                        overlap_fraction >= MIN_OVERLAP_FRACTION  # or for musc
                        and iou >= MIN_IOU
                    )

                    if should_join:
                        uf.union(node_a_id, node_b_id)

    groups = uf.groups()

    label_3d = np.zeros_like(mask_3d, dtype=np.uint16)
    rows = []

    new_label = 1

    for group in groups:
        z_list = sorted([nodes[node_id]["z"] for node_id in group])

        z_min = min(z_list)
        z_max = max(z_list)

        z_span = z_max - z_min + 1
        detected_z_slices = len(set(z_list))

        voxel_count = 0

        for node_id in group:
            z, label = node_id
            voxel_count += int((mask_3d[z] == label).sum())

        # Main MUSC-specific filter
        if z_span > MAX_Z_SPAN:
            continue

        if detected_z_slices < MIN_DETECTED_Z_SLICES:
            continue

        if voxel_count < MIN_3D_VOXELS:
            continue

        for node_id in group:
            z, label = node_id
            label_3d[z][mask_3d[z] == label] = new_label

        rows.append({
            "object_id_3d": new_label,
            "z_min_detected": z_min,
            "z_max_detected": z_max,
            "z_span": z_span,
            "detected_z_slices": detected_z_slices,
            "raw_2d_masks_joined": len(group),
            "voxel_count_from_2d_masks": voxel_count,
            "kept": True
        })

        new_label += 1

    reconstruction_df = pd.DataFrame(rows)

    return label_3d, reconstruction_df


def extract_3d_features(label_3d, raw_3d=None, file_name="sample", time_index=0):
    rows = []

    props = regionprops(label_3d, intensity_image=raw_3d)

    for p in props:
        z, y, x = p.centroid
        z_min, y_min, x_min, z_max, y_max, x_max = p.bbox

        coords = p.coords
        unique_z = np.unique(coords[:, 0])

        if raw_3d is not None:
            mean_intensity = float(p.mean_intensity)
            max_intensity = float(raw_3d[label_3d == p.label].max())
        else:
            mean_intensity = np.nan
            max_intensity = np.nan

        rows.append({
            "file": file_name,
            "time": int(time_index),
            "object_id_3d": int(p.label),

            "z": float(z),
            "y": float(y),
            "x": float(x),

            "volume_voxels": int(p.area),

            "z_min": int(z_min),
            "z_max": int(z_max - 1),
            "z_span": int(z_max - z_min),
            "detected_z_slices": int(len(unique_z)),

            "bbox_z_min": int(z_min),
            "bbox_y_min": int(y_min),
            "bbox_x_min": int(x_min),
            "bbox_z_max": int(z_max),
            "bbox_y_max": int(y_max),
            "bbox_x_max": int(x_max),

            "mean_intensity": mean_intensity,
            "max_intensity": max_intensity,
        })

    return pd.DataFrame(rows)
