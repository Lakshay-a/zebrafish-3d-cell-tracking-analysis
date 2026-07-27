from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import tifffile as tiff
import zarr
from qtpy.QtWidgets import (
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.draw import polygon



# Change only these settings if needed


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

BLOCKS_ROOT = Path(os.environ.get(
    "INJURY_BLOCKS_ROOT",
    PROJECT_ROOT / "overnight_batch_outputs" / "MMP",
)).expanduser().resolve()

OUTPUT_ROOT = Path(os.environ.get(
    "INJURY_ANNOTATION_ROOT",
    HERE / "manual_injury_annotations_mmp",
)).expanduser().resolve()

MUSC_MODEL_A_TABLE = Path(os.environ.get(
    "INJURY_MUSC_TABLE",
    HERE / "constrained_fish_features_time_corrected_mmp" / "musc"
    / "constrained_fish_level_mean_median.csv",
)).expanduser().resolve()

MACROPHAGE_MODEL_A_TABLE = Path(os.environ.get(
    "INJURY_MACROPHAGE_ALL_TABLE",
    HERE / "constrained_fish_features_time_corrected_mmp" / "macrophage_all"
    / "constrained_fish_level_mean_median.csv",
)).expanduser().resolve()

EXCLUDED_FISH = {"20240604_block06"}

MUSC_CHANNEL = 0
MACROPHAGE_CHANNEL = 1
INITIAL_TIME = 0
SKIP_ALREADY_SAVED = True

TIFF_NAMES = [
    "raw_TCZYX_drift_corrected.tif",
    "raw_TCZYX_drift_corrected.tiff",
    "raw_TCZYX.tif",
    "raw_TCZYX.tiff",
]

DEFAULT_XY_UM = 0.2959437779017855
XY_OVERRIDES_UM = {}




def detect_fish_column(df: pd.DataFrame) -> str:
    candidates = [
        "block_name",
        "fish_id",
        "block",
        "source_block",
        "sample_id",
    ]
    lookup = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"Could not find fish/block column in {list(df.columns)}"
    )


def fish_list() -> list[str]:
    """Annotate the union of fish used by the MUSC and macrophage analyses."""
    fish_names: set[str] = set()

    for table_path in (MUSC_MODEL_A_TABLE, MACROPHAGE_MODEL_A_TABLE):
        df = pd.read_csv(table_path, low_memory=False)
        fish_col = detect_fish_column(df)
        fish_names.update(
            df[fish_col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

    return sorted(
        name
        for name in fish_names
        if name and name not in EXCLUDED_FISH
    )


def find_tiff(block_dir: Path) -> Path | None:
    for name in TIFF_NAMES:
        matches = sorted(block_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def open_tiff_array(path: Path):
    with tiff.TiffFile(path) as tif:
        axes = str(tif.series[0].axes).upper()

    try:
        array = tiff.memmap(path)
        store = None
    except Exception:
        store = tiff.imread(path, aszarr=True)
        array = zarr.open(store, mode="r")

    if len(axes) != array.ndim:
        raise ValueError(
            f"Axes '{axes}' do not match TIFF shape {array.shape}"
        )
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"TIFF must contain Y and X axes: {axes}")

    return array, axes, store


def axis_size(array, axes: str, axis: str) -> int:
    return int(array.shape[axes.index(axis)]) if axis in axes else 1


def read_projection(
    array,
    axes: str,
    time_index: int,
    channel_index: int,
) -> np.ndarray:
    index = []
    remaining = []

    for axis in axes:
        if axis == "T":
            index.append(time_index)
        elif axis == "C":
            index.append(channel_index)
        elif axis in {"Z", "Y", "X"}:
            index.append(slice(None))
            remaining.append(axis)
        else:
            index.append(0)

    data = np.asarray(array[tuple(index)])
    wanted = [axis for axis in ("Z", "Y", "X") if axis in remaining]
    data = np.transpose(data, [remaining.index(axis) for axis in wanted])

    if data.ndim == 3:
        return np.max(data, axis=0)
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected ZYX or YX data, got {data.shape}")


def build_time_stack(array, axes: str, channel: int, status) -> np.ndarray:
    n_time = axis_size(array, axes, "T")
    n_channels = axis_size(array, axes, "C")

    if channel >= n_channels:
        raise IndexError(
            f"Requested channel {channel}; TIFF has {n_channels} channels"
        )

    frames = []
    for time_index in range(n_time):
        status(f"Creating max-Z stack: {time_index + 1}/{n_time}")
        frames.append(
            read_projection(array, axes, time_index, channel)
        )
    return np.stack(frames, axis=0)


def contrast_limits(stack: np.ndarray) -> tuple[float, float]:
    sample = stack
    if stack.shape[0] > 20:
        sample = stack[
            np.linspace(0, stack.shape[0] - 1, 20, dtype=int)
        ]
    values = sample[np.isfinite(sample)]
    low, high = np.percentile(values, [1, 99.8])
    if high <= low:
        high = low + 1
    return float(low), float(high)


# Polygon rasterisation: https://scikit-image.org/docs/stable/api/skimage.draw.html
def polygons_to_mask(
    polygon_data: list[np.ndarray],
    shape_yx: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape_yx, dtype=bool)

    for vertices in polygon_data:
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[0] < 3:
            continue
        if vertices.shape[1] != 2:
            raise ValueError(
                "ROI polygons must use 2D YX coordinates."
            )
        rr, cc = polygon(
            vertices[:, 0],
            vertices[:, 1],
            shape=shape_yx,
        )
        mask[rr, cc] = True

    return mask


def normalise(image: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    values = image[np.isfinite(image)]
    low, high = np.percentile(values, [1, 99.8])
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


class Annotator:
    def __init__(self):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        self.fish = fish_list()
        self.index = 0
        self.current_fish = None
        self.current_tiff = None
        self.current_array = None
        self.current_axes = None
        self.current_store = None
        self.musc_stack = None
        self.mac_stack = None
        self.roi_layer = None

        self.viewer = napari.Viewer(
            title="MMP manual injury ROI annotation"
        )
        self.viewer.dims.ndisplay = 2

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        instructions = QLabel(
            "Scroll through time if needed.\n"
            "Select the yellow Injury ROI layer.\n"
            "Use the polygon tool to draw the injury region.\n"
            "Multiple polygons are allowed and will be combined."
        )
        instructions.setWordWrap(True)

        save_button = QPushButton("Save ROI and next fish")
        clear_button = QPushButton("Clear polygons")
        skip_button = QPushButton("Skip fish")

        save_button.clicked.connect(self.save_next)
        clear_button.clicked.connect(self.clear)
        skip_button.clicked.connect(self.skip)

        layout = QVBoxLayout()
        layout.addWidget(instructions)
        layout.addWidget(self.status_label)
        layout.addWidget(clear_button)
        layout.addWidget(save_button)
        layout.addWidget(skip_button)
        layout.addStretch(1)

        panel = QWidget()
        panel.setLayout(layout)
        self.viewer.window.add_dock_widget(
            panel,
            area="right",
            name="Annotation controls",
        )

        self.viewer.bind_key("s", overwrite=True)(
            lambda viewer: self.save_next()
        )
        self.viewer.bind_key("c", overwrite=True)(
            lambda viewer: self.clear()
        )
        self.viewer.bind_key("n", overwrite=True)(
            lambda viewer: self.skip()
        )

        self.load_next()

    def status(self, text: str):
        message = (
            f"[{self.index + 1}/{len(self.fish)}] "
            f"{self.current_fish or ''}\n{text}"
        )
        self.status_label.setText(message)
        print(message, flush=True)

    def annotation_dir(self, fish_name: str) -> Path:
        return OUTPUT_ROOT / fish_name

    def mask_path(self, fish_name: str) -> Path:
        return (
            self.annotation_dir(fish_name)
            / "manual_injury_roi_mask_YX.tif"
        )

    def close_store(self):
        if self.current_store is not None:
            close = getattr(self.current_store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.current_store = None
        self.current_array = None

    def load_next(self):
        while self.index < len(self.fish):
            fish_name = self.fish[self.index]
            if (
                SKIP_ALREADY_SAVED
                and self.mask_path(fish_name).exists()
            ):
                self.index += 1
                continue
            self.load_fish(fish_name)
            return

        self.viewer.layers.clear()
        self.status_label.setText(
            "All fish completed.\n"
            f"Summary: {OUTPUT_ROOT / 'annotation_summary.csv'}"
        )
        QMessageBox.information(
            self.viewer.window._qt_window,
            "Finished",
            "All fish have been processed.",
        )

    def load_fish(self, fish_name: str):
        self.current_fish = fish_name
        self.close_store()
        self.viewer.layers.clear()
        self.musc_stack = None
        self.mac_stack = None

        block_dir = BLOCKS_ROOT / fish_name
        tiff_path = find_tiff(block_dir)

        if tiff_path is None:
            self.record("missing_tiff")
            self.index += 1
            self.load_next()
            return

        self.current_tiff = tiff_path
        self.status(f"Opening {tiff_path}")

        try:
            (
                self.current_array,
                self.current_axes,
                self.current_store,
            ) = open_tiff_array(tiff_path)

            self.musc_stack = build_time_stack(
                self.current_array,
                self.current_axes,
                MUSC_CHANNEL,
                lambda text: self.status(f"MUSC: {text}"),
            )

            n_channels = axis_size(
                self.current_array,
                self.current_axes,
                "C",
            )
            if MACROPHAGE_CHANNEL < n_channels:
                self.mac_stack = build_time_stack(
                    self.current_array,
                    self.current_axes,
                    MACROPHAGE_CHANNEL,
                    lambda text: self.status(
                        f"Macrophage: {text}"
                    ),
                )
            else:
                self.mac_stack = None
                self.status(
                    "No macrophage channel is present; "
                    "this fish will be annotated from MUSC only."
                )

        except Exception as exc:
            self.record("load_failed", notes=str(exc))
            QMessageBox.critical(
                self.viewer.window._qt_window,
                "Load failed",
                f"{fish_name}\n\n{exc}",
            )
            self.index += 1
            self.load_next()
            return

        self.viewer.add_image(
            self.musc_stack,
            name="MUSC max-Z",
            colormap="green",
            blending="additive",
            contrast_limits=contrast_limits(self.musc_stack),
        )
        if self.mac_stack is not None:
            self.viewer.add_image(
                self.mac_stack,
                name="Macrophage max-Z",
                colormap="magenta",
                blending="additive",
                contrast_limits=contrast_limits(self.mac_stack),
            )
        self.roi_layer = self.viewer.add_shapes(
            name="Injury ROI",
            edge_color="yellow",
            face_color=[1, 1, 0, 0.18],
            edge_width=2,
            ndim=2,
        )
        self.roi_layer.mode = "add_polygon"

        initial = min(INITIAL_TIME, self.musc_stack.shape[0] - 1)
        self.viewer.dims.set_current_step(0, initial)
        self.viewer.reset_view()

        channel_note = (
            "MUSC + macrophage channels available."
            if self.mac_stack is not None
            else "MUSC-only TIFF: define the injury from disrupted green tissue."
        )
        self.status(
            f"TIFF axes: {self.current_axes}\n"
            f"{channel_note}\n"
            "Draw the injury ROI, then save."
        )

    def clear(self):
        if self.roi_layer is not None:
            self.roi_layer.data = []
            self.roi_layer.mode = "add_polygon"

    def save_next(self):
        if self.roi_layer is None or self.current_fish is None:
            return

        polygons = [
            np.asarray(vertices, dtype=float)
            for vertices in self.roi_layer.data
        ]
        if not polygons:
            QMessageBox.warning(
                self.viewer.window._qt_window,
                "No ROI",
                "Draw at least one polygon before saving.",
            )
            return

        mask = polygons_to_mask(
            polygons,
            tuple(self.musc_stack.shape[-2:]),
        )
        if not mask.any():
            QMessageBox.warning(
                self.viewer.window._qt_window,
                "Empty ROI",
                "The drawn polygons produced an empty mask.",
            )
            return

        fish_name = self.current_fish
        output_dir = self.annotation_dir(fish_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = output_dir / "manual_injury_roi_mask_YX.tif"
        polygon_path = (
            output_dir / "manual_injury_roi_polygons.json"
        )
        metadata_path = (
            output_dir / "manual_injury_roi_metadata.json"
        )
        preview_path = (
            output_dir / "manual_injury_roi_preview.png"
        )

        tiff.imwrite(mask_path, mask.astype(np.uint8))
        polygon_path.write_text(
            json.dumps(
                {
                    "fish_name": fish_name,
                    "coordinate_order": "YX",
                    "polygons_yx": [
                        vertices.tolist() for vertices in polygons
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        xy_um = XY_OVERRIDES_UM.get(
            fish_name,
            DEFAULT_XY_UM,
        )
        area_pixels = int(mask.sum())
        area_um2 = float(area_pixels * xy_um * xy_um)
        time_index = int(self.viewer.dims.current_step[0])

        metadata_path.write_text(
            json.dumps(
                {
                    "fish_name": fish_name,
                    "source_tiff": str(self.current_tiff),
                    "source_axes": self.current_axes,
                    "source_shape": list(self.current_array.shape),
                    "musc_channel_index": MUSC_CHANNEL,
                    "macrophage_channel_index": (
                        MACROPHAGE_CHANNEL
                        if self.mac_stack is not None
                        else None
                    ),
                    "macrophage_channel_available": (
                        self.mac_stack is not None
                    ),
                    "roi_static_across_time": True,
                    "displayed_time_when_saved": time_index,
                    "roi_area_pixels": area_pixels,
                    "xy_pixel_size_um": xy_um,
                    "roi_area_um2": area_um2,
                    "saved_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.save_preview(preview_path, mask)
        self.record(
            "completed",
            area_pixels=area_pixels,
            area_um2=area_um2,
            mask_path=str(mask_path),
        )

        print(
            f"[SAVED] {fish_name}: {mask_path}",
            flush=True,
        )
        self.index += 1
        self.load_next()

    def save_preview(self, path: Path, mask: np.ndarray):
        green = normalise(np.max(self.musc_stack, axis=0))

        if self.mac_stack is not None:
            magenta = normalise(np.max(self.mac_stack, axis=0))
        else:
            magenta = np.zeros_like(green)

        rgb = np.zeros((*green.shape, 3), dtype=float)
        rgb[..., 0] = magenta
        rgb[..., 1] = green
        rgb[..., 2] = magenta

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.contour(mask.astype(float), levels=[0.5], linewidths=2)
        ax.set_title(f"{self.current_fish}: injury ROI")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def skip(self):
        if self.current_fish is not None:
            self.record("skipped_by_user")
        self.index += 1
        self.load_next()

    def record(
        self,
        status: str,
        area_pixels=None,
        area_um2=None,
        mask_path="",
        notes="",
    ):
        summary_path = OUTPUT_ROOT / "annotation_summary.csv"
        row = pd.DataFrame(
            [
                {
                    "fish_name": self.current_fish,
                    "status": status,
                    "source_tiff": str(self.current_tiff or ""),
                    "mask_path": mask_path,
                    "roi_area_pixels": area_pixels,
                    "roi_area_um2": area_um2,
                    "notes": notes,
                    "updated_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            ]
        )

        if summary_path.exists():
            old = pd.read_csv(summary_path, low_memory=False)
            old = old[
                old["fish_name"].astype(str)
                != str(self.current_fish)
            ]
            row = pd.concat([old, row], ignore_index=True)

        row.sort_values("fish_name").to_csv(
            summary_path,
            index=False,
        )


def main():
    if not BLOCKS_ROOT.exists():
        raise FileNotFoundError(BLOCKS_ROOT)
    if not MUSC_MODEL_A_TABLE.exists():
        raise FileNotFoundError(MUSC_MODEL_A_TABLE)
    if not MACROPHAGE_MODEL_A_TABLE.exists():
        raise FileNotFoundError(MACROPHAGE_MODEL_A_TABLE)

    annotator = Annotator()
    napari.run()


if __name__ == "__main__":
    main()
