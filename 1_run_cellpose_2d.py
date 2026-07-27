import numpy as np
import tifffile as tiff
from skimage.segmentation import relabel_sequential
from cellpose import models
import torch



from config import (
    RAW_TCZYX_TIF,
    CELLPOSE_MASKS_TZYX_TIF,
    CUSTOM_CELLPOSE_MODEL,
    USE_GPU,
    CELLPOSE_DIAMETER,
    FLOW_THRESHOLD,
    CELLPROB_THRESHOLD,
    NORMALIZE,
    CELLPOSE_BATCH_SIZE,
    RESAMPLE,
    CELL_TYPE,
    CHANNEL_INDEX,
)
print("[GPU CHECK] USE_GPU from config:", USE_GPU)
print("[GPU CHECK] torch version:", torch.__version__)
print("[GPU CHECK] CUDA available:", torch.cuda.is_available())
print("[GPU CHECK] MPS available:", torch.backends.mps.is_available())

def main():
    print(f"[INFO] Loading raw stack: {RAW_TCZYX_TIF}")
    raw = tiff.imread(RAW_TCZYX_TIF)

    print(f"[INFO] Loaded raw shape: {raw.shape}")

    if raw.ndim == 5:
        # Expected shape: T,C,Z,Y,X
        T, C, Z, Y, X = raw.shape

        if CHANNEL_INDEX >= C:
            raise ValueError(
                f"CHANNEL_INDEX={CHANNEL_INDEX} but raw only has {C} channels. "
                f"Raw shape: {raw.shape}"
            )

        raw_tzyx = raw[:, CHANNEL_INDEX, :, :, :]

    elif raw.ndim == 4:
        # Backward compatibility: already T,Z,Y,X
        T, Z, Y, X = raw.shape
        raw_tzyx = raw

    else:
        raise ValueError(
            f"Expected raw shape T,C,Z,Y,X or T,Z,Y,X. Got {raw.shape}"
        )

    masks_4d = np.zeros((T, Z, Y, X), dtype=np.uint16)

    print(
        f"[INFO] Running {CELL_TYPE} segmentation "
        f"using CHANNEL_INDEX={CHANNEL_INDEX}"
    )
    print(f"[INFO] Selected raw_tzyx shape: T={T}, Z={Z}, Y={Y}, X={X}")
    print(f"[INFO] Loading custom Cellpose model: {CUSTOM_CELLPOSE_MODEL}")

    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model=str(CUSTOM_CELLPOSE_MODEL)
    )
    print("[GPU CHECK] Cellpose model device:", getattr(model, "device", "unknown"))

    for t_idx in range(T):
        print(f"[INFO] Processing {CELL_TYPE} timepoint {t_idx + 1}/{T}")

        imgs = [raw_tzyx[t_idx, z_idx] for z_idx in range(Z)]

        for z_idx, img in enumerate(imgs):
            if img.ndim != 2:
                raise ValueError(
                    f"Expected 2D image at T={t_idx}, Z={z_idx}, got shape {img.shape}"
                )

        result = model.eval(
            imgs,
            batch_size=CELLPOSE_BATCH_SIZE,
            diameter=CELLPOSE_DIAMETER,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
            normalize=NORMALIZE,
            # resample=RESAMPLE,
        )
        # result = model.eval(
        #     imgs,
        #     # channels=[0, 0],
        #     # channel_axis=None,
        #     # z_axis=None,
        #     diameter=CELLPOSE_DIAMETER,
        #     cellprob_threshold=CELLPROB_THRESHOLD,
        #     flow_threshold=FLOW_THRESHOLD,
        #     normalize=NORMALIZE,
        #     resample=RESAMPLE,
        # )

        masks = result[0]

        if isinstance(masks, list):
            masks = np.stack(masks, axis=0)

        if masks.shape != (Z, Y, X):
            raise ValueError(
                f"Expected mask shape {(Z, Y, X)}, got {masks.shape}"
            )

        empty_z_slices = []
        mask_counts = []

        # for z_idx in range(Z):
        #     n_masks = len(np.unique(masks[z_idx])) - 1
        #     mask_counts.append(n_masks)

        #     if n_masks == 0:
        #         empty_z_slices.append(z_idx)

        # print(
        #     f"[MASK QC] {CELL_TYPE} T={t_idx}: "
        #     f"total masks={sum(mask_counts)}, "
        #     f"empty Z slices={len(empty_z_slices)}/{Z}"
        # )

        # if len(empty_z_slices) > 0:
        #     print(f"[MASK QC] Empty Z slices at T={t_idx}: {empty_z_slices}")

        for z_idx in range(Z):
            relabelled, _, _ = relabel_sequential(masks[z_idx].astype(np.uint16))
            masks_4d[t_idx, z_idx] = relabelled.astype(np.uint16)

    print(f"[INFO] Saving {CELL_TYPE} Cellpose masks: {CELLPOSE_MASKS_TZYX_TIF}")
    tiff.imwrite(CELLPOSE_MASKS_TZYX_TIF, masks_4d)

    print("[DONE] Cellpose 2D inference complete.")


if __name__ == "__main__":
    main()