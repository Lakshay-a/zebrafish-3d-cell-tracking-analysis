# Dataset preparation and Cellpose model training

These scripts reproduce acquisition-level dataset splitting, drift-corrected
slice extraction, macrophage preprocessing, Cellpose annotation conversion,
model training/fine-tuning, inference and evaluation. Dataset images, CZI files,
manual annotations and trained weights are external and are not committed to
the Git history. The final models are published as assets in the
[`segmentation-models-v1.0` GitHub release](https://github.com/Lakshay-a/zebrafish-3d-cell-tracking-analysis/releases/tag/segmentation-models-v1.0).

## Configuration

Set paths in the shell before running a script:

```bash
export CELLPOSE_PROJECT_ROOT="/path/to/project"
export CELLPOSE_SOURCE_PATHS="/path/to/source1:/path/to/source2"
export CELLPOSE_DATASET_DIR="/path/to/cellpose_dataset"
export CELLPOSE_INPUT_DATASET_DIR="/path/to/macrophage_initial"
export CELLPOSE_OUTPUT_DATASET_DIR="/path/to/macrophage_preprocessed"
export CELLPOSE_RAW_FRAMES_DIR="/path/to/manual_fiji_frames"
export CELLPOSE_MODEL_PATH="/path/to/trained_model"
export CELLPOSE_PRETRAINED_MODEL="/path/to/pilot_model"
```

Separate multiple `CELLPOSE_SOURCE_PATHS` entries with a colon on the
documented macOS environment.

## Files

| File | Purpose |
| --- | --- |
| `create_dataset_withDriftCorrection.py` | Builds the muSC dataset using acquisition-level 70/15/15 splitting and drift-corrected slice sampling. |
| `create_macrophage_withDriftCorrection.py` | Builds the initial macrophage dataset with the same split strategy. |
| `add_new_macrophage.py` | Appends later macrophage acquisitions without changing existing split assignments. |
| `manual_macrophages.py` | Imports manually exported Fiji macrophage frames. |
| `preprocess.py` | Applies percentile scaling and gamma preprocessing to macrophage images. |
| `annotation_model_training.py` | Trains the initial Cellpose annotation model from `cpsam`. |
| `annotation_model_finetuning.py` | Continues training from an external pilot model with learning-rate reduction and early stopping. |
| `convert_test_npy_to_masks.py` | Converts Cellpose GUI `_seg.npy` annotations to labelled TIFF masks. |
| `evaluate_model.py` | Evaluates instance masks using IoU matching and saves metrics and overlays. |
| `test_model.py` | Runs a trained model on the held-out image set. |
| `count_annotated_slices.py` | Counts Cellpose GUI annotations in a selected folder. |
| `mask_check.py` | Confirms that NPY and TIFF annotations contain identical masks. |
| `cleaning.py` | Moves auxiliary Cellpose outputs away from training images. |

The split seed is 42 and split fractions are 70% training, 15% validation and
15% test at acquisition/CZI level. Run scripts from the repository root with
the pinned environment described in `REPRODUCIBILITY.md`.

All model development and training was performed on the documented Mac Studio;
no Windows environment was used. Cellpose retraining used its Apple Silicon GPU
through PyTorch MPS and therefore requires compatible GPU hardware.
