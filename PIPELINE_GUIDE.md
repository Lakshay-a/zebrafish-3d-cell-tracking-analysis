# Pipeline guide

This guide shows where to enter the workflow according to the data available.
Run all commands from the repository root unless stated otherwise. Install the
pinned environment first by following `REPRODUCIBILITY.md`.

## Choose a starting point

| Data available | Start here | Main result |
| --- | --- | --- |
| Included `analysis_data/` CSV files only | [Final tables and plots](#final-tables-and-plots-from-included-csv-files) | Reproduced classifier tables and final plots |
| Fish-level feature tables | [Statistical modelling](#statistical-modelling-from-fish-level-tables) | Classifier metrics, predictions and selected features |
| Tracked-cell tables and cohort metadata | [Feature extraction](#feature-extraction-from-tracked-cell-tables) | Cell- and fish-level feature tables |
| Reconstructed 3D-object tables | [Tracking](#tracking-from-reconstructed-3d-objects) | Raw and filtered cell tracks |
| Cellpose 2D masks | [3D reconstruction](#3d-reconstruction-from-cellpose-masks) | Reconstructed 3D objects |
| Raw CZI files and trained Cellpose models | [Raw-image pipeline](#complete-pipeline-from-raw-czi-files) | CZI-to-feature workflow |
| Raw CZI files and manual annotations, but no trained model | [Cellpose training](#cellpose-dataset-preparation-and-training) | Trained segmentation model |

Large microscopy images, segmentation masks and manual annotations are external
to this repository. The final Cellpose models can be downloaded from the
[`segmentation-models-v1.0` GitHub release](https://github.com/Lakshay-a/zebrafish-3d-cell-tracking-analysis/releases/tag/segmentation-models-v1.0); exact direct links and
checksums are provided in `README.md`.

## Final tables and plots from included CSV files

This is the simplest fully packaged reproduction route. It does not require raw
images, Cellpose models or a GPU.

```bash
bash reproduce_final_analysis.sh
bash reproduce_final_plots.sh
```

Inputs are read from `analysis_data/`. Recreated tables are written beneath
`reproduced_analysis/`; plot destinations are documented in
`REPRODUCIBILITY.md`.

For a quick installation check rather than the submitted permutation run:

```bash
PERMUTATIONS=10 bash reproduce_final_analysis.sh
```

## Statistical modelling from fish-level tables

Use the analysis launchers in `Feature extraction/` when compatible fish-level
tables already exist. The principal public reproduction launcher remains
`reproduce_final_analysis.sh`; the individual model scripts are listed in the
main README.

Frozen untreated models are fitted with:

```bash
cd "Feature extraction"
python 11_fit_frozen_untreated_model.py
```

The fitted models can then be applied to the MMP or liraglutide cohorts:

```bash
bash run_frozen_mmp_analysis.sh
bash run_frozen_liraglutide_analysis.sh
```

Return to the repository root before running root-level commands.

## Feature extraction from tracked-cell tables

Required inputs:

- Filtered tracked-cell CSV tables
- Cohort metadata containing block, fish, condition and frame-interval fields
- Manual injury masks or annotations for injury-ROI analyses

Run the general feature workflow:

```bash
bash "Feature extraction/feature_extraction.sh"
```

For treatment-specific analyses:

```bash
bash "Feature extraction/run_mmp_feature_pipeline.sh"
bash "Feature extraction/run_liraglutide_feature_pipeline.sh"
```

The launchers call the numbered scripts in `Feature extraction/` in their
required order. Their outputs include cell-level features, fish-level summaries,
injury-region features and model-ready tables.

## Tracking from reconstructed 3D objects

Set the project location and cell type, then choose the tracker used for that
cell type.

MuSC tracking:

```bash
export BATCH_PROJECT_DIR="/path/to/processed_acquisition"
BATCH_CELL_TYPE=musc TRACKING_METHOD=nearest python 03_track_3d_objects.py
```

Macrophage tracking requires cluster detection first and uses LAP:

```bash
export BATCH_PROJECT_DIR="/path/to/processed_acquisition"
BATCH_CELL_TYPE=macrophage python 3b_detect_macrophage_clusters.py
BATCH_CELL_TYPE=macrophage TRACKING_METHOD=lap python 03_track_3d_objects.py
```

Then generate track-quality information and filtered tracks:

```bash
python 5_track_qc_summary.py
python 6_track_quality_features.py
python 7_filtered_good_tracks.py
```

The keyhole implementation is retained for tracker evaluation but is not the
main production tracker for either cell type.

## 3D reconstruction from Cellpose masks

Required inputs are the 2D labelled masks created by Cellpose and the associated
acquisition metadata.

```bash
export BATCH_PROJECT_DIR="/path/to/processed_acquisition"
BATCH_CELL_TYPE=musc python 2_reconstruct3D_objects.py
```

Replace `musc` with `macrophage` for macrophage masks. Continue with the
tracking section above.

## Complete pipeline from raw CZI files

Required inputs:

- Raw Zeiss CZI time-lapse file
- MuSC and/or macrophage Cellpose model
- Acquisition metadata
- Manual injury annotations if injury-region features will be calculated

Configure paths:

```bash
export BATCH_CZI_PATH="/path/to/input.czi"
export BATCH_PROJECT_DIR="/path/to/processed_acquisition"
export MUSC_CELLPOSE_MODEL="/path/to/musc_model"
export MACROPHAGE_CELLPOSE_MODEL="/path/to/macrophage_model"
```

Run the MuSC pipeline:

```bash
python 0_czi_to_tzxy_tif.py
BATCH_CELL_TYPE=musc python 1_run_cellpose_2d.py
BATCH_CELL_TYPE=musc python 2_reconstruct3D_objects.py
BATCH_CELL_TYPE=musc TRACKING_METHOD=nearest python 03_track_3d_objects.py
python 5_track_qc_summary.py
python 6_track_quality_features.py
python 7_filtered_good_tracks.py
```

For macrophages, use `BATCH_CELL_TYPE=macrophage`, run
`3b_detect_macrophage_clusters.py` after reconstruction, and track with
`TRACKING_METHOD=lap`.

Use `batch_run.py` or `run_overnight_tracking.sh` to process multiple
acquisitions after first validating the commands on one acquisition.

## Cellpose dataset preparation and training

This route is needed only when rebuilding the segmentation models. It requires
the original CZI acquisitions, manual Cellpose/Fiji annotations and compatible
GPU hardware. Training in this project used the Mac Studio Apple Silicon GPU
through PyTorch MPS.

Set the paths described in `model_training/README.md`, then prepare datasets:

```bash
python model_training/create_dataset_withDriftCorrection.py
python model_training/create_macrophage_withDriftCorrection.py
python model_training/add_new_macrophage.py
python model_training/preprocess.py
python model_training/convert_test_npy_to_masks.py
```

Train and evaluate:

```bash
python model_training/annotation_model_training.py
python model_training/evaluate_model.py
python model_training/test_model.py
```

See `model_training/README.md` for every required environment variable and for
the optional fine-tuning route.

## Optional quality-control and validation tools

- `4_view_in_napari.py` visually inspects reconstructed objects and tracks.
- `3c_view_cluster_mask.py` inspects macrophage cluster masks.
- `9_score_manual_tracking_ground_truth.py` scores tracking against manual
  ground truth.
- `10_manual_validation_summary.py` summarises manual validation.
- `qc_checks/` contains supplementary diagnostic and recovery scripts; these are
  not required for the standard pipeline.

## Configuration and further detail

- `config.py`: central path and processing configuration
- `.env.example`: example shell variables
- `README.md`: archive table of contents and file descriptions
- `REPRODUCIBILITY.md`: pinned environment, deterministic settings and expected
  outputs
- `model_training/README.md`: segmentation dataset and training instructions
