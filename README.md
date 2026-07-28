# Zebrafish 3D Cell Tracking and Analysis

Source code for a dissertation workflow that segments cells in zebrafish
time-lapse microscopy, reconstructs 3D objects, tracks them through time,
extracts cell- and fish-level features, and evaluates genotype and treatment
effects.

## Repository scope

This repository contains the software developed for the dissertation. Raw CZI
microscopy, TIFF stacks, trained Cellpose weights, generated figures, model
artefacts, and large derived tables are not included because they are either
study data, generated outputs, or third-party artefacts. The small metadata CSV
files required to define cohorts and frame intervals are included.

The required authorship statement is in
[`AUTHORSHIP_DECLARATION.txt`](AUTHORSHIP_DECLARATION.txt).

## Quick start

Python 3.11 was used during development.

```bash
git clone https://github.com/<username>/zebrafish-3d-cell-tracking-analysis.git
cd zebrafish-3d-cell-tracking-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

For reproduction of the submitted analyses, use the pinned environment and
follow [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md):

```bash
pip install -r requirements-reproducible.txt
python capture_environment.py
```

Edit `.env` or export the required variables before running the pipeline.
Large input data and model weights should remain outside Git:

```bash
export BATCH_CZI_PATH="/path/to/input.czi"
export BATCH_PROJECT_DIR="/path/to/output_directory"
export MUSC_CELLPOSE_MODEL="/path/to/musc_model"
export MACROPHAGE_CELLPOSE_MODEL="/path/to/macrophage_model"
export BATCH_CELL_TYPE="musc"
export TRACKING_METHOD="nearest"
```

The scripts read variables directly from the shell; `.env` is a documented
template and is not loaded automatically.

## Main workflow

Run commands from the repository root. Configuration is centralised in
`config.py`.

1. Convert CZI data and correct drift:

   ```bash
   python 0_czi_to_tzxy_tif.py
   ```

2. Segment one cell type with Cellpose:

   ```bash
   BATCH_CELL_TYPE=musc python 1_run_cellpose_2d.py
   ```

3. Reconstruct 3D objects:

   ```bash
   BATCH_CELL_TYPE=musc python 2_reconstruct3D_objects.py
   ```

4. For macrophages, detect clusters and exclusion regions:

   ```bash
   BATCH_CELL_TYPE=macrophage python 3b_detect_macrophage_clusters.py
   ```

5. Track reconstructed objects:

   ```bash
   BATCH_CELL_TYPE=musc TRACKING_METHOD=nearest python 03_track_3d_objects.py
   ```

6. Produce track QC tables and filtered tracks:

   ```bash
   python 5_track_qc_summary.py
   python 6_track_quality_features.py
   python 7_filtered_good_tracks.py
   ```

7. Extract biological features and run the analyses described below:

   ```bash
   bash "Feature extraction/feature_extraction.sh"
   ```

`batch_run.py` and `run_overnight_tracking.sh` automate repeated processing.
Set `CELL_TRACKING_DATA_ROOT`, `BATCH_OUTPUT_ROOT`, or `OVERNIGHT_DIR` instead
of editing machine-specific paths.

## Reproduce the final analyses

The small, analysis-ready tables used by the four final classifiers and final
model plots are included under [`analysis_data/`](analysis_data/). They are
separate from the excluded raw images and large intermediate cell tables.

```bash
bash reproduce_final_analysis.sh
bash reproduce_final_plots.sh
```

The classifier command uses seed 42 and 2,000 permutations by default. See
[`analysis_data/README.md`](analysis_data/README.md) and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for scope and environment details.

## Expected external inputs

- Raw Zeiss `.czi` time-lapse files.
- Custom Cellpose model weights for MUSC and macrophage segmentation.
- Cohort metadata CSVs with block/fish identifiers, genotype or condition, and
  frame intervals.
- Manual injury masks for injury-ROI analyses.
- Optional manual ground-truth tracks for tracker validation.

Generated outputs are written beneath `BATCH_PROJECT_DIR` or the corresponding
feature-analysis output directory.

## Archive table of contents

### Core segmentation, reconstruction, and tracking

| File | Description |
| --- | --- |
| `config.py` | Central paths, environment overrides, imaging parameters, reconstruction thresholds, and tracker settings. |
| `0_czi_to_tzxy_tif.py` | Reads CZI data lazily, estimates specimen drift, and writes corrected TCZYX TIFF stacks. |
| `1_run_cellpose_2d.py` | Applies a custom Cellpose model independently to each timepoint and Z plane. |
| `2_reconstruct3D_objects.py` | Reconstructs 3D labels from 2D masks and exports object measurements. |
| `3b_detect_macrophage_clusters.py` | Detects macrophage clusters and builds masks for region-aware tracking. |
| `03_track_3d_objects.py` | Runs nearest-neighbour, LAP, or keyhole 3D tracking. |
| `3c_view_cluster_mask.py` | Inspects macrophage cluster masks and related tracks in napari. |
| `4_view_in_napari.py` | Visualises raw images, reconstructed labels, and tracks in napari. |
| `5_track_qc_summary.py` | Calculates step distances and summarises raw track quality. |
| `6_track_quality_features.py` | Adds track-level quality, split/merge, and exclusion features. |
| `7_filtered_good_tracks.py` | Applies final track-quality rules and records exclusion reasons. |
| `9_score_manual_tracking_ground_truth.py` | Scores automatic tracks against manually annotated ground truth. |
| `10_manual_validation_summary.py` | Consolidates manual-validation results into a CSV summary. |
| `gt_tracking.py` | Creates and exports manual ground-truth point tracks with napari. |
| `batch_run.py` | Runs the main workflow over multiple CZI acquisitions. |
| `run_overnight_tracking.sh` | Shell launcher for resumable batch reconstruction and tracking. |
| `metadata.py` | Prints CZI dimensions and physical pixel sizes for configuration. |
| `utils_reconstruct_musc.py` | MUSC-specific 2D-to-3D reconstruction and measurements. |
| `utils_reconstruct_macrophage.py` | Macrophage-specific 2D-to-3D reconstruction and measurements. |

### Tracking implementation

| File | Description |
| --- | --- |
| `tracking_utils/common.py` | Shared input normalisation, assignment, gap closing, and track summaries. |
| `tracking_utils/nearest_tracker.py` | Nearest-neighbour tracking cost. |
| `tracking_utils/lap_tracker.py` | Linear-assignment tracking with distance and feature costs. |
| `tracking_utils/keyhole_tracker.py` | Direction-aware keyhole tracking cost. |
| `tracking_utils/__init__.py` | Public tracking utility exports. |

### Feature extraction and statistical analysis

| File | Description |
| --- | --- |
| `Feature extraction/01_extract_tracked_cell_features.py` | Extracts 3D morphology, intensity, movement, and track features. |
| `Feature extraction/01b_fish_level_features.py` | Aggregates tracked-cell features to fish-level summaries. |
| `Feature extraction/02_filter_outliers.py` | Flags implausible tracks using robust thresholds. |
| `Feature extraction/03_analyze_variability.py` | Analyses fish variability, ICC, effect sizes, PCA, and classification. |
| `Feature extraction/03b_recalculate_time_corrected_track_features.py` | Recalculates motion features using measured frame intervals. |
| `Feature extraction/03c_recalculate_time_corrected_track_features_capped.py` | Repeats time correction with a maximum elapsed-time window. |
| `Feature extraction/04_make_fish_level_features.py` | Builds general fish-level features and QC sensitivity summaries. |
| `Feature extraction/04b_make_constrained_fish_features.py` | Builds the constrained Model A fish-level feature table. |
| `Feature extraction/04c_make_model_b_d_fish_features.py` | Builds constrained Model B and D feature tables. |
| `Feature extraction/05_all_feature_fish_selection.py` | Performs broad feature screening, nested validation, PCA, and elimination. |
| `Feature extraction/05_fish_level_feature_selection.py` | Evaluates predefined and exhaustive fish-level feature subsets. |
| `Feature extraction/06_test_constrained_fish_separation.py` | Earlier constrained L1 fish-classification analysis. |
| `Feature extraction/06_test_constrained_fish_separation_final.py` | Final nested-LOFO logistic/elastic-net analysis and permutation test. |
| `Feature extraction/06b_test_constrained_cell_separation.py` | Cell-level classification with fish-held-out evaluation. |
| `Feature extraction/06c_test_constrained_fish_separation_svm.py` | Linear and RBF SVM fish-level analysis. |
| `Feature extraction/06d_test_constrained_fish_separation_l2_logistic.py` | L2-logistic wrapper using shared classifier code. |
| `Feature extraction/06e_test_constrained_fish_separation_lda_shrinkage.py` | Shrinkage-LDA wrapper using shared classifier code. |
| `Feature extraction/06f_test_constrained_fish_separation_calibrated_linear_svm.py` | Calibrated linear-SVM wrapper. |
| `Feature extraction/06g_summarize_classifier_results.py` | Combines classifier metrics and permutation results. |
| `Feature extraction/07_make_plots.py` | Produces final model-evaluation and feature-summary plots. |
| `Feature extraction/07b_make_pca_plots.py` | Produces PCA, loading, clustering, and composition plots. |
| `Feature extraction/08_extract_model_c_cluster_dynamics.py` | Extracts cluster-dynamics features for Model C. |
| `Feature extraction/08_make_best_feature_pca_biplot.py` | Creates PCA biplots from selected features. |
| `Feature extraction/09_injury_annotation.py` | Interactive napari tool for MMP injury-region annotation. |
| `Feature extraction/09_injury_annotation_liraglutide.py` | Liraglutide configuration wrapper for injury annotation. |
| `Feature extraction/10_injury_roi_feature_extraction.py` | Extracts distance-to-injury and injury-ROI movement features. |
| `Feature extraction/10_injury_roi_feature_extraction_liraglutide.py` | Liraglutide configuration wrapper for injury features. |
| `Feature extraction/10b_injury_roi_feature_extraction_musc_cap720.py` | Builds MUSC injury features with a 720-minute cap. |
| `Feature extraction/10c_merge_injury_features_into_fish_table.py` | Merges injury-derived variables into fish-level tables. |
| `Feature extraction/11_fit_frozen_untreated_model.py` | Fits and serialises a final model on untreated fish. |
| `Feature extraction/11_experiment2_apply_frozen_untreated_model.py` | Applies frozen untreated models to labelled experimental tables. |
| `Feature extraction/12_apply_frozen_untreated_model.py` | Scores a treatment cohort with a frozen model. |
| `Feature extraction/12_experiment_condition_specific_models.py` | Runs condition-specific frozen-model comparisons. |
| `Feature extraction/13_analyze_frozen_mmp_results.py` | Summarises MMP frozen-model scores and feature contributions. |
| `Feature extraction/14_audit_injury_roi_consistency.py` | Audits injury annotations and derived ROI consistency. |
| `Feature extraction/combine_feature_outputs.py` | Combines per-block feature tables by dataset. |
| `Feature extraction/extra_classifier_common.py` | Shared nested-validation code for additional classifiers. |
| `Feature extraction/extract_czi_frame_intervals.py` | Extracts acquisition intervals from CZI metadata. |

### Plot scripts

`Feature extraction/plot_scripts/` contains the final figure-generation
scripts. Each script reads previously generated CSV tables and writes figures:

| File | Description |
| --- | --- |
| `generate_all_final_model_plots.py` | Orchestrates the final model figure set. |
| `generate_combined_fish_block_feature_plots.py` | Generates combined fish/block feature figures. |
| `generate_cv_union_pca_plots.py` | Creates PCA plots from cross-validation-selected feature unions. |
| `generate_final_condition_plots.py` | Generates final treatment-condition comparisons. |
| `generate_frozen_pca_projections.py` | Generates PCA projections using frozen reference transformations. |
| `generate_six_group_fish_distributions.py` | Generates six-group fish distribution figures. |
| `investigate_musc_msd_drop.py` | Investigates the MUSC mean-squared-displacement decrease. |
| `make_combined_fish_block_feature_plots.py` | Implements combined fish/block feature plotting. |
| `make_directional_feature_stability_plots.py` | Plots stability of directional movement features. |
| `make_feature_selection_stability_plots.py` | Plots feature-selection frequencies and stability. |
| `make_feret_proxy_plots.py` | Plots Feret-diameter proxy measurements. |
| `make_fish_variability_plots.py` | Plots between-fish feature variability. |
| `make_frozen_untreated_pca_projection.py` | Projects treated fish into the untreated PCA space. |
| `make_msd_directionality_plots.py` | Plots MSD and directionality metrics. |
| `make_musc_msd_timecourse_excluding_fish.py` | Produces MUSC MSD sensitivity time courses. |
| `make_pca_component_loading_plots.py` | Plots PCA component loadings. |
| `make_pca_feature_contribution_plots.py` | Plots feature contributions to PCA components. |
| `make_shape_change_plots.py` | Plots changes in cell-shape metrics. |
| `make_shape_over_time_plots.py` | Plots longitudinal shape measurements. |
| `make_six_group_fish_distributions.py` | Implements six-group distribution plotting. |
| `make_z_movement_and_grouped_fish_plots.py` | Plots Z movement and grouped fish comparisons. |

### Pipeline launchers

| File | Description |
| --- | --- |
| `Feature extraction/feature_extraction.sh` | Runs feature extraction for all configured blocks. |
| `Feature extraction/feature_extraction_liraglutide.sh` | Configures feature extraction for Liraglutide data. |
| `Feature extraction/metadata.sh` | Creates a metadata template from discovered block folders. |
| `Feature extraction/run_mmp_feature_pipeline.sh` | Runs the complete MMP feature pipeline. |
| `Feature extraction/run_liraglutide_feature_pipeline.sh` | Runs the complete Liraglutide feature pipeline. |
| `Feature extraction/run_model_b_d_analyses.sh` | Builds and tests Models B and D. |
| `Feature extraction/run_model_c_outside_boundary_l1.sh` | Builds and tests Model C. |
| `Feature extraction/run_best_models.sh` | Runs the selected final classifier configurations. |
| `Feature extraction/run_additional_musc_l1_sensitivity.sh` | Runs additional MUSC L1 sensitivity analyses. |
| `Feature extraction/run_musc_cap720_model_screen.sh` | Screens MUSC models using 720-minute-capped tracks. |
| `Feature extraction/run_musc_cap720_injury_model_screen.sh` | Screens capped MUSC models with injury features. |
| `Feature extraction/run_frozen_mmp_analysis.sh` | Applies frozen untreated models to MMP data. |
| `Feature extraction/run_frozen_mmp_analysis_excluding_20240604_block02_20240702_block06.sh` | Runs the specified MMP exclusion sensitivity analysis. |
| `Feature extraction/run_frozen_liraglutide_analysis.sh` | Applies frozen untreated models to Liraglutide data. |

### Supplementary QC utilities

The scripts in `qc_checks/` are optional diagnostics and are not required for
the main pipeline:

| File | Description |
| --- | --- |
| `Mask_2dVS3dQC.py` | Compares 2D mask and reconstructed 3D-object counts. |
| `check.py` | Inspects large spatial jumps in a selected tracking table. |
| `check_completion.py` | Audits whether expected batch outputs are present. |
| `check_recovery.py` | Checks recovered TIFF structure and readable pages. |
| `check_shift.py` | Scores possible XY mask shifts against image intensity. |
| `overlay.py` | Produces raw-image and mask-boundary overlays. |
| `quick_fix.py` | Applies reviewed time-dependent shifts to label masks. |

### Metadata and repository files

| File | Description |
| --- | --- |
| `Feature extraction/block_metadata.csv` | Untreated cohort identifiers and metadata. |
| `Feature extraction/MMP_metadata.csv` | MMP cohort metadata and frame intervals. |
| `Feature extraction/MMP_analysis_metadata.csv` | Pre-specified MMP analysis cohort. |
| `Feature extraction/Liraglutide_metadata.csv` | Liraglutide cohort metadata and frame intervals. |
| `requirements.txt` | Cross-platform, unpinned Python dependencies. |
| `requirements-reproducible.txt` | Versions captured from the final macOS arm64 environment. |
| `environment.yml` | Conda definition for the pinned Python environment. |
| `REPRODUCIBILITY.md` | Environment, determinism, inputs, and rerun instructions. |
| `capture_environment.py` | Records platform and package versions for each rerun. |
| `analysis_data/` | Curated final fish-level inputs, result tables, coefficients, metadata, and checksums. |
| `reproduce_final_analysis.sh` | Recreates the four final classifier result tables. |
| `reproduce_final_plots.sh` | Recreates final-model stability and PCA plots. |
| `.env.example` | Portable path and runtime configuration examples. |
| `.gitignore` | Excludes raw data, generated outputs, caches, and local settings. |
| `AUTHORSHIP_DECLARATION.txt` | Required sole-authorship certification. |
| `create_submission_archive.sh` | Builds the source-only KEATS ZIP archive. |

## Reproducibility notes

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the pinned environment,
deterministic execution settings, required external inputs, and commands for
rerunning the final experiments.

- Randomised model scripts expose or fix random seeds.
- Feature selection is performed within training folds where required to avoid
  leakage.
- Metadata files retain frame intervals used for physical-time correction.
- Third-party algorithms are credited directly above the relevant functions.
- Absolute data locations are configurable; no main script depends on the
  original development machine.



