# Reproducing the analyses

This guide records the software environment and commands needed to rerun the
dissertation workflow.

## Reproducibility status

The complete workflow was developed and run on one Mac Studio environment;
no Windows or separate PC environment was used. The source code and libraries
specified in this repository reproduce the complete method when the required
external research data and Cellpose model inputs are available.

The final statistical analyses are directly reproducible from the included
`analysis_data/` package. A validation run reproduced all four reported balanced
accuracies exactly and regenerated all 52 final-model plots.

Full raw-image reproduction additionally requires the original microscopy data,
manual annotations and a compatible GPU for Cellpose training. Cellpose training
in this project used the Apple Silicon GPU through PyTorch MPS on the Mac Studio.

## Recorded environment

All development, image processing, Cellpose training, tracking, feature
extraction, modelling and final validation were performed on the Mac Studio
described below:

- macOS 26.5.2 (build 25F84)
- Apple arm64 architecture
- Python 3.11.6

`requirements-reproducible.txt` records the environment used for the final
validation run. Use this environment with the supplied CSV inputs to reproduce
the reported final results.

## Create the pinned environment

Using `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproducible.txt
python capture_environment.py
```

`requirements.txt` is the less restrictive cross-platform dependency
list. `requirements-reproducible.txt` is the captured pip environment and is
the preferred installation route for reproducing the reported analyses.
`environment.yml` is only an optional Conda convenience wrapper around that pip
file; it is not evidence that the original analysis was run with Conda.

## Deterministic model execution

The final modelling programs use a default random seed of 42. Pass it
explicitly and restrict numerical libraries to one thread when comparing
machines:

```bash
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
```

The reported classifiers use scikit-learn and do not require a GPU. The
pinned libraries and supplied analysis tables reproduced the submitted metrics
exactly. The single-thread settings below keep the rerun consistent with the
validated Mac Studio execution.

For a strict comparison, keep all of the following identical:

- Input CSV contents and row order
- Metadata and cohort inclusion files
- Python and dependency versions
- Random seed and permutation count
- Model options and hyperparameter grids
- Thread settings and processor architecture

## External inputs

Copy `.env.example` to `.env` for reference, then export the required paths.
The scripts read shell variables directly and do not automatically load `.env`.

```bash
export BATCH_CZI_PATH="/path/to/input.czi"
export BATCH_PROJECT_DIR="/path/to/output_directory"
export MUSC_CELLPOSE_MODEL="/path/to/musc_model"
export MACROPHAGE_CELLPOSE_MODEL="/path/to/macrophage_model"
```

The four cohort metadata CSV files supplied under `Feature extraction/` should
remain unchanged. Exact end-to-end reproduction also requires the original raw
CZI files and manually annotated injury masks. The final trained segmentation
models are published in the [`segmentation-models-v1.0` release](https://github.com/Lakshay-a/zebrafish-3d-cell-tracking-analysis/releases/tag/segmentation-models-v1.0);
their direct links and SHA-256 checksums are recorded in `README.md`.

## Rebuild the segmentation datasets and models

This stage is required only for full image-to-result reproduction. It requires
the original CZI acquisitions and manual Cellpose/Fiji annotations.

Set the external locations:

```bash
export CELLPOSE_SOURCE_PATHS="/path/to/czi_or_parent:/path/to/another_source"
export CELLPOSE_DATASET_DIR="/path/to/cellpose_dataset"
export CELLPOSE_INPUT_DATASET_DIR="/path/to/macrophage_initial"
export CELLPOSE_OUTPUT_DATASET_DIR="/path/to/macrophage_preprocessed"
export CELLPOSE_RAW_FRAMES_DIR="/path/to/manual_fiji_frames"
```

Create the acquisition-level datasets:

```bash
python model_training/create_dataset_withDriftCorrection.py
python model_training/create_macrophage_withDriftCorrection.py
python model_training/add_new_macrophage.py
python model_training/preprocess.py
```

The dataset builders preserve the recorded 70% training, 15% validation and
15% test split at CZI/acquisition level with seed 42. Splitting at acquisition
level prevents slices from the same acquisition appearing in multiple sets.

Convert GUI annotations and train:

```bash
python model_training/convert_test_npy_to_masks.py
python model_training/annotation_model_training.py
```

To continue from a saved pilot model:

```bash
export CELLPOSE_PRETRAINED_MODEL="/path/to/pilot_model"
python model_training/annotation_model_finetuning.py
```

Evaluate or inspect a trained model:

```bash
export CELLPOSE_MODEL_PATH="/path/to/trained_model"
python model_training/evaluate_model.py
python model_training/test_model.py
```

Cellpose model training was performed on the Mac Studio's Apple Silicon GPU
using the PyTorch MPS backend. Reproducing the training stage therefore requires
a compatible GPU environment; the closest match is the documented Apple
Silicon/MPS setup with the pinned libraries. All subsequent segmentation,
reconstruction, tracking and analysis commands are fully documented in this
repository.

## Rerun the image-to-feature workflow

Run the main commands from the repository root:

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
`3b_detect_macrophage_clusters.py` after reconstruction, and use
`TRACKING_METHOD=lap`. The repository also retains the keyhole tracker used
during tracker evaluation.

Use the condition-specific launchers to regenerate downstream feature tables:

```bash
bash "Feature extraction/run_mmp_feature_pipeline.sh"
bash "Feature extraction/run_liraglutide_feature_pipeline.sh"
```

## Rerun the reported classifier experiments

The curated final fish-level CSV inputs are supplied in `analysis_data/`.
From the repository root, recreate the four final classifiers, their fold
predictions, metrics, selected features, permutation tests, and summary:

```bash
bash reproduce_final_analysis.sh
```

The public launcher uses seed 42, one numerical thread, and 2,000 permutations.
For a quick installation test only, use:

```bash
PERMUTATIONS=10 bash reproduce_final_analysis.sh
```

Regenerate the final stability and PCA plots from the supplied final tables:

```bash
bash reproduce_final_plots.sh
```

The scripts called by this launcher default to seed 42. Their output directories
record fold predictions, metrics, selected features, and permutation results.

Fit the frozen untreated models and apply them without refitting to treatment
cohorts:

```bash
cd "Feature extraction"
bash run_additional_musc_l1_sensitivity.sh
bash run_frozen_mmp_analysis.sh
bash run_frozen_liraglutide_analysis.sh
```

The frozen-model launchers require their documented upstream fish-level tables
and saved `.joblib` models. Generated model files are excluded from Git and the
KEATS source archive, so either regenerate them using
`11_fit_frozen_untreated_model.py` and the supplied launcher or provide the
original model artefacts separately if exact binary reuse is required.

## Expected final-analysis outputs

`bash reproduce_final_analysis.sh` writes four result directories plus a
combined summary beneath `reproduced_analysis/final_best_models_1000_perm/`.
With the captured environment and supplied CSVs, the validated balanced
accuracies are:

| Final model | Balanced accuracy |
| --- | ---: |
| Macrophage all, Model B plus injury, calibrated linear SVM | 0.8397435897 |
| Macrophage outside boundary, Model B, calibrated linear SVM | 0.8782051282 |
| MuSC Model A plus injury, calibrated linear SVM | 0.9444444444 |
| MuSC Model A, L1 logistic regression | 0.8181818182 |

`bash reproduce_final_plots.sh` was validated to generate 52 PNG figures from
the included final tables. Compare supplied source-table checksums against
`analysis_data/MANIFEST.csv` before investigating numerical differences.

## Record each rerun

Before running an experiment, save a machine-readable environment report:

```bash
python capture_environment.py > environment_report.json
```

Keep this report beside the generated results. It records the platform, Python
version, dependency versions, and relevant reproducibility environment
variables.
