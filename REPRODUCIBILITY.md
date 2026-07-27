# Reproducing the analyses

This guide records the software environment and commands needed to rerun the
dissertation workflow. Raw microscopy data, trained Cellpose weights, manual
injury masks, and generated intermediate tables are external inputs and are not
distributed in this source repository.

## Recorded environment

The pinned environment was captured on 27 July 2026 from the available
development laptop:

- macOS 26.5.2 (build 25F84)
- Apple arm64 architecture
- Python 3.11.6

The earlier Windows/PC environment that produced different accuracy values was
not preserved, so its exact package builds cannot be reconstructed reliably.
`requirements-reproducible.txt` records the environment available at final
submission. Use one environment consistently when comparing or reporting
results; do not combine outputs generated under different environments.

## Create the pinned environment

Using `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproducible.txt
python capture_environment.py
```

Alternatively, with Conda:

```bash
conda env create -f environment.yml
conda activate zebrafish-tracking
python capture_environment.py
```

`requirements.txt` remains the less restrictive dependency list for installing
on other platforms. Use `requirements-reproducible.txt` when reproducing the
reported analyses.

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

Run the modelling scripts on CPU where possible. The reported classifiers use
scikit-learn; GPU acceleration is not required. Tiny floating-point differences
can still occur across operating systems, processor architectures, BLAS
implementations, or package builds. These can change a selected feature or
hyperparameter when two candidates have nearly identical scores.

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
CZI files, trained segmentation weights, and manually annotated injury masks.

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

The final untreated classifier commands, including the model choices and 2,000
permutations, are stored in one executable launcher:

```bash
cd "Feature extraction"
bash run_best_models.sh
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

## Record each rerun

Before running an experiment, save a machine-readable environment report:

```bash
python capture_environment.py > environment_report.json
```

Keep this report beside the generated results. It records the platform, Python
version, dependency versions, and relevant reproducibility environment
variables.
