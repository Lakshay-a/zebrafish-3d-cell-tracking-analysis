# Curated final analysis data

This folder contains only the analysis-ready CSV files needed to rerun the four
final classifier analyses and regenerate their final stability and PCA plots.
It excludes raw microscopy, masks, per-object/per-timepoint tables, exploratory
outputs, and generated figures.

The original relative directory structure is retained because the final
analysis and plotting scripts use these paths. `MANIFEST.csv` records the size
and SHA-256 checksum of every supplied source table.

From the repository root, reproduce the final classifiers and summary table:

```bash
bash reproduce_final_analysis.sh
```

For a faster smoke test, reduce the permutation count:

```bash
PERMUTATIONS=10 bash reproduce_final_analysis.sh
```

Regenerate the final-model stability and PCA plots directly from the supplied
final tables:

```bash
bash reproduce_final_plots.sh
```

The default classifier run uses seed 42, one numerical thread, and 2,000
permutations, matching `Feature extraction/run_best_models.sh`. Outputs are
written to `reproduced_analysis/`, which is ignored by Git.

This package reproduces the final fish-level statistical analyses. Rebuilding
all upstream cell trajectories from images additionally requires the excluded
raw CZI data, trained Cellpose weights, masks, and large per-cell tables.
