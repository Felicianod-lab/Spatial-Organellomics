# Fig.1 Liver and Pancreas PCA

## Purpose

Cross-organ principal component analysis for Figure 1. Reproduces the PCA scatter
plot in which cells are colored by label prefix:

| Label prefix | Color | Group |
|---|---|---|
| `P` | purple | Pancreas |
| `CNT` | pink | Liver |
| anything else | gray | Other |

The script retains only the data preparation, feature selection, PCA, sampling, and
plotting steps needed for that panel. Exploratory GMM, hierarchical clustering,
z-score, 3D, and hormone-cell-type plots from the original working script were removed.

## Inputs

**Processed-data archive:** https://doi.org/10.25378/janelia.32717316

| Script | Required file | Where the path is supplied |
|---|---|---|
| [`fig1_pca.py`](fig1_pca.py) | `fig1_liver_pancreas_cell_mask_features.csv` | `DATA_PATH` at [line 42](fig1_pca.py#L42), or pass `--data` |

The committed value of `DATA_PATH` is the placeholder `"...add path.../fig1_liver_pancreas_cell_mask_features.csv"`.
Either edit it or use the command-line flag.

### Feature selection

PCA is run on a curated set of quantitative organelle features. Columns listed in
`EXCLUDED_COLUMNS` are dropped, for the reasons documented inline above that list:

- metadata, identifiers, and spatial coordinates (`centroid-0`, `centroid-1`, `labels`, `label`, `Unnamed: 0`, `stack_folder`, `region_folder`)
- whole-cell `area`, so the PCA reflects organelle morphology rather than cell size
- hormone-derived measurements and hormone-based classifications, which are used for downstream annotation rather than to define the unsupervised space
- lipid-droplet morphology features, which are dominated by absence/sparsity in pancreatic cells
- large-peroxisome morphology features, which are largely restricted to endocrine cells

Mitochondrial features are retained after mitochondrial quality-control filtering and
are deliberately **not** excluded.

## Running

```bash
python fig1_pca.py --data /path/to/fig1_liver_pancreas_cell_mask_features.csv \
                   --output fig1_pca.png \
                   --sample-fraction 0.5 \
                   --random-state 42 \
                   --no-show
```

Available flags: `--data`, `--output`, `--sample-fraction`, `--random-state`, `--no-show`.

## Outputs

A single figure. Nothing is written unless an output path is given: `OUTPUT_PATH` is
`None` by default ([line 55](fig1_pca.py#L55)), so the plot is displayed only. Supply
`--output` or set `OUTPUT_PATH` to save. Saved figures use `dpi=300` and
`bbox_inches="tight"` ([line 351](fig1_pca.py#L351)).

No CSV tables are written by this script.

## Reproducibility notes

- `RANDOM_STATE = 42` ([line 45](fig1_pca.py#L45)) governs both the sampling and the PCA.
- `SAMPLE_FRACTION = 0.5` ([line 44](fig1_pca.py#L44)) — half the cells are plotted. Keep this value to reproduce the published panel.
- `MITO_COUNT_THRESHOLD = 20` ([line 46](fig1_pca.py#L46)) sets the mitochondrial quality-control cutoff.
- Plot styling is fixed to the published appearance: `FIGSIZE = (8, 8)`, point size 3, alpha 0.9, black edges at linewidth 0.02.
- PCA sign is not canonicalized; axis orientation may flip between runs on different package versions without changing the structure.

[Back to the analysis index](../README.md)

