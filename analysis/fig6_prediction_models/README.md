# Fig.6 Prediction models

## Purpose

Multilayer perceptron classifiers trained on single-cell organelle feature vectors, used
to predict (a) nutritional condition and organelle-defined hepatocyte category, and
(b) duration of Western-diet feeding as a model of early MASLD progression.

```text
Fig6_Experimental_Groups_Organelle_Features_Matrix.csv
   ├── MLP ──► nutritional condition (CNT / STV / WD)
   └── MLP ──► hepatocyte category (H1–H5)

Fig6_Early_MASLD_Model_Data_Matrix.csv
   └── MLP ──► Western-diet duration (6 classes: 0, 7, 17, 31, 42, 50 days)
```

## Inputs

**Processed-data archive:** https://doi.org/10.25378/janelia.32717316

| Script | Required file | Where the path is supplied |
|---|---|---|
| [`Fig6_MLP.py`](Fig6_MLP.py) | `Fig6_Experimental_Groups_Organelle_Features_Matrix.csv` | `DATA_PATH` at [line 20](Fig6_MLP.py#L20) |
| [`Fig6_Early_MASLD_MLP.py`](Fig6_Early_MASLD_MLP.py) | `Fig6_Early_MASLD_Model_Data_Matrix.csv` | `DATA_PATH` at [line 20](Fig6_Early_MASLD_MLP.py#L20) |

Both committed paths are the placeholder `"...add path..."` and must be edited before
running. Neither script has a command-line interface.

### Expected columns

Features are listed explicitly in the `FEATURE_COLUMNS` block of each script — roughly
100 columns spanning mitochondria (general plus subtypes 1 and 2), peroxisomes, and
lipid droplets, each with density, average area, aspect ratio, perimeter, percent total
area, solidity, circularity, and distance from cell edge.

Labels: `group` for nutritional condition and `Prediction` for hepatocyte category.
For the MASLD model, time point is parsed from the label strings by substring matching
on `7days_`, `18days_`, `31days_`, `42days_`, `50days_` ([lines 223–227](Fig6_Early_MASLD_MLP.py#L223-L227)).

**Note on class naming:** the label token is `18days_` but the published figure displays
that class as "17 days". This is documented inline at [line 32](Fig6_Early_MASLD_MLP.py#L32)
and is intentional — do not "fix" one without the other.

## Model configuration

| Setting | Fig6_MLP.py | Fig6_Early_MASLD_MLP.py |
|---|---|---|
| Architecture | 4-layer MLP: input → hidden → hidden → hidden → output, ReLU | same |
| Device | CPU (`torch.device("cpu")`) | CPU |
| Validation fraction | 0.20 | 0.20 |
| Batch size | 256 | 256 |
| Learning rate | 1e-5 | 1e-5 |
| Epochs | 2,000 (nutrient model); 4,000 (category model) | 8,000 |
| Classes | 3 conditions; 5 categories | 6 time points |
| Mitochondrial QC | — | `MINIMUM_MITO_COUNT = 10` |

Both scripts set `KMP_DUPLICATE_LIB_OK="TRUE"` at import to work around duplicate
OpenMP libraries in some local installations.

## Outputs

`Fig6_MLP.py` **writes no files.** Confusion-matrix heatmaps are displayed, and
validation accuracies for the nutrient and category models are printed to the console.
Save figures manually or add `savefig` calls if you need them on disk.

`Fig6_Early_MASLD_MLP.py` writes the filtered input matrix to
`FILTERED_DATA_SAVE_PATH` (`all_included.csv`, [line 21](Fig6_Early_MASLD_MLP.py#L21),
index included). Its confusion matrix is likewise displayed rather than saved.

## Reproducibility notes

- **No random seed is set in either script.** Weight initialization, the train/validation
  split, and batch shuffling are therefore not deterministic, and reported accuracies
  will vary between runs. This is a known limitation of these two scripts; treat the
  published accuracies as representative of a single run rather than exactly
  reproducible. Adding `torch.manual_seed()` and a fixed split seed would make future
  runs deterministic.
- Training runs are long by design (up to 8,000 epochs on CPU). Reduce the epoch count
  for a smoke test.
- Confusion matrices are row-normalized to percentages of the true class before display.
- Class label order in the plots is fixed by `TIMEPOINT_CLASS_LABELS`; keep that ordering
  consistent with the feature matrix.
- Neither script has a module docstring; behavior is documented here only.

[Back to the analysis index](../README.md)
