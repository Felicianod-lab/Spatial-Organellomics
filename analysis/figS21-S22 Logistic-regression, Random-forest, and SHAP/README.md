# Fig. S21–S22 Logistic regression, random forest, and SHAP

## Purpose

Interpretable-model counterparts to the Figure 6 neural networks. Logistic regression and random
forest classifiers are trained on the same organelle feature matrix, and feature
contributions are quantified by regression coefficients, mean decrease in impurity (MDI),
and SHAP values.

```text
figS21_S22_Experimental_Groups_Matrix.csv
   ├── logistic regression ──► per-feature coefficient contributions
   └── random forest ──► MDI importance
                     └──► SHAP values (per class, diet and category models)
```

## Inputs

**Processed-data archive:** https://doi.org/10.25378/janelia.32717316

| Script | Required file | Where the path is supplied |
|---|---|---|
| [`fig.S21_S22_Logistic_Regression.py`](fig.S21_S22_Logistic_Regression.py) | `figS21_S22_Experimental_Groups_Matrix.csv` | `pd.read_csv` at [line 24](fig.S21_S22_Logistic_Regression.py#L24) |
| [`fig.S21_S22_Random_Forest_with_MDI_and_SHAP.py`](fig.S21_S22_Random_Forest_with_MDI_and_SHAP.py) | `figS21_S22_Experimental_Groups_Matrix.csv` | `pd.read_csv` at [line 110](fig.S21_S22_Random_Forest_with_MDI_and_SHAP.py#L110) |

Both committed paths are the placeholder `"...add path..."`. Neither script has a
command-line interface. The random-forest script additionally requires `output_dir` to be
set at [line 39](fig.S21_S22_Random_Forest_with_MDI_and_SHAP.py#L39); it is created
automatically if absent.

### Row filtering

Both scripts drop rows where `cell_id_linked == 0`. These are cells for which `motile`
could not link organelles to a parent cell across z-slices. The surviving cell count is
printed at load time; record it, since it defines the analysis population.

### Labels

- `group` → diet / nutritional condition (CNT, STV, WD)
- `Prediction` → organelle-defined hepatocyte category (H1–H5)

Features are the same explicit organelle feature list used in Fig. 6.

## Outputs

### Logistic regression
Written to the working directory:
`nutrient_logistic_regression_feature_contribution.csv` and
`catagory_logistic_regression_feature_contribution.csv`.

### Random forest, MDI, and SHAP
All written to `output_dir`, figures at `dpi=300` with `bbox_inches="tight"`:

- Confusion matrices: `confusion_matrix_diet.png`, `confusion_matrix_hepatocyte_category.png`
- MDI importance: `MDI_feature_importance_diet.png`, `MDI_feature_importance_hepatocyte_category.png`
- SHAP, diet model: `shap_diet_all_groups_mean_abs_bar.png`, `shap_diet_legend_CNT_STV_WD.png`, and per-class `shap_diet_beeswarm_<token>.png`, `shap_diet_mean_abs_bar_<token>.png`, `shap_diet_class_<n>.csv`
- SHAP, category model: `shap_hepatocyte_category_all_categories_mean_abs_bar.png`, `shap_hepatocyte_category_legend.png`, and per-category `shap_hepatocyte_category_beeswarm_<token>.png`, `shap_hepatocyte_category_mean_abs_bar_<token>.png`

## Reproducibility notes

- Random forests: `n_estimators=300`, `random_state=42`, `n_jobs=-1` (lines [279](fig.S21_S22_Random_Forest_with_MDI_and_SHAP.py#L279) and [285](fig.S21_S22_Random_Forest_with_MDI_and_SHAP.py#L285)). Both models are seeded, so results are deterministic given identical input and package versions.
- MDI importance is biased toward high-cardinality and continuous features. SHAP values are provided alongside it for this reason; prefer SHAP for ranking claims.
- `shap 0.46.0` was used. SHAP output format has changed across releases; pin the version when rerunning.
- Note the spelling `catagory` in output filenames and variables — retained for consistency with the deposited files.
- These scripts have duplicated header docstrings from successive editing passes; only the last is meaningful.

[Back to the analysis index](../README.md)
