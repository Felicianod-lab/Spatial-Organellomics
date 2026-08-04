# Data dictionary

Describes the processed data files used by the analysis scripts in this repository.

**Deposit:** https://doi.org/10.25378/janelia.32717316
**Representative raw images:** https://doi.org/10.25378/janelia.31863250


All files are UTF-8 comma-separated text (`.csv`) with a single header row.

---

## Shared column conventions

Most fixed-tissue matrices are one row per segmented cell and draw on a common set of
columns.

### Identifiers and metadata

| Column | Meaning |
|---|---|
| `labels` | Acquisition label encoding condition, animal, and region, e.g. `CNT1TS1R2Z0_1`, `P3TS1R1Z0_2`. Condition and animal are parsed from the prefix. |
| `cell_id_linked` | Parent-cell identifier assigned when organelles are linked across z-slices. **`0` means linking failed**; rows with `cell_id_linked == 0` are excluded by the figs. S21–S22 scripts. |
| `stack_id`, `stack_folder`, `region_folder` | Acquisition provenance |
| `centroid-0`, `centroid-1` | Cell centroid position in image coordinates |
| `area` | Whole-cell area. Excluded from PCA feature sets so that components only reflect organelle architecture. |
| `group` | Nutritional condition: `CNT` (control), `STV` (fasted/starved), `WD` (Western diet) |

### Spatial position

| Column | Meaning |
|---|---|
| `ascini_position` | Normalized position along the portal vein → central vein axis of the hepatic acinus. (Spelled "ascini" throughout the code and deposited files.) Excluded from PCA feature sets so that components only reflect organelle architecture. |
| `ascini_position_binned` | Discretized position bin used for heterogeneity curves |
| `acinus_id`, `liver_id` | Grouping keys for per-acinus and per-animal aggregation |

### Category assignments

| Column | Meaning |
|---|---|
| `Prediction` | Organelle-defined hepatocyte category (H1–H5), after remapping to acinar order |
| `Prediction_original`, `Prediction_new` | Raw GMM component index and remapped index; retained for traceability |
| `Probability` | Maximum posterior probability of the assigned category |
| `0`, `1`, `2`, `3`, `4` | Per-category posterior probabilities. Used to select high-confidence cells. |
| `CAT` | Category index used by the trajectory and niche-metric scripts |
| `component_1`, `component_2`, … | PCA scores |

> **Always use the remapped `Prediction` column together with the saved
> `Cluster_remapping_by_ascini_position.csv` table.** Raw GMM component numbers are
> arbitrary and are not comparable across conditions or runs.

### Organelle feature columns

109 quantitative features per cell, extracted by the Liv-Zones workflow. Naming is
systematic: `[type_N_]<organelle>_<measurement>`.

| Organelle | Prefix | Subtypes |
|---|---|---|
| Mitochondria | `mito_` | `type_1_` – `type_3_` |
| Peroxisomes | `peroxisome_` | `type_1_` – `type_3_` |
| Lipid droplets | `ld_` | `type_1_` – `type_4_` |

| Measurement suffix | Meaning |
|---|---|
| `density` | Organelle count per unit cell area |
| `avg_area` | Mean area of individual organelles |
| `perimeter` | Mean organelle perimeter |
| `percent_total_area` | Fraction of cell area occupied by the organelle class |
| `solidity` | Area divided by convex-hull area |
| `circularity` | Shape circularity |
| `aspect_ratio` | Major/minor axis ratio (mitochondria and peroxisomes only; **not defined for lipid droplets**) |
| `distance_from_edge` / `dist_from_edge` | Mean distance from the cell boundary. General features use `distance_from_edge`; subtype features use `dist_from_edge`. |
| `percent_type_N_<organelle>` | Fraction of that organelle class assigned to subtype N |

> **Units are set at feature-extraction time in Liv-Zones and are not converted anywhere
> in this repository.**

### Intravital (IVM) tables

Separate schema, one row per segmented organelle or cell in intravital acquisitions.

| Column | Meaning |
|---|---|
| `cell_label` | Acquisition identifier, e.g. `CL1TS1R2Z0_1` (control liver 1) or `F1TS1R1Z0_1` (fasted liver 1) |
| `liver_id`, `tile_section`, `region_id` | Parsed from `cell_label` with `^(?P<liver_id>[^T]+)(?P<tile_section>TS\d+)(?P<region_id>R\d+)`; enter the mixed models as random effects |
| `total_TMRM` | TMRM intensity, reporting mitochondrial membrane potential |
| `PV_CV` | Region assignment, periportal (`PV`) or pericentral (`CV`) |
| `Condition` | `Control` or `Fasted` |
| `mean_cell_area` | Per-cell area used in score normalization |

---

## Deposited files

| File | One row is | Used for |
|---|---|---|
| `fig1_liver_pancreas_cell_mask_features.csv` | one segmented cell (liver and pancreas) | Fig. 1 cross-organ PCA |
| `Fig2_Hepatocyte _Clustering_Group_Control_data .csv` | one control-liver hepatocyte | Fig. 2 clustering and cluster/PC selection. **Filename contains literal spaces** after `Hepatocyte` and before `.csv`. |
| `figure2_benchmark_Organellomics_input_matrix.csv` | one cell; first column `cell_id`, last column excluded metadata | fig. S12 harmonized benchmarking (organellomics arm) |
| `figure2_benchmark_Transcriptomics_input_matrix.csv` | one cell (extracted from public scRNA-seq, GEO GSE218472) | fig. S12 (transcriptomics arm) |
| `figure2_benchmark_Proteomics_input_matrix.csv` | one cell (extracted from public proteomics, PRIDE PXD038699) | fig. S12 (proteomics arm) |
| `Fig3_Spatial_Analysis_Matrix.csv` | one hepatocyte with category and acinar position | Fig. 3 spatial distribution and entropy |
| `Fig3_Full_Acinar_Category_Spatial_Analysis_Matrix.csv` | one hepatocyte, with `acinus_id` and `liver_id` for per-acinus aggregation | fig. S13 niche metrics |
| `Fig3_Adapted_Spatial_Proteomic_Data.csv` | one cell from the reanalyzed spatial proteomics dataset, harmonized to acinar position bins | Fig. 3 heterogeneity comparison |
| `Fig3_Cross_Modal_Inference_Organellomics_Matrix.csv` | organellomics category × feature summary matrix | Fig. 3 cross-modal alignment |
| `Fig3_Cross_Modal_Inference_Proteomics_Matrix.csv` | proteomics category × pathway matrix, including nitrogen-pathway groupings | Fig. 3 cross-modal alignment |
| `Full_data_perturbations.csv` | one hepatocyte across all three nutritional conditions | Fig. 4 per-condition clustering; Fig. 5 PC1 trajectory merge |
| `Fig5_Perturbation_Data_Matrix.csv` | one hepatocyte, input to trajectory inference | Fig. 5 pseudotime and best-pair analysis |
| `Fig5_IVM_Control_Group_Mitochondria_Properties.csv` | one intravital mitochondrial measurement, control | Fig. 5 TMRM models |
| `Fig5_IVM_Fasted_Group_Mitochondria_Properties.csv` | one intravital mitochondrial measurement, fasted | Fig. 5 TMRM models |
| `Fig5_IVM_Control_Group_Lipid_Droplets_Properties.csv` | one intravital lipid-droplet measurement, control | Fig. 5 global organelle score |
| `Fig5_IVM_Fasted_Group_Lipid_Droplets_Properties.csv` | one intravital lipid-droplet measurement, fasted | Fig. 5 global organelle score |
| `Fig6_Experimental_Groups_Organelle_Features_Matrix.csv` | one hepatocyte with `group` and `Prediction` labels | Fig. 6 MLP prediction |
| `Fig6_Early_MASLD_Model_Data_Matrix.csv` | one hepatocyte from the Western-diet time course | Fig. 6 MASLD duration model. Time point parsed from `labels` via `7days_`, `18days_`, `31days_`, `42days_`, `50days_`. **The `18days_` class is displayed as "17 days" in the published figure.** |
| `figS21_S22_Experimental_Groups_Matrix.csv` | one hepatocyte with `group` and `Prediction` | figs. S21–S22 logistic regression, random forest, SHAP |

## Locally generated intermediates

Not deposited. Produced by an earlier script and consumed by later ones — see the run
order in [`figure_reproduction_guide.md`](figure_reproduction_guide.md).

| File | Produced by |
|---|---|
| `Clean_Data.csv` | `Fig2_PCA_GMM_plots.py` |
| `FULL_CONCAT_clusters.csv` | `Fig2_PCA_GMM_plots.py`; a per-condition version by `Fig4_PCA_GMM_Plots_by_Experimental_Group.py` |
| `combineHD_pittsburgh_indices.csv` | `Fig4_PCA_GMM_Plots_by_Experimental_Group.py` |
| `cells_with_pseudotime_H4_FH_only.csv` | `Fig5_Trajectory_Inference_from_Control_to_Fasted.py` |

## Derived index columns

`combineHD_pittsburgh_indices.csv` uses a wide layout keyed on
`ascini_position_binned`, with one column per condition and index:

| Column | Condition | Index |
|---|---|---|
| `C_H`, `C_D` | control | Shannon entropy (H), Simpson's dominance (D) |
| `F_H`, `F_D` | fasted | 〃 |
| `W_H`, `W_D` | Western diet | 〃 |

In the code these two indices are computed together by a function named
`pittsburgh_heterogeneity_indices()`. "Pittsburgh" is an internal project name, not a
published index; the underlying quantities are Simpson's dominance index and Shannon
entropy as defined in the Methods.
