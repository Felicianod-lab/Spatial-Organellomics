# Fig.4 Nutritional perturbation

## Purpose

PCA and Gaussian mixture model clustering applied independently to each nutritional
condition — control (CNT), fasted / starved (STV), and Western diet (WD) — followed by
heterogeneity and dominance index curves along the acinar axis.

```text
Full_data_perturbations.csv
  ├── per-experiment optimal PC and cluster-number selection
  ├── PCA + GMM per experimental group ──► FULL_CONCAT_clusters.csv
  │                                        Cluster_remapping_by_ascini_position.csv
  │                                        combineHD_pittsburgh_indices.csv
  └────────────────────────────────────►  H and D index curves, percent change
```

## Inputs

**Processed-data archive:** https://doi.org/10.25378/janelia.32717316

| Script | Required file | Where the path is supplied |
|---|---|---|
| [`Fig4_Optimal_Cluster_PCs_Selection_and_PCA_for_All_Experimental_Groups.py`](Fig4_Optimal_Cluster_PCs_Selection_and_PCA_for_All_Experimental_Groups.py) | `Full_data_perturbations.csv` | `DEFAULT_DATA_PATH` at [line 50](Fig4_Optimal_Cluster_PCs_Selection_and_PCA_for_All_Experimental_Groups.py#L50) — resolves relative to the script directory, so placing the CSV beside the script works without editing. Or pass `--data`. |
| [`Fig4_PCA_GMM_Plots_by_Experimental_Group.py`](Fig4_PCA_GMM_Plots_by_Experimental_Group.py) | `Full_data_perturbations.csv` | `INPUT_CSV` at [line 31](Fig4_PCA_GMM_Plots_by_Experimental_Group.py#L31) |
| [`Fig4_Pittsburgh_H_D_Plots.py`](Fig4_Pittsburgh_H_D_Plots.py) | `combineHD_pittsburgh_indices.csv` | `PITTSBURGH_CSV` at [line 29](Fig4_Pittsburgh_H_D_Plots.py#L29) |

`combineHD_pittsburgh_indices.csv` is **generated locally** by
`Fig4_PCA_GMM_Plots_by_Experimental_Group.py`; run that script first.

### Expected columns

`Fig4_Pittsburgh_H_D_Plots.py` requires the wide-format columns
`ascini_position_binned`, `C_H`, `F_H`, `W_H`, `C_D`, `F_D`, `W_D` — heterogeneity (H)
and dominance (D) for control, fasted, and Western diet respectively.

## Running

Only the selection script has a command-line interface:

```bash
python Fig4_Optimal_Cluster_PCs_Selection_and_PCA_for_All_Experimental_Groups.py \
  --data /path/to/Full_data_perturbations.csv \
  --output-dir cluster_selection_outputs \
  --label-column labels --group-column group \
  --min-pcs 2 --max-pcs 40 --max-clusters 10 \
  --mito-count-threshold 20 \
  --stability-runs 25 --stability-subsample-fraction 0.8 \
  --ari-std-penalty-weight 1.0 --cv-splits 5 \
  --random-state 42 --n-init 10 --jobs -1
```

Group membership can come from an existing group column or be derived from the labels
column. The other two scripts are run directly after editing their path constants.

## Outputs

### Cluster and PC selection
Per-experiment summaries, metric tables, diagnostic plots, and final cluster labels
under `cluster_selection_outputs/`, plus
`overall_experiment_cluster_selection_summary.csv`.

PC-set selection penalizes unstable results using `ARI_mean − penalty_weight × ARI_std`;
cluster-number selection uses the cluster metrics alone.

### Main PCA/GMM workflow
Written under `Full_Concat_data_by_experiment/` (derived from the input CSV's parent
directory, [line 58](Fig4_PCA_GMM_Plots_by_Experimental_Group.py#L58)):

- `Data_including_bad_cells.csv`, `Clean_Data.csv`
- `PCA.csv`, `PCA_feature_columns.csv`, `PCA_explained_variance_ratio.csv`
- `FULL_CONCAT.csv`, `FULL_CONCAT_clusters.csv`, `FULL_CONCAT_clusters_PC1_oriented.csv`
- `Cluster_remapping_by_ascini_position.csv`, `Cluster_remapping_by_ascini_position_PC1_oriented.csv`
- `Prediction_PC1_left_to_right_orientation_check.csv`
- `combineHD_pittsburgh_indices.csv`, `pittsburgh_indices_long.csv`
- PNG and SVG pairs: `Figure4_PCA_GMM_ellipses_same_style`, `Figure4_PCA_GMM_probability_same_style`, `Figure4_PCA_GMM_prediction_same_style`, `Figure4_PCA_ascini_position_rocket_same_style`

### Heterogeneity index curves
`combineHD_pittsburgh_indices_with_percentage_changes.csv` and plots under
`Pittsburgh_H_D_plots/`. Raw-index panels use `figsize=(12, 4.5)` with an inverted
x-axis; percent-change panels use `figsize=(12, 3)` with dash-dot lines and the control
trace at zero.

## Reproducibility notes

- PCA seed `RANDOM_STATE = 42`; GMM seed `RANDOM_STATE = 54` (lines [62](Fig4_PCA_GMM_Plots_by_Experimental_Group.py#L62) and [63](Fig4_PCA_GMM_Plots_by_Experimental_Group.py#L63)). These match the Fig. 2 workflow.
- Selection-script seed defaults to `42` ([line 85](Fig4_Optimal_Cluster_PCs_Selection_and_PCA_for_All_Experimental_Groups.py#L85)).
- PC and cluster choices from the selection script are transferred **manually** into the plotting script. Record the selected values with the outputs.
- Category labels and PC1 orientation are remapped for display. Use the saved remapping and orientation-check tables rather than raw GMM component numbers.
- Clustering is performed independently per condition, so category indices are not automatically comparable across conditions without the remapping tables.
- "Pittsburgh" in filenames and variables is an internal name for the Simpson's-dominance and Shannon-entropy pair; see `Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py` for the definition.

[Back to the analysis index](../README.md)
