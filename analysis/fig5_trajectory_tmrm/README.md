# Fig.5 Trajectory inference and TMRM measurements

## Purpose

Probabilistic trajectory inference from control to fasted hepatocyte states in fixed
tissue, and intravital (IVM) TMRM measurements linking organelle architecture to
mitochondrial membrane potential.

```text
Full_data_perturbations.csv ──► CNT→STV first-hit / best-pair analysis
                                hub selection, pseudotime
                                    └──► PC1 vs pseudotime, loadings,
                                         spatial modulation

IVM mitochondria + lipid-droplet property tables (control, fasted)
   ├── mixed model with liver and tile random effects
   ├── GMM on TMRM, state distribution, heterogeneity
   └── global organelle score vs TMRM, PV vs CV coupling
```

## Inputs

**Processed-data archive:** https://doi.org/10.25378/janelia.32717316

| Script | Required file or files | Where the path is supplied |
|---|---|---|
| [`Fig5_Trajectory_Inference_from_Control_to_Fasted.py`](Fig5_Trajectory_Inference_from_Control_to_Fasted.py) | `Full_data_perturbations.csv` | Pass `--input` ([line 2365](Fig5_Trajectory_Inference_from_Control_to_Fasted.py#L2365)) |
| [`Fig5_Trajectories_and PC1_Plots.py`](Fig5_Trajectories_and%20PC1_Plots.py) | `cells_with_pseudotime_H4_FH_only.csv`<br>`Full_data_perturbations.csv` | `PSEUDOTIME_FILE` at [line 34](Fig5_Trajectories_and%20PC1_Plots.py#L34), `FULL_DATA_FILE` at [line 38](Fig5_Trajectories_and%20PC1_Plots.py#L38) |
| [`Fig5_Control_to_Fasted_Mixed_Model_Mitochondria_Analysis_and_Plots.py`](Fig5_Control_to_Fasted_Mixed_Model_Mitochondria_Analysis_and_Plots.py) | `Fig5_IVM_Control_Group_Mitochondria_Properties.csv`<br>`Fig5_IVM_Fasted_Group_Mitochondria_Properties.csv` | `control_path` and `fasted_path` at [lines 17–18](Fig5_Control_to_Fasted_Mixed_Model_Mitochondria_Analysis_and_Plots.py#L17-L18) |
| [`Fig5_IVM_TMRM_Grouping_and_Heterogeneity.py`](Fig5_IVM_TMRM_Grouping_and_Heterogeneity.py) | same two IVM mitochondria tables | inline `pd.read_csv` calls at [lines 18](Fig5_IVM_TMRM_Grouping_and_Heterogeneity.py#L18) and [23](Fig5_IVM_TMRM_Grouping_and_Heterogeneity.py#L23) |
| [`Fig5_IVM_Control_to_Fasted_Organelle_TMRM_Coupling.py`](Fig5_IVM_Control_to_Fasted_Organelle_TMRM_Coupling.py) | `Fig5_IVM_Control_Group_Mitochondria_Properties.csv`<br>`Fig5_IVM_Control_Group_Lipid_Droplets_Properties.csv`<br>`Fig5_IVM_Fasted_Group_Mitochondria_Properties.csv`<br>`Fig5_IVM_Fasted_Group_Lipid_Droplets_Properties.csv` | `CONTROL_MITO_CSV`, `CONTROL_LD_CSV`, `FASTED_MITO_CSV`, `FASTED_LD_CSV` at [lines 43–55](Fig5_IVM_Control_to_Fasted_Organelle_TMRM_Coupling.py#L43-L55) |

`cells_with_pseudotime_H4_FH_only.csv` is **generated locally** by the trajectory
inference script; run it before the PC1 plots.

### IVM label parsing

The mixed-model script parses acquisition identifiers with the pattern
`^(?P<liver_id>[^T]+)(?P<tile_section>TS\d+)(?P<region_id>R\d+)` — for example
`CL1TS1R2Z0_1` (control liver 1) or `F1TS1R1Z0_1` (fasted liver 1). Liver and tile enter
the model as random effects, so this parsing must succeed for the model to be specified
correctly.

## Running

Only the trajectory inference script has a command-line interface:

```bash
python Fig5_Trajectory_Inference_from_Control_to_Fasted.py \
  --input /path/to/Full_data_perturbations.csv \
  --output-dir fig5_trajectory_outputs \
  --label-col labels --cat-col Prediction \
  --n-components 20 --graph-pcs 20 --knn 30 \
  --graph-metric euclidean \
  --cnt-cats 0,1,2,3,4 --stv-cats 0,1,2,3,4 --topk 10
```

Also available: `--forward-tol`, `--affinity-k`, `--affinity-pool`, and plot dimension
flags. The other four scripts are run directly after editing their path constants.

## Outputs

### Trajectory inference
`best_cnt_to_stv_pairs_ranked.csv`, `best_cnt_to_stv_lastcnt_breakdown.csv`,
`best_pairs_topk_plot_data.csv`, `best_pairs_topk_bars.png`, `hub_choice.csv`,
`pseudotime.csv`, `pseudotime_cnt_stv_only.csv`, `cells_with_pca_and_pseudotime.csv`,
`feature_columns_used.csv`, absorption and first-hit tables (`absorption_to_finals.csv`,
`first_stv_hit_probs.csv`, `two_stage_unconditional_paths.csv`), route analysis
(`route_metrics.csv`, `route_bars.png`, `route_compare_rwprob_affinity.png`,
`route_summary.txt`), root-sensitivity tables (`root_sensitivity_summary.csv`,
`root_sensitivity_all_pairs.csv`, `root_sensitivity_consensus.csv`), and PCA projection
figures.

### PC1 trajectory figures
`PC1_vs_Pseudotime.png`, `PC1_Loadings.png`, `PC1_Spatial_Modulation.png`, plus
`feature_<pseudotime_col>_correlations.csv`,
`cells_with_pseudotime_H4_FH_only_with_PCs.csv`, and `PCA_loadings.csv`.

### Organelle score and TMRM coupling
Written under `Global_organelle_score_outputs/`:
`analysis_cells_with_global_organelle_score.csv`,
`global_feature_correlations_and_weights.csv`, five-fold cross-validation tables
(`cross_validated_out_of_fold_predictions.csv`, `cross_validated_fold_regressions.csv`,
`cross_validated_fold_metrics.csv`, `cross_validated_summary.csv`), region-interaction
model (`region_interaction_coefficients.csv`, `region_interaction_model_summary.txt`),
mixed-effects model (`mixed_effects_fixed_coefficients.csv`,
`mixed_effects_liver_random_intercepts.csv`, `mixed_effects_model_summary.txt`), and six
numbered figures (`01_original_combined_pv_vs_cv.png` through
`06_mixed_effects_fasted_pv_vs_cv.png`).

### Mixed model and TMRM grouping
The mixed-model and TMRM-grouping scripts display figures and print model summaries;
they do not write files to disk. Capture the console output if you need a record.

## Reproducibility notes

- Coupling script: `RANDOM_STATE = 42`, five-fold CV with shuffling (`CV_N_SPLITS = 5`, `CV_SHUFFLE = True`, `CV_RANDOM_STATE = 42`, [lines 57–66](Fig5_IVM_Control_to_Fasted_Organelle_TMRM_Coupling.py#L57-L66)). Figures saved at `dpi=300`.
- TMRM grouping GMM: `random_state=42` ([line 63](Fig5_IVM_TMRM_Grouping_and_Heterogeneity.py#L63)).
- PC1 plots: features are retained at `|Spearman rho| >= FEATURE_RHO_THRESHOLD`, set to `0.3`; the pseudotime column is `pseudotime_full` ([lines 47–48](Fig5_Trajectories_and%20PC1_Plots.py#L47-L48)).
- The best-pair plot uses an initial pseudotime root (default CNT category 3, i.e. H4). The final saved pseudotime is then **recomputed** using the selected hub as root. The two are intentionally different; do not assume the plot and the saved CSV share a root.
- Two scripts have no module docstring (`Fig5_Control_to_Fasted_Mixed_Model_...`, `Fig5_IVM_TMRM_Grouping_...`); their behavior is documented here only.
- The filename `Fig5_Trajectory_Inference_from_Control_to_Fasted.py` contains a typo for "Fasted", and `Fig5_Trajectories_and PC1_Plots.py` contains a space. Quote these paths in the shell.

[Back to the analysis index](../README.md)
