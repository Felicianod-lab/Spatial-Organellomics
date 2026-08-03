# Figure reproduction guide

Maps manuscript figures to analysis folders, deposited input files, and the outputs each
script produces.

All input matrices are deposited at **https://doi.org/10.25378/janelia.32717316** unless
marked *generated locally*. Every subfolder README lists the exact constant and line
number where each path is set.

## Main figures

| Figure | Folder | Script | Input file | Key outputs |
|---|---|---|---|---|
| Fig. 1 | [`fig1_liver_pancreas_pca`](../analysis/fig1_liver_pancreas_pca/) | `fig1_pca.py` | `fig1_liver_pancreas_cell_mask_features.csv` | Cross-organ PCA scatter (figure only; no CSV written) |
| Fig. 2 | [`fig2_hepatocyte_clustering`](../analysis/fig2_hepatocyte_clustering/) | `Fig2_Optimal_Cluster_Number_and_PCs_selection.py` | `Fig2_Hepatocyte _Clustering_Group_Control_data .csv` | Cluster/PC selection metrics and diagnostic plots |
| Fig. 2 | 〃 | `Fig2_PCA_GMM_plots.py` | 〃 | `Clean_Data.csv`, `FULL_CONCAT_clusters.csv`, cluster remapping tables, PCA/GMM panels |
| Fig. 2 | 〃 | `Fig2_Organelle_Feature_Heatmaps.py` | `FULL_CONCAT_clusters.csv` *(generated locally)* | Per-category organelle feature heatmaps |
| Fig. 2 | 〃 | `Fig2_Organelle_Subtype_Proportion_Bars.py` | 〃 | Organelle subtype proportion bars |
| Fig. 3 | [`fig3_Spatial_distribution_analyses_and_cross_modal_alignment`](../analysis/fig3_Spatial_distribution_analyses_and_cross_modal_alignment/) | `Fig3_Hepatocyte_Category_Spatial_Distribution_Plots.py` | `Fig3_Spatial_Analysis_Matrix.csv` | H1–H5 frequency and entropy versus acinar position |
| Fig. 3 | 〃 | `Fig3_Hepatocyte_Category_Histograms.py` | `FULL_CONCAT_clusters.csv` *(generated locally)* | Category position histograms |
| Fig. 3 | 〃 | `Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py` | `Fig3_Adapted_Spatial_Proteomic_Data.csv` + `FULL_CONCAT_clusters.csv` | Heterogeneity index curves (Simpson's dominance, Shannon entropy) for both modalities |
| Fig. 3 | 〃 | `Fig3_Cross_Modal_Inference_Analysis_and_Plots.py` | `Fig3_Cross_Modal_Inference_Organellomics_Matrix.csv` + `Fig3_Cross_Modal_Inference_Proteomics_Matrix.csv` | 12 alignment figures, consensus and Hungarian matching tables, pathway enrichment, nitrogen indices, `Pipeline_parameters.csv` |
| Fig. 4 | [`fig4_nutritional_perturbation`](../analysis/fig4_nutritional_perturbation/) | `Fig4_Optimal_Cluster_PCs_Selection_and_PCA_for_All_Experimental_Groups.py` | `Full_data_perturbations.csv` | Per-condition cluster/PC selection, `overall_experiment_cluster_selection_summary.csv` |
| Fig. 4 | 〃 | `Fig4_PCA_GMM_Plots_by_Experimental_Group.py` | 〃 | Per-condition PCA/GMM panels, `FULL_CONCAT_clusters.csv`, `combineHD_pittsburgh_indices.csv` |
| Fig. 4 | 〃 | `Fig4_Pittsburgh_H_D_Plots.py` | `combineHD_pittsburgh_indices.csv` *(generated locally)* | Heterogeneity (H) and dominance (D) curves, percent-change panels |
| Fig. 5 | [`fig5_trajectory_tmrm`](../analysis/fig5_trajectory_tmrm/) | `Fig5_Trajectory_Inference_from_Control_to_Fatsed.py` | `Fig5_Perturbation_Data_Matrix.csv` | Best-pair ranking, hub selection, `pseudotime.csv`, `cells_with_pseudotime_H4_FH_only.csv`, root-sensitivity tables |
| Fig. 5 | 〃 | `Fig5_Trajectories_and PC1_Plots.py` | `cells_with_pseudotime_H4_FH_only.csv` *(generated locally)* + `Full_data_perturbations.csv` | `PC1_vs_Pseudotime.png`, `PC1_Loadings.png`, `PC1_Spatial_Modulation.png`, PCA loadings |
| Fig. 5 | 〃 | `Fig5_Control_to_Fasted_Mixed_Model_Mitochondria_Analysis_and_Plots.py` | `Fig5_IVM_Control_Group_Mitochondria_Properties.csv` + `Fig5_IVM_Fasted_Group_Mitochondria_Properties.csv` | Mixed model with liver and tile random effects (displayed; not written to disk) |
| Fig. 5 | 〃 | `Fig5_IVM_TMRM_Grouping_and_Heterogeneity.py` | 〃 | GMM on TMRM, state distribution, entropy (displayed; not written to disk) |
| Fig. 5 | 〃 | `Fig5_IVM_Control_to_Fasted_Organelle_TMRM_Coupling.py` | the two mitochondria tables + `Fig5_IVM_Control_Group_Lipid_Droplets_Properties.csv` + `Fig5_IVM_Fasted_Group_Lipid_Droplets_Properties.csv` | Global organelle score, five-fold CV tables, region-interaction and mixed-effects models, 6 figures |
| Fig. 6 | [`fig6_prediction_models`](../analysis/fig6_prediction_models/) | `Fig6_MLP.py` | `Fig6_Experimental_Groups_Organelle_Features_Matrix.csv` | Nutritional-condition and hepatocyte-category confusion matrices, printed accuracies (**no files written**) |
| Fig. 6 | 〃 | `Fig6_Early_MASLD_MLP.py` | `Fig6_Early_MASLD_Model_Data_Matrix.csv` | Western-diet duration confusion matrix, `all_included.csv` |

## Supplementary figures

| Figure | Folder | Script | Input file | Key outputs |
|---|---|---|---|---|
| fig. S8 (G) | [`fig2_hepatocyte_clustering`](../analysis/fig2_hepatocyte_clustering/) | `Fig2_Support_ANOVA_Levene_Welch_GamesHowell_Pairwise_for_S8_G.py` | `FULL_CONCAT_clusters.csv` *(generated locally)* | ANOVA, Levene, Welch, and Games–Howell pairwise statistics |
| fig. S12 | 〃 | `Fig2_Harmonized_Benchmarking_of_GMM.py` | `figure2_benchmark_Organellomics_input_matrix.csv`, `figure2_benchmark_Transcriptomics_input_matrix.csv`, `figure2_benchmark_Proteomics_input_matrix.csv` (one run per modality) | Per-modality GMM metrics in `GMM_Results/`, `modality_ranking.csv` |
| fig. S12 | 〃 | `Fig2_Harmonized_Benchmarking_of_Leiden.py` | the same three benchmark matrices | Per-modality Leiden metrics and composite bootstrap comparison |
| fig. S13 | [`fig3_...cross_modal_alignment`](../analysis/fig3_Spatial_distribution_analyses_and_cross_modal_alignment/) | `Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py` | `Fig3_Full_Acinar_Category_Spatial_Analysis_Matrix.csv` | Per-acinus adjacency, fraction explained by position, Moran's I residuals, bootstrap summaries, violin panel |
| figs. S21, S22 | [`figS21-S22 Logistic-regression, Random-forest, and SHAP`](../analysis/figS21-S22%20Logistic-regression,%20Random-forest,%20and%20SHAP/) | `fig.S21_S22_Logistic_Regression.py` | `figS21_S22_Experimental_Groups_Matrix.csv` | Per-feature logistic-regression contribution tables |
| figs. S21, S22 | 〃 | `fig.S21_S22_Random_Forest_with_MDI_and_SHAP.py` | 〃 | Confusion matrices, MDI importance, SHAP beeswarm and bar plots, per-class SHAP tables |

Panels not listed here are produced by the Liv-Zones workflow
(https://github.com/ahillsley/liv_zones) at the segmentation and feature-extraction
stage, or are schematics and representative micrographs rather than derived analyses.

## Required run order

Four files are intermediates produced by one script and consumed by others. Run the
producers first:

1. `Fig2_PCA_GMM_plots.py` → `Clean_Data.csv`, `FULL_CONCAT_clusters.csv`
   → needed by `Fig2_Organelle_Feature_Heatmaps.py`, `Fig2_Organelle_Subtype_Proportion_Bars.py`, `Fig2_Support_ANOVA_...py`, `Fig3_Hepatocyte_Category_Histograms.py`, `Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py`
2. `Fig4_PCA_GMM_Plots_by_Experimental_Group.py` → `combineHD_pittsburgh_indices.csv`
   → needed by `Fig4_Pittsburgh_H_D_Plots.py`
3. `Fig5_Trajectory_Inference_from_Control_to_Fatsed.py` → `cells_with_pseudotime_H4_FH_only.csv`
   → needed by `Fig5_Trajectories_and PC1_Plots.py`

All other scripts are independent and can be run in any order.

## Notes

- Clustering is performed separately per condition, so category indices are not directly comparable across conditions without the saved remapping tables. Always use `Cluster_remapping_by_ascini_position.csv` and the PC1-orientation check rather than raw GMM component numbers.
- Seeds: PCA uses `42` and GMM uses `54` in the clustering workflows; cross-modal inference uses `SEED = 0`; random forests use `random_state=42`. The two Fig. 6 MLP scripts set no seed and are not run-to-run deterministic.
- Two filenames and one folder name contain spaces or commas; quote them in the shell.
