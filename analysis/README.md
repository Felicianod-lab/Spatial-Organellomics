# Analysis code

Manuscript-level analysis scripts for **Multi-organelle signatures map cell state
diversity and metabolic adaptation in tissues**, organized by figure. Each subfolder has
its own README documenting required inputs, generated outputs, and random seeds.

| Folder | Figures | Contents |
|---|---|---|
| [`fig1_liver_pancreas_pca`](fig1_liver_pancreas_pca/) | Fig. 1 | Cross-organ PCA of liver and pancreas cells |
| [`fig2_hepatocyte_clustering`](fig2_hepatocyte_clustering/) | Fig. 2, figs. S8, S12 | Control-liver PCA and GMM, cluster/PC selection, organelle feature heatmaps, subtype proportions, ANOVA/Levene/Welch/Games–Howell tests, harmonized GMM and Leiden benchmarking against transcriptomics and proteomics |
| [`fig3_Spatial_distribution_analyses_and_cross_modal_alignment`](fig3_Spatial_distribution_analyses_and_cross_modal_alignment/) | Fig. 3, fig. S13 | Category distribution along the acinar axis, heterogeneity indices, per-acinus niche metrics, cross-modal alignment with spatial proteomics |
| [`fig4_nutritional_perturbation`](fig4_nutritional_perturbation/) | Fig. 4 | Per-condition PCA and GMM (control, fasted, Western diet), heterogeneity and dominance index curves |
| [`fig5_trajectory_tmrm`](fig5_trajectory_tmrm/) | Fig. 5 | Control-to-fasted trajectory inference and pseudotime, PC1 trajectory figures, intravital TMRM mixed models and organelle-score coupling |
| [`fig6_prediction_models`](fig6_prediction_models/) | Fig. 6 | MLP prediction of nutritional condition, hepatocyte category, and Western-diet duration |
| [`figS21-S22 Logistic-regression, Random-forest, and SHAP`](figS21-S22%20Logistic-regression,%20Random-forest,%20and%20SHAP/) | figs. S21, S22 | Logistic regression, random forest with MDI, and SHAP feature importance |

## Data

Input matrices are deposited on Figshare rather than stored in this repository:

- **Processed source data and analysis inputs:** https://doi.org/10.25378/janelia.32717316
- **Representative raw confocal images:** https://doi.org/10.25378/janelia.31863250

Most scripts carry a hard-coded input path committed as the placeholder
`"...add path..."`. Each subfolder README lists the constant and line number to edit, or
the command-line flag to use instead where one exists.

## Dependency order

Several scripts consume files produced by others. Run in this order:

1. `fig2_hepatocyte_clustering/Fig2_PCA_GMM_plots.py` → produces `FULL_CONCAT_clusters.csv`, required by parts of Fig. 2 and Fig. 3
2. `fig4_nutritional_perturbation/Fig4_PCA_GMM_Plots_by_Experimental_Group.py` → produces `combineHD_pittsburgh_indices.csv`, required by `Fig4_Pittsburgh_H_D_Plots.py`
3. `fig5_trajectory_tmrm/Fig5_Trajectory_Inference_from_Control_to_Fatsed.py` → produces `cells_with_pseudotime_H4_FH_only.csv`, required by `Fig5_Trajectories_and PC1_Plots.py`

Everything else can be run independently.

## Environment

See [`../requirements.txt`](../requirements.txt) for exact pinned versions and
[`../README.md`](../README.md) for setup instructions. Developed on Python 3.9.21.

## Conventions

- **Category labels.** GMM component indices are remapped for display so that categories order sensibly along the acinar axis. Always use the saved remapping and PC1-orientation tables rather than raw component numbers.
- **Paths with spaces.** Two filenames and one folder name contain spaces or commas. Quote them in the shell.
