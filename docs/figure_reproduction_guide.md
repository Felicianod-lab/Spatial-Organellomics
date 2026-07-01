# Figure reproduction guide

This guide maps manuscript figures to analysis folders, input data, and expected outputs.

| Figure | Analysis folder | Input data | Main outputs |
|---|---|---|---|
| Fig. 1 | analysis/fig1_liver_pancreas_pca | processed organelle feature tables | PCA plots and source data |
| Fig. 2 | analysis/fig2_hepatocyte_clustering | control liver feature table | H1-H5 labels, feature summaries, benchmark source data |
| Fig. 3 | analysis/fig3_zonation_model_validation; analysis/fig3_cross_modal_alignment | control liver feature table; spatial proteomics data | zonation metrics, cross-modal alignment tables |
| Fig. 4 | analysis/fig4_nutritional_perturbation | control/fasting/WD feature tables | condition PCA, category proportions, heterogeneity metrics |
| Fig. 5 | analysis/fig5_trajectory_tmrm | control/fasting fixed-tissue features; intravital TMRM tables | trajectory outputs, TMRM statistics, structure-function coupling |
| Fig. 6 | analysis/fig6_prediction_models | all-condition feature tables; MASLD progression tables | prediction metrics and confusion matrices |
| fig. S12 | analysis/figS12_benchmarking | organellomics/proteomics/transcriptomics benchmark inputs | harmonized benchmark outputs |
| fig. S22 | analysis/figS22_shap | trained RF/ML input tables | SHAP values and feature-importance plots |
