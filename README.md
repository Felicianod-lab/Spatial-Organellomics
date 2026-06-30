# Spatial Organellomics reproducibility hub

This repository provides the resource map and manuscript-level analysis code for:

**Multi-organelle signatures map cell state diversity and metabolic adaptation in tissues**

This hub links the reusable segmentation and feature-extraction workflow, manuscript analysis scripts, processed source data, trained models, and external datasets used in the study.

## Main resources

### Reusable workflow

The Liv-Zones pipeline for cell and organelle segmentation, feature extraction, example data, trained models, and tutorials is available at:

- GitHub: https://github.com/ahillsley/liv_zones
- Release: [insert release tag, e.g. v1.0.0]
- Archived DOI: [insert DOI if available]

### Cell-linking package

Motile was used for linking cells across z-slices:

- GitHub: https://github.com/funkelab/motile
- Version/commit used: [insert commit or version if available]

### Processed data and source files

Processed organelle feature tables, source CSV files, benchmark inputs, SHAP outputs, cross-modal alignment files, and intravital TMRM analysis tables are available at:

- Figshare: [insert DOI]

Representative raw confocal images from liver and pancreas are available at:

- Figshare: https://doi.org/10.25378/janelia.31863250

Due to file size, the complete full-resolution raw imaging dataset is available from the corresponding author upon reasonable request.

### Reused external datasets

- Spatial proteomics: PRIDE PXD038699
- Spatial transcriptomics: GEO GSE218472

## Contents of this repository

This repository contains manuscript-level code for:

- hepatocyte clustering
- organelle-feature analysis
- zonation-model validation
- spatial heterogeneity analysis
- cross-modal proteomic alignment
- proteome inference
- trajectory analysis
- machine-learning prediction
- SHAP feature-importance analysis
- spatial-omics benchmarking
- intravital TMRM analysis
- figure/source-data generation

## Recommended repository layout

```text
analysis/
  fig1_liver_pancreas_pca/
  fig2_hepatocyte_clustering/
  fig3_zonation_model_validation/
  fig3_cross_modal_alignment/
  fig4_nutritional_perturbation/
  fig5_trajectory_tmrm/
  fig6_prediction_models/
  figS12_benchmarking/
  figS22_shap/
docs/
  data_dictionary.md
  figure_reproduction_guide.md
  repository_map.md
data_manifest.tsv
code_manifest.tsv
environment.yml
release_notes.md
```

## Repository status

This repository is associated with the Science revision of the manuscript above. The final release corresponding to the submitted manuscript will be tagged as:

- Release: [insert release tag]
- Archived DOI: [insert archived DOI]

## Citation

Please cite the manuscript and the archived release DOI once available.
