# Spatial Organellomics hub

<img width="3000" height="2100" alt="Spatial Organellomics: multi-organelle segmentation and cell-state mapping across liver and pancreas" src="https://github.com/user-attachments/assets/74356908-97e8-463d-a53f-889467c96cb0" />

This repository provides the resource map and manuscript-level analysis code for:

**Multi-organelle signatures map cell state diversity and metabolic adaptation in tissues**

This hub links the reusable segmentation and feature-extraction workflow, manuscript analysis scripts, processed source data, trained models, and external datasets used in the study.

## Installation

The analysis was developed and run on **Python 3.9.21** (Windows). Exact package
versions are pinned in [`requirements.txt`](requirements.txt); these are the
versions used to produce the published figures.

Using conda:

```bash
conda env create -f environment.yml
conda activate sOrganellomics
```

Using pip, into an existing Python 3.9 environment:

```bash
python -m pip install -r requirements.txt
```

Note that `leidenalg`, `igraph`, and `umap-learn` are required by the clustering
and benchmarking scripts but are not installed automatically by `scanpy`. They
are included in both environment files.

## Main resources

### Reusable workflow

The Liv-Zones pipeline for cell and organelle segmentation, feature extraction, example data, trained models, and tutorials is available at:

- GitHub: https://github.com/ahillsley/liv_zones
- Release: liv_zones v1.0.0
- Archived DOI: https://doi.org/10.5281/zenodo.21808753
  
The pipeline is intended to be applied to new datasets. Note that transferring it to a second tissue in this study required retraining the segmentation models and adding tissue-specific postprocessing for clustered pancreatic peroxisomes; adapting it to other tissues should be expected to need comparable hands-on work.

**Cell-linking**

Within Liv-Zones `track_z_pos.py` links cells across z-slices using `motile` (v0.2.1) and `motile_toolbox` (v0.2.3).


### Processed data and source files

Processed organelle feature tables, source CSV files, benchmark inputs, SHAP outputs, cross-modal alignment files, and intravital TMRM analysis tables are available at:

- Figshare: https://doi.org/10.25378/janelia.32717316

Representative raw confocal images from liver and pancreas are available at:

- Figshare: https://doi.org/10.25378/janelia.31863250

Due to file size, the complete full-resolution raw imaging dataset is available from the corresponding author upon reasonable request.

### Reused external datasets

- Spatial proteomics: PRIDE PXD038699
- Spatial transcriptomics: GEO GSE218472

## Repository structure

```
analysis/
├── fig1_liver_pancreas_pca/                                    cross-organ PCA
├── fig2_hepatocyte_clustering/                                 GMM clustering, cluster/PC selection,
│                                                               organelle feature heatmaps, subtype
│                                                               proportions, harmonized benchmarking
│                                                               (GMM and Leiden), ANOVA/Levene/Welch/
│                                                               Games-Howell testing
├── fig3_Spatial_distribution_analyses_and_cross_modal_alignment/
│                                                               spatial distribution and niche metrics,
│                                                               heterogeneity indices, cross-modal
│                                                               inference against spatial proteomics
├── fig4_nutritional_perturbation/                              per-condition PCA and GMM, heterogeneity
│                                                               and dominance metrics
├── fig5_trajectory_tmrm/                                       trajectory inference (control to fasted),
│                                                               mixed-model mitochondrial analysis,
│                                                               intravital TMRM grouping and
│                                                               structure-function coupling
├── fig6_prediction_models/                                     MLP prediction of nutritional state and
│                                                               early MASLD progression
└── figS21-S22 Logistic-regression, Random-forest, and SHAP/    logistic regression, random forest with
                                                                MDI, and SHAP feature importance

docs/
├── data_dictionary.md                                          description of deposited data files
├── figure_reproduction_guide.md                                figure-to-folder mapping
└── repository_map.md                                           map of external resources
```

Each `analysis/` subfolder contains its own `README.md` documenting required
inputs, generated outputs, and random seeds. See
[`docs/figure_reproduction_guide.md`](docs/figure_reproduction_guide.md) for the
mapping from manuscript figures to folders.

## Citation

If you use this code, please cite both the software and the paper. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it via the
"Cite this repository" button in the sidebar.

Software:

> Feliciano, D., Adhikari, R., Hillsley, A. Spatial-Organellomics: manuscript
> analysis code (v1.0.0). 

Paper:

> Adhikari, R., Hillsley, A., Johnson, A. D., Gao, S. M., Espinosa-Medina, I.,
> Funke, J., Feliciano, D. Multi-organelle signatures map cell state diversity
> and metabolic adaptation in tissues. (2026).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).


