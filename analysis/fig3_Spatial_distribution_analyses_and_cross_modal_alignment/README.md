# Fig.3 Spatial distribution analyses and cross modal alignment

## Purpose

Spatial distribution of organelle-defined hepatocyte categories along the portal
vein–central vein axis, heterogeneity indices, per-acinus niche metrics, and
cross-modal alignment between sOrganellomics and spatial proteomics.

```text
FULL_CONCAT_clusters.csv  ──►  category histograms vs acinar position
                          └──►  heterogeneity indices (Simpson / Shannon)

Fig3_Spatial_Analysis_Matrix.csv  ──►  H1–H5 frequency and entropy vs position

Fig3_Full_Acinar_Category_Spatial_Analysis_Matrix.csv
                          ──►  per-acinus adjacency, fraction explained by
                               position, Moran's I residuals, bootstrap CIs

Organellomics + proteomics matrices  ──►  Wasserstein / Hungarian / consensus
                                          alignment, pathway enrichment,
                                          nitrogen functional indices
```

## Inputs

**Processed-data archive:** https://doi.org/10.25378/janelia.32717316

| Script | Required file or files | Where the path is supplied |
|---|---|---|
| [`Fig3_Hepatocyte_Category_Histograms.py`](Fig3_Hepatocyte_Category_Histograms.py) | `FULL_CONCAT_clusters.csv` | `FILE_PATH` at [line 37](Fig3_Hepatocyte_Category_Histograms.py#L37) |
| [`Fig3_Hepatocyte_Category_Spatial_Distribution_Plots.py`](Fig3_Hepatocyte_Category_Spatial_Distribution_Plots.py) | `Fig3_Spatial_Analysis_Matrix.csv` | `DATA_PATH` at [line 13](Fig3_Hepatocyte_Category_Spatial_Distribution_Plots.py#L13) |
| [`Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py`](Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py) | `Fig3_Adapted_Spatial_Proteomic_Data.csv`<br>`FULL_CONCAT_clusters.csv` | `PROTEOMICS_FILE_PATH` at [line 26](Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py#L26) and `ORGANELLOMICS_FILE_PATH` at [line 29](Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py#L29) |
| [`Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py`](Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py) | `Fig3_Full_Acinar_Category_Spatial_Analysis_Matrix.csv` | `DEFAULT_CSV` at [line 38](Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py#L38), or pass `--csv` |
| [`Fig3_Cross_Modal_Inference_Analysis_and_Plots.py`](Fig3_Cross_Modal_Inference_Analysis_and_Plots.py) | `Fig3_Cross_Modal_Inference_Organellomics_Matrix.csv`<br>`Fig3_Cross_Modal_Inference_Proteomics_Matrix.csv` | `ORG_PATH` at [line 37](Fig3_Cross_Modal_Inference_Analysis_and_Plots.py#L37) and `PROT_PATH` at [line 38](Fig3_Cross_Modal_Inference_Analysis_and_Plots.py#L38) |

All committed paths are the placeholder `"...add path..."` and must be edited before
running, except `Fig3_figS13_...`, which accepts `--csv`.

`FULL_CONCAT_clusters.csv` is produced upstream by
[`../fig2_hepatocyte_clustering/Fig2_PCA_GMM_plots.py`](../fig2_hepatocyte_clustering/Fig2_PCA_GMM_plots.py).

### Expected columns

The histogram script expects a prediction column (default `Prediction`), a position
column (default `ascini_position`), and per-category confidence columns named `0`–`4`.
Categories are selected from high-confidence rows only; histograms are then plotted
using all rows belonging to the selected categories. Set
`USE_FILTERED_ROWS_FOR_HISTOGRAMS = True` to restrict histograms to high-confidence
rows as well.

## Running

Only `Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py` has a command-line
interface. It exposes roughly 50 flags; the analysis-relevant ones are:

```bash
python Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py \
  --csv /path/to/Fig3_Full_Acinar_Category_Spatial_Analysis_Matrix.csv \
  --n_bins 10 --k 6 --radius 50 \
  --adj_perm 1000 --morans_perm 1000 --frac_perm 1000 \
  --boot_B 2000 --min_cells 20
```

The remaining flags control plot appearance (palettes, point size and jitter, figure
dimensions, font sizes, spines, layout). The other four scripts are run directly after
editing their path constants.

## Outputs

### Category histograms
`histograms.png` — one combined panel plus one panel per prediction category, written
to `OUTPUT_DIR` (default: the current directory).

### Spatial distribution
`H1_H5_frequency_vs_position_global_sem.png`, `entropy_vs_position_global_sem.png`,
`global_H1_H5_frequency_vs_position_stacked.png`,
`H1_H5_frequency_vs_position_stacked.png`.

### Heterogeneity indices
`proteomics_pittsburgh_indices_plot.png` and
`organellomics_pittsburgh_indices_plot.png`. The function
`pittsburgh_heterogeneity_indices()` ([line 136](Fig3_Spatial_Proteomics_and_Organellomics_PHI_Plots.py#L136))
returns Simpson's dominance index and Shannon entropy per position bin.

### Per-acinus niche metrics
`per_acinus_metrics_with_norm_and_fracP.csv`, `per_acinus_metrics_enhanced.csv`,
`fraction_explained_per_acinus_perm.csv`,
`fraction_explained_group_bootstrap_summary.csv`,
`norm_fraction_explained_group_bootstrap_summary.csv`,
`violin_control_fraction_unexplained_pct_points.png`, plus diagnostic files
(`DEBUG_acinus_sizes_after_dropna.csv`, `DEBUG_na_counts_before_dropna.csv`,
`DEBUG_README.txt`).

### Cross-modal inference
Twelve numbered figures (`00_baseline_CONSENSUS_OH_to_PH.png` through
`11_net_nitrogen_functional_indices_bubble_panel.png`) and fifteen tables, including
`Wasserstein_distance_matrix.csv`, `Hard_Hungarian_matching.csv`,
`CONSENSUS_OH_to_PH.csv`, `Ordering_agreement_soft_stats.csv`,
`Tau_sensitivity_corr_with_baseline.csv`, `Hard_matching_bootstrap_frequency.csv`,
`Tau_bootstrap_distribution.csv`, `Distribution_reconstruction_error.csv`,
`Organelle_density_zscores_by_OH.csv`, `Net_nitrogen_functional_indices.csv`, and
`Pipeline_parameters.csv`. The last of these records the settings used for the run and
should be kept alongside the outputs.

## Reproducibility notes

- Cross-modal inference seed: `SEED = 0` ([line 44](Fig3_Cross_Modal_Inference_Analysis_and_Plots.py#L44)).
- Niche metrics use two internal seeds, `2025` and `123` (lines [512](Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py#L512) and [719](Fig3_figS13_Spatial_Organellomics_Niche_Metrics.py#L719)), plus `--point_seed` for plot jitter only.
- Permutation and bootstrap counts affect runtime substantially. Validate with small values before running at publication settings.
- `Fig3_Hepatocyte_Category_Spatial_Distribution_Plots.py` has no module docstring; its behavior is documented here only.
- Category numbering follows the remapping tables written by the Fig. 2 workflow, not raw GMM component indices.

[Back to the analysis index](../README.md)
