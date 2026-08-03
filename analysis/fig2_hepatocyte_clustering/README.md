# Fig2 Hepatocyte clustering

## Purpose

This module contains the control-liver PCA/GMM workflow, visual and statistical summaries of GMM-defined hepatocyte categories, and harmonized GMM/Leiden comparisons across transcriptomics, proteomics, and organellomics.

```text
Control organellomics matrix
  ├── optimal PC/cluster selection
  └── PCA + GMM ──► FULL_CONCAT_clusters.csv
                         ├── feature heatmaps
                         ├── subtype proportions
                         └── ANOVA / Welch / pairwise tests

RNA + protein + organelle benchmark matrices
  ├── GMM benchmark
  └── Leiden benchmark
```

## Inputs

**Processed-data archive:** `https://doi.org/10.25378/janelia.32717316`

| Script | Required file or files | Provenance / DOI | Where the path is supplied |
|---|---|---|---|
| [`Fig2_Optimal_Cluster_Number_and_PCs_selection.py`](Fig2_Optimal_Cluster_Number_and_PCs_selection.py) | `Fig2_Hepatocyte _Clustering_Group_Control_data .csv` | External processed control-hepatocyte matrix; `https://doi.org/10.25378/janelia.32717316` | Replace `DATA_PATH` at [line 20](Fig2_Optimal_Cluster_Number_and_PCs_selection.py#L20). The spaces in the historical filename are literal. |
| [`Fig2_PCA_GMM_plots.py`](Fig2_PCA_GMM_plots.py) | `Fig2_Hepatocyte _Clustering_Group_Control_data .csv` | Same control matrix; `https://doi.org/10.25378/janelia.32717316` | Replace `INPUT_CSV` at [line 48](Fig2_PCA_GMM_plots.py#L48). |
| [`Fig2_Organelle_Feature_Heatmaps.py`](Fig2_Organelle_Feature_Heatmaps.py) | `FULL_CONCAT_clusters.csv` | Generated locally by `Fig2_PCA_GMM_plots.py`; generated locally, not deposited separately | Replace `INPUT_CSV` at [lines 35–37](Fig2_Organelle_Feature_Heatmaps.py#L35-L37). |
| [`Fig2_Organelle_Subtype_Proportion_Bars.py`](Fig2_Organelle_Subtype_Proportion_Bars.py) | `FULL_CONCAT_clusters.csv` | Generated locally by `Fig2_PCA_GMM_plots.py`; generated locally, not deposited separately | Replace `INPUT_CSV` at [lines 26–28](Fig2_Organelle_Subtype_Proportion_Bars.py#L26-L28). |
| [`Fig2_Support_ANOVA_Levene_Welch_GamesHowell_Pairwise_for_S8_G.py`](Fig2_Support_ANOVA_Levene_Welch_GamesHowell_Pairwise_for_S8_G.py) | `FULL_CONCAT_clusters.csv` | Generated locally by `Fig2_PCA_GMM_plots.py`; generated locally, not deposited separately | Replace `INPUT_CSV` at [line 43](Fig2_Support_ANOVA_Levene_Welch_GamesHowell_Pairwise_for_S8_G.py#L43). |
| [`Fig2_Harmonized_Benchmarking_of_GMM.py`](Fig2_Harmonized_Benchmarking_of_GMM.py) | Organellomics: `figure2_benchmark_Organellomics_input_matrix.csv`<br>Transcriptomics: `figure2_benchmark_Transcriptomics_input_matrix.csv`<br>Proteomics: `figure2_benchmark_Proteomics_input_matrix.csv` | External processed benchmark matrices; `https://doi.org/10.25378/janelia.32717316` | **Preferred:** run once per matrix with `--input`, declared at [line 1520](Fig2_Harmonized_Benchmarking_of_GMM.py#L1520), so no source edit is needed. The no-argument example hard-codes paths at [line 1461](Fig2_Harmonized_Benchmarking_of_GMM.py#L1461), [line 1476](Fig2_Harmonized_Benchmarking_of_GMM.py#L1476), and [line 1488](Fig2_Harmonized_Benchmarking_of_GMM.py#L1488). |
| [`Fig2_Harmonized_Benchmarking_of_Leiden.py`](Fig2_Harmonized_Benchmarking_of_Leiden.py) | `figure2_benchmark_Transcriptomics_input_matrix.csv`<br>`figure2_benchmark_Proteomics_input_matrix.csv`<br>`figure2_benchmark_Organellomics_input_matrix.csv` | External processed benchmark matrices; `https://doi.org/10.25378/janelia.32717316` | Pass `--rna`, `--protein`, and `--organelle`; defaults are declared at [lines 1179–1181](Fig2_Harmonized_Benchmarking_of_Leiden.py#L1179-L1181). The committed organelle default ends in `.csv.csv`; pass the correct filename explicitly. |

Both benchmark scripts consume the **same three harmonized input matrices**; that
shared input is what makes the GMM and Leiden results directly comparable. Only the
way the paths are supplied differs between the two scripts.

Common control-matrix columns include `cell_id_linked`, `labels`, `ascini_position`, `mito_aspect_ratio`, `mito_density`, `area`, and downstream `Prediction`/`Predictions`. The benchmark schemas differ: the Leiden script expects cell ID in the first column and excluded metadata in the final column; the GMM benchmark excludes conventional `cell_id` and `position` fields where present.

Example benchmark commands:

```bash
python analysis/fig2_hepatocyte_clustering/Fig2_Harmonized_Benchmarking_of_GMM.py \
  --input /path/to/figure2_benchmark_Organellomics_input_matrix.csv \
  --modality organellomics \
  --use-scanpy-preproc \
  --feature-budget 98 \
  --seed 42

python analysis/fig2_hepatocyte_clustering/Fig2_Harmonized_Benchmarking_of_Leiden.py \
  --rna /path/to/figure2_benchmark_Transcriptomics_input_matrix.csv \
  --protein /path/to/figure2_benchmark_Proteomics_input_matrix.csv \
  --organelle /path/to/figure2_benchmark_Organellomics_input_matrix.csv \
  --outdir /path/to/results/fig2_leiden
```

The five configuration-driven figure scripts are run directly after editing their linked constants.

## Outputs

### Optimal PC and cluster selection

`Fig2_Optimal_Cluster_Number_and_PCs_selection.py` evaluates cumulative PC sets and 1–10 GMM clusters. It displays diagnostic plots and explicitly saves `CNT_PC_ARI_stability_barplot.png` in the current working directory. Selection tables are otherwise printed or retained in memory.

### Main PCA/GMM workflow

`Fig2_PCA_GMM_plots.py` writes `Data_including_bad_cells.csv`, `Clean_Data.csv`, `PCA.csv`, PCA feature/variance tables, GMM parameter arrays, `FULL_CONCAT.csv`, `FULL_CONCAT_clusters.csv`, category-remapping/orientation tables, and PNG/SVG PCA/GMM figures. `FULL_CONCAT_clusters.csv` is the primary downstream input.

### Feature and statistical summaries

- `Fig2_Organelle_Feature_Heatmaps.py`: `cluster_mean_features.csv`, `feature_plotting_report.csv`, and organelle heatmaps in PNG/SVG.
- `Fig2_Organelle_Subtype_Proportion_Bars.py`: `subtype_proportions_by_cluster.csv` and stacked subtype plots in PNG/SVG.
- `Fig2_Support_ANOVA_...py`: filtered matrices, ANOVA/Levene/Welch tables, Games–Howell and Tukey results, category summaries, and many PNG/SVG pairwise plots under `ANOVA/`.

### Benchmark outputs

- GMM benchmark: modality cluster CSVs, PCA CSVs, summary JSON files, `modality_ranking.csv`, and an optional composite-score plot under `GMM_Results/`.
- Leiden benchmark: per-modality sweep/summary tables, focused comparisons, bootstrap statistics, `general_metrics_table.png`, `composite_bootstrap_barplot.png`, and verdict files under the selected `--outdir`.

## Reproducibility notes

- Main PCA seed: `42`; main GMM seed: `54`.
- Model-selection and benchmark defaults use fixed seeds where exposed, but exact results depend on input ordering and package versions.
- PC/cluster choices from the selection script are manually transferred into downstream settings; preserve the selected values with the outputs.
- Category labels and displayed PC1 orientation may be remapped. Use the saved mapping/orientation tables rather than raw GMM component numbers.
- The ANOVA script can produce hundreds of files; validate with smaller top-feature limits first.
- The Leiden benchmark requires the Scanpy/AnnData/igraph/Leiden stack.
- Do not run the GMM benchmark with no command-line arguments until its three example paths have been replaced; no-argument mode attempts all three modalities automatically.

[Back to the analysis index](../README.md)
