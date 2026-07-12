"""
Figure 1 PCA script.

Purpose
-------
Reproduce the Figure 1 PCA scatter plot where label-prefix groups are colored as:
    P     -> purple -> Pancreas
    CNT   -> pink   -> Liver
    Other -> gray

This script keeps only the data-preparation, feature-selection, PCA, sampling, and
plotting steps needed for that plot. It removes the unrelated GMM, hierarchy,
z-score, 3D, pancreas, and hormone-cell-type exploratory plots.

Default settings match the active pink/purple PCA block in the original script:
    SAMPLE_FRACTION = 0.5
    RANDOM_STATE = 42
    point size = 3
    alpha = 0.9
    edgecolor = black
    linewidth = 0.02
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==================================================
# User settings
# ==================================================
DATA_PATH = Path("...add path.../fig1_liver_pancreas_cell_mask_features.csv"
    
)

# Keep this at 0.5 to reproduce the current active code.

SAMPLE_FRACTION = 0.5
RANDOM_STATE = 42
MITO_COUNT_THRESHOLD = 20



# Optional: set to a filename to save, for example Path("fig1_pca_purple_pink.png").
OUTPUT_PATH: Optional[Path] = None

# Plot settings preserved from the pink/purple Figure 1 block.
FIGSIZE = (8, 8)
LABEL_PALETTE = {"Pancreas": "purple", "Liver": "pink", "Other": "gray"}
POINT_SIZE = 3
POINT_ALPHA = 0.9
POINT_EDGE_COLOR = "black"
POINT_LINEWIDTH = 0.02
Random_sample = SAMPLE_FRACTION * 100
PLOT_TITLE = f"PCA ({Random_sample}% Random Sample) \u2013 Pancreas and Liver Cells"


# ==================================================
# Feature selection for Fig. 1 PCA
# ==================================================
# PCA was performed on a curated set of quantitative organelle features.
# Mitochondrial features were retained after mitochondrial quality-control
# filtering and are therefore intentionally not included in EXCLUDED_COLUMNS.
#
# The columns below were excluded for the following reasons:
#
# 1. Metadata, identifiers, and spatial coordinates were excluded because they
#    do not represent biological organelle morphology and could introduce
#    acquisition- or sample-specific structure into the PCA.
#
# 2. Whole-cell area was excluded so that the PCA would reflect organelle-level
#    morphology rather than overall cell size.
#
# 3. Hormone-derived measurements and hormone-based classifications were
#    excluded because these variables are used for downstream annotation or
#    interpretation, not to define the unsupervised PCA space.
#
# 4. Lipid-droplet morphology features were excluded because lipid droplets
#    were sparse or absent in most pancreatic cells, making these variables
#    dominated by absence/sparsity rather than broadly comparable morphology.
#
# 5. Large-peroxisome morphology features were excluded because these structures
#    were predominantly observed in endocrine cells. Excluding these features
#    prevents the PCA from being driven by a cell-type-restricted structure
#    rather than the broader organelle-morphology landscape.

EXCLUDED_COLUMNS = [
    # Metadata / IDs / spatial coordinates present in the raw CSV
    "centroid-0",
    "centroid-1",
    "labels",
    "Unnamed: 0",
    "label",
    "stack_folder",
    "region_folder",
    "area",
    
    # Metadata columns created only if parse_label() is used
    # These are not present in the raw CSV but are safe to keep here.
    "pancreas",
    "scan_area",
    "region",
    "stack",
    "cell_id",
    

    # Hormone-derived columns excluded from PCA
    "normalized_insulin",
    "normalized_glucagon",
   

    # LD morphology excluded

    "ld_avg_area",
    "ld_perimeter",
    "ld_distance_from_edge",
    "ld_aspect_ratio",
    "ld_circularity",
    "ld_solidity",
    "type_1_ld_density",
    "type_2_ld_density",
    "type_3_ld_density",
    "type_4_ld_density",
    "type_1_ld_perimeter",
    "type_2_ld_perimeter",
    "type_3_ld_perimeter",
    "type_4_ld_perimeter",
    "type_1_ld_avg_area",
    "type_2_ld_avg_area",
    "type_3_ld_avg_area",
    "type_4_ld_avg_area",
    "type_1_ld_avg_aspect_ratio",
    "type_2_ld_avg_aspect_ratio",
    "type_3_ld_avg_aspect_ratio",
    "type_4_ld_avg_aspect_ratio",
    "type_1_ld_avg_solidity",
    "type_2_ld_avg_solidity",
    "type_3_ld_avg_solidity",
    "type_4_ld_avg_solidity",
    "type_1_ld_avg_circularity",
    "type_2_ld_avg_circularity",
    "type_3_ld_avg_circularity",
    "type_4_ld_avg_circularity",
    "type_1_ld_percent_total_area",
    "type_2_ld_percent_total_area",
    "type_3_ld_percent_total_area",
    "type_4_ld_percent_total_area",
    "type_1_ld_dist_from_edge",
    "type_2_ld_dist_from_edge",
    "type_3_ld_dist_from_edge",
    "type_4_ld_dist_from_edge",

    # Large peroxisome morphology excluded
    "large_peroxisome_avg_area",
    "large_peroxisome_perimeter",
    "large_peroxisome_distance_from_edge",
    "large_peroxisome_aspect_ratio",
    "large_peroxisome_circularity",
    "large_peroxisome_solidity",

   
]


# ==================================================
# Data helpers
# ==================================================
def parse_label(label: object) -> pd.Series:
    """Parse labels like P1S1R1S0_001 into hierarchy columns."""
    match = re.match(r"P(\d+)S(\d+)R(\d+)S(\d+)_(\d+)", str(label))
    if match:
        return pd.Series(
            {
                "pancreas": int(match.group(1)),
                "scan_area": int(match.group(2)),
                "region": int(match.group(3)),
                "stack": int(match.group(4)),
                "cell_id": int(match.group(5)),
            }
        )

    return pd.Series(
        {
            "pancreas": np.nan,
            "scan_area": np.nan,
            "region": np.nan,
            "stack": np.nan,
            "cell_id": np.nan,
        }
    )


def label_group(label: object) -> str:
    """Map the label prefix to the plotting group used in Figure 1."""
    label_text = str(label)
    if label_text.startswith("P"):
        return "Pancreas"
    if label_text.startswith("CNT"):
        return "Liver"
    return "Other"


def require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
        )


def load_and_filter_data(
    data_path: Path,
    mito_count_threshold: int = MITO_COUNT_THRESHOLD,
) -> pd.DataFrame:
    
    
    """Load data and apply only the filters needed for the Figure 1 PCA."""
    data = pd.read_csv(data_path, na_values="-")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(1e-8, inplace=True)

    cell_masks_before_filtering = len(data)
    
    
    require_columns(data, ["labels", "mito_aspect_ratio", "mito_density", "area"])

    parsed_labels = data["labels"].apply(parse_label)
    data = pd.concat([data, parsed_labels], axis=1)

    

# Filtering logic preserved from the original script.
    data = data[data["mito_aspect_ratio"] > 0].copy()
    mito_count = data["mito_density"] * data["area"]
    data = data[mito_count >= mito_count_threshold].copy()
    
    
    # ==================================================
    # Estimated cell numbers
    # ==================================================
    # Each physical cell is expected to appear across ~5 z positions.
    # Therefore, estimated cell number = cell mask count / 5.
    
    Z_POSITIONS_PER_CELL = 5
    
    liver_cell_masks = data["labels"].astype(str).str.startswith("CNT").sum()
    pancreas_cell_masks = data["labels"].astype(str).str.startswith("P").sum()
    
    estimated_liver_cells = liver_cell_masks / Z_POSITIONS_PER_CELL
    estimated_pancreas_cells = pancreas_cell_masks / Z_POSITIONS_PER_CELL
    
    print(f"Estimated liver cells: {estimated_liver_cells:,.1f}")
    print(f"Estimated pancreas cells: {estimated_pancreas_cells:,.1f}")
       
        

    data["Label_Group"] = data["labels"].apply(label_group)
    return data


def build_feature_matrix(
    data: pd.DataFrame,
    excluded_columns: Iterable[str] = EXCLUDED_COLUMNS,
) -> pd.DataFrame:
    """Drop excluded columns and keep numeric columns for PCA."""
    columns_to_drop = [column for column in excluded_columns if column in data.columns]
    feature_data = data.drop(columns=columns_to_drop)
    numeric_features = feature_data.select_dtypes(include=[np.number])

    if numeric_features.empty:
        raise ValueError("No numeric PCA features remain after applying EXCLUDED_COLUMNS.")

    return numeric_features


# ==================================================
# PCA and plotting
# ==================================================
def compute_sampled_pca(
    data: pd.DataFrame,
    numeric_features: pd.DataFrame,
    sample_fraction: float = SAMPLE_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, PCA, pd.DataFrame]:
    """Sample rows, standardize features, run 2D PCA, and return plot data."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be greater than 0 and less than or equal to 1.")

    sampled_data = data.sample(frac=sample_fraction, random_state=random_state)

    # Preserve the original row alignment behavior.
    sampled_features = numeric_features.loc[sampled_data.index]

    x_scaled = StandardScaler().fit_transform(sampled_features.values)

    pca = PCA(n_components=2, random_state=random_state)
    x_pca = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
    pca_df["Label_Group"] = sampled_data["Label_Group"].values
    
    
    # ----------------------------------
    # Mirror controls
    # ----------------------------------
    pca_df["PC1"] = -pca_df["PC1"]   # left-right mirror
    # pca_df["PC2"] = -pca_df["PC2"] # up-down mirror

    return pca_df, pca, sampled_data


def plot_fig1_pca(
    pca_df: pd.DataFrame,
    title: str = PLOT_TITLE,
    output_path: Optional[Path] = OUTPUT_PATH,
    show: bool = True,
):
    """Plot the Figure 1 PCA with the preserved pink/purple styling."""
    sns.set(style="white")

    plt.figure(figsize=FIGSIZE)
    ax = sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Label_Group",
        palette=LABEL_PALETTE,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolor=POINT_EDGE_COLOR,
        linewidth=POINT_LINEWIDTH,
    )

    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return ax


def run_fig1_pca(
    data_path: Path = DATA_PATH,
    output_path: Optional[Path] = OUTPUT_PATH,
    sample_fraction: float = SAMPLE_FRACTION,
    random_state: int = RANDOM_STATE,
    show: bool = True,
) -> Tuple[pd.DataFrame, PCA, pd.DataFrame]:
    """Run the full Figure 1 PCA workflow."""
    data = load_and_filter_data(data_path)
    numeric_features = build_feature_matrix(data)
    pca_df, pca, sampled_data = compute_sampled_pca(
        data=data,
        numeric_features=numeric_features,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )

    print(f"Cell masks after filtering: {len(data):,}")
    print(f"Cell masks plotted: {len(sampled_data):,} ({sample_fraction:.0%} sample)")
    print(f"PCA features used: {numeric_features.shape[1]:,}")
    print(
        "Explained variance: "
        f"PC1={pca.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2={pca.explained_variance_ratio_[1] * 100:.1f}%"
    )
    
    label_counts = sampled_data["Label_Group"].value_counts(dropna=False).sort_index()

    label_counts = label_counts.rename(index={
    "Liver": "Liver cell masks",
    "Pancreas": "Pancreas cell masks",

    # These two lines are useful if your Label_Group still uses P/CNT internally.
    "CNT": "Liver cell masks",
    "P": "Pancreas cell masks",

    "Other": "Other cell masks",
    })

    label_counts.index.name = None

    print("Cell mask counts in plotted sample:")
    print(label_counts.to_string())
    
    # print("Label group counts:")
    # print(sampled_data["Label_Group"].value_counts(dropna=False).sort_index())

    plot_fig1_pca(pca_df=pca_df, output_path=output_path, show=show)
    return pca_df, pca, sampled_data


# ==================================================
# Command line use
# ==================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Figure 1 pink/purple PCA plot."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Optional path for saving the figure, for example fig1_pca.png.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=SAMPLE_FRACTION,
        help="Fraction of cells to sample for plotting. Default matches the original active code: 0.5.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducible sampling and PCA.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save/process without opening the plot window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fig1_pca(
        data_path=args.data,
        output_path=args.output,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state,
        show=not args.no_show,
    )
