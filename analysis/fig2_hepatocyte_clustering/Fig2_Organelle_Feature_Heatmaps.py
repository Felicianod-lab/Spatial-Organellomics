"""
Organelle feature heatmaps by GMM cluster.


- matplotlib figure made with fig = plt.figure(figsize=(25, 25), dpi=400)
- one row/subplot per feature using GridSpec
- plt.imshow(..., cmap='bone', alpha=0.8)
- no x ticks and no y ticks
- feature names as horizontal y-axis labels, fontsize=5
- black frame around each feature row
- no gaps between rows: fig.subplots_adjust(hspace=0, wspace=0)

Input:
    FULL_CONCAT_clusters.csv

Default behavior:
    Plot general features for all three organelles:
        mitochondria, peroxisomes, lipid droplets

Optional:
    Set INCLUDE_SUBTYPES = True to append subtype features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# Editable settings
# -----------------------------------------------------------------------------
INPUT_CSV = Path(
    r"...add path.../FULL_CONCAT_clusters.csv"
)

OUTPUT_DIR = INPUT_CSV.parent / "Organelle_Feature_Heatmaps"

GROUP_COLUMN = "Prediction"
ORGANELLES_TO_PLOT = ["mito", "peroxisome", "ld"]

# Keep subtype features optional.
INCLUDE_SUBTYPES = False

# If the FULL_CONCAT_clusters.csv was already filtered upstream, this can be False.

APPLY_MITO_FILTERS = True
MIN_MITO_COUNT = 20
MIN_MITO_DENSITY = 0.0    #Use 0.28 for more stringency


CLUSTER_ORDER = [1, 2, 3, 4, 5]

# Save settings.
SAVE_FIGURES = True
SHOW_FIGURES = True

# Figure style.
FIGSIZE = (25, 25)
FIG_DPI = 400
GRID_NCOLS = 20
HEATMAP_CMAP = "bone"
HEATMAP_ALPHA = 0.8
FEATURE_LABEL_FONTSIZE = 5


# -----------------------------------------------------------------------------
# Feature lists
# -----------------------------------------------------------------------------
GENERAL_FEATURE_SUFFIXES = [
    "density",
    "avg_area",
    "percent_total_area",
    "aspect_ratio",
    "circularity",
    "perimeter",
    "solidity",
    "distance_from_edge",
]


ORG_PREFIX = {
    "mito": "mito",
    "peroxisome": "peroxisome",
    "ld": "ld",
}


ORG_DISPLAY_NAME = {
    "mito": "mitochondria",
    "peroxisome": "peroxisomes",
    "ld": "lipid_droplets",
}


SUBTYPE_COUNT = {
    "mito": 3,
    "peroxisome": 3,
    "ld": 4,
}


SUBTYPE_SUFFIXES_WITH_ASPECT_RATIO = [
    "density",
    "avg_area",
    "avg_aspect_ratio",
    "perimeter",
    "percent_total_area",
    "avg_solidity",
    "avg_circularity",
    "dist_from_edge",
]


SUBTYPE_SUFFIXES_WITHOUT_ASPECT_RATIO = [
    "density",
    "avg_area",
    "perimeter",
    "percent_total_area",
    "avg_solidity",
    "avg_circularity",
    "dist_from_edge",
]


def get_general_features(organelle: str) -> list[str]:
    """Return general feature names for one organelle."""
    prefix = ORG_PREFIX[organelle]
    return [f"{prefix}_{suffix}" for suffix in GENERAL_FEATURE_SUFFIXES]


def get_subtype_features(organelle: str) -> list[str]:
    """Return optional subtype feature names for one organelle."""
    prefix = ORG_PREFIX[organelle]
    subtype_features: list[str] = []

    if organelle == "ld":
        subtype_suffixes = SUBTYPE_SUFFIXES_WITHOUT_ASPECT_RATIO
    else:
        subtype_suffixes = SUBTYPE_SUFFIXES_WITH_ASPECT_RATIO

    for subtype_number in range(1, SUBTYPE_COUNT[organelle] + 1):
        for suffix in subtype_suffixes:
            subtype_features.append(f"type_{subtype_number}_{prefix}_{suffix}")
        subtype_features.append(f"percent_type_{subtype_number}_{prefix}")

    return subtype_features


def get_features_for_organelle(organelle: str, include_subtypes: bool = False) -> list[str]:
    """Return general features, optionally followed by subtype features."""
    features = get_general_features(organelle)

    if include_subtypes:
        features = features + get_subtype_features(organelle)

    return features


# -----------------------------------------------------------------------------
# Data loading and summarizing
# -----------------------------------------------------------------------------
def load_dataset(input_csv: Path) -> pd.DataFrame:
    """Load FULL_CONCAT_clusters.csv and apply basic missing/infinite cleanup."""
    dataset = pd.read_csv(input_csv, na_values="-")
    dataset.fillna(0, inplace=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)
    return dataset


def apply_optional_mito_filters(dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply the same mitochondrial filters used in the original heatmap code."""
    filtered = dataset.copy()

    if not APPLY_MITO_FILTERS:
        return filtered

    required_columns = ["mito_aspect_ratio", "mito_density", "area"]
    missing_columns = [col for col in required_columns if col not in filtered.columns]

    if missing_columns:
        raise KeyError(
            "Cannot apply mitochondrial filters because these columns are missing: "
            f"{missing_columns}"
        )

    filtered = filtered[filtered["mito_aspect_ratio"] > 0].copy()

    mito_count = filtered["mito_density"] * filtered["area"]
    filtered = filtered[mito_count >= MIN_MITO_COUNT].copy()

    filtered = filtered[filtered["mito_density"] >= MIN_MITO_DENSITY].copy()

    return filtered


def calculate_cluster_means(dataset: pd.DataFrame) -> pd.DataFrame:
    """Group by Prediction and calculate mean numeric feature values."""
    if GROUP_COLUMN not in dataset.columns:
        raise KeyError(f"Column '{GROUP_COLUMN}' was not found in the input CSV.")

    grouped = dataset.groupby(GROUP_COLUMN).mean(numeric_only=True)

    # Keep clusters in a stable biological order when possible.
    available_order = [cluster for cluster in CLUSTER_ORDER if cluster in grouped.index]
    remaining_order = [cluster for cluster in grouped.index if cluster not in available_order]

    if available_order:
        grouped = grouped.loc[available_order + remaining_order]

    return grouped


def filter_existing_features(binned_props: pd.DataFrame, features: list[str], organelle: str) -> list[str]:
    """Keep only features that are present in the input table and report missing ones."""
    existing_features = [feature for feature in features if feature in binned_props.columns]
    missing_features = [feature for feature in features if feature not in binned_props.columns]

    if missing_features:
        print(f"\nMissing columns skipped for {organelle}:")
        for feature in missing_features:
            print(f"  - {feature}")

    if not existing_features:
        raise ValueError(f"No requested features were found for {organelle}.")

    return existing_features


# -----------------------------------------------------------------------------
# Plotting: 
# -----------------------------------------------------------------------------
def plot_feature_rows_same_style(
    binned_props: pd.DataFrame,
    prop_list: list[str],
    output_name: str,
) -> None:
    """
    Plot feature rows.

    This intentionally uses plt.imshow and one GridSpec row per feature rather
    than sns.heatmap, because that is what produced the plot style.
    """
    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)
    gs = fig.add_gridspec(len(prop_list), GRID_NCOLS)

    for i, prop in enumerate(prop_list):
        ax = fig.add_subplot(gs[i, :])

        single_prop = binned_props[prop]
        prop_plot = np.expand_dims(np.array(single_prop), 0)

        
        sns.set_palette(sns.cubehelix_palette(start=0.5, rot=-0.5))

        plt.imshow(prop_plot, cmap=HEATMAP_CMAP, alpha=HEATMAP_ALPHA)

        plt.xticks([])
        plt.yticks([])
        plt.ylabel(
            prop,
            rotation=0,
            ha="right",
            va="center",
            fontsize=FEATURE_LABEL_FONTSIZE,
        )

        ax.set_frame_on(True)

    fig.subplots_adjust(hspace=0, wspace=0)

    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / f"{output_name}.png")
        fig.savefig(OUTPUT_DIR / f"{output_name}.svg")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_organelle_features(binned_props: pd.DataFrame, organelle: str) -> None:
    """Plot one organelle using the exact heatmap style."""
    requested_features = get_features_for_organelle(
        organelle=organelle,
        include_subtypes=INCLUDE_SUBTYPES,
    )

    prop_list = filter_existing_features(
        binned_props=binned_props,
        features=requested_features,
        organelle=organelle,
    )

    subtype_tag = "with_subtypes" if INCLUDE_SUBTYPES else "general_features"
    output_name = f"{organelle}_{subtype_tag}_same_style"

    plot_feature_rows_same_style(
        binned_props=binned_props,
        prop_list=prop_list,
        output_name=output_name,
    )


def save_summary_tables(binned_props: pd.DataFrame) -> None:
    """Save the grouped means used to generate the heatmaps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    binned_props.to_csv(OUTPUT_DIR / "cluster_mean_features.csv")

    report_rows = []
    for organelle in ORGANELLES_TO_PLOT:
        requested_features = get_features_for_organelle(organelle, INCLUDE_SUBTYPES)
        for feature in requested_features:
            report_rows.append(
                {
                    "organelle": organelle,
                    "feature": feature,
                    "present_in_input": feature in binned_props.columns,
                }
            )

    pd.DataFrame(report_rows).to_csv(
        OUTPUT_DIR / "feature_plotting_report.csv",
        index=False,
    )


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    dataset = load_dataset(INPUT_CSV)
    dataset = apply_optional_mito_filters(dataset)
    binned_props = calculate_cluster_means(dataset)

    save_summary_tables(binned_props)

    for organelle in ORGANELLES_TO_PLOT:
        plot_organelle_features(binned_props, organelle)


if __name__ == "__main__":
    main()
