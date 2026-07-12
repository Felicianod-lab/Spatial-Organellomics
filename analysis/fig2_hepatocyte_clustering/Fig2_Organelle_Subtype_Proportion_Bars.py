"""
Subtype-proportion bar plots for Figure 2 GMM clusters.

This script reads FULL_CONCAT_clusters.csv automatically making the plot for all three
organelles:

    mito
    peroxisome
    ld

Optional extra individual subtype plots are available but turned off
by default.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Editable settings
# -----------------------------------------------------------------------------
INPUT_CSV = Path(
    r"...add path.../FULL_CONCAT_clusters.csv"
)

OUTPUT_DIR = INPUT_CSV.parent / "Organelle_Subtype_Proportion_Bars"

PREDICTION_COLUMN = "Prediction"
ORGANELLES_TO_PLOT = ["mito", "peroxisome", "ld"]

# The plot is a stacked subtype-proportion bar plot.
MAKE_STACKED_SUBTYPE_PROPORTION_PLOTS = True

# Optional extra plots, one plot per individual subtype. Off by default.
MAKE_INDIVIDUAL_SUBTYPE_PLOTS = False


APPLY_MITO_FILTERS = True
MIN_MITO_COUNT = 20
MIN_MITO_DENSITY = 0.0

# Optional group filter. Leave as None to use all rows.
GROUP_FILTER = None
GROUP_COLUMN = "group"

# If CLUSTER_TO_CATEGORY is None, the script automatically maps sorted Prediction
# values to H1, H2, H3, ... .
# This works for both original labels 0-4 and remapped labels 1-5.
CLUSTER_TO_CATEGORY = None
# Example manual versions:
# CLUSTER_TO_CATEGORY = {0: "H1", 1: "H2", 2: "H3", 3: "H4", 4: "H5"}
# CLUSTER_TO_CATEGORY = {1: "H1", 2: "H2", 3: "H3", 4: "H4", 5: "H5"}

# Number of subtype classes for each organelle.
ORGANELLE_TYPE_COUNTS = {
    "mito": 3,
    "peroxisome": 3,
    "ld": 4,
}

# Visual style settings.
CUSTOM_COLOR = "#2F4F4F"
FIGSIZE = (1.8, 3)
DPI = 400
YLIM = (0, 1.01)

SHOW_FIGURES = True
SAVE_FIGURES = True
SAVE_SVG = True


# -----------------------------------------------------------------------------
# Data loading and filtering
# -----------------------------------------------------------------------------
def load_and_filter_data(input_csv: Path) -> pd.DataFrame:
    """Load FULL_CONCAT_clusters.csv and apply filters."""
    df = pd.read_csv(input_csv)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if GROUP_FILTER is not None:
        if GROUP_COLUMN not in df.columns:
            raise KeyError(f"GROUP_COLUMN '{GROUP_COLUMN}' was not found in the CSV.")
        df = df[df[GROUP_COLUMN] == GROUP_FILTER].copy()

    if APPLY_MITO_FILTERS:
        required_columns = ["mito_aspect_ratio", "mito_density", "area"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(
                "Cannot apply mitochondrial filters because these columns are missing: "
                f"{missing_columns}"
            )

        mito_positive = df[df["mito_aspect_ratio"] > 0].copy()
        mito_count = mito_positive["mito_density"] * mito_positive["area"]

        df = mito_positive[
            (mito_count >= MIN_MITO_COUNT)
            & (mito_positive["mito_density"] >= MIN_MITO_DENSITY)
        ].copy()

    if PREDICTION_COLUMN not in df.columns:
        raise KeyError(f"PREDICTION_COLUMN '{PREDICTION_COLUMN}' was not found in the CSV.")

    df[PREDICTION_COLUMN] = pd.to_numeric(df[PREDICTION_COLUMN], errors="raise").astype(int)

    return df


def make_cluster_categories(df: pd.DataFrame) -> Dict[int, str]:
    """Create H1, H2, ... labels from sorted Prediction values, unless manually set."""
    if CLUSTER_TO_CATEGORY is not None:
        return dict(CLUSTER_TO_CATEGORY)

    cluster_ids = sorted(df[PREDICTION_COLUMN].dropna().astype(int).unique().tolist())
    return {cluster_id: f"H{i + 1}" for i, cluster_id in enumerate(cluster_ids)}


# -----------------------------------------------------------------------------
# Proportion calculation
# -----------------------------------------------------------------------------
def subtype_percent_columns(organelle: str) -> List[str]:
    """Return percent_type columns for an organelle."""
    if organelle not in ORGANELLE_TYPE_COUNTS:
        raise KeyError(
            f"Unknown organelle '{organelle}'. Add it to ORGANELLE_TYPE_COUNTS."
        )

    n_types = ORGANELLE_TYPE_COUNTS[organelle]
    return [f"percent_type_{i}_{organelle}" for i in range(1, n_types + 1)]


def calculate_subtype_proportions(
    df: pd.DataFrame,
    organelle: str,
    categories: Dict[int, str],
) -> pd.DataFrame:
    """
    1. group by Prediction
    2. sum percent_type columns within each cluster
    3. divide each subtype sum by the total subtype sum for that cluster
    """
    type_columns = subtype_percent_columns(organelle)
    missing_columns = [col for col in type_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(
            f"Cannot plot '{organelle}'. These subtype columns are missing: {missing_columns}"
        )

    cluster_order = list(categories.keys())

    grouped_df = (
        df.groupby(PREDICTION_COLUMN)
        .sum(numeric_only=True)
        .reindex(cluster_order)
        .fillna(0)
    )

    total = grouped_df[type_columns].sum(axis=1)
    proportions = grouped_df[type_columns].div(total.replace(0, np.nan), axis=0).fillna(0)

    rename_dict = {
        column: f"Type{i + 1}_Proportion"
        for i, column in enumerate(type_columns)
    }
    proportions = proportions.rename(columns=rename_dict)

    proportions.insert(0, "HepCategory", [categories[idx] for idx in proportions.index])
    proportions.insert(0, PREDICTION_COLUMN, proportions.index)

    return proportions.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def style_y_axis_as_percent(ax) -> None:
   
    ax.set_yticklabels([f"{int(y * 100)}" for y in ax.get_yticks()])


def plot_stacked_subtype_proportions(
    proportions: pd.DataFrame,
    organelle: str,
    output_dir: Path,
) -> None:
    """Create the same stacked bar plot style."""
    category_labels = proportions["HepCategory"].tolist()
    n_types = ORGANELLE_TYPE_COUNTS[organelle]

    plt.figure(figsize=FIGSIZE, dpi=DPI)

    bottom = np.zeros(len(proportions), dtype=float)

    # Type 1: 
    plt.bar(
        category_labels,
        proportions["Type1_Proportion"],
        label="Type 1",
        color="black",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bottom = bottom + proportions["Type1_Proportion"].to_numpy(dtype=float)

    # Type 2:
    if n_types >= 2:
        plt.bar(
            category_labels,
            proportions["Type2_Proportion"],
            bottom=bottom,
            label="Type 2",
            color=CUSTOM_COLOR,
            alpha=0.1,
            edgecolor="black",
            linewidth=2,
        )
        bottom = bottom + proportions["Type2_Proportion"].to_numpy(dtype=float)

    # Type 3: 
    if n_types >= 3:
        plt.bar(
            category_labels,
            proportions["Type3_Proportion"],
            bottom=bottom,
            label="Type 3",
            color=CUSTOM_COLOR,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        bottom = bottom + proportions["Type3_Proportion"].to_numpy(dtype=float)

    # Type 4: 
    if n_types >= 4:
        plt.bar(
            category_labels,
            proportions["Type4_Proportion"],
            bottom=bottom,
            label="Type 4",
            color=CUSTOM_COLOR,
            alpha=1,
            edgecolor="black",
            linewidth=1,
        )

    plt.ylabel("Proportion", fontsize=5)

    legend = plt.legend(loc="upper left", bbox_to_anchor=(1.5, 1.5), fontsize=2)
    legend.get_frame().set_facecolor("none")

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.ylim(*YLIM)

    ax = plt.gca()
    ax.set_facecolor("none")
    style_y_axis_as_percent(ax)

    if SAVE_FIGURES:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{organelle}_subtype_proportion_stacked_bar.png")
        if SAVE_SVG:
            plt.savefig(output_dir / f"{organelle}_subtype_proportion_stacked_bar.svg")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def plot_individual_subtype_proportions(
    proportions: pd.DataFrame,
    organelle: str,
    output_dir: Path,
) -> None:
    """
    Optional extra plots: one separate bar plot per subtype.

    This is off by default. It keeps the same small figure, y-axis, font, and
    transparent-background settings as the original plot.
    """
    category_labels = proportions["HepCategory"].tolist()
    n_types = ORGANELLE_TYPE_COUNTS[organelle]

    for type_idx in range(1, n_types + 1):
        column = f"Type{type_idx}_Proportion"

        plt.figure(figsize=FIGSIZE, dpi=DPI)
        plt.bar(
            category_labels,
            proportions[column],
            label=f"Type {type_idx}",
            color="black" if type_idx == 1 else CUSTOM_COLOR,
            alpha={1: 0.8, 2: 0.1, 3: 0.8, 4: 1}.get(type_idx, 0.8),
            edgecolor="black",
            linewidth=2 if type_idx == 2 else 1,
        )

        plt.ylabel("Proportion", fontsize=5)

        legend = plt.legend(loc="upper left", bbox_to_anchor=(1.5, 1.5), fontsize=2)
        legend.get_frame().set_facecolor("none")

        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        plt.ylim(*YLIM)

        ax = plt.gca()
        ax.set_facecolor("none")
        style_y_axis_as_percent(ax)

        if SAVE_FIGURES:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"{organelle}_type_{type_idx}_proportion_bar.png")
            if SAVE_SVG:
                plt.savefig(output_dir / f"{organelle}_type_{type_idx}_proportion_bar.svg")

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    df = load_and_filter_data(INPUT_CSV)
    categories = make_cluster_categories(df)

    print("Cluster category mapping:")
    print(categories)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_proportion_tables = []

    for organelle in ORGANELLES_TO_PLOT:
        proportions = calculate_subtype_proportions(
            df=df,
            organelle=organelle,
            categories=categories,
        )

        proportions.insert(0, "organelle", organelle)
        all_proportion_tables.append(proportions.copy())

        proportions_for_plot = proportions.drop(columns=["organelle"])

        if MAKE_STACKED_SUBTYPE_PROPORTION_PLOTS:
            plot_stacked_subtype_proportions(
                proportions=proportions_for_plot,
                organelle=organelle,
                output_dir=OUTPUT_DIR,
            )

        if MAKE_INDIVIDUAL_SUBTYPE_PLOTS:
            plot_individual_subtype_proportions(
                proportions=proportions_for_plot,
                organelle=organelle,
                output_dir=OUTPUT_DIR,
            )

    all_proportions = pd.concat(all_proportion_tables, axis=0, ignore_index=True)
    all_proportions.to_csv(OUTPUT_DIR / "subtype_proportions_by_cluster.csv", index=False)

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
