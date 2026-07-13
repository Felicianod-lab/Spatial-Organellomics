"""Calculate and plot heterogeneity indices across binned positions.

Choose whether to analyze proteomics, organellomics, or both.
The plotting style is kept the same as the original script.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Choose what to plot
# -----------------------------------------------------------------------------
# Options: "proteomics", "organellomics", or "both"
PLOT_MODE = "both"


# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------
PROTEOMICS_FILE_PATH = "...add path.../Fig3_Adapted_Spatial_Proteomic_Data.csv"

# This File is generated in Fig2_PCA_GMM_plots.py. You will find it in the "Full_Concat_data" folder
ORGANELLOMICS_FILE_PATH = "...add path.../Full_Concat_data/FULL_CONCAT_clusters.csv"



PROTEOMICS_OUTPUT_PLOT_PATH = (
    "proteomics_pittsburgh_indices_plot.png"
)
ORGANELLOMICS_OUTPUT_PLOT_PATH = (
    "organellomics_pittsburgh_indices_plot.png"
)


# -----------------------------------------------------------------------------
# Column names
# -----------------------------------------------------------------------------
PROTEOMICS_POSITION_COLUMN = "RATIO"
ORGANELLOMICS_POSITION_COLUMN = "ascini_position"
PREDICTION_COLUMN = "Prediction"
BIN_COLUMN = "ascini_position_binned"


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
N_BINS = 20


# -----------------------------------------------------------------------------
# Plot settings: kept identical to the original plot style
# -----------------------------------------------------------------------------
FIGSIZE = (10, 5)
BAR_WIDTH = 0.5
D_COLOR = "lightgray"
H_COLOR = "purple"
D_ALPHA = 1.0
H_ALPHA = 0.5
AXIS_LABEL_SIZE = 18
Y_TICK_LABEL_SIZE = 18
X_TICK_LABEL_SIZE = 1


@dataclass(frozen=True)
class DatasetConfig:
    """Settings for one dataset."""

    name: str
    file_path: str
    position_column: str
    output_plot_path: str


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "proteomics": DatasetConfig(
        name="proteomics",
        file_path=PROTEOMICS_FILE_PATH,
        position_column=PROTEOMICS_POSITION_COLUMN,
        output_plot_path=PROTEOMICS_OUTPUT_PLOT_PATH,
    ),
    "organellomics": DatasetConfig(
        name="organellomics",
        file_path=ORGANELLOMICS_FILE_PATH,
        position_column=ORGANELLOMICS_POSITION_COLUMN,
        output_plot_path=ORGANELLOMICS_OUTPUT_PLOT_PATH,
    ),
}


def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file used for the heterogeneity analysis."""
    return pd.read_csv(file_path)


def validate_columns(
    df: pd.DataFrame,
    position_column: str,
    prediction_column: str = PREDICTION_COLUMN,
) -> None:
    """Check that the required columns are present before running the analysis."""
    required_columns = {position_column, prediction_column}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")


def add_position_bins(df: pd.DataFrame, position_column: str) -> pd.DataFrame:
    """Bin the position column into numbered position bins."""
    df = df.copy()
    df[BIN_COLUMN] = pd.cut(
        df[position_column],
        bins=N_BINS,
        labels=range(1, N_BINS + 1),
    )
    return df


def calculate_prediction_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the fraction of each prediction class within each position bin."""
    proportions = (
        df.groupby(BIN_COLUMN)[PREDICTION_COLUMN]
        .value_counts(normalize=True)
        .unstack()
    )
    return proportions.fillna(0)


def pittsburgh_heterogeneity_indices(probabilities) -> Tuple[float, float, float]:
    """Return Simpson's Index, Shannon's Entropy, and Shannon evenness."""
    probabilities = np.asarray(probabilities, dtype=float)
    total_prob = np.sum(probabilities)

    if total_prob == 0:
        return 0.0, 0.0, 0.0

    normalized_probs = probabilities / total_prob
    safe_probs = np.where(normalized_probs == 0, 1, normalized_probs)

    simpsons_index = np.sum(normalized_probs ** 2)
    shannons_entropy = -np.sum(normalized_probs * np.log2(safe_probs))

    num_categories = len(probabilities)
    evenness = shannons_entropy / np.log2(num_categories) if num_categories > 1 else 0.0

    return float(simpsons_index), float(shannons_entropy), float(evenness)


def calculate_indices(grouped_proportions: pd.DataFrame) -> pd.DataFrame:
    """Calculate heterogeneity indices for each binned position."""
    results = []

    for position_bin, probabilities in grouped_proportions.iterrows():
        simpsons_index, shannons_entropy, _ = pittsburgh_heterogeneity_indices(
            probabilities
        )
        results.append(
            {
                BIN_COLUMN: position_bin,
                "D": simpsons_index,
                "H": shannons_entropy,
            }
        )

    return pd.DataFrame(results)


def plot_indices(results_df: pd.DataFrame, output_path: str) -> None:
    """Plot Simpson's Index and Shannon's Entropy using the original style."""
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    plt.gca().invert_xaxis()

    ax1.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
    ax1.set_ylabel("Simpson’s Index (D)", color="gray", fontsize=AXIS_LABEL_SIZE)

    positions = np.arange(len(results_df[BIN_COLUMN]))

    ax1.bar(
        positions - BAR_WIDTH / 2,
        results_df["D"],
        color=D_COLOR,
        alpha=D_ALPHA,
        width=BAR_WIDTH,
        label="D Index",
    )

    ax2 = ax1.twinx()
    ax2.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
    ax2.set_ylabel("Shannon’s Entropy (H)", color="purple", fontsize=AXIS_LABEL_SIZE)

    ax2.bar(
        positions + BAR_WIDTH / 2,
        results_df["H"],
        color=H_COLOR,
        alpha=H_ALPHA,
        width=BAR_WIDTH,
        label="H Index",
    )

    plt.xticks(
        positions,
        results_df[BIN_COLUMN],
        rotation=0,
        fontsize=X_TICK_LABEL_SIZE,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def run_analysis(config: DatasetConfig) -> None:
    """Run the complete heterogeneity analysis and plotting workflow for one dataset."""
    print(f"Running {config.name} analysis...")
    print(f"Input file: {config.file_path}")
    print(f"Position column: {config.position_column}")

    df = load_data(config.file_path)
    validate_columns(df, position_column=config.position_column)

    # Optional example for analyzing only one label subset:
    # df = df[df["labels"].str.contains("CNT_L3", na=False)].copy()

    df = add_position_bins(df, position_column=config.position_column)
    grouped_proportions = calculate_prediction_proportions(df)
    results_df = calculate_indices(grouped_proportions)

    plot_indices(results_df, config.output_plot_path)

    print(
        "Combined plot with side-by-side bars for Simpson’s Index (D) and "
        f"Shannon’s Entropy (H) saved as {config.output_plot_path}"
    )


def get_selected_datasets(plot_mode: str) -> List[DatasetConfig]:
    """Return the dataset configuration(s) requested by the selected plot mode."""
    plot_mode = plot_mode.lower().strip()

    if plot_mode == "both":
        return [DATASET_CONFIGS["proteomics"], DATASET_CONFIGS["organellomics"]]

    if plot_mode not in DATASET_CONFIGS:
        valid_modes = ", ".join(["proteomics", "organellomics", "both"])
        raise ValueError(f"Invalid PLOT_MODE: {plot_mode}. Choose one of: {valid_modes}.")

    return [DATASET_CONFIGS[plot_mode]]


def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate and plot heterogeneity indices."
    )
    parser.add_argument(
        "--mode",
        choices=["proteomics", "organellomics", "both"],
        default=PLOT_MODE,
        help=(
            "Choose which dataset to plot. If not provided, the script uses "
            "the PLOT_MODE value set near the top of the file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected heterogeneity analysis workflow."""
    args = parse_args()
    selected_datasets = get_selected_datasets(args.mode)

    for config in selected_datasets:
        run_analysis(config)


if __name__ == "__main__":
    main()
