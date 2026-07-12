"""
Plot Pittsburgh heterogeneity-index curves for H and D from the consolidated CSV.

This script expects the CSV produced by:
    Fig4_PCA_GMM_Plots_by_Experimental_Group.py

Required wide columns:
    ascini_position_binned
    C_H, F_H, W_H
    C_D, F_D, W_D

The plot style for Pittsburgh-index script:
    - raw index plot: figsize=(12, 4.5), line width 4, inverted x-axis
    - percent-change plot: figsize=(12, 3), dash-dot lines, control line at 0
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Editable settings
# -----------------------------------------------------------------------------
# Point this to the CSV produced by the GMM/Pittsburgh export script.
PITTSBURGH_CSV = Path(
    r"...add path...\Full_Concat_data_by_experiment\combineHD_pittsburgh_indices.csv"
)

OUTPUT_DIR = PITTSBURGH_CSV.parent / "Pittsburgh_H_D_plots"

# Plot both H and D. You can also use ["H"] or ["D"].
INDEXES_TO_PLOT = ["H", "D"]

# These prefixes match the consolidated CSV:
# C = CNT/control, F = STV, W = WD.
CONTROL_PREFIX = "C"
STV_PREFIX = "F"
WD_PREFIX = "W"

CONTROL_LABEL = "Control"
STV_LABEL = "STV"
WD_LABEL = "WD"

CONTROL_COLOR = "black"
STV_COLOR = "blue"
WD_COLOR = "Purple"

SHOW_FIGURES = True
SAVE_FIGURES = True
DPI = 500


# -----------------------------------------------------------------------------
# Fixed y-axis limits to reproduce the old Pittsburgh figures
# -----------------------------------------------------------------------------
RAW_INDEX_YLIMS = {
    "D": (0.2, 1.10),
    "H": (0.0, 2.6),
}

RAW_INDEX_YTICKS = {
    "D": [0.2, 0.4, 0.6, 0.8, 1.0],
    "H": [0, 1, 2],
}

PERCENT_CHANGE_YLIMS = {
    "D": (-220, 220),
    "H": (-100, 100),
}

PERCENT_CHANGE_YTICKS = {
    "D": [-200, 0, 200],
    "H": [-100, 0, 100],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def index_ylabel(index_name: str) -> str:
    """Return the y-axis label used for each index."""
    if index_name == "H":
        return "Shannon’s Entropy (H)"
    if index_name == "D":
        return "Simpson’s Index (D)"
    if index_name == "E":
        return "Evenness (E)"
    return f"{index_name} Index"


def save_current_figure(
    output_path_png: Path,
    output_path_svg: Optional[Path] = None,
) -> None:
    """Save the current matplotlib figure as PNG and optionally SVG."""
    if not SAVE_FIGURES:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_png, dpi=DPI, bbox_inches="tight")

    if output_path_svg is not None:
        plt.savefig(output_path_svg, dpi=DPI, bbox_inches="tight")

    print(f"[INFO] Saved: {output_path_png}")


def plot_raw_index_curve(results_df: pd.DataFrame, index_name: str) -> None:
    """Plot raw Pittsburgh H or D curve with the same style as the original code."""
    x = results_df["ascini_position_binned"]

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    plt.gca().invert_xaxis()
    fig.patch.set_facecolor("white")

    ax1.plot(
        x,
        results_df[f"{CONTROL_PREFIX}_{index_name}"],
        label=f"{CONTROL_LABEL} {index_name} Index",
        linewidth=4,
        color=CONTROL_COLOR,
    )

    ax1.plot(
        x,
        results_df[f"{STV_PREFIX}_{index_name}"],
        label=f"{STV_LABEL} {index_name} Index",
        linewidth=4,
        color=STV_COLOR,
    )

    ax1.plot(
        x,
        results_df[f"{WD_PREFIX}_{index_name}"],
        label=f"{WD_LABEL} {index_name} Index",
        linewidth=4,
        color=WD_COLOR,
    )

    plt.xlabel("Binned Position")
    ax1.set_ylabel(index_ylabel(index_name), color="black", fontsize=28)
    ax1.tick_params(axis="y", labelsize=23)

    if index_name in RAW_INDEX_YLIMS:
        ax1.set_ylim(*RAW_INDEX_YLIMS[index_name])

    if index_name in RAW_INDEX_YTICKS:
        ax1.set_yticks(RAW_INDEX_YTICKS[index_name])

    save_current_figure(
        OUTPUT_DIR / f"pittsburgh_{index_name}_raw_curve.png",
        OUTPUT_DIR / f"pittsburgh_{index_name}_raw_curve.svg",
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_percentage_change_curve(
    results_df: pd.DataFrame,
    index_name: str,
) -> pd.DataFrame:
    """Plot STV and WD percent changes relative to control for H or D."""
    x = results_df["ascini_position_binned"]
    results_df = results_df.copy()

    control_values = pd.to_numeric(
        results_df[f"{CONTROL_PREFIX}_{index_name}"],
        errors="coerce",
    ).replace(0, np.nan)

    results_df[f"{STV_PREFIX}_{index_name}_percentage_change"] = (
        (
            pd.to_numeric(
                results_df[f"{STV_PREFIX}_{index_name}"],
                errors="coerce",
            )
            - control_values
        )
        / control_values
    ) * 100

    results_df[f"{WD_PREFIX}_{index_name}_percentage_change"] = (
        (
            pd.to_numeric(
                results_df[f"{WD_PREFIX}_{index_name}"],
                errors="coerce",
            )
            - control_values
        )
        / control_values
    ) * 100

    fig, ax2 = plt.subplots(figsize=(12, 3))
    plt.gca().invert_xaxis()
    fig.patch.set_facecolor("white")

    ax2.plot(
        x,
        results_df[f"{STV_PREFIX}_{index_name}_percentage_change"],
        label=f"{STV_LABEL} {index_name} % Change",
        linestyle="-.",
        linewidth=3,
        color=STV_COLOR,
    )

    ax2.plot(
        x,
        results_df[f"{WD_PREFIX}_{index_name}_percentage_change"],
        label=f"{WD_LABEL} {index_name} % Change",
        linestyle="-.",
        linewidth=3,
        color=WD_COLOR,
    )

    plt.axhline(
        y=0,
        color="black",
        linestyle="-.",
        label="Control (% Change = 0)",
        linewidth=4,
    )

    if index_name in PERCENT_CHANGE_YLIMS:
        ax2.set_ylim(*PERCENT_CHANGE_YLIMS[index_name])

    if index_name in PERCENT_CHANGE_YTICKS:
        ax2.set_yticks(PERCENT_CHANGE_YTICKS[index_name])

    plt.xlabel("Binned Position")
    ax2.set_ylabel("% Change", color="black", fontsize=28)
    ax2.tick_params(axis="y", labelsize=23)

    save_current_figure(
        OUTPUT_DIR / f"pittsburgh_{index_name}_percentage_change.png",
        OUTPUT_DIR / f"pittsburgh_{index_name}_percentage_change.svg",
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return results_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    if not PITTSBURGH_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {PITTSBURGH_CSV}. Update PITTSBURGH_CSV near the top "
            "or run the GMM/Pittsburgh export script first."
        )

    results_df = pd.read_csv(PITTSBURGH_CSV)
    all_results = results_df.copy()

    required_base = ["ascini_position_binned"]
    missing_base = [
        col for col in required_base
        if col not in results_df.columns
    ]

    if missing_base:
        raise KeyError(f"Missing required columns: {missing_base}")

    for index_name in INDEXES_TO_PLOT:
        required_columns = [
            f"{CONTROL_PREFIX}_{index_name}",
            f"{STV_PREFIX}_{index_name}",
            f"{WD_PREFIX}_{index_name}",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in results_df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing columns for index {index_name}: {missing_columns}. "
                "Check that the consolidated CSV was created for CNT, STV, and WD."
            )

        plot_raw_index_curve(results_df, index_name)
        all_results = plot_percentage_change_curve(all_results, index_name)

    output_with_changes = OUTPUT_DIR / "combineHD_pittsburgh_indices_with_percentage_changes.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_with_changes, index=False)

    print(
        "[INFO] Saved percentage-change table to: "
        f"{output_with_changes}"
    )


if __name__ == "__main__":
    main()