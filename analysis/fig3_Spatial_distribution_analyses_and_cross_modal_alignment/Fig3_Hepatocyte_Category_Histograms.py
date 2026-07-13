"""
Histograms of category-wise acini-position distributions.

This script:
  1) one combined histogram with all prediction categories
  2) one separate histogram per prediction category

The KDE sections were removed.

Expected input columns:
  - Prediction column, default: "Prediction"
  - Position column, default: "ascini_position"
  - Probability/confidence columns, default: "0", "1", "2", "3", "4"
  - Categories are selected from high-confidence rows only.
  - Histograms are plotted using all rows belonging to those selected categories.

To plot only high-confidence rows in the histograms, set:
  USE_FILTERED_ROWS_FOR_HISTOGRAMS = True
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================


# This File is generated in Fig2_PCA_GMM_plots.py. You will find it in the "Full_Concat_data" folder
FILE_PATH = Path("...add path.../Full_Concat_data/FULL_CONCAT_clusters.csv")

OUTPUT_DIR = Path(".")

PREDICTION_COL = "Prediction"       # For some WD files this may be "Predictions"
POSITION_COL = "ascini_position"    # Keep this spelling if that is the actual CSV column name
PROBABILITY_COLUMNS = ["0", "1", "2", "3", "4"]

PROBABILITY_THRESHOLD = 0.80
BINS = 40
HIST_RANGE = (-1, 1)
XLIM = (-1.3, 1.3)

# Categories are selected from filtered rows, but histograms use all rows.
# Set this to True if you want the histograms to include only rows passing the probability filter.
USE_FILTERED_ROWS_FOR_HISTOGRAMS = False

SHOW_PLOTS = True
INDIVIDUAL_HISTOGRAM_COLOR = "#2F4F4F"


# =============================================================================
# Helpers
# =============================================================================

def safe_filename(value: object) -> str:
    """Return a filename-safe version of a category label."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or "category"


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + f"\nAvailable columns are: {list(df.columns)}"
        )


def high_confidence_filter(
    df: pd.DataFrame,
    probability_columns: Sequence[str],
    threshold: float,
) -> pd.Series:
    """
    Return a boolean mask for rows where any probability column is >= threshold.
    Non-numeric probability values are treated as missing and do not pass the filter.
    """
    require_columns(df, probability_columns)

    probabilities = df.loc[:, probability_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return probabilities.ge(float(threshold)).any(axis=1)


def save_or_show(fig: plt.Figure, output_path: Path, show: bool = True) -> None:
    """Save a Matplotlib figure, optionally show it, then close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved: {output_path.resolve()}")

    if show:
        plt.show()

    # plt.close(fig)


# =============================================================================
# Plotting
# =============================================================================

def plot_combined_histogram(
    df: pd.DataFrame,
    categories: Sequence[object],
    prediction_col: str,
    position_col: str,
    output_path: Path,
    bins: int = 40,
    hist_range: tuple[float, float] = (-1, 1),
    xlim: tuple[float, float] = (-1.3, 1.3),
    show: bool = True,
) -> None:
    """Plot all selected categories on one overlaid histogram."""
    fig, ax = plt.subplots()

    for category in categories:
        subset = df[df[prediction_col] == category]
        if subset.empty:
            continue

        ax.hist(
            subset[position_col],
            bins=bins,
            alpha=0.5,
            range=hist_range,
            label=str(category),
        )

    ax.set_xlabel("Acini Position")
    ax.set_ylabel("Cell count")
    ax.set_title("Histograms of Acini Position by Prediction Category")
    ax.legend()
    ax.set_xlim(*xlim)
    ax.invert_xaxis()

    save_or_show(fig, output_path, show=show)


def plot_individual_histograms(
    df: pd.DataFrame,
    categories: Sequence[object],
    prediction_col: str,
    position_col: str,
    output_dir: Path,
    bins: int = 40,
    hist_range: tuple[float, float] = (-1, 1),
    color: str = "#2F4F4F",
    show: bool = True,
) -> None:
    """Plot one histogram per selected category."""
    for category in categories:
        subset = df[df[prediction_col] == category]
        if subset.empty:
            print(f"[WARNING] Skipping empty category: {category}")
            continue

        fig, ax = plt.subplots()
        ax.hist(
            subset[position_col],
            bins=bins,
            alpha=0.8,
            range=hist_range,
            color=color,
        )

        ax.set_xlabel("Acini Position")
        ax.set_ylabel("Cell count")
        ax.set_title(f"Histogram of Acini Position for Category: {category}")
        ax.invert_xaxis()

        filename = f"histogram_{safe_filename(category)}.png"
        save_or_show(fig, output_dir / filename, show=show)


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    require_path = FILE_PATH.expanduser()
    if not require_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {require_path}")

    df = pd.read_csv(require_path)

    require_columns(
        df,
        [PREDICTION_COL, POSITION_COL, *PROBABILITY_COLUMNS],
    )

    mask = high_confidence_filter(
        df,
        probability_columns=PROBABILITY_COLUMNS,
        threshold=PROBABILITY_THRESHOLD,
    )
    filtered_df = df.loc[mask].copy()

    categories = list(filtered_df[PREDICTION_COL].dropna().unique())
    if not categories:
        raise ValueError(
            f"No rows passed the probability threshold of {PROBABILITY_THRESHOLD}. "
            "Try lowering PROBABILITY_THRESHOLD or checking PROBABILITY_COLUMNS."
        )

    histogram_df = filtered_df if USE_FILTERED_ROWS_FOR_HISTOGRAMS else df

    print(f"[INFO] Loaded rows: {len(df)}")
    print(f"[INFO] Rows passing confidence filter: {len(filtered_df)}")
    print(f"[INFO] Categories plotted: {categories}")
    print(
        "[INFO] Histogram row source: "
        + ("filtered high-confidence rows" if USE_FILTERED_ROWS_FOR_HISTOGRAMS else "all rows from selected categories")
    )

    plot_combined_histogram(
        histogram_df,
        categories=categories,
        prediction_col=PREDICTION_COL,
        position_col=POSITION_COL,
        output_path=OUTPUT_DIR / "histograms.png",
        bins=BINS,
        hist_range=HIST_RANGE,
        xlim=XLIM,
        show=SHOW_PLOTS,
    )

    plot_individual_histograms(
        histogram_df,
        categories=categories,
        prediction_col=PREDICTION_COL,
        position_col=POSITION_COL,
        output_dir=OUTPUT_DIR,
        bins=BINS,
        hist_range=HIST_RANGE,
        color=INDIVIDUAL_HISTOGRAM_COLOR,
        show=SHOW_PLOTS,
    )

    print("[DONE] Histogram plotting complete. KDE plots were not generated.")


if __name__ == "__main__":
    main()
