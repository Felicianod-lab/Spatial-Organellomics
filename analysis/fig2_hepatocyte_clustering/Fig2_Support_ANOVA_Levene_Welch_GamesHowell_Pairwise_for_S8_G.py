"""
Clean one-way ANOVA + Levene + Tukey HSD workflow for cluster feature analysis.

This is a cleaned version of the Levene statistics script. It keeps the same
main visual style as the original script:
- horizontal Seaborn bar plots
- viridis palette for ANOVA p-value ranking
- organelle-colored / semi-transparent effect-size bars
- BuPu palette for Tukey percentage mean-difference bars
- dashed x-axis grid lines

Main outputs:
1. Levene test results for homogeneity of variance.
2. One-way ANOVA results for each numeric biological feature.
3. Partial eta squared effect size for each feature.
4. Welch ANOVA results for features that may violate equal variance assumptions.
5. Welch-compatible Games-Howell pairwise comparisons for selected top features.
6. Tukey HSD pairwise comparisons for selected top ANOVA features.
7. Clean figures with smaller text and legends/text boxes outside the bar area.

Important: This script still performs standard one-way ANOVA, even for features
that fail Levene's test. The Levene result is reported so you can decide whether
ANOVA assumptions are acceptable for each feature.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f as f_distribution, f_oneway, levene, studentized_range
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# -----------------------------------------------------------------------------
# Editable settings
# -----------------------------------------------------------------------------
INPUT_CSV = Path(r"...add path.../FULL_CONCAT_clusters.csv")  #<--- Generated from Fig2_PCA_GMM_plots.py 

OUTPUT_DIR = INPUT_CSV.parent / "ANOVA"

# Leave as None to auto-detect Prediction or Predictions.
# Set to "Prediction" or "Predictions" if you want to force one column.
CLUSTER_COLUMN: str | None = None
CLUSTER_COLUMN_CANDIDATES = ["Prediction", "Predictions"]

# Optional filter.
# Example: GROUP_FILTER_VALUE = 1 to keep only dataset["group"] == 1.
GROUP_FILTER_VALUE: int | float | str | None = None
GROUP_COLUMN = "group"

# Same mitochondrial quality filter used in the original script.
MIN_MITO_COUNT = 20

# Set to 0.28 if you want to add the newer mito_density filter.

MIN_MITO_DENSITY: float | None = None

ALPHA = 0.05

# Plot/output controls.
SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_DPI = 300

# Limit large summary plots so feature names remain readable.
# Set to None to plot every significant feature.
MAX_FEATURES_IN_SUMMARY_PLOTS: int | None = 80

# Run Welch ANOVA in addition to regular one-way ANOVA.
# Welch ANOVA is useful when Levene indicates unequal variances.
MAKE_WELCH_ANOVA = True
MAX_FEATURES_IN_WELCH_PLOT: int | None = 80

# Games-Howell plots are the Welch-compatible pairwise comparison plots.
# They can produce many images; start with a smaller number per organelle.
MAKE_GAMES_HOWELL_PLOTS = True
TOP_N_GAMES_HOWELL_FEATURES_PER_ORGANELLE = 200

# Tukey plots can produce many images. 
# Tukey is kept for the standard one-way ANOVA workflow.
MAKE_TUKEY_PLOTS = True
TOP_N_TUKEY_FEATURES_PER_ORGANELLE = 30

# Optional summary charts .
MAKE_CATEGORY_SUMMARY_CHARTS = True
TOP_N_FEATURES_FOR_CATEGORY_SUMMARY = 50


# -----------------------------------------------------------------------------
# Plot style settings
# -----------------------------------------------------------------------------
sns.set()
plt.rcParams["figure.dpi"] = FIGURE_DPI

FONT_TITLE = 9
FONT_AXIS = 8
FONT_TICK_X = 7
FONT_TICK_Y = 5
FONT_LEGEND = 6
FONT_TUKEY_TICK_X = 5
FONT_TUKEY_TICK_Y = 6
FONT_TUKEY_PVALUES = 5

ORGANELLE_COLORS = {
    "mito": "green",
    "peroxisome": "magenta",
    "ld": "darkgoldenrod",
    "other": "gray",
}

ORGANELLE_DISPLAY_NAMES = {
    "mito": "Mitochondria",
    "peroxisome": "Peroxisome",
    "ld": "Lipid droplets",
    "other": "Other",
}


# -----------------------------------------------------------------------------
# Columns that should not be tested as biological features
# -----------------------------------------------------------------------------
BASE_COLUMNS_TO_REMOVE = [
    # Metadata / identifiers
    "Unnamed: 0",
    "Unnamed: 0.1",
    "label",
    "labels",
    "stack_id",
    "cell_id_linked",
    "asinus",
    "cutoff",
    "HepCategory",

    # Position / geometry columns that should not be part of this feature screen
    "ascini_position",
    "centroid-0",
    "centroid-1",
    "area",

    # GMM/PCA model-derived columns
    "Prediction",
    "Predictions",
    "Prediction_original",
    "Old_Predictions",
    "Probability",

    # Optional experimental group column
    "group",

    # Columns excluded in the original script
    "ld_aspect_ratio",
    "type_1_ld_avg_aspect_ratio",
    "type_2_ld_avg_aspect_ratio",
    "type_3_ld_avg_aspect_ratio",
    "type_4_ld_avg_aspect_ratio",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def sanitize_filename(text: str, max_length: int = 120) -> str:
    """Create a safe file name from a feature name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return safe[:max_length] if len(safe) > max_length else safe


def detect_cluster_column(df: pd.DataFrame) -> str:
    """Return the cluster column used for grouping."""
    if CLUSTER_COLUMN is not None:
        if CLUSTER_COLUMN not in df.columns:
            raise KeyError(f"CLUSTER_COLUMN={CLUSTER_COLUMN!r} was not found in the input file.")
        return CLUSTER_COLUMN

    for candidate in CLUSTER_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        "Could not find a cluster column. Expected one of: "
        f"{CLUSTER_COLUMN_CANDIDATES}."
    )


def classify_organelle(feature: str) -> str:
    """Classify a feature name as mito, peroxisome, lipid droplet, or other."""
    lower = feature.lower()
    if "mito" in lower:
        return "mito"
    if "peroxisome" in lower:
        return "peroxisome"
    if "ld" in lower:
        return "ld"
    return "other"


def classify_feature_category(feature: str) -> str:
    """Classify a feature into broad biological descriptor categories."""
    lower = feature.lower()
    if any(word in lower for word in ["solidity", "aspect_ratio", "circularity"]):
        return "Morphology"
    if any(word in lower for word in ["area", "perimeter"]):
        return "Size"
    if any(word in lower for word in ["density", "percent"]):
        return "Amount"
    if any(word in lower for word in ["distance", "dist"]):
        return "Position"
    return "Other"


def show_or_close(fig: plt.Figure) -> None:
    """Show or close a figure depending on SHOW_FIGURES."""
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_figure(fig: plt.Figure, output_path_without_ext: Path) -> None:
    """Save PNG and SVG versions of a figure."""
    if not SAVE_FIGURES:
        return

    output_path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_without_ext.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(output_path_without_ext.with_suffix(".svg"), bbox_inches="tight")


def dynamic_height(n_items: int, base: float = 2.0, per_item: float = 0.18, min_height: float = 5.0, max_height: float = 30.0) -> float:
    """Choose a readable figure height based on the number of y-axis labels."""
    return min(max(min_height, base + per_item * max(n_items, 1)), max_height)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to numeric and replace infinite values with NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_and_filter_data(input_csv: Path) -> tuple[pd.DataFrame, str]:
    """Load and apply the same filtering logic as the Levene script."""
    df = pd.read_csv(input_csv, na_values="-")

   
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    cluster_col = detect_cluster_column(df)

    if GROUP_FILTER_VALUE is not None:
        if GROUP_COLUMN not in df.columns:
            raise KeyError(f"GROUP_FILTER_VALUE was set, but {GROUP_COLUMN!r} is not in the input file.")
        before = len(df)
        df = df[df[GROUP_COLUMN] == GROUP_FILTER_VALUE].copy()
        print(f"Group filter {GROUP_COLUMN} == {GROUP_FILTER_VALUE}: kept {len(df)} of {before} rows")

    required = ["mito_aspect_ratio", "mito_density", "area", cluster_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for filtering/statistics: {missing}")

    before = len(df)
    df = df[df["mito_aspect_ratio"] > 0].copy()
    print(f"mito_aspect_ratio > 0: kept {len(df)} of {before} rows")

    mito_count = df["mito_density"] * df["area"]
    before = len(df)
    df = df[mito_count >= MIN_MITO_COUNT].copy()
    print(f"mito_density * area >= {MIN_MITO_COUNT}: kept {len(df)} of {before} rows")

    if MIN_MITO_DENSITY is not None:
        before = len(df)
        df = df[df["mito_density"] >= MIN_MITO_DENSITY].copy()
        print(f"mito_density >= {MIN_MITO_DENSITY}: kept {len(df)} of {before} rows")

    # Ensure cluster labels are clean and usable.
    df[cluster_col] = clean_numeric_series(df[cluster_col])
    df = df.dropna(subset=[cluster_col]).copy()
    df[cluster_col] = df[cluster_col].astype(int)

    if df[cluster_col].nunique() < 2:
        raise ValueError("At least two clusters are required for Levene/ANOVA tests.")

    return df, cluster_col


def build_feature_matrix(df: pd.DataFrame, cluster_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric biological features and aligned cluster labels."""
    clusters = df[cluster_col].copy()

    columns_to_remove = list(BASE_COLUMNS_TO_REMOVE)

    # Remove all PCA components generated by earlier scripts.
    columns_to_remove.extend([col for col in df.columns if str(col).startswith("component_")])

    # Remove GMM probability columns named 0, 1, 2, 3, ...
    columns_to_remove.extend([col for col in df.columns if str(col).isdigit()])

    features = df.drop(columns=columns_to_remove, errors="ignore")
    features = features.select_dtypes(include=[np.number]).copy()

    # Do not allow the cluster column to leak into the feature matrix.
    features = features.drop(columns=[cluster_col], errors="ignore")

    # Remove all-NaN, infinite, and constant columns.
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    features = features.loc[:, features.nunique(dropna=True) > 1]

    # Keep only rows where all selected feature columns are finite.
    # This prevents scipy tests from receiving NaNs.
    valid_rows = features.notna().all(axis=1)
    features = features.loc[valid_rows].copy()
    clusters = clusters.loc[features.index].copy()

    print(f"Rows used for statistics: {len(features)}")
    print(f"Numeric biological features tested: {features.shape[1]}")
    print("Cluster counts used for statistics:")
    print(clusters.value_counts().sort_index())

    return features, clusters


# -----------------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------------
def compute_partial_eta_squared(f_statistic: float, df_between: int, df_within: int) -> float:
    """Partial eta squared from one-way ANOVA F statistic."""
    if not np.isfinite(f_statistic) or f_statistic <= 0:
        return 0.0
    denominator = f_statistic * df_between + df_within
    if denominator == 0:
        return np.nan
    return float((f_statistic * df_between) / denominator)


def get_groups(feature_values: pd.Series, clusters: pd.Series) -> list[np.ndarray]:
    """Split feature values into arrays by sorted cluster label."""
    temp = pd.DataFrame({"value": feature_values, "cluster": clusters}).dropna()
    groups = [
        temp.loc[temp["cluster"] == cluster, "value"].to_numpy()
        for cluster in sorted(temp["cluster"].unique())
    ]
    groups = [group for group in groups if len(group) > 0]
    return groups


def run_levene_and_anova(features: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Run Levene and one-way ANOVA feature by feature."""
    unique_clusters = sorted(clusters.unique())
    n_clusters = len(unique_clusters)
    df_between = n_clusters - 1
    df_within = len(features) - n_clusters

    records: list[dict[str, object]] = []

    for feature in features.columns:
        groups = get_groups(features[feature], clusters)
        if len(groups) < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                levene_stat, levene_p = levene(*groups)
            except ValueError:
                levene_stat, levene_p = np.nan, np.nan

            try:
                anova_result = f_oneway(*groups)
                anova_f = float(anova_result.statistic)
                anova_p = float(anova_result.pvalue)
            except ValueError:
                anova_f, anova_p = np.nan, np.nan

        if not np.isfinite(anova_p):
            continue

        eta_squared = compute_partial_eta_squared(anova_f, df_between, df_within)

        records.append(
            {
                "feature": feature,
                "organelle": classify_organelle(feature),
                "feature_category": classify_feature_category(feature),
                "n_total": int(sum(len(group) for group in groups)),
                "n_groups": int(len(groups)),
                "levene_statistic": levene_stat,
                "levene_p_value": levene_p,
                "levene_violates_homogeneity": bool(np.isfinite(levene_p) and levene_p < ALPHA),
                "anova_F_statistic": anova_f,
                "anova_p_value": anova_p,
                "minus_log10_anova_p": -np.log10(max(anova_p, 1e-300)),
                "partial_eta_squared": eta_squared,
                "significant_anova": bool(anova_p < ALPHA),
            }
        )

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("No usable features were tested. Check filtering and feature columns.")

    return results.sort_values("anova_p_value", ascending=True).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Welch ANOVA
# -----------------------------------------------------------------------------
def welch_anova_one_feature(feature_values: pd.Series, clusters: pd.Series) -> dict[str, object]:
    """
    Manual Welch ANOVA for one feature across clusters.

    Welch ANOVA tests whether group means differ while allowing unequal
    variances across clusters. It is useful when Levene's test suggests that
    the equal-variance assumption of regular ANOVA is violated.
    """
    temp = pd.DataFrame({"value": feature_values, "cluster": clusters}).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    groups: list[np.ndarray] = []
    group_labels: list[object] = []

    for group_label, group_df in temp.groupby("cluster", sort=True):
        values = group_df["value"].to_numpy(dtype=float)
        if len(values) >= 2:
            groups.append(values)
            group_labels.append(group_label)

    k = len(groups)
    n_total = int(sum(len(group) for group in groups))

    if k < 2:
        return {
            "status": "skipped_less_than_2_groups",
            "welch_F_statistic": np.nan,
            "welch_df_between": np.nan,
            "welch_df_within": np.nan,
            "welch_p_value": np.nan,
            "n_groups_welch": k,
            "n_total_welch": n_total,
            "groups_used_welch": ",".join(map(str, group_labels)),
        }

    n = np.array([len(group) for group in groups], dtype=float)
    means = np.array([np.mean(group) for group in groups], dtype=float)
    variances = np.array([np.var(group, ddof=1) for group in groups], dtype=float)

    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        return {
            "status": "skipped_zero_or_invalid_group_variance",
            "welch_F_statistic": np.nan,
            "welch_df_between": np.nan,
            "welch_df_within": np.nan,
            "welch_p_value": np.nan,
            "n_groups_welch": k,
            "n_total_welch": n_total,
            "groups_used_welch": ",".join(map(str, group_labels)),
        }

    weights = n / variances
    weight_sum = weights.sum()
    weighted_mean = np.sum(weights * means) / weight_sum

    df_between = k - 1
    numerator = np.sum(weights * (means - weighted_mean) ** 2) / df_between

    variance_correction_terms = ((1 - weights / weight_sum) ** 2) / (n - 1)
    correction = 1 + ((2 * (k - 2)) / (k**2 - 1)) * np.sum(variance_correction_terms)

    welch_f = numerator / correction
    df_within = (k**2 - 1) / (3 * np.sum(variance_correction_terms))
    welch_p = f_distribution.sf(welch_f, df_between, df_within)

    return {
        "status": "ok",
        "welch_F_statistic": float(welch_f),
        "welch_df_between": float(df_between),
        "welch_df_within": float(df_within),
        "welch_p_value": float(welch_p),
        "n_groups_welch": k,
        "n_total_welch": n_total,
        "groups_used_welch": ",".join(map(str, group_labels)),
    }


def run_welch_anova(features: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Run Welch ANOVA feature by feature."""
    records: list[dict[str, object]] = []

    for feature in features.columns:
        record = welch_anova_one_feature(features[feature], clusters)
        record = {
            "feature": feature,
            "organelle": classify_organelle(feature),
            "feature_category": classify_feature_category(feature),
            **record,
        }

        p_value = record.get("welch_p_value", np.nan)
        if np.isfinite(p_value):
            record["minus_log10_welch_p"] = -np.log10(max(float(p_value), 1e-300))
            record["significant_welch"] = bool(float(p_value) < ALPHA)
        else:
            record["minus_log10_welch_p"] = np.nan
            record["significant_welch"] = False

        records.append(record)

    welch_results = pd.DataFrame(records)
    if welch_results.empty:
        raise ValueError("No usable features were available for Welch ANOVA.")

    return welch_results.sort_values(
        ["status", "welch_p_value"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def plot_welch_pvalues(welch_results: pd.DataFrame, output_dir: Path) -> None:
    """Plot significant Welch ANOVA features by -log10(p-value)."""
    valid = welch_results[welch_results["status"] == "ok"].copy()
    plot_df = valid[valid["significant_welch"]].copy()

    if plot_df.empty:
        print("No significant Welch ANOVA features to plot.")
        return

    plot_df = plot_df.sort_values("minus_log10_welch_p", ascending=False)
    if MAX_FEATURES_IN_WELCH_PLOT is not None:
        plot_df = plot_df.head(MAX_FEATURES_IN_WELCH_PLOT)

    height = dynamic_height(len(plot_df), per_item=0.20, max_height=32)
    fig, ax = plt.subplots(figsize=(10, height), dpi=FIGURE_DPI)

    sns.barplot(
        data=plot_df,
        x="minus_log10_welch_p",
        y="feature",
        palette="viridis",
        ax=ax,
    )

    ax.axvline(-np.log10(ALPHA), color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("-log10(Welch ANOVA p-value)", fontsize=FONT_AXIS)
    ax.set_ylabel("Features", fontsize=FONT_AXIS)
    ax.set_title("Significant Features Sorted by Welch ANOVA p-values", fontsize=FONT_TITLE)
    ax.tick_params(axis="x", labelsize=FONT_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Put the threshold explanation outside the bars rather than using an internal legend.
    ax.text(
        1.01,
        1.0,
        f"dashed line:\np = {ALPHA}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_LEGEND,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85, "linewidth": 0.5},
    )
    fig.subplots_adjust(right=0.82)

    save_figure(fig, output_dir / "welch_anova_significant_features_minus_log10_p")
    show_or_close(fig)


def save_welch_outputs(welch_results: pd.DataFrame, anova_results: pd.DataFrame, output_dir: Path) -> None:
    """Save Welch ANOVA tables and a combined ANOVA/Welch comparison table."""
    welch_dir = output_dir / "Welch_ANOVA_results"
    welch_dir.mkdir(parents=True, exist_ok=True)

    valid = welch_results[welch_results["status"] == "ok"].copy()
    skipped = welch_results[welch_results["status"] != "ok"].copy()
    significant = valid[valid["significant_welch"]].copy()

    welch_results.to_csv(welch_dir / "welch_anova_all_features.csv", index=False)
    valid.to_csv(welch_dir / "welch_anova_valid_features.csv", index=False)
    significant.to_csv(welch_dir / "welch_anova_significant_features.csv", index=False)
    skipped.to_csv(welch_dir / "welch_anova_skipped_features.csv", index=False)

    comparison_cols = [
        "feature",
        "levene_p_value",
        "levene_violates_homogeneity",
        "anova_F_statistic",
        "anova_p_value",
        "minus_log10_anova_p",
        "partial_eta_squared",
        "significant_anova",
    ]
    combined = anova_results[comparison_cols].merge(
        welch_results[
            [
                "feature",
                "status",
                "welch_F_statistic",
                "welch_df_between",
                "welch_df_within",
                "welch_p_value",
                "minus_log10_welch_p",
                "significant_welch",
            ]
        ],
        on="feature",
        how="left",
    )
    combined.to_csv(welch_dir / "anova_levene_welch_comparison.csv", index=False)

    print("\nWelch ANOVA summary")
    print("-------------------")
    print(f"Welch valid features: {len(valid)}")
    print(f"Welch skipped features: {len(skipped)}")
    print(f"Welch significant features p < {ALPHA}: {len(significant)}")
    print(f"Welch outputs saved to: {welch_dir}")



# -----------------------------------------------------------------------------
# Plotting summary bar plots
# -----------------------------------------------------------------------------
def plot_anova_pvalues(results: pd.DataFrame, output_dir: Path) -> None:
    """Plot significant ANOVA features by -log10(p-value), preserving viridis bar style."""
    plot_df = results[results["significant_anova"]].copy()
    if plot_df.empty:
        print("No significant ANOVA features to plot.")
        return

    plot_df = plot_df.sort_values("minus_log10_anova_p", ascending=False)
    if MAX_FEATURES_IN_SUMMARY_PLOTS is not None:
        plot_df = plot_df.head(MAX_FEATURES_IN_SUMMARY_PLOTS)

    height = dynamic_height(len(plot_df), per_item=0.20, max_height=32)
    fig, ax = plt.subplots(figsize=(10, height), dpi=FIGURE_DPI)

    sns.barplot(
        data=plot_df,
        x="minus_log10_anova_p",
        y="feature",
        palette="viridis",
        ax=ax,
    )

    ax.axvline(-np.log10(ALPHA), color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("-log10(ANOVA p-value)", fontsize=FONT_AXIS)
    ax.set_ylabel("Features", fontsize=FONT_AXIS)
    ax.set_title("Significant Features Sorted by ANOVA p-values", fontsize=FONT_TITLE)
    ax.tick_params(axis="x", labelsize=FONT_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    fig.tight_layout()
    save_figure(fig, output_dir / "anova_significant_features_minus_log10_p")
    show_or_close(fig)


def plot_eta_squared(results: pd.DataFrame, output_dir: Path) -> None:
    """Plot significant features by partial eta squared with organelle colors."""
    plot_df = results[results["significant_anova"]].copy()
    if plot_df.empty:
        print("No significant ANOVA features for eta-squared plot.")
        return

    plot_df = plot_df.sort_values("partial_eta_squared", ascending=False)
    if MAX_FEATURES_IN_SUMMARY_PLOTS is not None:
        plot_df = plot_df.head(MAX_FEATURES_IN_SUMMARY_PLOTS)

    colors = [ORGANELLE_COLORS.get(org, "gray") for org in plot_df["organelle"]]
    height = dynamic_height(len(plot_df), per_item=0.22, max_height=34)

    fig, ax = plt.subplots(figsize=(10, height), dpi=FIGURE_DPI)
    sns.barplot(
        data=plot_df,
        x="partial_eta_squared",
        y="feature",
        palette=colors,
        ax=ax,
        alpha=0.5,
    )

    ax.set_xlabel("Partial Eta Squared (η²)", fontsize=FONT_AXIS)
    ax.set_ylabel("Features", fontsize=FONT_AXIS)
    ax.set_title("Significant Features Sorted by Effect Size (Partial Eta Squared)", fontsize=FONT_TITLE)
    ax.tick_params(axis="x", labelsize=FONT_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Legend outside the plotting area so it does not cover bars.
    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, color=color, alpha=0.5)
        for color in ORGANELLE_COLORS.values()
    ]
    legend_labels = [ORGANELLE_DISPLAY_NAMES[key] for key in ORGANELLE_COLORS.keys()]
    ax.legend(
        legend_handles,
        legend_labels,
        title="Feature group",
        fontsize=FONT_LEGEND,
        title_fontsize=FONT_LEGEND,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        borderaxespad=0.0,
    )

    # Reserve right side for the external legend.
    fig.subplots_adjust(right=0.78)
    save_figure(fig, output_dir / "anova_significant_features_partial_eta_squared")
    show_or_close(fig)


def plot_levene_summary(results: pd.DataFrame, output_dir: Path) -> None:
    """Plot counts of features that violate/do not violate Levene assumption."""
    total = len(results)
    violating = int(results["levene_violates_homogeneity"].sum())
    non_violating = total - violating

    summary = pd.DataFrame(
        {
            "Levene_result": ["Violates\np < 0.05", "Does not violate\np ≥ 0.05"],
            "count": [violating, non_violating],
            "percent": [100 * violating / total, 100 * non_violating / total],
        }
    )
    summary.to_csv(output_dir / "levene_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=FIGURE_DPI)
    sns.barplot(data=summary, x="Levene_result", y="percent", palette=["firebrick", "gray"], ax=ax, alpha=0.75)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Features (%)", fontsize=FONT_AXIS)
    ax.set_xlabel("", fontsize=FONT_AXIS)
    ax.set_title("Levene Homogeneity-of-Variance Summary", fontsize=FONT_TITLE)
    ax.tick_params(axis="x", labelsize=FONT_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TICK_X)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, row in summary.iterrows():
        ax.text(i, row["percent"] + 2, f"{row['count']}\n({row['percent']:.1f}%)", ha="center", va="bottom", fontsize=FONT_TICK_X)

    fig.tight_layout()
    save_figure(fig, output_dir / "levene_homogeneity_summary")
    show_or_close(fig)


# -----------------------------------------------------------------------------
# Tukey HSD analysis and plotting
# -----------------------------------------------------------------------------
def tukey_to_dataframe(tukey_result) -> pd.DataFrame:
    """Convert statsmodels Tukey HSD summary to a DataFrame."""
    summary = tukey_result.summary()
    tukey_df = pd.DataFrame(np.array(summary.data[1:], dtype=object), columns=summary.data[0])

    numeric_cols = ["meandiff", "p-adj", "lower", "upper"]
    for col in numeric_cols:
        tukey_df[col] = pd.to_numeric(tukey_df[col], errors="coerce")

    tukey_df["group1"] = tukey_df["group1"].astype(str)
    tukey_df["group2"] = tukey_df["group2"].astype(str)
    tukey_df["Comparison"] = tukey_df["group1"] + " vs " + tukey_df["group2"]

    return tukey_df


def add_percentage_mean_difference(tukey_df: pd.DataFrame, feature: str, data: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Add percentage mean differences using actual group means."""
    result = tukey_df.copy()

    group_means = data.groupby(cluster_col)[feature].mean()
    group_means.index = group_means.index.astype(str)

    result["group1_mean"] = result["group1"].map(group_means)
    result["group2_mean"] = result["group2"].map(group_means)
    denominator = (result["group1_mean"] + result["group2_mean"]) / 2

    # Avoid division by zero for features where both group means are zero.
    denominator = denominator.replace(0, np.nan)

    result["percentage_mean_diff"] = (result["meandiff"] / denominator) * 100
    result["lower_percentage"] = (result["lower"] / denominator) * 100
    result["upper_percentage"] = (result["upper"] / denominator) * 100
    result["lower_error"] = np.abs(result["percentage_mean_diff"] - result["lower_percentage"])
    result["upper_error"] = np.abs(result["upper_percentage"] - result["percentage_mean_diff"])

    return result


def select_tukey_features(results: pd.DataFrame) -> dict[str, list[str]]:
    """Select top significant features per organelle for Tukey plots."""
    significant = results[results["significant_anova"]].copy()
    selected: dict[str, list[str]] = {}

    for organelle in ["mito", "peroxisome", "ld"]:
        organelle_features = significant[significant["organelle"] == organelle]
        organelle_features = organelle_features.sort_values("partial_eta_squared", ascending=False)
        selected[organelle] = organelle_features["feature"].head(TOP_N_TUKEY_FEATURES_PER_ORGANELLE).tolist()

    return selected


def plot_tukey_percentage_bar(tukey_df: pd.DataFrame, feature: str, organelle: str, output_dir: Path) -> None:
    """Plot Tukey HSD percentage mean differences with p-values outside the bar plot."""
    plot_df = tukey_df.dropna(
        subset=["percentage_mean_diff", "lower_error", "upper_error"]
    ).copy()

    if plot_df.empty:
        print(f"Skipping Tukey plot for {feature}: no finite percentage mean differences.")
        return

    n_comparisons = len(plot_df)
    fig_width = max(8.0, min(18.0, 7.0 + 0.18 * n_comparisons))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=FIGURE_DPI)

    sns.barplot(
        data=plot_df,
        x="Comparison",
        y="percentage_mean_diff",
        palette="BuPu",
        ax=ax,
        ci=None,
    )

    for i, row in plot_df.reset_index(drop=True).iterrows():
        ax.errorbar(
            i,
            row["percentage_mean_diff"],
            yerr=[[row["lower_error"]], [row["upper_error"]]],
            fmt="none",
            color="black",
            capsize=3,
            elinewidth=1.0,
        )

    ax.axhline(0, color="black", linewidth=0.8)

    y_max = max(
        (plot_df["percentage_mean_diff"] + plot_df["upper_error"]).max(),
        plot_df["percentage_mean_diff"].max(),
    )
    y_min = min(
        (plot_df["percentage_mean_diff"] - plot_df["lower_error"]).min(),
        plot_df["percentage_mean_diff"].min(),
    )

    if np.isfinite(y_max) and np.isfinite(y_min):
        if y_max == y_min:
            margin = max(abs(y_max) * 0.1, 1.0)
        else:
            margin = 0.10 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_title(f"Tukey HSD percentage mean difference: {feature}", fontsize=FONT_TITLE)
    ax.set_xlabel("Group comparisons", fontsize=FONT_AXIS)
    ax.set_ylabel("Percentage mean difference", fontsize=FONT_AXIS)
    ax.tick_params(axis="x", labelrotation=90, labelsize=FONT_TUKEY_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TUKEY_TICK_Y)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Place p-values outside the axes, not over the bars.
    p_values_text = "\n".join(
        f"{row['Comparison']}: p={row['p-adj']:.4g}"
        for _, row in plot_df.iterrows()
    )

    ax.text(
        1.02,
        1.0,
        p_values_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_TUKEY_PVALUES,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85, "linewidth": 0.5},
    )

    # Reserve right side for the p-value text box and bottom for rotated labels.
    fig.subplots_adjust(right=0.70, bottom=0.35)

    filename = sanitize_filename(f"{organelle}_{feature}_tukey_percent_mean_diff")
    save_figure(fig, output_dir / "Tukey_HSD_percent_mean_diff" / organelle / filename)
    show_or_close(fig)


def run_tukey_for_selected_features(
    data_for_tukey: pd.DataFrame,
    results: pd.DataFrame,
    features: pd.DataFrame,
    clusters: pd.Series,
    cluster_col: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Run Tukey HSD for selected top features and save/plot results."""
    if not MAKE_TUKEY_PLOTS:
        print("MAKE_TUKEY_PLOTS is False; skipping Tukey plots.")
        return pd.DataFrame()

    selected_by_organelle = select_tukey_features(results)
    records: list[pd.DataFrame] = []

    # Use one dataframe with cluster labels and feature columns for group means.
    tukey_data = features.copy()
    tukey_data[cluster_col] = clusters.astype(str)

    for organelle, selected_features in selected_by_organelle.items():
        if not selected_features:
            print(f"No selected {organelle} features for Tukey plots.")
            continue

        for feature in selected_features:
            print(f"Running Tukey HSD for {feature}...")

            temp = tukey_data[[feature, cluster_col]].dropna().copy()
            if temp[cluster_col].nunique() < 2:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                try:
                    tukey = pairwise_tukeyhsd(
                        endog=temp[feature],
                        groups=temp[cluster_col],
                        alpha=ALPHA,
                    )
                except Exception as exc:  # statsmodels can fail on degenerate features
                    print(f"Skipping Tukey for {feature}: {exc}")
                    continue

            tukey_df = tukey_to_dataframe(tukey)
            tukey_df = add_percentage_mean_difference(tukey_df, feature, temp, cluster_col)
            tukey_df.insert(0, "feature", feature)
            tukey_df.insert(1, "organelle", organelle)
            records.append(tukey_df)

            tukey_csv_dir = output_dir / "Tukey_HSD_tables" / organelle
            tukey_csv_dir.mkdir(parents=True, exist_ok=True)
            tukey_df.to_csv(tukey_csv_dir / f"{sanitize_filename(feature)}_tukey.csv", index=False)

            plot_tukey_percentage_bar(tukey_df, feature, organelle, output_dir)

    if records:
        all_tukey = pd.concat(records, axis="rows", ignore_index=True)
        all_tukey.to_csv(output_dir / "tukey_hsd_selected_features_all.csv", index=False)
        return all_tukey

    return pd.DataFrame()




# -----------------------------------------------------------------------------
# Games-Howell posthoc analysis and plotting
# -----------------------------------------------------------------------------
def games_howell_for_feature(feature_values: pd.Series, clusters: pd.Series, feature: str, organelle: str) -> pd.DataFrame:
    """
    Run Games-Howell pairwise comparisons for one feature.

    Games-Howell is the pairwise follow-up that matches the Welch ANOVA logic:
    it allows unequal variances and unequal group sizes.
    """
    temp = pd.DataFrame({"value": feature_values, "cluster": clusters}).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    grouped = []
    for group_label, group_df in temp.groupby("cluster", sort=True):
        values = group_df["value"].to_numpy(dtype=float)
        if len(values) >= 2:
            grouped.append(
                {
                    "label": group_label,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "variance": float(np.var(values, ddof=1)),
                }
            )

    k = len(grouped)
    if k < 2:
        return pd.DataFrame()

    records: list[dict[str, object]] = []

    for i in range(k - 1):
        for j in range(i + 1, k):
            g1 = grouped[i]
            g2 = grouped[j]

            var1 = g1["variance"]
            var2 = g2["variance"]
            n1 = g1["n"]
            n2 = g2["n"]

            if not np.isfinite(var1) or not np.isfinite(var2) or var1 <= 0 or var2 <= 0:
                continue

            se = np.sqrt((var1 / n1) + (var2 / n2))
            if not np.isfinite(se) or se == 0:
                continue

            mean_diff = g2["mean"] - g1["mean"]
            t_stat = abs(mean_diff) / se

            numerator_df = ((var1 / n1) + (var2 / n2)) ** 2
            denominator_df = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
            df = numerator_df / denominator_df if denominator_df != 0 else np.nan

            if not np.isfinite(df) or df <= 0:
                continue

            q_stat = np.sqrt(2) * t_stat
            p_adj = float(studentized_range.sf(q_stat, k, df))

            q_crit = float(studentized_range.ppf(1 - ALPHA, k, df))
            ci_half_width = q_crit * se / np.sqrt(2)
            lower = mean_diff - ci_half_width
            upper = mean_diff + ci_half_width

            denominator = (g1["mean"] + g2["mean"]) / 2
            if abs(denominator) < 1e-12:
                percentage_mean_diff = np.nan
                lower_percentage = np.nan
                upper_percentage = np.nan
                lower_error = np.nan
                upper_error = np.nan
            else:
                percentage_mean_diff = (mean_diff / denominator) * 100
                lower_percentage = (lower / denominator) * 100
                upper_percentage = (upper / denominator) * 100
                lower_error = abs(percentage_mean_diff - lower_percentage)
                upper_error = abs(upper_percentage - percentage_mean_diff)

            group1 = str(g1["label"])
            group2 = str(g2["label"])

            records.append(
                {
                    "feature": feature,
                    "organelle": organelle,
                    "group1": group1,
                    "group2": group2,
                    "Comparison": f"{group1} vs {group2}",
                    "group1_n": n1,
                    "group2_n": n2,
                    "group1_mean": g1["mean"],
                    "group2_mean": g2["mean"],
                    "mean_diff": mean_diff,
                    "standard_error": se,
                    "df": df,
                    "q_statistic": q_stat,
                    "p_adj": p_adj,
                    "lower": lower,
                    "upper": upper,
                    "reject": bool(p_adj < ALPHA),
                    "percentage_mean_diff": percentage_mean_diff,
                    "lower_percentage": lower_percentage,
                    "upper_percentage": upper_percentage,
                    "lower_error": lower_error,
                    "upper_error": upper_error,
                }
            )

    return pd.DataFrame(records)


def select_games_howell_features(welch_results: pd.DataFrame) -> dict[str, list[str]]:
    """Select top Welch-significant features per organelle for Games-Howell plots."""
    valid = welch_results[
        (welch_results["status"] == "ok")
        & (welch_results["significant_welch"])
    ].copy()

    selected: dict[str, list[str]] = {}

    for organelle in ["mito", "peroxisome", "ld"]:
        organelle_features = valid[valid["organelle"] == organelle]
        organelle_features = organelle_features.sort_values("welch_p_value", ascending=True)
        selected[organelle] = organelle_features["feature"].head(
            TOP_N_GAMES_HOWELL_FEATURES_PER_ORGANELLE
        ).tolist()

    return selected


def plot_games_howell_percentage_bar(games_df: pd.DataFrame, feature: str, organelle: str, output_dir: Path) -> None:
    """Plot Games-Howell percentage mean differences with p-values outside the bar plot."""
    plot_df = games_df.dropna(
        subset=["percentage_mean_diff", "lower_error", "upper_error"]
    ).copy()

    if plot_df.empty:
        print(f"Skipping Games-Howell plot for {feature}: no finite percentage mean differences.")
        return

    n_comparisons = len(plot_df)
    fig_width = max(8.0, min(18.0, 7.0 + 0.18 * n_comparisons))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=FIGURE_DPI)

    sns.barplot(
        data=plot_df,
        x="Comparison",
        y="percentage_mean_diff",
        palette="BuPu",
        ax=ax,
        ci=None,
    )

    for i, row in plot_df.reset_index(drop=True).iterrows():
        ax.errorbar(
            i,
            row["percentage_mean_diff"],
            yerr=[[row["lower_error"]], [row["upper_error"]]],
            fmt="none",
            color="black",
            capsize=3,
            elinewidth=1.0,
        )

    ax.axhline(0, color="black", linewidth=0.8)

    y_max = max(
        (plot_df["percentage_mean_diff"] + plot_df["upper_error"]).max(),
        plot_df["percentage_mean_diff"].max(),
    )
    y_min = min(
        (plot_df["percentage_mean_diff"] - plot_df["lower_error"]).min(),
        plot_df["percentage_mean_diff"].min(),
    )

    if np.isfinite(y_max) and np.isfinite(y_min):
        if y_max == y_min:
            margin = max(abs(y_max) * 0.1, 1.0)
        else:
            margin = 0.10 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_title(f"Games-Howell percentage mean difference: {feature}", fontsize=FONT_TITLE)
    ax.set_xlabel("Group comparisons", fontsize=FONT_AXIS)
    ax.set_ylabel("Percentage mean difference", fontsize=FONT_AXIS)
    ax.tick_params(axis="x", labelrotation=90, labelsize=FONT_TUKEY_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_TUKEY_TICK_Y)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    p_values_text = "\n".join(
        f"{row['Comparison']}: p={row['p_adj']:.4g}"
        for _, row in plot_df.iterrows()
    )

    ax.text(
        1.02,
        1.0,
        p_values_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_TUKEY_PVALUES,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85, "linewidth": 0.5},
    )

    fig.subplots_adjust(right=0.70, bottom=0.35)

    filename = sanitize_filename(f"{organelle}_{feature}_games_howell_percent_mean_diff")
    save_figure(fig, output_dir / "Games_Howell_percent_mean_diff" / organelle / filename)
    show_or_close(fig)


def run_games_howell_for_selected_features(
    welch_results: pd.DataFrame,
    features: pd.DataFrame,
    clusters: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    """Run Games-Howell pairwise comparisons for selected Welch-significant features."""
    if not MAKE_GAMES_HOWELL_PLOTS:
        print("MAKE_GAMES_HOWELL_PLOTS is False; skipping Games-Howell plots.")
        return pd.DataFrame()

    selected_by_organelle = select_games_howell_features(welch_results)
    records: list[pd.DataFrame] = []

    for organelle, selected_features in selected_by_organelle.items():
        if not selected_features:
            print(f"No selected {organelle} features for Games-Howell plots.")
            continue

        for feature in selected_features:
            print(f"Running Games-Howell for {feature}...")
            try:
                games_df = games_howell_for_feature(
                    feature_values=features[feature],
                    clusters=clusters,
                    feature=feature,
                    organelle=organelle,
                )
            except Exception as exc:
                print(f"Skipping Games-Howell for {feature}: {exc}")
                continue

            if games_df.empty:
                print(f"No Games-Howell results for {feature}.")
                continue

            records.append(games_df)

            games_csv_dir = output_dir / "Games_Howell_tables" / organelle
            games_csv_dir.mkdir(parents=True, exist_ok=True)
            games_df.to_csv(games_csv_dir / f"{sanitize_filename(feature)}_games_howell.csv", index=False)

            plot_games_howell_percentage_bar(games_df, feature, organelle, output_dir)

    if records:
        all_games = pd.concat(records, axis="rows", ignore_index=True)
        all_games.to_csv(output_dir / "games_howell_selected_features_all.csv", index=False)
        return all_games

    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Category summary plots
# -----------------------------------------------------------------------------
def plot_category_summary(results: pd.DataFrame, output_dir: Path) -> None:
    """Make compact category summary charts similar to the original Levene script."""
    if not MAKE_CATEGORY_SUMMARY_CHARTS:
        return

    significant = results[results["significant_anova"]].copy()
    if significant.empty:
        return

    top = significant.sort_values("partial_eta_squared", ascending=False).head(TOP_N_FEATURES_FOR_CATEGORY_SUMMARY)

    # Donut charts: for each feature category, show organelle composition.
    categories = ["Morphology", "Size", "Amount", "Position"]
    organelles = ["mito", "peroxisome", "ld", "other"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), dpi=FIGURE_DPI)
    for ax, category in zip(axes, categories):
        subset = top[top["feature_category"] == category]
        counts = subset["organelle"].value_counts().reindex(organelles, fill_value=0)
        values = counts.to_numpy()

        if values.sum() == 0:
            ax.axis("off")
            ax.set_title(f"{category}\n(no features)", fontsize=FONT_TITLE)
            continue

        ax.pie(
            values,
            labels=[ORGANELLE_DISPLAY_NAMES[o] for o in organelles],
            autopct="%1.1f%%",
            colors=[ORGANELLE_COLORS[o] for o in organelles],
            startangle=140,
            textprops={"fontsize": 5},
            wedgeprops={"width": 0.25, "edgecolor": "white", "alpha": 0.5},
        )
        ax.set_title(f"{category} Features", fontsize=FONT_TITLE)

    fig.tight_layout()
    save_figure(fig, output_dir / "category_organelle_donut_summary")
    show_or_close(fig)

    # Overall feature category distribution.
    category_counts = top["feature_category"].value_counts().reindex(categories + ["Other"], fill_value=0)
    category_counts = category_counts[category_counts > 0]
    if not category_counts.empty:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=FIGURE_DPI)
        ax.pie(
            category_counts.values,
            labels=category_counts.index,
            autopct="%1.1f%%",
            colors=["blue", "red", "purple", "orange", "gray"][: len(category_counts)],
            startangle=140,
            textprops={"fontsize": 6},
            wedgeprops={"width": 0.25, "edgecolor": "white", "alpha": 0.5},
        )
        ax.set_title("Feature Distribution Across Categories", fontsize=FONT_TITLE)
        fig.tight_layout()
        save_figure(fig, output_dir / "feature_category_distribution_donut")
        show_or_close(fig)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filtered_df, cluster_col = load_and_filter_data(INPUT_CSV)
    features, clusters = build_feature_matrix(filtered_df, cluster_col)

    # Save the exact data used for the tests.
    filtered_df.to_csv(OUTPUT_DIR / "filtered_data_used_for_anova.csv", index=False)
    features.to_csv(OUTPUT_DIR / "feature_matrix_used_for_anova.csv", index=False)

    results = run_levene_and_anova(features, clusters)
    results.to_csv(OUTPUT_DIR / "anova_levene_results.csv", index=False)

    significant = results[results["significant_anova"]].copy()
    significant.to_csv(OUTPUT_DIR / "anova_significant_features.csv", index=False)

    cluster_feature_means = features.copy()
    cluster_feature_means[cluster_col] = clusters
    cluster_feature_means.groupby(cluster_col).mean(numeric_only=True).to_csv(OUTPUT_DIR / "feature_means_by_cluster.csv")
    cluster_feature_means.groupby(cluster_col).std(numeric_only=True).to_csv(OUTPUT_DIR / "feature_stds_by_cluster.csv")

    n_total = len(results)
    n_violating = int(results["levene_violates_homogeneity"].sum())
    n_significant = int(results["significant_anova"].sum())

    print("\nSummary")
    print("-------")
    print(f"Cluster column: {cluster_col}")
    print(f"Total features tested: {n_total}")
    print(f"Features violating Levene homogeneity p < {ALPHA}: {n_violating} ({100 * n_violating / n_total:.2f}%)")
    print(f"ANOVA significant features p < {ALPHA}: {n_significant} ({100 * n_significant / n_total:.2f}%)")
    print(f"Results saved to: {OUTPUT_DIR}")

    if MAKE_WELCH_ANOVA:
        welch_results = run_welch_anova(features, clusters)
        save_welch_outputs(welch_results, results, OUTPUT_DIR)
        plot_welch_pvalues(welch_results, OUTPUT_DIR / "Welch_ANOVA_results")
        run_games_howell_for_selected_features(
            welch_results=welch_results,
            features=features,
            clusters=clusters,
            output_dir=OUTPUT_DIR / "Welch_ANOVA_results",
        )

    plot_levene_summary(results, OUTPUT_DIR)
    plot_anova_pvalues(results, OUTPUT_DIR)
    plot_eta_squared(results, OUTPUT_DIR)
    plot_category_summary(results, OUTPUT_DIR)

    run_tukey_for_selected_features(
        data_for_tukey=filtered_df,
        results=results,
        features=features,
        clusters=clusters,
        cluster_col=cluster_col,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
