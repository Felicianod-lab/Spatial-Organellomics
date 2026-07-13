"""
Clean, standalone analysis for the figure:

    Combined (Control + Fasted): PV vs CV
    x-axis: Global Organelle Score
    y-axis: TMRM

The global score construction and the appearance of the original overlay plot are
kept identical to the source script. Additional outputs are included for:

1. Five-fold cross-validated regional calibration regressions.
2. The PV-versus-CV region interaction model plotted directly.
3. The liver-level random-intercept mixed-effects model plotted directly.

The main figure uses the same data processing, score formula, model parameters,
plot dimensions, colors, line styles, point styles, confidence-band settings,
font sizes, axis limits, and title as the original code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# USER SETTINGS
# =============================================================================

CONTROL_MITO_CSV = Path(
    r"...add path...\Fig5_IVM_Control_Group_Mitochondria_Properties.csv"
)
CONTROL_LD_CSV = Path(
    r"...add path...\Fig5_IVM_Control_Group_Lipid_Droplets_Properties.csv"
)    
FASTED_MITO_CSV = Path(
    r"...add path...\Fig5_IVM_Fasted_Group_Mitochondria_Properties.csv"
   
)
FASTED_LD_CSV = Path(
    r"...add path...\Fig5_IVM_Fasted_Group_Lipid_Droplets_Properties.csv"
)

OUTPUT_DIR = Path("Global_organelle_score_outputs")
SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_DPI = 300

# These are the exact cross-validation settings in the original script.
CV_N_SPLITS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 42

# The original mixed-model call used statsmodels' default REML fit.
MIXED_MODEL_REML = True

# In the mixed-model plots, thin conditional lines show fitted random intercepts
# for individual livers. The thick lines and red bands are population-average
# fixed-effect estimates.
SHOW_LIVER_SPECIFIC_MIXED_LINES = True
LIVER_LINE_ALPHA = 0.12
LIVER_LINE_WIDTH = 0.6


# =============================================================================
# EXACT STYLE SETTINGS FROM THE ORIGINAL PV-VERSUS-CV OVERLAY
# =============================================================================

@dataclass(frozen=True)
class OverlayStyle:
    # Figure
    figure_size: tuple[float, float] = (4.8, 5.7)

    # Scatter
    pv_scatter_style: str = "outline"
    cv_scatter_style: str = "outline"
    pv_dot_edge_color: str = "grey"
    cv_dot_edge_color: str = "lightgrey"#"#ADD8E6"
    pv_dot_size: float = 25
    cv_dot_size: float = 25
    pv_dot_alpha: float = 0.4
    cv_dot_alpha: float = 0.4
    pv_dot_edge_width: float = 0.4
    cv_dot_edge_width: float = 0.4

    # Regression lines
    pv_line_color: str = "black"
    cv_line_color: str = "black"
    pv_line_width: float = 1.8
    cv_line_width: float = 1.8
    pv_line_style: str = ":"
    cv_line_style: str = "-"

    # Confidence bands
    show_confidence_band: bool = True
    pv_confidence_color: str = "red"
    cv_confidence_color: str = "red"
    confidence_alpha: float = 0.3

    # Prediction intervals
    show_prediction_band: bool = False
    prediction_line_style: str = ":"
    prediction_line_width: float = 0.8
    pv_prediction_color: str = "black"
    cv_prediction_color: str = "red"

    # Statistics text
    show_pearson: bool = True
    show_spearman: bool = True
    show_r2: bool = True
    stats_fontsize: float = 10

    # Labels
    title_fontsize: float = 14
    label_fontsize: float = 12
    tick_fontsize: float = 18

    # Axes
    x_padding_fraction: float = 0.05
    y_limits: tuple[float, float] = (-0.18, 1.1)


STYLE = OverlayStyle()

MORPH_FEATURES = [
    "perimeter",
    "solidity",
    "aspect_ratio",
    "circularity",
    "boundry_dist",
    "area",
]


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class GlobalScoreArtifacts:
    mito_features: list[str]
    ld_features: list[str]
    scaler_mito: StandardScaler
    scaler_ld: StandardScaler
    mito_correlations: np.ndarray
    ld_correlations: np.ndarray
    mito_signs: np.ndarray
    ld_signs: np.ndarray
    mito_weights: np.ndarray
    ld_weights: np.ndarray
    beta_mito: float
    beta_ld: float
    global_regression: LinearRegression
    feature_weight_table: pd.DataFrame


@dataclass
class CrossValidationResults:
    predictions: pd.DataFrame
    fold_models: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finish_figure(fig: plt.Figure, output_path: Path | None) -> None:
    fig.tight_layout()
    if SAVE_FIGURES and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIGURE_DPI)
    if SHOW_FIGURES:
        plt.show()
    # plt.close(fig)


def calculate_x_limits(scores: pd.Series) -> tuple[float, float]:
    """Reproduce the original 5% global x-axis padding exactly."""
    x_min = float(scores.min())
    x_max = float(scores.max())
    x_pad = STYLE.x_padding_fraction * (x_max - x_min)
    return x_min - x_pad, x_max + x_pad


def region_style(region: str) -> dict[str, object]:
    if region == "PV":
        return {
            "scatter_style": STYLE.pv_scatter_style,
            "dot_edge_color": STYLE.pv_dot_edge_color,
            "dot_size": STYLE.pv_dot_size,
            "dot_alpha": STYLE.pv_dot_alpha,
            "dot_edge_width": STYLE.pv_dot_edge_width,
            "line_color": STYLE.pv_line_color,
            "line_style": STYLE.pv_line_style,
            "line_width": STYLE.pv_line_width,
            "confidence_color": STYLE.pv_confidence_color,
            "prediction_color": STYLE.pv_prediction_color,
            "text_y_offset": 0.95,
        }
    if region == "CV":
        return {
            "scatter_style": STYLE.cv_scatter_style,
            "dot_edge_color": STYLE.cv_dot_edge_color,
            "dot_size": STYLE.cv_dot_size,
            "dot_alpha": STYLE.cv_dot_alpha,
            "dot_edge_width": STYLE.cv_dot_edge_width,
            "line_color": STYLE.cv_line_color,
            "line_style": STYLE.cv_line_style,
            "line_width": STYLE.cv_line_width,
            "confidence_color": STYLE.cv_confidence_color,
            "prediction_color": STYLE.cv_prediction_color,
            "text_y_offset": 0.80,
        }
    raise ValueError(f"Unknown region: {region!r}")


def draw_region_scatter(df: pd.DataFrame, region: str) -> None:
    style = region_style(region)
    x = df["OrgScore"].to_numpy()
    y = df["total_TMRM"].to_numpy()

    if style["scatter_style"] == "filled":
        plt.scatter(
            x,
            y,
            color=style["dot_edge_color"],
            alpha=style["dot_alpha"],
            s=style["dot_size"],
        )
    else:
        plt.scatter(
            x,
            y,
            facecolors="none",
            edgecolors=style["dot_edge_color"],
            linewidth=style["dot_edge_width"],
            s=style["dot_size"],
            alpha=style["dot_alpha"],
        )


def apply_original_axes(title: str, x_limits: tuple[float, float]) -> None:
    plt.title(title, fontsize=STYLE.title_fontsize)
    plt.xlabel("Global Organelle Score", fontsize=STYLE.label_fontsize)
    plt.ylabel("TMRM", fontsize=STYLE.label_fontsize)
    plt.tick_params(axis="both", labelsize=STYLE.tick_fontsize)
    plt.xlim(x_limits)
    plt.ylim(STYLE.y_limits)


# =============================================================================
# LOAD AND AGGREGATE DATA - SAME OPERATIONS AS THE ORIGINAL SCRIPT
# =============================================================================

def load_and_aggregate_cell_data() -> pd.DataFrame:
    mito_c = pd.read_csv(CONTROL_MITO_CSV, low_memory=False)
    ld_c = pd.read_csv(CONTROL_LD_CSV, low_memory=False)
    mito_f = pd.read_csv(FASTED_MITO_CSV, low_memory=False)
    ld_f = pd.read_csv(FASTED_LD_CSV, low_memory=False)

    mito_required = [
        "cell_label",
        "ratio_01_00",
        "area_cell",
        "PV_CV",
        *MORPH_FEATURES,
    ]
    ld_required = ["cell_label", *MORPH_FEATURES]
    require_columns(mito_c, mito_required, "Control mitochondrial table")
    require_columns(mito_f, mito_required, "Fasted mitochondrial table")
    require_columns(ld_c, ld_required, "Control lipid-droplet table")
    require_columns(ld_f, ld_required, "Fasted lipid-droplet table")

    mito_c["Condition"] = "Control"
    mito_f["Condition"] = "Fasted"

    # Keep the original concat behavior; row order and grouping match the source.
    mito = pd.concat([mito_c, mito_f])
    ld = pd.concat([ld_c, ld_f])

    mito["liver_id"] = mito["cell_label"].astype(str).str.split("TS").str[0]

    # This validation does not change valid data. It catches a failure mode in
    # which identical cell labels occur in both conditions and would otherwise
    # be combined by the original groupby.
    condition_counts = mito.groupby("cell_label")["Condition"].nunique()
    duplicated_between_conditions = condition_counts[condition_counts > 1]
    if not duplicated_between_conditions.empty:
        example_labels = duplicated_between_conditions.index.astype(str).tolist()[:10]
        raise ValueError(
            "Some cell_label values occur in more than one condition. The original "
            "groupby would combine those cells. Example labels: "
            f"{example_labels}"
        )

    mito_cell = (
        mito.groupby("cell_label")
        .agg(
            total_TMRM=("ratio_01_00", "mean"),
            mito_count=("cell_label", "count"),
            mean_cell_area=("area_cell", "mean"),
            PV_CV=("PV_CV", "first"),
            liver_id=("liver_id", "first"),
            Condition=("Condition", "first"),
            **{f"{feature}_mito": (feature, "mean") for feature in MORPH_FEATURES},
        )
    )

    ld_cell = (
        ld.groupby("cell_label")
        .agg(
            ld_count=("cell_label", "count"),
            **{f"{feature}_ld": (feature, "mean") for feature in MORPH_FEATURES},
        )
    )

    # The original merge is an inner join, so only cells present in both
    # organelle tables are retained.
    cell_df = mito_cell.merge(ld_cell, left_index=True, right_index=True)

    cell_df["mito_density"] = cell_df["mito_count"] / cell_df["mean_cell_area"]
    cell_df["ld_density"] = cell_df["ld_count"] / cell_df["mean_cell_area"]

    print(f"Total cells after mitochondrial/LD inner merge: {cell_df.shape[0]}")
    return cell_df


# =============================================================================
# GLOBAL ORGANELE SCORE - FORMULA PRESERVED EXACTLY
# =============================================================================

def fit_global_organelle_score(
    cell_df: pd.DataFrame,
) -> tuple[pd.DataFrame, GlobalScoreArtifacts]:
    # Preserve the original feature-selection rule and column order.
    mito_features = [column for column in cell_df.columns if "_mito" in column] + [
        "mito_density"
    ]
    ld_features = [column for column in cell_df.columns if "_ld" in column] + [
        "ld_density"
    ]

    required = mito_features + ld_features + ["total_TMRM", "PV_CV"]
    analysis_df = cell_df.dropna(subset=required).copy()

    if analysis_df.empty:
        raise ValueError("No complete cells remain for the global score analysis.")

    y_all = analysis_df["total_TMRM"].to_numpy()

    # Exact global standardization used by the original code.
    scaler_mito = StandardScaler()
    scaler_ld = StandardScaler()
    mito_z_all = scaler_mito.fit_transform(analysis_df[mito_features])
    ld_z_all = scaler_ld.fit_transform(analysis_df[ld_features])

    # Exact outcome-derived signs and |r|-normalized weights.
    mito_correlations = np.array(
        [pearsonr(analysis_df[feature], y_all)[0] for feature in mito_features]
    )
    ld_correlations = np.array(
        [pearsonr(analysis_df[feature], y_all)[0] for feature in ld_features]
    )

    if not np.isfinite(mito_correlations).all() or not np.isfinite(ld_correlations).all():
        raise ValueError(
            "At least one feature has an undefined Pearson correlation, usually "
            "because it is constant in the analysis data."
        )

    mito_signs = np.sign(mito_correlations)
    ld_signs = np.sign(ld_correlations)

    mito_weight_denominator = np.sum(np.abs(mito_correlations))
    ld_weight_denominator = np.sum(np.abs(ld_correlations))
    if mito_weight_denominator == 0 or ld_weight_denominator == 0:
        raise ValueError("Correlation-based weights cannot be normalized because all r values are zero.")

    mito_weights = np.abs(mito_correlations) / mito_weight_denominator
    ld_weights = np.abs(ld_correlations) / ld_weight_denominator

    mito_axis = np.sum(mito_z_all * mito_signs * mito_weights, axis=1)
    ld_axis = np.sum(ld_z_all * ld_signs * ld_weights, axis=1)

    # Exact two-axis global regression from the original code.
    x_global = np.column_stack([mito_axis, ld_axis])
    global_regression = LinearRegression().fit(x_global, y_all)
    beta_mito, beta_ld = global_regression.coef_

    # Important: preserve the original score definition, which omits the
    # regression intercept and uses only beta_M * M + beta_L * L.
    analysis_df["M_score"] = mito_axis
    analysis_df["L_score"] = ld_axis
    analysis_df["OrgScore"] = beta_mito * mito_axis + beta_ld * ld_axis

    feature_weight_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "compartment": "Mitochondria",
                    "feature": mito_features,
                    "pearson_r_with_TMRM": mito_correlations,
                    "sign": mito_signs,
                    "normalized_abs_r_weight": mito_weights,
                }
            ),
            pd.DataFrame(
                {
                    "compartment": "Lipid droplet",
                    "feature": ld_features,
                    "pearson_r_with_TMRM": ld_correlations,
                    "sign": ld_signs,
                    "normalized_abs_r_weight": ld_weights,
                }
            ),
        ],
        ignore_index=True,
    )

    artifacts = GlobalScoreArtifacts(
        mito_features=mito_features,
        ld_features=ld_features,
        scaler_mito=scaler_mito,
        scaler_ld=scaler_ld,
        mito_correlations=mito_correlations,
        ld_correlations=ld_correlations,
        mito_signs=mito_signs,
        ld_signs=ld_signs,
        mito_weights=mito_weights,
        ld_weights=ld_weights,
        beta_mito=float(beta_mito),
        beta_ld=float(beta_ld),
        global_regression=global_regression,
        feature_weight_table=feature_weight_table,
    )

    print("\n=======================================")
    print("GLOBAL WEIGHTS")
    print("beta_M:", artifacts.beta_mito)
    print("beta_L:", artifacts.beta_ld)
    print("Global regression intercept (not included in OrgScore):", global_regression.intercept_)
    print("Cells in score analysis:", len(analysis_df))

    return analysis_df, artifacts


# =============================================================================
# FIGURE 1 - EXACT ORIGINAL OVERLAY
# =============================================================================

def plot_original_pv_cv_overlay(
    analysis_df: pd.DataFrame,
    x_limits: tuple[float, float],
    output_path: Path | None,
) -> None:
    """Generate the original figure with the original parameters and style."""
    fig = plt.figure(figsize=STYLE.figure_size)

    df_pv = analysis_df[analysis_df["PV_CV"] == "PV"]
    df_cv = analysis_df[analysis_df["PV_CV"] == "CV"]

    # Preserve the original plotting order: PV first, then CV.
    for df, region in [(df_pv, "PV"), (df_cv, "CV")]:
        style = region_style(region)
        x = df["OrgScore"].to_numpy()
        y = df["total_TMRM"].to_numpy()

        x_design = sm.add_constant(x)
        model = sm.OLS(y, x_design).fit()

        x_sorted_index = np.argsort(x)
        x_sorted = x[x_sorted_index]
        x_sorted_design = sm.add_constant(x_sorted)

        prediction = model.get_prediction(x_sorted_design)
        y_predicted = prediction.predicted_mean
        confidence_interval = prediction.conf_int()
        prediction_frame = prediction.summary_frame(alpha=0.05)

        pearson_r, _ = pearsonr(x, y)
        spearman_rho, _ = spearmanr(x, y)
        r_squared = r2_score(y, model.predict(x_design))

        if style["scatter_style"] == "filled":
            plt.scatter(
                x,
                y,
                color=style["dot_edge_color"],
                alpha=style["dot_alpha"],
                s=style["dot_size"],
            )
        else:
            plt.scatter(
                x,
                y,
                facecolors="none",
                edgecolors=style["dot_edge_color"],
                linewidth=style["dot_edge_width"],
                s=style["dot_size"],
                alpha=style["dot_alpha"],
            )

        plt.plot(
            x_sorted,
            y_predicted,
            color=style["line_color"],
            linewidth=style["line_width"],
            linestyle=style["line_style"],
        )

        if STYLE.show_confidence_band:
            plt.fill_between(
                x_sorted,
                confidence_interval[:, 0],
                confidence_interval[:, 1],
                alpha=STYLE.confidence_alpha,
                color=style["confidence_color"],
            )

        if STYLE.show_prediction_band:
            plt.plot(
                x_sorted,
                prediction_frame["obs_ci_lower"].to_numpy(),
                linestyle=STYLE.prediction_line_style,
                linewidth=STYLE.prediction_line_width,
                color=style["prediction_color"],
            )
            plt.plot(
                x_sorted,
                prediction_frame["obs_ci_upper"].to_numpy(),
                linestyle=STYLE.prediction_line_style,
                linewidth=STYLE.prediction_line_width,
                color=style["prediction_color"],
            )

        stats_text = ""
        if STYLE.show_pearson:
            stats_text += f"{region} Pearson r = {pearson_r:.3f}\n"
        if STYLE.show_spearman:
            stats_text += f"{region} Spearman rho = {spearman_rho:.3f}\n"
        if STYLE.show_r2:
            stats_text += f"{region} R2 = {r_squared:.3f}"

        # The original code creates this text object even when it is empty.
        plt.text(
            0.05,
            style["text_y_offset"],
            stats_text.strip(),
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=STYLE.stats_fontsize,
        )

    apply_original_axes("Combined (Control + Fasted): PV vs CV", x_limits)
    finish_figure(fig, output_path)


# =============================================================================
# CROSS-VALIDATED REGIONAL REGRESSIONS
# =============================================================================

def cross_validate_region_regressions(
    analysis_df: pd.DataFrame,
) -> CrossValidationResults:
    """
    Reproduce the original regional CV logic:

    - OrgScore geometry is fixed from the full-data global score.
    - Within each region, five-fold KFold is used.
    - In each fold, only the slope and intercept mapping OrgScore to TMRM are fit.

    This matches the existing code's CV target. It is not an end-to-end nested
    validation of the score-construction step.
    """
    prediction_records: list[dict[str, object]] = []
    fold_model_records: list[dict[str, object]] = []
    fold_metric_records: list[dict[str, object]] = []

    for region in ["PV", "CV"]:
        sub = analysis_df[analysis_df["PV_CV"] == region].copy()
        if len(sub) < CV_N_SPLITS:
            raise ValueError(
                f"Region {region} has {len(sub)} cells, fewer than {CV_N_SPLITS} folds."
            )

        x = sub["OrgScore"].to_numpy()
        y = sub["total_TMRM"].to_numpy()
        cell_labels = sub.index.astype(str).to_numpy()

        splitter = KFold(
            n_splits=CV_N_SPLITS,
            shuffle=CV_SHUFFLE,
            random_state=CV_RANDOM_STATE,
        )

        for fold_number, (train_index, test_index) in enumerate(splitter.split(x), start=1):
            model = LinearRegression().fit(x[train_index].reshape(-1, 1), y[train_index])
            predicted = model.predict(x[test_index].reshape(-1, 1))
            fold_r2 = r2_score(y[test_index], predicted)

            fold_model_records.append(
                {
                    "region": region,
                    "fold": fold_number,
                    "intercept": float(model.intercept_),
                    "slope": float(model.coef_[0]),
                    "train_n": int(len(train_index)),
                    "test_n": int(len(test_index)),
                    "test_x_min": float(np.min(x[test_index])),
                    "test_x_max": float(np.max(x[test_index])),
                }
            )
            fold_metric_records.append(
                {
                    "region": region,
                    "fold": fold_number,
                    "r2": float(fold_r2),
                }
            )

            for local_position, prediction_value in zip(test_index, predicted):
                prediction_records.append(
                    {
                        "cell_label": cell_labels[local_position],
                        "region": region,
                        "fold": fold_number,
                        "OrgScore": float(x[local_position]),
                        "observed_TMRM": float(y[local_position]),
                        "oof_predicted_TMRM": float(prediction_value),
                    }
                )

    predictions = pd.DataFrame(prediction_records)
    fold_models = pd.DataFrame(fold_model_records)
    fold_metrics = pd.DataFrame(fold_metric_records)

    summary_records = []
    for region in ["PV", "CV"]:
        region_predictions = predictions[predictions["region"] == region]
        region_metrics = fold_metrics[fold_metrics["region"] == region]
        summary_records.append(
            {
                "region": region,
                "mean_fold_r2": float(region_metrics["r2"].mean()),
                "sd_fold_r2": float(region_metrics["r2"].std(ddof=0)),
                "pooled_oof_r2": float(
                    r2_score(
                        region_predictions["observed_TMRM"],
                        region_predictions["oof_predicted_TMRM"],
                    )
                ),
                "n_cells": int(len(region_predictions)),
            }
        )

    summary = pd.DataFrame(summary_records)

    print("\n=======================================")
    print("FIVE-FOLD CROSS-VALIDATED REGIONAL REGRESSIONS")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return CrossValidationResults(
        predictions=predictions,
        fold_models=fold_models,
        fold_metrics=fold_metrics,
        summary=summary,
    )


def plot_cross_validated_regression_lines(
    analysis_df: pd.DataFrame,
    cv_results: CrossValidationResults,
    x_limits: tuple[float, float],
    output_path: Path | None,
) -> None:
    """Plot all training-derived fold regressions and their mean line."""
    fig = plt.figure(figsize=STYLE.figure_size)

    for region in ["PV", "CV"]:
        sub = analysis_df[analysis_df["PV_CV"] == region]
        style = region_style(region)
        draw_region_scatter(sub, region)

        x_line = np.linspace(float(sub["OrgScore"].min()), float(sub["OrgScore"].max()), 250)
        models = cv_results.fold_models[cv_results.fold_models["region"] == region]

        # Thin lines: one regression trained in each fold.
        for row in models.itertuples(index=False):
            y_line = row.intercept + row.slope * x_line
            plt.plot(
                x_line,
                y_line,
                color=style["line_color"],
                linewidth=0.8,
                linestyle=style["line_style"],
                alpha=0.22,
            )

        # Thick line: average of the five fitted intercepts and slopes.
        mean_intercept = float(models["intercept"].mean())
        mean_slope = float(models["slope"].mean())
        plt.plot(
            x_line,
            mean_intercept + mean_slope * x_line,
            color=style["line_color"],
            linewidth=style["line_width"],
            linestyle=style["line_style"],
        )

    apply_original_axes("5-fold cross-validated regressions: PV vs CV", x_limits)
    finish_figure(fig, output_path)


def plot_out_of_fold_predictions(
    cv_results: CrossValidationResults,
    output_path: Path | None,
) -> None:
    """Plot observed TMRM against out-of-fold predicted TMRM."""
    fig = plt.figure(figsize=STYLE.figure_size)

    for region in ["PV", "CV"]:
        style = region_style(region)
        sub = cv_results.predictions[cv_results.predictions["region"] == region]
        x = sub["oof_predicted_TMRM"].to_numpy()
        y = sub["observed_TMRM"].to_numpy()

        if style["scatter_style"] == "filled":
            plt.scatter(
                x,
                y,
                color=style["dot_edge_color"],
                alpha=style["dot_alpha"],
                s=style["dot_size"],
            )
        else:
            plt.scatter(
                x,
                y,
                facecolors="none",
                edgecolors=style["dot_edge_color"],
                linewidth=style["dot_edge_width"],
                s=style["dot_size"],
                alpha=style["dot_alpha"],
            )

    lower, upper = STYLE.y_limits
    plt.plot([lower, upper], [lower, upper], color="black", linewidth=1.0, linestyle="--")
    plt.title("Out-of-fold predictions: PV vs CV", fontsize=STYLE.title_fontsize)
    plt.xlabel("Out-of-fold predicted TMRM", fontsize=STYLE.label_fontsize)
    plt.ylabel("Observed TMRM", fontsize=STYLE.label_fontsize)
    plt.tick_params(axis="both", labelsize=STYLE.tick_fontsize)
    plt.xlim(STYLE.y_limits)
    plt.ylim(STYLE.y_limits)
    finish_figure(fig, output_path)


# =============================================================================
# REGION INTERACTION MODEL - FIT AND PLOT THE MODEL DIRECTLY
# =============================================================================

def fit_region_interaction_model(analysis_df: pd.DataFrame):
    model_df = analysis_df.copy()
    model_df["Region_binary"] = (model_df["PV_CV"] == "CV").astype(int)
    model_df["Interaction_region"] = model_df["OrgScore"] * model_df["Region_binary"]

    x_region = sm.add_constant(
        model_df[["OrgScore", "Region_binary", "Interaction_region"]]
    )
    model = sm.OLS(model_df["total_TMRM"], x_region).fit()

    print("\n=======================================")
    print("REGION SLOPE-DIFFERENCE MODEL: PV VS CV")
    print(model.summary())
    print(
        "Interaction p-value:",
        model.pvalues.get("Interaction_region", np.nan),
    )
    return model_df, model


def plot_region_interaction_model(
    model_df: pd.DataFrame,
    model,
    x_limits: tuple[float, float],
    output_path: Path | None,
) -> None:
    """Plot predictions and 95% mean-response CIs from the interaction model."""
    fig = plt.figure(figsize=STYLE.figure_size)

    for region in ["PV", "CV"]:
        style = region_style(region)
        region_binary = 0 if region == "PV" else 1
        sub = model_df[model_df["PV_CV"] == region]
        x = sub["OrgScore"].to_numpy()
        x_sorted = np.sort(x)

        draw_region_scatter(sub, region)

        new_design = pd.DataFrame(
            {
                "OrgScore": x_sorted,
                "Region_binary": region_binary,
                "Interaction_region": x_sorted * region_binary,
            }
        )
        new_design = sm.add_constant(new_design, has_constant="add")
        new_design = new_design[model.model.exog_names]

        prediction = model.get_prediction(new_design)
        y_predicted = prediction.predicted_mean
        confidence_interval = prediction.conf_int(alpha=0.05)

        plt.plot(
            x_sorted,
            y_predicted,
            color=style["line_color"],
            linewidth=style["line_width"],
            linestyle=style["line_style"],
        )
        if STYLE.show_confidence_band:
            plt.fill_between(
                x_sorted,
                confidence_interval[:, 0],
                confidence_interval[:, 1],
                alpha=STYLE.confidence_alpha,
                color=style["confidence_color"],
            )

    apply_original_axes("Region interaction model: PV vs CV", x_limits)
    finish_figure(fig, output_path)


# =============================================================================
# LIVER-LEVEL MIXED-EFFECTS MODEL - FIT AND PLOT DIRECTLY
# =============================================================================

def fit_liver_mixed_effects_model(analysis_df: pd.DataFrame):
    required = ["OrgScore", "total_TMRM", "PV_CV", "Condition", "liver_id"]
    require_columns(analysis_df, required, "Analysis table")
    model_df = analysis_df.dropna(subset=required).copy()

    model_df["Region_binary"] = (model_df["PV_CV"] == "CV").astype(int)
    model_df["Condition_binary"] = (model_df["Condition"] == "Fasted").astype(int)

    mixed_model = smf.mixedlm(
        "total_TMRM ~ OrgScore * Region_binary * Condition_binary",
        model_df,
        groups=model_df["liver_id"],
    ).fit(reml=MIXED_MODEL_REML)

    print("\n=======================================")
    print("MIXED-EFFECTS MODEL (random intercept: liver_id)")
    print(mixed_model.summary())
    return model_df, mixed_model


def mixed_fixed_prediction(
    mixed_result,
    new_data: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population-average prediction and CI from fixed-effect uncertainty."""
    design_info = mixed_result.model.data.design_info
    design = build_design_matrices(
        [design_info],
        new_data,
        return_type="dataframe",
    )[0]

    fixed_effects = mixed_result.fe_params
    design = design.loc[:, fixed_effects.index]
    design_array = design.to_numpy(dtype=float)

    mean = design_array @ fixed_effects.to_numpy(dtype=float)
    covariance = mixed_result.cov_params().loc[
        fixed_effects.index,
        fixed_effects.index,
    ].to_numpy(dtype=float)
    variance = np.einsum("ij,jk,ik->i", design_array, covariance, design_array)
    standard_error = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
    critical_value = norm.ppf(1.0 - alpha / 2.0)

    lower = mean - critical_value * standard_error
    upper = mean + critical_value * standard_error
    return mean, lower, upper


def random_intercept_for_liver(mixed_result, liver_id: object) -> float:
    random_effect = mixed_result.random_effects[liver_id]
    return float(np.asarray(random_effect, dtype=float).ravel()[0])


def plot_mixed_effects_by_condition(
    model_df: pd.DataFrame,
    mixed_result,
    condition: str,
    x_limits: tuple[float, float],
    output_path: Path | None,
) -> None:
    """
    Plot fixed-effect mixed-model lines for one condition.

    If enabled, thin lines add each liver's fitted random intercept. The thick
    black lines and red bands remain the population-average fixed-effect result.
    """
    condition_binary = 0 if condition == "Control" else 1
    condition_df = model_df[model_df["Condition"] == condition]
    fig = plt.figure(figsize=STYLE.figure_size)

    for region in ["PV", "CV"]:
        style = region_style(region)
        region_binary = 0 if region == "PV" else 1
        sub = condition_df[condition_df["PV_CV"] == region]
        if sub.empty:
            continue

        draw_region_scatter(sub, region)
        x_sorted = np.sort(sub["OrgScore"].to_numpy())

        fixed_new_data = pd.DataFrame(
            {
                "OrgScore": x_sorted,
                "Region_binary": region_binary,
                "Condition_binary": condition_binary,
            }
        )
        fixed_mean, fixed_lower, fixed_upper = mixed_fixed_prediction(
            mixed_result,
            fixed_new_data,
        )

        if SHOW_LIVER_SPECIFIC_MIXED_LINES:
            for liver_id, liver_sub in sub.groupby("liver_id"):
                liver_x = np.linspace(
                    float(liver_sub["OrgScore"].min()),
                    float(liver_sub["OrgScore"].max()),
                    60,
                )
                liver_new_data = pd.DataFrame(
                    {
                        "OrgScore": liver_x,
                        "Region_binary": region_binary,
                        "Condition_binary": condition_binary,
                    }
                )
                liver_fixed_mean, _, _ = mixed_fixed_prediction(
                    mixed_result,
                    liver_new_data,
                )
                try:
                    random_intercept = random_intercept_for_liver(
                        mixed_result,
                        liver_id,
                    )
                except (KeyError, ValueError):
                    continue

                plt.plot(
                    liver_x,
                    liver_fixed_mean + random_intercept,
                    color=style["line_color"],
                    linewidth=LIVER_LINE_WIDTH,
                    linestyle=style["line_style"],
                    alpha=LIVER_LINE_ALPHA,
                )

        plt.plot(
            x_sorted,
            fixed_mean,
            color=style["line_color"],
            linewidth=style["line_width"],
            linestyle=style["line_style"],
        )
        if STYLE.show_confidence_band:
            plt.fill_between(
                x_sorted,
                fixed_lower,
                fixed_upper,
                alpha=STYLE.confidence_alpha,
                color=style["confidence_color"],
            )

    apply_original_axes(
        f"Mixed effects ({condition}): PV vs CV",
        x_limits,
    )
    finish_figure(fig, output_path)


# =============================================================================
# OUTPUT TABLES
# =============================================================================

def save_analysis_outputs(
    analysis_df: pd.DataFrame,
    score_artifacts: GlobalScoreArtifacts,
    cv_results: CrossValidationResults,
    region_model,
    mixed_result,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis_df.to_csv(OUTPUT_DIR / "analysis_cells_with_global_organelle_score.csv")
    score_artifacts.feature_weight_table.to_csv(
        OUTPUT_DIR / "global_feature_correlations_and_weights.csv",
        index=False,
    )
    cv_results.predictions.to_csv(
        OUTPUT_DIR / "cross_validated_out_of_fold_predictions.csv",
        index=False,
    )
    cv_results.fold_models.to_csv(
        OUTPUT_DIR / "cross_validated_fold_regressions.csv",
        index=False,
    )
    cv_results.fold_metrics.to_csv(
        OUTPUT_DIR / "cross_validated_fold_metrics.csv",
        index=False,
    )
    cv_results.summary.to_csv(
        OUTPUT_DIR / "cross_validated_summary.csv",
        index=False,
    )

    region_parameter_table = pd.DataFrame(
        {
            "coefficient": region_model.params,
            "standard_error": region_model.bse,
            "p_value": region_model.pvalues,
            "ci_95_lower": region_model.conf_int()[0],
            "ci_95_upper": region_model.conf_int()[1],
        }
    )
    region_parameter_table.to_csv(OUTPUT_DIR / "region_interaction_coefficients.csv")
    save_text(
        region_model.summary().as_text(),
        OUTPUT_DIR / "region_interaction_model_summary.txt",
    )

    fixed_effect_names = mixed_result.fe_params.index
    mixed_ci = mixed_result.conf_int().loc[fixed_effect_names]
    mixed_parameter_table = pd.DataFrame(
        {
            "coefficient": mixed_result.fe_params,
            "standard_error": mixed_result.bse.loc[fixed_effect_names],
            "p_value": mixed_result.pvalues.loc[fixed_effect_names],
            "ci_95_lower": mixed_ci[0],
            "ci_95_upper": mixed_ci[1],
        }
    )
    mixed_parameter_table.to_csv(OUTPUT_DIR / "mixed_effects_fixed_coefficients.csv")

    random_effect_records = []
    for liver_id, random_effect in mixed_result.random_effects.items():
        random_effect_records.append(
            {
                "liver_id": liver_id,
                "random_intercept": float(np.asarray(random_effect).ravel()[0]),
            }
        )
    pd.DataFrame(random_effect_records).to_csv(
        OUTPUT_DIR / "mixed_effects_liver_random_intercepts.csv",
        index=False,
    )
    save_text(
        mixed_result.summary().as_text(),
        OUTPUT_DIR / "mixed_effects_model_summary.txt",
    )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cell_df = load_and_aggregate_cell_data()
    analysis_df, score_artifacts = fit_global_organelle_score(cell_df)
    x_limits = calculate_x_limits(analysis_df["OrgScore"])

    print("X limits:", x_limits)
    print("Y limits:", STYLE.y_limits)

    # Figure 1: exact reproduction of the original plot.
    plot_original_pv_cv_overlay(
        analysis_df,
        x_limits,
        OUTPUT_DIR / "01_original_combined_pv_vs_cv.png",
    )

    # Figures 2-3: regional cross-validation views.
    cv_results = cross_validate_region_regressions(analysis_df)
    plot_cross_validated_regression_lines(
        analysis_df,
        cv_results,
        x_limits,
        OUTPUT_DIR / "02_cross_validated_regression_lines.png",
    )
    plot_out_of_fold_predictions(
        cv_results,
        OUTPUT_DIR / "03_cross_validated_out_of_fold_predictions.png",
    )

    # Figure 4: direct predictions from the region interaction model.
    region_model_df, region_model = fit_region_interaction_model(analysis_df)
    plot_region_interaction_model(
        region_model_df,
        region_model,
        x_limits,
        OUTPUT_DIR / "04_region_interaction_model.png",
    )

    # Figures 5-6: direct mixed-model predictions for each condition.
    mixed_model_df, mixed_result = fit_liver_mixed_effects_model(analysis_df)
    plot_mixed_effects_by_condition(
        mixed_model_df,
        mixed_result,
        "Control",
        x_limits,
        OUTPUT_DIR / "05_mixed_effects_control_pv_vs_cv.png",
    )
    plot_mixed_effects_by_condition(
        mixed_model_df,
        mixed_result,
        "Fasted",
        x_limits,
        OUTPUT_DIR / "06_mixed_effects_fasted_pv_vs_cv.png",
    )

    save_analysis_outputs(
        analysis_df,
        score_artifacts,
        cv_results,
        region_model,
        mixed_result,
    )

    print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
