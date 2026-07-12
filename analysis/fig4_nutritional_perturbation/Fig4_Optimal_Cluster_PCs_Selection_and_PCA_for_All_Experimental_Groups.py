"""
Optimal PC and GMM cluster selection by experiment/group.

This script can:
  1. Load one combined CSV containing multiple experiments.
  2. Use an existing group column OR derive the experiment from the labels column.
  3. Run the PCA + GMM cluster-number selection independently for each experiment.
  4. Penalize unstable ARI results using ARI_mean - penalty_weight * ARI_std
     only when selecting the final PC set.
  5. Keep cluster-number selection based on the selected cluster metrics only.
  6. Save per-experiment summaries, metric tables, plots, and final cluster labels.

"""

from __future__ import annotations

import argparse
import ast
import re
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so the script can save plots on servers/Windows terminals.
# Use --show-plots if you also want figures displayed interactively.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

sns.set()


# ----------------------------
# Constants and Configuration
# ----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "Full_data_perturbations.csv" #<----This will look for the csv in the same directory where the script has been saved.
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "cluster_selection_outputs"

LABEL_COLUMN = "labels"
GROUP_COLUMN = "group"

# Assign Type and experiment names based on labels.
# You can add additional experiments here, e.g. "XYZ_": 4.
LABEL_MAP = {
    "CNT_": 1,
    "STV_": 2,
    "WD_": 3,
}

LABEL_PREFIX_TO_EXPERIMENT = {
    prefix: prefix.rstrip("_")
    for prefix in LABEL_MAP
}


# Orientation controls for the selected-experiment PCA plot only.
# Mirror X = left/right mirror: PC1 -> -PC1
# Mirror Y = up/down mirror: PC2 -> -PC2
# Rotate 180 = PC1 -> -PC1 and PC2 -> -PC2

SCRIPT_SELECTED_PCA_MIRROR_X = False
SCRIPT_SELECTED_PCA_MIRROR_Y = False
SCRIPT_SELECTED_PCA_ROTATE_180 = True


MITO_COUNT_THRESHOLD = 20
MIN_PCS_TO_TEST = 2
MAX_PCS_TO_TEST = 10
MAX_CLUSTERS = 10
CV_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = -1
N_INIT = 5
N_STABILITY_RUNS = 50
STABILITY_SUBSAMPLE_FRACTION = 0.8
MIN_SAMPLES_PER_EXPERIMENT = 30

# ARI variability penalty settings.
# ARI_penalized_score = ARI_mean - ARI_STD_PENALTY_WEIGHT * ARI_std
#   0.0 = no penalty
#   0.5 = mild penalty
#   1.0 = one-standard-deviation lower-bound stability score
#   2.0 = conservative penalty
ARI_STD_PENALTY_WEIGHT = 1.0

# Metric weights used when cluster_selection_metric="combined_score".
# Higher is better for log-likelihood, silhouette, and CV score.
# Lower is better for BIC and AIC; these are inverted in the combined score.
WEIGHTS = {
    "log_likelihood": 0.05,
    "bic": 0.25,
    "aic": 0.20,
    "silhouette": 0.15,
    "cv_score": 0.35,
}

# Columns that should not be used as numeric features.
# errors="ignore" is used later, so it is safe if some columns are absent.
DEFAULT_EXCLUDED_COLUMNS = [
    "area",
    "centroid-0",
    "centroid-1",
    "labels",
    "group",
    "Experiment",
    "Type",
    "type_1_ld_avg_aspect_ratio",
    "type_2_ld_avg_aspect_ratio",
    "type_3_ld_avg_aspect_ratio",
    "type_4_ld_avg_aspect_ratio",
    "type_1_ld_avg_solidity",
    "type_2_ld_avg_solidity",
    "type_3_ld_avg_solidity",
    "type_4_ld_avg_solidity",
    "stack_id",
    "cell_id_linked",
    "ascini_position",
]

# ----------------------------
# No-Bash Run Settings
# ----------------------------
# Keep USE_SCRIPT_SETTINGS = True when running this file by pressing Run in
# Spyder, VS Code, PyCharm, or Jupyter. Change the values below instead of
# typing command-line options in Bash/Terminal.
#
# Set SCRIPT_EXPERIMENTS_TO_RUN like this:
#   None          -> run all detected experiments
#   ["CNT"]       -> run only CNT
#   ["STV", "WD"] -> run only STV and WD
#
# Set USE_SCRIPT_SETTINGS = False only if you want to use command-line options.
USE_SCRIPT_SETTINGS = True

SCRIPT_DATA_PATH = DEFAULT_DATA_PATH
SCRIPT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
SCRIPT_LABEL_COLUMN = LABEL_COLUMN
SCRIPT_GROUP_COLUMN = GROUP_COLUMN
SCRIPT_GROUP_SOURCE = "labels"
SCRIPT_EXPERIMENTS_TO_RUN = None #["STV","WD"]   <--------Select experiment here (None is for all)
SCRIPT_POOLED = False

SCRIPT_MIN_PCS = MIN_PCS_TO_TEST
SCRIPT_MAX_PCS = MAX_PCS_TO_TEST
SCRIPT_MAX_CLUSTERS = MAX_CLUSTERS
# Recommended when you want to penalize ARI variability only during PC selection:
#   cluster_selection_metric = "combined_score"
#   pc_selection_metric = "ari_penalized_then_combined"
SCRIPT_CLUSTER_SELECTION_METRIC = "combined_score"
SCRIPT_PC_SELECTION_METRIC = "ari_penalized_then_combined"
SCRIPT_ARI_STD_PENALTY_WEIGHT = ARI_STD_PENALTY_WEIGHT
SCRIPT_MITO_COUNT_THRESHOLD = MITO_COUNT_THRESHOLD
SCRIPT_STABILITY_RUNS = N_STABILITY_RUNS
SCRIPT_STABILITY_SUBSAMPLE_FRACTION = STABILITY_SUBSAMPLE_FRACTION
SCRIPT_CV_SPLITS = CV_SPLITS
SCRIPT_RANDOM_STATE = RANDOM_STATE
SCRIPT_JOBS = N_JOBS
SCRIPT_N_INIT = N_INIT
SCRIPT_MIN_SAMPLES = MIN_SAMPLES_PER_EXPERIMENT
SCRIPT_SHOW_PLOTS = False

# Apply this only to the combined selected-experiment PCA plot.
# True  = remove cells where cell_id_linked == 0
# False = plot all cells
SCRIPT_SELECTED_PCA_EXCLUDE_CELL_ID_LINKED_ZERO = True

# ----------------------------
# Label/group helpers
# ----------------------------

def parse_labels_cell(value) -> List[str]:
    """
    Convert one value from the labels column into a list of label strings.

    The uploaded CSV stores labels as strings that look like Python lists, e.g.
    "['CNT_L1L1a0s0_1', 'CNT_L1L1a0s1_1']".
    This function also supports plain strings or actual lists/tuples.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [str(item) for item in value if item is not None]

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray, pd.Series)):
            return [str(item) for item in parsed if item is not None]
        return [str(parsed)]
    except Exception:
        return [text]


def label_to_type(label_value, label_map: Dict[str, int] = LABEL_MAP) -> int:
    """
    Assign numeric Type from labels using LABEL_MAP.

    This is a safer version of:
        if "CNT_" in label: return 1
    because it also handles list-like label strings.
    """
    labels = parse_labels_cell(label_value)
    for label in labels:
        for prefix, type_value in label_map.items():
            if label.startswith(prefix) or prefix in label:
                return type_value
    return 0


def label_to_experiment(
    label_value,
    prefix_to_experiment: Dict[str, str] = LABEL_PREFIX_TO_EXPERIMENT,
    unknown_name: str = "Unknown",
) -> str:
    """
    Derive an experiment/group name from the labels column.

    First uses known prefixes from LABEL_PREFIX_TO_EXPERIMENT, e.g. CNT_, STV_, WD_.
    If no known prefix is found, it falls back to the text before the first underscore.
    """
    labels = parse_labels_cell(label_value)
    if len(labels) == 0:
        return unknown_name

    for label in labels:
        for prefix, experiment in prefix_to_experiment.items():
            if label.startswith(prefix) or prefix in label:
                return experiment

    first_label = labels[0]
    match = re.match(r"^([^_]+)_", first_label)
    if match:
        return match.group(1)

    return unknown_name


def add_type_and_experiment_columns(
    data: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
    group_column: str = GROUP_COLUMN,
    group_source: str = "auto",
) -> pd.DataFrame:
    """
    Add two metadata columns:
      - Type: numeric code derived from labels using LABEL_MAP.
      - Experiment: experiment/group name used for per-experiment clustering.

    group_source options:
      - "auto": use group_column if it exists; otherwise derive from labels.
      - "group": require group_column and use it.
      - "labels": derive from labels even if group_column exists.
    """
    data = data.copy()

    if label_column not in data.columns:
        raise ValueError(
            f"Could not find label column '{label_column}'. Available columns include: "
            f"{list(data.columns[:20])} ..."
        )

    data["Type"] = data[label_column].apply(label_to_type)

    if group_source not in {"auto", "group", "labels"}:
        raise ValueError("group_source must be one of: auto, group, labels")

    if group_source == "group":
        if group_column not in data.columns:
            raise ValueError(
                f"group_source='group' was requested, but column '{group_column}' was not found. "
                f"Use group_source='labels' to derive experiments from '{label_column}'."
            )
        data["Experiment"] = data[group_column].astype(str)

    elif group_source == "auto" and group_column in data.columns:
        data["Experiment"] = data[group_column].astype(str)

    else:
        data["Experiment"] = data[label_column].apply(label_to_experiment)

    return data


# ----------------------------
# General helpers
# ----------------------------

def safe_name(value: str) -> str:
    """Make a string safe for filenames."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unnamed"


def normalize(array) -> np.ndarray:
    """
    Normalize values to 0-1 range.
    Handles NaN, inf, and constant arrays safely.
    """
    array = np.array(array, dtype=float)
    out = np.zeros_like(array, dtype=float)
    valid = np.isfinite(array)

    if valid.sum() == 0:
        return out

    min_val = np.nanmin(array[valid])
    max_val = np.nanmax(array[valid])

    if np.isclose(max_val, min_val):
        out[valid] = 0.5
    else:
        out[valid] = (array[valid] - min_val) / (max_val - min_val)

    out[~valid] = np.nan
    return out


def clean_numeric_values(data: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN and fill missing numeric values with zero."""
    data = data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)
    return data


def filter_data(
    data: pd.DataFrame,
    mito_count_threshold: float = MITO_COUNT_THRESHOLD,
) -> pd.DataFrame:
    """Apply the same mitochondrial filters as the original script."""
    required = ["mito_aspect_ratio", "mito_density", "area"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required filter columns: {missing}")

    data_filtered = data[data["mito_aspect_ratio"] > 0].copy()
    mito_count = data_filtered["mito_density"] * data_filtered["area"]
    data_filtered = data_filtered[mito_count >= mito_count_threshold].copy()
    
    return data_filtered


def get_feature_data(
    data_filtered: pd.DataFrame,
    excluded_columns: Sequence[str] = DEFAULT_EXCLUDED_COLUMNS,
) -> pd.DataFrame:
    """Drop metadata/excluded columns and keep numeric features only."""
    data_features = data_filtered.drop(columns=list(excluded_columns), errors="ignore")
    data_features = data_features.select_dtypes(include=[np.number]).copy()

    # Drop columns that are entirely missing, if any remain.
    data_features = data_features.dropna(axis=1, how="all")

    # Drop constant columns because they add no clustering information.
    nunique = data_features.nunique(dropna=False)
    constant_columns = nunique[nunique <= 1].index.tolist()
    if constant_columns:
        data_features = data_features.drop(columns=constant_columns)

    if data_features.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after exclusions.")

    return data_features


# ----------------------------
# GMM/PCA metric functions
# ----------------------------

def calculate_metrics_for_X(
    n: int,
    X_train: pd.DataFrame,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
    n_init: int = N_INIT,
) -> Tuple[float, float, float, float, float]:
    """
    Fit GMM with n clusters and calculate:
    log-likelihood, BIC, AIC, silhouette, and CV score.
    """
    try:
        gmm = GaussianMixture(
            n_components=n,
            random_state=random_state,
            n_init=n_init,
        )

        labels = gmm.fit_predict(X_train)

        log_likelihood = gmm.score(X_train) * len(X_train)
        bic = gmm.bic(X_train)
        aic = gmm.aic(X_train)

        unique_labels = np.unique(labels)
        if n > 1 and len(unique_labels) > 1 and len(unique_labels) < len(X_train):
            silhouette = silhouette_score(X_train, labels)
        else:
            silhouette = np.nan

        fold_scores = []
        n_splits = min(cv_splits, len(X_train))

        if n_splits >= 2:
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )

            for train_index, val_index in kf.split(X_train):
                X_train_fold = X_train.iloc[train_index]
                X_val_fold = X_train.iloc[val_index]

           
                if len(X_train_fold) <= n:
                    continue

                gmm_fold = GaussianMixture(
                    n_components=n,
                    random_state=random_state,
                    n_init=n_init,
                )
                gmm_fold.fit(X_train_fold)
                fold_scores.append(gmm_fold.score(X_val_fold) * len(X_val_fold))

        cv_score = np.mean(fold_scores) if len(fold_scores) > 0 else np.nan
        return log_likelihood, bic, aic, silhouette, cv_score

    except Exception as exc:
        warnings.warn(f"GMM failed for n_clusters={n}: {exc}")
        return np.nan, np.nan, np.nan, np.nan, np.nan


def add_combined_score(metrics_df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Normalize metrics and calculate a weighted combined score."""
    metrics_df = metrics_df.copy()

    metrics_df["log_likelihood_norm"] = normalize(metrics_df["log_likelihood"])
    metrics_df["bic_norm"] = normalize(metrics_df["bic"])
    metrics_df["aic_norm"] = normalize(metrics_df["aic"])
    metrics_df["silhouette_norm"] = normalize(metrics_df["silhouette"])
    metrics_df["cv_score_norm"] = normalize(metrics_df["cv_score"])

    metrics_df["combined_score"] = (
        weights["log_likelihood"] * np.nan_to_num(metrics_df["log_likelihood_norm"], nan=0.0)
        + weights["bic"] * (1 - np.nan_to_num(metrics_df["bic_norm"], nan=1.0))
        + weights["aic"] * (1 - np.nan_to_num(metrics_df["aic_norm"], nan=1.0))
        + weights["silhouette"] * np.nan_to_num(metrics_df["silhouette_norm"], nan=0.0)
        + weights["cv_score"] * np.nan_to_num(metrics_df["cv_score_norm"], nan=0.0)
    )

    return metrics_df


def choose_optimal_cluster_number(
    metrics_df: pd.DataFrame,
    metric: str = "combined_score",
) -> int:
    
    valid_metrics = {
        "combined_score",
        "bic",
        "aic",
        "silhouette",
        "cv_score",
        "log_likelihood",
    }
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {sorted(valid_metrics)}")

    if metric in {"bic", "aic"}:
        idx = metrics_df[metric].replace([np.inf, -np.inf], np.nan).idxmin()
    else:
        idx = metrics_df[metric].replace([np.inf, -np.inf], np.nan).idxmax()

    if pd.isna(idx):
        raise ValueError(f"Could not choose optimal cluster number using metric '{metric}'.")

    return int(metrics_df.loc[idx, "n_clusters"])



def calculate_ari_penalized_score(
    ari_mean: float,
    ari_std: float,
    ari_std_penalty_weight: float = ARI_STD_PENALTY_WEIGHT,
    n_clusters: Optional[int] = None,
) -> float:
    """
    Calculate the ARI-variability-penalized score used for final PC selection.

    Formula:
        ARI_penalized_score = ARI_mean - ari_std_penalty_weight * ARI_std

    This is applied only after the optimal cluster number has already been
    selected by the cluster-selection metric. It does not influence the
    cluster-number choice.

    """
    if n_clusters is not None and int(n_clusters) < 2:
        return 0.0

    score = np.nan_to_num(ari_mean, nan=0.0) - ari_std_penalty_weight * np.nan_to_num(ari_std, nan=0.0)
    return float(np.clip(score, 0.0, 1.0))
def find_optimal_clusters_for_pc_set(
    X_pc: pd.DataFrame,
    n_clusters_range: Iterable[int],
    cluster_selection_metric: str = "combined_score",
    weights: Dict[str, float] = WEIGHTS,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    n_init: int = N_INIT,
) -> Tuple[int, pd.DataFrame]:
    """
    For one PC set, evaluate all cluster numbers and return:
      - optimal cluster number
      - metrics dataframe
    """
    if len(X_pc) < 3:
        raise ValueError("Need at least 3 samples to run train/test cluster selection.")

    X_train, _ = train_test_split(
        X_pc,
        test_size=0.2,
        random_state=random_state,
    )

    valid_cluster_range = [n for n in n_clusters_range if n < len(X_train)]
    if not valid_cluster_range:
        raise ValueError(
            "No valid cluster numbers to test. Reduce max_clusters or use more samples."
        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_metrics_for_X)(n, X_train, cv_splits, random_state, n_init)
        for n in valid_cluster_range
    )

    log_likelihoods, bics, aics, silhouettes, cv_scores = zip(*results)

    metrics_df = pd.DataFrame(
        {
            "n_clusters": valid_cluster_range,
            "log_likelihood": log_likelihoods,
            "bic": bics,
            "aic": aics,
            "silhouette": silhouettes,
            "cv_score": cv_scores,
        }
    )

    metrics_df = add_combined_score(metrics_df, weights)

    # Cluster-number selection stops here: it uses the chosen cluster metric only.
    # ARI stability is calculated later for the selected cluster number and is
    # used only to select the final PC set.

    optimal_clusters = choose_optimal_cluster_number(metrics_df, cluster_selection_metric)

    return optimal_clusters, metrics_df


def calculate_ari_stability(
    X_pc: pd.DataFrame,
    n_clusters: int,
    n_runs: int = N_STABILITY_RUNS,
    subsample_fraction: float = STABILITY_SUBSAMPLE_FRACTION,
    random_state: int = RANDOM_STATE,
    n_init: int = N_INIT,
) -> Dict[str, float]:
    """
    Estimate cluster stability using ARI.

    Method:
      - Repeatedly subsample the data.
      - Fit a GMM on each subsample.
      - Predict cluster labels for the full dataset.
      - Compare all pairs of full-dataset labelings using ARI.
    """
    if n_runs < 2 or n_clusters < 2:
        # ARI is not meaningful for a one-cluster solution; it is trivially stable.
        return {
            "ARI_mean": np.nan,
            "ARI_std": np.nan,
            "ARI_min": np.nan,
            "ARI_max": np.nan,
            "ARI_n_comparisons": 0,
        }

    rng = np.random.default_rng(random_state)
    X_np = X_pc.to_numpy()
    n_samples = X_np.shape[0]

    sample_size = int(np.floor(subsample_fraction * n_samples))
    sample_size = max(sample_size, n_clusters + 1)
    sample_size = min(sample_size, n_samples)

    all_labels = []

    for run in range(n_runs):
        sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state + run,
            n_init=n_init,
        )
        gmm.fit(X_np[sample_idx])
        all_labels.append(gmm.predict(X_np))

    ari_scores = []
    for i, j in combinations(range(len(all_labels)), 2):
        ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    ari_scores = np.array(ari_scores, dtype=float)

    return {
        "ARI_mean": float(np.mean(ari_scores)),
        "ARI_std": float(np.std(ari_scores)),
        "ARI_min": float(np.min(ari_scores)),
        "ARI_max": float(np.max(ari_scores)),
        "ARI_n_comparisons": int(len(ari_scores)),
    }


def choose_final_pc_set(
    pc_summary_df: pd.DataFrame,
    pc_selection_metric: str = "ari_then_combined",
) -> pd.Series:
    """
    Choose final number of PCs.

    pc_selection_metric options:
      - ari_penalized_then_combined: highest ARI_penalized_score, then highest best_combined_score.
      - ari_then_combined: highest ARI_mean, then highest best_combined_score.
      - combined_then_ari: highest best_combined_score, then highest ARI_mean.
      - ARI_penalized_score: highest ARI_penalized_score.
      - ARI_mean: highest ARI_mean.
      - best_combined_score: highest best_combined_score.
    """
    if pc_selection_metric == "ari_penalized_then_combined":
        return pc_summary_df.sort_values(
            by=["ARI_penalized_score", "best_combined_score"], ascending=[False, False]
        ).iloc[0]

    if pc_selection_metric == "ari_then_combined":
        return pc_summary_df.sort_values(
            by=["ARI_mean", "best_combined_score"], ascending=[False, False]
        ).iloc[0]

    if pc_selection_metric == "combined_then_ari":
        return pc_summary_df.sort_values(
            by=["best_combined_score", "ARI_mean"], ascending=[False, False]
        ).iloc[0]

    if pc_selection_metric in {"ARI_penalized_score", "ARI_mean", "best_combined_score"}:
        return pc_summary_df.sort_values(
            by=pc_selection_metric, ascending=False
        ).iloc[0]

    raise ValueError(
        "pc_selection_metric must be one of: ari_penalized_then_combined, "
        "ari_then_combined, combined_then_ari, "
        "ARI_penalized_score, ARI_mean, best_combined_score"
    )


# ----------------------------
# Plotting
# ----------------------------

def plot_pc_ari_bars(
    pc_results: pd.DataFrame,
    modality: Optional[str] = None,
    output_path: Optional[Path] = None,
    dpi: int = 400,
    show_plots: bool = False,
) -> None:
    """Plot ARI mean with std error bars for each number of PCs."""
    df = pc_results.copy()

    if "n_pcs" in df.columns:
        x_col = "n_pcs"
    elif "n_pc" in df.columns:
        x_col = "n_pc"
    else:
        raise ValueError("Could not find PC number column. Expected 'n_pcs' or 'n_pc'.")

    if "ARI_mean" in df.columns:
        ari_mean_col = "ARI_mean"
    elif "ari_mean" in df.columns:
        ari_mean_col = "ari_mean"
    else:
        raise ValueError("Could not find ARI mean column. Expected 'ARI_mean' or 'ari_mean'.")

    if "ARI_std" in df.columns:
        ari_std_col = "ARI_std"
    elif "ari_std" in df.columns:
        ari_std_col = "ari_std"
    else:
        raise ValueError("Could not find ARI std column. Expected 'ARI_std' or 'ari_std'.")

    df = df.sort_values(x_col)
    x = df[x_col].astype(int).values
    y = df[ari_mean_col].values
    yerr = df[ari_std_col].values

    plt.figure(figsize=(4, 4))
    plt.bar(
        x,
        y,
        yerr=yerr,
        capsize=5,
        edgecolor="black",
        color="purple",
        alpha=0.5,
    )
    plt.xlabel("Number of PCs", fontsize=14)
    plt.ylabel("ARI (mean)", fontsize=14)
    plt.tick_params(axis="both", labelsize=14)
    plt.title(modality if modality else "Clustering Stability vs PCA Dimensionality", fontsize=8)
    plt.xticks(x)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        print(f"[INFO] Saved plot to: {output_path}")

    if show_plots:
        plt.show()
    plt.close()


def plot_final_clusters(
    X_pca_all: pd.DataFrame,
    final_labels: np.ndarray,
    selected_n_pcs: int,
    selected_clusters: int,
    experiment_name: str,
    output_path: Optional[Path] = None,
    dpi: int = 400,
    show_plots: bool = False,
) -> None:
    """Plot final clusters on PC1 vs PC2."""
    X_plot = X_pca_all[["PC1", "PC2"]].copy()
    X_plot["Cluster"] = final_labels

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=X_plot,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        s=100,
    )
    plt.title(
        f"{experiment_name}: GMM using PC1-PC{selected_n_pcs}, {selected_clusters} clusters",
        fontsize=18,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        print(f"[INFO] Saved plot to: {output_path}")

    if show_plots:
        plt.show()
    plt.close()

# ---------------------------------------
# PCA plot of selected experiments together
# ---------------------------------------
def plot_selected_experiments_pca(
    data: pd.DataFrame,
    experiments_to_plot,
    output_dir: Path,
    experiment_column: str = "Experiment",
    mito_count_threshold: float = MITO_COUNT_THRESHOLD,
    random_state: int = RANDOM_STATE,
    point_size: float = 70,
    point_alpha: float = 0.95,
    dpi: int = 400,
    show_plots: bool = False,
    show_axes: bool = False,
    exclude_cell_id_linked_zero: bool = False,
    cell_id_linked_column: str = "cell_id_linked",
    mirror_x: bool = False,
    mirror_y: bool = False,
    rotate_180: bool = False,
  ) -> None:

    """
    Plot the selected experiments together in one shared PCA space.

    This is different from the per-experiment PCA used for cluster selection.
    Here, PCA is fit once using all selected experiments together so that
    PC1 and PC2 are directly comparable across experiments.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_column not in data.columns:
        raise ValueError(
            f"Column '{experiment_column}' was not found. "
            "Run add_type_and_experiment_columns() before making this plot."
        )

    # Decide which experiments to plot.
    # None means use all detected experiments.
    if experiments_to_plot is None:
        experiment_order = [
            str(x) for x in sorted(data[experiment_column].dropna().unique())
        ]
    else:
        experiment_order = [str(x) for x in experiments_to_plot]

    if len(experiment_order) == 0:
        warnings.warn("No experiments were selected for the PCA plot.")
        return

    # If running pooled mode, use all rows but still color by original Experiment.
    if "Pooled" in experiment_order:
        selected_data = data.copy()
        experiment_order = [
            str(x) for x in sorted(selected_data[experiment_column].dropna().unique())
        ]
    else:
        selected_data = data[
            data[experiment_column].astype(str).isin(experiment_order)
        ].copy()

  
    
    if selected_data.empty:
        warnings.warn(f"No rows found for selected experiments: {experiment_order}")
        return

    # Optional filter for removing cells where cell_id_linked == 0.
    # This affects only this selected-experiment PCA plot.
    # It does not affect cluster selection, PC selection, or final model fitting.
    if exclude_cell_id_linked_zero:
        if cell_id_linked_column not in selected_data.columns:
            warnings.warn(
                f"Requested removal of {cell_id_linked_column} == 0, but column "
                f"'{cell_id_linked_column}' was not found. Plotting without this filter."
            )
        else:
            before_n = selected_data.shape[0]

            cell_id_numeric = pd.to_numeric(
                selected_data[cell_id_linked_column],
                errors="coerce"
            )

            # Keep everything except cells where cell_id_linked is exactly 0.
            # NaN values are kept because they are not equal to 0.
            keep_mask = cell_id_numeric.ne(0) | cell_id_numeric.isna()

            removed_n = int((~keep_mask).sum())

            selected_data = selected_data.loc[keep_mask].copy()

            after_n = selected_data.shape[0]

            print(
                f"[INFO] Selected-experiment PCA removed "
                f"{cell_id_linked_column} == 0: "
                f"{before_n} -> {after_n} rows "
                f"({removed_n} removed)"
            )

            if selected_data.empty:
                warnings.warn(
                    f"No rows remain after removing {cell_id_linked_column} == 0."
                )
                return
    
    
            
    # Use the same cleaning, filtering, and feature-selection logic as the main pipeline.
    selected_data = clean_numeric_values(selected_data)
        
    
    data_filtered = filter_data(
        selected_data,
        mito_count_threshold=mito_count_threshold,
    )
    data_filtered = data_filtered[data_filtered["mito_density"] > 0.28].copy()

    if data_filtered.shape[0] < 3:
        warnings.warn("Not enough rows remain after filtering to make a 2D PCA plot.")
        return

    data_features = get_feature_data(data_filtered)

    if data_features.shape[1] < 2:
        warnings.warn("Need at least two numeric features to make a 2D PCA plot.")
        return

    # Normalize features and fit one shared PCA.
    X_scaled = StandardScaler().fit_transform(data_features)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Experiment": data_filtered[experiment_column].astype(str).values,
        }
    )
    # Optional PCA display orientation changes.
    # These affect only the saved PCA plot, not PCA fitting or clustering.
    if mirror_x:
        plot_df["PC1"] = -plot_df["PC1"]
    
    if mirror_y:
        plot_df["PC2"] = -plot_df["PC2"]
    
    if rotate_180:
        plot_df["PC1"] = -plot_df["PC1"]
        plot_df["PC2"] = -plot_df["PC2"]

    # Keep only experiments that survived filtering.
    present_experiments = [
        exp for exp in experiment_order
        if exp in set(plot_df["Experiment"])
    ]

    if len(present_experiments) == 0:
        warnings.warn("No selected experiments remain after filtering.")
        return

    # Color scheme similar to your attached figure.
    experiment_palette = {
        "WD": "#8B008B",   # dark purple
        "STV": "#A8DDE2",    # light cyan
        "CNT": "#D49AD0",   # pink/lavender
            
    }

    # Fallback colors for any future experiment names not listed above.
    missing_experiments = [
        exp for exp in present_experiments
        if exp not in experiment_palette
    ]

    fallback_colors = sns.color_palette(
        "tab10",
        n_colors=max(len(missing_experiments), 1),
    ).as_hex()

    for exp, color in zip(missing_experiments, fallback_colors):
        experiment_palette[exp] = color

    # Figure style similar to the attached plot:
    # white plotting area, black top legend area, black marker outlines.
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("white")

    for exp in present_experiments:
        exp_df = plot_df[plot_df["Experiment"] == exp]

        ax.scatter(
            exp_df["PC1"],
            exp_df["PC2"],
            s=point_size,
            c=experiment_palette[exp],
            edgecolors="black",
            linewidths=0.55,
            alpha=point_alpha,
            label=exp,
            rasterized=True,
        )

    if show_axes:
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
            fontsize=13,
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
            fontsize=13,
        )
        ax.tick_params(axis="both", labelsize=12)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=min(len(present_experiments), 4),
        frameon=False,
        fontsize=13,
        markerscale=1.8,
        handletextpad=0.4,
        columnspacing=1.5,
    )

    for text in legend.get_texts():
        text.set_color("white")

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.88,
    )

    plot_tag = "_".join(safe_name(exp) for exp in present_experiments)
    output_path = output_dir / f"selected_experiments_PCA_{plot_tag}.png"

    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )

    print(f"[INFO] Saved selected-experiment PCA plot to: {output_path}")

    if show_plots:
        plt.show()

    plt.close(fig)
    
    
    
# ----------------------------
# Main analysis function
# ----------------------------

def run_cluster_selection_for_experiment(
    data_subset: pd.DataFrame,
    experiment_name: str,
    output_dir: Path,
    min_pcs: int = MIN_PCS_TO_TEST,
    max_pcs: int = MAX_PCS_TO_TEST,
    max_clusters: int = MAX_CLUSTERS,
    cluster_selection_metric: str = "combined_score",
    pc_selection_metric: str = "ari_then_combined",
    mito_count_threshold: float = MITO_COUNT_THRESHOLD,
    n_stability_runs: int = N_STABILITY_RUNS,
    stability_subsample_fraction: float = STABILITY_SUBSAMPLE_FRACTION,
    ari_std_penalty_weight: float = ARI_STD_PENALTY_WEIGHT,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    n_init: int = N_INIT,
    min_samples: int = MIN_SAMPLES_PER_EXPERIMENT,
    show_plots: bool = False,
) -> Optional[Dict[str, object]]:
    """Run the complete PC/cluster-selection workflow for one experiment."""
    experiment_safe = safe_name(experiment_name)
    experiment_output_dir = output_dir / experiment_safe
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Rows before filtering: {len(data_subset)}")

    data_clean = clean_numeric_values(data_subset)
    data_filtered = filter_data(data_clean, mito_count_threshold=mito_count_threshold)
    print(f"Rows after filtering: {len(data_filtered)}")

    if len(data_filtered) < min_samples:
        warnings.warn(
            f"Skipping {experiment_name}: only {len(data_filtered)} rows after filtering; "
            f"minimum is {min_samples}."
        )
        return None

    data_features = get_feature_data(data_filtered)
    print(f"Numeric features used: {data_features.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_features)

    max_possible_pcs = min(max_pcs, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if max_possible_pcs < min_pcs:
        warnings.warn(
            f"Skipping {experiment_name}: max_possible_pcs={max_possible_pcs} is less than min_pcs={min_pcs}."
        )
        return None

    pca = PCA(n_components=max_possible_pcs, random_state=random_state)
    X_pca_all = pd.DataFrame(
        pca.fit_transform(X_scaled),
        columns=[f"PC{i}" for i in range(1, max_possible_pcs + 1)],
        index=data_filtered.index,
    )

    explained_variance = pd.DataFrame(
        {
            "Experiment": experiment_name,
            "PC": [f"PC{i}" for i in range(1, max_possible_pcs + 1)],
            "Explained_Variance_Ratio": pca.explained_variance_ratio_,
            "Cumulative_Explained_Variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained_variance_path = experiment_output_dir / f"{experiment_safe}_explained_variance.csv"
    explained_variance.to_csv(explained_variance_path, index=False)

    print("\nExplained variance by PCA component:")
    print(explained_variance)

    n_clusters_range = range(1, max_clusters + 1)
    pc_summary = []
    all_pc_metrics = []

    for n_pcs in range(min_pcs, max_possible_pcs + 1):
        pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
        X_pc = X_pca_all[pc_cols].copy()

        print(f"\nTesting PC set: PC1 to PC{n_pcs}")

        optimal_clusters, metrics_df = find_optimal_clusters_for_pc_set(
            X_pc,
            n_clusters_range,
            cluster_selection_metric=cluster_selection_metric,
            weights=WEIGHTS,
            cv_splits=cv_splits,
            random_state=random_state,
            n_jobs=n_jobs,
            n_init=n_init,
        )

        metrics_df.insert(0, "Experiment", experiment_name)
        metrics_df.insert(1, "n_pcs", n_pcs)
        metrics_df.insert(2, "PC_set", f"PC1-PC{n_pcs}")
        all_pc_metrics.append(metrics_df)

        row_for_optimal_clusters = metrics_df[metrics_df["n_clusters"] == optimal_clusters].iloc[0]

        stability = calculate_ari_stability(
            X_pc,
            n_clusters=optimal_clusters,
            n_runs=n_stability_runs,
            subsample_fraction=stability_subsample_fraction,
            random_state=random_state,
            n_init=n_init,
        )

        best_combined_score = row_for_optimal_clusters["combined_score"]
        ari_penalized_score = calculate_ari_penalized_score(
            stability["ARI_mean"],
            stability["ARI_std"],
            ari_std_penalty_weight=ari_std_penalty_weight,
            n_clusters=optimal_clusters,
        )
        cumulative_variance = float(np.sum(pca.explained_variance_ratio_[:n_pcs]))

        pc_summary.append(
            {
                "Experiment": experiment_name,
                "PC_set": f"PC1-PC{n_pcs}",
                "n_pcs": n_pcs,
                "optimal_clusters": optimal_clusters,
                "cluster_selection_metric": cluster_selection_metric,
                "best_combined_score": best_combined_score,
                "ARI_penalized_score": ari_penalized_score,
                "ARI_std_penalty_weight": ari_std_penalty_weight,
                "best_bic": row_for_optimal_clusters["bic"],
                "best_aic": row_for_optimal_clusters["aic"],
                "best_silhouette": row_for_optimal_clusters["silhouette"],
                "best_cv_score": row_for_optimal_clusters["cv_score"],
                "ARI_mean": stability["ARI_mean"],
                "ARI_std": stability["ARI_std"],
                "ARI_min": stability["ARI_min"],
                "ARI_max": stability["ARI_max"],
                "ARI_n_comparisons": stability["ARI_n_comparisons"],
                "cumulative_explained_variance": cumulative_variance,
            }
        )

        print(
            f"PC1-PC{n_pcs}: optimal clusters = {optimal_clusters}, "
            f"mean ARI = {stability['ARI_mean']:.3f}, "
            f"ARI penalty score = {ari_penalized_score:.3f}, "
            f"cumulative variance = {cumulative_variance:.3f}"
        )

    pc_summary_df = pd.DataFrame(pc_summary)
    all_pc_metrics_df = pd.concat(all_pc_metrics, ignore_index=True)

    pc_summary_path = experiment_output_dir / f"{experiment_safe}_pc_summary.csv"
    cluster_metrics_path = experiment_output_dir / f"{experiment_safe}_cluster_metrics_by_pc.csv"
    pc_summary_df.to_csv(pc_summary_path, index=False)
    all_pc_metrics_df.to_csv(cluster_metrics_path, index=False)

    print("\nSummary of PC-set clustering results:")
    print(pc_summary_df)

    best_pc_row = choose_final_pc_set(pc_summary_df, pc_selection_metric=pc_selection_metric)
    selected_n_pcs = int(best_pc_row["n_pcs"])
    selected_clusters = int(best_pc_row["optimal_clusters"])

    print("\nSelected final model:")
    print(f"Experiment: {experiment_name}")
    print(f"PC set: PC1-PC{selected_n_pcs}")
    print(f"Optimal clusters: {selected_clusters}")
    print(f"Mean ARI stability: {best_pc_row['ARI_mean']:.3f}")
    print(f"ARI std: {best_pc_row['ARI_std']:.3f}")
    print(f"ARI penalized score: {best_pc_row.get('ARI_penalized_score', np.nan):.3f}")
    print(f"Cumulative explained variance: {best_pc_row['cumulative_explained_variance']:.3f}")

    selected_pc_cols = [f"PC{i}" for i in range(1, selected_n_pcs + 1)]
    X_selected = X_pca_all[selected_pc_cols].copy()

    gmm_final = GaussianMixture(
        n_components=selected_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    final_labels = gmm_final.fit_predict(X_selected)

    metadata_columns = [
        col
        for col in ["Experiment", "Type", GROUP_COLUMN, LABEL_COLUMN, "stack_id", "cell_id_linked"]
        if col in data_filtered.columns
    ]
    assignments = data_filtered[metadata_columns].copy()
    assignments["Original_Index"] = data_filtered.index
    assignments["Selected_n_pcs"] = selected_n_pcs
    assignments["Selected_n_clusters"] = selected_clusters
    assignments["Cluster"] = final_labels
    assignments = pd.concat([assignments, X_pca_all], axis=1)

    assignments_path = experiment_output_dir / f"{experiment_safe}_final_cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    ari_plot_path = experiment_output_dir / f"{experiment_safe}_PC_ARI_stability_barplot.png"
    plot_pc_ari_bars(
        pc_summary_df,
        modality=str(experiment_name),
        output_path=ari_plot_path,
        show_plots=show_plots,
    )

    cluster_plot_path = experiment_output_dir / f"{experiment_safe}_final_clusters_PC1_PC2.png"
    plot_final_clusters(
        X_pca_all,
        final_labels,
        selected_n_pcs,
        selected_clusters,
        str(experiment_name),
        output_path=cluster_plot_path,
        show_plots=show_plots,
    )

    return {
        "Experiment": experiment_name,
        "rows_before_filtering": len(data_subset),
        "rows_after_filtering": len(data_filtered),
        "n_features": data_features.shape[1],
        "selected_n_pcs": selected_n_pcs,
        "selected_clusters": selected_clusters,
        "selected_ARI_mean": best_pc_row["ARI_mean"],
        "selected_ARI_std": best_pc_row["ARI_std"],
        "selected_ARI_penalized_score": best_pc_row.get("ARI_penalized_score", np.nan),
        "selected_best_combined_score": best_pc_row["best_combined_score"],
        "selected_cumulative_explained_variance": best_pc_row["cumulative_explained_variance"],
        "pc_summary_path": str(pc_summary_path),
        "cluster_metrics_path": str(cluster_metrics_path),
        "assignments_path": str(assignments_path),
        "ari_plot_path": str(ari_plot_path),
        "cluster_plot_path": str(cluster_plot_path),
    }


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optimal PC and GMM cluster selection per experiment/group."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to input CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument("--label-column", default=LABEL_COLUMN, help="Column containing labels.")
    parser.add_argument("--group-column", default=GROUP_COLUMN, help="Column containing group names, if present.")
    parser.add_argument(
        "--group-source",
        choices=["auto", "group", "labels"],
        default="auto",
        help="Use existing group column, labels column, or auto-detect. Use 'labels' for this uploaded CSV.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment names to run, e.g. CNT STV WD. Omit to run all detected experiments.",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Run one pooled analysis across all rows instead of one analysis per experiment.",
    )
    parser.add_argument("--min-pcs", type=int, default=MIN_PCS_TO_TEST)
    parser.add_argument("--max-pcs", type=int, default=MAX_PCS_TO_TEST)
    parser.add_argument("--max-clusters", type=int, default=MAX_CLUSTERS)
    parser.add_argument(
        "--cluster-selection-metric",
        choices=[
            "combined_score",
            "bic",
            "aic",
            "silhouette",
            "cv_score",
            "log_likelihood",
        ],
        default="combined_score",
        help="Metric used to choose optimal clusters for each PC set.",
    )
    parser.add_argument(
        "--pc-selection-metric",
        choices=[
            "ari_penalized_then_combined",
            "ari_then_combined",
            "combined_then_ari",
            "ARI_penalized_score",
            "ARI_mean",
            "best_combined_score",
        ],
        default="ari_penalized_then_combined",
        help="Metric used to choose the final PC set.",
    )
    parser.add_argument("--mito-count-threshold", type=float, default=MITO_COUNT_THRESHOLD)
    parser.add_argument("--stability-runs", type=int, default=N_STABILITY_RUNS)
    parser.add_argument("--stability-subsample-fraction", type=float, default=STABILITY_SUBSAMPLE_FRACTION)
    parser.add_argument("--ari-std-penalty-weight", type=float, default=ARI_STD_PENALTY_WEIGHT)
    parser.add_argument("--cv-splits", type=int, default=CV_SPLITS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_EXPERIMENT)
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def _normalize_experiments_setting(value):
    """Return None, or a clean list of experiment names from the no-Bash setting."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip() != ""]


def get_script_settings() -> argparse.Namespace:
    """Build argparse-like settings from the editable no-Bash variables above."""
    return argparse.Namespace(
        data=Path(SCRIPT_DATA_PATH),
        output_dir=Path(SCRIPT_OUTPUT_DIR),
        label_column=SCRIPT_LABEL_COLUMN,
        group_column=SCRIPT_GROUP_COLUMN,
        group_source=SCRIPT_GROUP_SOURCE,
        experiments=_normalize_experiments_setting(SCRIPT_EXPERIMENTS_TO_RUN),
        pooled=SCRIPT_POOLED,
        min_pcs=SCRIPT_MIN_PCS,
        max_pcs=SCRIPT_MAX_PCS,
        max_clusters=SCRIPT_MAX_CLUSTERS,
        cluster_selection_metric=SCRIPT_CLUSTER_SELECTION_METRIC,
        pc_selection_metric=SCRIPT_PC_SELECTION_METRIC,
        mito_count_threshold=SCRIPT_MITO_COUNT_THRESHOLD,
        stability_runs=SCRIPT_STABILITY_RUNS,
        stability_subsample_fraction=SCRIPT_STABILITY_SUBSAMPLE_FRACTION,
        ari_std_penalty_weight=SCRIPT_ARI_STD_PENALTY_WEIGHT,
        cv_splits=SCRIPT_CV_SPLITS,
        random_state=SCRIPT_RANDOM_STATE,
        jobs=SCRIPT_JOBS,
        n_init=SCRIPT_N_INIT,
        min_samples=SCRIPT_MIN_SAMPLES,
        show_plots=SCRIPT_SHOW_PLOTS,
    )


def main() -> None:
    if USE_SCRIPT_SETTINGS:
        args = get_script_settings()
    else:
        args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        raise FileNotFoundError(f"Could not find input CSV: {args.data}")

    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data, na_values="-")

    data = add_type_and_experiment_columns(
        data,
        label_column=args.label_column,
        group_column=args.group_column,
        group_source=args.group_source,
    )

    experiment_counts = data["Experiment"].value_counts(dropna=False).sort_index()
    print("\nDetected experiments:")
    print(experiment_counts)

    if args.pooled:
        experiments_to_run = ["Pooled"]
        subsets = {"Pooled": data}
    else:
        all_experiments = [str(x) for x in sorted(data["Experiment"].dropna().unique())]
        if args.experiments:
            requested = set(args.experiments)
            missing = sorted(requested - set(all_experiments))
            if missing:
                warnings.warn(f"Requested experiments were not found and will be skipped: {missing}")
            experiments_to_run = [exp for exp in all_experiments if exp in requested]
        else:
            experiments_to_run = all_experiments

        subsets = {
            exp: data[data["Experiment"].astype(str) == exp].copy()
            for exp in experiments_to_run
        }
        
    
        plot_selected_experiments_pca(
        data=data,
        experiments_to_plot=experiments_to_run,
        output_dir=args.output_dir,
        experiment_column="Experiment",
        mito_count_threshold=args.mito_count_threshold,
        random_state=args.random_state,
        show_plots=args.show_plots,
        show_axes=False,
        exclude_cell_id_linked_zero=SCRIPT_SELECTED_PCA_EXCLUDE_CELL_ID_LINKED_ZERO,
        cell_id_linked_column="cell_id_linked",
        mirror_x=SCRIPT_SELECTED_PCA_MIRROR_X,
        mirror_y=SCRIPT_SELECTED_PCA_MIRROR_Y,
        rotate_180=SCRIPT_SELECTED_PCA_ROTATE_180
    )

    overall_rows = []    
        
        
    for experiment_name in experiments_to_run:
        result = run_cluster_selection_for_experiment(
            subsets[experiment_name],
            experiment_name=experiment_name,
            output_dir=args.output_dir,
            min_pcs=args.min_pcs,
            max_pcs=args.max_pcs,
            max_clusters=args.max_clusters,
            cluster_selection_metric=args.cluster_selection_metric,
            pc_selection_metric=args.pc_selection_metric,
            mito_count_threshold=args.mito_count_threshold,
            n_stability_runs=args.stability_runs,
            stability_subsample_fraction=args.stability_subsample_fraction,
            ari_std_penalty_weight=args.ari_std_penalty_weight,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
            n_jobs=args.jobs,
            n_init=args.n_init,
            min_samples=args.min_samples,
            show_plots=args.show_plots,
        )
        if result is not None:
            overall_rows.append(result)

    if len(overall_rows) > 0:
        overall_summary_df = pd.DataFrame(overall_rows)
        overall_summary_path = args.output_dir / "overall_experiment_cluster_selection_summary.csv"
        overall_summary_df.to_csv(overall_summary_path, index=False)

        print("\n" + "=" * 80)
        print("Overall summary:")
        print(overall_summary_df)
        print(f"\n[INFO] Saved overall summary to: {overall_summary_path}")
    else:
        print("\nNo experiments were successfully analyzed.")


if __name__ == "__main__":
    main()
