"""
Figure 4 PCA/GMM workflow for a combined CNT/STV/WD CSV.

"""

from pathlib import Path
import ast
import copy
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



sns.set()
plt.rcParams["figure.dpi"] = 500


# -----------------------------------------------------------------------------
# Editable settings
# -----------------------------------------------------------------------------
# Use your combined CSV containing CNT, STV, and WD.
# Example:
INPUT_CSV = Path(r"...add path...\Full_data_perturbations.csv")


# Choose the experiment(s) to run.
# Examples:
#   EXPERIMENTS_TO_RUN = ["CNT"]
#   EXPERIMENTS_TO_RUN = ["STV"]
#   EXPERIMENTS_TO_RUN = ["WD"]
#   EXPERIMENTS_TO_RUN = ["CNT", "STV", "WD"] Use all experiments when generating combineHD_pittsburgh_indices.csv
EXPERIMENTS_TO_RUN = ["CNT", "STV", "WD"]

# How to identify the experiment in the combined CSV.
#   "labels" = derive CNT/STV/WD from the labels column, e.g. CNT_, STV_, WD_
#   "group"  = use an existing group column
#   "auto"   = use group if present, otherwise labels
EXPERIMENT_SOURCE = "labels"
LABEL_COLUMN = "labels"
GROUP_COLUMN = "group"

# Manual settings from the optimal-PC/cluster-selection code.
# Change these numbers to the selected values from the other script.
EXPERIMENT_SETTINGS = {
    "CNT": {"n_pcs": 2, "n_clusters": 5},
    "STV": {"n_pcs": 2, "n_clusters": 4},
    "WD":  {"n_pcs": 2, "n_clusters": 2},
}

OUTPUT_ROOT_DIR = INPUT_CSV.parent / "Full_Concat_data_by_experiment"

HUE_COLUMN = "Prediction"
MIN_MITO_COUNT = 20
PCA_RANDOM_STATE = 42
GMM_RANDOM_STATE = 54
MIN_MITO_DENSITY = 0.1

#Adjust per experiment if needed.
PLOT_XLIM = (-15, 20)
PLOT_YLIM = (-11.5, 20)
FIGSIZE = (9, 9)

# Keep True if you run this as a script and want the figures to appear.
# In Jupyter, the figures will also display naturally.
SHOW_FIGURES = True

# Set True if you also want image files saved in OUTPUT_PLOTS_DIR.
SAVE_FIGURES = True

# Random plotting subset only. This does not change PCA, GMM, probabilities,
# cluster labels, or saved full tables.
PLOT_FRACTION = 1.0
PLOT_RANDOM_STATE = 42

# Automatically mirror the displayed PC1 axis so the ordered categories go from
# category 1 on the left to the last category on the right.
# This changes the orientation of the saved plots only; PCA/GMM fitting stays the same.
ORIENT_PC1_BY_CATEGORY_ORDER = True
ORIENTATION_CATEGORY_COLUMN = HUE_COLUMN
SAVE_ORIENTED_FULL_CONCAT = True

# -----------------------------------------------------------------------------
# Pittsburgh H/D export settings
# -----------------------------------------------------------------------------
# This creates one consolidated CSV that can be used by the Pittsburgh
# heterogeneity plotting script.

MAKE_PITTSBURGH_INDEX_CSV = True

# Use all three for the consolidated H/D file.
PITTSBURGH_EXPERIMENTS_TO_EXPORT = ["CNT", "STV", "WD"]

# These prefixes match your Pittsburgh plotting script:
# C_H, F_H, W_H and C_D, F_D, W_D.
# Here F is used for STV so the old plotting code keeps working.
PITTSBURGH_PREFIX_MAP = {
    "CNT": "C",
    "STV": "F",
    "WD": "W",
}

PITTSBURGH_POSITION_COLUMN = "ascini_position"
PITTSBURGH_CATEGORY_COLUMN = HUE_COLUMN


PITTSBURGH_N_BINS = 60

# ascini_position is expected to be scaled from -1 (CV) to 1 (PV).
PITTSBURGH_POSITION_RANGE =  (-0.95, 0.95) 

PITTSBURGH_WIDE_OUTPUT_NAME = "combineHD_pittsburgh_indices.csv"
PITTSBURGH_LONG_OUTPUT_NAME = "pittsburgh_indices_long.csv"



# Label prefixes used when EXPERIMENT_SOURCE = "labels".
LABEL_PREFIX_TO_EXPERIMENT = {
    "CNT_": "CNT",
    "STV_": "STV",
    "WD_": "WD",
}

# Experiment-specific palettes for ordered categories 1..N.
# CNT is the original control palette from your current script.
# STV and WD were sampled from the attached example thumbnails.
# Cluster/category 1 is intentionally the darkest color for STV and WD.
EXPERIMENT_CLUSTER_PALETTES = {
    "CNT": [
        "#fcfbfd",  # category 1, original control palette
        "#9e9ac8",
        "#6950a3",
        "#dadaeb",
        "#3f007d",  # category 5 / last category
    ],
    "STV": [
        "#0C2A56",  # category 1, darkest blue
        "#92A7B6",
        "white",  
        "#3D7097",
       
    ],
    "WD": [
        "#262026",  # dark charcoal plum
        "#4A334A",  # dark plum
        "#693D69",  # muted deep purple
        
    ],
    
    
}

# ----------------------------------------------------------------------------
# Columns excluded from the PCA feature matrix
# ----------------------------------------------------------------------------
EXCLUDE_FROM_PCA = [
    "Unnamed: 0",
    "label",
    "area",
    "centroid-0",
    "centroid-1",
    "type_1_ld_avg_aspect_ratio",
    "type_2_ld_avg_aspect_ratio",
    "type_3_ld_avg_aspect_ratio",
    "type_4_ld_avg_aspect_ratio",
    "type_1_ld_avg_solidity",
    "type_2_ld_avg_solidity",
    "type_3_ld_avg_solidity",
    "type_4_ld_avg_solidity",
    "ascini_position",
    "labels",
    "stack_id",
    "cell_id_linked",
    "group",
    "Experiment",
    "Type",
    "Prediction",
    "Prediction_original",
    "Probability",
]


# -----------------------------------------------------------------------------
# Label / experiment helpers
# -----------------------------------------------------------------------------
def safe_name(value) -> str:
    """Make a string safe for filenames and folder names."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unnamed"


def parse_labels_cell(value) -> list[str]:
    """
    Convert one value from the labels column into a list of label strings.

    Supports strings that look like Python lists, plain strings, and actual
    list/tuple/array-like values.
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


def label_to_experiment(label_value) -> str:
    """Derive CNT/STV/WD from the labels column."""
    labels = parse_labels_cell(label_value)
    if len(labels) == 0:
        return "Unknown"

    for label in labels:
        for prefix, experiment in LABEL_PREFIX_TO_EXPERIMENT.items():
            if label.startswith(prefix) or prefix in label:
                return experiment

    # Fallback: use text before the first underscore.
    first_label = labels[0]
    match = re.match(r"^([^_]+)_", first_label)
    if match:
        return match.group(1)

    return "Unknown"


def add_experiment_column(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Add an Experiment column using labels, group, or auto-detection."""
    data = raw_data.copy()

    if EXPERIMENT_SOURCE not in {"labels", "group", "auto"}:
        raise ValueError("EXPERIMENT_SOURCE must be 'labels', 'group', or 'auto'.")

    if EXPERIMENT_SOURCE == "group":
        if GROUP_COLUMN not in data.columns:
            raise KeyError(f"Column '{GROUP_COLUMN}' was not found in the CSV.")
        data["Experiment"] = data[GROUP_COLUMN].astype(str)

    elif EXPERIMENT_SOURCE == "auto" and GROUP_COLUMN in data.columns:
        data["Experiment"] = data[GROUP_COLUMN].astype(str)

    else:
        if LABEL_COLUMN not in data.columns:
            raise KeyError(f"Column '{LABEL_COLUMN}' was not found in the CSV.")
        data["Experiment"] = data[LABEL_COLUMN].apply(label_to_experiment)

    return data


def get_experiment_settings(experiment_name: str) -> tuple[int, int]:
    """Return manually selected n_pcs and n_clusters for one experiment."""
    if experiment_name not in EXPERIMENT_SETTINGS:
        raise KeyError(
            f"No manual PC/cluster settings found for experiment '{experiment_name}'. "
            "Add it to EXPERIMENT_SETTINGS."
        )

    n_pcs = int(EXPERIMENT_SETTINGS[experiment_name]["n_pcs"])
    n_clusters = int(EXPERIMENT_SETTINGS[experiment_name]["n_clusters"])

    if n_pcs < 2:
        raise ValueError("n_pcs must be at least 2 because the plots use PC1 and PC2.")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    return n_pcs, n_clusters


def get_experiment_cluster_colors(experiment_name: str, n_clusters: int) -> list[str]:
    """
    Return ordered colors for categories 1..n_clusters.

    If the requested number of clusters differs from the number of fixed colors,
    colors are interpolated through the experiment-specific palette.
    """
    base_colors = EXPERIMENT_CLUSTER_PALETTES.get(
        experiment_name,
        sns.color_palette("tab10", n_colors=max(n_clusters, 1)).as_hex(),
    )

    if len(base_colors) == n_clusters:
        return list(base_colors)

    return sns.blend_palette(base_colors, n_colors=n_clusters).as_hex()


def make_ordered_prediction_palette(
    n_clusters: int,
    experiment_name: str,
):
    """
    Return a palette and hue order for remapped categories 1..N.

    This is used after remap_prediction_by_ascini_position(), so the prediction
    plot legend and colors follow the new ordered category numbers.
    """
    colors = get_experiment_cluster_colors(experiment_name, n_clusters)
    categories = list(range(1, n_clusters + 1))
    prediction_palette = dict(zip(categories, colors))
    prediction_hue_order = categories
    return prediction_palette, prediction_hue_order


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_combined_csv(input_csv: Path) -> pd.DataFrame:
    """Load the combined CSV and add the Experiment column."""
    raw_data = pd.read_csv(input_csv, na_values="-")
    raw_data = add_experiment_column(raw_data)
    return raw_data


def load_and_filter_data_for_experiment(
    all_data: pd.DataFrame,
    experiment_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one experiment and apply the same cleaning/filtering as before."""
    raw_data = all_data[all_data["Experiment"].astype(str) == str(experiment_name)].copy()

    if raw_data.empty:
        detected = sorted(all_data["Experiment"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows found for experiment '{experiment_name}'. Detected experiments: {detected}"
        )

    print("\n" + "=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Rows before cleaning/filtering: {len(raw_data)}")

    # Same order as your notebook code, after experiment selection.
    raw_data.fillna(0, inplace=True)
    raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_data.dropna(inplace=True)

    # -------------------------------------------------------------------------
    # Remove rows where cell_id_linked is 0
    # -------------------------------------------------------------------------
    if "cell_id_linked" not in raw_data.columns:
        raise KeyError("Column 'cell_id_linked' was not found in the input CSV.")

    raw_data["cell_id_linked"] = pd.to_numeric(
        raw_data["cell_id_linked"],
        errors="coerce",
    )

    before_cell_id_filter = len(raw_data)
    raw_data = raw_data[raw_data["cell_id_linked"] != 0].copy()
    after_cell_id_filter = len(raw_data)

    print(
        "Removed rows with cell_id_linked == 0:",
        before_cell_id_filter - after_cell_id_filter,
    )

    # -------------------------------------------------------------------------
    # Existing mitochondrial filters
    # -------------------------------------------------------------------------
    required_columns = ["mito_aspect_ratio", "mito_density", "area"]
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        raise KeyError(f"Missing required filter columns: {missing_columns}")

    mito_positive = raw_data[raw_data["mito_aspect_ratio"] > 0].copy()

    mito_density_filtered = mito_positive[
        mito_positive["mito_density"] >= MIN_MITO_DENSITY
    ].copy()

    mito_count = mito_density_filtered["mito_density"] * mito_density_filtered["area"]
    clean_data = mito_density_filtered[mito_count >= MIN_MITO_COUNT].copy()

    print(f"Rows after filters: {len(clean_data)}")

    return raw_data, clean_data


def build_pca_feature_matrix(clean_data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Drop excluded columns and return the feature matrix used for PCA."""
    pca_features = clean_data.drop(EXCLUDE_FROM_PCA, axis=1, errors="ignore")

  
    non_numeric_columns = pca_features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "These non-numeric columns are still present in the PCA feature matrix: "
            f"{non_numeric_columns}. Add them to EXCLUDE_FROM_PCA or convert them to numbers."
        )

    # Drop constant columns; StandardScaler can handle them, but they add no information.
    nunique = pca_features.nunique(dropna=False)
    constant_columns = nunique[nunique <= 1].index.tolist()
    if constant_columns:
        pca_features = pca_features.drop(columns=constant_columns)

    if pca_features.shape[1] == 0:
        raise ValueError("No numeric PCA feature columns remain after exclusions.")

    return np.array(pca_features), pca_features.columns.tolist()


def run_pca(feature_matrix: np.ndarray) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """Z-score the feature matrix and calculate all PCA components."""
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)

    # n_components=None keeps all possible components, as in your latest code.
    pca = PCA(n_components=None, random_state=PCA_RANDOM_STATE)
    pca_data = pca.fit_transform(normalized_features)

    pca_columns = [f"component_{i + 1}" for i in range(pca_data.shape[1])]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns)

    return pca_df, pca, normalized_features


# -----------------------------------------------------------------------------
# GMM
# -----------------------------------------------------------------------------
def fit_gmm_on_selected_pcs(
    pca_df: pd.DataFrame,
    n_pcs: int,
    n_clusters: int,
):
    """
    Fit the GMM using component_1 through component_N.

    The scatter plots still display component_1 vs component_2, but the GMM
    labels/probabilities are calculated from all selected PCs.
    """
    selected_columns = [f"component_{i}" for i in range(1, n_pcs + 1)]
    missing_columns = [col for col in selected_columns if col not in pca_df.columns]
    if missing_columns:
        raise ValueError(
            f"The requested n_pcs={n_pcs} requires columns that do not exist: {missing_columns}"
        )

    gmm_input = pca_df[selected_columns].copy()

    gmm = mixture.GaussianMixture(
        n_components=n_clusters,
        random_state=GMM_RANDOM_STATE,
        covariance_type="full",
        max_iter=100,
        tol=1e-3,
    ).fit(gmm_input)

    labels = gmm.predict(gmm_input)
    probabilities = gmm.predict_proba(gmm_input)

    return gmm, gmm_input, labels, probabilities, selected_columns


def make_gmm_plot_input(pca_df: pd.DataFrame) -> pd.DataFrame:
    """Return PC1/PC2 coordinates used for all scatter plots."""
    return pca_df[["component_1", "component_2"]].copy()


def save_gmm_parameters(gmm, output_dir: Path, gmm_name: str) -> None:
    """Save GMM weights, means, and covariance matrices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("the weights are :", gmm.weights_)
    np.save(output_dir / f"{gmm_name}_weights.npy", gmm.weights_, allow_pickle=False)
    np.save(output_dir / f"{gmm_name}_means.npy", gmm.means_, allow_pickle=False)
    np.save(output_dir / f"{gmm_name}_covariances.npy", gmm.covariances_, allow_pickle=False)


# -----------------------------------------------------------------------------
# Output tables
# -----------------------------------------------------------------------------
def save_tables(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    pca_df: pd.DataFrame,
    pca_feature_columns: list[str],
    pca,
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    experiment_name: str,
    n_pcs: int,
    n_clusters: int,
) -> pd.DataFrame:
    """Save the same main CSVs plus useful PCA documentation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data.to_csv(output_dir / "Data_including_bad_cells.csv", index=False)
    clean_data.to_csv(output_dir / "Clean_Data.csv", index=False)

    pca_df = pca_df.reset_index(drop=True)
    pca_df.to_csv(output_dir / "PCA.csv", index=False)

    pd.Series(pca_feature_columns, name="PCA_feature_column").to_csv(
        output_dir / "PCA_feature_columns.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "component": [f"component_{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(output_dir / "PCA_explained_variance_ratio.csv", index=False)

    # Keep the original probability column style: 0, 1, 2, 3, ...
    gmm_df = pd.DataFrame(probabilities)
    gmm_df["Probability"] = probabilities.max(axis=1)
    gmm_df["Prediction"] = labels
    gmm_df["Experiment_Selected"] = experiment_name
    gmm_df["GMM_n_pcs"] = n_pcs
    gmm_df["GMM_n_clusters"] = n_clusters

    gmm_and_pca_df = pd.concat([gmm_df, pca_df], axis="columns")

    # Preserve original row identity, but reset index for clean concatenation.
    clean_data_reset = clean_data.reset_index(drop=False).rename(
        columns={"index": "Original_Index"}
    )

    full_concat = pd.concat(
        [gmm_and_pca_df.reset_index(drop=True), clean_data_reset.reset_index(drop=True)],
        axis="columns",
    )

    output_file = output_dir / "FULL_CONCAT.csv"
    full_concat.to_csv(output_file, index=False)

    return pd.read_csv(output_file)


# -----------------------------------------------------------------------------
# Plotting helpers 
# -----------------------------------------------------------------------------
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance."""
    ax = ax or plt.gca()

    covariance = np.asarray(covariance)
    position = np.asarray(position)

    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(
                position,
                nsig * width,
                nsig * height,
                angle=angle,
                **kwargs,
            )
        )


def get_gmm_display_means_and_covariances(gmm):
    """
    Return GMM means/covariances projected to PC1 and PC2 for ellipse plotting.

    This allows GMM fitting on PC1..PCN while preserving PC1/PC2 plots.
    """
    means_2d = np.asarray(gmm.means_)[:, :2]
    covariance_type = getattr(gmm, "covariance_type", "full")

    if covariance_type == "full":
        covariances_2d = np.asarray(gmm.covariances_)[:, :2, :2]
    elif covariance_type == "tied":
        cov_2d = np.asarray(gmm.covariances_)[:2, :2]
        covariances_2d = np.repeat(cov_2d[None, :, :], gmm.n_components, axis=0)
    elif covariance_type == "diag":
        covariances_2d = np.array(
            [np.diag(row[:2]) for row in np.asarray(gmm.covariances_)]
        )
    elif covariance_type == "spherical":
        covariances_2d = np.array(
            [np.eye(2) * value for value in np.asarray(gmm.covariances_)]
        )
    else:
        raise ValueError(f"Unsupported GMM covariance_type: {covariance_type}")

    return means_2d, covariances_2d


def plot_gmm_same_as_before(
    gmm,
    gmm_plot_input: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: Path,
    prediction_palette=None,
    labels_override=None,
    plot_xlim=None,
    label=True,
):
    """
    Recreate the original active GMM ellipse plot.

    If GMM was fit on more than two PCs, labels must be passed with
    labels_override. Ellipses are drawn from the PC1/PC2 slice of the GMM.
    """
    if labels_override is None:
        if gmm.means_.shape[1] != gmm_plot_input.shape[1]:
            raise ValueError(
                "labels_override is required when the GMM was fit on more PCs "
                "than the two plotted PCs."
            )
        labels = gmm.predict(gmm_plot_input)
    else:
        labels = np.asarray(labels_override)

    size = 5 * probabilities.max(1) ** 3
    xlim_to_use = PLOT_XLIM if plot_xlim is None else plot_xlim

    plt.figure(figsize=FIGSIZE)
    plt.xlim(*xlim_to_use)
    plt.ylim(*PLOT_YLIM)

    if label:
        if prediction_palette is not None:
            point_colors = [prediction_palette.get(int(label_value), "#808080") for label_value in labels]
            plt.scatter(
                gmm_plot_input["component_1"],
                gmm_plot_input["component_2"],
                c=point_colors,
                linewidths=0,
                s=40,
                zorder=1,
            )
        else:
            plt.scatter(
                gmm_plot_input["component_1"],
                gmm_plot_input["component_2"],
                c=labels,
                linewidths=0,
                s=40,
                cmap="coolwarm_r",
                zorder=1,
            )
    else:
        plt.scatter(
            gmm_plot_input["component_1"],
            gmm_plot_input["component_2"],
            linewidths=0,
            edgecolor="black",
            s=size,
            zorder=1,
        )

    means_2d, covariances_2d = get_gmm_display_means_and_covariances(gmm)
    w_factor = 0.1 / gmm.weights_.max()
    for pos, covar, w in zip(means_2d, covariances_2d, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

    if SAVE_FIGURES:
        plt.savefig(output_dir / "Figure4_PCA_GMM_ellipses_same_style.png")
        plt.savefig(output_dir / "Figure4_PCA_GMM_ellipses_same_style.svg")


def plot_probability_same_as_before(
    full_df: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: Path,
    plot_xlim=None,
):
    """Recreate the probability-colored PCA scatterplot from the latest code."""
    size = 1 * probabilities.max(1) ** 4
    max_probability = probabilities.max(1)
    xlim_to_use = PLOT_XLIM if plot_xlim is None else plot_xlim

    plt.figure(figsize=FIGSIZE)
    plt.ylim(*PLOT_YLIM)
    plt.xlim(*xlim_to_use)

    sns.scatterplot(
        x="component_1",
        y="component_2",
        s=100,
        data=full_df,
        hue=max_probability,
        size=size,
        sizes=(30, 100),
        legend="brief",
        edgecolor="black",
        palette="magma",
    )

    if SAVE_FIGURES:
        plt.savefig(output_dir / "Figure4_PCA_GMM_probability_same_style.png")
        plt.savefig(output_dir / "Figure4_PCA_GMM_probability_same_style.svg")


def plot_ascini_position_same_as_before(
    full_df: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: Path,
    position_column: str = "ascini_position",
    palette: str = "rocket",
    vmin: float = -1,
    vmax: float = 1,
    plot_xlim=None,
):
    """
    PCA scatter plot colored by ascini_position using the rocket color scheme.

    This does not refit PCA or GMM.
    It uses the existing component_1 and component_2 coordinates.
    """
    required_columns = ["component_1", "component_2", position_column]
    missing_columns = [col for col in required_columns if col not in full_df.columns]

    if missing_columns:
        raise KeyError(f"These columns are missing from full_df: {missing_columns}")

    plot_df = full_df.copy()

    for col in required_columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df["_max_probability"] = probabilities.max(axis=1)
    plot_df["_point_size"] = 15 + 60 * (plot_df["_max_probability"] ** 4)

    plot_df = plot_df.dropna(subset=["component_1", "component_2", position_column]).copy()

    cmap = sns.color_palette(palette, as_cmap=True)
    xlim_to_use = PLOT_XLIM if plot_xlim is None else plot_xlim

    fig, ax = plt.subplots(figsize=FIGSIZE)

    scatter = ax.scatter(
        plot_df["component_1"],
        plot_df["component_2"],
        c=plot_df[position_column],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=plot_df["_point_size"],
        edgecolors="black",
        linewidths=0.35,
        alpha=0.95,
        rasterized=True,
    )

    ax.set_xlim(*xlim_to_use)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_xlabel("component_1")
    ax.set_ylabel("component_2")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(position_column)

    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(output_dir / "Figure4_PCA_ascini_position_rocket_same_style.png", dpi=500)
        plt.savefig(output_dir / "Figure4_PCA_ascini_position_rocket_same_style.svg", dpi=500)


def copy_gmm_with_pc1_mirrored(gmm):
    """
    Make a display-only copy of a fitted GMM after PC1 is mirrored.

    The fitted GMM is not changed. This copy is only used so ellipse positions
    and shapes match the mirrored PCA coordinates in the saved figures.
    """
    mirrored_gmm = copy.deepcopy(gmm)

    if hasattr(mirrored_gmm, "means_"):
        mirrored_gmm.means_ = np.array(mirrored_gmm.means_, copy=True)
        mirrored_gmm.means_[:, 0] = -mirrored_gmm.means_[:, 0]

    if hasattr(mirrored_gmm, "covariances_"):
        mirrored_gmm.covariances_ = np.array(mirrored_gmm.covariances_, copy=True)

        if getattr(mirrored_gmm, "covariance_type", None) == "full":
            # Reflecting PC1 changes the sign of all covariance terms involving PC1.
            mirrored_gmm.covariances_[:, 0, :] *= -1
            mirrored_gmm.covariances_[:, :, 0] *= -1

        elif getattr(mirrored_gmm, "covariance_type", None) == "tied":
            mirrored_gmm.covariances_[0, :] *= -1
            mirrored_gmm.covariances_[:, 0] *= -1

        # For diag and spherical covariance, reflection does not change variances.

    return mirrored_gmm


def orient_pc1_by_category_order(
    full_df: pd.DataFrame,
    gmm_plot_input: pd.DataFrame,
    gmm,
    output_dir: Path,
    cluster_mapping=None,
    category_column: str = HUE_COLUMN,
    plot_xlim: tuple[float, float] = PLOT_XLIM,
    auto_orient: bool = ORIENT_PC1_BY_CATEGORY_ORDER,
    save_outputs: bool = SAVE_ORIENTED_FULL_CONCAT,
):
    """
    Mirror PC1, if needed, so category 1 appears to the left of the last category.

    This is a display-orientation step. It does not refit PCA, GMM, probabilities,
    or cluster assignments. It only changes the PC1 sign used in the saved plots.
    """
    if category_column not in full_df.columns:
        raise KeyError(f"Column '{category_column}' was not found in full_df.")

    oriented_full_df = full_df.copy()
    oriented_gmm_plot_input = gmm_plot_input.copy()
    oriented_gmm = gmm
    oriented_cluster_mapping = None if cluster_mapping is None else cluster_mapping.copy()
    oriented_xlim = plot_xlim
    flipped_pc1 = False

    if not auto_orient:
        return (
            oriented_full_df,
            oriented_gmm_plot_input,
            oriented_gmm,
            oriented_cluster_mapping,
            flipped_pc1,
            oriented_xlim,
        )

    categories_numeric = pd.to_numeric(oriented_full_df[category_column], errors="coerce")
    valid_categories = sorted(categories_numeric.dropna().astype(int).unique().tolist())

    if len(valid_categories) < 2:
        print(
            "[INFO] PC1 orientation skipped because fewer than two categories "
            f"were found in '{category_column}'."
        )
        return (
            oriented_full_df,
            oriented_gmm_plot_input,
            oriented_gmm,
            oriented_cluster_mapping,
            flipped_pc1,
            oriented_xlim,
        )

    first_category = valid_categories[0]
    last_category = valid_categories[-1]

    category_means_before = (
        oriented_full_df.assign(_category_for_orientation=categories_numeric)
        .dropna(subset=["_category_for_orientation", "component_1"])
        .groupby("_category_for_orientation")["component_1"]
        .mean()
        .sort_index()
    )

    first_pc1 = category_means_before.loc[first_category]
    last_pc1 = category_means_before.loc[last_category]

    # If category 1 is to the right of the last category, mirror PC1.
    flipped_pc1 = bool(first_pc1 > last_pc1)

    if flipped_pc1:
        oriented_full_df["component_1"] = -pd.to_numeric(
            oriented_full_df["component_1"],
            errors="coerce",
        )
        oriented_gmm_plot_input["component_1"] = -pd.to_numeric(
            oriented_gmm_plot_input["component_1"],
            errors="coerce",
        )
        oriented_gmm = copy_gmm_with_pc1_mirrored(gmm)
        oriented_xlim = (-plot_xlim[1], -plot_xlim[0])

        if oriented_cluster_mapping is not None:
            for pc1_col in ["centroid_component_1", "centroid_PC1"]:
                if pc1_col in oriented_cluster_mapping.columns:
                    oriented_cluster_mapping[pc1_col] = -pd.to_numeric(
                        oriented_cluster_mapping[pc1_col],
                        errors="coerce",
                    )

    category_means_after = (
        oriented_full_df.assign(
            _category_for_orientation=pd.to_numeric(
                oriented_full_df[category_column],
                errors="coerce",
            )
        )
        .dropna(subset=["_category_for_orientation", "component_1"])
        .groupby("_category_for_orientation")["component_1"]
        .mean()
        .sort_index()
    )

    orientation_df = pd.DataFrame(
        {
            "category": category_means_after.index.astype(int),
            "mean_component_1_after_orientation": category_means_after.values,
        }
    )
    orientation_df["PC1_flipped"] = flipped_pc1
    orientation_df["left_to_right_rank_by_mean_PC1"] = (
        orientation_df["mean_component_1_after_orientation"]
        .rank(method="first")
        .astype(int)
    )

    print("\nPC1 orientation by ordered categories:")
    print(orientation_df.to_string(index=False))
    print(f"PC1 flipped: {flipped_pc1}")

    is_monotonic = orientation_df["mean_component_1_after_orientation"].is_monotonic_increasing
    if not is_monotonic:
        print(
            "[WARNING] The first and last categories are oriented left-to-right, "
            "but the intermediate category means are not perfectly increasing along PC1. "
            "A PC1 mirror can reverse direction, but it cannot reorder non-monotonic clusters."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    orientation_df.to_csv(
        output_dir / "Prediction_PC1_left_to_right_orientation_check.csv",
        index=False,
    )

    if save_outputs:
        oriented_full_df.to_csv(
            output_dir / "FULL_CONCAT_clusters_PC1_oriented.csv",
            index=False,
        )
        if oriented_cluster_mapping is not None:
            oriented_cluster_mapping.to_csv(
                output_dir / "Cluster_remapping_by_ascini_position_PC1_oriented.csv",
                index=False,
            )

    return (
        oriented_full_df,
        oriented_gmm_plot_input,
        oriented_gmm,
        oriented_cluster_mapping,
        flipped_pc1,
        oriented_xlim,
    )


def plot_prediction_same_as_before(
    full_df: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: Path,
    prediction_palette=None,
    prediction_hue_order=None,
    plot_xlim=None,
):
    """Prediction-colored PCA scatterplot using experiment-specific fixed colors."""
    size = 1 * probabilities.max(1) ** 4

    plot_df = full_df.copy()
    plot_df[HUE_COLUMN] = plot_df[HUE_COLUMN].astype(int)
    xlim_to_use = PLOT_XLIM if plot_xlim is None else plot_xlim

    plt.figure(figsize=FIGSIZE)
    plt.ylim(*PLOT_YLIM)
    plt.xlim(*xlim_to_use)

    sns.scatterplot(
        x="component_1",
        y="component_2",
        s=45,
        data=plot_df,
        hue=HUE_COLUMN,
        hue_order=prediction_hue_order,
        size=size,
        sizes=(15, 45),
        legend="brief",
        edgecolor="black",
        palette=prediction_palette,
    )

    if SAVE_FIGURES:
        plt.savefig(output_dir / "Figure4_PCA_GMM_prediction_same_style.png")
        plt.savefig(output_dir / "Figure4_PCA_GMM_prediction_same_style.svg")


def make_plots_same_as_before(
    full_df: pd.DataFrame,
    gmm,
    gmm_plot_input: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: Path,
    prediction_palette=None,
    prediction_hue_order=None,
    labels_override=None,
    plot_xlim=None,
):
    """Make the same active plots as the latest working code."""
    plt.style.use("default")

    plot_gmm_same_as_before(
        gmm,
        gmm_plot_input,
        probabilities,
        output_dir=output_dir,
        prediction_palette=prediction_palette,
        labels_override=labels_override,
        plot_xlim=plot_xlim,
    )

    plot_probability_same_as_before(
        full_df,
        probabilities,
        output_dir=output_dir,
        plot_xlim=plot_xlim,
    )

    plot_ascini_position_same_as_before(
        full_df,
        probabilities,
        output_dir=output_dir,
        position_column="ascini_position",
        palette="rocket",
        vmin=-1,
        vmax=1,
        plot_xlim=plot_xlim,
    )

    plot_prediction_same_as_before(
        full_df,
        probabilities,
        output_dir=output_dir,
        prediction_palette=prediction_palette,
        prediction_hue_order=prediction_hue_order,
        plot_xlim=plot_xlim,
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


# -----------------------------------------------------------------------------
# Remap GMM cluster numbers by ascini_position direction
# -----------------------------------------------------------------------------
def remap_prediction_by_ascini_position(
    full_df: pd.DataFrame,
    gmm,
    output_dir: Path,
    n_clusters: int,
    experiment_name: str,
    position_column: str = "ascini_position",
    prediction_column: str = HUE_COLUMN,
    extreme_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remap original GMM cluster IDs to ordered cluster IDs based on ascini_position.

    New logic:
        Cluster 1 = GMM centroid closest to ascini_position +1 side
        Cluster 2 = next centroid along the +1 to -1 direction
        ...
        Cluster N = GMM centroid closest to ascini_position -1 side

    The GMM may be fit on PC1..PCN. Remapping uses the PC1/PC2 centroid slice
    because these are the dimensions shown in the plots.
    """
    required_columns = [
        "component_1",
        "component_2",
        position_column,
        prediction_column,
    ]

    missing_columns = [col for col in required_columns if col not in full_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in full_df: {missing_columns}")

    remapped_df = full_df.copy()

    original_prediction = pd.to_numeric(
        remapped_df[prediction_column],
        errors="raise",
    ).astype(int)

    reference_df = remapped_df[["component_1", "component_2", position_column]].copy()

    for col in ["component_1", "component_2", position_column]:
        reference_df[col] = pd.to_numeric(reference_df[col], errors="coerce")

    reference_df = reference_df.dropna()

    if reference_df.empty:
        raise ValueError(
            "No usable rows remain after removing NaN values from "
            "component_1, component_2, and ascini_position."
        )

    n_extreme = max(1, int(round(len(reference_df) * extreme_fraction)))
    n_extreme = min(n_extreme, max(1, len(reference_df) // 2))

    near_plus_one = reference_df.nlargest(n_extreme, position_column)
    near_minus_one = reference_df.nsmallest(n_extreme, position_column)

    plus_one_reference = near_plus_one[["component_1", "component_2"]].mean().to_numpy(dtype=float)
    minus_one_reference = near_minus_one[["component_1", "component_2"]].mean().to_numpy(dtype=float)

    # Direction from ascini_position +1 toward ascini_position -1.
    ascini_axis = minus_one_reference - plus_one_reference
    ascini_axis_length = np.linalg.norm(ascini_axis)

    if ascini_axis_length == 0:
        raise ValueError(
            "Could not define an ascini_position axis. "
            "The +1 and -1 reference points are identical in PC space."
        )

    ascini_axis_unit = ascini_axis / ascini_axis_length

    records = []
    means_2d = np.asarray(gmm.means_)[:, :2]

    for original_cluster_id in range(gmm.n_components):
        centroid_pc1, centroid_pc2 = means_2d[original_cluster_id]
        centroid = np.array([centroid_pc1, centroid_pc2], dtype=float)

        projection = float(np.dot(centroid - plus_one_reference, ascini_axis_unit))

        # Approximate where the cluster centroid lies on the +1 to -1 scale.
        estimated_ascini_position = float(1 - 2 * (projection / ascini_axis_length))

        records.append(
            {
                "Prediction_original": original_cluster_id,
                "centroid_component_1": centroid_pc1,
                "centroid_component_2": centroid_pc2,
                "projection_from_plus_to_minus": projection,
                "estimated_ascini_position": estimated_ascini_position,
            }
        )

    cluster_mapping = (
        pd.DataFrame(records)
        .sort_values("projection_from_plus_to_minus")
        .reset_index(drop=True)
    )

    cluster_mapping["Prediction_new"] = np.arange(1, len(cluster_mapping) + 1)

    colors = get_experiment_cluster_colors(experiment_name, n_clusters)
    if len(colors) == len(cluster_mapping):
        cluster_mapping["suggested_color"] = colors

    remap_dict = dict(
        zip(
            cluster_mapping["Prediction_original"].astype(int),
            cluster_mapping["Prediction_new"].astype(int),
        )
    )

    new_prediction = original_prediction.map(remap_dict)

    if new_prediction.isna().any():
        missing_labels = sorted(original_prediction[new_prediction.isna()].unique().tolist())
        raise ValueError(f"Some Prediction labels could not be remapped: {missing_labels}")

    if "Prediction_original" not in remapped_df.columns:
        remapped_df.insert(
            remapped_df.columns.get_loc(prediction_column),
            "Prediction_original",
            original_prediction,
        )
    else:
        remapped_df["Prediction_original"] = original_prediction

    remapped_df[prediction_column] = new_prediction.astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    remapped_output_file = output_dir / "FULL_CONCAT_clusters.csv"
    mapping_output_file = output_dir / "Cluster_remapping_by_ascini_position.csv"

    remapped_df.to_csv(remapped_output_file, index=False)
    cluster_mapping.to_csv(mapping_output_file, index=False)

    print("\nCluster remapping based on ascini_position:")
    print(cluster_mapping.to_string(index=False))

    print(f"\nSaved remapped table to: {remapped_output_file}")
    print(f"Saved cluster mapping to: {mapping_output_file}")

    return remapped_df, cluster_mapping




# -----------------------------------------------------------------------------
# Pittsburgh H/D export
# -----------------------------------------------------------------------------
def read_pittsburgh_source_table(output_dir: Path) -> tuple[pd.DataFrame, Path]:
    """
    Read the best available per-experiment cluster table.

    Preference:
        1. FULL_CONCAT_clusters_PC1_oriented.csv
        2. FULL_CONCAT_clusters.csv

    The oriented file is preferred because it matches the final displayed
    PC1 orientation, but H/D only require ascini_position and Prediction.
    """

    oriented_file = output_dir / "FULL_CONCAT_clusters_PC1_oriented.csv"
    remapped_file = output_dir / "FULL_CONCAT_clusters.csv"

    if oriented_file.exists():
        return pd.read_csv(oriented_file), oriented_file

    if remapped_file.exists():
        return pd.read_csv(remapped_file), remapped_file

    raise FileNotFoundError(
        f"Could not find either {oriented_file.name} or {remapped_file.name} "
        f"in {output_dir}"
    )


def calculate_pittsburgh_indices_for_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    n_clusters: int,
    bin_edges: np.ndarray,
    position_column: str = PITTSBURGH_POSITION_COLUMN,
    category_column: str = PITTSBURGH_CATEGORY_COLUMN,
) -> pd.DataFrame:
    """
    Calculate Pittsburgh heterogeneity indices for one experiment.

    H = Shannon entropy:
        H = -sum(p_i * log2(p_i))

    D = Simpson dominance index:
        D = sum(p_i ** 2)

    E = Shannon evenness:
        E = H / log2(number_of_possible_categories)

    p_i is the fraction of cells in category i within each ascini_position bin.
    """

    required_columns = [position_column, category_column]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{experiment_name}: missing required columns for Pittsburgh export: "
            f"{missing_columns}"
        )

    work = df.copy()

    work[position_column] = pd.to_numeric(
        work[position_column],
        errors="coerce",
    )

    work[category_column] = pd.to_numeric(
        work[category_column],
        errors="coerce",
    )

    work = work.dropna(subset=[position_column, category_column]).copy()
    work[category_column] = work[category_column].astype(int)

    # Use the expected ordered categories from the manual GMM setting.
    # This keeps H/E comparable within each experiment even if one bin is missing
    # one of the categories.
    expected_categories = list(range(1, int(n_clusters) + 1))

    # Assign each cell to an ascini_position bin.
    work["_position_bin"] = pd.cut(
        work[position_column],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=True,
    )

    outside_range_n = int(work["_position_bin"].isna().sum())
    if outside_range_n > 0:
        print(
            f"[WARNING] {experiment_name}: {outside_range_n} cells were outside "
            f"the Pittsburgh bin range and were excluded."
        )

    work = work.dropna(subset=["_position_bin"]).copy()
    work["_position_bin"] = work["_position_bin"].astype(int)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    rows = []

    for bin_idx, bin_center in enumerate(bin_centers):
        bin_df = work[work["_position_bin"] == bin_idx]

        counts = (
            bin_df[category_column]
            .value_counts()
            .reindex(expected_categories, fill_value=0)
            .astype(float)
        )

        total_cells = counts.sum()

        if total_cells == 0:
            H = np.nan
            D = np.nan
            E = np.nan
        else:
            p = counts / total_cells
            p_nonzero = p[p > 0]

            H = float(-(p_nonzero * np.log2(p_nonzero)).sum())
            D = float((p ** 2).sum())

            if len(expected_categories) > 1:
                E = float(H / np.log2(len(expected_categories)))
            else:
                E = np.nan

        row = {
            "Experiment": experiment_name,
            "bin_index": bin_idx,
            "ascini_position_binned": float(bin_center),
            "bin_left": float(bin_edges[bin_idx]),
            "bin_right": float(bin_edges[bin_idx + 1]),
            "n_cells": int(total_cells),
            "n_possible_categories": int(len(expected_categories)),
            "H": H,
            "D": D,
            "E": E,
        }

        # Add category counts for QC.
        for category in expected_categories:
            row[f"count_category_{category}"] = int(counts.loc[category])

        rows.append(row)

    return pd.DataFrame(rows)


def export_pittsburgh_indices_from_gmm_outputs(
    output_root_dir: Path = OUTPUT_ROOT_DIR,
    experiments_to_export=None,
    experiment_settings=None,
    prefix_map=None,
    n_bins: int = PITTSBURGH_N_BINS,
    position_range: tuple[float, float] = PITTSBURGH_POSITION_RANGE,
    wide_output_name: str = PITTSBURGH_WIDE_OUTPUT_NAME,
    long_output_name: str = PITTSBURGH_LONG_OUTPUT_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine CNT/STV/WD remapped GMM outputs into one Pittsburgh H/D CSV.

    The wide output is compatible with your Pittsburgh plotting script:
        ascini_position_binned
        C_H, F_H, W_H
        C_D, F_D, W_D

    It also includes readable aliases:
        CNT_H, STV_H, WD_H
        CNT_D, STV_D, WD_D
    """

    if experiments_to_export is None:
        experiments_to_export = PITTSBURGH_EXPERIMENTS_TO_EXPORT

    if experiment_settings is None:
        experiment_settings = EXPERIMENT_SETTINGS

    if prefix_map is None:
        prefix_map = PITTSBURGH_PREFIX_MAP

    output_root_dir = Path(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(
        float(position_range[0]),
        float(position_range[1]),
        int(n_bins) + 1,
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    wide_df = pd.DataFrame(
        {
            "ascini_position_binned": bin_centers,
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
        }
    )

    all_long_dfs = []

    for experiment_name in experiments_to_export:
        experiment_name = str(experiment_name)

        if experiment_name not in experiment_settings:
            print(
                f"[WARNING] Skipping {experiment_name}: not found in EXPERIMENT_SETTINGS."
            )
            continue

        n_pcs = int(experiment_settings[experiment_name]["n_pcs"])
        n_clusters = int(experiment_settings[experiment_name]["n_clusters"])

        experiment_safe = safe_name(experiment_name)
        experiment_output_dir = output_root_dir / f"{experiment_safe}_PC{n_pcs}_K{n_clusters}"

        try:
            experiment_df, source_file = read_pittsburgh_source_table(
                experiment_output_dir
            )
        except FileNotFoundError as exc:
            print(f"[WARNING] {exc}")
            continue

        print(
            f"[INFO] Pittsburgh export using {experiment_name} source file: "
            f"{source_file}"
        )

        experiment_indices = calculate_pittsburgh_indices_for_experiment(
            df=experiment_df,
            experiment_name=experiment_name,
            n_clusters=n_clusters,
            bin_edges=bin_edges,
            position_column=PITTSBURGH_POSITION_COLUMN,
            category_column=PITTSBURGH_CATEGORY_COLUMN,
        )

        all_long_dfs.append(experiment_indices)

        prefix = prefix_map.get(experiment_name, experiment_safe)

        # Old plotting-script compatible names.
        wide_df[f"{prefix}_H"] = experiment_indices["H"].values
        wide_df[f"{prefix}_D"] = experiment_indices["D"].values
        wide_df[f"{prefix}_E"] = experiment_indices["E"].values
        wide_df[f"{prefix}_n_cells"] = experiment_indices["n_cells"].values

        # Readable aliases.
        wide_df[f"{experiment_name}_H"] = experiment_indices["H"].values
        wide_df[f"{experiment_name}_D"] = experiment_indices["D"].values
        wide_df[f"{experiment_name}_E"] = experiment_indices["E"].values
        wide_df[f"{experiment_name}_n_cells"] = experiment_indices["n_cells"].values

    if len(all_long_dfs) == 0:
        raise RuntimeError(
            "No Pittsburgh indices were calculated. Check that the per-experiment "
            "FULL_CONCAT_clusters files exist."
        )

    long_df = pd.concat(all_long_dfs, ignore_index=True)

    wide_output_path = output_root_dir / wide_output_name
    long_output_path = output_root_dir / long_output_name

    wide_df.to_csv(wide_output_path, index=False)
    long_df.to_csv(long_output_path, index=False)

    print(f"\n[INFO] Saved Pittsburgh wide H/D CSV to: {wide_output_path}")
    print(f"[INFO] Saved Pittsburgh long H/D CSV to: {long_output_path}")

    return wide_df, long_df


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def run_one_experiment(all_data: pd.DataFrame, experiment_name: str) -> None:
    n_pcs, n_clusters = get_experiment_settings(experiment_name)

    experiment_safe = safe_name(experiment_name)
    output_dir = OUTPUT_ROOT_DIR / f"{experiment_safe}_PC{n_pcs}_K{n_clusters}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manual GMM settings for {experiment_name}: PC1-PC{n_pcs}, K={n_clusters}")
    print(f"Output directory: {output_dir}")

    raw_data, clean_data = load_and_filter_data_for_experiment(all_data, experiment_name)
    feature_matrix, pca_feature_columns = build_pca_feature_matrix(clean_data)
    pca_df, pca, _ = run_pca(feature_matrix)

    if n_pcs > pca_df.shape[1]:
        raise ValueError(
            f"Requested n_pcs={n_pcs}, but only {pca_df.shape[1]} PCA components are available."
        )

    gmm, gmm_input_selected_pcs, labels, probabilities, selected_pc_columns = fit_gmm_on_selected_pcs(
        pca_df,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
    )

    gmm_plot_input = make_gmm_plot_input(pca_df)

    gmm_name = f"Full_cell_linked_model_{experiment_safe}_PC{n_pcs}_K{n_clusters}"
    save_gmm_parameters(gmm, output_dir=output_dir, gmm_name=gmm_name)

    print(gmm.get_params())
    print(probabilities[:5].round(3))
    print("Selected GMM PC columns:", selected_pc_columns)

    full_df = save_tables(
        raw_data=raw_data,
        clean_data=clean_data,
        pca_df=pca_df,
        pca_feature_columns=pca_feature_columns,
        pca=pca,
        probabilities=probabilities,
        labels=labels,
        output_dir=output_dir,
        experiment_name=experiment_name,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
    )

    # -------------------------------------------------------------------------
    # Remap Prediction labels based on ascini_position direction.
    # This creates FULL_CONCAT_clusters.csv with categories 1..N.
    # -------------------------------------------------------------------------
    full_df, cluster_mapping = remap_prediction_by_ascini_position(
        full_df=full_df,
        gmm=gmm,
        output_dir=output_dir,
        n_clusters=n_clusters,
        experiment_name=experiment_name,
        position_column="ascini_position",
        prediction_column=HUE_COLUMN,
        extreme_fraction=0.10,
    )

    # -------------------------------------------------------------------------
    # Orient PC1 for display so category 1 is on the left and the last category
    # is on the right. This does not refit PCA or GMM.
    # -------------------------------------------------------------------------
    (
        full_df_for_plot,
        gmm_plot_input_for_plot,
        gmm_for_plot,
        cluster_mapping_for_plot,
        pc1_flipped,
        plot_xlim_for_display,
    ) = orient_pc1_by_category_order(
        full_df=full_df,
        gmm_plot_input=gmm_plot_input,
        gmm=gmm,
        output_dir=output_dir,
        cluster_mapping=cluster_mapping,
        category_column=ORIENTATION_CATEGORY_COLUMN,
        plot_xlim=PLOT_XLIM,
        auto_orient=ORIENT_PC1_BY_CATEGORY_ORDER,
        save_outputs=SAVE_ORIENTED_FULL_CONCAT,
    )

    prediction_palette, prediction_hue_order = make_ordered_prediction_palette(
        n_clusters=n_clusters,
        experiment_name=experiment_name,
    )

    # -------------------------------------------------------------------------
    # Randomly select only a fraction of cells for plotting.
    # This does NOT change PCA, GMM fitting, probabilities, or FULL_CONCAT.csv.
    # It only controls which points are shown in the plots.
    # -------------------------------------------------------------------------
    if not (0 < PLOT_FRACTION <= 1):
        raise ValueError("PLOT_FRACTION must be > 0 and <= 1.")

    plot_indices = full_df_for_plot.sample(
        frac=PLOT_FRACTION,
        random_state=PLOT_RANDOM_STATE,
    ).index.sort_values()

    full_df_plot = full_df_for_plot.loc[plot_indices].copy()
    gmm_plot_input_plot = gmm_plot_input_for_plot.loc[plot_indices].copy()
    probabilities_plot = probabilities[plot_indices.to_numpy(), :]
    labels_plot = full_df_plot[HUE_COLUMN].astype(int).to_numpy()

    make_plots_same_as_before(
        full_df_plot,
        gmm_for_plot,
        gmm_plot_input_plot,
        probabilities_plot,
        output_dir=output_dir,
        prediction_palette=prediction_palette,
        prediction_hue_order=prediction_hue_order,
        labels_override=labels_plot,
        plot_xlim=plot_xlim_for_display,
    )

    print(f"\nFinished {experiment_name}. Files saved in: {output_dir}")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Could not find input CSV: {INPUT_CSV}")

    all_data = load_combined_csv(INPUT_CSV)

    detected = all_data["Experiment"].value_counts(dropna=False).sort_index()
    print("\nDetected experiments:")
    print(detected)

    for experiment_name in EXPERIMENTS_TO_RUN:
        run_one_experiment(all_data, str(experiment_name))

    if MAKE_PITTSBURGH_INDEX_CSV:
        export_pittsburgh_indices_from_gmm_outputs(
            output_root_dir=OUTPUT_ROOT_DIR,
            experiments_to_export=PITTSBURGH_EXPERIMENTS_TO_EXPORT,
            experiment_settings=EXPERIMENT_SETTINGS,
            prefix_map=PITTSBURGH_PREFIX_MAP,
            n_bins=PITTSBURGH_N_BINS,
            position_range=PITTSBURGH_POSITION_RANGE,
            wide_output_name=PITTSBURGH_WIDE_OUTPUT_NAME,
            long_output_name=PITTSBURGH_LONG_OUTPUT_NAME,
        )
        
    


if __name__ == "__main__":
    main()
