"""
Figure 2 PCA/GMM workflow - cleaned version with the same plot style/scale
as the latest working notebook code.

This script intentionally preserves the visual choices from the original code:
- seaborn `sns.set()` at import time
- matplotlib `plt.style.use("default")` before plotting
- 8 x 8 inch figures
- x/y limits used in the latest plotting block
- same palettes: coolwarm_r, magma, Purples
- same point-size formulas
- same GMM ellipse plotting behavior

Pipeline:
1. Load the group-level cell table.
2. Replace '-' with NaN, fill NaN with 0, remove +/- infinity rows.
3. Filter cells with mito_aspect_ratio > 0 and mito_density * area >= 20.
4. Drop non-PCA columns and z-score the remaining features.
5. Compute all PCA components and save them to PCA.csv.
6. Fit a 5-component GMM using only component_1 and component_2.
7. Save GMM parameters and FULL_CONCAT.csv.
8. Generate the same three plots as the latest notebook code:
   A. GMM ellipse plot
   B. PCA plot colored by maximum GMM probability
   C. PCA plot colored by GMM Prediction
"""

from pathlib import Path
import copy

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
INPUT_CSV = Path(r"...add path.../Fig2_Hepatocyte _Clustering_Group_Control_data .csv")


OUTPUT_PLOTS_DIR = INPUT_CSV.parent / "Full_Concat_data"

N_CLUSTERS = 5
HUE_COLUMN = "Prediction"
GMM_NAME = "Full_cell_linked_model_CNT_M_data"

MIN_MITO_COUNT = 20 
PCA_RANDOM_STATE = 42
GMM_RANDOM_STATE = 54  
MIN_MITO_DENSITY = 0.28 # use 0.10 or 0.28 if want to be more stringent 

# These match your latest plotting block.
PLOT_XLIM = (-15, 20) 
PLOT_YLIM = (-11.5, 20)
FIGSIZE = (9, 9)

# Keep True if you run this as a script and want the figures to appear.
# In Jupyter, the figures will also display naturally.
SHOW_FIGURES = True

# Set True if you also want image files saved in OUTPUT_PLOTS_DIR.
SAVE_FIGURES = True

# Automatically mirror the displayed PC1 axis so the ordered categories
# go from category 1 on the left to the last category on the right.
# This changes the orientation of the saved plots only; PCA/GMM fitting stays the same.
ORIENT_PC1_BY_CATEGORY_ORDER = True
ORIENTATION_CATEGORY_COLUMN = HUE_COLUMN
SAVE_ORIENTED_FULL_CONCAT = True


# -----------------------------------------------------------------------------
# Randomly select only a fraction of cells for plotting.
# This does NOT change PCA, GMM fitting, probabilities, or FULL_CONCAT.csv.
# It only controls which points are shown in the plots.
# -----------------------------------------------------------------------------
PLOT_FRACTION = 1.0 
PLOT_RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Columns excluded from the PCA feature matrix
# -----------------------------------------------------------------------------
EXCLUDE_FROM_PCA = [
    "Unnamed: 0", #
    "label", #
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
  
]


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_and_filter_data(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the input CSV and apply the same cleaning/filtering as before."""
    raw_data = pd.read_csv(input_csv, na_values="-")

    # Same order as your notebook code.
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
        errors="coerce"
    )

    before_cell_id_filter = len(raw_data)

    raw_data = raw_data[
        raw_data["cell_id_linked"] != 0
    ].copy()

    after_cell_id_filter = len(raw_data)

    print(
        "Removed rows with cell_id_linked == 0:",
        before_cell_id_filter - after_cell_id_filter
    )

    # -------------------------------------------------------------------------
    # Existing mitochondrial filters
    # -------------------------------------------------------------------------
    
    
    mito_positive = raw_data[raw_data["mito_aspect_ratio"] > 0].copy()

    mito_density_filtered = mito_positive[
        mito_positive["mito_density"] >= MIN_MITO_DENSITY
    ].copy()
    
    mito_count = mito_density_filtered["mito_density"] * mito_density_filtered["area"]
    
    clean_data = mito_density_filtered[
        mito_count >= MIN_MITO_COUNT
    ].copy()

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
def fit_gmm_on_pc1_pc2(pca_df: pd.DataFrame):
    """Fit the GMM using only component_1 and component_2, matching the plot."""
    gmm_input = pca_df[["component_1", "component_2"]]

    gmm = mixture.GaussianMixture(
        n_components=N_CLUSTERS,
        random_state=GMM_RANDOM_STATE,
        covariance_type="full",
        max_iter=100,
        tol=1e-3,
    ).fit(gmm_input)
    labels = gmm.predict(gmm_input)
    probabilities = gmm.predict_proba(gmm_input)

    return gmm, gmm_input, labels, probabilities


def save_gmm_parameters(gmm) -> None:
    """Save GMM weights, means, and covariance matrices using original names."""
    print("the weights are :", gmm.weights_)
    np.save(GMM_NAME + "_weights", gmm.weights_, allow_pickle=False)
    np.save(GMM_NAME + "_means", gmm.means_, allow_pickle=False)
    np.save(GMM_NAME + "_covariances", gmm.covariances_, allow_pickle=False)


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
) -> pd.DataFrame:
    """Save the same main CSVs plus useful PCA documentation."""
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    
    raw_data.to_csv("Data_including_bad_cells.csv")
    clean_data.to_csv("Clean_Data.csv")

    
    pca_df.to_csv("PCA.csv", index=False)

  
    pd.Series(pca_feature_columns, name="PCA_feature_column").to_csv(
        OUTPUT_PLOTS_DIR / "PCA_feature_columns.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "component": [f"component_{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(OUTPUT_PLOTS_DIR / "PCA_explained_variance_ratio.csv", index=False)

  
    # Keep the original probability column style: 0, 1, 2, 3, ...
    gmm_df = pd.DataFrame(probabilities)
    
    # Maximum GMM probability for each cell
    gmm_df["Probability"] = probabilities.max(axis=1)
    
    # Original GMM prediction label
    gmm_df["Prediction"] = labels



    gmm_and_pca_df = pd.concat([gmm_df, pca_df], axis="columns")

 
    clean_data_from_file = pd.read_csv("Clean_Data.csv")

    full_concat = pd.concat([gmm_and_pca_df, clean_data_from_file], axis="columns")
    output_file = OUTPUT_PLOTS_DIR / "FULL_CONCAT.csv"
    full_concat.to_csv(output_file, index=False)

    return pd.read_csv(output_file)


# -----------------------------------------------------------------------------
# Plotting helpers - intentionally preserve original visual behavior
# -----------------------------------------------------------------------------
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance."""
    ax = ax or plt.gca()

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


def plot_gmm_same_as_before(
    gmm,
    gmm_input: pd.DataFrame,
    probabilities: np.ndarray,
    label=True,
    labels_override=None,
    plot_xlim=None,
):
    """
    Recreate the original active GMM ellipse plot.

    Important: the old plotting function refit the GMM inside the plotting
    function. That is unusual, but this version keeps that behavior so the plot
    matches the old code as closely as possible.

    The original function also called ax.axis("equal") on an axes object that
    was created before the plotting figure. Visually, that did not force the
    displayed PCA plot to equal aspect. To match the displayed plot and avoid an
    extra blank figure, this cleaned version does not force equal aspect here.
    """
    
    if labels_override is None:
        labels = gmm.predict(gmm_input)
    else:
        labels = np.asarray(labels_override)

    size = 5 * probabilities.max(1) ** 3
    xlim_to_use = PLOT_XLIM if plot_xlim is None else plot_xlim

    plt.figure(figsize=FIGSIZE)
    plt.xlim(*xlim_to_use)
    plt.ylim(*PLOT_YLIM)

    if label:
        plt.scatter(
            gmm_input["component_1"],
            gmm_input["component_2"],
            c=labels,
            linewidths=0,
            s=40,
            cmap="coolwarm_r",
            zorder=1,
        )
    else:
        plt.scatter(
            gmm_input["component_1"],
            gmm_input["component_2"],
            linewidths=0,
            edgecolor="black",
            s=size,
            zorder=1,
        )

    w_factor = 0.1 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

    if SAVE_FIGURES:
        plt.savefig(OUTPUT_PLOTS_DIR / "Figure2_PCA_GMM_ellipses_same_style.png")
        plt.savefig(OUTPUT_PLOTS_DIR / "Figure2_PCA_GMM_ellipses_same_style.svg")


def plot_probability_same_as_before(
    full_df: pd.DataFrame,
    probabilities: np.ndarray,
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
        s=100,  #70
        data=full_df,
        hue=max_probability,
        size=size,
        sizes=(30, 100),
        legend="brief",
        edgecolor="black",
        palette="magma",
    )

    if SAVE_FIGURES:
        plt.savefig(OUTPUT_PLOTS_DIR / "Figure2_PCA_GMM_probability_same_style.png")
        plt.savefig(OUTPUT_PLOTS_DIR / "Figure2_PCA_GMM_probability_same_style.svg")


def plot_ascini_position_same_as_before(
    full_df: pd.DataFrame,
    probabilities: np.ndarray,
    position_column: str = "ascini_position",
    palette: str = "rocket",
    vmin: float = -1,
    vmax: float = 1,
):
    """
    PCA scatter plot colored by ascini_position using the rocket color scheme.

    This does not refit PCA or GMM.
    It uses the existing component_1 and component_2 coordinates.
    """

    required_columns = ["component_1", "component_2", position_column]
    missing_columns = [
        col for col in required_columns
        if col not in full_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"These columns are missing from full_df: {missing_columns}"
        )

    plot_df = full_df.copy()

    for col in required_columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    # Match the probability-based point-size style from your other PCA plots.
    plot_df["_max_probability"] = probabilities.max(axis=1)
    plot_df["_point_size"] = 15 + 60 * (plot_df["_max_probability"] ** 4)

    plot_df = plot_df.dropna(
        subset=["component_1", "component_2", position_column]
    ).copy()

    cmap = sns.color_palette(palette, as_cmap=True)

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

    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)

    ax.set_xlabel("component_1")
    ax.set_ylabel("component_2")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(position_column)

    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(
            OUTPUT_PLOTS_DIR / "Figure2_PCA_ascini_position_rocket_same_style.png",
            dpi=500,
        )
        plt.savefig(
            OUTPUT_PLOTS_DIR / "Figure2_PCA_ascini_position_rocket_same_style.svg",
            dpi=500,
        )


def make_ordered_prediction_palette(n_clusters: int = N_CLUSTERS):
    """
    Return a palette and hue order for remapped categories 1..N.

    This is used after remap_prediction_by_ascini_position(), so the prediction
    plot legend and colors follow the new ordered category numbers.
    """
    fixed_purple_colors = [
        "#fcfbfd",  # category 1
        "#9e9ac8",  # category 2
        "#6950a3",  # category 3
        "#dadaeb",  # category 4
        "#3f007d",  # category 5 / last category
    ]

    if len(fixed_purple_colors) != n_clusters:
        fixed_purple_colors = sns.color_palette(
            "Purples",
            n_colors=n_clusters,
        ).as_hex()

    categories = list(range(1, n_clusters + 1))
    prediction_palette = dict(zip(categories, fixed_purple_colors))
    prediction_hue_order = categories

    return prediction_palette, prediction_hue_order


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
            # Reflecting PC1 changes the sign of the PC1/PC2 covariance terms.
            mirrored_gmm.covariances_[:, 0, 1] = -mirrored_gmm.covariances_[:, 0, 1]
            mirrored_gmm.covariances_[:, 1, 0] = -mirrored_gmm.covariances_[:, 1, 0]

        elif getattr(mirrored_gmm, "covariance_type", None) == "tied":
            mirrored_gmm.covariances_[0, 1] = -mirrored_gmm.covariances_[0, 1]
            mirrored_gmm.covariances_[1, 0] = -mirrored_gmm.covariances_[1, 0]

        # For diag and spherical covariance, reflection does not change variances.

    return mirrored_gmm


def orient_pc1_by_category_order(
    full_df: pd.DataFrame,
    gmm_input: pd.DataFrame,
    gmm,
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
    oriented_gmm_input = gmm_input.copy()
    oriented_gmm = gmm
    oriented_cluster_mapping = None if cluster_mapping is None else cluster_mapping.copy()
    oriented_xlim = plot_xlim
    flipped_pc1 = False

    if not auto_orient:
        return (
            oriented_full_df,
            oriented_gmm_input,
            oriented_gmm,
            oriented_cluster_mapping,
            flipped_pc1,
            oriented_xlim,
        )

    categories_numeric = pd.to_numeric(
        oriented_full_df[category_column],
        errors="coerce",
    )

    valid_categories = sorted(
        categories_numeric.dropna().astype(int).unique().tolist()
    )

    if len(valid_categories) < 2:
        print(
            "[INFO] PC1 orientation skipped because fewer than two categories "
            f"were found in '{category_column}'."
        )
        return (
            oriented_full_df,
            oriented_gmm_input,
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
        oriented_gmm_input["component_1"] = -pd.to_numeric(
            oriented_gmm_input["component_1"],
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

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    orientation_df.to_csv(
        OUTPUT_PLOTS_DIR / "Prediction_PC1_left_to_right_orientation_check.csv",
        index=False,
    )

    if save_outputs:
        oriented_full_df.to_csv(
            OUTPUT_PLOTS_DIR / "FULL_CONCAT_clusters_PC1_oriented.csv",
            index=False,
        )
        if oriented_cluster_mapping is not None:
            oriented_cluster_mapping.to_csv(
                OUTPUT_PLOTS_DIR / "Cluster_remapping_by_ascini_position_PC1_oriented.csv",
                index=False,
            )

    return (
        oriented_full_df,
        oriented_gmm_input,
        oriented_gmm,
        oriented_cluster_mapping,
        flipped_pc1,
        oriented_xlim,
    )


def make_ascini_axis_prediction_palette(
    full_df: pd.DataFrame,
    gmm,
    position_column: str = "ascini_position",
    extreme_fraction: float = 0.10,
):
    """
    Assign fixed Prediction colors using ascini_position as a reference axis.

    The first color is assigned to the GMM centroid closest to ascini_position +1.
    The last color is assigned to the GMM centroid closest to ascini_position -1.

    This uses only PC1/PC2 centroid coordinates for color assignment.
    It does not refit PCA or GMM.
    """

    fixed_purple_colors = [
        "#fcfbfd",  # closest to ascini_position +1
        "#9e9ac8",
        "#6950a3",
        "#dadaeb",
        "#3f007d",  # closest to ascini_position -1
    ]

    if len(fixed_purple_colors) != N_CLUSTERS:
        raise ValueError(
            f"You have {N_CLUSTERS} clusters but {len(fixed_purple_colors)} colors."
        )

    required_columns = ["component_1", "component_2", position_column]
    missing_columns = [col for col in required_columns if col not in full_df.columns]

    if missing_columns:
        raise KeyError(
            f"These columns are missing from full_df: {missing_columns}"
        )

    # Keep only the PCA coordinates and ascini_position.
    reference_df = full_df[required_columns].copy()

    for col in required_columns:
        reference_df[col] = pd.to_numeric(reference_df[col], errors="coerce")

    reference_df = reference_df.dropna()

    if reference_df.empty:
        raise ValueError(
            "No usable rows remain after removing NaN values from "
            "component_1, component_2, and ascini_position."
        )

    # Use the cells nearest to ascini_position +1 and -1 as reference points.
    n_extreme = max(1, int(round(len(reference_df) * extreme_fraction)))
    n_extreme = min(n_extreme, max(1, len(reference_df) // 2))

    near_plus_one = reference_df.nlargest(n_extreme, position_column)
    near_minus_one = reference_df.nsmallest(n_extreme, position_column)

    plus_one_reference = near_plus_one[["component_1", "component_2"]].mean().to_numpy()
    minus_one_reference = near_minus_one[["component_1", "component_2"]].mean().to_numpy()

    # Axis direction from ascini_position +1 to ascini_position -1.
    ascini_axis = minus_one_reference - plus_one_reference
    ascini_axis_length = np.linalg.norm(ascini_axis)

    if ascini_axis_length == 0:
        raise ValueError(
            "Could not define an ascini_position axis in PCA space. "
            "The +1 and -1 reference points are identical."
        )

    ascini_axis_unit = ascini_axis / ascini_axis_length

    records = []

    for cluster_id in range(N_CLUSTERS):
        centroid_pc1, centroid_pc2 = gmm.means_[cluster_id]
        centroid = np.array([centroid_pc1, centroid_pc2])

        # Projection of the GMM centroid onto the +1 to -1 axis.
        projection = np.dot(
            centroid - plus_one_reference,
            ascini_axis_unit,
        )

        # Approximate position on the +1 to -1 scale.
        # +1 reference gives about +1.
        # -1 reference gives about -1.
        estimated_ascini_position = 1 - 2 * (projection / ascini_axis_length)

        records.append(
            {
                "cluster": cluster_id,
                "centroid_PC1": centroid_pc1,
                "centroid_PC2": centroid_pc2,
                "projection_plus_to_minus": projection,
                "estimated_ascini_position": estimated_ascini_position,
            }
        )

    assignment_df = (
        pd.DataFrame(records)
        .sort_values("projection_plus_to_minus")
        .reset_index(drop=True)
    )

    # First centroid along the +1 to -1 axis gets the first color.
    assignment_df["color"] = fixed_purple_colors

    prediction_palette = dict(
        zip(
            assignment_df["cluster"].astype(int),
            assignment_df["color"],
        )
    )

    prediction_hue_order = assignment_df["cluster"].astype(int).tolist()

    print("\nPrediction color assignment based on ascini_position axis:")
    print(
        assignment_df[
            [
                "cluster",
                "centroid_PC1",
                "centroid_PC2",
                "estimated_ascini_position",
                "color",
            ]
        ].to_string(index=False)
    )

    assignment_df.to_csv(
        OUTPUT_PLOTS_DIR / "Prediction_color_assignment_by_ascini_position.csv",
        index=False,
    )

    return prediction_palette, prediction_hue_order


def plot_prediction_same_as_before(
    full_df: pd.DataFrame,
    probabilities: np.ndarray,
    prediction_palette=None,
    prediction_hue_order=None,
    plot_xlim=None,
):
    """Prediction-colored PCA scatterplot using ascini_position-based fixed colors."""

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
        plt.savefig(OUTPUT_PLOTS_DIR / "Figure2_PCA_GMM_prediction_same_style.png")
        plt.savefig(OUTPUT_PLOTS_DIR / "Figure2_PCA_GMM_prediction_same_style.svg")


def make_plots_same_as_before(
    full_df: pd.DataFrame,
    gmm,
    gmm_input: pd.DataFrame,
    probabilities: np.ndarray,
    prediction_palette=None,
    prediction_hue_order=None,
    labels_override=None,
    plot_xlim=None,
):
    """Make the same active plots as the latest working code."""
    plt.style.use("default")

  
    plot_gmm_same_as_before(gmm, gmm_input, probabilities)
    plot_probability_same_as_before(full_df, probabilities)
    
    plot_ascini_position_same_as_before(
        full_df,
        probabilities,
        position_column="ascini_position",
        palette="rocket",
        vmin=-1,
        vmax=1,
    )
    
    plot_prediction_same_as_before(
        full_df,
        probabilities,
        prediction_palette=prediction_palette,
        prediction_hue_order=prediction_hue_order,
    )
            
# -----------------------------------------------------------------------------
# Remap GMM cluster numbers by ascini_position direction
# -----------------------------------------------------------------------------
def remap_prediction_by_ascini_position(
    full_df: pd.DataFrame,
    gmm,
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

    This does not refit PCA or GMM.
    It only changes the cluster numbering in the output table.
    """

    required_columns = [
        "component_1",
        "component_2",
        position_column,
        prediction_column,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in full_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns in full_df: {missing_columns}"
        )

    if gmm.means_.shape[1] != 2:
        raise ValueError(
            "This remapping expects the GMM to have been fit on PC1 and PC2 only."
        )

    remapped_df = full_df.copy()

    # Make sure original Prediction values are integer GMM labels: 0, 1, 2, ...
    original_prediction = pd.to_numeric(
        remapped_df[prediction_column],
        errors="raise",
    ).astype(int)

    # -------------------------------------------------------------------------
    # Define the ascini_position axis in PC1/PC2 space.
    # We use the cells closest to +1 and -1 as reference clouds.
    # -------------------------------------------------------------------------
    reference_df = remapped_df[
        ["component_1", "component_2", position_column]
    ].copy()

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

    plus_one_reference = near_plus_one[
        ["component_1", "component_2"]
    ].mean().to_numpy(dtype=float)

    minus_one_reference = near_minus_one[
        ["component_1", "component_2"]
    ].mean().to_numpy(dtype=float)

    # Direction from ascini_position +1 toward ascini_position -1
    ascini_axis = minus_one_reference - plus_one_reference
    ascini_axis_length = np.linalg.norm(ascini_axis)

    if ascini_axis_length == 0:
        raise ValueError(
            "Could not define an ascini_position axis. "
            "The +1 and -1 reference points are identical in PC space."
        )

    ascini_axis_unit = ascini_axis / ascini_axis_length

    # -------------------------------------------------------------------------
    # Project each GMM centroid onto the +1 to -1 ascini_position axis.
    # -------------------------------------------------------------------------
    records = []

    for original_cluster_id in range(gmm.n_components):
        centroid_pc1, centroid_pc2 = gmm.means_[original_cluster_id]
        centroid = np.array([centroid_pc1, centroid_pc2], dtype=float)

        projection = float(
            np.dot(
                centroid - plus_one_reference,
                ascini_axis_unit,
            )
        )

        # Approximate where the cluster centroid lies on the +1 to -1 scale.
        # Near +1 reference -> about +1
        # Near -1 reference -> about -1
        estimated_ascini_position = float(
            1 - 2 * (projection / ascini_axis_length)
        )

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

    # New labels:
    # 1 = closest to +1 side
    # N = closest to -1 side
    cluster_mapping["Prediction_new"] = np.arange(
        1,
        len(cluster_mapping) + 1,
    )

    # Optional color documentation using your requested order
    fixed_purple_colors = [
        "#fcfbfd",  # cluster 1: closest to ascini_position +1
        "#9e9ac8",  # cluster 2
        "#6950a3",  # cluster 3
        "#dadaeb",  # cluster 4
        "#3f007d",  # cluster 5: closest to ascini_position -1
    ]

    if len(fixed_purple_colors) == len(cluster_mapping):
        cluster_mapping["suggested_color"] = fixed_purple_colors

    remap_dict = dict(
        zip(
            cluster_mapping["Prediction_original"].astype(int),
            cluster_mapping["Prediction_new"].astype(int),
        )
    )

    new_prediction = original_prediction.map(remap_dict)

    if new_prediction.isna().any():
        missing_labels = sorted(
            original_prediction[new_prediction.isna()].unique().tolist()
        )
        raise ValueError(
            "Some Prediction labels could not be remapped: "
            f"{missing_labels}"
        )

    # Keep the original GMM label for traceability.
    if "Prediction_original" not in remapped_df.columns:
        remapped_df.insert(
            remapped_df.columns.get_loc(prediction_column),
            "Prediction_original",
            original_prediction,
        )
    else:
        remapped_df["Prediction_original"] = original_prediction

    # Replace Prediction with the new ordered cluster number.
    remapped_df[prediction_column] = new_prediction.astype(int)


    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    remapped_output_file = OUTPUT_PLOTS_DIR / "FULL_CONCAT_clusters.csv"
    mapping_output_file = OUTPUT_PLOTS_DIR / "Cluster_remapping_by_ascini_position.csv"

    remapped_df.to_csv(remapped_output_file, index=False)
    cluster_mapping.to_csv(mapping_output_file, index=False)

    print("\nCluster remapping based on ascini_position:")
    print(cluster_mapping.to_string(index=False))

    print(f"\nSaved remapped table to: {remapped_output_file}")
    print(f"Saved cluster mapping to: {mapping_output_file}")

    return remapped_df, cluster_mapping       
        

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    raw_data, clean_data = load_and_filter_data(INPUT_CSV)
    feature_matrix, pca_feature_columns = build_pca_feature_matrix(clean_data)
    pca_df, pca, _ = run_pca(feature_matrix)

    gmm, gmm_input, labels, probabilities = fit_gmm_on_pc1_pc2(pca_df)
    save_gmm_parameters(gmm)

    print(gmm.get_params())
    print(probabilities[:5].round(3))

    full_df = save_tables(
        raw_data=raw_data,
        clean_data=clean_data,
        pca_df=pca_df,
        pca_feature_columns=pca_feature_columns,
        pca=pca,
        probabilities=probabilities,
        labels=labels,
    )
    
    # -----------------------------------------------------------------------------
    # Remap Prediction labels based on ascini_position direction.
    # This creates FULL_CONCAT_clusters.csv with categories 1..N.
    # -----------------------------------------------------------------------------
    full_df, cluster_mapping = remap_prediction_by_ascini_position(
        full_df=full_df,
        gmm=gmm,
        position_column="ascini_position",
        prediction_column=HUE_COLUMN,
        extreme_fraction=0.10,
    )

    # -----------------------------------------------------------------------------
    # Orient PC1 for display so category 1 is on the left and the last category
    # is on the right. This does not refit PCA or GMM.
    # -----------------------------------------------------------------------------
    (
        full_df_for_plot,
        gmm_input_for_plot,
        gmm_for_plot,
        cluster_mapping_for_plot,
        pc1_flipped,
        plot_xlim_for_display,
    ) = orient_pc1_by_category_order(
        full_df=full_df,
        gmm_input=gmm_input,
        gmm=gmm,
        cluster_mapping=cluster_mapping,
        category_column=ORIENTATION_CATEGORY_COLUMN,
        plot_xlim=PLOT_XLIM,
        auto_orient=ORIENT_PC1_BY_CATEGORY_ORDER,
        save_outputs=SAVE_ORIENTED_FULL_CONCAT,
    )

    prediction_palette, prediction_hue_order = make_ordered_prediction_palette(
        n_clusters=N_CLUSTERS
    )

    

    plot_indices = full_df_for_plot.sample(
        frac=PLOT_FRACTION,
        random_state=PLOT_RANDOM_STATE
    ).index.sort_values()

    full_df_plot = full_df_for_plot.loc[plot_indices].copy()
    gmm_input_plot = gmm_input_for_plot.loc[plot_indices].copy()
    probabilities_plot = probabilities[plot_indices.to_numpy(), :]
    labels_plot = full_df_plot[HUE_COLUMN].astype(int).to_numpy()

    make_plots_same_as_before(
        full_df_plot,
        gmm_for_plot,
        gmm_input_plot,
        probabilities_plot,
        prediction_palette=prediction_palette,
        prediction_hue_order=prediction_hue_order,
        labels_override=labels_plot,
        plot_xlim=plot_xlim_for_display,
    )
    
    

if __name__ == "__main__":
    main()
