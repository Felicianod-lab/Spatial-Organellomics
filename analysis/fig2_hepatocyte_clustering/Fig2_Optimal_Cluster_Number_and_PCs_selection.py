import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score, adjusted_rand_score
from itertools import combinations
# Set seaborn style
sns.set()

# ----------------------------
# Constants and Configuration
# ----------------------------


DATA_PATH = r"...add path...\Fig2_Hepatocyte _Clustering_Group_Control_data .csv"

MITO_COUNT_THRESHOLD = 20
PCA_COMPONENTS = 2
CV_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = -1

# Selection settings
# Cluster number selection stays based on the existing combined score.
# PC set selection uses ARI_mean penalized by ARI_std.
CLUSTER_SELECTION_METRIC = "combined_score"
PC_SELECTION_METRIC = "ari_penalized_then_combined"
ARI_STD_PENALTY_WEIGHT = 1.0

# Evaluation Weights
weights = {
    "log_likelihood": 0.05,
    "bic": 0.25,
    "aic": 0.2,
    "silhouette": 0.15,
    "cv_score": 0.35
}

# ----------------------------
# Load and Preprocess Data
# ----------------------------
data = pd.read_csv(DATA_PATH, na_values='-')
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)
data.dropna(inplace=True)


# ----------------------------
# Filter Data
# ----------------------------

data_filtered = data[data["mito_aspect_ratio"] > 0]
# Compute mito_count and apply threshold
mito_count = data_filtered["mito_density"] * data_filtered["area"]
data_filtered = data_filtered[mito_count >= MITO_COUNT_THRESHOLD].copy()

# ----------------------------
# Feature Selection
# ----------------------------
excluded_columns = [
    "area", "centroid-0", "centroid-1",  "labels",
    "type_1_ld_avg_aspect_ratio", "type_2_ld_avg_aspect_ratio",
    "type_3_ld_avg_aspect_ratio", "type_4_ld_avg_aspect_ratio",
    "type_1_ld_avg_solidity", "type_2_ld_avg_solidity",
    "type_3_ld_avg_solidity", "type_4_ld_avg_solidity",
    "stack_id", "cell_id_linked", "ascini_position",
                           
]

data_features = data_filtered.drop(columns=excluded_columns)
data_features = data_features.select_dtypes(include=[np.number])  # Ensure numerical only


# ----------------------------
# Normalize and Reduce Dimensionality
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_features)

# Test cumulative PC sets: PC1-PC2, PC1-PC3, PC1-PC4, etc.
MIN_PCS_TO_TEST = 2
MAX_PCS_TO_TEST = 10        # Change this if you want more or fewer PCs tested
MAX_CLUSTERS = 10
N_STABILITY_RUNS = 50       # More runs = better stability estimate, but slower
STABILITY_SUBSAMPLE_FRACTION = 0.8

max_possible_pcs = min(MAX_PCS_TO_TEST, X_scaled.shape[1], X_scaled.shape[0] - 1)

pca = PCA(n_components=max_possible_pcs, random_state=RANDOM_STATE)
X_pca_all = pd.DataFrame(
    pca.fit_transform(X_scaled),
    columns=[f"PC{i}" for i in range(1, max_possible_pcs + 1)]
)

explained_variance = pd.DataFrame({
    "PC": [f"PC{i}" for i in range(1, max_possible_pcs + 1)],
    "Explained_Variance_Ratio": pca.explained_variance_ratio_,
    "Cumulative_Explained_Variance": np.cumsum(pca.explained_variance_ratio_)
})

print("\nExplained variance by PCA component:")
print(explained_variance)


# ----------------------------
# Helper Functions
# ----------------------------
def normalize(array):
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


def calculate_metrics_for_X(n, X_train):
    """
    Fit GMM with n clusters and calculate:
    log-likelihood, BIC, AIC, silhouette, and CV score.
    """
    try:
        gmm = GaussianMixture(
            n_components=n,
            random_state=RANDOM_STATE,
            n_init=5
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

        # Cross-validation
        fold_scores = []
        n_splits = min(CV_SPLITS, len(X_train))

        if n_splits >= 2:
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE
            )

            for train_index, val_index in kf.split(X_train):
                X_train_fold = X_train.iloc[train_index]
                X_val_fold = X_train.iloc[val_index]

                # GMM needs more training samples than components
                if len(X_train_fold) <= n:
                    continue

                gmm_fold = GaussianMixture(
                    n_components=n,
                    random_state=RANDOM_STATE,
                    n_init=5
                )

                gmm_fold.fit(X_train_fold)
                fold_scores.append(gmm_fold.score(X_val_fold) * len(X_val_fold))

        cv_score = np.mean(fold_scores) if len(fold_scores) > 0 else np.nan

        return log_likelihood, bic, aic, silhouette, cv_score

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def find_optimal_clusters_for_pc_set(X_pc, n_clusters_range):
    """
    For one PC set, evaluate all cluster numbers and return:
    - optimal cluster number
    - metrics dataframe
    """
    X_train, X_val = train_test_split(
        X_pc,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    valid_cluster_range = [
        n for n in n_clusters_range
        if n < len(X_train)
    ]

    results = Parallel(n_jobs=N_JOBS)(
        delayed(calculate_metrics_for_X)(n, X_train)
        for n in valid_cluster_range
    )

    log_likelihoods, bics, aics, silhouettes, cv_scores = zip(*results)

    metrics_df = pd.DataFrame({
        "n_clusters": valid_cluster_range,
        "log_likelihood": log_likelihoods,
        "bic": bics,
        "aic": aics,
        "silhouette": silhouettes,
        "cv_score": cv_scores
    })

    # Normalize metrics
    metrics_df["log_likelihood_norm"] = normalize(metrics_df["log_likelihood"])
    metrics_df["bic_norm"] = normalize(metrics_df["bic"])
    metrics_df["aic_norm"] = normalize(metrics_df["aic"])
    metrics_df["silhouette_norm"] = normalize(metrics_df["silhouette"])
    metrics_df["cv_score_norm"] = normalize(metrics_df["cv_score"])

    # Combine scores
    # Higher is better for log-likelihood, silhouette, and CV score.
    # Lower is better for BIC and AIC, so we use 1 - normalized value.
    metrics_df["combined_score"] = (
        weights["log_likelihood"] * np.nan_to_num(metrics_df["log_likelihood_norm"], nan=0.0) +
        weights["bic"] * (1 - np.nan_to_num(metrics_df["bic_norm"], nan=1.0)) +
        weights["aic"] * (1 - np.nan_to_num(metrics_df["aic_norm"], nan=1.0)) +
        weights["silhouette"] * np.nan_to_num(metrics_df["silhouette_norm"], nan=0.0) +
        weights["cv_score"] * np.nan_to_num(metrics_df["cv_score_norm"], nan=0.0)
    )

    # Cluster-number selection stays independent from ARI stability.
    # With CLUSTER_SELECTION_METRIC = "combined_score", this is the same behavior
    # as the original script.
    best_idx = metrics_df[CLUSTER_SELECTION_METRIC].idxmax()
    optimal_clusters = int(metrics_df.loc[best_idx, "n_clusters"])

    return optimal_clusters, metrics_df


def calculate_ari_stability(
    X_pc,
    n_clusters,
    n_runs=50,
    subsample_fraction=0.7,
    random_state=42
):
    """
    Estimate cluster stability using ARI.

    Method:
    - Repeatedly subsample the data.
    - Fit a GMM on each subsample.
    - Predict cluster labels for the full dataset.
    - Compare all pairs of full-dataset labelings using ARI.

    Higher ARI means more stable clustering.
    ARI close to 1 means highly stable.
    ARI close to 0 means weak stability.
    """
    rng = np.random.default_rng(random_state)

    X_np = X_pc.to_numpy()
    n_samples = X_np.shape[0]

    sample_size = int(np.floor(subsample_fraction * n_samples))
    sample_size = max(sample_size, n_clusters + 1)
    sample_size = min(sample_size, n_samples)

    all_labels = []

    for run in range(n_runs):
        sample_idx = rng.choice(
            n_samples,
            size=sample_size,
            replace=False
        )

        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state + run,
            n_init=5
        )

        gmm.fit(X_np[sample_idx])

        # Predict all cells using the model trained on the subsample
        labels_full = gmm.predict(X_np)
        all_labels.append(labels_full)

    ari_scores = []

    for i, j in combinations(range(len(all_labels)), 2):
        ari = adjusted_rand_score(all_labels[i], all_labels[j])
        ari_scores.append(ari)

    ari_scores = np.array(ari_scores)

    return {
        "ARI_mean": np.mean(ari_scores),
        "ARI_std": np.std(ari_scores),
        "ARI_min": np.min(ari_scores),
        "ARI_max": np.max(ari_scores),
        "ARI_n_comparisons": len(ari_scores)
    }


def calculate_ari_penalized_score(
    ari_mean,
    ari_std,
    ari_std_penalty_weight=ARI_STD_PENALTY_WEIGHT,
    n_clusters=None
):
    """
    Penalize ARI stability by its run-to-run variability.

    Formula:
        ARI_penalized_score = ARI_mean - ari_std_penalty_weight * ARI_std

    This score is used only for final PC-set selection. It does not affect
    cluster-number selection.
    """
    # Avoid rewarding a trivial one-cluster solution for being artificially stable.
    if n_clusters is not None and int(n_clusters) < 2:
        return 0.0

    score = (
        np.nan_to_num(ari_mean, nan=0.0)
        - ari_std_penalty_weight * np.nan_to_num(ari_std, nan=0.0)
    )

    return float(np.clip(score, 0.0, 1.0))


# ----------------------------
# Test Different PC Sets
# ----------------------------
n_clusters_range = range(1, MAX_CLUSTERS + 1)

pc_summary = []
all_pc_metrics = {}

for n_pcs in range(MIN_PCS_TO_TEST, max_possible_pcs + 1):

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    X_pc = X_pca_all[pc_cols].copy()

    print(f"\nTesting PC set: PC1 to PC{n_pcs}")

    optimal_clusters, metrics_df = find_optimal_clusters_for_pc_set(
        X_pc,
        n_clusters_range
    )

    stability = calculate_ari_stability(
        X_pc,
        n_clusters=optimal_clusters,
        n_runs=N_STABILITY_RUNS,
        subsample_fraction=STABILITY_SUBSAMPLE_FRACTION,
        random_state=RANDOM_STATE
    )

    best_combined_score = metrics_df.loc[
        metrics_df["n_clusters"] == optimal_clusters,
        "combined_score"
    ].values[0]

    ari_penalized_score = calculate_ari_penalized_score(
        stability["ARI_mean"],
        stability["ARI_std"],
        ari_std_penalty_weight=ARI_STD_PENALTY_WEIGHT,
        n_clusters=optimal_clusters
    )

    cumulative_variance = np.sum(pca.explained_variance_ratio_[:n_pcs])

    pc_summary.append({
        "PC_set": f"PC1-PC{n_pcs}",
        "n_pcs": n_pcs,
        "optimal_clusters": optimal_clusters,
        "cluster_selection_metric": CLUSTER_SELECTION_METRIC,
        "pc_selection_metric": PC_SELECTION_METRIC,
        "best_combined_score": best_combined_score,
        "ARI_mean": stability["ARI_mean"],
        "ARI_std": stability["ARI_std"],
        "ARI_penalized_score": ari_penalized_score,
        "ARI_std_penalty_weight": ARI_STD_PENALTY_WEIGHT,
        "ARI_min": stability["ARI_min"],
        "ARI_max": stability["ARI_max"],
        "ARI_n_comparisons": stability["ARI_n_comparisons"],
        "cumulative_explained_variance": cumulative_variance
    })

    all_pc_metrics[n_pcs] = metrics_df

    print(
        f"PC1-PC{n_pcs}: "
        f"optimal clusters = {optimal_clusters}, "
        f"mean ARI = {stability['ARI_mean']:.3f}, "
        f"ARI penalty score = {ari_penalized_score:.3f}, "
        f"cumulative variance = {cumulative_variance:.3f}"
    )


pc_summary_df = pd.DataFrame(pc_summary)


print("\nSummary of PC-set clustering results:")
print(pc_summary_df)



# ---------------------------------------
# Plot ARI Bar Graphs for PC Stability
# ---------------------------------------
def plot_pc_ari_bars(pc_results, modality=None, output_path=None, dpi=400):
    """
    Plot ARI mean with std error bars for each number of PCs.

    This function accepts either:
        1. pc_summary_df from the PCA-testing block:
            columns = ["n_pcs", "ARI_mean", "ARI_std"]

        2. pc_results from the attached modality-aware pipeline:
            columns = ["n_pc", "ari_mean", "ari_std"]

    Parameters:
        pc_results (pd.DataFrame)
        modality (str, optional): e.g. "CNT", "organellomics", "RNA", etc.
        output_path (str, optional): path to save the figure
        dpi (int): figure resolution if saving
    """
 

    df = pc_results.copy()

    # ----------------------------
    # Make compatible column names
    # ----------------------------
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
        alpha=0.5
    )

    plt.xlabel("Number of PCs", fontsize=14)
    plt.ylabel("ARI (mean)", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)

    # Title with modality
    if modality:
        title = f"{modality}"
    else:
        title = "Clustering Stability vs PCA Dimensionality"

    plt.title(title, fontsize=8)

    plt.xticks(x)
    plt.ylim(0, 1)

    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        print(f"[INFO] Saved plot to: {output_path}")

    plt.show()


# ----------------------------
# Select Final PC Set
# ----------------------------

# Choose the final PC set.
#
# With PC_SELECTION_METRIC = "ari_penalized_then_combined", the selected PC set
# is the one with the highest ARI_penalized_score, using best_combined_score
# only as the secondary tie-breaker.
if PC_SELECTION_METRIC == "ari_penalized_then_combined":
    best_pc_row = pc_summary_df.sort_values(
        by=["ARI_penalized_score", "best_combined_score"],
        ascending=[False, False]
    ).iloc[0]
elif PC_SELECTION_METRIC == "ari_then_combined":
    best_pc_row = pc_summary_df.sort_values(
        by=["ARI_mean", "best_combined_score"],
        ascending=[False, False]
    ).iloc[0]
else:
    raise ValueError(
        "PC_SELECTION_METRIC must be 'ari_penalized_then_combined' "
        "or 'ari_then_combined'."
    )

selected_n_pcs = int(best_pc_row["n_pcs"])
selected_clusters = int(best_pc_row["optimal_clusters"])

print("\nSelected final model:")
print(f"PC set: PC1-PC{selected_n_pcs}")
print(f"Optimal clusters: {selected_clusters}")
print(f"Mean ARI stability: {best_pc_row['ARI_mean']:.3f}")
print(f"ARI std: {best_pc_row['ARI_std']:.3f}")
print(f"ARI penalized score: {best_pc_row['ARI_penalized_score']:.3f}")
print(f"Cumulative explained variance: {best_pc_row['cumulative_explained_variance']:.3f}")


# ----------------------------
# Final Model Fit and Visualization
# ----------------------------
selected_pc_cols = [f"PC{i}" for i in range(1, selected_n_pcs + 1)]
X_selected = X_pca_all[selected_pc_cols].copy()

gmm_final = GaussianMixture(
    n_components=selected_clusters,
    random_state=RANDOM_STATE,
    n_init=5
)

final_labels = gmm_final.fit_predict(X_selected)

X_plot = X_pca_all[["PC1", "PC2"]].copy()
X_plot["Cluster"] = final_labels

plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=X_plot,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=100
)
plt.title(
    f'GMM Clustering using PC1-PC{selected_n_pcs}, '
    f'{selected_clusters} Clusters',
    fontsize=20
)

plt.show()




plot_pc_ari_bars(
    pc_summary_df,
    modality=None,
    output_path="CNT_PC_ARI_stability_barplot.png",
    dpi=400
)

