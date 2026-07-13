"""
Generate three PC1 trajectory figures from the H4/FH pseudotime dataset:

1. PC1_vs_Pseudotime.png
2. PC1_Loadings.png
3. PC1_Spatial_Modulation.png

Workflow
--------
- Correlate numeric features with the selected pseudotime column.
- Keep features with |Spearman rho| >= FEATURE_RHO_THRESHOLD.
- Z-score those features and run PCA.
- Save PC scores and PCA loadings.
- Merge acinar position from the full dataset.
- Generate only the three requested figures, preserving the original style.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


# =============================================================================
# EDITABLE SETTINGS
# =============================================================================

PSEUDOTIME_FILE = Path(
    r"...add path...\Fig5_Perturbation_Data_Matrix__best_pairs\cells_with_pseudotime_H4_FH_only.csv"
)

FULL_DATA_FILE = Path(
     r"...add path...\Full_data_perturbations.csv"
    
)

OUTPUT_DIR = Path(
    r"...add path...\Fig5_Perturbation_Data_Matrix__best_pairs"
)

PSEUDOTIME_COL = "pseudotime_full"
FEATURE_RHO_THRESHOLD = 0.3 # Set 0.3 for stringency. 
#Note: results are still concistent even with RHO_THRESHOLD = 0.0. 

N_PCA_COMPONENTS = 5
N_POSITION_BINS = 8

# Files saved in the parent analysis directory.
ANALYSIS_DIR = OUTPUT_DIR.parent
CORRELATION_FILE = ANALYSIS_DIR / f"feature_{PSEUDOTIME_COL}_correlations.csv"
PC_SCORE_FILE = ANALYSIS_DIR / "cells_with_pseudotime_H4_FH_only_with_PCs.csv"
PCA_LOADING_FILE = ANALYSIS_DIR / "PCA_loadings.csv"

# Columns that must never be treated as candidate biological features.
EXCLUDED_NUMERIC_COLUMNS = {
    "pseudotime_H4_FH_only",
    "pseudotime_full",
    "CAT",
}


# =============================================================================
# HELPERS
# =============================================================================

def clean_feature_name(name: str) -> str:
    """Convert raw column names into display labels used in the loading plot."""
    replacements = {
        "mito": "mitochondria",
        "ld": "LD",
    }

    words = name.split("_")
    cleaned_words = [replacements.get(word.lower(), word) for word in words]
    return " ".join(cleaned_words).title().replace("Ld", "LD")


def validate_columns(df: pd.DataFrame, required: list[str], dataframe_name: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def calculate_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlations between numeric features and pseudotime."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        column
        for column in numeric_cols
        if column not in EXCLUDED_NUMERIC_COLUMNS
    ]

    results = []

    for feature in feature_cols:
        valid = df[[PSEUDOTIME_COL, feature]].dropna()

        if len(valid) < 3 or valid[feature].nunique() < 2:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = spearmanr(
                valid[PSEUDOTIME_COL],
                valid[feature],
            )

        results.append((feature, rho, p_value))

    results_df = pd.DataFrame(
        results,
        columns=["feature", "rho", "pval"],
    )

    valid_p = results_df["pval"].notna()
    results_df["pval_adj"] = np.nan
    results_df["significant"] = False

    if valid_p.any():
        reject, adjusted, _, _ = multipletests(
            results_df.loc[valid_p, "pval"],
            method="fdr_bh",
        )
        results_df.loc[valid_p, "pval_adj"] = adjusted
        results_df.loc[valid_p, "significant"] = reject

    results_df["abs_rho"] = results_df["rho"].abs()
    return results_df.sort_values("abs_rho", ascending=False)


def run_trajectory_pca(
    df: pd.DataFrame,
    correlation_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Run PCA on features whose absolute correlation exceeds the threshold."""
    selected_features = correlation_results.loc[
        correlation_results["abs_rho"] >= FEATURE_RHO_THRESHOLD,
        "feature",
    ].tolist()

    if len(selected_features) < 2:
        raise ValueError(
            "Too few features passed the correlation threshold. "
            f"Found {len(selected_features)} feature(s) with "
            f"|rho| >= {FEATURE_RHO_THRESHOLD}."
        )

    pca_input = df[selected_features].copy()

    if pca_input.isna().any().any():
        missing_counts = pca_input.isna().sum()
        missing_features = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            "Selected PCA features contain missing values. "
            f"Missing counts: {missing_features}"
        )

    n_components = min(
        N_PCA_COMPONENTS,
        len(selected_features),
        len(df),
    )

    scaled = StandardScaler().fit_transform(pca_input)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    output_df = df.copy()
    for index in range(n_components):
        output_df[f"PC{index + 1}"] = scores[:, index]

    loading_columns = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=selected_features,
        columns=loading_columns,
    )

    print(
        f"Number of features with |rho| >= {FEATURE_RHO_THRESHOLD}: "
        f"{len(selected_features)}"
    )
    print("Explained variance ratio:")
    for index, ratio in enumerate(pca.explained_variance_ratio_[:3], start=1):
        print(f"  PC{index}: {ratio:.3f}")

    return output_df, loadings, pca


def merge_acinar_position(df: pd.DataFrame) -> pd.DataFrame:
    """Merge acinar position and define PV/CV side from its sign."""
    full_data = pd.read_csv(FULL_DATA_FILE)
    validate_columns(full_data, ["labels", "ascini_position"], "Full dataset")

    position_lookup = full_data[["labels", "ascini_position"]].drop_duplicates(
        subset="labels"
    )

    merged = df.merge(
        position_lookup,
        on="labels",
        how="left",
        validate="many_to_one",
    )

    merged["zone"] = np.where(
        merged["ascini_position"] >= 0,
        "PV_side",
        "CV_side",
    )

    print("Missing ascini_position:", merged["ascini_position"].isna().sum())
    return merged


# =============================================================================
#PC1 VS PSEUDOTIME
# =============================================================================

def plot_pc1_vs_pseudotime(df: pd.DataFrame) -> None:
    """Create the final cubehelix PC1-versus-pseudotime figure."""
    valid = df[[PSEUDOTIME_COL, "PC1"]].dropna()
    rho, _ = spearmanr(valid[PSEUDOTIME_COL], valid["PC1"])

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)

    plt.figure(figsize=(4, 5))
    plt.scatter(
        df[PSEUDOTIME_COL],
        df["PC1"],
        c=df[PSEUDOTIME_COL],
        cmap=cmap,
        s=12,
        alpha=0.6,
        edgecolor="none",
    )

    sns.regplot(
        data=df,
        x=PSEUDOTIME_COL,
        y="PC1",
        scatter=False,
        lowess=True,
        color="black",
        line_kws={"linewidth": 1.2, "alpha": 0.5},
    )

    plt.xlabel("Pseudotime")
    plt.ylabel("PC1 (Remodeling Axis)")
    plt.title(f"PC1 vs Pseudotime (ρ = {rho:.2f})")

    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "PC1_vs_Pseudotime.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# TOP PC1 LOADINGS
# =============================================================================

def plot_pc1_loadings(loadings: pd.DataFrame) -> None:
    """Create the top positive and negative PC1 loading bar plot."""
    top_features = (
        loadings["PC1"]
        .abs()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    pc1_values = loadings.loc[top_features, "PC1"].sort_values()
    display_names = [clean_feature_name(name) for name in pc1_values.index]

    positive_color = "#2C3E50"
    negative_color = "#7F8C8D"
    bar_colors = [
        positive_color if value > 0 else negative_color
        for value in pc1_values.values
    ]

    plt.figure(figsize=(5, 5))
    plt.barh(
        display_names,
        pc1_values.values,
        color=bar_colors,
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.xlim(-0.2, 0.2)
    plt.xlabel("PC1 Loading")

    axis = plt.gca()
    axis.yaxis.tick_right()
    axis.yaxis.set_label_position("right")
    axis.tick_params(axis="y", labelsize=8)

    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "PC1_Loadings.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# SPATIAL MODULATION OF PC1
# =============================================================================

def plot_pc1_spatial_modulation(df: pd.DataFrame) -> None:
    """Create the binned PV/CV spatial-modulation figure."""
    plot_df = df.dropna(subset=["ascini_position", "PC1", "zone"]).copy()

    plot_df["position_bin"] = plot_df.groupby("zone", observed=True)[
        "ascini_position"
    ].transform(
        lambda values: pd.qcut(
            values,
            q=N_POSITION_BINS,
            duplicates="drop",
        )
    )

    binned = (
        plot_df.groupby(["zone", "position_bin"], observed=True)
        .agg(
            mean_position=("ascini_position", "mean"),
            mean_pc1=("PC1", "mean"),
            sem_pc1=("PC1", "sem"),
        )
        .reset_index()
        .sort_values("mean_position")
    )

    cv_data = binned[binned["zone"] == "CV_side"].sort_values("mean_position")
    pv_data = binned[binned["zone"] == "PV_side"].sort_values("mean_position")

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
    pv_color = cmap(0.05)
    cv_color = cmap(0.95)

    line_width = 1.9
    line_alpha = 0.9
    shade_alpha = 0.2
    divider_alpha = 0.5
    divider_width = 0.8

    plt.figure(figsize=(4, 5))

    plt.plot(
        cv_data["mean_position"],
        cv_data["mean_pc1"],
        color=cv_color,
        linewidth=line_width,
        alpha=line_alpha,
    )
    plt.fill_between(
        cv_data["mean_position"],
        cv_data["mean_pc1"] - cv_data["sem_pc1"],
        cv_data["mean_pc1"] + cv_data["sem_pc1"],
        color=cv_color,
        alpha=shade_alpha,
    )

    plt.plot(
        pv_data["mean_position"],
        pv_data["mean_pc1"],
        color=pv_color,
        linewidth=line_width,
        alpha=line_alpha,
    )
    plt.fill_between(
        pv_data["mean_position"],
        pv_data["mean_pc1"] - pv_data["sem_pc1"],
        pv_data["mean_pc1"] + pv_data["sem_pc1"],
        color=pv_color,
        alpha=shade_alpha,
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=divider_width,
        alpha=divider_alpha,
    )

    plt.gca().invert_xaxis()
    plt.xlabel("Relative Acinar Position")
    plt.ylabel("Mean PC1")
    plt.title("Spatial Modulation of Remodeling Axis")

    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "PC1_Spatial_Modulation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set(style="white", context="talk")

    df = pd.read_csv(PSEUDOTIME_FILE)
    validate_columns(df, ["labels", PSEUDOTIME_COL], "Pseudotime dataset")

    correlation_results = calculate_feature_correlations(df)
    correlation_results.to_csv(CORRELATION_FILE, index=False)

    df_with_pcs, loadings, _ = run_trajectory_pca(df, correlation_results)
    loadings.to_csv(PCA_LOADING_FILE)

    df_with_position = merge_acinar_position(df_with_pcs)
    df_with_position.to_csv(PC_SCORE_FILE, index=False)

    plot_pc1_vs_pseudotime(df_with_position)
    plot_pc1_loadings(loadings)
    plot_pc1_spatial_modulation(df_with_position)

    print("\nGenerated figures:")
    print("  PC1_vs_Pseudotime.png")
    print("  PC1_Loadings.png")
    print("  PC1_Spatial_Modulation.png")
    print(f"\nSaved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
