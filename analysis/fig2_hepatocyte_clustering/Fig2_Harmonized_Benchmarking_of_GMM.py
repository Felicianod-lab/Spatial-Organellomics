# -*- coding: utf-8 -*-
"""
Modality-Aware GMM Benchmarking Pipeline

- Preprocess via Scanpy AnnData routines per modality (RNA / Protein / Organelle)
  OR a generic CSV path (log-detect + scale).
- Feature budget: select top-variance features on layers['pre_scale'] (scaling-insensitive)
- PCA -> GMM with multi-metric model selection (seeded, CV, stability)
- Robustness (bootstrap ARI), Separation (overlap & Mahalanobis), Resolution (silhouette & DB),
  Uncertainty (soft cluster entropy), Effective Resolution (granularity + k±1 stability), Composite score
- Summary JSON per modality + a cross-modality ranking table (CSV).
- All generated files are automatically saved by default in a Results folder beside this script.

Usage:
  - Set use_scanpy_preproc=True to route through the AnnData preprocessors.
  - Control feature budget with `feature_budget` (default 98) or CLI flag `--feature-budget`.
  - You do not need to provide output paths; default output names are routed into Results/.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Create the script-local Results folder immediately, before heavy imports.
# This means the folder appears even if a later package import fails.
# -----------------------------------------------------------------------------
from pathlib import Path

RESULTS_FOLDER_NAME = "GMM_Results"

def get_script_dir() -> Path:
    """Return the folder containing this script; fall back to cwd in interactive use."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def get_results_dir() -> Path:
    """Create and return ./Results next to this script."""
    results_dir = get_script_dir() / RESULTS_FOLDER_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    marker = results_dir / "README_Results_folder_created_by_GMM_script.txt"
    if not marker.exists():
        marker.write_text(
            "This folder was created automatically by the GMM benchmarking script.\n"
            "All generated CSV, JSON, and PNG outputs are saved here by default.\n",
            encoding="utf-8",
        )
    return results_dir

DEFAULT_RESULTS_DIR = get_results_dir()
print(f"[INFO] Results folder is: {DEFAULT_RESULTS_DIR}", flush=True)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

from typing import Optional, Iterable, Sequence, Tuple, Dict, Any, List
from scipy import sparse, linalg as la

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed


try:
    from anndata import AnnData
    import scanpy as sc
    _SCANPY_IMPORT_ERROR = None
except Exception as _e:
    AnnData = None  # type: ignore[assignment]
    sc = None       # type: ignore[assignment]
    _SCANPY_IMPORT_ERROR = _e


def _require_scanpy_anndata():
    if AnnData is None or sc is None:
        raise ImportError(
            "scanpy and anndata are required for modality-aware preprocessing. "
            "Install them with `pip install scanpy anndata`, or run without "
            "--use-scanpy-preproc if your input is already suitable for the generic PCA path."
        ) from _SCANPY_IMPORT_ERROR


# =============================================================================
# -------------------------- Output folder helpers ----------------------------
# =============================================================================

# RESULTS_FOLDER_NAME, get_script_dir(), get_results_dir(), and DEFAULT_RESULTS_DIR
# are defined at the very top of the file so the folder is created before any
# heavy package imports can fail.


def default_output_filename_for_modality(modality: str) -> str:
    """Return the standard output prefix used when no output filename is supplied."""
    m = str(modality).lower()
    if "organell" in m:
        return "organellomics_clusters.csv"
    if "transcript" in m or m == "rna":
        return "rna_clusters.csv"
    if "protein" in m or "proteom" in m:
        return "protein_clusters.csv"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(modality)).strip("_")
    return f"{safe or 'modality'}_clusters.csv"


def default_summary_files() -> List[str]:
    """Default summary JSON files used when --rank is supplied without paths."""
    return [
        "organellomics_clusters_summary.json",
        "rna_clusters_summary.json",
        "protein_clusters_summary.json",
    ]


def output_path_in_results(path_like: str) -> str:
    """Place an output file in the script-local Results folder, using its basename."""
    name = Path(str(path_like)).name
    if not name:
        raise ValueError("Output filename cannot be empty.")
    return str(DEFAULT_RESULTS_DIR / name)


def output_csv_prefix_in_results(output_csv: str) -> str:
    """Return a CSV-named output prefix inside Results, regardless of input directory."""
    name = Path(str(output_csv)).name
    if not name.lower().endswith(".csv"):
        name = f"{name}.csv"
    return str(DEFAULT_RESULTS_DIR / name)


def derived_output_path(output_csv: str, suffix: str) -> str:
    """Create a derived filename beside output_csv, e.g. *_pca.csv or *_summary.json."""
    p = Path(output_csv)
    return str(p.with_name(f"{p.stem}{suffix}"))


def resolve_input_or_results(path_like: str) -> str:
    """Resolve a supplied input path; if missing, try Results/<basename>."""
    p = Path(str(path_like))
    if p.exists():
        return str(p)
    candidate = DEFAULT_RESULTS_DIR / p.name
    if candidate.exists():
        return str(candidate)
    return str(p)


# =============================================================================
# ----------------------------- JSON helpers ----------------------------------
# =============================================================================

def json_safe(obj: Any):
    """Recursively convert numpy/scipy types to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    # Fallback (e.g., pandas scalars)
    try:
        return json_safe(obj.item())  # type: ignore[attr-defined]
    except Exception:
        return str(obj)


# =============================================================================
# ----------------------- Modality-specific preprocessors ---------------------
# =============================================================================

def preprocess_rna(
    adata: AnnData,
    n_top_genes: Optional[int] = 2000,
    n_pcs: Optional[int] = 50,
    scale_max: float = 10.0,
    metric: str = "cosine",
    seed: int = 0,
) -> AnnData:
    _require_scanpy_anndata()
    A = adata.copy()
    if "counts" not in A.layers:
        A.layers["counts"] = A.X.copy()
    norm = sc.pp.normalize_total(A, target_sum=1e4, inplace=False)  # TP10K
    A.layers["tp10k"] = norm["X"]
    A.layers["log1p_tp10k"] = A.layers["tp10k"].copy()
    sc.pp.log1p(A, layer="log1p_tp10k", base=None)

    # HVG selection on counts-like layer (tp10k). Then proceed on log1p layer.
    if n_top_genes is not None and A.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(
            A, flavor="seurat_v3", n_top_genes=n_top_genes, inplace=True, layer="tp10k"
        )
        A = A[:, A.var["highly_variable"]].copy()

    A.X = A.layers["log1p_tp10k"]
    A.layers["pre_scale"] = A.X.copy()
    sc.pp.scale(A, max_value=scale_max)
    if n_pcs is not None:
        sc.tl.pca(A, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    A.uns["preferred_metric"] = metric
    return A


def preprocess_protein(
    adata: AnnData,
    arcsinh_cofactor: float = 5.0,        # kept for API compatibility (ignored)
    n_pcs: Optional[int] = 30,
    scale: bool = False,                  # default False: center-only
    metric: str = 'euclidean',
    seed: int = 0,
    do_median_normalize: bool = True,
    norm_set: Optional[Iterable[str]] = None,
    norm_set_size: int = 175,
    coverage_steps: Sequence[float] = (1.00, 0.995, 0.99, 0.98, 0.97, 0.95, 0.90),
    eps: float = 1e-12,
) -> AnnData:
    _require_scanpy_anndata()
    A = adata.copy()

    def _to_dense(X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)

    def _choose_norm_set(X_dense: np.ndarray, var_names: np.ndarray) -> np.ndarray:
        present_counts = (X_dense > 0).sum(axis=0)
        n_cells = X_dense.shape[0]
        for cov in coverage_steps:
            mask = present_counts >= cov * n_cells
            if mask.sum() >= norm_set_size:
                U = X_dense[:, mask]
                med = np.median(U, axis=0)
                sd = U.std(axis=0, ddof=1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    cv = sd / np.where(med == 0, np.nan, med)
                idx_sorted = np.argsort(np.nan_to_num(cv, nan=np.inf))
                chosen_cols = np.flatnonzero(mask)[idx_sorted[:norm_set_size]]
                return chosen_cols
        mask = present_counts >= coverage_steps[-1] * n_cells
        chosen_cols = np.flatnonzero(mask)[:norm_set_size]
        return chosen_cols

    X = _to_dense(A.X).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=None, neginf=0.0)
    var_names = A.var_names.to_numpy()

    if do_median_normalize:
        if norm_set is not None:
            norm_set = [v for v in norm_set if v in A.var_names]
            if len(norm_set) == 0:
                raise ValueError("Provided 'norm_set' has no overlap with adata.var_names.")
            norm_idx = np.array([A.var_names.get_loc(v) for v in norm_set], dtype=int)
            used_norm_names = np.array(norm_set, dtype=object)
        else:
            norm_idx = _choose_norm_set(X, var_names)
            used_norm_names = var_names[norm_idx]
        ref = X[:, norm_idx]
        pos_mask = ref > 0
        with np.errstate(invalid='ignore'):
            sample_medians = np.nanmedian(np.where(pos_mask, ref, np.nan), axis=1)
        if np.any(~np.isfinite(sample_medians)):
            global_pos = ref[ref > 0]
            global_med = float(np.median(global_pos)) if global_pos.size else 1.0
            sample_medians = np.where(np.isfinite(sample_medians), sample_medians, global_med)

        grand_median = float(np.median(sample_medians[sample_medians > 0])) if np.any(sample_medians > 0) else 1.0
        norm_factors = sample_medians / grand_median
        norm_factors[norm_factors == 0] = 1.0
        X = X / norm_factors[:, None]
        A.obs['norm_factor'] = norm_factors
        A.uns['normalization_set'] = used_norm_names.tolist()

    X = np.log2(np.clip(X, eps, None))
    A.layers['pre_scale'] = X.copy()
    col_means = X.mean(axis=0, keepdims=True)
    X = X - col_means
    if scale:
        col_std = X.std(axis=0, ddof=1, keepdims=True)
        col_std[col_std == 0] = 1.0
        X = X / col_std
    A.X = X
    if n_pcs is not None and n_pcs > 0:
        sc.tl.pca(A, n_comps=int(min(n_pcs, min(A.n_obs, A.n_vars))), svd_solver='arpack', random_state=seed)
    A.uns['preferred_metric'] = metric
    return A


def preprocess_organelle(
    adata: AnnData,
    log1p: bool = False,
    n_pcs: Optional[int] = 30,
    scale: bool = True,
    metric: str = 'euclidean',
    seed: int = 0,
) -> AnnData:
    _require_scanpy_anndata()
    A = adata.copy()
    if log1p:
        A.X = np.log1p(np.maximum(A.X, 0.0))
    A.layers['pre_scale'] = A.X.copy()
    if scale:
        sc.pp.scale(A, max_value=None)
    if n_pcs is not None:
        sc.tl.pca(A, n_comps=n_pcs, svd_solver='arpack', random_state=seed)
    A.uns['preferred_metric'] = metric
    return A


# =============================================================================
# ---------------- Feature budget (variance on pre_scale) selection -----------
# =============================================================================

def select_top_var_features(X: np.ndarray, n: int) -> np.ndarray:
    X = np.asarray(X)
    if n >= X.shape[1]:
        return np.arange(X.shape[1])
    var = np.nanvar(X, axis=0)  # NaN-safe
    return np.argsort(var)[::-1][:n]

def downsample_features(adata: AnnData, n_features: int) -> AnnData:
    Xref = adata.layers.get('pre_scale', adata.X)
    Xref = np.asarray(Xref)
    if n_features >= Xref.shape[1]:
        return adata.copy()
    sel = select_top_var_features(Xref, n_features)
    return adata[:, sel].copy()


# =============================================================================
# ---------------- CSV <-> AnnData adapters  ----------
# =============================================================================

def _df_to_adata(df: pd.DataFrame) -> AnnData:
    """
    Expects a dataframe with at least:
      - 'cell_id' column (else row index is used)
      - 'position' column (used as obs['position'])
      - feature columns = everything else
    """
    _require_scanpy_anndata()
    if 'cell_id' in df.columns:
        obs_names = df['cell_id'].astype(str).to_numpy()
        df_work = df.drop(columns=['cell_id'])
    else:
        obs_names = df.index.astype(str).to_numpy()
        df_work = df.copy()

    pos_col = 'position' if 'position' in df_work.columns else df_work.columns[-1]
    features = [c for c in df_work.columns if c != pos_col]
    X = df_work[features].to_numpy(dtype=float)
    obs = pd.DataFrame(index=obs_names)
    obs['position'] = df_work[pos_col].to_numpy()
    var = pd.DataFrame(index=pd.Index(features, name="features"))
    return AnnData(X=X, obs=obs, var=var)


# =============================================================================
# -------- Preprocess-by-modality wrapper WITH feature budget (pre_scale) -----
# =============================================================================

def preprocess_by_modality(
    df: pd.DataFrame,
    modality: str,
    seed: int = 42,
    feature_budget: int = 98,
    **modality_kwargs
):
    """
    Wraps the AnnData preprocessors so the rest of the pipeline can keep using numpy arrays.
    Uses layers['pre_scale'] to select top-variance features (budget) BEFORE PCA (scaling-insensitive).

    Returns: X_pca, cell_ids, positions, preferred_metric, adata_preprocessed
    """
    adata = _df_to_adata(df)

    # We compute PCA AFTER feature selection, so skip PCA inside modality preprocessors.
    local_kwargs = dict(modality_kwargs)
    requested_n_pcs = local_kwargs.pop("n_pcs", None)

    mod = str(modality).lower()
    if mod in ("rna", "transcriptomics", "transcriptome"):
        A = preprocess_rna(adata, n_pcs=None, seed=seed, **{k: v for k, v in local_kwargs.items()
                                                            if k in {"n_top_genes", "scale_max", "metric"}})
        default_pcs = 50
    elif mod in ("protein", "proteomics", "cytof", "citel"):
        A = preprocess_protein(adata, n_pcs=None, seed=seed, **{k: v for k, v in local_kwargs.items()
                                                                if k in {"arcsinh_cofactor", "scale", "metric",
                                                                         "do_median_normalize", "norm_set", "norm_set_size",
                                                                         "coverage_steps", "eps"}})
        default_pcs = 30
    elif mod in ("organelle", "organellomics"):
        A = preprocess_organelle(adata, n_pcs=None, seed=seed, **{k: v for k, v in local_kwargs.items()
                                                                  if k in {"log1p", "scale", "metric"}})
        default_pcs = 30
    else:
        raise ValueError(f"Unknown modality '{modality}'. "
                         "Use one of: transcriptomics/rna, proteomics/protein, organellomics/organelle.")

    # --- Feature budget (pre_scale variance) ---
    n_vars_before = A.n_vars
    if feature_budget is not None and feature_budget > 0 and feature_budget < n_vars_before:
        A = downsample_features(A, int(feature_budget))
        print(f"[INFO] Feature selection: kept {A.n_vars} / {n_vars_before} features "
              f"(budget={feature_budget}) using variance on layers['pre_scale'].")

    # --- PCA after selection (once) ---
    n_pcs_final = requested_n_pcs if requested_n_pcs is not None else default_pcs
    n_pcs_final = int(min(n_pcs_final, A.n_vars, A.n_obs))
    sc.tl.pca(A, n_comps=n_pcs_final, svd_solver="arpack", random_state=seed)

    X_pca = A.obsm["X_pca"]
    cell_ids = A.obs_names.to_numpy()
    positions = A.obs["position"].to_numpy()
    preferred_metric = A.uns.get("preferred_metric", "euclidean")
    return X_pca, cell_ids, positions, preferred_metric, A


# =============================================================================
# -------------------------- Generic CSV-based preprocess ---------------------
# =============================================================================

def preprocess_dataset(df, modality, default_log=True, random_state=42):
    """
    Excludes 'cell_id' and 'position' from features.
    Auto-detects log1p for typical count-like modalities; standardizes features.
    Returns: X_scaled, cell_ids, positions
    """
    non_features = {"cell_id", "position"}
    feature_cols = [c for c in df.columns if c not in non_features]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding non-features (cell_id, position).")

    X = df[feature_cols].to_numpy(dtype=float)

    # Heuristic log-detect
    apply_log = False
    if default_log and modality in ['transcriptomics', 'proteomics']:
        if np.nanmin(X) >= 0 and (np.nanmax(X) > 20 or np.percentile(X, 95) > 50):
            apply_log = True
        else:
            print(f"[INFO] {modality}: values are small, skipping log-transform.")
    elif modality == 'organellomics':
        if np.nanmax(X) > 100 or (np.percentile(X, 75) / (np.percentile(X, 25) + 1e-9) > 10):
            print(f"[INFO] {modality}: high range/skewed values; consider log-transform.")

    if apply_log:
        print(f"[INFO] {modality}: applying log1p transform.")
        X = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pos_col = 'position' if 'position' in df.columns else df.columns[-1]
    cell_ids = df['cell_id'].to_numpy() if 'cell_id' in df.columns else np.arange(len(df))
    positions = df[pos_col].to_numpy()

    return X_scaled, cell_ids, positions


def load_precomputed_pca(input_csv):
    df = pd.read_csv(input_csv)
    if 'cell_id' not in df.columns:
        raise ValueError("Expected a 'cell_id' column in the precomputed PCA CSV.")
    pos_col = 'position' if 'position' in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c not in ['cell_id', pos_col]]
    if not feature_cols:
        raise ValueError("No PCA feature columns found after excluding 'cell_id' and position column.")
    X_pca = df[feature_cols].to_numpy(dtype=float)
    cell_ids = df['cell_id'].to_numpy()
    positions = df[pos_col].to_numpy()
    print(f"[INFO] Using '{pos_col}' as position column. Feature count={len(feature_cols)}; rows={len(df)}")
    return X_pca, cell_ids, positions


# =============================================================================
# ----------------------------- Core ML blocks --------------------------------
# =============================================================================

def apply_pca(X, n_components=20, random_state=42):
    max_valid = int(min(X.shape[0], X.shape[1]))
    n = int(min(max(3, n_components), max_valid))
    svd_solver = 'randomized' if n < min(X.shape) else 'auto'
    pca = PCA(n_components=n, svd_solver=svd_solver,
              random_state=(random_state if svd_solver == 'randomized' else None))
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def save_gmm_results(cell_ids, positions, X_pca, clusters, filename):
    """Save GMM cluster assignments with PCA coordinates only."""
    df_out = pd.DataFrame({
        'cell_id': cell_ids,
        'position': positions,
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'cluster': clusters,
    })
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(filename, index=False)
    print(f"[INFO] Saved: {filename}")


# =============================================================================
# --------------------- GMM model selection + quality -------------------------
# =============================================================================

def gmm_cluster_selection(
    X,
    max_clusters=10,
    random_state=42,
    cv_splits=5,
    n_jobs=-1,
    covariance_type: str = "full",
):
    """
    Picks k via a weighted combination of:
    - log-likelihood (↑), BIC (↓), AIC (↓), silhouette (↑), CV log-likelihood (↑).
    Uses train split + KFold CV for stability. Adds reg_covar for numerical robustness.
    Returns: labels, metrics_dict, final_gmm
    """
    weights = {"log_likelihood": 0.05, "bic": 0.25, "aic": 0.20, "silhouette": 0.15, "cv_score": 0.35}

    n_samples = X.shape[0]
    max_k = int(min(max_clusters, max(2, n_samples - 1)))
    if max_k < 2:
        raise ValueError("Not enough samples for GMM model selection.")

    X_train, _ = train_test_split(X, test_size=0.2, random_state=random_state)
    cv_splits_eff = int(max(2, min(cv_splits, len(X_train))))

    def calc_metrics(n):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type=covariance_type,
            random_state=random_state,
            reg_covar=1e-6,
        )
        gmm.fit(X_train)
        log_like = gmm.score(X_train) * len(X_train)
        bic = gmm.bic(X_train)
        aic = gmm.aic(X_train)
        if n > 1:
            try:
                sil = silhouette_score(X_train, gmm.predict(X_train))
            except Exception:
                sil = np.nan
        else:
            sil = np.nan

        kf = KFold(n_splits=cv_splits_eff, shuffle=True, random_state=random_state)
        fold_scores = []
        for tr, va in kf.split(X_train):
            g = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                random_state=random_state,
                reg_covar=1e-6,
            ).fit(X_train[tr])
            fold_scores.append(g.score(X_train[va]) * len(va))
        cv_score = float(np.mean(fold_scores))
        return log_like, bic, aic, sil, cv_score

    n_clusters_range = list(range(1, max_k + 1))
    results = Parallel(n_jobs=n_jobs)(delayed(calc_metrics)(n) for n in n_clusters_range)
    log_likelihoods, bics, aics, silhouettes, cv_scores = map(np.array, zip(*results))

    def normalize(arr):
        arr = np.array(arr, dtype=float)
        return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-10)

    combined_scores = (
        weights["log_likelihood"] * normalize(log_likelihoods) +
        weights["bic"] * (1 - normalize(bics)) +
        weights["aic"] * (1 - normalize(aics)) +
        weights["silhouette"] * normalize(silhouettes) +
        weights["cv_score"] * normalize(cv_scores)
    )

    optimal_k = n_clusters_range[int(np.nanargmax(combined_scores))]
    print(f"[GMM Cluster Selection - Multi-Metric] Optimal clusters: {optimal_k}")

    final_gmm = GaussianMixture(
        n_components=optimal_k,
        covariance_type=covariance_type,
        random_state=random_state,
        reg_covar=1e-6,
    ).fit(X)
    labels = final_gmm.predict(X)

    return labels, {
        'log_likelihood': log_likelihoods.tolist(),
        'bic': bics.tolist(),
        'aic': aics.tolist(),
        'silhouette': silhouettes.tolist(),
        'cv': cv_scores.tolist(),
        'combined_score': combined_scores.tolist(),
        'chosen_k': int(optimal_k),
        'n_clusters_range': n_clusters_range
    }, final_gmm


def _cov_matrix_from_gmm(gmm: GaussianMixture, k: int, i: int) -> np.ndarray:
    """Return covariance matrix for component i regardless of covariance_type."""
    cov_type = gmm.covariance_type
    if cov_type == 'full':
        return gmm.covariances_[i]
    elif cov_type == 'diag':
        return np.diag(gmm.covariances_[i])
    elif cov_type == 'spherical':
        return np.eye(k) * gmm.covariances_[i]
    elif cov_type == 'tied':
        return gmm.covariances_
    else:
        C = gmm.covariances_[i] if hasattr(gmm, "covariances_") else np.eye(k)
        return C


def gmm_quality_metrics(X: np.ndarray, gmm: GaussianMixture, labels: np.ndarray) -> Dict[str, Any]:
    """
    Compute:
      - silhouette ([-1,1]), db_index (>=0),
      - overlap_max from responsibilities (higher overlap = worse separation),
      - mahal_min (pairwise min symmetric Mahalanobis distance),
      - entropy_mean (mean normalized entropy; 0=confident,1=uncertain), entropy_score=1-mean.
    """
    # Silhouette & DBI (guard for single-cluster)
    unique = np.unique(labels)
    if len(unique) > 1:
        sil = float(silhouette_score(X, labels)) if len(unique) > 1 else float('nan')
        dbi = float(davies_bouldin_score(X, labels))
    else:
        sil, dbi = float('nan'), float('nan')

    # Responsibilities
    resp = gmm.predict_proba(X)  # (n_samples, k)
    k = resp.shape[1]

    # Pairwise overlap (mean min prob)
    overlap_vals = []
    for i in range(k):
        for j in range(i + 1, k):
            overlap_ij = float(np.mean(np.minimum(resp[:, i], resp[:, j])))
            overlap_vals.append(overlap_ij)
    overlap_max = float(np.max(overlap_vals)) if overlap_vals else float('nan')

    # Mahalanobis separations between component means
    means = gmm.means_
    mahal_d = []
    for i in range(k):
        for j in range(i + 1, k):
            mu = means[i] - means[j]
            Ci = _cov_matrix_from_gmm(gmm, X.shape[1], i)
            Cj = _cov_matrix_from_gmm(gmm, X.shape[1], j)
            Cavg = (Ci + Cj) / 2.0
            # regularize for stability
            Cavg = Cavg + np.eye(Cavg.shape[0]) * 1e-6
            try:
                d2 = float(mu.T @ la.solve(Cavg, mu))
            except Exception:
                d2 = float(mu.T @ np.linalg.pinv(Cavg) @ mu)
            d = float(np.sqrt(max(d2, 0.0)))
            mahal_d.append(d)
    mahal_min = float(np.min(mahal_d)) if mahal_d else float('nan')

    # Uncertainty via normalized entropy of responsibilities
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.sum(resp * np.log(resp + 1e-12), axis=1) / np.log(max(k, 2))
    ent_mean = float(np.mean(ent))
    ent_score = float(1.0 - ent_mean)  # higher is better

    return {
        "silhouette": sil,
        "db_index": dbi,
        "overlap_max": overlap_max,
        "mahal_min": mahal_min,
        "entropy_mean": ent_mean,
        "entropy_score": ent_score,
    }


# =============================================================================
# --------------------- Effective Resolution ----------------------------
# =============================================================================

def effective_resolution_metrics(
    X: np.ndarray,
    labels_k: np.ndarray,
    k: int,
    random_state: int = 42,
    covariance_type: str = "full",
    tau_small: float = 0.01,   # 1% tiny-cluster floor
) -> Dict[str, Any]:
    """
    Measures how 'usable' the chosen k is:
      - Granularity/balance (avoid many tiny clusters)
      - Stability to k±1 (prefers stable plateaus over brittle frontiers)
    Returns a dict including an overall 'effres_score' in [0,1] where higher is better.
    """
    n = len(labels_k)
    # Balance / granularity
    sizes = np.bincount(labels_k, minlength=k).astype(float)
    p = sizes / max(n, 1)
    s2 = float(np.sum(p ** 2))
    n_eff = (1.0 / s2) if s2 > 0 else np.nan
    S_balance = float(min(1.0, (n_eff / max(k, 1.0)))) if np.isfinite(n_eff) else np.nan
    S_minsize = float(min(1.0, (np.min(p) / tau_small))) if len(p) and tau_small > 0 else np.nan
    S_gran = float(np.nanmean([S_balance, S_minsize]))

    # Neighboring-k stability
    def _fit_labels(k_):
        if k_ < 2:
            return None
        g = GaussianMixture(
            n_components=k_,
            covariance_type=covariance_type,
            random_state=random_state,
            reg_covar=1e-6
        ).fit(X)
        return g.predict(X)

    labs_km1 = _fit_labels(k - 1)
    labs_kp1 = _fit_labels(k + 1)
    aris = []
    if labs_km1 is not None:
        aris.append(adjusted_rand_score(labels_k, labs_km1))
    if labs_kp1 is not None:
        aris.append(adjusted_rand_score(labels_k, labs_kp1))
    S_stab = float(np.mean(aris)) if len(aris) else np.nan

    S_effres = float(np.nanmean([S_gran, S_stab]))
    return {
        "effres_score": S_effres,
        "effres_balance": S_balance,
        "effres_minsize": S_minsize,
        "effres_stability": S_stab,
        "n_eff": float(n_eff) if np.isfinite(n_eff) else np.nan,
        "min_cluster_prop": float(np.min(p)) if len(p) else np.nan,
    }


# =============================================================================
# --------------------- Bootstrap robustness ----------------------------------
# =============================================================================

def bootstrap_cluster_robustness(
    X: np.ndarray,
    base_labels: np.ndarray,
    k: int,
    n_boot: int = 30,
    frac: float = 0.8,
    random_state: int = 42,
    covariance_type: str = "full",
) -> Dict[str, Any]:
    """
    Bootstrap robustness for GMM:
      - repeatedly subsample frac of data, fit GMM(k), predict labels for full X,
        compute ARI vs base_labels. Report mean & std.
    Ensures subsample size >= k to avoid GMM failures when k is large.
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    aris: List[float] = []

    for b in range(n_boot):
        size = int(max(k, 2, np.floor(frac * n)))
        idx = rng.choice(n, size=size, replace=False)
        gmm_b = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state + 1000 + b,
            reg_covar=1e-6
        ).fit(X[idx])
        pred_full = gmm_b.predict(X)
        ari_b = float(adjusted_rand_score(base_labels, pred_full))
        aris.append(ari_b)

    return {
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris, ddof=1) if len(aris) > 1 else 0.0),
        "n_boot": int(n_boot),
        "frac": float(frac),
        "ari_samples": aris,
    }


# =============================================================================
# -----------------------------Composite score --------------------------------
# =============================================================================

def compute_modality_composite(
    gmm_quality: Dict[str, Any],
    boot_stats: Dict[str, Any],
    er_metrics: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Combine metrics into a single [0,1] score.
    Components (all scaled to [0,1], higher=better):
      - Robustness: boot ARI mean
      - Separation: 1 - overlap_max, and 1 - exp(-mahal_min)
      - Resolution: (silhouette+1)/2, and 1/(1+DB)
      - Uncertainty: entropy_score (1 - mean normalized entropy)
      - Effective Resolution: combines cluster-size granularity and k±1 stability

    Default weights sum to 1.0:
      robustness: 0.15
      separation_overlap: 0.15
      separation_mahal: 0.15
      resolution_sil: 0.15
      resolution_db: 0.15
      uncertainty: 0.15
      effective_resolution: 0.10
    """
    if weights is None:
        weights = {
            "robustness": 0.15,
            "sep_overlap": 0.15,
            "sep_mahal": 0.15,
            "res_sil": 0.15,
            "res_db": 0.15,
            "uncertainty": 0.15,
            "effective_resolution": 0.1,
        }

    # Robustness
    robustness = float(boot_stats.get("ari_mean", np.nan))

    # Separation
    sep_overlap = 1.0 - float(gmm_quality.get("overlap_max", np.nan))  # higher better
    mahal = float(gmm_quality.get("mahal_min", np.nan))
    sep_mahal = float(1.0 - np.exp(-max(mahal, 0.0))) if np.isfinite(mahal) else np.nan

    # Resolution
    sil = float(gmm_quality.get("silhouette", np.nan))
    res_sil = float((sil + 1.0) / 2.0) if np.isfinite(sil) else np.nan
    db = float(gmm_quality.get("db_index", np.nan))
    res_db = float(1.0 / (1.0 + db)) if np.isfinite(db) else np.nan

    # Uncertainty
    uncertainty = float(gmm_quality.get("entropy_score", np.nan))

    # Effective Resolution (new)
    effres = float(er_metrics.get("effres_score")) if (er_metrics and np.isfinite(er_metrics.get("effres_score", np.nan))) else np.nan

    parts = {
        "robustness": robustness,
        "sep_overlap": sep_overlap,
        "sep_mahal": sep_mahal,
        "silhouette_scaled": res_sil,
        "db_scaled": res_db,
        "entropy_score": uncertainty,
        "effective_resolution": effres,
    }

    # Weighted sum, skipping NaNs
    score = 0.0
    wsum = 0.0
    for key, w in weights.items():
        if key == "robustness":
            val = parts["robustness"]
        elif key == "sep_overlap":
            val = parts["sep_overlap"]
        elif key == "sep_mahal":
            val = parts["sep_mahal"]
        elif key == "res_sil":
            val = parts["silhouette_scaled"]
        elif key == "res_db":
            val = parts["db_scaled"]
        elif key == "uncertainty":
            val = parts["entropy_score"]
        elif key == "effective_resolution":
            val = parts["effective_resolution"]
        else:
            continue

        if np.isfinite(val):
            score += w * val
            wsum += w

    score = float(score / wsum) if wsum > 0 else float('nan')
    return score, parts


# =============================================================================
# --------------- Bootstrap composite score (error / CI) ----------------------
# =============================================================================

def bootstrap_composite_score(
    X: np.ndarray,
    k: int,
    boot_stats_full: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    n_boot: int = 30,
    frac: float = 0.8,
    random_state: int = 42,
    covariance_type: str = "full",
) -> Dict[str, Any]:
    """
    Bootstrap the composite score by resampling the data, refitting GMM(k),
    recomputing quality + effective-resolution metrics, and then recomputing
    the composite while holding the full-dataset robustness stats fixed.

    Returns:
      {
        "n_boot": int,
        "frac": float,
        "mean": float,
        "std": float,
        "sem": float,
        "ci_low": float,
        "ci_high": float,
        "samples": List[float],
      }
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    samples: List[float] = []

    for b in range(n_boot):
        size = int(max(k, 2, np.floor(frac * n)))
        idx = rng.choice(n, size=size, replace=False)
        X_b = X[idx]

        gmm_b = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state + 2000 + b,
            reg_covar=1e-6,
        ).fit(X_b)
        labels_b = gmm_b.predict(X_b)

        gmm_quality_b = gmm_quality_metrics(X_b, gmm_b, labels_b)
        effres_b = effective_resolution_metrics(
            X_b, labels_b, k,
            random_state=random_state + 3000 + b,
            covariance_type=covariance_type,
        )

        comp_b, _ = compute_modality_composite(
            gmm_quality_b,
            boot_stats_full,
            er_metrics=effres_b,
            weights=weights,
        )
        samples.append(float(comp_b))

    samples_arr = np.array(samples, dtype=float)
    mean = float(np.mean(samples_arr)) if len(samples_arr) else float('nan')
    std = float(np.std(samples_arr, ddof=1)) if len(samples_arr) > 1 else 0.0
    sem = float(std / np.sqrt(len(samples_arr))) if len(samples_arr) > 0 else float('nan')
    if len(samples_arr):
        ci_low, ci_high = np.percentile(samples_arr, [2.5, 97.5])
    else:
        ci_low = ci_high = float('nan')

    return {
        "n_boot": int(n_boot),
        "frac": float(frac),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "samples": samples,
    }


def save_run_summary(
    summary_path: str,
    modality: str,
    n_cells: int,
    n_features_after: Optional[int],
    feature_budget: Optional[int],
    preferred_metric: str,
    gmm_metrics: Dict[str, Any],
    gmm_quality: Dict[str, Any],
    boot_stats: Dict[str, Any],
    composite_score: float,
    composite_parts: Dict[str, float],
    effres_metrics: Optional[Dict[str, Any]] = None,
    composite_bootstrap: Optional[Dict[str, Any]] = None,
):
    payload = {
        "modality": modality,
        "n_cells": int(n_cells),
        "n_features_after_selection": (int(n_features_after) if n_features_after is not None else None),
        "feature_budget": (int(feature_budget) if feature_budget is not None else None),
        "preferred_metric": preferred_metric,
        "gmm_model_selection": gmm_metrics,
        "gmm_quality": gmm_quality,
        "bootstrap_ari_stats": boot_stats,
        "composite": {"score": float(composite_score), "parts": composite_parts},
        "effective_resolution": effres_metrics or {},
        "composite_bootstrap": composite_bootstrap or {},
    }
    payload = json_safe(payload)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Summary saved: {summary_path}")


# =============================================================================
# ------------------------------ Plotting helpers -----------------------------
# =============================================================================

def plot_pca_clusters(X_pca, clusters, modality):
    """Plot GMM clusters in PCA space only; no UMAP or Leiden output."""
    clusters = clusters.astype(str)
    unique_clusters = np.unique(clusters)
    plt.figure(figsize=(6, 5))
    for c in unique_clusters:
        idx = clusters == c
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {c}", s=10)
    plt.title(f"{modality} - GMM clusters in PCA (PC1 vs PC2)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.show()


def plot_metric(ax, x, y, title, xlabel, ylabel, optimal_clusters):
    ax.plot(x, y, marker='o', color='black')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.axvline(x=optimal_clusters, color='red', linestyle='--', label='Optimal Clusters')
    ax.legend(); ax.grid(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")


def plot_gmm_metrics(n_clusters_range, metrics_dict):
    optimal_clusters = metrics_dict['chosen_k']
    log_likelihoods = metrics_dict['log_likelihood']
    bics = metrics_dict['bic']
    aics = metrics_dict['aic']
    silhouettes = metrics_dict['silhouette']
    cv_scores = metrics_dict['cv']
    combined_scores = metrics_dict['combined_score']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plot_metric(axes[0, 0], n_clusters_range, log_likelihoods, 'Log-Likelihood', 'Clusters', 'Log-Likelihood', optimal_clusters)
    plot_metric(axes[0, 1], n_clusters_range, bics, 'BIC', 'Clusters', 'BIC', optimal_clusters)
    plot_metric(axes[0, 2], n_clusters_range, aics, 'AIC', 'Clusters', 'AIC', optimal_clusters)
    plot_metric(axes[1, 0], n_clusters_range[1:], silhouettes[1:], 'Silhouette Score', 'Clusters', 'Score', optimal_clusters)
    plot_metric(axes[1, 1], n_clusters_range, cv_scores, 'CV Log-Likelihood', 'Clusters', 'Score', optimal_clusters)
    plot_metric(axes[1, 2], n_clusters_range, combined_scores, 'Combined Score', 'Score', 'Score', optimal_clusters)
    plt.tight_layout(pad=4); plt.show()


def plot_pca_by_position(X_pca, positions, modality, point_size=40):
    print("[INFO] Plotting PCA colored by position...")
    try:
        # Numeric (continuous)
        pos = positions.astype(float)
        vmin, vmax = float(np.min(pos)), float(np.max(pos))
        vcenter = (vmin + vmax) / 2.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('viridis')  # Matplotlib-native
        plt.figure(figsize=(7, 6), dpi=300)
        sca = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pos, cmap=cmap, norm=norm, s=point_size)
        plt.title(f"{modality} - PCA (PC1 vs PC2) colored by position")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        cbar = plt.colorbar(sca)
        cbar.set_ticks([vmin, vcenter, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vcenter:.2f}", f"{vmax:.2f}"])
        cbar.set_label("Position", rotation=270, labelpad=15)
    except ValueError:
        # Categorical (discrete)
        pos = positions.astype(str)
        unique_pos = np.unique(pos)
        idx_map = {p: i for i, p in enumerate(unique_pos)}
        color_idx = np.array([idx_map[p] for p in pos])
        cmap = cm.get_cmap('viridis', len(unique_pos))
        norm = mcolors.BoundaryNorm(np.arange(len(unique_pos) + 1) - 0.5, cmap.N)
        plt.figure(figsize=(7, 6))
        sca = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_idx, cmap=cmap, norm=norm, s=point_size)
        plt.title(f"{modality} - PCA (PC1 vs PC2) colored by position")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        cbar = plt.colorbar(sca, ticks=range(len(unique_pos)))
        cbar.ax.set_yticklabels(unique_pos)
        cbar.set_label("Position", rotation=270, labelpad=15)
    plt.tight_layout(); plt.show()


# =============================================================================
# ---------------------------- Main processing --------------------------------
# =============================================================================

def process_modality(
    input_csv,
    modality,
    output_csv=None,
    precomputed_pca=False,
    max_gmm_clusters=10,
    sample_size=None,
    random_state=42,
    use_scanpy_preproc=False,
    feature_budget: int = 98,
    boot_n: int = 30,
    boot_frac: float = 0.8,
    covariance_type: str = "full",
    gmm_n_jobs: int = -1,
    plot_pca: bool = True,
    plot_gmm_diagnostics: bool = True,
    **modality_kwargs
):
    """
    End-to-end GMM-only runner. Returns arrays/metrics for programmatic use.

    This version keeps PCA preprocessing, GMM model selection, GMM quality metrics,
    bootstrap robustness, effective resolution, composite scoring, summary JSONs,
    ranking tables, and the same composite-score bar plot.

    """
    print(f"\nProcessing {modality} from {input_csv}...")
    if output_csv is None or str(output_csv).strip() == "":
        output_csv = default_output_filename_for_modality(modality)
    output_csv = output_csv_prefix_in_results(output_csv)
    print(f"[INFO] Results folder: {DEFAULT_RESULTS_DIR}")

    df = pd.read_csv(input_csv)

    # Optional deterministic sampling BEFORE preprocessing
    if sample_size is not None and sample_size < len(df):
        print(f"[INFO] Randomly sampling {sample_size} cells from {len(df)} total...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    preferred_metric = "euclidean"  # kept in summary for preprocessing provenance
    n_features_after = None

    if use_scanpy_preproc:
        # >>> Modality-aware AnnData preprocessing + feature budget <<<
        X_pca, cell_ids, positions, preferred_metric, A_pre = preprocess_by_modality(
            df, modality=modality, seed=random_state,
            feature_budget=feature_budget,
            **modality_kwargs
        )
        n_features_after = int(A_pre.n_vars)

        # Save PCA CSV for parity
        df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        df_pca.insert(0, "cell_id", cell_ids)
        df_pca["position"] = positions
        pca_output_path = derived_output_path(output_csv, "_pca.csv")
        df_pca.to_csv(pca_output_path, index=False)
        print(f"[INFO] PCA components saved to {pca_output_path}")

    else:
        # >>> Original CSV-based path <<<
        if precomputed_pca:
            X_pca, cell_ids, positions = load_precomputed_pca(input_csv)
        else:
            X_scaled, cell_ids, positions = preprocess_dataset(df, modality, default_log=True, random_state=random_state)
            X_pca, pca_model = apply_pca(X_scaled, n_components=20, random_state=random_state)
            # Save PCA CSV
            df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
            df_pca.insert(0, "cell_id", cell_ids)
            df_pca["position"] = positions
            pca_output_path = derived_output_path(output_csv, "_pca.csv")
            df_pca.to_csv(pca_output_path, index=False)
            print(f"[INFO] PCA components saved to {pca_output_path}")

    if X_pca.shape[1] < 2:
        raise ValueError("At least two PCA components are required because GMM clustering uses PC1 and PC2.")

    # Optional PCA colored by position plot; 
    if plot_pca:
        plot_pca_by_position(X_pca, positions, modality)

    # GMM on first 2 PCs.
    X_for_gmm = X_pca[:, :2]
    clusters_gmm, gmm_metrics, gmm_model = gmm_cluster_selection(
        X_for_gmm,
        max_clusters=max_gmm_clusters,
        random_state=random_state,
        n_jobs=gmm_n_jobs,
        covariance_type=covariance_type,
    )

    if plot_gmm_diagnostics:
        plot_gmm_metrics(gmm_metrics['n_clusters_range'], gmm_metrics)
        plot_pca_clusters(X_pca, clusters_gmm, modality)

    # GMM quality metrics (separation/resolution/uncertainty)
    gmm_quality = gmm_quality_metrics(X_for_gmm, gmm_model, clusters_gmm)

    # Bootstrap robustness (ARI vs base GMM labels)
    boot_stats = bootstrap_cluster_robustness(
        X_for_gmm, base_labels=clusters_gmm, k=int(gmm_metrics["chosen_k"]),
        n_boot=boot_n, frac=boot_frac, random_state=random_state, covariance_type=covariance_type
    )

    # Effective Resolution (granularity + k±1 stability)
    effres = effective_resolution_metrics(
        X_for_gmm, clusters_gmm, int(gmm_metrics["chosen_k"]),
        random_state=random_state, covariance_type=covariance_type
    )

    # Composite score (includes effective resolution)
    composite, comp_parts = compute_modality_composite(gmm_quality, boot_stats, er_metrics=effres)

    # Bootstrap the composite score itself for error / CI
    composite_boot = bootstrap_composite_score(
        X_for_gmm,
        k=int(gmm_metrics["chosen_k"]),
        boot_stats_full=boot_stats,
        weights=None,
        n_boot=boot_n,
        frac=boot_frac,
        random_state=random_state,
        covariance_type=covariance_type,
    )

    # Save GMM-only output. 
    save_gmm_results(cell_ids, positions, X_pca, clusters_gmm, derived_output_path(output_csv, "_gmm.csv"))

    # JSON summary
    summary_path = derived_output_path(output_csv, "_summary.json")
    save_run_summary(
        summary_path, modality,
        n_cells=len(cell_ids),
        n_features_after=n_features_after,
        feature_budget=feature_budget if use_scanpy_preproc else None,
        preferred_metric=preferred_metric,
        gmm_metrics=gmm_metrics,
        gmm_quality=gmm_quality,
        boot_stats=boot_stats,
        composite_score=composite,
        composite_parts=comp_parts,
        effres_metrics=effres,
        composite_bootstrap=composite_boot,
    )

    print(f"[DONE] {modality} processed. Outputs saved as *_gmm.csv and summary JSON.")
    return {
        "cell_ids": cell_ids,
        "positions": positions,
        "X_pca": X_pca,
        "clusters_gmm": clusters_gmm,
        "gmm_metrics": gmm_metrics,
        "gmm_quality": gmm_quality,
        "bootstrap": boot_stats,
        "effective_resolution": effres,
        "preferred_metric": preferred_metric,
        "composite": {"score": composite, "parts": comp_parts},
        "composite_bootstrap": composite_boot,
    }


# =============================================================================
# ------------------------------ Ranking helper -------------------------------
# =============================================================================

def rank_modalities(summary_paths, output_csv="modality_ranking.csv", round_ndigits=3):
    """
    Build a ranking table from one or more *_summary.json files produced by process_modality().
    Sorts by CompositeScore (desc) and saves a CSV in the script-local Results folder.
    """
    output_csv = output_path_in_results(output_csv)
    rows = []
    for path in summary_paths:
        path = resolve_input_or_results(path)
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)

        comp      = s.get("composite", {}) or {}
        parts     = comp.get("parts", {}) or {}
        gmm_sel   = s.get("gmm_model_selection", {}) or {}
        boot      = s.get("bootstrap_ari_stats", {}) or {}
        er        = s.get("effective_resolution", {}) or {}
        comp_boot = s.get("composite_bootstrap", {}) or {}

        rows.append({
            "Modality":                    s.get("modality"),
            "CompositeScore":             float(comp.get("score", float("nan"))),
            "CompositeStd":               float(comp_boot.get("std", float("nan"))),
            "CompositeSEM":               float(comp_boot.get("sem", float("nan"))),
            "CompositeCI_low":            float(comp_boot.get("ci_low", float("nan"))),
            "CompositeCI_high":           float(comp_boot.get("ci_high", float("nan"))),
            "Robustness(ARI_mean)":       float(boot.get("ari_mean", float("nan"))),
            "Separation(overlap→1−max)":  float(parts.get("sep_overlap", float("nan"))),
            "Separation(mahal→1−e^-d)":   float(parts.get("sep_mahal", float("nan"))),
            "Resolution(sil_scaled)":     float(parts.get("silhouette_scaled", float("nan"))),
            "Resolution(DB_scaled)":      float(parts.get("db_scaled", float("nan"))),
            "UncertaintyScore":           float(parts.get("entropy_score", float("nan"))),
            "EffectiveResolution":        float(parts.get("effective_resolution", float("nan"))),
            "EffResBalance":              float(er.get("effres_balance", float("nan"))),
            "EffResStability":            float(er.get("effres_stability", float("nan"))),
            "N_eff":                      float(er.get("n_eff", float("nan"))),
            "MinClusterProp":             float(er.get("min_cluster_prop", float("nan"))),
            "k(chosen)":                  int(gmm_sel.get("chosen_k", 0)),
            "n_cells":                    int(s.get("n_cells", 0)),
            "n_features":                 (int(s.get("n_features_after_selection"))
                                           if s.get("n_features_after_selection") is not None else None),
            "PreferredMetric":            s.get("preferred_metric"),
            "SummaryPath":                path,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("CompositeScore", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    # Optional rounding for readability
    to_round = [
        "CompositeScore", "CompositeStd", "CompositeSEM",
        "CompositeCI_low", "CompositeCI_high",
        "Robustness(ARI_mean)",
        "Separation(overlap→1−max)", "Separation(mahal→1−e^-d)",
        "Resolution(sil_scaled)", "Resolution(DB_scaled)", "UncertaintyScore",
        "EffectiveResolution", "EffResBalance", "EffResStability", "N_eff", "MinClusterProp"
    ]
    for c in to_round:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float).round(round_ndigits)

    # Save & print
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("\n=== Modality Ranking (higher is better) ===")
    print(df.drop(columns=["SummaryPath"]).to_string(index=False))
    print(f"\n[INFO] Ranking table saved to: {output_csv}")

    return df


# =============================================================================
# --------- Bar plot of bootstrap composite score with STD (horizontal) -------
# =============================================================================

def modality_color(mod: str):
    """
    Return RGBA color with alpha=0.5 based on modality string:
      - Organellomics / Organelle -> magenta
      - Proteomics / Protein      -> grey
      - Transcriptomics / RNA     -> blue
    """
    m = str(mod).lower()
    if "organellomics" in m or "organelle" in m:
        return (1.0, 0.0, 1.0, 0.3)  # magenta, alpha=0.5
    elif "proteomics" in m or "protein" in m:
        return (0.5, 0.5, 0.5, 0.1)  # grey, alpha=0.5
    elif "transcriptomics" in m or "rna" in m:
        return (0.0, 0.0, 1.0, 0.1)  # blue, alpha=0.5
    else:
        return (0.0, 0.0, 0.0, 0.01)  # fallback dark grey  #(0.2, 0.2, 0.2, 0.5)


def plot_composite_bars_from_ranking(
    ranking_csv: str = "modality_ranking.csv",
    score_col: str = "CompositeScore",
    std_col: str = "CompositeStd",
    output_path: str = "composite_bootstrap_barplot.png",
    dpi: int = 300,
):
    """
    Plot a horizontal bar graph of composite scores with STD error bars and save it.

    - Y-axis: method names (modalities)
    - X-axis: 0–1 (composite score)
    - Bars extend to the right from the Y-axis
    - Bars sorted from highest to lowest score (top to bottom: high → low)
    - Colors:
        Organellomics -- magenta (alpha=0.5)
        Proteomics    -- grey    (alpha=0.5)
        Transcriptomics -- blue  (alpha=0.5)
    - Font size for both axes: 15
    - Gridlines: only very faint vertical ones; no horizontal gridlines.
    """
    ranking_csv = resolve_input_or_results(ranking_csv)
    output_path = output_path_in_results(output_path)
    df = pd.read_csv(ranking_csv)

    if score_col not in df.columns or std_col not in df.columns:
        raise ValueError(f"Required columns '{score_col}' and/or '{std_col}' not found in {ranking_csv}")

    # Keep only needed columns and drop rows with missing values
    df_plot = df[["Modality", score_col, std_col]].dropna()

    # Sort so that highest score appears at the top (top→bottom: high→low)
    df_plot = df_plot.sort_values(score_col, ascending=True)

    scores = df_plot[score_col].values
    errs   = df_plot[std_col].values
    labels = df_plot["Modality"].values
    colors = [modality_color(m) for m in labels]

    y_pos = np.arange(len(df_plot))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Horizontal bar plot with STD as error bars
    ax.barh(
        y_pos,
        scores,
        xerr=errs,
        color=colors,
        edgecolor="black",
        capsize=5,
        height=0.6,
    )

    # Y-axis: method names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=15)

    # X-axis: 0–1 range
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Composite score", fontsize=15)
    ax.tick_params(axis="x", labelsize=15)

    # Optional y-axis label
    ax.set_ylabel("Method", fontsize=15)

    # Very faint vertical gridlines, no horizontal ones
    ax.xaxis.grid(False, linestyle="--", alpha=0.5)
    ax.yaxis.grid(False)

    # Clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)

    plt.tight_layout()

    # Save the figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"[INFO] Composite bar plot saved to: {output_path}")

    plt.show()


# =============================================================================
# --------------------------------- CLI / Main --------------------------------
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print(f"[INFO] All generated files will be saved in: {DEFAULT_RESULTS_DIR}")
        # --- Organelle / Organellomics via AnnData preprocessor ---
        path = "...add path.../figure2_benchmark_Organellomics_input_matrix.csv"
        output_path = "organellomics_clusters.csv"
        process_modality(
            path, "organellomics", output_path,
            max_gmm_clusters=10,
            use_scanpy_preproc=True,
            sample_size=390, random_state=42,
            feature_budget=98,  # feature cap
            boot_n=30, boot_frac=0.5,
            # modality kwargs:
            log1p=False, n_pcs=40, scale=True, metric="euclidean"
        )

        # --- RNA / Transcriptomics via AnnData preprocessor ---
        process_modality(
            "...add path.../figure2_benchmark_Transcriptomics_input_matrix.csv",
            "transcriptomics", "rna_clusters.csv",
            max_gmm_clusters=10,
            use_scanpy_preproc=True,
            sample_size=390, random_state=42,
            feature_budget=98,
            boot_n=30, boot_frac=0.5,
            n_top_genes=2000, n_pcs=80, scale_max=10.0, metric="cosine"
        )

        # --- Protein / Proteomics via AnnData preprocessor ---
        process_modality(
            "...add path.../figure2_benchmark_Proteomics_input_matrix.csv",
            "proteomics", "protein_clusters.csv",
            max_gmm_clusters=10,
            use_scanpy_preproc=True,
            sample_size=390, random_state=42,
            feature_budget=98,
            boot_n=30, boot_frac=0.5,
            do_median_normalize=True, n_pcs=40, scale=False, metric="euclidean"
        )

        # Build a ranking table from the three summary JSONs
        rank_modalities(
            [
                "organellomics_clusters_summary.json",
                "rna_clusters_summary.json",
                "protein_clusters_summary.json",
            ],
            output_csv="modality_ranking.csv",
            round_ndigits=3
        )

        # Plot composite bar chart with STD error bars and save it
        plot_composite_bars_from_ranking(
            ranking_csv="modality_ranking.csv",
            output_path="composite_bootstrap_barplot.png",
            dpi=300,
        )

    else:
        # ---------------- CLI path ----------------
        import argparse
        p = argparse.ArgumentParser(description="Modality-aware GMM-only benchmarking with feature budget, GMM quality, effective resolution, and ranking")
        p.add_argument("--input", default=None, help="Input CSV path")
        p.add_argument("--modality", default=None,
                       choices=["transcriptomics","rna","proteomics","protein","organellomics","organelle"],
                       help="Modality to process")
        p.add_argument("--output", default=None, help="Output CSV filename/prefix. Generated files are saved automatically in Results/ beside this script.")
        p.add_argument("--use-scanpy-preproc", action="store_true", help="Use modality-specific AnnData preprocessing")
        p.add_argument("--precomputed-pca", action="store_true", help="Treat input as precomputed PCA CSV (fallback path)")
        p.add_argument("--sample-size", type=int, default=None, help="Number of cells to sample before preprocessing")
        p.add_argument("--max-gmm-clusters", type=int, default=10)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--feature-budget", type=int, default=98,
                       help="Max number of features to retain by pre_scale variance")
        p.add_argument("--boot-n", type=int, default=30, help="Bootstrap iterations for robustness & composite")
        p.add_argument("--boot-frac", type=float, default=0.8, help="Bootstrap subsample fraction")
        p.add_argument("--covariance-type", default="full", choices=["full","diag","spherical","tied"],
                       help="Covariance type for GMMs (selection, robustness, effective-resolution, composite bootstrap)")
        p.add_argument("--gmm-n-jobs", type=int, default=-1,
                       help="Parallel jobs for GMM model-selection sweep; use -1 for all cores")
        # modality knobs
        p.add_argument("--rna-n-top-genes", type=int, default=2000)
        p.add_argument("--rna-n-pcs", type=int, default=50)
        p.add_argument("--protein-n-pcs", type=int, default=30)
        p.add_argument("--organelle-n-pcs", type=int, default=30)
        p.add_argument("--protein-scale", action="store_true", help="Enable scaling for proteomics (default off)")
        # optional: ranking on a list of summaries
        p.add_argument("--rank", nargs="*", help="Paths to *_summary.json files to rank; outputs Results/modality_ranking.csv")
        p.add_argument("--plot-ranking", action="store_true",
                       help="If set, also plot composite bar graph with STD from modality_ranking.csv")
        p.add_argument("--plot-output", default="composite_bootstrap_barplot.png",
                       help="Output filename for composite bar plot; saved automatically in Results/ beside this script")

        args = p.parse_args()

        print(f"[INFO] All generated files will be saved in: {DEFAULT_RESULTS_DIR}")

        run_analysis = args.input is not None or args.modality is not None

        if run_analysis:
            missing = [name for name, value in {
                "--input": args.input,
                "--modality": args.modality,
            }.items() if value is None]
            if missing:
                p.error("To process a modality, provide: " + ", ".join(missing))

            if args.output is None:
                args.output = default_output_filename_for_modality(args.modality)
                print(f"[INFO] No --output supplied; using default output prefix: {args.output}")

            kwargs = {}
            if args.modality in ("transcriptomics","rna"):
                kwargs.update(dict(n_top_genes=args.rna_n_top_genes, n_pcs=args.rna_n_pcs, scale_max=10.0, metric="cosine"))
            elif args.modality in ("proteomics","protein"):
                kwargs.update(dict(do_median_normalize=True, n_pcs=args.protein_n_pcs,
                                   scale=bool(args.protein_scale), metric="euclidean"))
            else:
                kwargs.update(dict(log1p=False, n_pcs=args.organelle_n_pcs, scale=True, metric="euclidean"))

            result = process_modality(
                args.input, args.modality, args.output,
                precomputed_pca=args.precomputed_pca,
                max_gmm_clusters=args.max_gmm_clusters,
                sample_size=args.sample_size,
                random_state=args.seed,
                use_scanpy_preproc=args.use_scanpy_preproc,
                feature_budget=args.feature_budget,
                boot_n=args.boot_n,
                boot_frac=args.boot_frac,
                covariance_type=args.covariance_type,
                gmm_n_jobs=args.gmm_n_jobs,
                **kwargs
            )

        if args.rank is not None:
            summary_paths = args.rank if len(args.rank) > 0 else default_summary_files()
            rank_modalities(summary_paths, output_csv="modality_ranking.csv", round_ndigits=3)
            if args.plot_ranking:
                plot_composite_bars_from_ranking(
                    ranking_csv="modality_ranking.csv",
                    output_path=args.plot_output,
                    dpi=300,
                )
        elif not run_analysis:
            p.error("Provide --input and --modality to process data, or use --rank to rank existing summaries.")
 