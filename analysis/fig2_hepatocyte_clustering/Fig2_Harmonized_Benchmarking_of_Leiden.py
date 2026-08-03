"""
Single-cell Transcriptomics vs Proteomics vs Organellomics

This script determine:
  1) the best Leiden setting per modality,
  2) general_metrics_table.png, and
  3) composite_bootstrap_barplot.png.

Execution order:
1) Modality-specific preprocessing.
2) Feature-budget matching using variance in layers['pre_scale'].
3) Reproducible cell-budget matching.
4) PCA after matching (PCA is still required for clustering, but no PCA/UMAP
   embedding plots are generated).
5) kNN/Leiden sweep, internal metrics, cell-subsampling stability, composite
   scoring, the composite-score bootstrap, focused tables, and robustness plots.

Input convention:
  - first CSV column: cell ID
  - middle CSV columns: numeric features
  - last CSV column: non-feature metadata (historically called "position")

The final metadata column is excluded from the feature matrix but is no longer
stored in AnnData because it is not used by this benchmark.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Sequence, Iterable

import numpy as np
import pandas as pd

# --- Ensure PNG saving works headless (e.g., servers/SSH) ---
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Core single-cell & ML stack
try:
    import scanpy as sc
    from anndata import AnnData
except Exception as e:
    raise ImportError("This template requires scanpy/anndata. Install with `pip install scanpy anndata`. " + str(e))

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from scipy import sparse

# Optional: intrinsic dimensionality
try:
    from skdim.id import MLE
    HAS_SKDIM = True
except Exception:
    HAS_SKDIM = False

import json

# =============================================================================
# Helpers for picking best rows / scoring tables / decisions
# =============================================================================

def pick_best_row(df: pd.DataFrame, min_clusters: int = 2) -> pd.Series:
    sel = df[df['n_clusters'] >= min_clusters]
    if sel.empty:
        sel = df
    sel = sel.assign(_sil=sel['silhouette'].fillna(-1e9))
    best = sel.sort_values(['stability_ari_mean', '_sil'], ascending=[False, False]).iloc[0]
    return best

def _modality_core_metrics(df: pd.DataFrame, min_clusters: int) -> Dict[str, float]:
    if 'n_cells_used' in df.columns and df['n_cells_used'].notna().any():
        cells_used = int(df['n_cells_used'].iloc[0])
    else:
        cells_used = float('nan')
    if 'n_features_used' in df.columns and df['n_features_used'].notna().any():
        features_used = int(df['n_features_used'].iloc[0])
    else:
        features_used = float('nan')

    df2 = df[df['n_clusters'] >= min_clusters].copy()
    if df2.empty:
        return {
            'stab_median': 0.0,
            'stab_q90': 0.0,
            'sil_median': float('nan'),
            'sil_q90': float('nan'),
            'eff_clusters': 0.0,
            'best_k': 0,
            'best_resolution': float('nan'),
            'best_stability': 0.0,
            'best_n_clusters': 0,
            'best_silhouette': float('nan'),
            'cells_used': cells_used,
            'features_used': features_used,
        }

    stab_median = float(df2['stability_ari_mean'].median())
    stab_q90    = float(df2['stability_ari_mean'].quantile(0.90))
    sil_median  = float(df2['silhouette'].median())
    sil_q90     = float(df2['silhouette'].quantile(0.90))
    eff_clusters = float((df2['n_clusters'] * df2['stability_ari_mean']).mean())

    best = pick_best_row(df2, min_clusters=min_clusters)

    return {
        'stab_median': stab_median,
        'stab_q90': stab_q90,
        'sil_median': sil_median,
        'sil_q90': sil_q90,
        'eff_clusters': eff_clusters,
        'best_k': int(best['k']),
        'best_resolution': float(best['resolution']),
        'best_stability': float(best['stability_ari_mean']),
        'best_n_clusters': int(best['n_clusters']),
        'best_silhouette': float(best['silhouette']),
        'cells_used': cells_used,
        'features_used': features_used,
    }

def _minmax(col: pd.Series) -> pd.Series:
    lo, hi = float(col.min()), float(col.max())
    if hi - lo <= 1e-12:
        return pd.Series([0.5] * len(col), index=col.index)
    return (col - lo) / (hi - lo)

def build_scores_table(
    df_rna: Optional[pd.DataFrame]=None,
    df_protein: Optional[pd.DataFrame]=None,
    df_org: Optional[pd.DataFrame]=None,
    min_clusters: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = {}
    if df_rna is not None and not df_rna.empty:
        rows['RNA'] = _modality_core_metrics(df_rna, min_clusters=min_clusters)
    if df_protein is not None and not df_protein.empty:
        rows['Protein'] = _modality_core_metrics(df_protein, min_clusters=min_clusters)
    if df_org is not None and not df_org.empty:
        rows['Organelle'] = _modality_core_metrics(df_org, min_clusters=min_clusters)
    if not rows:
        raise ValueError("No modalities provided to build_scores_table")
    raw = pd.DataFrame(rows).T

    raw['robustness'] = 0.7 * raw['stab_q90'] + 0.3 * raw['stab_median']
    raw['separation'] = 0.7 * raw['sil_q90'] + 0.3 * raw['sil_median']
    raw['resolution_eff'] = raw['eff_clusters']

    norm = pd.DataFrame(index=raw.index)
    for axis in ['robustness', 'separation', 'resolution_eff']:
        norm[axis] = _minmax(raw[axis])

    return raw, norm

def decide_winner(
    raw: pd.DataFrame,
    norm: pd.DataFrame,
    w_robust: float = 0.5,
    w_sep: float = 0.3,
    w_res: float = 0.2
) -> Tuple[str, pd.DataFrame, str, np.ndarray]:
    w = np.array([w_robust, w_sep, w_res], dtype=float)
    if w.sum() <= 0:
        w = np.array([0.5, 0.3, 0.2])
    w = w / w.sum()

    axes = ['robustness', 'separation', 'resolution_eff']
    labels = {
        'robustness': 'bootstrap stability',
        'separation': 'silhouette (separation)',
        'resolution_eff': 'effective #clusters (clusters×stability)'
    }

    scores = norm[axes].dot(w)
    winner = scores.idxmax()

    others = [i for i in norm.index if i != winner]
    howto = (
        "\nHow to read these metrics:\n"
        "• bootstrap stability (ARI): agreement between the full-data clustering and bootstrap subsamples; higher is better (1.0 = perfect).\n"
        "• silhouette: cluster separation/compactness in the chosen representation; higher is better.\n"
        "• effective resolution: mean over all (k,res) of (#clusters × stability). Rewards many clusters only when stably recoverable.\n"
        "• best n_clusters: number of clusters at the parameter setting that achieved the highest stability.\n"
    )

    if not others:
        r = raw.loc[winner]
        narrative = (
            f"Winner: {winner}. Only one modality was evaluated.\n"
            f"Best setting: k={r['best_k']}, res={r['best_resolution']:.2f}.\n"
            f"Stability ARI median/q90/best: {r['stab_median']:.3f}/{r['stab_q90']:.3f}/{r['best_stability']:.3f}.\n"
            f"Silhouette median/q90/best: {r['sil_median']:.3f}/{r['sil_q90']:.3f}/{r['best_silhouette']:.3f}.\n"
            f"Effective resolution: {r['eff_clusters']:.2f}; best n_clusters = {r['best_n_clusters']}.\n"
        ) + howto
        out = pd.DataFrame({'composite_score': scores})
        return winner, out, narrative, w

    diffs_vs_mean = {axis: float(norm.loc[winner, axis] - norm.loc[others, axis].mean()) for axis in axes}
    top_axes = sorted(diffs_vs_mean.items(), key=lambda x: -abs(x[1]))[:2]
    r = raw.loc[winner]
    narrative = (
        f"Winner: {winner}. Highest composite of robustness, separation, resolution.\n"
        f"Why: strongest {labels[top_axes[0][0]]} and {labels[top_axes[1][0]]} relative to peers.\n"
        f"Best setting: k={r['best_k']}, res={r['best_resolution']:.2f}.\n"
        f"Stability ARI median/q90/best: {r['stab_median']:.3f}/{r['stab_q90']:.3f}/{r['best_stability']:.3f}.\n"
        f"Silhouette median/q90/best: {r['sil_median']:.3f}/{r['sil_q90']:.3f}/{r['best_silhouette']:.3f}.\n"
        f"Effective resolution: {r['eff_clusters']:.2f}; best n_clusters = {r['best_n_clusters']}.\n"
    )
    narrative += "Where others fall short:\n"
    for mod in others:
        gaps = {axis: float(norm.loc[mod, axis] - norm.loc[winner, axis]) for axis in axes}
        worst = [item for item in sorted(gaps.items(), key=lambda x: x[1])[:2] if item[1] < 0]
        rm = raw.loc[mod]
        pieces = []
        for ax, val in worst:
            pieces.append(f"lower {labels[ax]} (Δnorm={val:.2f})")
        if not pieces:
            pieces.append(f"lower composite score (Δ={scores.loc[mod]-scores.loc[winner]:.2f})")
        narrative += (
            f"- {mod}: " + ", ".join(pieces) + ". "
            f"Key numbers — stability median/q90/best: {rm['stab_median']:.3f}/{rm['stab_q90']:.3f}/{rm['best_stability']:.3f}; "
            f"silhouette median/q90/best: {rm['sil_median']:.3f}/{rm['sil_q90']:.3f}/{rm['best_silhouette']:.3f}; "
            f"effective clusters: {rm['eff_clusters']:.2f}; best n_clusters={rm['best_n_clusters']}."
        )
    out = pd.DataFrame({'composite_score': scores})
    narrative += howto
    return winner, out, narrative, w

def _first_or_nan(series: pd.Series) -> float:
    try:
        return float(series.iloc[0])
    except Exception:
        return float('nan')

def build_general_metrics_table_from_dfs(
    df_rna: Optional[pd.DataFrame],
    df_protein: Optional[pd.DataFrame],
    df_org: Optional[pd.DataFrame],
    comp_scores: Optional[pd.DataFrame] = None,
    min_clusters: int = 2,
) -> pd.DataFrame:
    rows = []
    for mod, df in [('RNA', df_rna), ('Protein', df_protein), ('Organelle', df_org)]:
        if df is None or df.empty:
            continue
        stab_med = float(df['stability_ari_mean'].median())
        stab_q90 = float(df['stability_ari_mean'].quantile(0.90))
        sil_med = float(df['silhouette'].median())
        sil_q90 = float(df['silhouette'].quantile(0.90))
        eff = float((df['n_clusters'] * df['stability_ari_mean']).mean())
        pcs80 = _first_or_nan(df['pcs80']) if 'pcs80' in df.columns else float('nan')
        pcs90 = _first_or_nan(df['pcs90']) if 'pcs90' in df.columns else float('nan')
        idmle = _first_or_nan(df['id_mle']) if 'id_mle' in df.columns else float('nan')
        n_cells = _first_or_nan(df['n_cells_used']) if 'n_cells_used' in df.columns else float('nan')
        n_features = _first_or_nan(df['n_features_used']) if 'n_features_used' in df.columns else float('nan')
        best = pick_best_row(df, min_clusters=min_clusters)
        row = {
            'modality': mod,
            'composite_score': float(comp_scores.loc[mod, 'composite_score']) if comp_scores is not None and mod in comp_scores.index else float('nan'),
            'stability_ari_median': stab_med,
            'stability_ari_q90': stab_q90,
            'silhouette_median': sil_med,
            'silhouette_q90': sil_q90,
            'effective_resolution_mean': eff,
            'Number of Cells': n_cells,
            'Number of Features': n_features,
            'best_n_clusters': int(best['n_clusters']),
            'best_stability_ari': float(best['stability_ari_mean']),
            'best_silhouette': float(best['silhouette']),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index('modality')
    if 'composite_score' in out.columns and out['composite_score'].notna().any():
        out = out.sort_values('composite_score', ascending=False)
    else:
        out = out.sort_values('stability_ari_median', ascending=False)
    return out

def save_table_as_png(
    df: pd.DataFrame,
    path: str,
    dpi: int = 200,
    *,
    target_aspect: float = 0.5,
    base_cell_w: float = 0.9,
    base_cell_h: float = 0.5,
    min_fig_w: float = 6.0,
    min_fig_h: float = 4.5,
    max_fig_w: float = 14.0,
    max_fig_h: float = 14.0,
    wrap_chars: int = 12,
    header_fontsize: int = 9,
    body_fontsize: int = 8,
    scale_x: float = 1.0,
    scale_y: float = 1.2,
):
    if df is None or df.empty:
        return
    import textwrap
    col_labels = [textwrap.fill(str(c), width=wrap_chars) for c in df.columns]
    row_labels = [str(i) for i in df.index]
    n_rows = len(df.index)
    n_cols = len(df.columns)
    fig_w = float(np.clip(base_cell_w * (n_cols + 1), min_fig_w, max_fig_w))
    fig_h = float(np.clip(base_cell_h * (n_rows + 1.2), min_fig_h, max_fig_h))
    current_aspect = fig_w / fig_h
    if current_aspect > target_aspect * 1.05:
        fig_h = min(max_fig_h, max(fig_h, fig_w / max(target_aspect, 1e-6)))
    elif current_aspect < target_aspect / 1.05:
        fig_w = min(max_fig_w, max(fig_w, fig_h * target_aspect))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.axis('off')
    body = df.round(3).values
    table = ax.table(cellText=body, colLabels=col_labels, rowLabels=row_labels, loc='center', cellLoc='center', rowLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(body_fontsize)
    for (r, c), cell in table.get_celld().items():
        if r == -1:
            cell.set_text_props(fontsize=header_fontsize, fontweight='bold')
            cell.set_height(cell.get_height() * 1.15)
    if hasattr(table, "auto_set_column_width"):
        try:
            table.auto_set_column_width(col=list(range(n_cols)))
        except Exception:
            pass
    table.scale(scale_x, scale_y)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# NEW: Composite bootstrap + bar plot helpers
# =============================================================================

def modality_color(mod: str):
    """
    Return the original RGBA colors used by the published-style bar plot.

    The alpha values are intentionally preserved exactly from v8:
      - Organelle / Organellomics -> magenta, alpha 0.3
      - Protein / Proteomics      -> grey, alpha 0.1
      - RNA / Transcriptomics     -> blue, alpha 0.1
    """
    m = str(mod).lower()
    if "organelle" in m or "organellomics" in m:
        return (1.0, 0.0, 1.0, 0.3)  # preserved v8 style
    elif "protein" in m or "proteomics" in m:
        return (0.5, 0.5, 0.5, 0.1)  # preserved v8 style
    elif "rna" in m or "transcript" in m:
        return (0.0, 0.0, 1.0, 0.1)  # preserved v8 style
    else:
        return (0.0, 0.0, 0.0, 0.1)  # fallback dark grey

def bootstrap_composite_scores(
    df_rna: Optional[pd.DataFrame],
    df_protein: Optional[pd.DataFrame],
    df_org: Optional[pd.DataFrame],
    *,
    min_clusters: int = 2,
    w_robust: float = 0.33,
    w_sep: float = 0.33,
    w_res: float = 0.33,
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Bootstrap the *composite score* by resampling the (k, res) rows per modality.

    For each bootstrap:
      - Sample each modality's metrics table with replacement.
      - Rebuild raw/norm via build_scores_table().
      - Recompute composite scores with the same weights.
    Returns a DataFrame indexed by modality with:
      - composite_bootstrap_mean
      - composite_bootstrap_std
    """
    rng = np.random.RandomState(seed)

    # Nothing to do if we have no metrics
    if (df_rna is None or df_rna.empty) and \
       (df_protein is None or df_protein.empty) and \
       (df_org is None or df_org.empty):
        return pd.DataFrame()

    # Collect bootstrap samples per modality
    samples: Dict[str, List[float]] = {"RNA": [], "Protein": [], "Organelle": []}

    for b in range(n_boot):
        def _resample(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            # sample same number of rows with replacement
            return df.sample(
                n=len(df),
                replace=True,
                random_state=int(rng.randint(0, 2**31 - 1))
            )

        brna = _resample(df_rna)
        bprot = _resample(df_protein)
        borg = _resample(df_org)

        try:
            raw_b, norm_b = build_scores_table(brna, bprot, borg, min_clusters=min_clusters)
        except ValueError:
            # No modalities? Skip
            continue

        w_vec = np.array([w_robust, w_sep, w_res], dtype=float)
        if w_vec.sum() <= 0:
            w_vec = np.array([0.33, 0.33, 0.34])
        w_vec = w_vec / w_vec.sum()
        axes = ['robustness', 'separation', 'resolution_eff']
        scores_b = norm_b[axes].dot(w_vec)

        for mod in scores_b.index:
            if mod in samples:
                samples[mod].append(float(scores_b.loc[mod]))

    rows = []
    for mod in ["RNA", "Protein", "Organelle"]:
        vals = np.asarray(samples[mod], dtype=float)
        if vals.size == 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1) if vals.size > 1 else 0.0)
        rows.append({
            "modality": mod,
            "composite_bootstrap_mean": mean,
            "composite_bootstrap_std": std,
        })

    out = pd.DataFrame(rows).set_index("modality")
    return out

def plot_composite_bars(gm: pd.DataFrame, output_path: str, dpi: int = 300):
    """
    Horizontal bar plot of composite score (0–1) with bootstrap STD as x-error.

    - Y axis: modalities (RNA, Protein, Organelle)
    - X axis: composite score [0,1]
    - Sorted from highest to lowest score (top = best)
    - Colors and transparency are preserved exactly from v8.
    """
    if gm is None or gm.empty:
        return

    df = gm.copy()

    # Choose which columns to use: prefer bootstrap mean/std if present
    mean_col = "composite_bootstrap_mean" if "composite_bootstrap_mean" in df.columns else "composite_score"
    if mean_col not in df.columns:
        # Nothing to plot
        return

    err_col = "composite_bootstrap_std" if "composite_bootstrap_std" in df.columns else None

    # Sort from highest to lowest
    df = df.sort_values(mean_col, ascending=False)

    labels = df.index.to_numpy()
    scores = df[mean_col].to_numpy()
    errs = df[err_col].to_numpy() if err_col is not None and err_col in df.columns else None

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.5, 4 ))

    colors = [modality_color(m) for m in labels]

    ax.barh(
        y_pos,
        scores,
        xerr=errs,
        color=colors,
        edgecolor="black",
        capsize=5,
        height=0.6,
       
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=15)

    ax.set_xlabel("Composite score", fontsize=15)
    ax.set_xlim(0.0, 1.0)

    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    # Optional y-axis label
    ax.set_ylabel("Method", fontsize=15)
  
    # Highest score at top
    ax.invert_yaxis()

    # Very faint vertical gridlines
    ax.xaxis.grid(False, linestyle="--", alpha=0.5)
    ax.yaxis.grid(False)

    # Ensure all spines visible
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception:
        pass
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] composite_bootstrap_barplot.png: {os.path.abspath(output_path)}")

# =============================================================================
# IO / budgets / preprocessing
# =============================================================================

def set_seed(seed: int = 0):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

# --- robust, noisy saving helpers ---
def _abs(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p

def _write_csv(df: Optional[pd.DataFrame], path: str, label: str):
    try:
        ensure_outdir(os.path.dirname(path))
        (df if df is not None else pd.DataFrame()).to_csv(path, index=False)
        print(f"[write] {label}: {_abs(path)}  rows={0 if df is None else len(df)}")
    except Exception as e:
        print(f"[error] failed to write {label} -> {path}: {e}")

def _savefig(fig: Figure, path: str, label: str, dpi: int = 200):
    try:
        ensure_outdir(os.path.dirname(path))
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"[write] {label}: {_abs(path)}")
    except Exception as e:
        print(f"[error] failed to write {label} -> {path}: {e}")
    finally:
        plt.close(fig)

def load_csv_to_adata(path: str, position_col_is_last: bool = True) -> AnnData:
    """Load a modality matrix without retaining unused row metadata.

    For backward compatibility, ``position_col_is_last=True`` still means that
    the last CSV column is excluded from the feature matrix. Unlike v8, that
    column is not copied into ``adata.obs`` because no downstream calculation
    reads it. Excluding it in the same way preserves the clustering inputs.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 3:
        raise ValueError(
            f"CSV at {path} must have >=3 columns: cell_id, >=1 feature, metadata"
        )

    cell_id_col = df.columns[0]
    metadata_col = df.columns[-1] if position_col_is_last else None
    features_df = (
        df.iloc[:, 1:-1].copy()
        if position_col_is_last
        else df.iloc[:, 1:].copy()
    )

    for column in features_df.columns:
        features_df[column] = pd.to_numeric(features_df[column], errors='coerce')
    features_df = features_df.fillna(0.0)

    adata = AnnData(features_df.to_numpy())
    adata.obs_names = df[cell_id_col].astype(str).to_numpy()
    adata.var_names = features_df.columns.astype(str)

    if metadata_col is not None:
        print(
            f"[info] excluded non-feature metadata column '{metadata_col}' "
            f"from {os.path.abspath(path)}"
        )
    return adata

def downsample_cells(adata: AnnData, n_cells: int, seed: int = 0) -> AnnData:
    if n_cells >= adata.n_obs:
        return adata.copy()
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(adata.n_obs, n_cells, replace=False))
    return adata[idx].copy()

def select_top_var_features(X: np.ndarray, n: int) -> np.ndarray:
    X = np.asarray(X)
    if n >= X.shape[1]:
        return np.arange(X.shape[1])
    var = np.var(X, axis=0)
    return np.argsort(var)[::-1][:n]

def downsample_features(adata: AnnData, n_features: int) -> AnnData:
    Xref = adata.layers.get('pre_scale', adata.X)
    Xref = np.asarray(Xref)
    if n_features >= Xref.shape[1]:
        return adata.copy()
    sel = select_top_var_features(Xref, n_features)
    return adata[:, sel].copy()

# =============================================================================
# Preprocessing per modality
# =============================================================================

def preprocess_rna(
    adata: AnnData,
    n_top_genes: Optional[int] = 2000,
    n_pcs: Optional[int] = 50,
    scale_max: float = 10.0,
    metric: str = "cosine",
    seed: int = 0,
) -> AnnData:
    adata = adata.copy()
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    norm = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)  # TP10K
    adata.layers["tp10k"] = norm["X"]
    adata.layers["log1p_tp10k"] = adata.layers["tp10k"].copy()
    sc.pp.log1p(adata, layer="log1p_tp10k", base=None)
    adata.X = adata.layers["log1p_tp10k"]
    if n_top_genes is not None and adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes, inplace=True)
        adata = adata[:, adata.var["highly_variable"]].copy()
    adata.layers["pre_scale"] = adata.X.copy()
    sc.pp.scale(adata, max_value=scale_max)
    if n_pcs is not None:
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    adata.uns["preferred_metric"] = metric
    return adata

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
    n_cells, n_prot = X.shape
    var_names = A.var_names.to_numpy()

    norm_factors = np.ones(n_cells, dtype=float)
    used_norm_names = None

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
    adata = adata.copy()
    if log1p:
        adata.X = np.log1p(np.maximum(adata.X, 0.0))
    adata.layers['pre_scale'] = adata.X.copy()
    if scale:
        sc.pp.scale(adata, max_value=None)
    if n_pcs is not None:
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack', random_state=seed)
    adata.uns['preferred_metric'] = metric
    return adata

# =============================================================================
# Graphs / clustering / metrics
# =============================================================================

def build_graph_and_cluster(
    adata: AnnData,
    n_neighbors: int,
    resolution: float,
    use_rep: str = 'X_pca',
    metric: Optional[str] = None,
    random_state: int = 0,
    key_added: str = 'leiden',
) -> None:
    metric = metric or adata.uns.get('preferred_metric', 'euclidean')
    
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        metric=metric,
        random_state=random_state,
        method="umap",
    )
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, random_state=random_state)

def internal_scores(adata: AnnData, labels_key: str = 'leiden', rep: str = 'X_pca', _max_cells_eval: int = 5000) -> Dict[str, float]:
    X = adata.obsm.get(rep, None)
    if X is None:
        raise ValueError(f"Representation {rep} not found in adata.obsm")
    labels = adata.obs[labels_key].astype(str).values
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    if X.shape[0] > _max_cells_eval:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(X.shape[0], _max_cells_eval, replace=False))
        X_eval = X[idx]; labels_eval = labels[idx]
    else:
        X_eval = X; labels_eval = labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sil = silhouette_score(X_eval, labels_eval)
        ch = calinski_harabasz_score(X_eval, labels_eval)
        db = davies_bouldin_score(X_eval, labels_eval)
    return {"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db)}

def variance_explained(adata: AnnData) -> np.ndarray:
    vr = adata.uns.get('pca', {}).get('variance_ratio')
    if vr is None:
        nmax = min(100, adata.X.shape[1])
        pca = PCA(n_components=nmax)
        pca.fit(adata.X)
        vr = pca.explained_variance_ratio_
    return np.asarray(vr)

def pcs_to_threshold(vr: np.ndarray, threshold: float = 0.9) -> int:
    csum = np.cumsum(vr)
    k = int(np.searchsorted(csum, threshold) + 1)
    return min(k, len(vr))

def choose_pcs_from_variance(
    adata: AnnData,
    threshold: float = 0.90,
    lo: int = 15,
    hi: int = 60,
) -> int:
    vr = variance_explained(adata)
    k = pcs_to_threshold(vr, threshold=threshold)
    return int(np.clip(k, lo, min(hi, adata.n_vars)))

def intrinsic_dimensionality(adata: AnnData, rep: str = 'X_pca') -> Optional[float]:
    if not HAS_SKDIM:
        return None
    X = adata.obsm[rep]
    try:
        est = MLE().fit(X).dimension_
        return float(est)
    except Exception:
        return None

def bootstrap_stability(
    adata: AnnData,
    n_neighbors: int,
    resolution: float,
    rep: str = 'X_pca',
    metric: Optional[str] = None,
    repeats: int = 100,
    frac: float = 0.8,
    seed: int = 0,
    labels_key: str = 'leiden',
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    tmp = adata.copy()
    build_graph_and_cluster(tmp, n_neighbors=n_neighbors, resolution=resolution, use_rep=rep, metric=metric, random_state=seed, key_added=labels_key)
    base_labels = tmp.obs[labels_key].astype(str).copy()
    aris = []
    n = tmp.n_obs
    m = max(2, int(frac * n))
    for i in range(repeats):
        idx = np.sort(rng.choice(n, m, replace=False))
        sub = tmp[idx].copy()
        build_graph_and_cluster(sub, n_neighbors=n_neighbors, resolution=resolution, use_rep=rep, metric=metric, random_state=seed + i + 1, key_added=labels_key)
        ari = adjusted_rand_score(base_labels.iloc[idx].values, sub.obs[labels_key].astype(str).values)
        aris.append(ari)
    aris = np.asarray(aris)
    return {
        "stability_ari_mean": float(np.nanmean(aris)),
        "stability_ari_std": float(np.nanstd(aris)),
        "stability_ari_q10": float(np.nanpercentile(aris, 10)),
        "stability_ari_q90": float(np.nanpercentile(aris, 90)),
    }

# =============================================================================
# Sweeps / summaries
# =============================================================================

@dataclass
class SweepConfig:
    k_list: List[int]
    res_list: List[float]
    metric: str
    repeats: int = 100
    frac: float = 0.8
    seed: int = 0

def run_sweep(adata: AnnData, modality: str, sweep: SweepConfig, rep: str = 'X_pca', labels_key: str = 'leiden', min_clusters: int = 2) -> pd.DataFrame:
    rows = []
    vr = variance_explained(adata)
    pcs80 = pcs_to_threshold(vr, 0.80)
    pcs90 = pcs_to_threshold(vr, 0.90)
    id_est = intrinsic_dimensionality(adata, rep=rep)

    for k in sweep.k_list:
        for res in sweep.res_list:
            tmp = adata.copy()
            build_graph_and_cluster(tmp, n_neighbors=k, resolution=res, use_rep=rep, metric=sweep.metric, random_state=sweep.seed, key_added=labels_key)
            n_clusters = int(tmp.obs[labels_key].nunique())
            if n_clusters < min_clusters:
                continue
            metrics = internal_scores(tmp, labels_key=labels_key, rep=rep)
            stab = bootstrap_stability(adata, n_neighbors=k, resolution=res, rep=rep, metric=sweep.metric,
                                       repeats=sweep.repeats, frac=sweep.frac, seed=sweep.seed, labels_key=labels_key)

            rows.append({
                "modality": modality,
                "k": k,
                "resolution": res,
                "n_clusters": n_clusters,
                **metrics,
                **stab,
                "pcs80": pcs80,
                "pcs90": pcs90,
                "id_mle": id_est if id_est is not None else np.nan,
            })
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame, min_clusters: int = 2) -> pd.DataFrame:
    """Return the same top-five selections as v8 with CSV-safe labels."""
    df2 = df[df['n_clusters'] >= min_clusters]
    if df2.empty:
        df2 = df

    best_stab = df2.sort_values(
        ['stability_ari_mean', 'silhouette'], ascending=[False, False]
    ).head(5).copy()
    best_stab.insert(0, 'ranking_group', 'top_by_stability')

    best_sil = df2.sort_values(
        ['silhouette', 'stability_ari_mean'], ascending=[False, False]
    ).head(5).copy()
    best_sil.insert(0, 'ranking_group', 'top_by_silhouette')

    return pd.concat([best_stab, best_sil], ignore_index=True)

# =============================================================================
# ORDER:(preprocess → feature budget → cell budget → final PCA → sweeps)
# =============================================================================

INCLUDE_MODALITIES = {
    'RNA': True,
    'Protein': True,
    'Organelle': True,
}

def comparative_power_three(
    rna_csv: str,
    protein_csv: str,
    organelle_csv: str,
    outdir: str,
    # Preprocess params
    rna_n_top_genes: Optional[int] = 2000,
    protein_arcsinh: float = 5.0,
    organelle_log1p: bool = False,
    # Metrics
    rna_metric: str = 'cosine',
    protein_metric: str = 'euclidean',
    organelle_metric: str = 'euclidean',
    # PCA (allow "auto" or int)
    n_pcs_rna: Union[int, str] = "auto",
    n_pcs_protein: Union[int, str] = "auto",
    n_pcs_organelle: Union[int, str] = "auto",
    # Sweeps
    k_list: List[int] = [5, 8, 10, 15, 30],
    res_list: List[float] = [0.2, 0.4, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0],
    # Stability
    bootstrap_repeats: int = 50,
    bootstrap_frac: float = 0.8,
    seed: int = 0,
    # Fairness controls
    feature_budget: Optional[Union[str,int]] = 'match',
    cell_budget: Optional[Union[str,int]] = 'match',
    # Inclusion toggles (None -> use INCLUDE_MODALITIES)
    include_rna: Optional[bool] = None,
    include_protein: Optional[bool] = None,
    include_organelle: Optional[bool] = None,
    # Selection threshold
    min_clusters: int = 2,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """
    ORDER:
      1) Load CSVs
      2) Preprocess per modality (creates layers['pre_scale']); early PCA DISABLED
      3) Feature selection (feature budget) using pre_scale variance
      4) Downsample cells (cell budget, uniform random)
      5) Final PCA (after budgets)
      6) kNN/Leiden, metrics, stability, focused outputs

   
    """
    ensure_outdir(outdir)
    print(f"[info] writing outputs under: {os.path.abspath(outdir)}")
    set_seed(seed)

    # Resolve includes
    if include_rna is None:
        include_rna = bool(INCLUDE_MODALITIES.get('RNA', True))
    if include_protein is None:
        include_protein = bool(INCLUDE_MODALITIES.get('Protein', True))
    if include_organelle is None:
        include_organelle = bool(INCLUDE_MODALITIES.get('Organelle', True))

    include_map = {'RNA': include_rna, 'Protein': include_protein, 'Organelle': include_organelle}
    if not any(include_map.values()):
        raise ValueError("All modalities disabled; enable at least one in INCLUDE_MODALITIES or via include_* params.")

    # 1) Load included CSVs
    adatas: Dict[str, AnnData] = {}
    paths = {'RNA': rna_csv, 'Protein': protein_csv, 'Organelle': organelle_csv}
    for mod, use in include_map.items():
        if use:
            adatas[mod] = load_csv_to_adata(paths[mod])

    # 2) Preprocess per modality — disable early PCA (n_pcs=None)
    if include_rna and 'RNA' in adatas:
        adatas['RNA'] = preprocess_rna(adatas['RNA'],
                                       n_top_genes=rna_n_top_genes,
                                       n_pcs=None,                      # disable early PCA
                                       metric=rna_metric, seed=seed)
    if include_protein and 'Protein' in adatas:
        adatas['Protein'] = preprocess_protein(adatas['Protein'],
                                               arcsinh_cofactor=protein_arcsinh,
                                               n_pcs=None,                 # disable early PCA
                                               metric=protein_metric, seed=seed)
    if include_organelle and 'Organelle' in adatas:
        adatas['Organelle'] = preprocess_organelle(adatas['Organelle'],
                                                   log1p=organelle_log1p,
                                                   n_pcs=None,              # disable early PCA
                                                   metric=organelle_metric, seed=seed)

    # 3) Feature selection (feature budget) using *pre-scale* variance
    if isinstance(feature_budget, int):
        nfeat = feature_budget
    elif feature_budget == 'match':
        nfeat = min(ad.n_vars for ad in adatas.values())
    else:
        nfeat = None

    if nfeat is not None:
        for mod in list(adatas.keys()):
            adatas[mod] = downsample_features(adatas[mod], nfeat)

    # 4) Downsample cells (cell budget, uniform random)
    if isinstance(cell_budget, int):
        target_cells = cell_budget
    elif cell_budget == 'match':
        target_cells = min(ad.n_obs for ad in adatas.values())
    else:
        target_cells = None

    if target_cells is not None:
        for mod in list(adatas.keys()):
            adatas[mod] = downsample_cells(adatas[mod], target_cells, seed=seed)

    # 5) Final PCA (after budgets). Respect "auto" vs explicit ints.
    bounds = {'RNA': (30, 80), 'Protein': (10, 40), 'Organelle': (15, 50)}
    user_targets = {'RNA': n_pcs_rna, 'Protein': n_pcs_protein, 'Organelle': n_pcs_organelle}
    for mod, ad in adatas.items():
        ut = user_targets[mod]
        if isinstance(ut, str) and ut.lower() == "auto":
            lo, hi = bounds[mod]
            k = choose_pcs_from_variance(ad, threshold=0.90, lo=lo, hi=hi)
        else:
            k = int(ut)
        k = int(min(k, ad.n_vars))
        svd_solver = 'randomized' if max(ad.shape) > 2000 else 'arpack'
        sc.tl.pca(ad, n_comps=k, svd_solver=svd_solver, random_state=seed)

    # Configure sweeps per modality
    sweeps: Dict[str, SweepConfig] = {
        'RNA': SweepConfig(k_list=k_list, res_list=res_list, metric=rna_metric, repeats=bootstrap_repeats, frac=bootstrap_frac, seed=seed),
        'Protein': SweepConfig(k_list=k_list, res_list=res_list, metric=protein_metric, repeats=bootstrap_repeats, frac=bootstrap_frac, seed=seed),
        'Organelle': SweepConfig(k_list=k_list, res_list=res_list, metric=organelle_metric, repeats=bootstrap_repeats, frac=bootstrap_frac, seed=seed),
    }

    # Capture counts per modality after all matching/preprocessing
    counts = {mod: {'n_cells': adatas[mod].n_obs, 'n_features': adatas[mod].n_vars} for mod in adatas.keys()}

    # 6) Run sweeps
    results: Dict[str, pd.DataFrame] = {}
    for mod, ad in adatas.items():
        results[mod] = run_sweep(
            ad,
            modality=mod,
            sweep=sweeps[mod],
            min_clusters=min_clusters,
        )
        # Keep the modality label as a real CSV column even if a sweep happens
        # to return zero valid parameter settings.
        results[mod]['modality'] = mod
        results[mod]['n_cells_used'] = counts[mod]['n_cells']
        results[mod]['n_features_used'] = counts[mod]['n_features']

    # Save setting-level metrics. Each CSV has an explicit modality column.
    _write_csv(
        results.get('RNA'),
        os.path.join(outdir, 'metrics_rna.csv'),
        'metrics_rna.csv',
    )
    _write_csv(
        results.get('Protein'),
        os.path.join(outdir, 'metrics_protein.csv'),
        'metrics_protein.csv',
    )
    _write_csv(
        results.get('Organelle'),
        os.path.join(outdir, 'metrics_organelle.csv'),
        'metrics_organelle.csv',
    )

    nonempty_results = [df for df in results.values() if df is not None and not df.empty]
    metrics_all = (
        pd.concat(nonempty_results, ignore_index=True)
        if nonempty_results
        else pd.DataFrame()
    )
    _write_csv(
        metrics_all,
        os.path.join(outdir, 'metrics_all_modalities.csv'),
        'metrics_all_modalities.csv',
    )

    # A focused setting-level robustness table used by the robustness plots.
    robustness_columns = [
        'modality', 'k', 'resolution', 'n_clusters',
        'stability_ari_mean', 'stability_ari_std',
        'stability_ari_q10', 'stability_ari_q90',
        'silhouette', 'n_cells_used', 'n_features_used',
    ]
    robustness_by_setting = (
        metrics_all[[c for c in robustness_columns if c in metrics_all.columns]].copy()
        if not metrics_all.empty
        else pd.DataFrame(columns=robustness_columns)
    )
    _write_csv(
        robustness_by_setting,
        os.path.join(outdir, 'robustness_by_setting.csv'),
        'robustness_by_setting.csv',
    )

    # Preserve the original top-five stability and silhouette selections.
    summary_frames = []
    for mod in ['RNA', 'Protein', 'Organelle']:
        df = results.get(mod)
        if df is not None and not df.empty:
            summary_frames.append(summarize(df, min_clusters=min_clusters))
    summary = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )
    _write_csv(summary, os.path.join(outdir, 'summary.csv'), 'summary.csv')

  
    try:
        for mod, df in results.items():
            if df is None or df.empty:
                continue
            pivot = df.pivot_table(
                index='resolution',
                columns='k',
                values='stability_ari_mean',
                aggfunc='mean',
            )
            fig, ax = plt.subplots()
            pivot.plot(ax=ax, marker='o')
            ax.set_title(f"{mod}: Stability (ARI mean) vs Resolution")
            ax.set_xlabel('Resolution')
            ax.set_ylabel('Stability ARI (mean)')
            plt.tight_layout()
            _savefig(
                fig,
                os.path.join(outdir, f"{mod.lower()}_stability_vs_resolution.png"),
                f"{mod} stability_vs_resolution",
            )
    except Exception as exc:
        print(f"Robustness plotting failed (non-fatal): {exc}")

    df_rna = results.get('RNA')
    df_protein = results.get('Protein')
    df_org = results.get('Organelle')
    return df_rna, df_protein, df_org, summary

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Comparative : RNA vs Protein vs Organelle (Three Modalities)")
    p.add_argument('--rna', default='...add path.../figure2_benchmark_Transcriptomics_input_matrix.csv', help='Path to RNA CSV (first col cell_id, last col excluded metadata).')
    p.add_argument('--protein', default='...add path.../figure2_benchmark_Proteomics_input_matrix.csv', help='Path to Protein CSV (first col cell_id, last col excluded metadata).')
    p.add_argument('--organelle', default='...add path.../figure2_benchmark_Organellomics_input_matrix.csv', help='Path to Organelle CSV (first col cell_id, last col excluded metadata).')
    p.add_argument('--outdir', default='results', help='Output directory (default: results)')

    # Preprocess options
    p.add_argument('--rna_hvgs', type=int, default=2000, help='Top HVGs for RNA (None/0 to skip)')
    p.add_argument('--protein_arcsinh', type=float, default=5.0, help='Arcsinh cofactor for protein (kept for API compatibility)')
    p.add_argument(
        '--organelle_log1p',
        action='store_true',
        default=False,
        help='Apply log1p to organelle features before scaling (default: off).',
    )

    # Metrics / neighbors
    p.add_argument('--rna_metric', type=str, default='cosine')
    p.add_argument('--protein_metric', type=str, default='euclidean')
    p.add_argument('--organelle_metric', type=str, default='euclidean')

    # PCA components: accept integer or "auto" (string)
    p.add_argument('--n_pcs_rna', type=str, default="80",
                   help='#PCs for RNA (int) or "auto" to pick by 90% variance with clip [30,80]')
    p.add_argument('--n_pcs_protein', type=str, default="40",
                   help='#PCs for Protein (int) or "auto" with clip [10,40]')
    p.add_argument('--n_pcs_organelle', type=str, default="50",
                   help='#PCs for Organelle (int) or "auto" with clip [15,50]')

    # Sweeps
    p.add_argument('--k_list', type=int, nargs='+', default=[5, 8, 10, 15, 30])
    p.add_argument('--res_list', type=float, nargs='+', default=[0.2, 0.4, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0])

    # Stability
    p.add_argument('--bootstrap_repeats', type=int, default=30)
    p.add_argument('--bootstrap_frac', type=float, default=0.8)

    # Repro
    p.add_argument('--seed', type=int, default=0)

    # Fairness controls
    p.add_argument('--feature_budget', default='match', help="'match' or integer")
    p.add_argument('--cell_budget', default='match', help="'match' or integer")

    # Composite decision weights (sum auto-normalized)
    p.add_argument('--w_robust', type=float, default=0.33, help='Weight for robustness (stability) in composite verdict')
    p.add_argument('--w_sep', type=float, default=0.33, help='Weight for separation (silhouette) in composite verdict')
    p.add_argument('--w_res', type=float, default=0.33, help='Weight for effective resolution (#clusters×stability) in composite verdict')

    # Cluster threshold
    p.add_argument('--min_clusters', type=int, default=3,
                   help='Minimum number of clusters a setting must have to be considered (>=2).')

    return p.parse_args()

def main():
    args = parse_args()

    feature_budget = args.feature_budget
    if isinstance(feature_budget, str) and feature_budget != 'match':
        try:
            feature_budget = int(feature_budget)
        except Exception:
            feature_budget = 'match'

    cell_budget = args.cell_budget
    if isinstance(cell_budget, str) and cell_budget != 'match':
        try:
            cell_budget = int(cell_budget)
        except Exception:
            cell_budget = 'match'

    def _parse_npcs(s: str) -> Union[int, str]:
        try:
            return int(s)
        except Exception:
            return "auto"

    min_clusters = max(2, int(args.min_clusters))

    df_rna, df_protein, df_org, summary = comparative_power_three(
        rna_csv=args.rna,
        protein_csv=args.protein,
        organelle_csv=args.organelle,
        outdir=args.outdir,
        rna_n_top_genes=None if args.rna_hvgs in [None, 0, -1] else int(args.rna_hvgs),
        protein_arcsinh=float(args.protein_arcsinh),
        organelle_log1p=bool(args.organelle_log1p),
        rna_metric=args.rna_metric,
        protein_metric=args.protein_metric,
        organelle_metric=args.organelle_metric,
        n_pcs_rna=_parse_npcs(args.n_pcs_rna),
        n_pcs_protein=_parse_npcs(args.n_pcs_protein),
        n_pcs_organelle=_parse_npcs(args.n_pcs_organelle),
        k_list=list(args.k_list),
        res_list=list(args.res_list),
        bootstrap_repeats=int(args.bootstrap_repeats),
        bootstrap_frac=float(args.bootstrap_frac),
        seed=int(args.seed),
        feature_budget=feature_budget,
        cell_budget=cell_budget,
        min_clusters=min_clusters,
    )

    raw, norm = build_scores_table(df_rna, df_protein, df_org, min_clusters=min_clusters)
    winner, comp_scores, narrative, w_norm = decide_winner(
        raw, norm,
        w_robust=float(args.w_robust),
        w_sep=float(args.w_sep),
        w_res=float(args.w_res),
    )

    # CSVs directly supporting best-setting selection and robustness scoring.
    best_settings = raw[[
        'best_k', 'best_resolution', 'best_n_clusters',
        'best_stability', 'best_silhouette',
        'cells_used', 'features_used',
    ]].copy()
    best_settings.index.name = 'modality'
    _write_csv(
        best_settings.reset_index(),
        os.path.join(args.outdir, 'best_clustering_settings.csv'),
        'best_clustering_settings.csv',
    )

    robustness_summary = raw[[
        'stab_median', 'stab_q90', 'best_stability', 'robustness',
        'best_k', 'best_resolution', 'best_n_clusters',
    ]].copy()
    robustness_summary['robustness_normalized'] = norm['robustness']
    robustness_summary.index.name = 'modality'
    _write_csv(
        robustness_summary.reset_index(),
        os.path.join(args.outdir, 'robustness_summary.csv'),
        'robustness_summary.csv',
    )

    score_components = raw.copy()
    for axis in ['robustness', 'separation', 'resolution_eff']:
        score_components[f'{axis}_normalized'] = norm[axis]
    score_components['composite_score'] = comp_scores['composite_score']
    score_components.index.name = 'modality'
    _write_csv(
        score_components.reset_index(),
        os.path.join(args.outdir, 'composite_score_components.csv'),
        'composite_score_components.csv',
    )

    os.makedirs(args.outdir, exist_ok=True)
    verdict_json = {
        'winner': winner,
        'weights_input': {
            'robustness': float(args.w_robust),
            'separation': float(args.w_sep),
            'resolution_eff': float(args.w_res),
        },
        'weights_used': {
            'robustness': float(w_norm[0]),
            'separation': float(w_norm[1]),
            'resolution_eff': float(w_norm[2]),
        },
        'raw_metrics': raw.to_dict(orient='index'),
        'normalized_axes': norm.to_dict(orient='index'),
        'composite_scores': comp_scores['composite_score'].to_dict(),
        'narrative': narrative,
    }
    verdict_json_path = os.path.join(args.outdir, 'verdict.json')
    with open(verdict_json_path, 'w') as f:
        json.dump(verdict_json, f, indent=2)
    print(f"[write] verdict.json: {os.path.abspath(verdict_json_path)}")

    verdict_txt_path = os.path.join(args.outdir, 'verdict.txt')
    with open(verdict_txt_path, 'w') as f:
        f.write(narrative + "")
    print(f"[write] verdict.txt: {os.path.abspath(verdict_txt_path)}")

    # General metrics table (per modality)
    gm = build_general_metrics_table_from_dfs(df_rna, df_protein, df_org, comp_scores, min_clusters=min_clusters)

    # NEW: Bootstrap composite scores over the sweep grid + bar plot
    try:
        boot_df = bootstrap_composite_scores(
            df_rna,
            df_protein,
            df_org,
            min_clusters=min_clusters,
            w_robust=float(args.w_robust),
            w_sep=float(args.w_sep),
            w_res=float(args.w_res),
            n_boot=200,
            seed=int(args.seed),
        )
        if boot_df is not None and not boot_df.empty:
            # merge into gm
            for col in boot_df.columns:
                gm[col] = boot_df[col]

            boot_csv_path = os.path.join(args.outdir, 'composite_bootstrap_stats.csv')
            _write_csv(boot_df.reset_index(), boot_csv_path, "composite_bootstrap_stats.csv")

            bar_data_columns = [
                'composite_score',
                'composite_bootstrap_mean',
                'composite_bootstrap_std',
            ]
            bar_data = gm[[c for c in bar_data_columns if c in gm.columns]].copy()
            bar_data.index.name = 'modality'
            _write_csv(
                bar_data.reset_index(),
                os.path.join(args.outdir, 'composite_bootstrap_barplot_data.csv'),
                'composite_bootstrap_barplot_data.csv',
            )

            barplot_path = os.path.join(args.outdir, 'composite_bootstrap_barplot.png')
            plot_composite_bars(gm, barplot_path, dpi=300)
    except Exception as e:
        print(f"Composite bootstrap / barplot failed (non-fatal): {e}")

    gm_path_csv = os.path.join(args.outdir, 'general_metrics_table.csv')
    _write_csv(gm.reset_index(), gm_path_csv, "general_metrics_table.csv")

    png_path = os.path.join(args.outdir, 'general_metrics_table.png')
    save_table_as_png(gm, png_path)
    print(f"[write] general_metrics_table.png: {os.path.abspath(png_path)}")

    print("Saved to:", os.path.abspath(args.outdir))
    print(" - general_metrics_table.png and general_metrics_table.csv")
    print(" - composite_bootstrap_barplot.png and its two supporting CSVs")
    print(" - best_clustering_settings.csv and composite_score_components.csv")
    print(" - robustness_summary.csv, robustness_by_setting.csv, and stability plots")
    print(" - per-modality and combined setting-level metrics CSVs")
    print(" - summary.csv, verdict.json, and verdict.txt")

if __name__ == '__main__':
    main()
