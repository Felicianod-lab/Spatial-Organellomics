#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial organellomics niche metrics: 
------------------------------------

This script reads a spatial organellomics CSV, computes one set of
position- and neighborhood-based category metrics per acinus, and writes a
small Control-focused figure set plus CSV summary tables.

Generated plots:
- Control zero-centered normalized bars with points for adjacency,
  fraction explained, position accuracy, and residual Moran's p
- Control violin plot with points for percent not explained by position
"""

import argparse
import re
from pathlib import Path
from typing import Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

# ---------------------- Config / Labels ----------------------

FRIENDLY_LABELS = {
    0:"H1", 1:"H2", 2:"H3", 3:"H4", 4:"H5",
    5:"FH1", 6:"FH2", 7:"FH3", 8:"FH4",
    9:"WH1", 10:"WH2"
}
DEFAULT_CSV =r"...add path.../Fig3_Full_Acinar_Category_Spatial_Analysis_Matrix.csv"

# ---------------------- Utilities ----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_acinus_id(label: str) -> str:
    """
    Extract canonical 'L#L#a#' (e.g., L1L1a0) from labels like 'CNT_L1L1a0s0_FOO'.
    - Robust to case (L/l, A/a) and trailing scan suffix (s##).
    - If underscore is present, search only after the first underscore.
    """
    if not isinstance(label, str):
        return "NA"
    rest = label.split("_", 1)[1] if "_" in label else label
    m = re.search(r'([Ll]\d+[Ll]\d+[Aa]\d+)', rest)
    if m:
        token = m.group(1)
        token = token.replace("A", "a").replace("l", "L")  # normalize casing
        return token
    rest2 = re.sub(r's\d+.*$', '', rest)  # Fallback: trim trailing scan suffix like s12
    return rest2 if rest2 else "NA"

def parse_liver_id(acinus_id: str) -> str:
    if not isinstance(acinus_id, str):
        return "NA"
    m = re.match(r'^(L\d+)', acinus_id)
    return m.group(1) if m else "NA"

def to_cat_levels(series: pd.Series) -> List[Any]:
    as_num = pd.to_numeric(series, errors='coerce')
    if as_num.notna().mean() > 0.8:
        levels = sorted(pd.unique(as_num.dropna()))
        if np.all(np.isclose(levels, np.round(levels))):
            levels = [int(x) for x in levels]
        return levels
    else:
        return sorted(pd.unique(series.astype(str)))

def map_cats(series: pd.Series, levels: List[Any]) -> np.ndarray:
    if all(isinstance(l, (int, np.integer)) for l in levels):
        vals = pd.to_numeric(series, errors='coerce')
    else:
        vals = series.astype(str)
    mapping = {levels[i]: i for i in range(len(levels))}
    out = []
    for v in vals:
        if pd.isna(v):
            out.append(np.nan)
        else:
            key = v
            if isinstance(v, str) and all(isinstance(l, int) for l in levels):
                try:
                    key = int(v)
                except Exception:
                    key = v
            out.append(mapping.get(key, np.nan))
    return np.asarray(out, dtype=float)

def canonical_group_name(group_val, cat_val):
    """Map to {'Control','Fasted','WD'} preferring 'group' text; fallback to CAT range."""
    s = str(group_val).lower().strip()
    if ("fast" in s) or (s in {"2","fasted"}):
        return "Fasted"
    if ("wd" in s) or ("western" in s) or (s in {"3"}):
        return "WD"
    if ("ctrl" in s) or ("control" in s) or (s in {"1","cnt","ct"}):
        return "Control"
    try:
        c = int(cat_val)
        if 0 <= c <= 4: return "Control"
        if 5 <= c <= 8: return "Fasted"
        if 9 <= c <= 10: return "WD"
    except Exception:
        pass
    return "Control"

# ---- Label/color parsing ----

def _parse_kv_mapping(s: str) -> dict:
    if not s:
        return {}
    out = {}
    for tok in re.split(r'[;,]', s):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        out[k.strip()] = v.strip()
    return out

def _parse_palette_spec(palette: str, n: int):
    if not palette:
        return [None]*n
    s = palette.strip()
    # If explicit list or hex color(s)
    if any(sep in s for sep in (',',';')) or s.startswith('#'):
        parts = [t.strip() for t in re.split(r'[;,]', s) if t.strip()]
        return [parts[i % len(parts)] for i in range(n)] if parts else [None]*n
    # Try as a colormap name
    try:
        cmap = plt.get_cmap(s)
        if n == 1:
            return [cmap(0.6)]
        return [cmap(i/(n-1)) for i in range(n)]
    except Exception:
        pass
    # Try as a named color
    try:
        _ = to_rgba(s)
        return [s for _ in range(n)]
    except Exception:
        return [None]*n

def _group_colors(groups, palette: str, group_color_map: str, bar_alpha: float = 1.0):
    m = _parse_kv_mapping(group_color_map)
    base = _parse_palette_spec(palette, len(groups))
    out = []
    for i, g in enumerate(groups):
        c = m.get(g, base[i] if i < len(base) else None)
        if c is None:
            out.append(None)
        else:
            rgba = list(to_rgba(c))
            rgba[3] = bar_alpha
            out.append(tuple(rgba))
    return out

def _apply_ticklabels(ax, groups, ticklabels=None):
    x = np.arange(len(groups)) + 1
    if ticklabels is None:
        ax.set_xticks(x); ax.set_xticklabels(groups, rotation=0)
    else:
        ax.set_xticks(x); ax.set_xticklabels(ticklabels, rotation=0)

# ---------------------- Binning & Entropy ----------------------

def compute_bins(df_all: pd.DataFrame, n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    vmin = np.nanmin(df_all["ascini_position"].to_numpy())
    vmax = np.nanmax(df_all["ascini_position"].to_numpy())
    lo = -1.0 if vmin >= -1.2 else vmin
    hi = 1.0 if vmax <= 1.2 else vmax
    bins = np.linspace(lo, hi, n_bins+1)
    centers = 0.5*(bins[:-1]+bins[1:])
    return bins, centers

def safe_weights_from_bins(bin_idx: np.ndarray, n_bins: int):
    tot_raw = np.bincount(bin_idx, minlength=n_bins).astype(float)
    tot_safe = np.where(tot_raw == 0, 1.0, tot_raw)
    w = np.where(tot_raw == 0, 0.0, tot_raw)
    ws = w.sum()
    if ws > 0: w = w / ws
    return tot_raw, tot_safe, w

def per_acinus_P_c_given_bin(sdf: pd.DataFrame, cat_levels: List[Any], bins: np.ndarray):
    vals = sdf["ascini_position"].to_numpy(float)
    cats_idx = map_cats(sdf["CAT"], cat_levels)  # float with possible NaN
    n_bins = len(bins)-1
    bin_idx = np.digitize(vals, bins) - 1
    valid = (bin_idx>=0) & (bin_idx < n_bins) & np.isfinite(cats_idx)
    bin_idx = bin_idx[valid]
    cats_valid = cats_idx[valid].astype(int)
    tot_raw, tot_safe, w = safe_weights_from_bins(bin_idx, n_bins)
    C = len(cat_levels)
    P = np.zeros((n_bins, C), float)
    for ci in range(C):
        cnt = np.bincount(bin_idx[cats_valid==ci], minlength=n_bins).astype(float)
        P[:, ci] = cnt / tot_safe
    # w already normalized in safe_weights_from_bins
    return P, w, bin_idx, cats_idx, valid

def entropy(p):
    p = np.asarray(p, float)
    p = p[p>0]
    return -np.sum(p*np.log2(p)) if p.size>0 else 0.0

def acinus_fraction_explained_and_accuracy(sdf: pd.DataFrame, cat_levels: List[Any], bins: np.ndarray):
    P, w, bin_idx, cats_idx, valid = per_acinus_P_c_given_bin(sdf, cat_levels, bins)
    mapped = map_cats(sdf["CAT"], cat_levels)
    counts = np.bincount(mapped[np.isfinite(mapped)].astype(int), minlength=len(cat_levels)).astype(float)
    p_overall = counts / counts.sum() if counts.sum()>0 else counts
    H = entropy(p_overall)
    Hcx = 0.0
    for i in range(P.shape[0]):
        Hcx += w[i] * entropy(P[i])
    frac_expl = (1.0 - Hcx/H) if H>0 else np.nan
    e_bayes = float(np.sum(w*(1.0 - np.max(P, axis=1))))
    acc = 1.0 - e_bayes
    return frac_expl, acc

def fraction_explained_perm_p(sdf: pd.DataFrame, cat_levels: List[Any], bins: np.ndarray,
                              n_perm: int = 500, rng_seed: int = 2025):
    FE_obs, _ = acinus_fraction_explained_and_accuracy(sdf, cat_levels, bins)
    if not np.isfinite(FE_obs):
        return np.nan, np.nan, np.nan
    vals = sdf["ascini_position"].to_numpy(float)
    n_bins = len(bins) - 1
    bin_idx = np.digitize(vals, bins) - 1
    cats_f = map_cats(sdf["CAT"], cat_levels)
    valid = (bin_idx >= 0) & (bin_idx < n_bins) & np.isfinite(cats_f)
    bin_idx = bin_idx[valid]
    cats_valid = cats_f[valid].astype(int)
    rng = np.random.default_rng(rng_seed)
    perm_vals = np.full(n_perm, np.nan, float)

    def FE_from_assign(bin_idx_local, cats_local):
        tot_raw = np.bincount(bin_idx_local, minlength=n_bins).astype(float)
        tot_safe = np.where(tot_raw == 0, 1.0, tot_raw)
        w = tot_raw / np.sum(tot_raw) if np.sum(tot_raw) > 0 else tot_raw
        C = len(cat_levels)
        P = np.zeros((n_bins, C), float)
        for ci in range(C):
            cnt = np.bincount(bin_idx_local[cats_local==ci], minlength=n_bins).astype(float)
            P[:, ci] = cnt / tot_safe
        counts = np.bincount(cats_local, minlength=C).astype(float)
        p_overall = counts / counts.sum() if counts.sum()>0 else counts
        H = entropy(p_overall)
        if H <= 0: 
            return np.nan
        Hcx = 0.0
        for i in range(n_bins):
            Hcx += (w[i] if i < len(w) else 0.0) * entropy(P[i])
        return 1.0 - Hcx/H

    for t in range(n_perm):
        cats_perm = rng.permutation(cats_valid)
        perm_vals[t] = FE_from_assign(bin_idx, cats_perm)

    perm_finite = perm_vals[np.isfinite(perm_vals)]
    if perm_finite.size == 0:
        return np.nan, np.nan, np.nan
    mu = float(np.nanmean(perm_finite))
    sd = float(np.nanstd(perm_finite, ddof=1)) if perm_finite.size>1 else np.nan
    p = (1.0 + np.sum(perm_finite >= FE_obs)) / (perm_finite.size + 1.0)
    return float(p), mu, sd

# ---------------------- Graphs & residual Moran's I ----------------------

def build_neighbors(coords: np.ndarray, k: int = 8, radius: float = None):
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=min(k+1, len(coords)))
    if k == 1 or len(idxs.shape) == 1:
        idxs = np.expand_dims(idxs, 1)
    neighbors = []
    for i in range(len(coords)):
        nbrs = [j for j in idxs[i] if j != i]
        neighbors.append(nbrs)
    if radius is not None and radius > 0:
        within = tree.query_ball_point(coords, r=radius)
        for i in range(len(coords)):
            extra = [j for j in within[i] if j != i]
            neighbors[i] = list(sorted(set(neighbors[i]).union(extra)))
    return neighbors

def symmetrize_edges(neighbors):
    edges = set()
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return sorted(edges)

def residual_morans_I_mean(sdf: pd.DataFrame, cat_levels: List[Any], bins: np.ndarray, k: int = 8, radius: float = None):
    coords = sdf[["centroid-0","centroid-1"]].to_numpy(float)
    if len(coords) < 3:
        return np.nan
    neighbors = build_neighbors(coords, k=k, radius=radius)
    edges = symmetrize_edges(neighbors)
    if len(edges)==0:
        return np.nan
    cats_idx = map_cats(sdf["CAT"], cat_levels)  # float with possible NaN
    n_bins = len(bins)-1
    vals = sdf["ascini_position"].to_numpy(float)
    bin_idx = np.digitize(vals, bins) - 1
    valid_mask = (bin_idx>=0) & (bin_idx < n_bins) & np.isfinite(cats_idx)

    tot_raw, tot_safe, w = safe_weights_from_bins(bin_idx[valid_mask], n_bins)
    C = len(cat_levels)
    p_hat_bin = np.zeros((n_bins, C), float)
    for ci in range(C):
        cnt = np.bincount(bin_idx[(cats_idx==ci) & valid_mask], minlength=n_bins).astype(float)
        p_hat_bin[:, ci] = cnt / np.where(tot_raw==0, 1.0, tot_raw)

    W = np.zeros((len(sdf), len(sdf)), float)
    for i,j in edges:
        W[i,j] = 1.0; W[j,i] = 1.0
    W_sum = W.sum()
    if W_sum==0:
        return np.nan

    I_vals = []
    for ci in range(C):
        y = (cats_idx==ci).astype(float)
        ph = np.array([p_hat_bin[bin_idx[i], ci] if valid_mask[i] else np.nan for i in range(len(sdf))])
        keep = np.isfinite(ph)
        if keep.sum()<3:
            continue
        yk = y[keep]; phk = ph[keep]
        # Build W only on kept subset
        Wk = np.zeros((keep.sum(), keep.sum()), float)
        idx_keep = np.where(keep)[0]
        pos = -np.ones(len(keep), dtype=int)
        pos[idx_keep] = np.arange(len(idx_keep))
        for (i,j) in edges:
            ii = pos[i]; jj = pos[j]
            if ii >= 0 and jj >= 0 and ii != jj:
                Wk[ii, jj] = 1.0; Wk[jj, ii] = 1.0
        Wk_sum=Wk.sum()
        if Wk_sum==0 or np.all(yk==phk):
            continue
        r = yk - phk; r_bar = np.mean(r)
        num = (r - r_bar).T @ Wk @ (r - r_bar)
        den = ((r - r_bar)**2).sum()
        I = (len(r)/Wk_sum) * (num/den) if den>0 else np.nan
        if np.isfinite(I):
            I_vals.append(I)
    return float(np.nanmean(I_vals)) if I_vals else np.nan

def residual_morans_p_per_acinus(sdf: pd.DataFrame, cat_levels: List[Any], bins: np.ndarray, k: int = 8, radius: float = None, n_perm: int = 200):
    coords = sdf[["centroid-0","centroid-1"]].to_numpy(float)
    if len(coords) < 3:
        return np.nan
    neighbors = build_neighbors(coords, k=k, radius=radius)
    edges = symmetrize_edges(neighbors)
    if len(edges)==0:
        return np.nan
    cats_idx = map_cats(sdf["CAT"], cat_levels)  # float with possible NaN
    n_bins = len(bins)-1
    vals = sdf["ascini_position"].to_numpy(float)
    bin_idx = np.digitize(vals, bins) - 1
    valid_mask = (bin_idx>=0) & (bin_idx < n_bins) & np.isfinite(cats_idx)

    tot_raw, tot_safe, w = safe_weights_from_bins(bin_idx[valid_mask], n_bins)
    C = len(cat_levels)
    p_hat_bin = np.zeros((n_bins, C), float)
    for ci in range(C):
        cnt = np.bincount(bin_idx[(cats_idx==ci) & valid_mask], minlength=n_bins).astype(float)
        p_hat_bin[:, ci] = cnt / np.where(tot_raw==0, 1.0, tot_raw)

    def mean_I_for_labels(lbl_idx):
        I_vals = []
        for ci in range(C):
            y = (lbl_idx==ci).astype(float)
            ph = np.array([p_hat_bin[bin_idx[i], ci] if valid_mask[i] else np.nan for i in range(len(sdf))])
            keep = np.isfinite(ph)
            if keep.sum()<3: continue
            yk = y[keep]; phk = ph[keep]
            # Build W only once per subset to avoid shape mismatch
            W = np.zeros((keep.sum(), keep.sum()), float)
            idx_keep = np.where(keep)[0]
            pos = -np.ones(len(keep), dtype=int)
            pos[idx_keep] = np.arange(len(idx_keep))
            for (i,j) in edges:
                ii = pos[i]; jj = pos[j]
                if ii >= 0 and jj >= 0 and ii != jj:
                    W[ii, jj] = 1.0; W[jj, ii] = 1.0
            W_sum = W.sum()
            if W_sum==0 or np.all(yk==phk): continue
            r = yk - phk; r_bar = np.mean(r)
            num = (r - r_bar).T @ W @ (r - r_bar)
            den = ((r - r_bar)**2).sum()
            I = (len(r)/W_sum) * (num/den) if den>0 else np.nan
            if np.isfinite(I):
                I_vals.append(I)
        return float(np.nanmean(I_vals)) if I_vals else np.nan

    I_obs = mean_I_for_labels(cats_idx)
    if not np.isfinite(I_obs):
        return np.nan

    rng = np.random.default_rng(12345)
    null_I = np.full(n_perm, np.nan, float)
    bin_to_idx = {b: np.where((bin_idx==b) & valid_mask)[0] for b in range(n_bins)}
    for t in range(n_perm):
        y_perm = cats_idx.copy()
        for b, idxs in bin_to_idx.items():
            if len(idxs)>1:
                y_perm[idxs] = rng.permutation(y_perm[idxs])
        null_I[t] = mean_I_for_labels(y_perm)

    mu = np.nanmean(null_I)
    diffs = np.abs(null_I - mu)
    o = abs(I_obs - mu)
    p = (1 + np.nansum(diffs >= o)) / (np.sum(np.isfinite(null_I)) + 1)
    return float(p)

# ---------------------- Adjacency pos-conditioned ----------------------

def adjacency_sig_count_per_acinus(sdf: pd.DataFrame, cat_levels: List[Any], bins: np.ndarray,
                                   k: int = 8, radius: float = None, n_perm: int = 200, alpha: float = 0.05):
    coords = sdf[["centroid-0","centroid-1"]].to_numpy(float)
    if len(coords) < 3:
        return np.nan, np.nan
    neighbors = build_neighbors(coords, k=k, radius=radius)
    edges = symmetrize_edges(neighbors)
    if len(edges)==0:
        return np.nan, np.nan
    cats_f = map_cats(sdf["CAT"], cat_levels)  # float with possible NaN
    keep = np.isfinite(cats_f)
    cats_idx = np.full(len(cats_f), -1, dtype=int)
    cats_idx[keep] = cats_f[keep].astype(int)
    vals = sdf["ascini_position"].to_numpy(float)
    n_bins = len(bins)-1
    bin_idx = np.digitize(vals, bins) - 1

    C = len(cat_levels)
    obs = np.zeros((C,C), float)
    for (i,j) in edges:
        if cats_idx[i] < 0 or cats_idx[j] < 0:
            continue
        ci = cats_idx[i]; cj = cats_idx[j]
        obs[ci, cj] += 1; obs[cj, ci] += 1

    rng = np.random.default_rng(2468)
    null = np.zeros((n_perm, C, C), float)
    for t in range(n_perm):
        M = np.zeros((C,C), float)
        tmp = cats_idx.copy()
        for b in range(n_bins):
            idxs = np.where((bin_idx==b) & (cats_idx>=0))[0]  # permute only valid labels
            if len(idxs)>1:
                tmp[idxs] = rng.permutation(tmp[idxs])
        for (i,j) in edges:
            if tmp[i] < 0 or tmp[j] < 0:
                continue
            ci = tmp[i]; cj = tmp[j]
            M[ci, cj] += 1; M[cj, ci] += 1
        null[t] = M

    mu = null.mean(axis=0); diffs = np.abs(null - mu[None, :, :])
    obs_dev = np.abs(obs - mu)
    p = np.zeros((C,C), float)
    present_cats = np.unique(cats_idx[cats_idx>=0]).astype(int)
    Nc = len(present_cats)
    for i in range(C):
        for j in range(C):
            arr = diffs[:, i, j]; o = obs_dev[i, j]
            p[i, j] = (1 + np.sum(arr >= o)) / (arr.size + 1)

    count = 0
    for i in present_cats:
        for j in present_cats:
            if j < i: continue
            if p[i, j] < alpha:
                count += 1
    return int(count), int(Nc)

# ---------------------- Per-acinus metrics ----------------------

PERA_COLS = [
    "group","acinus_id","fraction_explained","fraction_explained_p",
    "fraction_explained_null_mean","fraction_explained_null_sd",
    "position_accuracy","residual_moransI_mean","residual_morans_p",
    "adjacency_sig_pairs","Nc_present",
    "norm_fraction_explained","norm_position_accuracy",
    "norm_residual_morans_p","norm_adjacency_sig"
]

def compute_per_acinus_metrics(df: pd.DataFrame, n_bins: int = 30, k: int = 8, radius: float = None, 
                               adj_perm: int = 200, morans_perm: int = 200, frac_perm: int = 500,
                               min_cells: int = 5) -> pd.DataFrame:
    cat_levels = to_cat_levels(df["CAT"])
    bins, _ = compute_bins(df, n_bins=n_bins)
    rows = []
    for (group, acinus_id), sdf in df.groupby(["group_canon", "acinus_id"]):
        if len(sdf) < min_cells:
            continue
        frac, acc = acinus_fraction_explained_and_accuracy(sdf, cat_levels, bins)
        morI = residual_morans_I_mean(sdf, cat_levels, bins, k=k, radius=radius)
        morP = residual_morans_p_per_acinus(sdf, cat_levels, bins, k=k, radius=radius, n_perm=morans_perm)
        adjN, Nc = adjacency_sig_count_per_acinus(sdf.copy(), cat_levels, bins, k=k, radius=radius, n_perm=adj_perm, alpha=0.05)
        frac_p, frac_null_mean, frac_null_sd = fraction_explained_perm_p(sdf, cat_levels, bins, n_perm=frac_perm, rng_seed=2025)

        # Zero-centered scores
        norm_acc  = (acc  - 0.80) / 0.20 if np.isfinite(acc)  else np.nan
        norm_frac = (frac - 0.70) / 0.20 if np.isfinite(frac) else np.nan   # or 0.60
        norm_morP = np.log10(morP / 0.05) if (np.isfinite(morP) and morP>0) else np.nan
        if np.isfinite(Nc) and Nc>0:
            E = 0.05 * (Nc*(Nc+1)/2.0)
            denom = max(E, 1.0)
            norm_adj = (E - adjN) / denom if np.isfinite(adjN) else np.nan
        else:
            norm_adj = np.nan

        rows.append({
            "group": group, "acinus_id": acinus_id,
            "fraction_explained": frac,
            "fraction_explained_p": frac_p,
            "fraction_explained_null_mean": frac_null_mean,
            "fraction_explained_null_sd": frac_null_sd,
            "position_accuracy": acc,
            "residual_moransI_mean": morI,
            "residual_morans_p": morP,
            "adjacency_sig_pairs": adjN,
            "Nc_present": Nc,
            "norm_fraction_explained": norm_frac,
            "norm_position_accuracy": norm_acc,
            "norm_residual_morans_p": norm_morP,
            "norm_adjacency_sig": norm_adj
        })
    if not rows:
        return pd.DataFrame(columns=PERA_COLS)
    return pd.DataFrame(rows)

# ---------------------- Plot helpers ----------------------

def _finite_concat(values_by_group):
    arrs = []
    for g, v in values_by_group.items():
        a = np.asarray(v, float)
        a = a[np.isfinite(a)]
        if a.size:
            arrs.append(a)
    if not arrs:
        return np.array([])
    return np.concatenate(arrs)

def _compute_bar_ylim(means, sems, zero_line=None, symmetric_about_zero=False, min_span=1e-6):
    mm = np.array(means, float); ss = np.array(sems, float)
    lo = np.nanmin(mm - np.nan_to_num(ss, nan=0.0)) if np.any(np.isfinite(mm)) else 0.0
    hi = np.nanmax(mm + np.nan_to_num(ss, nan=0.0)) if np.any(np.isfinite(mm)) else 1.0
    if zero_line is not None:
        lo = min(lo, zero_line)
        hi = max(hi, zero_line)
    if symmetric_about_zero or (zero_line == 0.0):
        span = max(abs(lo), abs(hi))
        lo, hi = -span, +span
    if hi - lo < min_span:
        lo -= 0.5
        hi += 0.5
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad

def _compute_dist_ylim(values_by_group, zero_line=None, symmetric_about_zero=False, min_span=1e-6):
    x = _finite_concat(values_by_group)
    if x.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if zero_line is not None:
        lo = min(lo, zero_line)
        hi = max(hi, zero_line)
    if symmetric_about_zero or (zero_line == 0.0):
        span = max(abs(lo), abs(hi))
        lo, hi = -span, +span
    if hi - lo < min_span:
        lo -= 0.5
        hi += 0.5
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad

def bar_with_sem(ax, groups, values_by_group, title, ylabel, zero_line=None, symmetric_about_zero=False,
                 colors=None, ticklabels=None, bar_width: float = 0.6):
    means = []
    sems = []
    for g in groups:
        vals = np.asarray(values_by_group[g], float)
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)) if vals.size else np.nan)
        sems.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan)
    x = np.arange(len(groups)) + 1
    kw = {"yerr":sems, "capsize":5, "width": bar_width}
    if colors is not None: kw["color"] = colors
    ax.bar(x, means, **kw)
    if zero_line is not None:
        ax.axhline(zero_line, linestyle='--', linewidth=1, zorder=1)
    y0, y1 = _compute_bar_ylim(means, sems, zero_line=zero_line, symmetric_about_zero=symmetric_about_zero)
    ax.set_ylim(y0, y1)
    _apply_ticklabels(ax, groups, ticklabels=ticklabels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    return means, sems, x, (y0, y1)

def bar_with_bootci(ax, groups, values_by_group, title, ylabel, ci_by_group, zero_line=None, symmetric_about_zero=False,
                    colors=None, ticklabels=None, bar_width: float = 0.6):
    means = [np.nanmean(values_by_group[g]) if len(values_by_group[g])>0 else np.nan for g in groups]
    lows  = [ci_by_group.get(g, (np.nan, np.nan))[0] for g in groups]
    highs = [ci_by_group.get(g, (np.nan, np.nan))[1] for g in groups]
    x = np.arange(len(groups)) + 1
    kw = {"width": bar_width}
    if colors is not None: kw["color"] = colors
    ax.bar(x, means, **kw)
    for i, (m, lo, hi) in enumerate(zip(means, lows, highs), start=1):
        if np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi):
            ax.vlines(i, lo, hi, linewidth=2)
            ax.hlines([lo, hi], i-0.15, i+0.15, linewidth=2)
    if zero_line is not None:
        ax.axhline(zero_line, linestyle='--', linewidth=1, zorder=1)
    mm = np.array(means, float); lo_arr = np.array(lows, float); hi_arr = np.array(highs, float)
    finite = np.isfinite(mm) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
    if np.any(finite):
        lo_y = np.nanmin(np.vstack([mm[finite], lo_arr[finite]]))
        hi_y = np.nanmax(np.vstack([mm[finite], hi_arr[finite]]))
    else:
        lo_y, hi_y = (0.0, 1.0)
    if zero_line is not None:
        lo_y = min(lo_y, zero_line)
        hi_y = max(hi_y, zero_line)
    if symmetric_about_zero or (zero_line == 0.0):
        span = max(abs(lo_y), abs(hi_y))
        lo_y, hi_y = -span, +span
    pad = 0.08 * (hi_y - lo_y)
    ax.set_ylim(lo_y - pad, hi_y + pad)
    _apply_ticklabels(ax, groups, ticklabels=ticklabels)
    ax.set_ylabel(ylabel); ax.set_title(title)

def safe_distribution_plot(ax, groups, values_by_group, title, ylabel, zero_line=None, symmetric_about_zero=False,
                           ticklabels=None, violin_colors=None, violin_alpha=0.8, violin_width=0.6):
    """
    Draw violin if each group has >=2 finite values; otherwise fall back to scatter.
    Returns True if a violin was drawn (so caller may overlay points), False otherwise.
    """
    data = []
    use_violin = True
    for g in groups:
        vals = np.array(values_by_group[g], float)
        finite = vals[np.isfinite(vals)]
        data.append(finite if finite.size>0 else np.array([]))
        if finite.size < 2:
            use_violin = False

    x = np.arange(len(groups)) + 1
    if use_violin:
        vp = ax.violinplot(data, showmeans=True, showextrema=False, widths=violin_width)
        # Color violins if colors provided
        if violin_colors is not None:
            for i, b in enumerate(vp['bodies']):
                if i < len(violin_colors) and violin_colors[i] is not None:
                    b.set_facecolor(violin_colors[i])
                    b.set_edgecolor('black')
                    b.set_alpha(violin_alpha)
    else:
        for i, vals in enumerate(data, start=1):
            if vals.size == 0:
                continue
            jitter = (np.random.rand(vals.size)-0.5)*0.15
            ax.scatter(np.full(vals.size, i)+jitter, vals, alpha=0.8)

    if zero_line is not None:
        ax.axhline(zero_line, linestyle='--', linewidth=1, zorder=1)
    y0, y1 = _compute_dist_ylim(values_by_group, zero_line=zero_line, symmetric_about_zero=symmetric_about_zero)
    ax.set_ylim(y0, y1)
    _apply_ticklabels(ax, groups, ticklabels=ticklabels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    return use_violin

def annotate_pvals_vs_control(ax, groups, values_by_group, base_ylim=None, control_label="Control",
                              y_padding_ratio=0.05, annot_size: Optional[float] = None):
    if control_label not in groups:
        return
    ctrl_vals = np.array(values_by_group[control_label])
    ctrl_vals = ctrl_vals[np.isfinite(ctrl_vals)]
    if ctrl_vals.size < 2:
        return
    y0, y1 = ax.get_ylim() if base_ylim is None else base_ylim
    max_y = y1
    span = max(1e-9, y1 - y0)
    for idx, g in enumerate(groups):
        if g == control_label:
            continue
        vals = np.array(values_by_group[g])
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        try:
            stat, p = mannwhitneyu(ctrl_vals, vals, alternative='two-sided')
        except Exception:
            p = np.nan
        x = idx + 1
        y = np.nanmean(vals) if np.isfinite(np.nanmean(vals)) else 0.0
        pad = max(1e-9, y_padding_ratio * span)
        y_text = y + pad*3
        fs = annot_size if annot_size is not None else float(plt.rcParams.get("ytick.labelsize", 10))
        ax.text(x, y_text, f"p={p:.3g}", ha='center', va='bottom', fontsize=fs)
        max_y = max(max_y, y_text + pad*2)
    if max_y > y1:
        ax.set_ylim(y0, max_y)

def overlay_points_on_bars(ax, groups, values_by_group, colors=None,
                           jitter=0.18, alpha=0.65, size=18, seed=123,
                           zero_line=None, symmetric_about_zero=False,
                           edgecolor='white', edgewidth=0.5):
    rng = np.random.default_rng(seed)
    all_vals = []
    for i, g in enumerate(groups, start=1):
        v = np.asarray(values_by_group[g], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        x = rng.uniform(i - jitter, i + jitter, size=v.size)
        color = (colors[i-1] if (colors is not None and (i-1) < len(colors)) else None)
        ax.scatter(x, v, s=size, alpha=alpha, c=[color] if color else None,
            edgecolors=edgecolor, linewidths=edgewidth, zorder=3)
        all_vals.append(v)
    if all_vals:
        values_by_group_tmp = {g: np.asarray(values_by_group[g], float) for g in groups}
        y0, y1 = _compute_dist_ylim(values_by_group_tmp, zero_line=zero_line, symmetric_about_zero=symmetric_about_zero)
        cy0, cy1 = ax.get_ylim()
        ax.set_ylim(min(y0, cy0), max(y1, cy1))


def save_publication_figure(fig, path: Path, dpi: int = 300):
    """Save a figure while preserving the original styling and preventing label clipping."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def write_metrics_readme(outdir_path: Path):
    """Write a README explaining what the code measures and how normalized scores are interpreted."""
    readme = """ # Spatial organellomics niche metrics

This code analyzes a spatial organellomics CSV containing per-cell category labels, acinar position, and cell coordinates. It computes one value per acinus for several metrics that ask whether category identity is mainly explained by acinar position or whether additional spatial/neighborhood structure remains after accounting for position.

The plotted values are Control acini. Each point represents one acinus. Bars summarize the mean across acini. The violin plot shows the distribution across acini.

## Required input columns

The input CSV must contain these columns:

- `labels`: used to parse acinus identity.
- `group`: used only to identify the rows plotted as Control. For a Control-only file, set every row to `Control`.
- `CAT`: category label being analyzed.
- `centroid-0` and `centroid-1`: spatial coordinates used to build neighbor graphs and compute residual spatial statistics.
- `ascini_position`: acinar position coordinate used for position binning.

## Unit of analysis

The code groups cells by acinus. Most metrics are computed once per acinus. The figure points therefore represent acini, not individual cells.

The code parses `acinus_id` from the `labels` column, drops rows with missing required values, and ignores acini with fewer than `min_cells` cells.

## Position binning

The column `ascini_position` is divided into `n_bins` position bins. The default is 30 bins.

If the observed position range is close to `-1` to `1`, the bin range is fixed to `-1` to `1`. Otherwise, the observed minimum and maximum are used.

Position bins are used to estimate how category frequencies change along acinar position.

## Raw metrics

### `fraction_explained`

This measures the fraction of category entropy explained by acinar position.

```text
fraction_explained = 1 - H(CAT | position_bin) / H(CAT)
```

Interpretation:

- `1.0`: category identity is fully explained by position bins.
- `0.0`: knowing position does not reduce category uncertainty.
- Higher values indicate stronger position-driven category structure.

This metric is also permutation-tested by shuffling category labels and recomputing the value. The resulting p-value is saved as `fraction_explained_p`.

### `fraction_unexplained_pct`

This is the complement of `fraction_explained`, expressed as a percentage.

```text
fraction_unexplained_pct = 100 * (1 - fraction_explained)
```

Interpretation:

- `0%`: all category structure is explained by position.
- `100%`: none of the category structure is explained by position.
- Higher values indicate more category structure remains after accounting for position.

This metric is not zero-centered. It is plotted on a 0 to 100 percent scale.

### `position_accuracy`

This measures how accurately category identity can be predicted using only position bin.

For each position bin, the most frequent category in that bin is treated as the position-only prediction.

Interpretation:

- `1.0`: position bins perfectly predict category identity.
- Lower values: position bins are less predictive.
- Higher values indicate stronger position-driven category structure.

### `residual_moransI_mean`

This measures residual spatial autocorrelation after accounting for the category frequencies expected from position bins.

Interpretation:

- Higher positive values indicate neighboring cells retain category similarity beyond what position explains.
- Values near zero indicate little residual spatial autocorrelation after accounting for position.

### `residual_morans_p`

This is a permutation p-value for residual Moran's I.

Category labels are permuted within position bins. This preserves broad position structure while testing whether additional residual spatial organization remains.

Interpretation:

- `p < 0.05`: residual spatial organization is detected after accounting for position.
- `p >= 0.05`: residual spatial organization is not detected at the 0.05 threshold.

### `adjacency_sig_pairs`

This counts category-pair neighbor relationships that are significant after position-conditioned permutation testing.

Interpretation:

- Higher values indicate more residual neighborhood enrichment or depletion beyond position.
- Lower values indicate fewer residual neighborhood effects beyond position.

## Zero-centered normalized metrics

Several raw metrics are converted into zero-centered decision scores. These normalized values are used in the normalized bar plots.

A normalized value of `0` means the metric is exactly at the chosen reference threshold. Positive values support the position-explained interpretation, called Model A. Negative values support residual structure beyond position, called Model B.

The shaded background follows the same interpretation:

- Above zero: more Model A-like; category structure is more position-explained.
- Below zero: more Model B-like; residual spatial/neighborhood structure remains after accounting for position.

### `norm_fraction_explained`

```text
norm_fraction_explained = (fraction_explained - 0.70) / 0.20
```

Scale interpretation:

- `0` means `fraction_explained = 0.70`.
- `+1` means `fraction_explained = 0.90`.
- `-1` means `fraction_explained = 0.50`.
- Positive values mean position explains more than the 0.70 reference.
- Negative values mean position explains less than the 0.70 reference.

### `norm_position_accuracy`

```text
norm_position_accuracy = (position_accuracy - 0.80) / 0.20
```

Scale interpretation:

- `0` means `position_accuracy = 0.80`.
- `+1` means `position_accuracy = 1.00`.
- `-1` means `position_accuracy = 0.60`.
- Positive values mean position-only prediction is stronger than the 0.80 reference.
- Negative values mean position-only prediction is weaker than the 0.80 reference.

### `norm_residual_morans_p`

```text
norm_residual_morans_p = log10(residual_morans_p / 0.05)
```

Scale interpretation:

- `0` means `residual_morans_p = 0.05`.
- `+1` means `residual_morans_p = 0.50`.
- `-1` means `residual_morans_p = 0.005`.
- Positive values mean residual Moran's I is not significant at p = 0.05, supporting Model A.
- Negative values mean residual Moran's I is significant, supporting Model B.

### `norm_adjacency_sig`

The expected number of significant category-pair relationships under a 5% false-positive rate is:

```text
E = 0.05 * Nc * (Nc + 1) / 2
```

where `Nc` is the number of category labels present in that acinus.

The normalized score is:

```text
norm_adjacency_sig = (E - observed_significant_pairs) / max(E, 1.0)
```

Scale interpretation:

- `0` means the observed number of significant residual adjacency pairs equals the expected number.
- Positive values mean fewer residual adjacency effects than expected, supporting Model A.
- Negative values mean more residual adjacency effects than expected, supporting Model B.
- Because the denominator is at least 1, a change of one significant pair changes the score by approximately one unit when `E < 1`.

## Output files

Main tables:

- `per_acinus_metrics_with_norm_and_fracP.csv`
- `per_acinus_metrics_enhanced.csv`
- `fraction_explained_per_acinus_perm.csv`

Main plots:

- `bar_norm_control_norm_adjacency_sig_points.png`
- `bar_norm_control_norm_fraction_explained_points.png`
- `bar_norm_control_norm_position_accuracy_points.png`
- `bar_norm_control_norm_residual_morans_p_points.png`
- `violin_control_fraction_unexplained_pct_points.png`

Use `per_acinus_metrics_enhanced.csv` for exact numerical interpretation. The figures are visual summaries of those per-acinus values.
"""
    (outdir_path / "README_metrics_and_normalization.md").write_text(readme, encoding="utf-8")

# ---------------------- Bootstrap CI utilities ----------------------

def bootstrap_ci_mean(vals: np.ndarray, B: int = 2000, alpha: float = 0.05, rng_seed: int = 2026):
    x = np.array(vals, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.nanmean(x[idx])
    lo, hi = np.nanpercentile(boots, [100*alpha/2, 100*(1 - alpha/2)])
    return float(np.nanmean(x)), float(lo), float(hi)

def group_bootstrap_summary(perA: pd.DataFrame, col: str, groups: List[str], B: int = 2000, alpha: float = 0.05):
    rows = []
    for g in groups:
        vals = perA.loc[perA["group"]==g, col].to_numpy(float)
        mean, lo, hi = bootstrap_ci_mean(vals, B=B, alpha=alpha)
        sem = (np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals)))) if np.sum(np.isfinite(vals))>1 else np.nan
        rows.append({"group": g, "n_acini": int(np.sum(np.isfinite(vals))), "mean": mean, "sem": sem, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)

# ---------------------- Zero-centered band helper ----------------------

def fill_zero_bands(ax, zero_line=0.0, above_color="#eeeeee", below_color="#f3e8ff", alpha=0.35):
    """Shade areas above/below zero on zero-centered plots."""
    if zero_line is None:
        return
    y0, y1 = ax.get_ylim()
    # Draw behind everything else
    ax.axhspan(zero_line, y1, facecolor=above_color, alpha=alpha, zorder=0)
    ax.axhspan(y0, zero_line, facecolor=below_color, alpha=alpha, zorder=0)

# ---------------------- Spine helpers ----------------------

def _parse_list(s: str):
    if not s:
        return []
    return [t.strip().lower() for t in re.split(r'[;,]', s) if t.strip()]

def apply_spines(ax, spines_off: str = "top,right", spines_on: str = ""):
    """Set visibility of axis spines (plot borders)."""
    all_sp = {"left","right","top","bottom"}
    off = set(_parse_list(spines_off))
    on  = set(_parse_list(spines_on))
    # start from all visible
    for sp in all_sp:
        if sp in ax.spines:
            ax.spines[sp].set_visible(True)
    # apply offs, then ons override
    for sp in off:
        if sp in ax.spines:
            ax.spines[sp].set_visible(False)
    for sp in on:
        if sp in ax.spines:
            ax.spines[sp].set_visible(True)

def new_axes(figsize, spines_off: str, spines_on: str, layout_mode: str = "fixed",
             left: float = 0.30, right: float = 0.98, top: float = 0.90, bottom: float = 0.14):
    fig, ax = plt.subplots(figsize=figsize)
    apply_spines(ax, spines_off=spines_off, spines_on=spines_on)
    if layout_mode == "fixed":
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    return fig, ax

# ---------------------- Orchestrator ----------------------

def run(csv: str = DEFAULT_CSV, n_bins: int = 30, k: int = 8, radius: float = None,
        adj_perm: int = 200, morans_perm: int = 200, frac_perm: int = 500, boot_B: int = 2000,
        min_cells: int = 5,
        with_points: bool = True,
        palette: str = "tab10",
        group_colors: str = "",
        group_labels: str = "",
        title_prefix: str = "",
        ylabel_overrides: str = "",
        point_size: float = 18.0,
        point_alpha: float = 0.65,
        point_jitter: float = 0.18,
        point_seed: int = 123,
        bar_alpha: float = 1.0,
        # Point colors/outline
        point_color: str = "",
        point_edgecolor: str = "white",
        point_edgewidth: float = 0.5,
        # Palettes
        control_palette: str = "Purples",
        violin_palette_groups: str = "",
        violin_alpha: float = 0.8,
        # Zero-band fills
        zero_fill_above: str =  "#f3e8ff",
        zero_fill_below: str = "#eeeeee",
        zero_fill_alpha: float = 0.35,
        # Widths
        bar_width: float = 0.6,
        violin_width: float = 0.6,
        # Figure sizes
        figw_control: float = 2.0, figh_control: float = 4.2,
        figw_groups: float = 7.5, figh_groups: float = 4.6,
        # Spines
        spines_off: str = "top,right",
        spines_on: str = "",
        # Layout
        layout_mode: str = "fixed",  # 'fixed' for consistent margins, 'tight' to auto-fit
        layout_left: float = 0.30,
        layout_right: float = 0.98,
        layout_top: float = 0.90,
        layout_bottom: float = 0.14,
        # Fonts
        font_size_all: Optional[float] = None,
        title_size: float = 12.0,
        axis_label_size: float = 11.0,
        tick_label_size: float = 10.0,
        annotation_size: float = 10.0):
    csv_path = Path(csv)
    outdir_path = csv_path.parent / csv_path.stem / "model_test" / "model_plots"
    ensure_dir(outdir_path)

    # ---- Layout helper ----
    def maybe_tight(fig):
        if layout_mode == "tight":
            fig.tight_layout()

    # ---- Fonts: single knob or individual sizes ----
    if font_size_all is not None:
        title_size = axis_label_size = tick_label_size = annotation_size = float(font_size_all)
    plt.rcParams["axes.titlesize"] = float(title_size)
    plt.rcParams["axes.labelsize"] = float(axis_label_size)
    plt.rcParams["xtick.labelsize"] = float(tick_label_size)
    plt.rcParams["ytick.labelsize"] = float(tick_label_size)

    # Load
    df = pd.read_csv(csv_path)
    required = {"labels","group","CAT","centroid-0","centroid-1","ascini_position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean & augment
    df["group"] = df["group"].astype(str).str.strip()
    df["centroid-0"] = pd.to_numeric(df["centroid-0"], errors="coerce")
    df["centroid-1"] = pd.to_numeric(df["centroid-1"], errors="coerce")
    df["ascini_position"] = pd.to_numeric(df["ascini_position"], errors="coerce")
    df["acinus_id"] = df["labels"].astype(str).map(parse_acinus_id)
    df["liver_id"] = df["acinus_id"].map(parse_liver_id)
    df["group_canon"] = [canonical_group_name(g, c) for g, c in zip(df["group"], df["CAT"])]

    # Pre-dropna snapshot for debugging
    df_before = df.copy()
    df = df.dropna(subset=["centroid-0","centroid-1","ascini_position","group","acinus_id","CAT","liver_id"]).copy()

    # Compute per-acinus metrics
    perA = compute_per_acinus_metrics(df, n_bins=n_bins, k=k, radius=radius,
                                      adj_perm=adj_perm, morans_perm=morans_perm, frac_perm=frac_perm,
                                      min_cells=min_cells)
    perA.to_csv(outdir_path / "per_acinus_metrics_with_norm_and_fracP.csv", index=False)

    # Early exit with debug bundle if empty
    if perA.empty:
        (df.groupby(["group_canon","acinus_id"]).size()
           .rename("n_cells")
           .reset_index()
           .to_csv(outdir_path / "DEBUG_acinus_sizes_after_dropna.csv", index=False))
        na_report = df_before[["centroid-0","centroid-1","ascini_position","group","acinus_id","CAT","liver_id"]].isna().sum()
        na_report.to_csv(outdir_path / "DEBUG_na_counts_before_dropna.csv")
        with open(outdir_path / "DEBUG_README.txt","w") as f:
            f.write(
                "per_acinus_metrics table is empty.\n"
                f"Possible reasons:\n"
                f"  • After dropna, no acinus had >= min_cells (min_cells={min_cells}). Try --min_cells 3.\n"
                f"  • acinus_id parsing failed; see DEBUG_acinus_sizes_after_dropna.csv for unexpected IDs.\n"
                f"  • 'ascini_position' missing/NA for many cells; see DEBUG_na_counts_before_dropna.csv.\n"
                "\nNo plots were generated to avoid KeyError. Re-run after addressing the above.\n"
            )
        print(f"[WARN] No acini passed filters. Wrote DEBUG files to: {outdir_path}")
        return

    # ---------- Derived metrics ----------
    # % not explained by position
    perA["fraction_unexplained"] = 1.0 - perA["fraction_explained"]
    perA["fraction_unexplained_pct"] = 100.0 * perA["fraction_unexplained"]

    # Enhanced table
    perA.to_csv(outdir_path / "per_acinus_metrics_enhanced.csv", index=False)

    # ---------- Plot label/color plumbing ----------
    label_map = _parse_kv_mapping(group_labels)
    groups = ["Control", "Fasted", "WD"]
    display_groups = [label_map.get(g, g) for g in groups]
    bar_colors = _group_colors(groups, palette=palette, group_color_map=group_colors, bar_alpha=bar_alpha)
    ylab_over = _parse_kv_mapping(ylabel_overrides)

    # Control-only color (bars + violins) from control_palette
    control_color_list = _parse_palette_spec(control_palette, 1)
    cc = control_color_list[0] if control_color_list else None
    if cc is not None:
        c_rgba = list(to_rgba(cc)); c_rgba[3] = bar_alpha; control_bar_color = tuple(c_rgba)
    else:
        control_bar_color = bar_colors[0] if bar_colors else None
    control_violin_colors = [control_bar_color]

    # Group violin colors
    if violin_palette_groups:
        _vcols = _parse_palette_spec(violin_palette_groups, len(groups))
        group_violin_colors = []
        for c in _vcols:
            if c is None:
                group_violin_colors.append(None)
            else:
                rgba = list(to_rgba(c)); rgba[3] = violin_alpha; group_violin_colors.append(tuple(rgba))
    else:
        group_violin_colors = bar_colors

    # Point face colors (default: match bar colors)
    if point_color:
        ctrl_point_cols = _parse_palette_spec(point_color, 1)
        control_point_colors = [ctrl_point_cols[0]]
        group_point_colors = _parse_palette_spec(point_color, len(groups))
    else:
        control_point_colors = [control_bar_color]
        group_point_colors = bar_colors

    # Figure sizes
    control_figsize = (figw_control, figh_control)
    group_figsize   = (figw_groups,   figh_groups)

    # convenience creator
    def make_axes(figsize):
        return new_axes(figsize, spines_off=spines_off, spines_on=spines_on,
                        layout_mode=layout_mode, left=layout_left, right=layout_right, top=layout_top, bottom=layout_bottom)

    # ---------- PLOTS: Control-only reduced output set ----------
    raw_metric_labels = {
        "fraction_unexplained_pct": ylab_over.get(
            "fraction_unexplained_pct",
            "Percent not explained by position (%)",
        ),
    }
    norm_metric_labels = {
        "norm_adjacency_sig": ylab_over.get(
            "norm_adjacency_sig",
            "Zero-centered score: adjacency sig-pairs\n(0 = expected; + = A, - = B)",
        ),
        "norm_fraction_explained": ylab_over.get(
            "norm_fraction_explained",
            "Zero-centered score: fraction explained\n(0 = 0.70; + = A, - = B)",
        ),
        "norm_position_accuracy": ylab_over.get(
            "norm_position_accuracy",
            "Zero-centered score: position-only accuracy\n(0 = 0.80; + = A, - = B)",
        ),
        "norm_residual_morans_p": ylab_over.get(
            "norm_residual_morans_p",
            "Zero-centered score: residual Moran's p\n(0 = p=0.05; + = A, - = B)",
        ),
    }

    control_df = perA[perA["group"] == "Control"].copy()
    if control_df.empty:
        print("[WARNING] No Control acini found. Control-only plots will contain no finite values.")

    # 1) Control-only normalized bars with points for requested metrics.
    requested_norm_metrics = [
        "norm_adjacency_sig",
        "norm_fraction_explained",
        "norm_position_accuracy",
        "norm_residual_morans_p",
    ]
    for key in requested_norm_metrics:
        ylabel = norm_metric_labels[key]
        vals = control_df[key].to_numpy(float) if key in control_df.columns else np.array([], dtype=float)
        values_by_group = {"Control": vals}
        title = (title_prefix + f"{key} (zero-centered) — Control only") if title_prefix else f"{key} (zero-centered) — Control only"

        fig, ax = make_axes(control_figsize)
        bar_with_sem(
            ax,
            ["Control"],
            values_by_group,
            title,
            ylabel,
            zero_line=0.0,
            symmetric_about_zero=True,
            colors=[control_bar_color],
            ticklabels=[display_groups[0]],
            bar_width=bar_width,
        )
        if with_points:
            overlay_points_on_bars(
                ax,
                ["Control"],
                values_by_group,
                colors=control_point_colors,
                zero_line=0.0,
                symmetric_about_zero=True,
                jitter=point_jitter,
                alpha=point_alpha,
                size=point_size,
                seed=point_seed,
                edgecolor=point_edgecolor,
                edgewidth=point_edgewidth,
            )
        fill_zero_bands(ax, 0.0, zero_fill_above, zero_fill_below, zero_fill_alpha)
        maybe_tight(fig)
        save_publication_figure(fig, outdir_path / f"bar_norm_control_{key}_points.png", dpi=300)
        plt.close(fig)

    # 2) Control-only violin with points for percent not explained by position.
    key = "fraction_unexplained_pct"
    vals = control_df[key].to_numpy(float) if key in control_df.columns else np.array([], dtype=float)
    values_by_group = {"Control": vals}
    title = (title_prefix + f"{key} — Control only (distribution across acini)") if title_prefix else f"{key} — Control only (distribution across acini)"
    fig, ax = make_axes(control_figsize)
    used_violin = safe_distribution_plot(
        ax,
        ["Control"],
        values_by_group,
        title,
        raw_metric_labels[key],
        ticklabels=[display_groups[0]],
        violin_colors=control_violin_colors,
        violin_alpha=violin_alpha,
        violin_width=violin_width,
    )
    ax.set_ylim(0, 100)
    if with_points and used_violin:
        overlay_points_on_bars(
            ax,
            ["Control"],
            values_by_group,
            colors=control_point_colors,
            jitter=point_jitter,
            alpha=point_alpha,
            size=point_size,
            seed=point_seed,
            edgecolor=point_edgecolor,
            edgewidth=point_edgewidth,
        )
        ax.set_ylim(0, 100)
    maybe_tight(fig)
    save_publication_figure(fig, outdir_path / "violin_control_fraction_unexplained_pct_points.png", dpi=300)
    plt.close(fig)

    # ---------- Save permutation & bootstrap summaries ----------
    perA[[
        "group", "acinus_id", "fraction_explained", "fraction_explained_p",
        "fraction_explained_null_mean", "fraction_explained_null_sd",
    ]].to_csv(outdir_path / "fraction_explained_per_acinus_perm.csv", index=False)

    fe_summary = group_bootstrap_summary(perA, "fraction_explained", groups, B=boot_B, alpha=0.05)
    nfe_summary = group_bootstrap_summary(perA, "norm_fraction_explained", groups, B=boot_B, alpha=0.05)
    fe_summary.to_csv(outdir_path / "fraction_explained_group_bootstrap_summary.csv", index=False)
    nfe_summary.to_csv(outdir_path / "norm_fraction_explained_group_bootstrap_summary.csv", index=False)

    # README explaining metrics and zero-centered scales.
    write_metrics_readme(outdir_path)

    print(f"[OK] Outputs written to: {outdir_path}")
    if with_points:
        print("[INFO] Points are ENABLED (bars + violins). Use --no_points to disable.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to input CSV")
    ap.add_argument("--n_bins", type=int, default=30, help="Bins along ascini_position")
    ap.add_argument("--k", type=int, default=8, help="k for kNN graph")
    ap.add_argument("--radius", type=float, default=None, help="Optional neighbor radius")
    ap.add_argument("--adj_perm", type=int, default=200, help="Permutations for per-acinus adjacency null")
    ap.add_argument("--morans_perm", type=int, default=200, help="Permutations for per-acinus residual Moran's p")
    ap.add_argument("--frac_perm", type=int, default=500, help="Permutations for per-acinus fraction_explained p-value")
    ap.add_argument("--boot_B", type=int, default=2000, help="Bootstrap replicates for group CIs")
    ap.add_argument("--min_cells", type=int, default=5, help="Minimum cells per acinus to compute metrics")

    # Styling/points args (default points ON)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--with_points", dest="with_points", action="store_true",
                       help="Enable overlaid points on bars & violins (default: enabled)")
    group.add_argument("--no_points", dest="with_points", action="store_false",
                       help="Disable overlaid points on bars & violins (default: enabled)")
    ap.set_defaults(with_points=True)

    ap.add_argument("--palette", default="tab10", help="Matplotlib colormap name OR comma/semicolon-separated colors")
    ap.add_argument("--group_colors", default="", help="Explicit mapping 'Control:#4C78A8,Fasted:#F58518,WD:#54A24B'")
    ap.add_argument("--group_labels", default="", help="Tick-label mapping 'Control:CTL,Fasted:FAST,WD:WD'")
    ap.add_argument("--title_prefix", default="", help="Prefix string added to the start of every plot title")
    ap.add_argument("--ylabel_overrides", default="", help="Comma/semicolon list 'metric:new y label'")
    ap.add_argument("--point_size", type=float, default=19.0, help="Point size for overlaid dots")
    ap.add_argument("--point_alpha", type=float, default=0.9, help="Alpha for overlaid dots")
    ap.add_argument("--point_jitter", type=float, default=0.18, help="Half-width of horizontal jitter window")
    ap.add_argument("--point_seed", type=int, default=123, help="Seed to jitter reproducibly")
    ap.add_argument("--bar_alpha", type=float, default=1.0, help="Alpha for bar colors")

    # Point color & outline
    ap.add_argument("--point_color", default="black", help="Color(s) for points; comma/semicolon list, colormap name, named color, or hex. Default matches bar colors")
    ap.add_argument("--point_edgecolor", default="white", help="Outline (edge) color for points")
    ap.add_argument("--point_edgewidth", type=float, default=0.5, help="Outline (edge) width for points")

    # Bar/violin color palettes
    ap.add_argument("--control_palette", default="Purples", help="Colormap or color for Control-only plots (bars + violins)")
    ap.add_argument("--violin_palette_groups", default="", help="Colormap/colors for group violins; default matches bar colors")
    ap.add_argument("--violin_alpha", type=float, default=0.55, help="Alpha for violin bodies")

    # Zero-centered background bands
    ap.add_argument("--zero_fill_above", default="#f3e8ff", help="Fill color above zero for zero-centered plots (light gray)")
    ap.add_argument("--zero_fill_below", default="#eeeeee", help="Fill color below zero for zero-centered plots (light purple)")
    ap.add_argument("--zero_fill_alpha", type=float, default=0.85, help="Alpha for zero-centered fill bands")

    # Widths
    ap.add_argument("--bar_width", type=float, default=0.45, help="Matplotlib bar width (0–1)")
    ap.add_argument("--violin_width", type=float, default=0.55, help="Width of violins (0–1)")

    # Figure sizes
    ap.add_argument("--figw_control", type=float, default=1.6, help="Figure width (inches) for Control-only plots")
    ap.add_argument("--figh_control", type=float, default=4.2, help="Figure height (inches) for Control-only plots")
    ap.add_argument("--figw_groups",  type=float, default=7.5, help="Figure width (inches) for group plots")
    ap.add_argument("--figh_groups",  type=float, default=4.6, help="Figure height (inches) for group plots")

    # Spines (plot borders)
    ap.add_argument("--spines_off", default="top,right", help="Comma/semicolon list of spines to hide: any of left,right,top,bottom (default: top,right)")
    ap.add_argument("--spines_on",  default="", help="Comma/semicolon list of spines to force visible (applied after --spines_off)")

    # Layout
    ap.add_argument("--layout_mode", choices=["fixed","tight"], default="fixed",
                    help="Use 'fixed' for consistent margins (equal axes width), 'tight' to auto-fit label text")
    ap.add_argument("--layout_left", type=float, default=0.30, help="When layout_mode=fixed: left margin frac (0-1)")
    ap.add_argument("--layout_right", type=float, default=0.98, help="When layout_mode=fixed: right margin frac (0-1)")
    ap.add_argument("--layout_top", type=float, default=0.90, help="When layout_mode=fixed: top margin frac (0-1)")
    ap.add_argument("--layout_bottom", type=float, default=0.14, help="When layout_mode=fixed: bottom margin frac (0-1)")

    # Font sizes
    ap.add_argument("--font_size_all", type=float, default=None,
                    help="Set a single size for titles, axis labels, tick labels, and annotations")
    ap.add_argument("--title_size", type=float, default=3.0, help="Font size for plot titles")
    ap.add_argument("--axis_label_size", type=float, default=11.0, help="Font size for axis labels")
    ap.add_argument("--tick_label_size", type=float, default=15.0, help="Font size for tick labels")
    ap.add_argument("--annotation_size", type=float, default=10.0, help="Font size for on-plot annotations (e.g., p-values)")

    args = ap.parse_args()
    run(csv=args.csv, n_bins=args.n_bins, k=args.k, radius=args.radius,
        adj_perm=args.adj_perm, morans_perm=args.morans_perm, frac_perm=args.frac_perm, boot_B=args.boot_B,
        min_cells=args.min_cells,
        with_points=args.with_points, palette=args.palette, group_colors=args.group_colors,
        group_labels=args.group_labels, title_prefix=args.title_prefix, ylabel_overrides=args.ylabel_overrides,
        point_size=args.point_size, point_alpha=args.point_alpha, point_jitter=args.point_jitter,
        point_seed=args.point_seed, bar_alpha=args.bar_alpha,
        point_color=args.point_color, point_edgecolor=args.point_edgecolor, point_edgewidth=args.point_edgewidth,
        control_palette=args.control_palette, violin_palette_groups=args.violin_palette_groups, violin_alpha=args.violin_alpha,
        zero_fill_above=args.zero_fill_above, zero_fill_below=args.zero_fill_below, zero_fill_alpha=args.zero_fill_alpha,
        bar_width=args.bar_width, violin_width=args.violin_width,
        figw_control=args.figw_control, figh_control=args.figh_control,
        figw_groups=args.figw_groups,   figh_groups=args.figh_groups,
        spines_off=args.spines_off, spines_on=args.spines_on,
        layout_mode=args.layout_mode, layout_left=args.layout_left, layout_right=args.layout_right, layout_top=args.layout_top, layout_bottom=args.layout_bottom,
        font_size_all=args.font_size_all, title_size=args.title_size,
        axis_label_size=args.axis_label_size, tick_label_size=args.tick_label_size,
        annotation_size=args.annotation_size)

if __name__ == "__main__":
    main()
