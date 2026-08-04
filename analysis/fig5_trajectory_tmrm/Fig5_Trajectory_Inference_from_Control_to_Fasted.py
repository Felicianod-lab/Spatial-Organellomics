#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control (CNT)-to- Fasted (STV) best-pair and pseudotime pipeline.

  1. CNT->STV first-hit analysis for best pairs.
  2. The CNT-only stacked bar plot: DIRECT + via CNTx, with WD/OTHER excluded.
  3. Hub/anchor selection from the strongest CNT->STV pair.
  4. Pseudotime computation with the same graph-distance strategy.

Important workflow note:
  - The best-pair plot is computed with an initial pseudotime root, default CNT CAT=3
    (H4).
  - After the best pair is selected, the final pseudotime CSV is recomputed using the
    selected hub CNT as the root by default. This avoids changing the best-pair plot
    while still making the saved pseudotime informed by the hub/anchor decision.

Default outputs:
  best_cnt_to_stv_pairs_ranked.csv
  best_cnt_to_stv_lastcnt_breakdown.csv
  best_pairs_topk_plot_data.csv
  best_pairs_topk_bars.png
  hub_choice.csv
  pseudotime.csv
  root_sensitivity_summary.csv
  root_sensitivity_all_pairs.csv
  root_sensitivity_consensus.csv
  absorption_to_finals.csv
  first_stv_hit_probs.csv
  two_stage_unconditional_paths.csv
  route_metrics.csv
  route_bars.png
  cells_with_pca_and_pseudotime.csv
  pseudotime_cnt_stv_only.csv
  cells_with_pseudotime_H4_FH_only.csv  (H4/FH filtered cells; pseudotime_full rescaled 0-1)
  pca_projection_with_final_pseudotime_filtered.png  (filtered cells rescaled 0-1)
  pca_signed_control_to_fh_pseudotime_summary.csv
  pca_projection_with_signed_control_to_fh_pseudotime.png
  pca_pseudotime_filter_summary.csv
 

"""

import argparse
import ast
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ARROW = "\u2192"
EN_DASH = "\u2014"

FH_NAME = {"5": "FH1", "6": "FH2", "7": "FH3", "8": "FH4"}
CNT_FRIENDLY = {"0": "H1", "1": "H2", "2": "H3", "3": "H4", "4": "H5"}


def friendly_cnt(cat: object) -> str:
    """Return H1/H2/... names for CNT category IDs."""
    return CNT_FRIENDLY.get(str(cat), f"CNT{cat}")


def friendly_stv(cat: object) -> str:
    """Return FH1/FH2/... names for STV category IDs."""
    return FH_NAME.get(str(cat), f"STV{cat}")


def friendly_route(route: object) -> str:
    """Convert a numeric route like 3>5>6 to H4→FH1→FH2."""
    parts = [part.strip() for part in str(route).split(">") if part.strip()]
    if not parts:
        return str(route)

    labels = [friendly_cnt(parts[0])]
    labels.extend(friendly_stv(part) for part in parts[1:])
    return ARROW.join(labels)


def _normalize_label(label: object) -> str:
    """Normalize labels that may arrive as strings or as stringified lists."""
    s = str(label).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                s = str(val[0])
        except Exception:
            s = s.strip("[]'\" ").split(",")[0].strip(" '\"")
    return s.strip(" '\"")


def parse_group(label: object) -> str:
    """Classify labels as CNT, STV, WD, or OTHER from their prefix."""
    s = _normalize_label(label)
    match = re.search(r"(CNT|STV|WD)_", s, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "OTHER"


def find_feature_columns(df: pd.DataFrame, label_col: str, extra_drop: Optional[List[str]]) -> List[str]:
    """Select numeric feature columns after removing known metadata columns."""
    drop_cols = {label_col, "group", "CAT"}
    drop_cols.update(extra_drop or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after excluding labels/metadata.")
    return feature_cols


def run_pca(features: pd.DataFrame, n_components: int) -> Tuple[PCA, np.ndarray, StandardScaler]:
    """Standardize features and compute PCA scores."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(features.values)

    max_components = min(x_scaled.shape[0], x_scaled.shape[1])
    n_keep = min(max(3, int(n_components)), max_components)
    pca = PCA(n_components=n_keep)
    scores = pca.fit_transform(x_scaled)
    return pca, scores, scaler


def build_knn_graph(embedding: np.ndarray, k: int = 15, metric: str = "euclidean") -> csr_matrix:
    """Build a symmetric affinity-weighted kNN graph."""
    n = embedding.shape[0]
    if n <= 1:
        return csr_matrix((n, n))

    k = max(1, min(int(k), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    distances = distances[:, 1:]
    indices = indices[:, 1:]
    local_scale = distances[:, -1] + 1e-8

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for i in range(n):
        for d, j in zip(distances[i], indices[i]):
            scale = local_scale[i] * local_scale[j]
            weight = float(np.exp(-(d * d) / (2.0 * scale)))
            rows.append(i)
            cols.append(int(j))
            vals.append(weight)
            rows.append(int(j))
            cols.append(i)
            vals.append(weight)

    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def connected_component_mask(graph: csr_matrix, root_indices: np.ndarray) -> np.ndarray:
    """Keep the component containing the largest number of root cells."""
    binary = graph.copy()
    binary.data[:] = 1.0
    n_components, labels = connected_components(binary, directed=False)

    if root_indices.size == 0:
        component_sizes = np.bincount(labels)
        component_id = int(np.argmax(component_sizes))
    else:
        root_components = labels[root_indices]
        counts = np.bincount(root_components, minlength=n_components)
        component_id = int(np.argmax(counts))

    return labels == component_id


def _lengths_from_affinity(graph: csr_matrix) -> csr_matrix:
    """Convert graph affinities to edge lengths with the original 1/weight rule."""
    eps = 1e-12
    rows, cols = graph.nonzero()
    lengths = 1.0 / (graph.data + eps)
    return csr_matrix((lengths, (rows, cols)), shape=graph.shape)


def _normalize_pseudotime(dist_min: np.ndarray) -> np.ndarray:
    """Normalize finite graph distances to pseudotime in [0, 1]."""
    is_finite = np.isfinite(dist_min)
    is_root_like = (dist_min <= 1e-12) & is_finite
    is_nonzero_finite = is_finite & (~is_root_like)

    pt = np.zeros_like(dist_min, dtype=float)
    if np.any(is_nonzero_finite):
        vals = dist_min[is_nonzero_finite]
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if hi > lo:
            pt[is_nonzero_finite] = (vals - lo) / (hi - lo)
        else:
            pt[is_nonzero_finite] = 0.5

    pt[~is_finite] = 1.0
    return np.clip(pt, 0.0, 1.0)


def _rescale_values_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Rescale finite values to [0, 1], preserving shape.

    This is used for display-only filtered PCA plots so the remaining
    non-outlier cells occupy the full colorbar range.
    """
    out = np.full_like(np.asarray(values, dtype=float), np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    vals = np.asarray(values, dtype=float)[finite]
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if hi > lo:
        out[finite] = (vals - lo) / (hi - lo)
    else:
        out[finite] = 0.5
    return np.clip(out, 0.0, 1.0)


def _compute_signed_control_to_fh_display_pseudotime(
    df_sub: pd.DataFrame,
    root_distance_sub: np.ndarray,
    filtered_mask_sub: np.ndarray,
    root_cnt: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Create an Option-B signed display pseudotime for filtered CNT/STV cells.

    This is a visualization-only transformation. It does not change the raw
    graph pseudotime, the forward Markov chain, first-hit probabilities, route
    probabilities, hub selection, or any bar-plot output.

    Option B definition on the filtered display cells:
      - non-root CNT categories are placed on the negative side and scaled so
        the farthest remaining non-root CNT cell is -1.
      - root CNT cells are fixed at 0.
      - STV/FH cells are placed on the positive side and scaled so the farthest
        remaining STV/FH cell is +1.
      - the signed scale [-1, +1] is then converted to [0, 1], so
        0.0 = farthest non-root control side, 0.5 = root/H4, and
        1.0 = farthest FH/STV side.
    """
    # Use raw graph distance here, i.e. before the normal 0-1 pseudotime
    # normalization. This keeps the signed display transform faithful to the
    # requested sequence: compute distance -> filter -> assign control side
    # negative / FH side positive -> normalize the signed values to 0-1.
    pt = np.asarray(root_distance_sub, dtype=float)
    filtered = np.asarray(filtered_mask_sub, dtype=bool) & np.isfinite(pt)

    signed_minus1_to_1 = np.full(df_sub.shape[0], np.nan, dtype=float)
    signed_0_to_1 = np.full(df_sub.shape[0], np.nan, dtype=float)

    group = df_sub["group"].astype(str).str.upper().to_numpy()
    cat = df_sub["CAT"].astype(str).to_numpy()
    root_cnt = str(root_cnt)

    root_mask = filtered & (group == "CNT") & (cat == root_cnt)
    control_side_mask = filtered & (group == "CNT") & (cat != root_cnt)
    fh_side_mask = filtered & (group == "STV")

    # Balanced side-specific scaling. The negative and positive sides are
    # intentionally scaled independently so the root is centered at 0.5 after
    # conversion to [0, 1].
    control_vals = pt[control_side_mask]
    fh_vals = pt[fh_side_mask]

    control_max = float(np.nanmax(control_vals)) if control_vals.size else np.nan
    fh_max = float(np.nanmax(fh_vals)) if fh_vals.size else np.nan

    signed_minus1_to_1[root_mask] = 0.0

    if control_vals.size:
        if np.isfinite(control_max) and control_max > 0:
            signed_minus1_to_1[control_side_mask] = -np.clip(pt[control_side_mask] / control_max, 0.0, 1.0)
        else:
            signed_minus1_to_1[control_side_mask] = 0.0

    if fh_vals.size:
        if np.isfinite(fh_max) and fh_max > 0:
            signed_minus1_to_1[fh_side_mask] = np.clip(pt[fh_side_mask] / fh_max, 0.0, 1.0)
        else:
            signed_minus1_to_1[fh_side_mask] = 0.0

    valid = np.isfinite(signed_minus1_to_1)
    signed_0_to_1[valid] = (signed_minus1_to_1[valid] + 1.0) / 2.0
    signed_0_to_1[valid] = np.clip(signed_0_to_1[valid], 0.0, 1.0)

    summary = {
        "signed_display_definition": (
            "Option B balanced: non-root CNT scaled -1..0, root CNT fixed at 0, "
            "STV scaled 0..1, then converted to 0..1"
        ),
        "root_cnt": root_cnt,
        "root_label": friendly_cnt(root_cnt),
        "n_filtered_cells_available": int(filtered.sum()),
        "n_signed_cells": int(valid.sum()),
        "n_control_side_nonroot_cnt": int(control_side_mask.sum()),
        "n_root_cnt": int(root_mask.sum()),
        "n_fh_side_stv": int(fh_side_mask.sum()),
        "control_side_raw_graph_distance_max_used_for_scaling": control_max,
        "fh_side_raw_graph_distance_max_used_for_scaling": fh_max,
        "signed_minus1_to_1_min": float(np.nanmin(signed_minus1_to_1)) if valid.any() else np.nan,
        "signed_minus1_to_1_median": float(np.nanmedian(signed_minus1_to_1)) if valid.any() else np.nan,
        "signed_minus1_to_1_max": float(np.nanmax(signed_minus1_to_1)) if valid.any() else np.nan,
        "signed_display_pseudotime_0_1_min": float(np.nanmin(signed_0_to_1)) if valid.any() else np.nan,
        "signed_display_pseudotime_0_1_median": float(np.nanmedian(signed_0_to_1)) if valid.any() else np.nan,
        "signed_display_pseudotime_0_1_max": float(np.nanmax(signed_0_to_1)) if valid.any() else np.nan,
    }

    return signed_minus1_to_1, signed_0_to_1, summary


def compute_pseudotime(graph: csr_matrix, root_indices: np.ndarray) -> np.ndarray:
    """Compute pseudotime by Dijkstra distance from root cells on graph lengths."""
    mask = connected_component_mask(graph, root_indices)
    idx_map = np.where(mask)[0]

    lookup = -np.ones(graph.shape[0], dtype=int)
    lookup[idx_map] = np.arange(len(idx_map))

    graph_sub = graph[mask][:, mask]
    roots_sub = lookup[root_indices[(root_indices >= 0) & (root_indices < graph.shape[0])]]
    roots_sub = roots_sub[roots_sub >= 0]
    if roots_sub.size == 0:
        roots_sub = np.array([0], dtype=int)

    lengths = _lengths_from_affinity(graph_sub)
    dist = dijkstra(lengths, directed=False, indices=roots_sub)
    dist_min = np.min(dist, axis=0)

    pt_sub = _normalize_pseudotime(dist_min)
    pt = np.ones(graph.shape[0], dtype=float)
    pt[mask] = pt_sub
    return pt


def compute_graph_distance_from_roots(graph: csr_matrix, root_indices: np.ndarray) -> np.ndarray:
    """Compute raw Dijkstra graph distance from root cells before normalization."""
    mask = connected_component_mask(graph, root_indices)
    idx_map = np.where(mask)[0]

    lookup = -np.ones(graph.shape[0], dtype=int)
    lookup[idx_map] = np.arange(len(idx_map))

    graph_sub = graph[mask][:, mask]
    roots_sub = lookup[root_indices[(root_indices >= 0) & (root_indices < graph.shape[0])]]
    roots_sub = roots_sub[roots_sub >= 0]
    if roots_sub.size == 0:
        roots_sub = np.array([0], dtype=int)

    lengths = _lengths_from_affinity(graph_sub)
    dist = dijkstra(lengths, directed=False, indices=roots_sub)
    dist_min = np.min(dist, axis=0)

    out = np.full(graph.shape[0], np.inf, dtype=float)
    out[mask] = dist_min
    return out


def build_forward_markov(graph: csr_matrix, pseudotime: np.ndarray, tol: float = 1e-3) -> csr_matrix:
    """Orient the graph forward in pseudotime and row-normalize to a Markov chain."""
    graph = graph.tocsr().copy()
    rows, cols = graph.nonzero()

    keep = pseudotime[cols] + tol >= pseudotime[rows]
    data = graph.data.copy()
    data[~keep] = 0.0

    p = csr_matrix((data, (rows, cols)), shape=graph.shape)
    p.eliminate_zeros()

    row_sums = np.array(p.sum(axis=1)).ravel()
    dead = row_sums <= 0
    if np.any(dead):
        p = p.tolil()
        for i in np.where(dead)[0]:
            p[i, i] = 1.0
        p = p.tocsr()
        row_sums = np.array(p.sum(axis=1)).ravel()

    inv = 1.0 / row_sums
    d_inv = csr_matrix((inv, (np.arange(p.shape[0]), np.arange(p.shape[0]))), shape=p.shape)
    return d_inv @ p


def _reachable_from(p: csr_matrix, starts: np.ndarray) -> np.ndarray:
    """Nodes reachable from start nodes in a directed Markov graph."""
    p = p.tocsr()
    n = p.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = list(np.asarray(starts, dtype=int))

    if not stack:
        return seen

    seen[stack] = True
    indptr, indices = p.indptr, p.indices
    while stack:
        i = stack.pop()
        neighbors = indices[indptr[i] : indptr[i + 1]]
        neighbors = neighbors[~seen[neighbors]]
        if neighbors.size:
            seen[neighbors] = True
            stack.extend(neighbors.tolist())
    return seen


def _can_reach(p: csr_matrix, targets: np.ndarray) -> np.ndarray:
    """Nodes that can reach target nodes in a directed Markov graph."""
    pt = p.transpose().tocsr()
    n = p.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = list(np.asarray(targets, dtype=int))

    if not stack:
        return seen

    seen[stack] = True
    indptr, indices = pt.indptr, pt.indices
    while stack:
        j = stack.pop()
        preds = indices[indptr[j] : indptr[j + 1]]
        preds = preds[~seen[preds]]
        if preds.size:
            seen[preds] = True
            stack.extend(preds.tolist())
    return seen


def augment_absorbing_with_sinks(
    p: csr_matrix,
    absorbing_idx: np.ndarray,
    start_support: Optional[np.ndarray],
) -> np.ndarray:
    """Add nodes that cannot reach target absorbers as sink absorbers.

    The probabilities from the requested start distribution are unchanged by adding
    unreachable sink nodes, but doing so prevents singular linear systems when a
    different connected component contains a closed class unrelated to the current
    start cells.
    """
    n = p.shape[0]
    absorbing_idx = np.unique(np.asarray(absorbing_idx, dtype=int))

    can_reach_absorbing = _can_reach(p, absorbing_idx)
    sink_idx = np.where(~can_reach_absorbing)[0]
    return np.unique(np.concatenate([absorbing_idx, sink_idx]))


def absorbing_blocks(p: csr_matrix, absorbing_idx: np.ndarray):
    """Return Q/R blocks for an absorbing Markov chain."""
    n = p.shape[0]
    absorbing_mask = np.zeros(n, dtype=bool)
    absorbing_mask[absorbing_idx] = True
    transient_mask = ~absorbing_mask

    order = np.concatenate([np.where(transient_mask)[0], np.where(absorbing_mask)[0]])
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    p_perm = p[order][:, order]
    t = int(transient_mask.sum())
    q = p_perm[:t, :t]
    r = p_perm[:t, t:]
    return q, r, order, inv_order, transient_mask, absorbing_mask


def compute_firsthit_and_lastcnt(
    p_fwd: csr_matrix,
    df: pd.DataFrame,
    start_idx: np.ndarray,
    stv_map: Dict[str, np.ndarray],
    cnt_cats: List[str],
    exclude_non_cnt_in_breakdown: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute CNT->STV first-hit probability and last-CNT contribution breakdown.

    The probability p_abs[stv_cat] is the first-hit probability for that STV class.
    The lastcnt dict stores only CNT last-hop contributions by default, which gives
    the DIRECT + via CNT stack used in the bar plot.
    """
    n = df.shape[0]
    finals_list = [idx for idx in stv_map.values() if len(idx) > 0]
    finals_all = np.unique(np.concatenate(finals_list)) if finals_list else np.array([], dtype=int)

    absorbing_nodes = augment_absorbing_with_sinks(p_fwd, finals_all, start_support=start_idx)
    q, r, order, _, transient_mask, _ = absorbing_blocks(p_fwd, absorbing_nodes)
    t = int(transient_mask.sum())

    p_abs = {str(s): 0.0 for s in stv_map.keys()}
    lastcnt = {str(s): {} for s in stv_map.keys()}
    if t == 0 or start_idx.size == 0:
        return p_abs, lastcnt

    start_dist = np.zeros(n, dtype=float)
    start_dist[start_idx] = 1.0 / len(start_idx)
    start_perm = start_dist[order]
    start_transient = start_perm[:t]

    # Same occupancy solve as the original code: v = solve((I - Q).T, sT).
    lu_t = splu((sparse_eye(t, format="csr") - q).transpose().tocsc())
    occupancy = lu_t.solve(start_transient)

    abs_nodes_perm = order[t:]
    abs_groups = df["group"].to_numpy()[abs_nodes_perm]
    abs_cats = df["CAT"].astype(str).to_numpy()[abs_nodes_perm]
    trans_nodes = order[:t]
    trans_groups = df["group"].to_numpy()[trans_nodes]
    trans_cats = df["CAT"].astype(str).to_numpy()[trans_nodes]

    for stv_cat in stv_map.keys():
        stv_cat = str(stv_cat)
        cols = np.where((abs_groups == "STV") & (abs_cats == stv_cat))[0]
        if cols.size == 0:
            continue

        row_sums_to_stv = np.array(r[:, cols].sum(axis=1)).ravel()
        contrib_per_transient = occupancy * row_sums_to_stv
        p_abs[stv_cat] = float(contrib_per_transient.sum())

        contrib_by_cnt: Dict[str, float] = {}
        for cnt_cat in cnt_cats:
            cnt_cat = str(cnt_cat)
            mask = (trans_groups == "CNT") & (trans_cats == cnt_cat)
            value = float(contrib_per_transient[mask].sum())
            if value > 0:
                contrib_by_cnt[cnt_cat] = value

        if not exclude_non_cnt_in_breakdown:
            other_mask = trans_groups != "CNT"
            other_value = float(contrib_per_transient[other_mask].sum())
            if other_value > 0:
                contrib_by_cnt["OTHER"] = other_value

        lastcnt[stv_cat] = contrib_by_cnt

    return p_abs, lastcnt


def choose_cnt_hub_from_top_pair(cnt2stv_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], float]:
    """Select hub CNT and anchor STV from the highest absolute first-hit pair."""
    if cnt2stv_df.empty or float(cnt2stv_df["prob"].max()) <= 0:
        return None, None, 0.0
    row = cnt2stv_df.loc[int(cnt2stv_df["prob"].idxmax())]
    return str(row["start_cnt"]), str(row["stv_cat"]), float(row["prob"])


def build_index_maps(
    df: pd.DataFrame,
    cnt_cats: List[str],
    stv_cats: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return node indices for each CNT and STV category."""
    cnt_map = {
        str(c): np.where((df["group"] == "CNT") & (df["CAT"] == str(c)))[0]
        for c in cnt_cats
    }
    stv_map = {
        str(s): np.where((df["group"] == "STV") & (df["CAT"] == str(s)))[0]
        for s in stv_cats
    }
    return cnt_map, stv_map


def run_best_pair_analysis(
    p_fwd: csr_matrix,
    df: pd.DataFrame,
    cnt_map: Dict[str, np.ndarray],
    stv_map: Dict[str, np.ndarray],
    cnt_cats: List[str],
    stv_cats: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ranked CNT->STV first-hit probabilities and CNT-only breakdowns."""
    rows: List[Dict[str, object]] = []
    breakdown_rows: List[Dict[str, object]] = []

    for cnt_cat in cnt_cats:
        cnt_cat = str(cnt_cat)
        start_idx = cnt_map.get(cnt_cat, np.array([], dtype=int))

        if start_idx.size == 0:
            for stv_cat in stv_cats:
                rows.append({"start_cnt": cnt_cat, "stv_cat": str(stv_cat), "prob": 0.0})
            continue

        p_abs, lastcnt = compute_firsthit_and_lastcnt(
            p_fwd=p_fwd,
            df=df,
            start_idx=start_idx,
            stv_map=stv_map,
            cnt_cats=cnt_cats,
            exclude_non_cnt_in_breakdown=True,
        )

        for stv_cat in stv_cats:
            stv_cat = str(stv_cat)
            rows.append({"start_cnt": cnt_cat, "stv_cat": stv_cat, "prob": float(p_abs.get(stv_cat, 0.0))})
            for last_cnt, contrib in (lastcnt.get(stv_cat, {}) or {}).items():
                breakdown_rows.append(
                    {
                        "start_cnt": cnt_cat,
                        "stv_cat": stv_cat,
                        "last_cnt": str(last_cnt),
                        "last_cnt_label": friendly_cnt(last_cnt),
                        "contrib": float(contrib),
                    }
                )

    cnt2stv = pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)
    cnt2stv["start_cnt_label"] = cnt2stv["start_cnt"].map(friendly_cnt)
    cnt2stv["stv_label"] = cnt2stv["stv_cat"].map(friendly_stv)
    cnt2stv["pair_label"] = cnt2stv["start_cnt_label"] + ARROW + cnt2stv["stv_label"]

    breakdown_df = pd.DataFrame(breakdown_rows)
    if breakdown_df.empty:
        breakdown_df = pd.DataFrame(columns=["start_cnt", "stv_cat", "last_cnt", "last_cnt_label", "contrib"])

    return cnt2stv, breakdown_df


def build_plot_data(
    cnt2stv: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    cnt_cats: List[str],
    topk: int,
    force_include_stv: Optional[str],
    rank_by: str = "prob",
) -> pd.DataFrame:
    """Prepare the exact stack columns used by the best-pairs bar plot."""
    segment_names = ["DIRECT"] + [f"via {friendly_cnt(c)}" for c in cnt_cats]
    records: List[Dict[str, object]] = []

    grouped = {}
    if not breakdown_df.empty:
        for key, sub in breakdown_df.groupby(["start_cnt", "stv_cat"], sort=False):
            grouped[(str(key[0]), str(key[1]))] = {str(r.last_cnt): float(r.contrib) for r in sub.itertuples(index=False)}

    for row in cnt2stv.itertuples(index=False):
        start_cnt = str(row.start_cnt)
        stv_cat = str(row.stv_cat)
        contrib_by_last_cnt = grouped.get((start_cnt, stv_cat), {})

        rec: Dict[str, object] = {
            "start_cnt": start_cnt,
            "start_cnt_label": friendly_cnt(start_cnt),
            "stv_cat": stv_cat,
            "stv_label": friendly_stv(stv_cat),
            "pair_label": f"{friendly_cnt(start_cnt)}{ARROW}{friendly_stv(stv_cat)}",
            "prob": float(row.prob),
        }
        for seg in segment_names:
            rec[seg] = 0.0

        for last_cnt, value in contrib_by_last_cnt.items():
            if last_cnt == start_cnt:
                rec["DIRECT"] = float(rec["DIRECT"]) + float(value)
            elif last_cnt in [str(c) for c in cnt_cats]:
                rec[f"via {friendly_cnt(last_cnt)}"] = float(rec[f"via {friendly_cnt(last_cnt)}"]) + float(value)

        cnt_only_total = float(sum(float(rec[seg]) for seg in segment_names))
        direct = float(rec["DIRECT"])
        rec["cnt_only_total"] = cnt_only_total
        rec["direct_fraction"] = direct / cnt_only_total if cnt_only_total > 0 else 0.0
        rec["via_cnt_fraction"] = 1.0 - float(rec["direct_fraction"]) if cnt_only_total > 0 else 0.0
        records.append(rec)

    all_plot = pd.DataFrame(records)
    if all_plot.empty:
        return all_plot

    if rank_by not in all_plot.columns:
        raise ValueError(f"rank_by must be one of the plot-data columns; got {rank_by!r}")

    ranked = all_plot.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top = ranked.head(max(int(topk), 0)).copy()

    if force_include_stv is not None and str(force_include_stv).strip() != "":
        force = str(force_include_stv)
        forced_candidates = ranked[ranked["stv_cat"].astype(str) == force]
        if not forced_candidates.empty:
            forced = forced_candidates.head(1).copy()
            forced_key = (str(forced.iloc[0]["start_cnt"]), str(forced.iloc[0]["stv_cat"]))
            top_keys = set(zip(top["start_cnt"].astype(str), top["stv_cat"].astype(str)))
            if forced_key not in top_keys:
                top = pd.concat([top, forced], ignore_index=True)

    # Preserve the original top-k order, then append the forced pair at the end.
    return top.reset_index(drop=True)


def plot_best_pairs_stacked(
    plot_df: pd.DataFrame,
    cnt_cats: List[str],
    output_png: str,
    plot_width: float = 16.0,
    plot_height: float = 9.21,
    plot_dpi: int = 128,
) -> None:
    """Draw the CNT-only DIRECT/via-CNT stacked bar plot."""
    if plot_df.empty:
        print("Note: no rows available for best-pairs plot.")
        return

    segment_order = ["DIRECT"] + [f"via {friendly_cnt(c)}" for c in cnt_cats]
    pastel = plt.get_cmap("Pastel2")
    colors = {seg: pastel(i) for i, seg in enumerate(segment_order)}
    edge = "#444"

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    y = np.arange(len(plot_df))
    left = np.zeros(len(plot_df), dtype=float)

    for seg in segment_order:
        values = plot_df[seg].astype(float).to_numpy() if seg in plot_df.columns else np.zeros(len(plot_df))
        ax.barh(
            y,
            values,
            left=left,
            color=colors[seg],
            edgecolor=edge,
            linewidth=0.8,
            label=seg,
        )
        left += values

    max_total = float(max(plot_df["cnt_only_total"].max(), 1e-12))
    text_pad = max_total * 0.003
    for i, row in enumerate(plot_df.itertuples(index=False)):
        total = float(row.cnt_only_total)
        direct_fraction = float(row.direct_fraction) if total > 0 else 0.0
        direct_pct = int(round(100.0 * direct_fraction))
        via_pct = max(0, 100 - direct_pct)
        label = f"{total:.3f}\n{direct_pct}% direct / {via_pct}% via-CNT"
        ax.text(total + text_pad, y[i], label, va="center", fontsize=10, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["pair_label"].values, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("CNT-only first-hit contribution (DIRECT + via CNT)", fontsize=12)
    ax.set_title(
        f"Top CNT{ARROW}STV pairs {EN_DASH} DIRECT vs CNT{ARROW}CNT detours (WD/OTHER excluded)",
        fontsize=15,
        pad=10,
    )

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[seg]) for seg in segment_order]
    ax.legend(
        handles,
        segment_order,
        loc="lower right",
        bbox_to_anchor=(0.965, 0.03),
        ncol=2,
        fontsize=16,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.8,
        borderpad=0.6,
        labelspacing=0.8,
        columnspacing=1.4,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="both", which="both", length=3, bottom=True, left=True, top=False, right=False)
    ax.margins(x=0.02)
    ax.set_xlim(0, max_total * 1.05)

    fig.tight_layout()
    fig.savefig(output_png, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)


def make_pseudotime_table(
    df: pd.DataFrame,
    label_col: str,
    pseudotime: np.ndarray,
    final_root_cnt: Optional[str],
    hub_anchor_stv: Optional[str],
    initial_root_cnt: str,
) -> pd.DataFrame:
    """Create the CSV table for final pseudotime output."""
    final_root_cnt = None if final_root_cnt is None else str(final_root_cnt)
    hub_anchor_stv = None if hub_anchor_stv is None else str(hub_anchor_stv)

    out = pd.DataFrame(
        {
            "row_index": np.arange(df.shape[0]),
            label_col: df[label_col].values,
            "group": df["group"].values,
            "CAT": df["CAT"].astype(str).values,
            "friendly_CAT": [
                friendly_cnt(c) if g == "CNT" else friendly_stv(c) if g == "STV" else str(c)
                for g, c in zip(df["group"].values, df["CAT"].astype(str).values)
            ],
            "pseudotime": pseudotime,
            "initial_root_cnt_for_best_pair_plot": str(initial_root_cnt),
            "initial_root_label_for_best_pair_plot": friendly_cnt(initial_root_cnt),
            "final_root_cnt": final_root_cnt,
            "final_root_label": friendly_cnt(final_root_cnt) if final_root_cnt is not None else None,
            "selected_anchor_stv": hub_anchor_stv,
            "selected_anchor_label": friendly_stv(hub_anchor_stv) if hub_anchor_stv is not None else None,
        }
    )

    out["is_final_root_cell"] = (
        (out["group"] == "CNT") & (out["CAT"].astype(str) == str(final_root_cnt))
        if final_root_cnt is not None
        else False
    )
    out["is_selected_anchor_cell"] = (
        (out["group"] == "STV") & (out["CAT"].astype(str) == str(hub_anchor_stv))
        if hub_anchor_stv is not None
        else False
    )
    return out


def save_pca_pseudotime_outputs(
    df: pd.DataFrame,
    label_col: str,
    scores: np.ndarray,
    pseudotime: np.ndarray,
    output_dir: str,
    final_root_cnt: Optional[str],
    hub_anchor_stv: Optional[str],
    initial_root_cnt: str,
    plot_groups: Optional[List[str]] = None,
    label_centroids: bool = False,
    make_filtered_plot: bool = True,
    filter_upper_quantile: float = 0.99,
    filter_max_pseudotime: float = 0.999,
) -> Dict[str, str]:
    """
    Save a PCA + final-pseudotime CSV and a PC1/PC2 plot.

    This is intentionally an output-only helper: it uses the PCA scores and final
    graph-distance pseudotime already computed by the clean pipeline and does not
    recompute PCA, pseudotime, the Markov chain, best pairs, routes, or sensitivity.

    The plot style follows the standalone PCA pseudotime plotting script:
      - figsize=(9.6, 7.2), dpi=200
      - seaborn cubehelix_palette(start=.5, rot=-.5)
      - PC1 vs PC2 scatter, s=25, edgecolors="none", alpha=0.9
      - black centroid dots per CAT, s=100, linewidth=1.5
      - equal aspect ratio
      - saved at dpi=300

    The optional filtered plot keeps the exact same PCA/pseudotime logic, but
    removes high-end pseudotime outliers from the display only. It does not
    change the graph, pseudotime values, first-hit analysis, route analysis,
    or any existing outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    if scores.ndim != 2 or scores.shape[0] != df.shape[0]:
        raise ValueError("scores must be a 2D array with one row per input cell.")
    if len(pseudotime) != df.shape[0]:
        raise ValueError("pseudotime must have one value per input cell.")

    final_root_cnt = None if final_root_cnt is None else str(final_root_cnt)
    hub_anchor_stv = None if hub_anchor_stv is None else str(hub_anchor_stv)

    cells = df.copy()
    cells.insert(0, "row_index", np.arange(df.shape[0]))

    # Ensure these columns are present and friendly for downstream plotting/inspection.
    cells["group"] = df["group"].values
    cells["CAT"] = df["CAT"].astype(str).values
    cells["friendly_CAT"] = [
        friendly_cnt(c) if g == "CNT" else friendly_stv(c) if g == "STV" else str(c)
        for g, c in zip(df["group"].values, df["CAT"].astype(str).values)
    ]

    for j in range(scores.shape[1]):
        cells[f"PC{j + 1}"] = scores[:, j]

    cells["pseudotime"] = np.asarray(pseudotime, dtype=float)
    cells["final_pseudotime"] = np.asarray(pseudotime, dtype=float)
    cells["initial_root_cnt_for_best_pair_plot"] = str(initial_root_cnt)
    cells["initial_root_label_for_best_pair_plot"] = friendly_cnt(initial_root_cnt)
    cells["final_root_cnt"] = final_root_cnt
    cells["final_root_label"] = friendly_cnt(final_root_cnt) if final_root_cnt is not None else None
    cells["selected_anchor_stv"] = hub_anchor_stv
    cells["selected_anchor_label"] = friendly_stv(hub_anchor_stv) if hub_anchor_stv is not None else None

    if plot_groups is None or len(plot_groups) == 0:
        plot_mask = np.ones(df.shape[0], dtype=bool)
        cells["included_in_pca_pseudotime_plot"] = True
    else:
        wanted_groups = {str(g).upper() for g in plot_groups}
        plot_mask = df["group"].astype(str).str.upper().isin(wanted_groups).to_numpy()
        cells["included_in_pca_pseudotime_plot"] = plot_mask

    # Display-only filtering for the optional second PCA plot.
    # This keeps the clean graph-distance pseudotime unchanged, but removes
    # cells with extreme high pseudotime values that can compress the colormap.
    pt_all = np.asarray(pseudotime, dtype=float)
    finite_plot_mask = plot_mask & np.isfinite(pt_all)
    filtered_plot_mask = np.zeros(df.shape[0], dtype=bool)
    pca_filter_cutoff = np.nan
    pca_filter_quantile_value = np.nan

    if make_filtered_plot and np.any(finite_plot_mask):
        q = float(np.clip(filter_upper_quantile, 0.0, 1.0))
        pca_filter_quantile_value = float(np.quantile(pt_all[finite_plot_mask], q))
        pca_filter_cutoff = min(pca_filter_quantile_value, float(filter_max_pseudotime))
        filtered_plot_mask = finite_plot_mask & (pt_all < pca_filter_cutoff)

        # If the cutoff is too aggressive for a small dataset, fall back to
        # removing only exact/near-1 values. This prevents an accidental empty plot.
        if filtered_plot_mask.sum() < 3 and finite_plot_mask.sum() >= 3:
            pca_filter_cutoff = float(filter_max_pseudotime)
            filtered_plot_mask = finite_plot_mask & (pt_all < pca_filter_cutoff)

    cells["included_in_pca_pseudotime_filtered_plot"] = filtered_plot_mask
    cells["pca_pseudotime_filter_upper_quantile"] = float(filter_upper_quantile)
    cells["pca_pseudotime_filter_quantile_value"] = pca_filter_quantile_value
    cells["pca_pseudotime_filter_max_pseudotime"] = float(filter_max_pseudotime)
    cells["pca_pseudotime_filter_cutoff"] = pca_filter_cutoff

    csv_path = os.path.join(output_dir, "cells_with_pca_and_pseudotime.csv")
    cells.to_csv(csv_path, index=False)

    png_path = os.path.join(output_dir, "pca_projection_with_final_pseudotime.png")
    filtered_png_path = os.path.join(output_dir, "pca_projection_with_final_pseudotime_filtered.png")
    signed_png_path = os.path.join(output_dir, "pca_projection_with_signed_control_to_fh_pseudotime.png")
    filter_summary_path = os.path.join(output_dir, "pca_pseudotime_filter_summary.csv")
    signed_summary_path = os.path.join(output_dir, "pca_signed_control_to_fh_pseudotime_summary.csv")
    if not np.any(plot_mask):
        print("No cells matched --pca-plot-groups; saved CSV but skipped PCA pseudotime plot.")
        return {
            "cells_with_pca_and_pseudotime": csv_path,
            "pca_projection_with_final_pseudotime": "",
            "pca_projection_with_final_pseudotime_filtered": "",
            "pca_pseudotime_filter_summary": "",
        }

    plot_scores = scores[plot_mask]
    plot_pt = np.asarray(pseudotime, dtype=float)[plot_mask]
    plot_df = df.loc[plot_mask].copy().reset_index(drop=True)

    # Use a small fallback only if
    # seaborn is unavailable in the user's Python environment.
    try:
        import seaborn as sns
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    except Exception:
        cmap = plt.get_cmap("cubehelix")

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=200)
    sc = ax.scatter(
        plot_scores[:, 0],
        plot_scores[:, 1],
        c=plot_pt,
        cmap=cmap,
        s=25,
        edgecolors="none",
        alpha=0.9,
    )

    # Compute and plot black centroid dots per CAT, matching the standalone style.
    cat_labels = plot_df["CAT"].astype(str).values
    unique_cats = sorted(np.unique(cat_labels), key=lambda x: (len(x), x))

    for cat in unique_cats:
        mask = cat_labels == cat
        if mask.sum() == 0:
            continue

        centroid = plot_scores[mask].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=100,
            facecolor="black",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )

        if label_centroids:
            groups_for_cat = plot_df.loc[mask, "group"].astype(str).unique().tolist()
            if "CNT" in groups_for_cat:
                label = friendly_cnt(cat)
            elif "STV" in groups_for_cat:
                label = friendly_stv(cat)
            else:
                label = f"CAT:{cat}"
            ax.text(
                centroid[0],
                centroid[1] + 0.4,
                label,
                fontsize=11,
                color="black",
                ha="center",
                va="bottom",
                zorder=11,
            )

    root_label = friendly_cnt(final_root_cnt) if final_root_cnt is not None else "selected root"
    cbar_label = f"Final pseudotime ({root_label} root)"
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal")

    plt.tight_layout()
   
    plt.close(fig)

    filtered_png_written = ""
    filter_summary_written = ""
    if make_filtered_plot:
        # Save a small summary table so it is transparent which cells were
        # excluded from only the filtered PCA display.
        finite_pt = pt_all[finite_plot_mask]
        filtered_pt_all = pt_all[filtered_plot_mask]
        summary_row = {
            "plot_groups": ";".join(plot_groups) if plot_groups else "ALL",
            "n_plot_cells_before_filter": int(finite_plot_mask.sum()),
            "n_plot_cells_after_filter": int(filtered_plot_mask.sum()),
            "n_removed_by_filter": int(finite_plot_mask.sum() - filtered_plot_mask.sum()),
            "filter_upper_quantile": float(filter_upper_quantile),
            "filter_quantile_value": pca_filter_quantile_value,
            "filter_max_pseudotime": float(filter_max_pseudotime),
            "filter_cutoff_used": pca_filter_cutoff,
            "before_min": float(np.nanmin(finite_pt)) if finite_pt.size else np.nan,
            "before_median": float(np.nanmedian(finite_pt)) if finite_pt.size else np.nan,
            "before_max": float(np.nanmax(finite_pt)) if finite_pt.size else np.nan,
            "after_min": float(np.nanmin(filtered_pt_all)) if filtered_pt_all.size else np.nan,
            "after_median": float(np.nanmedian(filtered_pt_all)) if filtered_pt_all.size else np.nan,
            "after_max": float(np.nanmax(filtered_pt_all)) if filtered_pt_all.size else np.nan,
        }
        pd.DataFrame([summary_row]).to_csv(filter_summary_path, index=False)
        filter_summary_written = filter_summary_path

        if filtered_plot_mask.sum() >= 3:
            filtered_scores = scores[filtered_plot_mask]
            filtered_pt = pt_all[filtered_plot_mask]
            filtered_df = df.loc[filtered_plot_mask].copy().reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=200)
            sc = ax.scatter(
                filtered_scores[:, 0],
                filtered_scores[:, 1],
                c=filtered_pt,
                cmap=cmap,
                s=25,
                edgecolors="none",
                alpha=0.9,
            )

            cat_labels = filtered_df["CAT"].astype(str).values
            unique_cats = sorted(np.unique(cat_labels), key=lambda x: (len(x), x))

            for cat in unique_cats:
                mask = cat_labels == cat
                if mask.sum() == 0:
                    continue

                centroid = filtered_scores[mask].mean(axis=0)
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    s=100,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=10,
                )

                if label_centroids:
                    groups_for_cat = filtered_df.loc[mask, "group"].astype(str).unique().tolist()
                    if "CNT" in groups_for_cat:
                        label = friendly_cnt(cat)
                    elif "STV" in groups_for_cat:
                        label = friendly_stv(cat)
                    else:
                        label = f"CAT:{cat}"
                    ax.text(
                        centroid[0],
                        centroid[1] + 0.4,
                        label,
                        fontsize=11,
                        color="black",
                        ha="center",
                        va="bottom",
                        zorder=11,
                    )

            cbar_label = f"Final pseudotime ({root_label} root; filtered display)"
            plt.colorbar(sc, ax=ax, label=cbar_label)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_aspect("equal")

            plt.tight_layout()
            fig.savefig(filtered_png_path, dpi=300)
            plt.close(fig)
            filtered_png_written = filtered_png_path
        else:
            print("Filtered PCA pseudotime plot skipped: fewer than 3 cells remained after filtering.")

    return {
        "cells_with_pca_and_pseudotime": csv_path,
        "pca_projection_with_final_pseudotime": png_path,
        "pca_projection_with_final_pseudotime_filtered": filtered_png_written,
        "pca_pseudotime_filter_summary": filter_summary_written,
    }



def save_cnt_stv_only_pca_pseudotime_outputs(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    output_dir: str,
    final_root_cnt: Optional[str],
    hub_anchor_stv: Optional[str],
    initial_root_cnt: str,
    n_components: int,
    graph_pcs: int,
    knn: int,
    graph_metric: str,
    all_group_pseudotime: Optional[np.ndarray] = None,
    analysis_groups: Optional[List[str]] = None,
    plot_groups: Optional[List[str]] = None,
    label_centroids: bool = False,
    make_filtered_plot: bool = True,
    filter_upper_quantile: float = 0.99,
    filter_max_pseudotime: float = 0.999,
) -> Dict[str, str]:
    """
    Save PCA scatter/pseudotime outputs using a CNT/STV-only PCA+graph
    pseudotime calculation for visualization.

    This helper is output-only. It does NOT change the main clean analysis
    graph, first-hit probabilities, hub/anchor selection, root sensitivity,
    route probabilities, best_pairs_topk_bars.png, or route_bars.png.

    Visualization strategy, applied only to --pca-pseudotime-analysis-groups
    (default CNT STV):
      1. Run PCA on the same numeric feature columns.
      2. Build the same local-scale kNN graph in PCA space.
      3. Convert graph affinities to graph lengths.
      4. Use Dijkstra graph distance from H4 cells (CNT CAT=3).
      5. Normalize graph distances to 0-1.
      6. Remove high-pseudotime outliers and rescale the retained cells to 0-1.
      7. Export retained H4/FH cells with that value named pseudotime_full.
    """
    os.makedirs(output_dir, exist_ok=True)

    if analysis_groups is None or len(analysis_groups) == 0:
        analysis_groups = ["CNT", "STV"]
    wanted_analysis = {str(g).upper() for g in analysis_groups}
    analysis_mask = df["group"].astype(str).str.upper().isin(wanted_analysis).to_numpy()
    analysis_indices = np.where(analysis_mask)[0]

    if analysis_indices.size < 3:
        raise RuntimeError(
            "Need at least 3 cells in --pca-pseudotime-analysis-groups "
            f"for the PCA pseudotime display. Found {analysis_indices.size}."
        )

    final_root_cnt = None if final_root_cnt is None else str(final_root_cnt)
    hub_anchor_stv = None if hub_anchor_stv is None else str(hub_anchor_stv)

    # The remodeling-axis input is explicitly an H4-to-FH analysis.  Keep the
    # display/export pseudotime rooted at H4 (CNT CAT=3), independently of the
    # hub-selected root used by pseudotime.csv.
    root_for_plot = "3"

    df_sub = df.loc[analysis_mask].copy().reset_index(drop=True)
    features_sub = df.loc[analysis_mask, feature_cols].copy()
    sub_medians = features_sub.median(numeric_only=True)
    full_medians = df[feature_cols].median(numeric_only=True)
    features_sub = features_sub.fillna(sub_medians).fillna(full_medians).fillna(0.0)

    _, scores_sub, _ = run_pca(features_sub, n_components)
    if scores_sub.shape[1] < 2:
        raise RuntimeError("Need at least 2 PCs for the PCA pseudotime display plot.")

    graph_pcs_sub = min(int(graph_pcs), scores_sub.shape[1])
    graph_sub = build_knn_graph(scores_sub[:, :graph_pcs_sub], k=knn, metric=graph_metric)

    roots_sub = np.where(
        (df_sub["group"] == "CNT") & (df_sub["CAT"].astype(str) == root_for_plot)
    )[0]
    if roots_sub.size == 0:
        raise RuntimeError(
            "No H4 root cells (CNT CAT=3) were found in the CNT/STV analysis subset."
        )

    pt_sub = compute_pseudotime(graph_sub, roots_sub)
    dist_sub = compute_graph_distance_from_roots(graph_sub, roots_sub)

    cells = df.copy()
    cells.insert(0, "row_index", np.arange(df.shape[0]))
    cells["group"] = df["group"].values
    cells["CAT"] = df["CAT"].astype(str).values
    cells["friendly_CAT"] = [
        friendly_cnt(c) if g == "CNT" else friendly_stv(c) if g == "STV" else str(c)
        for g, c in zip(df["group"].values, df["CAT"].astype(str).values)
    ]

    if all_group_pseudotime is not None and len(all_group_pseudotime) == df.shape[0]:
        cells["final_pseudotime_all_groups"] = np.asarray(all_group_pseudotime, dtype=float)

    for j in range(scores_sub.shape[1]):
        col = f"PC{j + 1}"
        cells[col] = np.nan
        cells.loc[analysis_mask, col] = scores_sub[:, j]

    # The plotting pseudotime is explicitly CNT/STV-only. WD/OTHER rows are kept
    # in the CSV for traceability but receive NaN for these display-specific fields.
    cells["pseudotime"] = np.nan
    cells["final_pseudotime"] = np.nan
    cells["cnt_stv_graph_pseudotime"] = np.nan
    cells["cnt_stv_raw_graph_distance_from_root"] = np.nan
    cells.loc[analysis_mask, "pseudotime"] = pt_sub
    cells.loc[analysis_mask, "final_pseudotime"] = pt_sub
    cells.loc[analysis_mask, "cnt_stv_graph_pseudotime"] = pt_sub
    cells.loc[analysis_mask, "cnt_stv_raw_graph_distance_from_root"] = dist_sub

    cells["pca_pseudotime_basis"] = "CNT/STV-only PCA+graph pseudotime"
    cells["pca_pseudotime_analysis_groups"] = ";".join(analysis_groups)
    cells["initial_root_cnt_for_best_pair_plot"] = str(initial_root_cnt)
    cells["initial_root_label_for_best_pair_plot"] = friendly_cnt(initial_root_cnt)
    cells["final_root_cnt"] = str(root_for_plot)
    cells["final_root_label"] = friendly_cnt(root_for_plot)
    cells["selected_anchor_stv"] = hub_anchor_stv
    cells["selected_anchor_label"] = friendly_stv(hub_anchor_stv) if hub_anchor_stv is not None else None

    if plot_groups is None or len(plot_groups) == 0:
        plot_groups = analysis_groups
    wanted_plot = {str(g).upper() for g in plot_groups}
    plot_mask_sub = df_sub["group"].astype(str).str.upper().isin(wanted_plot).to_numpy()
    plot_mask_full = np.zeros(df.shape[0], dtype=bool)
    plot_mask_full[analysis_indices[plot_mask_sub]] = True
    cells["included_in_pca_pseudotime_plot"] = plot_mask_full
    cells["included_in_pca_pseudotime_analysis"] = analysis_mask

    finite_plot_mask_sub = plot_mask_sub & np.isfinite(pt_sub)
    filtered_plot_mask_sub = np.zeros(df_sub.shape[0], dtype=bool)
    pca_filter_cutoff = np.nan
    pca_filter_quantile_value = np.nan

    if make_filtered_plot and np.any(finite_plot_mask_sub):
        q = float(np.clip(filter_upper_quantile, 0.0, 1.0))
        pca_filter_quantile_value = float(np.quantile(pt_sub[finite_plot_mask_sub], q))
        pca_filter_cutoff = min(pca_filter_quantile_value, float(filter_max_pseudotime))
        filtered_plot_mask_sub = finite_plot_mask_sub & (pt_sub < pca_filter_cutoff)

        if filtered_plot_mask_sub.sum() < 3 and finite_plot_mask_sub.sum() >= 3:
            pca_filter_cutoff = float(filter_max_pseudotime)
            filtered_plot_mask_sub = finite_plot_mask_sub & (pt_sub < pca_filter_cutoff)

    filtered_plot_mask_full = np.zeros(df.shape[0], dtype=bool)
    if filtered_plot_mask_sub.size:
        filtered_plot_mask_full[analysis_indices[filtered_plot_mask_sub]] = True

    cells["included_in_pca_pseudotime_filtered_plot"] = filtered_plot_mask_full

    # Display-only rescaling for the filtered PCA plot.
    # The underlying CNT/STV graph pseudotime is not changed; this rescales only
    # the cells that remain after the high-pseudotime outlier filter so the
    # visible cloud spans the full 0-1 colorbar range.
    filtered_rescaled_sub = np.full(df_sub.shape[0], np.nan, dtype=float)
    if filtered_plot_mask_sub.sum() > 0:
        filtered_rescaled_sub[filtered_plot_mask_sub] = _rescale_values_to_unit_interval(
            pt_sub[filtered_plot_mask_sub]
        )
    cells["cnt_stv_graph_pseudotime_filtered_rescaled_0_1"] = np.nan
    cells.loc[analysis_mask, "cnt_stv_graph_pseudotime_filtered_rescaled_0_1"] = filtered_rescaled_sub

    # Create the clean H4 + FH remodeling input from the *filtered and
    # rescaled* CNT/STV-only graph pseudotime.  No additional PCA, graph, or
    # pseudotime calculation is performed for this export.
    h4_fh_csv_path = ""
    if make_filtered_plot and filtered_plot_mask_sub.sum() >= 3:
        h4_fh_csv_path = save_filtered_root_cnt_fh_plot_input_csv(
            df_sub=df_sub,
            label_col=label_col,
            feature_cols=feature_cols,
            filtered_mask_sub=filtered_plot_mask_sub,
            filtered_rescaled_pseudotime=filtered_rescaled_sub,
            output_dir=output_dir,
            root_cnt="3",
            stv_cats=list(FH_NAME.keys()),
        )
        print("Filtered/rescaled H4 + FH CSV:", h4_fh_csv_path)

    signed_minus1_to_1_sub, signed_display_0_1_sub, signed_display_summary = (
        _compute_signed_control_to_fh_display_pseudotime(
            df_sub=df_sub,
            root_distance_sub=dist_sub,
            filtered_mask_sub=filtered_plot_mask_sub,
            root_cnt=root_for_plot,
        )
    )
    cells["cnt_stv_signed_control_to_fh_minus1_to_1"] = np.nan
    cells["cnt_stv_signed_control_to_fh_pseudotime_0_1"] = np.nan
    cells.loc[analysis_mask, "cnt_stv_signed_control_to_fh_minus1_to_1"] = signed_minus1_to_1_sub
    cells.loc[analysis_mask, "cnt_stv_signed_control_to_fh_pseudotime_0_1"] = signed_display_0_1_sub

    cells["pca_pseudotime_filter_upper_quantile"] = float(filter_upper_quantile)
    cells["pca_pseudotime_filter_quantile_value"] = pca_filter_quantile_value
    cells["pca_pseudotime_filter_max_pseudotime"] = float(filter_max_pseudotime)
    cells["pca_pseudotime_filter_cutoff"] = pca_filter_cutoff

    csv_path = os.path.join(output_dir, "cells_with_pca_and_pseudotime.csv")
    cells.to_csv(csv_path, index=False)

    cnt_stv_csv_path = os.path.join(output_dir, "pseudotime_cnt_stv_only.csv")
    cells.loc[analysis_mask].copy().to_csv(cnt_stv_csv_path, index=False)

    # The unfiltered PCA projection is intentionally no longer generated.
    obsolete_png_path = os.path.join(output_dir, "pca_projection_with_final_pseudotime.png")
    if os.path.exists(obsolete_png_path):
        os.remove(obsolete_png_path)

    filtered_png_path = os.path.join(output_dir, "pca_projection_with_final_pseudotime_filtered.png")
    signed_png_path = os.path.join(output_dir, "pca_projection_with_signed_control_to_fh_pseudotime.png")
    filter_summary_path = os.path.join(output_dir, "pca_pseudotime_filter_summary.csv")
    signed_summary_path = os.path.join(output_dir, "pca_signed_control_to_fh_pseudotime_summary.csv")

    if not np.any(plot_mask_sub):
        print("No cells matched --pca-plot-groups inside the PCA pseudotime analysis subset; saved CSV but skipped plot.")
        return {
            "cells_with_pca_and_pseudotime": csv_path,
            "pseudotime_cnt_stv_only": cnt_stv_csv_path,
            "cells_with_pseudotime_H4_FH_only": h4_fh_csv_path,
            "pca_projection_with_final_pseudotime": "",
            "pca_projection_with_final_pseudotime_filtered": "",
            "pca_projection_with_signed_control_to_fh_pseudotime": "",
            "pca_pseudotime_filter_summary": "",
            "pca_signed_control_to_fh_pseudotime_summary": "",
        }

    try:
        import seaborn as sns
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    except Exception:
        cmap = plt.get_cmap("cubehelix")

    def _draw_pca_plot(
        scores_plot: np.ndarray,
        pt_plot: np.ndarray,
        df_plot: pd.DataFrame,
        output_png: str,
        colorbar_label: str,
        colorbar_ticks: Optional[List[float]] = None,
        colorbar_ticklabels: Optional[List[str]] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=200)
        sc = ax.scatter(
            scores_plot[:, 0],
            scores_plot[:, 1],
            c=pt_plot,
            cmap=cmap,
            s=25,
            edgecolors="none",
            alpha=0.9,
        )

        cat_labels = df_plot["CAT"].astype(str).values
        unique_cats = sorted(np.unique(cat_labels), key=lambda x: (len(x), x))
        for cat in unique_cats:
            mask = cat_labels == cat
            if mask.sum() == 0:
                continue
            centroid = scores_plot[mask].mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                s=100,
                facecolor="black",
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )
            if label_centroids:
                groups_for_cat = df_plot.loc[mask, "group"].astype(str).unique().tolist()
                if "CNT" in groups_for_cat:
                    label = friendly_cnt(cat)
                elif "STV" in groups_for_cat:
                    label = friendly_stv(cat)
                else:
                    label = f"CAT:{cat}"
                ax.text(
                    centroid[0],
                    centroid[1] + 0.4,
                    label,
                    fontsize=11,
                    color="black",
                    ha="center",
                    va="bottom",
                    zorder=11,
                )

        cbar = plt.colorbar(sc, ax=ax, label=colorbar_label)
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
        if colorbar_ticklabels is not None:
            cbar.set_ticklabels(colorbar_ticklabels)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300)
        plt.close(fig)

    root_label = friendly_cnt(root_for_plot)

    filtered_png_written = ""
    filter_summary_written = ""
    if make_filtered_plot:
        finite_pt = pt_sub[finite_plot_mask_sub]
        filtered_pt = pt_sub[filtered_plot_mask_sub]
        filtered_pt_rescaled = _rescale_values_to_unit_interval(filtered_pt)
        summary_row = {
            "pseudotime_basis": "CNT/STV-only PCA+graph pseudotime",
            "analysis_groups": ";".join(analysis_groups),
            "plot_groups": ";".join(plot_groups) if plot_groups else "ALL_ANALYSIS_GROUPS",
            "root_cnt_used_for_plot_pseudotime": str(root_for_plot),
            "root_label_used_for_plot_pseudotime": root_label,
            "n_analysis_cells": int(analysis_mask.sum()),
            "n_plot_cells_before_filter": int(finite_plot_mask_sub.sum()),
            "n_plot_cells_after_filter": int(filtered_plot_mask_sub.sum()),
            "n_removed_by_filter": int(finite_plot_mask_sub.sum() - filtered_plot_mask_sub.sum()),
            "filter_upper_quantile": float(filter_upper_quantile),
            "filter_quantile_value": pca_filter_quantile_value,
            "filter_max_pseudotime": float(filter_max_pseudotime),
            "filter_cutoff_used": pca_filter_cutoff,
            "before_min": float(np.nanmin(finite_pt)) if finite_pt.size else np.nan,
            "before_median": float(np.nanmedian(finite_pt)) if finite_pt.size else np.nan,
            "before_max": float(np.nanmax(finite_pt)) if finite_pt.size else np.nan,
            "after_min": float(np.nanmin(filtered_pt)) if filtered_pt.size else np.nan,
            "after_median": float(np.nanmedian(filtered_pt)) if filtered_pt.size else np.nan,
            "after_max": float(np.nanmax(filtered_pt)) if filtered_pt.size else np.nan,
            "after_rescaled_min": float(np.nanmin(filtered_pt_rescaled)) if filtered_pt_rescaled.size else np.nan,
            "after_rescaled_median": float(np.nanmedian(filtered_pt_rescaled)) if filtered_pt_rescaled.size else np.nan,
            "after_rescaled_max": float(np.nanmax(filtered_pt_rescaled)) if filtered_pt_rescaled.size else np.nan,
            "filtered_plot_color_scale": "filtered remaining cells rescaled to 0-1",
        }
        pd.DataFrame([summary_row]).to_csv(filter_summary_path, index=False)
        filter_summary_written = filter_summary_path

        signed_summary = dict(signed_display_summary)
        signed_summary.update({
            "pseudotime_basis": "CNT/STV-only raw graph distance from root, signed for display after filtering",
            "analysis_groups": ";".join(analysis_groups),
            "plot_groups": ";".join(plot_groups) if plot_groups else "ALL_ANALYSIS_GROUPS",
            "filter_cutoff_used_before_signed_transform": pca_filter_cutoff,
        })
        pd.DataFrame([signed_summary]).to_csv(signed_summary_path, index=False)

        if filtered_plot_mask_sub.sum() >= 3:
            filtered_scores = scores_sub[filtered_plot_mask_sub]
            filtered_df = df_sub.loc[filtered_plot_mask_sub].copy().reset_index(drop=True)
            filtered_label = f"CNT/STV-only graph pseudotime ({root_label} root; filtered cells rescaled 0-1)"
            _draw_pca_plot(filtered_scores, filtered_pt_rescaled, filtered_df, filtered_png_path, filtered_label)
            filtered_png_written = filtered_png_path

            signed_values_plot = signed_display_0_1_sub[filtered_plot_mask_sub]
            finite_signed = np.isfinite(signed_values_plot)
            if finite_signed.sum() >= 3:
                signed_label = (
                    f"Signed control→{root_label}→FH display pseudotime "
                    "(0=control side, 0.5=root, 1=FH side)"
                )
                _draw_pca_plot(
                    filtered_scores[finite_signed],
                    signed_values_plot[finite_signed],
                    filtered_df.loc[finite_signed].copy().reset_index(drop=True),
                    signed_png_path,
                    signed_label,
                    colorbar_ticks=[0.0, 0.5, 1.0],
                    colorbar_ticklabels=["control side", root_label, "FH side"],
                )
            else:
                print("Signed control-to-FH PCA plot skipped: fewer than 3 finite signed-display values.")
        else:
            print("Filtered CNT/STV-only PCA pseudotime plot skipped: fewer than 3 cells remained after filtering.")

    return {
        "cells_with_pca_and_pseudotime": csv_path,
        "pseudotime_cnt_stv_only": cnt_stv_csv_path,
        "cells_with_pseudotime_H4_FH_only": h4_fh_csv_path,
        "pca_projection_with_final_pseudotime": "",
        "pca_projection_with_final_pseudotime_filtered": filtered_png_written,
        "pca_projection_with_signed_control_to_fh_pseudotime": signed_png_path if os.path.exists(signed_png_path) else "",
        "pca_pseudotime_filter_summary": filter_summary_written,
        "pca_signed_control_to_fh_pseudotime_summary": signed_summary_path if os.path.exists(signed_summary_path) else "",
    }


def save_filtered_root_cnt_fh_plot_input_csv(
    df_sub: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    filtered_mask_sub: np.ndarray,
    filtered_rescaled_pseudotime: np.ndarray,
    output_dir: str,
    root_cnt: str = "3",
    stv_cats: Optional[List[str]] = None,
) -> str:
    """Export filtered H4/FH cells with rescaled pseudotime named pseudotime_full.

    The input dataframe is already the CNT/STV analysis subset.  This function
    performs only row/column selection and file writing.  It does not refit PCA,
    rebuild a graph, recompute graph distances, or rescale pseudotime again.

    Rows retained:
      - H4: CNT, CAT=3 (or ``root_cnt``);
      - FH1-FH4: STV, CAT in 5,6,7,8 (or ``stv_cats``);
      - only cells that passed ``filtered_mask_sub`` and have finite rescaled
        pseudotime.

    The filtered/rescaled values are deliberately written under the column name
    ``pseudotime_full`` so the cleaned remodeling-axis plotting script can be
    used without modification.
    """
    os.makedirs(output_dir, exist_ok=True)

    root_cnt = str(root_cnt)
    stv_set = {str(cat) for cat in (stv_cats or list(FH_NAME.keys()))}

    if df_sub.shape[0] == 0:
        raise RuntimeError("The CNT/STV analysis subset is empty.")

    filtered_mask = np.asarray(filtered_mask_sub, dtype=bool)
    filtered_pt = np.asarray(filtered_rescaled_pseudotime, dtype=float)
    if filtered_mask.ndim != 1 or filtered_mask.shape[0] != df_sub.shape[0]:
        raise ValueError("filtered_mask_sub must contain one value per CNT/STV row.")
    if filtered_pt.ndim != 1 or filtered_pt.shape[0] != df_sub.shape[0]:
        raise ValueError(
            "filtered_rescaled_pseudotime must contain one value per CNT/STV row."
        )

    group_values = df_sub["group"].astype(str).str.upper()
    cat_values = df_sub["CAT"].astype(str)

    root_mask = (group_values == "CNT") & (cat_values == root_cnt)
    fh_mask = (group_values == "STV") & cat_values.isin(stv_set)
    biological_subset = (root_mask | fh_mask).to_numpy()

    export_mask = (
        filtered_mask
        & biological_subset
        & np.isfinite(filtered_pt)
    )

    n_root = int((export_mask & root_mask.to_numpy()).sum())
    n_fh = int((export_mask & fh_mask.to_numpy()).sum())
    if n_root == 0:
        raise RuntimeError(
            f"No filtered CNT root cells remained at CAT={root_cnt} "
            f"({friendly_cnt(root_cnt)})."
        )
    if n_fh == 0:
        raise RuntimeError("No filtered FH/STV cells remained for the H4/FH export.")

    out = df_sub.loc[export_mask, feature_cols + [label_col]].copy().reset_index(drop=True)
    out["group"] = group_values.loc[export_mask].to_numpy()
    out["CAT"] = cat_values.loc[export_mask].to_numpy()
    out["pseudotime_full"] = filtered_pt[export_mask]
    out["pseudotime_basis"] = (
        "H4-rooted CNT/STV-only graph pseudotime; high-end cells filtered; "
        "retained cells rescaled to 0-1"
    )

    root_label = friendly_cnt(root_cnt)
    safe_root_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", root_label).strip("_") or f"CNT{root_cnt}"
    output_path = os.path.join(
        output_dir,
        f"cells_with_pseudotime_{safe_root_label}_FH_only.csv",
    )
    out.to_csv(output_path, index=False)
    return output_path


# ---------- Route testing helpers: selected CNT/hub -> STV routes ----------
def build_rw_markov(graph: csr_matrix) -> csr_matrix:
    """Build an ordinary row-normalized random-walk Markov chain on the graph."""
    graph = graph.tocsr().copy()
    row_sums = np.array(graph.sum(axis=1)).ravel()
    dead = row_sums <= 0
    if np.any(dead):
        graph = graph.tolil()
        for i in np.where(dead)[0]:
            graph[i, i] = 1.0
        graph = graph.tocsr()
        row_sums = np.array(graph.sum(axis=1)).ravel()

    inv = 1.0 / row_sums
    d_inv = csr_matrix((inv, (np.arange(graph.shape[0]), np.arange(graph.shape[0]))), shape=graph.shape)
    return d_inv @ graph


def _solve_absorption_system(q: csr_matrix, r: csr_matrix) -> np.ndarray:
    """Solve (I - Q)^-1 R with the same sparse-LU strategy used elsewhere."""
    t = q.shape[0]
    if t == 0:
        return np.empty((0, r.shape[1]))
    lu = splu((sparse_eye(t, format="csr") - q).tocsc())
    r_dense = r.toarray()
    if r_dense.ndim == 1:
        return lu.solve(r_dense)
    return np.column_stack([lu.solve(r_dense[:, j]) for j in range(r_dense.shape[1])])


def route_first_hit_distribution(
    p: csr_matrix,
    start_dist: np.ndarray,
    target_idx: np.ndarray,
    forbid_idx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    First-hit distribution for route testing from an arbitrary start distribution.

    Returns:
      phi: conditional entry distribution over target_idx, given the target is hit.
      p_hit_target: probability of hitting target_idx before forbid_idx/sinks.
    """
    n = p.shape[0]
    target_idx = np.asarray(target_idx, dtype=int)
    forbid_idx = np.asarray(forbid_idx, dtype=int)
    s = np.asarray(start_dist, dtype=float).copy()

    if s.ndim != 1 or s.shape[0] != n:
        raise ValueError("start_dist must be a length-n vector.")
    if target_idx.size == 0:
        return np.zeros(0, dtype=float), 0.0

    total = float(s.sum())
    if total <= 0:
        return np.zeros(len(target_idx), dtype=float), 0.0
    s /= total

    base_absorbing = np.unique(np.concatenate([target_idx, forbid_idx]))
    absorbing_idx = augment_absorbing_with_sinks(p, base_absorbing, start_support=np.where(s > 0)[0])
    q, r, order, _, transient_mask, absorbing_mask = absorbing_blocks(p, absorbing_idx)
    t = int(transient_mask.sum())
    a = int(absorbing_mask.sum())
    if a == 0:
        return np.zeros(len(target_idx), dtype=float), 0.0

    if t > 0:
        nr = _solve_absorption_system(q, r)
        s_perm = s[order]
        b_cols = s_perm[:t] @ nr + s_perm[t:]
    else:
        s_perm = s[order]
        b_cols = s_perm

    target_mask_full = np.zeros(n, dtype=bool)
    target_mask_full[target_idx] = True
    target_cols_mask = target_mask_full[order][t:]
    if target_cols_mask.sum() == 0:
        return np.zeros(len(target_idx), dtype=float), 0.0

    probs_target_nodes = b_cols[target_cols_mask]
    p_hit_target = float(probs_target_nodes.sum())

    phi = np.zeros(len(target_idx), dtype=float)
    if p_hit_target > 0:
        target_nodes_in_perm = order[t:][np.where(target_cols_mask)[0]]
        node_to_pos = {int(node): i for i, node in enumerate(target_idx.tolist())}
        for node, prob in zip(target_nodes_in_perm, probs_target_nodes):
            phi[node_to_pos[int(node)]] = float(prob) / p_hit_target

    return phi, p_hit_target


def route_probability(
    p: csr_matrix,
    start_idx: np.ndarray,
    waypoint_idx: List[np.ndarray],
    final_idx: np.ndarray,
    forbid_idx: np.ndarray,
) -> float:
    """
    Route probability/score used by the route_bars plot.

    Important interpretation:
      - Direct routes, for example 3>6, are first-hit probabilities.
      - Routes with waypoints, for example 3>5>6, renormalize after each waypoint;
        they are conditional waypoint route scores rather than full unconditional
        multi-stage probabilities. For unconditional two-stage probabilities, use
        two_stage_unconditional_paths.csv.
    """
    n = p.shape[0]
    start_idx = np.asarray(start_idx, dtype=int)
    final_idx = np.asarray(final_idx, dtype=int)
    forbid_idx = np.asarray(forbid_idx, dtype=int)

    if start_idx.size == 0 or final_idx.size == 0:
        return 0.0

    mass = np.zeros(n, dtype=float)
    mass[start_idx] = 1.0 / len(start_idx)

    visited = np.array([], dtype=int)
    for waypoint in waypoint_idx:
        waypoint = np.asarray(waypoint, dtype=int)
        if waypoint.size == 0:
            return 0.0

        absorbing_now = np.unique(np.concatenate([waypoint, final_idx, forbid_idx]))
        forbid_now = np.setdiff1d(absorbing_now, waypoint)
        phi_waypoint, p_hit_waypoint = route_first_hit_distribution(p, mass, waypoint, forbid_now)
        if p_hit_waypoint <= 0:
            return 0.0

        mass = np.zeros(n, dtype=float)
        mass[waypoint] = phi_waypoint
        visited = np.unique(np.concatenate([visited, waypoint]))

    forbid_final = np.setdiff1d(forbid_idx, visited)
    base_absorbing = np.unique(np.concatenate([final_idx, forbid_final]))
    absorbing_nodes = augment_absorbing_with_sinks(p, base_absorbing, start_support=np.where(mass > 0)[0])
    q, r, order, _, transient_mask, _ = absorbing_blocks(p, absorbing_nodes)
    t = int(transient_mask.sum())

    if t > 0:
        nr = _solve_absorption_system(q, r)
        mass_perm = mass[order]
        b_cols = mass_perm[:t] @ nr + mass_perm[t:]
    else:
        mass_perm = mass[order]
        b_cols = mass_perm

    final_mask_full = np.zeros(n, dtype=bool)
    final_mask_full[final_idx] = True
    final_cols_mask = final_mask_full[order][t:]
    return float(b_cols[final_cols_mask].sum())


def set_to_set_kstep_flow(
    p_rw: csr_matrix,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    k_max: int = 3,
    pool: str = "max",
) -> float:
    """Random-walk k-step set-to-set flow used as an affinity-style route score."""
    source_idx = np.asarray(source_idx, dtype=int)
    target_idx = np.asarray(target_idx, dtype=int)
    if source_idx.size == 0 or target_idx.size == 0:
        return 0.0

    n = p_rw.shape[0]
    mass = np.zeros(n, dtype=float)
    mass[source_idx] = 1.0 / len(source_idx)
    target_mask = np.zeros(n, dtype=bool)
    target_mask[target_idx] = True

    vals: List[float] = []
    current = mass.copy()
    for _ in range(1, int(k_max) + 1):
        current = current @ p_rw
        vals.append(float(current[target_mask].sum()))

    if pool == "sum":
        return float(sum(vals))
    return float(max(vals)) if vals else 0.0


def affinity_path_score(
    p_rw: csr_matrix,
    start_idx: np.ndarray,
    waypoint_idx: List[np.ndarray],
    final_idx: np.ndarray,
    k_max: int = 3,
    pool: str = "max",
) -> float:
    """Product of set-to-set k-step flows along a route."""
    if len(final_idx) == 0:
        return 0.0

    score = 1.0
    previous = start_idx
    for target in waypoint_idx + [final_idx]:
        step = set_to_set_kstep_flow(p_rw, previous, target, k_max=k_max, pool=pool)
        score *= step
        previous = target
        if score == 0.0:
            break
    return float(score)


def _absorption_breakdown(
    p: csr_matrix,
    start_mass: np.ndarray,
    sets: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Absorption probabilities into named sets from an arbitrary start mass.

    Used for:
      - first_stv_hit_probs.csv
      - two_stage_unconditional_paths.csv
    """
    if len(sets) == 0:
        return {}, np.array([], dtype=int), np.array([], dtype=float)

    start_mass = np.asarray(start_mass, dtype=float).copy()
    if start_mass.sum() > 0:
        start_mass /= float(start_mass.sum())

    finals_list = [np.asarray(v, dtype=int) for v in sets.values() if len(v) > 0]
    if not finals_list:
        return {str(k): 0.0 for k in sets.keys()}, np.array([], dtype=int), np.array([], dtype=float)

    finals = np.unique(np.concatenate(finals_list))
    absorbing_idx = augment_absorbing_with_sinks(p, finals, start_support=np.where(start_mass > 0)[0])
    q, r, order, _, transient_mask, _ = absorbing_blocks(p, absorbing_idx)
    t = int(transient_mask.sum())

    if t > 0:
        nr = _solve_absorption_system(q, r)
        start_perm = start_mass[order]
        b_cols = start_perm[:t] @ nr + start_perm[t:]
    else:
        start_perm = start_mass[order]
        b_cols = start_perm

    absorbing_nodes_in_perm = order[t:]
    per_set: Dict[str, float] = {}
    for name, idxs in sets.items():
        idxs = np.asarray(idxs, dtype=int)
        mask_cols = np.isin(absorbing_nodes_in_perm, idxs)
        per_set[str(name)] = float(b_cols[mask_cols].sum()) if mask_cols.any() else 0.0

    return per_set, absorbing_nodes_in_perm, b_cols


def default_route_list(route_start_cnt: str, stv_cats: List[str]) -> List[str]:
    """Return the original route list, with the start CNT replaced by route_start_cnt."""
    route_start_cnt = str(route_start_cnt)
    stv_set = {str(s) for s in stv_cats}
    original_tails = [
        ["5"], ["6"], ["5", "6"], ["6", "5"],
        ["5", "8"], ["5", "7"], ["6", "7"], ["6", "8"],
        ["6", "5", "7"], ["6", "5", "8"], ["5", "6", "7"], ["5", "6", "8"],
        ["7"], ["8"],
    ]

    if all(all(x in stv_set for x in tail) for tail in original_tails):
        return [route_start_cnt + ">" + ">".join(tail) for tail in original_tails]

    # Fallback for non-standard STV categories: direct + all ordered two-stage routes.
    routes = [route_start_cnt + ">" + str(s) for s in stv_cats]
    routes.extend(
        route_start_cnt + ">" + str(a) + ">" + str(b)
        for a in stv_cats
        for b in stv_cats
        if str(a) != str(b)
    )
    return routes


def plot_route_bars(metrics_df: pd.DataFrame, output_png: str, title: str = "Route probabilities (forward)") -> None:
    """Plot horizontal route-probability bars using friendly H/FH route labels."""
    if metrics_df.empty:
        print("Note: no route metrics available for route_bars.png.")
        return

    df_plot = metrics_df.sort_values("prob_forward", ascending=False).reset_index(drop=True)
    label_col = "route_label" if "route_label" in df_plot.columns else "route"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_plot[label_col], df_plot["prob_forward"])
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title(title)

    for i, prob in enumerate(df_plot["prob_forward"].astype(float).values):
        ax.text(prob + 1e-3, i, f"{prob:.3f}", va="center")

    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_compare_rw_aff(metrics_df: pd.DataFrame, output_png: str, output_png_log: str) -> None:
    """Optional route comparison plots: random-walk route probability vs affinity score."""
    if metrics_df.empty:
        return

    df_plot = metrics_df.copy().sort_values(["prob_rw", "affinity"], ascending=[False, False])
    label_col = "route_label" if "route_label" in df_plot.columns else "route"
    routes = df_plot[label_col].tolist()
    x = np.arange(len(routes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width / 4, df_plot["prob_rw"].values, width / 2, label="rwprob")
    bars2 = ax2.bar(x + width / 4, df_plot["affinity"].values, width / 2, label="affinity", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=30, ha="right")
    ax.set_ylabel("Probability (rwprob)")
    ax2.set_ylabel("Affinity (pooled multi-step)")
    ax.set_title("Route comparison: rwprob (probability) vs affinity (score)")
    ax.legend([bars1[0], bars2[0]], ["rwprob", "affinity"], loc="best")
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    eps = 1e-12
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, np.log10(np.maximum(df_plot["prob_rw"].values, eps)), width / 2, label="rwprob")
    ax.bar(x, np.log10(np.maximum(df_plot["affinity"].values, eps)), width / 2, label="affinity", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=30, ha="right")
    ax.set_ylabel("log10(value)")
    ax.set_title("Route comparison (log scale)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_png_log, dpi=220)
    plt.close(fig)


def run_route_analysis(
    graph: csr_matrix,
    df: pd.DataFrame,
    cnt_map: Dict[str, np.ndarray],
    stv_map: Dict[str, np.ndarray],
    route_start_cnt: str,
    stv_cats: List[str],
    output_dir: str,
    forward_tol: float,
    routes: Optional[List[str]] = None,
    affinity_k: int = 3,
    affinity_pool: str = "max",
) -> Dict[str, str]:
    """
    Add the route-test outputs CNT3->STV .

    This function is intentionally separate from the best-pair/pseudotime pipeline so
    route outputs do not change any existing best-pair, hub, sensitivity, or
    pseudotime files.
    """
    route_start_cnt = str(route_start_cnt)
    route_start_idx = cnt_map.get(route_start_cnt, np.array([], dtype=int))
    if route_start_idx.size == 0:
        print(f"No CNT cells found for route start CAT={route_start_cnt}; skipping route analysis.")
        return {}

    finals_order = [str(s) for s in stv_cats if str(s) in stv_map and stv_map[str(s)].size > 0]
    finals_dict = {s: stv_map[s] for s in finals_order}
    if not finals_dict:
        print("No STV/FH cells found for route analysis; skipping route analysis.")
        return {}

    pt_route = compute_pseudotime(graph, route_start_idx)
    p_fwd_route = build_forward_markov(graph, pt_route, tol=forward_tol)
    p_rw_markov = build_rw_markov(graph)

    n = df.shape[0]
    start_mass = np.zeros(n, dtype=float)
    start_mass[route_start_idx] = 1.0 / len(route_start_idx)

    # Baseline absorption / first STV hit from the route-start CNT.
    first_probs, abs_nodes_perm, b_cols_first = _absorption_breakdown(p_fwd_route, start_mass, finals_dict)

    absorption_rows = [
        {"final": cat, "final_label": friendly_stv(cat), "prob": float(first_probs.get(cat, 0.0))}
        for cat in finals_order
    ]
    absorption_path = os.path.join(output_dir, "absorption_to_finals.csv")
    pd.DataFrame(absorption_rows).to_csv(absorption_path, index=False)

    first_stv_path = os.path.join(output_dir, "first_stv_hit_probs.csv")
    pd.DataFrame(
        [
            {"CAT": cat, "CAT_label": friendly_stv(cat), "p_first": float(first_probs.get(cat, 0.0))}
            for cat in finals_order
        ]
    ) \
        .sort_values("p_first", ascending=False) \
        .to_csv(first_stv_path, index=False)

    # Entry distributions within each first-hit STV category.
    phi_map: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    for cat in finals_order:
        idxs = finals_dict[cat]
        mask_cols = np.isin(abs_nodes_perm, idxs)
        p_cat = float(b_cols_first[mask_cols].sum()) if len(b_cols_first) > 0 and mask_cols.any() else 0.0
        if p_cat > 0:
            nodes_cat = abs_nodes_perm[mask_cols]
            phi_cat = b_cols_first[mask_cols] / p_cat
            phi_map[cat] = (nodes_cat, phi_cat, p_cat)

    # True unconditional two-stage paths: p(first STV) * p(next STV | entry distribution in first STV).
    two_stage_rows: List[Dict[str, float]] = []
    for first_cat, (nodes_cat, phi_cat, p_first_cat) in phi_map.items():
        mass = np.zeros(n, dtype=float)
        mass[nodes_cat] = phi_cat

        other_sets = {cat: finals_dict[cat] for cat in finals_order if cat != first_cat}
        next_probs, _, _ = _absorption_breakdown(p_fwd_route, mass, other_sets)
        for next_cat, p_next_given_first in next_probs.items():
            two_stage_rows.append(
                {
                    "route": f"{route_start_cnt}>{first_cat}>{next_cat}",
                    "route_label": friendly_route(f"{route_start_cnt}>{first_cat}>{next_cat}"),
                    "start_cnt": route_start_cnt,
                    "start_cnt_label": friendly_cnt(route_start_cnt),
                    "first_stv": first_cat,
                    "first_stv_label": friendly_stv(first_cat),
                    "next_stv": next_cat,
                    "next_stv_label": friendly_stv(next_cat),
                    "p_unconditional": float(p_first_cat) * float(p_next_given_first),
                    "p_first": float(p_first_cat),
                    "p_next_given_first": float(p_next_given_first),
                }
            )

    two_stage_path = os.path.join(output_dir, "two_stage_unconditional_paths.csv")
    two_stage_df = pd.DataFrame(
        two_stage_rows,
        columns=[
            "route",
            "route_label",
            "start_cnt",
            "start_cnt_label",
            "first_stv",
            "first_stv_label",
            "next_stv",
            "next_stv_label",
            "p_unconditional",
            "p_first",
            "p_next_given_first",
        ],
    )
    if not two_stage_df.empty:
        two_stage_df = two_stage_df.sort_values("p_unconditional", ascending=False)
    two_stage_df.to_csv(two_stage_path, index=False)

    # Comparative route metrics and the route_bars.png plot.
    route_list = [r.strip() for r in (routes or default_route_list(route_start_cnt, finals_order)) if str(r).strip()]
    metrics_rows: List[Dict[str, float]] = []
    for route in route_list:
        parts = [p.strip() for p in route.split(">") if p.strip()]
        if len(parts) < 2:
            continue
        if parts[0] != route_start_cnt:
            raise ValueError(f"Route {route!r} must start at CNT:{route_start_cnt}.")

        waypoint_cats = parts[1:-1]
        final_cat = parts[-1]
        waypoint_idx = [stv_map.get(str(w), np.array([], dtype=int)) for w in waypoint_cats]
        final_idx = stv_map.get(str(final_cat), np.array([], dtype=int))

        if final_idx.size == 0:
            metrics_rows.append(
                {
                    "route": route,
                    "route_label": friendly_route(route),
                    "prob_forward": 0.0,
                    "prob_rw": 0.0,
                    "affinity": 0.0,
                }
            )
            continue

        forbid_cats = [cat for cat in finals_order if cat != str(final_cat)]
        forbid_idx = (
            np.unique(np.concatenate([stv_map[cat] for cat in forbid_cats]))
            if forbid_cats
            else np.array([], dtype=int)
        )

        p_forward = route_probability(p_fwd_route, route_start_idx, waypoint_idx, final_idx, forbid_idx)
        prob_rw = route_probability(p_rw_markov, route_start_idx, waypoint_idx, final_idx, forbid_idx)
        affinity = affinity_path_score(
            p_rw_markov,
            route_start_idx,
            waypoint_idx,
            final_idx,
            k_max=affinity_k,
            pool=affinity_pool,
        )
        metrics_rows.append(
            {
                "route": route,
                "route_label": friendly_route(route),
                "prob_forward": float(p_forward),
                "prob_rw": float(prob_rw),
                "affinity": float(affinity),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows, columns=["route", "route_label", "prob_forward", "prob_rw", "affinity"])
    metrics_df = metrics_df.sort_values("prob_forward", ascending=False) if not metrics_df.empty else metrics_df
    route_metrics_path = os.path.join(output_dir, "route_metrics.csv")
    metrics_df.to_csv(route_metrics_path, index=False)

    route_bars_path = os.path.join(output_dir, "route_bars.png")
    plot_route_bars(metrics_df, route_bars_path)

    # Only keep the route probability bar plot. 
    for old_plot in [
        "route_compare_rwprob_affinity.png",
        "route_compare_rwprob_affinity_log.png",
    ]:
        old_plot_path = os.path.join(output_dir, old_plot)
        if os.path.exists(old_plot_path):
            os.remove(old_plot_path)

    route_summary_path = os.path.join(output_dir, "route_summary.txt")
    with open(route_summary_path, "w", encoding="utf-8") as handle:
        handle.write(
            "Route analysis start CNT: "
            f"{route_start_cnt} ({friendly_cnt(route_start_cnt)})\n"
            "Direct routes are first-hit probabilities. Routes with waypoints are conditional waypoint route scores.\n"
            "Use two_stage_unconditional_paths.csv for true unconditional two-stage route probabilities.\n"
            "Only route_bars.png is generated for route plotting.\n"
        )

    return {
        "absorption_to_finals": absorption_path,
        "first_stv_hit_probs": first_stv_path,
        "two_stage_unconditional_paths": two_stage_path,
        "route_metrics": route_metrics_path,
        "route_bars": route_bars_path,
        "route_summary": route_summary_path,
    }

def run_root_sensitivity(
    graph: csr_matrix,
    df: pd.DataFrame,
    cnt_map: Dict[str, np.ndarray],
    stv_map: Dict[str, np.ndarray],
    cnt_cats: List[str],
    stv_cats: List[str],
    sensitivity_roots: List[str],
    forward_tol: float,
    primary_hub_cnt: Optional[str],
    primary_anchor_stv: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Re-run the full CNT->STV first-hit analysis using each requested CNT category
    as the pseudotime root.

    This checks whether the selected hub/anchor is robust or is mainly a result of
    the original H4-rooted direction. The best-pair plot itself is not changed by
    this analysis; it is diagnostic output only.
    """
    summary_rows: List[Dict[str, object]] = []
    all_pair_frames: List[pd.DataFrame] = []

    primary_hub_cnt = None if primary_hub_cnt is None else str(primary_hub_cnt)
    primary_anchor_stv = None if primary_anchor_stv is None else str(primary_anchor_stv)
    primary_pair_label = (
        f"{friendly_cnt(primary_hub_cnt)}{ARROW}{friendly_stv(primary_anchor_stv)}"
        if primary_hub_cnt is not None and primary_anchor_stv is not None
        else None
    )

    all_pairs_columns = [
        "sensitivity_root_cnt",
        "sensitivity_root_label",
        "rank_under_root",
        "start_cnt",
        "stv_cat",
        "prob",
        "start_cnt_label",
        "stv_label",
        "pair_label",
    ]

    for root_cnt in [str(r) for r in sensitivity_roots]:
        root_idx = cnt_map.get(root_cnt, np.array([], dtype=int))
        root_label = friendly_cnt(root_cnt)
        base: Dict[str, object] = {
            "sensitivity_root_cnt": root_cnt,
            "sensitivity_root_label": root_label,
            "n_root_cells": int(root_idx.size),
            "primary_pair_from_initial_analysis": primary_pair_label,
            "primary_hub_cnt": primary_hub_cnt,
            "primary_hub_label": friendly_cnt(primary_hub_cnt) if primary_hub_cnt is not None else None,
            "primary_anchor_stv": primary_anchor_stv,
            "primary_anchor_label": friendly_stv(primary_anchor_stv) if primary_anchor_stv is not None else None,
        }

        if root_idx.size == 0:
            summary_rows.append(
                {
                    **base,
                    "status": "no_root_cells",
                    "top_pair": None,
                    "top_pair_prob": np.nan,
                    "top_hub_cnt": None,
                    "top_hub_label": None,
                    "top_anchor_stv": None,
                    "top_anchor_label": None,
                    "root_equals_top_hub": False,
                    "top_pair_matches_primary_pair": False,
                    "top_hub_matches_primary_hub": False,
                    "top_anchor_matches_primary_anchor": False,
                    "primary_pair_prob_under_this_root": np.nan,
                    "primary_pair_rank_under_this_root": np.nan,
                }
            )
            continue

        try:
            pt_root = compute_pseudotime(graph, root_idx)
            p_root = build_forward_markov(graph, pt_root, tol=forward_tol)
            cnt2stv_root, _ = run_best_pair_analysis(
                p_fwd=p_root,
                df=df,
                cnt_map=cnt_map,
                stv_map=stv_map,
                cnt_cats=cnt_cats,
                stv_cats=stv_cats,
            )
        except Exception as exc:
            summary_rows.append(
                {
                    **base,
                    "status": f"error: {type(exc).__name__}: {exc}",
                    "top_pair": None,
                    "top_pair_prob": np.nan,
                    "top_hub_cnt": None,
                    "top_hub_label": None,
                    "top_anchor_stv": None,
                    "top_anchor_label": None,
                    "root_equals_top_hub": False,
                    "top_pair_matches_primary_pair": False,
                    "top_hub_matches_primary_hub": False,
                    "top_anchor_matches_primary_anchor": False,
                    "primary_pair_prob_under_this_root": np.nan,
                    "primary_pair_rank_under_this_root": np.nan,
                }
            )
            continue

        cnt2stv_root = cnt2stv_root.copy()
        cnt2stv_root.insert(0, "sensitivity_root_cnt", root_cnt)
        cnt2stv_root.insert(1, "sensitivity_root_label", root_label)
        cnt2stv_root.insert(2, "rank_under_root", np.arange(1, len(cnt2stv_root) + 1))
        all_pair_frames.append(cnt2stv_root)

        top_hub, top_anchor, top_prob = choose_cnt_hub_from_top_pair(cnt2stv_root)
        top_pair_label = (
            f"{friendly_cnt(top_hub)}{ARROW}{friendly_stv(top_anchor)}"
            if top_hub is not None and top_anchor is not None
            else None
        )

        primary_pair_prob = np.nan
        primary_pair_rank = np.nan
        if primary_hub_cnt is not None and primary_anchor_stv is not None:
            primary_match = cnt2stv_root[
                (cnt2stv_root["start_cnt"].astype(str) == primary_hub_cnt)
                & (cnt2stv_root["stv_cat"].astype(str) == primary_anchor_stv)
            ]
            if not primary_match.empty:
                primary_pair_prob = float(primary_match.iloc[0]["prob"])
                primary_pair_rank = int(primary_match.iloc[0]["rank_under_root"])

        summary_rows.append(
            {
                **base,
                "status": "ok",
                "top_pair": top_pair_label,
                "top_pair_prob": float(top_prob),
                "top_hub_cnt": top_hub,
                "top_hub_label": friendly_cnt(top_hub) if top_hub is not None else None,
                "top_anchor_stv": top_anchor,
                "top_anchor_label": friendly_stv(top_anchor) if top_anchor is not None else None,
                "root_equals_top_hub": bool(top_hub is not None and str(root_cnt) == str(top_hub)),
                "top_pair_matches_primary_pair": bool(
                    top_hub is not None
                    and top_anchor is not None
                    and primary_hub_cnt is not None
                    and primary_anchor_stv is not None
                    and str(top_hub) == primary_hub_cnt
                    and str(top_anchor) == primary_anchor_stv
                ),
                "top_hub_matches_primary_hub": bool(
                    top_hub is not None and primary_hub_cnt is not None and str(top_hub) == primary_hub_cnt
                ),
                "top_anchor_matches_primary_anchor": bool(
                    top_anchor is not None
                    and primary_anchor_stv is not None
                    and str(top_anchor) == primary_anchor_stv
                ),
                "primary_pair_prob_under_this_root": primary_pair_prob,
                "primary_pair_rank_under_this_root": primary_pair_rank,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    all_pairs_df = pd.concat(all_pair_frames, ignore_index=True) if all_pair_frames else pd.DataFrame(columns=all_pairs_columns)

    ok = summary_df[summary_df["status"] == "ok"].copy() if not summary_df.empty else pd.DataFrame()
    if ok.empty:
        consensus_df = pd.DataFrame(
            columns=[
                "top_pair",
                "top_hub_cnt",
                "top_hub_label",
                "top_anchor_stv",
                "top_anchor_label",
                "n_roots_supporting_pair",
                "roots_supporting_pair",
                "mean_top_pair_prob",
                "min_top_pair_prob",
                "max_top_pair_prob",
            ]
        )
    else:
        consensus_rows: List[Dict[str, object]] = []
        group_cols = ["top_pair", "top_hub_cnt", "top_hub_label", "top_anchor_stv", "top_anchor_label"]
        for key, sub in ok.groupby(group_cols, dropna=False, sort=False):
            roots = sub["sensitivity_root_label"].astype(str).tolist()
            probs = sub["top_pair_prob"].astype(float)
            consensus_rows.append(
                {
                    "top_pair": key[0],
                    "top_hub_cnt": key[1],
                    "top_hub_label": key[2],
                    "top_anchor_stv": key[3],
                    "top_anchor_label": key[4],
                    "n_roots_supporting_pair": int(len(sub)),
                    "roots_supporting_pair": ";".join(roots),
                    "mean_top_pair_prob": float(probs.mean()),
                    "min_top_pair_prob": float(probs.min()),
                    "max_top_pair_prob": float(probs.max()),
                }
            )
        consensus_df = pd.DataFrame(consensus_rows).sort_values(
            ["n_roots_supporting_pair", "mean_top_pair_prob"], ascending=[False, False]
        ).reset_index(drop=True)

    return summary_df, all_pairs_df, consensus_df

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Control (CNT)-> Fatsed (STV) best pairs, stacked plot, hub/anchor choice, final pseudotime CSV, and optional route tests."
    )
    ap.add_argument("--input", default=r"...add path...\Fig5_Perturbation_Data_Matrix.csv")  #<<<<--------ADD PATH HERE
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--label-col", default="labels")
    ap.add_argument("--cat-col", default="CAT")

    ap.add_argument("--n-components", type=int, default=20)
    ap.add_argument("--graph-pcs", type=int, default=20)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--graph-metric", choices=["euclidean", "cosine"], default="euclidean")
    ap.add_argument("--forward-tol", type=float, default=1e-3)

    ap.add_argument(
        "--initial-root-cnt",
        "--pseudotime_root_cnt",
        dest="initial_root_cnt",
        default="3",
        help="CNT CAT used for the first pass that reproduces the original best-pair plot. Default: 3 (H4).",
    )
    ap.add_argument(
        "--final-root-mode",
        choices=["hub", "initial"],
        default="hub",
        help="Root used for pseudotime.csv. 'hub' uses the best-pair hub; 'initial' preserves the initial root.",
    )

    ap.add_argument("--cnt-cats", nargs="*", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--stv-cats", nargs="*", default=["5", "6", "7", "8"])
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument(
        "--force-include-stv",
        default="7",
        help="Append the best pair for this STV category if absent from topk. Use empty string to disable. Default: 7 (FH3).",
    )
    ap.add_argument(
        "--plot-rank-by",
        choices=["prob", "cnt_only_total"],
        default="prob",
        help="Metric used to select top-k plot rows. Default keeps the original absolute-probability ranking.",
    )

    ap.add_argument("--plot-width", type=float, default=16.0)
    ap.add_argument("--plot-height", type=float, default=9.21)
    ap.add_argument("--plot-dpi", type=int, default=128)
    ap.add_argument(
        "--skip-root-sensitivity",
        action="store_true",
        help="Skip the all-roots diagnostic. By default, the script tests each CNT root H1-H5.",
    )
    ap.add_argument(
        "--sensitivity-roots",
        nargs="*",
        default=None,
        help="CNT CAT values to use for root-sensitivity diagnostics. Default: all --cnt-cats.",
    )

    ap.add_argument(
        "--skip-pca-pseudotime-output",
        action="store_true",
        help=("Skip CNT/STV pseudotime display outputs, including the filtered "
              "H4/FH remodeling-input CSV."),
    )
    ap.add_argument(
        "--pca-plot-groups",
        nargs="*",
        default=["CNT", "STV"],
        help="Groups considered for the CNT/STV pseudotime display outputs. Default: CNT STV.",
    )
    ap.add_argument(
        "--pca-pseudotime-analysis-groups",
        nargs="*",
        default=["CNT", "STV"],
        help=(
            "Groups used to recompute the PCA/graph pseudotime for the PCA color plot only. "
            "Default: CNT STV. This does not affect best-pair, route, or sensitivity bar graphs."
        ),
    )
    ap.add_argument(
        "--label-pca-centroids",
        action="store_true",
        help="Add H/FH text labels above PCA centroid dots. Default is off, matching the standalone plotting script.",
    )
    ap.add_argument(
        "--skip-filtered-pca-pseudotime-plot",
        action="store_true",
        help=("Skip the filtered/rescaled PCA display and the filtered H4/FH "
              "cells_with_pseudotime_H4_FH_only.csv export."),
    )
    ap.add_argument(
        "--pca-pseudotime-filter-upper-quantile",
        type=float,
        default=0.99,
        help="Upper quantile used to remove high-end pseudotime outliers from the filtered PCA display. Default: 0.99.",
    )
    ap.add_argument(
        "--pca-pseudotime-filter-max",
        type=float,
        default=0.999,
        help="Also remove cells with final pseudotime at/above this value in the filtered PCA display. Default: 0.999.",
    )

    ap.add_argument(
        "--skip-route-analysis",
        action="store_true",
        help="Skip selected-CNT route tests. By default, route_bars.png and two_stage_unconditional_paths.csv are created.",
    )
    ap.add_argument(
        "--route-start-mode",
        choices=["hub", "initial"],
        default="hub",
        help="CNT used for route tests when --route-start-cnt is not provided. Default: selected hub CNT.",
    )
    ap.add_argument(
        "--route-start-cnt",
        default=None,
        help="Optional CNT CAT override for route tests. If omitted, uses --route-start-mode.",
    )
    ap.add_argument(
        "--routes",
        nargs="*",
        default=None,
        help="Optional explicit route list, e.g. 3>6 3>5>6. If omitted, uses the original default route list with the route-start CNT.",
    )
    ap.add_argument(
        "--route-final-cats",
        nargs="*",
        default=None,
        help="STV CATs treated as route endpoints. Default: same as --stv-cats.",
    )
    ap.add_argument("--affinity-k", type=int, default=3)
    ap.add_argument("--affinity-pool", choices=["max", "sum"], default="max")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    in_path = os.path.abspath(args.input)
    in_dir = os.path.dirname(in_path)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = args.output_dir or os.path.join(in_dir, f"{stem}__best_pairs")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column {args.label_col!r} not found in input CSV.")
    if args.cat_col not in df.columns:
        raise ValueError(f"CAT column {args.cat_col!r} not found in input CSV.")

    df["group"] = df[args.label_col].apply(parse_group)
    df["CAT"] = df[args.cat_col].astype(str)

    feature_cols = find_feature_columns(df, args.label_col, extra_drop=[args.cat_col])
    features = df[feature_cols].copy().fillna(df[feature_cols].median(numeric_only=True))

    _, scores, _ = run_pca(features, args.n_components)
    n_kept = scores.shape[1]
    if n_kept < 3:
        raise RuntimeError("Need at least 3 PCs for the graph step.")

    graph_pcs = min(args.graph_pcs, n_kept)
    graph = build_knn_graph(scores[:, :graph_pcs], k=args.knn, metric=args.graph_metric)

    cnt_cats = [str(c) for c in args.cnt_cats]
    stv_cats = [str(s) for s in args.stv_cats]
    cnt_map, stv_map = build_index_maps(df, cnt_cats, stv_cats)

    initial_root_cnt = str(args.initial_root_cnt)
    initial_roots = cnt_map.get(initial_root_cnt, np.array([], dtype=int))
    if initial_roots.size == 0:
        raise RuntimeError(f"No CNT cells found for initial root CAT={initial_root_cnt} ({friendly_cnt(initial_root_cnt)}).")

    # Pass 1: reproduce original best-pair/plot logic.
    pt_initial = compute_pseudotime(graph, initial_roots)
    p_initial = build_forward_markov(graph, pt_initial, tol=args.forward_tol)
    cnt2stv, breakdown_df = run_best_pair_analysis(
        p_fwd=p_initial,
        df=df,
        cnt_map=cnt_map,
        stv_map=stv_map,
        cnt_cats=cnt_cats,
        stv_cats=stv_cats,
    )

    cnt2stv.to_csv(os.path.join(out_dir, "best_cnt_to_stv_pairs_ranked.csv"), index=False)
    breakdown_df.to_csv(os.path.join(out_dir, "best_cnt_to_stv_lastcnt_breakdown.csv"), index=False)

    force_include_stv = args.force_include_stv if str(args.force_include_stv).strip() else None
    plot_df = build_plot_data(
        cnt2stv=cnt2stv,
        breakdown_df=breakdown_df,
        cnt_cats=cnt_cats,
        topk=args.topk,
        force_include_stv=force_include_stv,
        rank_by=args.plot_rank_by,
    )
    plot_df.to_csv(os.path.join(out_dir, "best_pairs_topk_plot_data.csv"), index=False)
    plot_best_pairs_stacked(
        plot_df=plot_df,
        cnt_cats=cnt_cats,
        output_png=os.path.join(out_dir, "best_pairs_topk_bars.png"),
        plot_width=args.plot_width,
        plot_height=args.plot_height,
        plot_dpi=args.plot_dpi,
    )

    # Hub/anchor selection from the same ranked best-pair table used by the original code.
    hub_cnt, hub_anchor_stv, hub_anchor_prob = choose_cnt_hub_from_top_pair(cnt2stv)

    if args.final_root_mode == "hub" and hub_cnt is not None and cnt_map.get(hub_cnt, np.array([], dtype=int)).size > 0:
        final_root_cnt = hub_cnt
    else:
        final_root_cnt = initial_root_cnt

    final_roots = cnt_map.get(str(final_root_cnt), np.array([], dtype=int))
    if final_roots.size == 0:
        raise RuntimeError(f"No CNT cells found for final root CAT={final_root_cnt} ({friendly_cnt(final_root_cnt)}).")

    # Pass 2: final pseudotime root informed by the selected hub by default.
    pt_final = compute_pseudotime(graph, final_roots)

    hub_choice = pd.DataFrame(
        [
            {
                "hub_select_method": "top_pair_by_absolute_probability",
                "initial_root_cnt_for_best_pair_plot": initial_root_cnt,
                "initial_root_label_for_best_pair_plot": friendly_cnt(initial_root_cnt),
                "hub_cnt": hub_cnt,
                "hub_cnt_label": friendly_cnt(hub_cnt) if hub_cnt is not None else None,
                "hub_anchor_stv": hub_anchor_stv,
                "hub_anchor_stv_label": friendly_stv(hub_anchor_stv) if hub_anchor_stv is not None else None,
                "hub_anchor_pair_prob": hub_anchor_prob,
                "final_pseudotime_root_mode": args.final_root_mode,
                "final_pseudotime_root_cnt": final_root_cnt,
                "final_pseudotime_root_label": friendly_cnt(final_root_cnt),
            }
        ]
    )
    hub_choice.to_csv(os.path.join(out_dir, "hub_choice.csv"), index=False)

    # Diagnostic pass: repeat the best-pair analysis with each CNT category as the
    # initial pseudotime root. This does not change the best-pair plot; it tests
    # whether the selected hub/anchor is stable across rooting choices.
    if not args.skip_root_sensitivity:
        sensitivity_roots = (
            [str(r) for r in args.sensitivity_roots]
            if args.sensitivity_roots is not None and len(args.sensitivity_roots) > 0
            else cnt_cats
        )
        sensitivity_summary, sensitivity_all_pairs, sensitivity_consensus = run_root_sensitivity(
            graph=graph,
            df=df,
            cnt_map=cnt_map,
            stv_map=stv_map,
            cnt_cats=cnt_cats,
            stv_cats=stv_cats,
            sensitivity_roots=sensitivity_roots,
            forward_tol=args.forward_tol,
            primary_hub_cnt=hub_cnt,
            primary_anchor_stv=hub_anchor_stv,
        )
        sensitivity_summary.to_csv(os.path.join(out_dir, "root_sensitivity_summary.csv"), index=False)
        sensitivity_all_pairs.to_csv(os.path.join(out_dir, "root_sensitivity_all_pairs.csv"), index=False)
        sensitivity_consensus.to_csv(os.path.join(out_dir, "root_sensitivity_consensus.csv"), index=False)

    pseudotime_df = make_pseudotime_table(
        df=df,
        label_col=args.label_col,
        pseudotime=pt_final,
        final_root_cnt=final_root_cnt,
        hub_anchor_stv=hub_anchor_stv,
        initial_root_cnt=initial_root_cnt,
    )
    pseudotime_df.to_csv(os.path.join(out_dir, "pseudotime.csv"), index=False)

    pca_pseudotime_outputs: Dict[str, str] = {}
    if not args.skip_pca_pseudotime_output:
        pca_pseudotime_outputs = save_cnt_stv_only_pca_pseudotime_outputs(
            df=df,
            label_col=args.label_col,
            feature_cols=feature_cols,
            output_dir=out_dir,
            final_root_cnt=final_root_cnt,
            hub_anchor_stv=hub_anchor_stv,
            initial_root_cnt=initial_root_cnt,
            n_components=args.n_components,
            graph_pcs=args.graph_pcs,
            knn=args.knn,
            graph_metric=args.graph_metric,
            all_group_pseudotime=pt_final,
            analysis_groups=args.pca_pseudotime_analysis_groups,
            plot_groups=args.pca_plot_groups,
            label_centroids=args.label_pca_centroids,
            make_filtered_plot=not args.skip_filtered_pca_pseudotime_plot,
            filter_upper_quantile=args.pca_pseudotime_filter_upper_quantile,
            filter_max_pseudotime=args.pca_pseudotime_filter_max,
        )

    pd.DataFrame({"feature_column": feature_cols}).to_csv(os.path.join(out_dir, "feature_columns_used.csv"), index=False)

    route_outputs: Dict[str, str] = {}
    if not args.skip_route_analysis:
        if args.route_start_cnt is not None and str(args.route_start_cnt).strip() != "":
            route_start_cnt = str(args.route_start_cnt)
        elif args.route_start_mode == "hub" and hub_cnt is not None:
            route_start_cnt = str(hub_cnt)
        else:
            route_start_cnt = str(initial_root_cnt)

        route_final_cats = (
            [str(s) for s in args.route_final_cats]
            if args.route_final_cats is not None and len(args.route_final_cats) > 0
            else stv_cats
        )

        try:
            route_outputs = run_route_analysis(
                graph=graph,
                df=df,
                cnt_map=cnt_map,
                stv_map=stv_map,
                route_start_cnt=route_start_cnt,
                stv_cats=route_final_cats,
                output_dir=out_dir,
                forward_tol=args.forward_tol,
                routes=args.routes,
                affinity_k=args.affinity_k,
                affinity_pool=args.affinity_pool,
            )
        except Exception as exc:
            error_path = os.path.join(out_dir, "route_analysis_error.txt")
            with open(error_path, "w", encoding="utf-8") as handle:
                handle.write(f"Route analysis failed: {type(exc).__name__}: {exc}\n")
            print(f"Route analysis failed but existing outputs were kept. Details: {error_path}")

    print("Saved to:", os.path.abspath(out_dir))
    print("Best-pair plot:", os.path.join(os.path.abspath(out_dir), "best_pairs_topk_bars.png"))
    print("Pseudotime CSV:", os.path.join(os.path.abspath(out_dir), "pseudotime.csv"))
    if pca_pseudotime_outputs:
        print("PCA + pseudotime CSV:", pca_pseudotime_outputs.get("cells_with_pca_and_pseudotime"))
        print("PCA + CNT/STV-only pseudotime CSV:", pca_pseudotime_outputs.get("pseudotime_cnt_stv_only"))
        print("Filtered/rescaled H4 + FH CSV:", pca_pseudotime_outputs.get("cells_with_pseudotime_H4_FH_only"))
        print("PCA + filtered CNT/STV-only final pseudotime plot:", pca_pseudotime_outputs.get("pca_projection_with_final_pseudotime_filtered"))
        print("PCA + signed control-to-FH display pseudotime plot:", pca_pseudotime_outputs.get("pca_projection_with_signed_control_to_fh_pseudotime"))
        print("PCA pseudotime filter summary:", pca_pseudotime_outputs.get("pca_pseudotime_filter_summary"))
        print("PCA signed display summary:", pca_pseudotime_outputs.get("pca_signed_control_to_fh_pseudotime_summary"))
    if route_outputs:
        print("Route bars:", route_outputs.get("route_bars"))
        print("Two-stage unconditional routes:", route_outputs.get("two_stage_unconditional_paths"))
    if not args.skip_root_sensitivity:
        print("Root-sensitivity summary:", os.path.join(os.path.abspath(out_dir), "root_sensitivity_summary.csv"))
    print(
        "Hub/anchor:",
        f"{friendly_cnt(hub_cnt)}{ARROW}{friendly_stv(hub_anchor_stv)}" if hub_cnt is not None else "not selected",
    )


if __name__ == "__main__":
    main()
