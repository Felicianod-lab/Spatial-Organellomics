import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================
# Configuration
# ==============================

DATA_PATH = r"...add path...\Fig3_Spatial_Analysis_Matrix.csv"
N_AXIS_BINS = 60
OUT_DIR_NAME = "Spatial_boundary_results"

# Mapping: CAT code -> H label
CAT_TO_H = {
    1: "H1",
    2: "H2",
    3: "H3",
    4: "H4",
    5: "H5",
}

H_LEVELS = ["H1", "H2", "H3", "H4", "H5"]

# Color map: pick specific entries from tab20b for each H label
CMAP = plt.get_cmap("tab20b")
H_COLOR_INDICES = {
    "H1": 0,
    "H2": 4,
    "H3": 9,
    "H4": 14,
    "H5": 19,
}
H_COLORS = {h: CMAP(idx) for h, idx in H_COLOR_INDICES.items()}


# ==============================
# Helpers
# ==============================

def extract_acinus_id(label: str):
    """
    Extract acinus ID from a label string.

    Examples
    --------
    'CNT_L1L1a0s0_1'  -> 'L1L1a0'
    'CNT_L2L3a4s2_13' -> 'L2L3a4'
    """
    label = str(label)
    parts = label.split("_", 1)
    if len(parts) < 2:
        return None

    tail = parts[1]

    # Preferred: cut at the first 's'
    s_idx = tail.find("s")
    if s_idx > 0:
        return tail[:s_idx]

    # Fallback: regex on the tail
    match = re.search(r"L\d+L\d+a\d+", tail)
    if match:
        return match.group(0)

    return None


def add_acinus_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'acinus_id' column based on the 'labels' column."""
    df = df.copy()
    df["acinus_id"] = df["labels"].astype(str).apply(extract_acinus_id)
    return df


def calculate_entropy(proportions: np.ndarray) -> np.ndarray:
    """
    Calculate Shannon entropy for each row of a proportions matrix.

    Parameters
    ----------
    proportions:
        Array with shape (n_bins, n_categories). Each row should sum to 1,
        except empty bins can be all zeros.
    """
    p_safe = np.where(proportions > 0, proportions, 1.0)
    return -(proportions * np.log(p_safe)).sum(axis=1)


def make_position_bins(values: pd.Series, n_bins: int) -> np.ndarray:
    """Create global bins from the min and max of ascini_position."""
    min_pos = float(values.min())
    max_pos = float(values.max())

    if np.isclose(min_pos, max_pos):
        # Avoid duplicate bin edges if all cells have the same position.
        min_pos -= 0.5
        max_pos += 0.5

    return np.linspace(min_pos, max_pos, n_bins + 1)


def bin_dataframe(df: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    """Return a copy of df with a 'bin_index' column."""
    df_binned = df.copy()
    df_binned["bin_index"] = pd.cut(
        df_binned["ascini_position"],
        bins=bins,
        include_lowest=True,
        labels=False,
    )
    return df_binned


def compute_proportions_by_bin(
    df_sub: pd.DataFrame,
    bins: np.ndarray,
    bin_centers: np.ndarray,
) -> np.ndarray:
    """
    Compute H1-H5 proportions per position bin for a dataframe.

    Returns
    -------
    proportions:
        Matrix with shape (n_bins, 5), where rows are position bins and
        columns are H1-H5 in H_LEVELS order.
    """
    n_bins = len(bin_centers)
    if df_sub.empty:
        return np.zeros((n_bins, len(H_LEVELS)), dtype=float)

    df_binned = bin_dataframe(df_sub, bins)

    counts = (
        df_binned
        .groupby(["bin_index", "H_label"])
        .size()
        .reset_index(name="count")
    )

    if counts.empty:
        return np.zeros((n_bins, len(H_LEVELS)), dtype=float)

    counts["prop"] = counts["count"] / counts.groupby("bin_index")["count"].transform("sum")

    pivot = (
        counts
        .pivot(index="bin_index", columns="H_label", values="prop")
        .reindex(index=range(n_bins), columns=H_LEVELS)
        .fillna(0.0)
    )

    return pivot.to_numpy()


def nansem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """SEM along an axis, ignoring NaNs and matching the original ddof=0 behavior."""
    n_eff = np.sum(np.isfinite(values), axis=axis)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(values, axis=axis)

    sem = np.full_like(std, np.nan, dtype=float)
    np.divide(std, np.sqrt(n_eff), out=sem, where=n_eff > 0)
    return sem


def compute_global_and_sem_stats(df: pd.DataFrame, bins: np.ndarray):
    """
    Compute the values needed for the three global outputs:

    1. pooled global H1-H5 proportions per position bin
    2. SEM of H1-H5 proportions across acini per position bin
    3. pooled global entropy and SEM of entropy across acini per position bin
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    n_bins = len(bin_centers)
    n_h = len(H_LEVELS)

    global_props = compute_proportions_by_bin(df, bins, bin_centers)
    if np.all(global_props == 0):
        return None, None, None, None, None

    global_entropy = calculate_entropy(global_props)

    acini = sorted(df["acinus_id"].dropna().unique())
    prop_acini = np.full((len(acini), n_bins, n_h), np.nan, dtype=float)
    entropy_acini = np.full((len(acini), n_bins), np.nan, dtype=float)

    for i, acinus_id in enumerate(acini):
        df_ac = df[df["acinus_id"] == acinus_id]
        df_ac_binned = bin_dataframe(df_ac, bins)

        for b in range(n_bins):
            df_bin = df_ac_binned[df_ac_binned["bin_index"] == b]
            if df_bin.empty:
                continue

            counts = df_bin["H_label"].value_counts()
            total = float(counts.sum())
            if total <= 0:
                continue

            p_vec = np.array([counts.get(h, 0) / total for h in H_LEVELS], dtype=float)
            prop_acini[i, b, :] = p_vec
            entropy_acini[i, b] = calculate_entropy(p_vec.reshape(1, -1))[0]

    sem_props = nansem(prop_acini, axis=0)
    sem_entropy = nansem(entropy_acini, axis=0)

    return bin_centers, global_props, sem_props, global_entropy, sem_entropy


# ==============================
# Plotting
# ==============================

def plot_stacked_frequency(
    x: np.ndarray,
    proportions: np.ndarray,
    out_dir: str,
    filename: str,
    title: str,
):
    """Save one stacked H1-H5 frequency-vs-position bar plot."""
    os.makedirs(out_dir, exist_ok=True)

    if len(x) == 0:
        return

    width = (x.max() - x.min()) / (len(x) * 1.2) if len(x) > 1 else 0.1

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros_like(x, dtype=float)

    for h_idx, h in enumerate(H_LEVELS):
        vals = proportions[:, h_idx]
        ax.bar(x, vals, width=width, bottom=bottom, label=h, color=H_COLORS[h])
        bottom += vals

    ax.set_xlabel("ascini_position (CV -> PV)")
    ax.set_ylabel("Frequency (fraction of cells)")
    ax.set_title(title)
    ax.legend(title="Category")
    ax.invert_xaxis()

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved stacked frequency plot to: {out_path}")


def plot_global_curves_with_sem(
    x: np.ndarray,
    global_props: np.ndarray,
    sem_props: np.ndarray,
    global_entropy: np.ndarray,
    sem_entropy: np.ndarray,
    out_dir: str,
):
    """
    Save the two global SEM plots:

    1. H1-H5 frequency vs position with SEM shading
    2. Shannon entropy vs position with SEM shading
    """
    os.makedirs(out_dir, exist_ok=True)

    # H1-H5 frequency curves with SEM
    fig, ax = plt.subplots(figsize=(9, 5))

    for h_idx, h in enumerate(H_LEVELS):
        mean_vals = global_props[:, h_idx]
        sem_vals = sem_props[:, h_idx]
        valid = np.isfinite(mean_vals) & np.isfinite(sem_vals)

        if not np.any(valid):
            continue

        x_v = x[valid]
        mean_v = mean_vals[valid]
        sem_v = sem_vals[valid]

        ax.plot(x_v, mean_v, label=h, color=H_COLORS[h])
        ax.fill_between(
            x_v,
            mean_v - sem_v,
            mean_v + sem_v,
            alpha=0.2,
            color=H_COLORS[h],
        )

    ax.set_xlabel("ascini_position (CV -> PV)")
    ax.set_ylabel("Frequency (fraction of cells)")
    ax.set_title("H1-H5 frequency vs position (global +/- SEM across acini)")
    ax.legend(title="Category")
    ax.invert_xaxis()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "H1_H5_frequency_vs_position_global_sem.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved global frequency SEM curves to: {out_path}")

    # Entropy curve with SEM
    fig, ax = plt.subplots(figsize=(9, 5))

    valid = np.isfinite(global_entropy) & np.isfinite(sem_entropy)
    if np.any(valid):
        x_v = x[valid]
        entropy_v = global_entropy[valid]
        sem_v = sem_entropy[valid]

        ax.plot(x_v, entropy_v, color="black")
        ax.fill_between(x_v, entropy_v - sem_v, entropy_v + sem_v, alpha=0.2, color="gray")

    ax.set_xlabel("ascini_position (CV -> PV)")
    ax.set_ylabel("Shannon entropy")
    ax.set_title("Heterogeneity vs position (global +/- SEM across acini)")
    ax.invert_xaxis()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "entropy_vs_position_global_sem.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved global entropy SEM curve to: {out_path}")


# ==============================
# Main workflow
# ==============================

def load_and_prepare_data() -> pd.DataFrame:
    """Load the CSV and prepare H labels plus acinus IDs."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_cols = ["labels", "CAT", "ascini_position"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[df["CAT"].isin(CAT_TO_H.keys())].copy()
    if df.empty:
        raise ValueError("No rows found with CAT in 1-5. Check your CSV.")

    df["CAT"] = df["CAT"].astype(int)
    df["ascini_position"] = df["ascini_position"].astype(float)
    df["H_label"] = df["CAT"].map(CAT_TO_H)

    before = len(df)
    df = add_acinus_id(df)
    df = df[df["acinus_id"].notna()].copy()
    after = len(df)

    print(f"Dropped {before - after} cells with unparseable acinus_id; {after} cells remain.")
    return df


def main():
    df = load_and_prepare_data()

    base_dir = os.path.dirname(DATA_PATH)
    results_dir = os.path.join(base_dir, OUT_DIR_NAME)
    os.makedirs(results_dir, exist_ok=True)

    bins = make_position_bins(df["ascini_position"], N_AXIS_BINS)

    print("Computing global plots...")
    bin_centers, global_props, sem_props, global_entropy, sem_entropy = compute_global_and_sem_stats(
        df,
        bins,
    )

    if global_props is None:
        print("Not enough data for global plots.")
        return

    # Global output 1: stacked frequency bars
    plot_stacked_frequency(
        x=bin_centers,
        proportions=global_props,
        out_dir=results_dir,
        filename="global_H1_H5_frequency_vs_position_stacked.png",
        title="Global H1-H5 frequency vs position along CV-PV axis",
    )

    # Global outputs 2 and 3: SEM curves
    plot_global_curves_with_sem(
        x=bin_centers,
        global_props=global_props,
        sem_props=sem_props,
        global_entropy=global_entropy,
        sem_entropy=sem_entropy,
        out_dir=results_dir,
    )

    # Per-acinus output: only the stacked frequency bar plot in each folder
    acini = sorted(df["acinus_id"].unique())
    print(f"\nFound {len(acini)} acini. Making one stacked plot per acinus...")

    for acinus_id in acini:
        df_ac = df[df["acinus_id"] == acinus_id].copy()
        ac_props = compute_proportions_by_bin(df_ac, bins, bin_centers)

        if np.all(ac_props == 0):
            print(f"Acinus {acinus_id}: all-zero proportions; skipping.")
            continue

        safe_ac = re.sub(r"[^A-Za-z0-9]+", "-", str(acinus_id))
        acinus_dir = os.path.join(results_dir, f"acinus_{safe_ac}")

        plot_stacked_frequency(
            x=bin_centers,
            proportions=ac_props,
            out_dir=acinus_dir,
            filename="H1_H5_frequency_vs_position_stacked.png",
            title=f"Acinus {acinus_id}: H1-H5 frequency vs position along CV-PV axis",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
