# ==========================================================
# MINIMAL SCRIPT
# GMM on TMRM + State Distribution + Heterogeneity
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import scipy.stats as stats

# ==========================================================
# LOAD DATA
# ==========================================================


mito_c = pd.read_csv(
    r"...add path...\Fig5_IVM_Control_Group_Mitochondria_Properties.csv",
    low_memory=False
)

mito_f = pd.read_csv(
    r"...add path...\Fig5_IVM_Fasted_Group_Mitochondria_Properties.csv",
    low_memory=False
)

mito_c["Condition"] = "Control"
mito_f["Condition"] = "Fasted"

mito = pd.concat([mito_c, mito_f])
mito["liver_id"] = mito["cell_label"].str.split("TS").str[0]

# ==========================================================
# AGGREGATE TO CELL LEVEL
# ==========================================================

cell_df = (
    mito.groupby("cell_label")
    .agg(
        total_TMRM=("ratio_01_00", "mean"),
        PV_CV=("PV_CV", "first"),
        Condition=("Condition", "first"),
    )
)

print("Total cells:", cell_df.shape[0])

# ==========================================================
# GMM ON log(TMRM)
# ==========================================================

cell_df = cell_df[cell_df["total_TMRM"] > 0].copy()
cell_df["log_TMRM"] = np.log(cell_df["total_TMRM"])

X = cell_df[["log_TMRM"]].values

# Determine optimal k via BIC
bic_scores = []
models = []

for k in range(1, 5):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    models.append(gmm)

optimal_k = np.argmin(bic_scores) + 1
best_gmm = models[optimal_k - 1]

print("Optimal number of states:", optimal_k)

# Assign states
cell_df["oxphos_state"] = best_gmm.predict(X)

# ==========================================================
# PLOT GMM FIT
# ==========================================================

x_vals = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
pdf = np.exp(best_gmm.score_samples(x_vals))

plt.figure(figsize=(7,5))
sns.histplot(cell_df["log_TMRM"], bins=50, stat="density", alpha=0.5)
plt.plot(x_vals, pdf, linewidth=3)
plt.title("Gaussian Mixture Fit on log(TMRM)")
plt.xlabel("log(TMRM)")
plt.ylabel("Density")
plt.show()

# ==========================================================
# STACKED BAR: STATE DISTRIBUTION
# (Ordered, grouped, bone colormap, thin bars, spaced)
# ==========================================================

zone_order = ["PV", "CV"]
condition_order = ["Control", "Fasted"]

group_counts = (
    cell_df
    .groupby(["PV_CV", "Condition", "oxphos_state"])
    .size()
    .reset_index(name="count")
)

# Enforce ordering
group_counts["PV_CV"] = pd.Categorical(
    group_counts["PV_CV"],
    categories=zone_order,
    ordered=True
)

group_counts["Condition"] = pd.Categorical(
    group_counts["Condition"],
    categories=condition_order,
    ordered=True
)

# Convert to percentages
group_counts["percent"] = (
    group_counts["count"] /
    group_counts.groupby(["PV_CV", "Condition"])["count"].transform("sum")
) * 100

# Pivot table (Condition first so Control bars stay together)
pivot_df = (
    group_counts
    .pivot_table(index=["Condition", "PV_CV"],
                 columns="oxphos_state",
                 values="percent",
                 fill_value=0)
    .sort_index()
)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(6,3.5))

# ---- CONTROL SPACING HERE ----
x_positions = np.arange(len(pivot_df)) * 0.5   # ← Increase 1.4 for more spacing

x_labels = [f"{cond}-{zone}" for cond, zone in pivot_df.index]

bottom = np.zeros(len(pivot_df))


cmap = plt.cm.get_cmap("rocket_r", optimal_k)

for i, state in enumerate(sorted(pivot_df.columns)):
    
    values = pivot_df[state].values
    
    ax.bar(
        x_positions,
        values,
        bottom=bottom,
        label=f"State {state}",
        color=cmap(i),
        width=0.3,   # ← thinner bars
        # alpha=0.9   # ← decrease transparency here
    )
    
    bottom += values

# Axis formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=45)
ax.set_ylabel("Percentage of Cells")
ax.set_title("OxPhos State Distribution")
# ax.legend(title="State", bbox_to_anchor=(1.05,1))

plt.tight_layout()
ax.tick_params(axis='y', labelsize=20)
plt.show()

# ==========================================================
# HETEROGENEITY INDEX (ORDERED + GROUPED BY CONDITION)
# ==========================================================

def compute_entropy(states, k):
    probs = pd.Series(states).value_counts(normalize=True)
    probs = probs.reindex(range(k), fill_value=0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

H_max = np.log2(optimal_k)

entropy_results = []

for (zone, condition), group in cell_df.groupby(["PV_CV", "Condition"]):
    
    H = compute_entropy(group["oxphos_state"], optimal_k)
    H_norm = H / H_max
    
    entropy_results.append({
        "PV_CV": zone,
        "Condition": condition,
        "Entropy_normalized": H_norm
    })

entropy_df = pd.DataFrame(entropy_results)

# ----------------------------------------------------------
# Enforce desired order
# ----------------------------------------------------------

zone_order = ["PV", "CV"]
condition_order = ["Control", "Fasted"]

entropy_df["PV_CV"] = pd.Categorical(
    entropy_df["PV_CV"],
    categories=zone_order,
    ordered=True
)

entropy_df["Condition"] = pd.Categorical(
    entropy_df["Condition"],
    categories=condition_order,
    ordered=True
)

entropy_df = entropy_df.sort_values(["Condition", "PV_CV"])

print("\nNormalized Entropy (Ordered):")
print(entropy_df.round(3))

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(6,4))

x_labels = [
    f"{cond}-{zone}"
    for cond, zone in zip(entropy_df["Condition"], entropy_df["PV_CV"])
]

plt.bar(
    x_labels,
    entropy_df["Entropy_normalized"]
)

plt.axhline(1, linestyle="--", color="gray")
plt.ylim(0,1.05)
plt.ylabel("Normalized Entropy")
plt.title("Metabolic Heterogeneity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================================
# HETEROGENEITY INDEX — DOTTED LINE WITH POINTS
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

# Ensure correct ordering
zone_order = ["PV", "CV"]
condition_order = ["Control", "Fasted"]

entropy_df["PV_CV"] = pd.Categorical(
    entropy_df["PV_CV"],
    categories=zone_order,
    ordered=True
)

entropy_df["Condition"] = pd.Categorical(
    entropy_df["Condition"],
    categories=condition_order,
    ordered=True
)

entropy_df = entropy_df.sort_values(["Condition", "PV_CV"])

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(6,3))

# Control spacing here
x_positions = np.arange(len(entropy_df)) * 1.4

x_labels = [
    f"{cond}-{zone}"
    for cond, zone in zip(entropy_df["Condition"], entropy_df["PV_CV"])
]

y_values = entropy_df["Entropy_normalized"].values

# Dotted pink line with markers
ax.plot(
    x_positions,
    y_values,
    linestyle=":",
    marker="o",
    color="pink",
    linewidth=3,
    markersize=10,
    markerfacecolor="white",   # ← makes circle open
    markeredgecolor="black",   # ← outline color
    markeredgewidth=2         # ← thickness of outline
)

# Reference line at max entropy
ax.axhline(1, linestyle="--", color="gray", alpha=0.6)

# Formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=45)
ax.set_ylabel("Normalized Entropy (H / Hmax)")
ax.set_title("Metabolic Heterogeneity")

plt.tight_layout()
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', labelsize=12)  # X-axis tick size
ax.tick_params(axis='y', labelsize=20)  # Y-axis tick size
plt.show()