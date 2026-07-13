# ==========================================================
# CONTROL vs FASTED — TMRM  ANALYSIS
# Mixed model with liver + tile random effects
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import norm, levene

# ==========================================================
# LOAD DATA
# ==========================================================

control_path = r"...add path...\Fig5_IVM_Control_Group_Mitochondria_Properties.csv"
fasted_path  = r"...add path...\Fig5_IVM_Fasted_Group_Mitochondria_Properties.csv"

mito_control = pd.read_csv(control_path, low_memory=False)
mito_fasted  = pd.read_csv(fasted_path,  low_memory=False)

# ==========================================================
# PARSE LABELS
# Example: CL1TS1R2Z0_1 or F1TS1R1Z0_1
# ==========================================================

pattern = r'^(?P<liver_id>[^T]+)(?P<tile_section>TS\d+)(?P<region_id>R\d+)'

for d in [mito_control, mito_fasted]:
    parsed = d["cell_label"].astype(str).str.extract(pattern)
    d["liver_id"] = parsed["liver_id"]
    d["tile_section"] = parsed["tile_section"]
    d["region_id"] = parsed["region_id"]
    d["tile_key"] = d["liver_id"].astype(str) + "_" + d["tile_section"].astype(str)
    d["region_key"] = (
        d["liver_id"].astype(str) + "_" +
        d["tile_section"].astype(str) + "_" +
        d["region_id"].astype(str)
    )

# ==========================================================
# AGGREGATE MITOCHONDRIA TO CELL LEVEL
# ==========================================================

control_cells = (
    mito_control
    .dropna(subset=["cell_label", "PV_CV", "ratio_01_00", "liver_id", "tile_section"])
    .groupby("cell_label", observed=True)
    .agg(
        total_TMRM=("ratio_01_00", "mean"),
        PV_CV=("PV_CV", "first"),
        liver_id=("liver_id", "first"),
        tile_section=("tile_section", "first"),
        region_id=("region_id", "first"),
        tile_key=("tile_key", "first"),
        region_key=("region_key", "first")
    )
    .reset_index()
)

fasted_cells = (
    mito_fasted
    .dropna(subset=["cell_label", "PV_CV", "ratio_01_00", "liver_id", "tile_section"])
    .groupby("cell_label", observed=True)
    .agg(
        total_TMRM=("ratio_01_00", "mean"),
        PV_CV=("PV_CV", "first"),
        liver_id=("liver_id", "first"),
        tile_section=("tile_section", "first"),
        region_id=("region_id", "first"),
        tile_key=("tile_key", "first"),
        region_key=("region_key", "first")
    )
    .reset_index()
)

control_cells["Condition"] = "Control"
fasted_cells["Condition"]  = "Fasted"

df = pd.concat([control_cells, fasted_cells], ignore_index=True)

df["Condition"] = pd.Categorical(df["Condition"], categories=["Control", "Fasted"])
df["PV_CV"] = pd.Categorical(df["PV_CV"], categories=["PV", "CV"])

df = df.dropna(subset=["total_TMRM", "PV_CV", "Condition", "liver_id", "tile_key"]).copy()

print("Total cells:", df.shape[0])
print("Livers:", df["liver_id"].nunique())
print("Tiles:", df["tile_key"].nunique())
print("\nCells per group:")
print(df.groupby(["Condition", "PV_CV"], observed=True).size())

# ==========================================================
# MIXED MODEL WITH LIVER + TILE RANDOM EFFECTS
# ==========================================================

print("\n===================================================")
print("Mixed Model: total_TMRM ~ PV_CV * Condition")
print("Random effects: liver + tile")
print("===================================================")

model_tiles = smf.mixedlm(
    "total_TMRM ~ PV_CV * Condition",
    data=df,
    groups=df["liver_id"],
    vc_formula={"tile": "0 + C(tile_key)"}
).fit(reml=True, method="lbfgs")

print(model_tiles.summary())

# ==========================================================
# CONTRAST HELPERS
# ==========================================================

beta = model_tiles.fe_params
cov = model_tiles.cov_params().loc[beta.index, beta.index]
names = beta.index.tolist()
name_to_idx = {name: i for i, name in enumerate(names)}

def make_L(term_weights):
    L = np.zeros(len(names))
    for term, weight in term_weights.items():
        if term not in name_to_idx:
            raise KeyError(f"{term} not found. Available terms: {names}")
        L[name_to_idx[term]] = weight
    return L

def wald_test(L, label):
    L = np.asarray(L).reshape(1, -1)
    est = (L @ beta.values.reshape(-1, 1)).item()
    se = np.sqrt((L @ cov.values @ L.T)).item()
    z = est / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {
        "Comparison": label,
        "Estimate": est,
        "SE": se,
        "z": z,
        "p_value": p,
        "signif": sig_label(p)
    }

def sig_label(p):
    if pd.isna(p):
        return "NA"
    if p <= 0.0001:
        return "***"
    if p <= 0.05:
        return "**"
    return "N.S."

# ==========================================================
# PLANNED CONTRASTS FROM ONE MIXED MODEL
# ==========================================================

results = []

results.append(
    wald_test(
        make_L({"Condition[T.Fasted]": 1}),
        "PV: Fasted vs Control"
    )
)

results.append(
    wald_test(
        make_L({
            "Condition[T.Fasted]": 1,
            "PV_CV[T.CV]:Condition[T.Fasted]": 1
        }),
        "CV: Fasted vs Control"
    )
)

results.append(
    wald_test(
        make_L({"PV_CV[T.CV]": 1}),
        "Control: CV vs PV"
    )
)

results.append(
    wald_test(
        make_L({
            "PV_CV[T.CV]": 1,
            "PV_CV[T.CV]:Condition[T.Fasted]": 1
        }),
        "Fasted: CV vs PV"
    )
)

results_table = pd.DataFrame(results)

print("\n===================================================")
print("PLANNED CONTRASTS")
print("===================================================")
print(results_table)

# ==========================================================
# MIXED-MODEL GAP ANALYSIS
# ==========================================================

b_cv = beta["PV_CV[T.CV]"]
b_int = beta["PV_CV[T.CV]:Condition[T.Fasted]"]

control_gap = -b_cv
fasted_gap = -(b_cv + b_int)

abs_reduction = control_gap - fasted_gap
pct_reduction = abs_reduction / control_gap * 100
interaction_p = model_tiles.pvalues["PV_CV[T.CV]:Condition[T.Fasted]"]

gap_table = pd.DataFrame([{
    "Control_gap_PV_minus_CV": control_gap,
    "Fasted_gap_PV_minus_CV": fasted_gap,
    "Absolute_reduction": abs_reduction,
    "Percent_reduction": pct_reduction,
    "Interaction_p_value": interaction_p,
    "signif": sig_label(interaction_p)
}])

print("\n===================================================")
print("MIXED-MODEL GAP CHANGE")
print("===================================================")
print(gap_table)

# ==========================================================
# MODEL-PREDICTED MEANS ± 95% CI
# ==========================================================

def estimate_and_ci(L):
    L = np.asarray(L).reshape(1, -1)
    est = (L @ beta.values.reshape(-1, 1)).item()
    se = np.sqrt((L @ cov.values @ L.T)).item()
    ci95 = 1.96 * se
    return est, se, ci95

rows = []

est, se, ci = estimate_and_ci(make_L({"Intercept": 1}))
rows.append({"Condition": "Control", "PV_CV": "PV", "mean": est, "se": se, "ci95": ci})

est, se, ci = estimate_and_ci(make_L({"Intercept": 1, "PV_CV[T.CV]": 1}))
rows.append({"Condition": "Control", "PV_CV": "CV", "mean": est, "se": se, "ci95": ci})

est, se, ci = estimate_and_ci(make_L({"Intercept": 1, "Condition[T.Fasted]": 1}))
rows.append({"Condition": "Fasted", "PV_CV": "PV", "mean": est, "se": se, "ci95": ci})

est, se, ci = estimate_and_ci(make_L({
    "Intercept": 1,
    "PV_CV[T.CV]": 1,
    "Condition[T.Fasted]": 1,
    "PV_CV[T.CV]:Condition[T.Fasted]": 1
}))
rows.append({"Condition": "Fasted", "PV_CV": "CV", "mean": est, "se": se, "ci95": ci})

pred_summary = pd.DataFrame(rows)

print("\n===================================================")
print("MODEL-PREDICTED MEANS ± 95% CI")
print("===================================================")
print(pred_summary)

# ==========================================================
# PLOT 1 — MIXED-MODEL INTERACTION PLOT
# ==========================================================

plt.figure(figsize=(4,5))
plt.yticks(fontsize=20)

for zone in ["PV", "CV"]:
    sub = pred_summary[pred_summary["PV_CV"] == zone]
    plt.errorbar(
        sub["Condition"],
        sub["mean"],
        yerr=sub["ci95"],
        marker="o",
        capsize=4
    )

plt.tight_layout()
plt.show()

# ==========================================================
# PLOT 2 — MIXED-MODEL PV-CV GAP
# ==========================================================

gap_df = pred_summary.pivot(index="Condition", columns="PV_CV", values="mean")
gap_df["PV_minus_CV"] = gap_df["PV"] - gap_df["CV"]

print("\nMixed-model gap table:")
print(gap_df)

plt.figure(figsize=(4,5))
plt.yticks(fontsize=20)
plt.plot(gap_df.index, gap_df["PV_minus_CV"], marker="o")
plt.tight_layout()
plt.show()

# ==========================================================
# PLOT 3 — RAW VIOLIN + MODEL MEANS ± 95% CI
# ==========================================================

df["Group"] = df["PV_CV"].astype(str) + " - " + df["Condition"].astype(str)

order = [
    "PV - Control",
    "PV - Fasted",
    "CV - Control",
    "CV - Fasted"
]

palette = {
    "PV - Control": "#4C72B0",
    "PV - Fasted": "#4C72B0",
    "CV - Control": "#DD8452",
    "CV - Fasted": "#DD8452"
}

plt.figure(figsize=(4.5,6.5))
plt.xticks(fontsize=10)
plt.yticks(fontsize=28)

ax = sns.violinplot(
    data=df,
    x="Group",
    y="total_TMRM",
    order=order,
    hue="Group",
    palette=palette,
    inner="quartile",
    cut=0,
    linewidth=1,
    saturation=0.8,
    legend=False
)

for i, line in enumerate(ax.lines):
    if i % 3 == 1:
        line.set_color("black")
        line.set_linewidth(3)

sns.stripplot(
    data=df,
    x="Group",
    y="total_TMRM",
    order=order,
    color="black",
    size=0.8,
    alpha=0.3,
    jitter=0.25
)

overlay_map = {
    "PV - Control": ("Control", "PV"),
    "PV - Fasted": ("Fasted", "PV"),
    "CV - Control": ("Control", "CV"),
    "CV - Fasted": ("Fasted", "CV")
}

for i, grp in enumerate(order):
    cond, zone = overlay_map[grp]
    row = pred_summary[
        (pred_summary["Condition"] == cond) &
        (pred_summary["PV_CV"] == zone)
    ].iloc[0]

    plt.errorbar(
        i,
        row["mean"],
        yerr=row["ci95"],
        fmt="o",
        color="black",
        capsize=4,
        markersize=7,
        linewidth=1.5,
        zorder=10
    )

plt.xlabel("")
plt.ylim(0, 0.7)
plt.xticks(rotation=15)
plt.margins(x=0.20)
sns.despine()
plt.tight_layout()
plt.show()

# ==========================================================
# PLOT 4 — RAW KDE DISTRIBUTIONS
# ==========================================================

plt.figure(figsize=(7,5))

kde_palette = {
    ("PV", "Control"): "#1f77b4",
    ("PV", "Fasted"): "#ff7f0e",
    ("CV", "Control"): "#2ca02c",
    ("CV", "Fasted"): "#d62728"
}

for zone in ["PV", "CV"]:
    for cond in ["Control", "Fasted"]:
        subset = df[(df["PV_CV"] == zone) & (df["Condition"] == cond)]

        sns.kdeplot(
            subset["total_TMRM"],
            fill=True,
            alpha=0.3,
            linewidth=1.5,
            color=kde_palette[(zone, cond)],
            label=f"{zone} — {cond}",
            common_norm=False
        )

plt.xlabel("TMRM (ratio_01_00)")
plt.ylabel("Density")
plt.xlim(0, 0.65)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# ==========================================================
# VARIANCE TESTING — RAW CELL DISTRIBUTIONS
# Descriptive only; not the main inferential model
# ==========================================================

groups = [
    df[(df.Condition == "Control") & (df.PV_CV == "PV")]["total_TMRM"],
    df[(df.Condition == "Fasted")  & (df.PV_CV == "PV")]["total_TMRM"],
    df[(df.Condition == "Control") & (df.PV_CV == "CV")]["total_TMRM"],
    df[(df.Condition == "Fasted")  & (df.PV_CV == "CV")]["total_TMRM"]
]

pv_ctrl = groups[0]
pv_fast = groups[1]
cv_ctrl = groups[2]
cv_fast = groups[3]

stat_pv, p_pv = levene(pv_ctrl, pv_fast)
stat_cv, p_cv = levene(cv_ctrl, cv_fast)

print("\n===================================================")
print("Levene Variance Tests — raw cell distributions")
print("===================================================")
print(f"PV variance change p-value: {p_pv}")
print(f"CV variance change p-value: {p_cv}")

print("\n===================================================")
print("Descriptive variance summary")
print("===================================================")

stats_table = df.groupby(["Condition", "PV_CV"], observed=True)["total_TMRM"].agg(["mean", "std", "var"])
print(stats_table)

print("\n===================================================")
print("Coefficient of Variation")
print("===================================================")

g = df.groupby(["Condition", "PV_CV"], observed=True)["total_TMRM"]
cv_values = g.std() / g.mean()
print(cv_values)