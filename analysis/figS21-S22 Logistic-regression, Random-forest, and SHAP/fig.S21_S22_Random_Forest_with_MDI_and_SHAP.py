# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:09:58 2026

@author: adhikarir
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:00:37 2026

@author: adhikarir
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:38:02 2026

@author: adhikarir

Updated: SHAP value analysis added for both Diet and Hepatocyte Category models.
         All plots and CSVs saved to output_dir.
"""

#%%
import os
import numpy as np
import pandas as pd
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# =============================================================================
# SET YOUR OUTPUT FOLDER HERE — all plots and CSVs will be saved here
# =============================================================================
output_dir = r"...add path...\Plots_and_Files"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# READABLE LABELS USED ONLY FOR PLOT TITLES, LEGENDS, AND FILENAMES
# =============================================================================
# Existing project convention for the diet/group labels:
#     1 = CNT, 2 = STV, 3 = WD
# The string versions are included so this also works when the CSV already
# stores the group names as "CNT", "STV", and "WD".
DIET_CLASS_NAME_MAP = {
    1: "CNT",
    2: "STV",
    3: "WD",
    "1": "CNT",
    "2": "STV",
    "3": "WD",
    "CNT": "CNT",
    "STV": "STV",
    "WD": "WD",
}

# Optional: add biological names here if the numbered hepatocyte categories
# later receive descriptive names. Example: {10: "Your category name"}
HEPATOCYTE_CATEGORY_NAME_MAP = {}


def normalize_class_value(raw_class):
    """Convert NumPy scalar labels to simple Python values for display."""
    if isinstance(raw_class, np.generic):
        raw_class = raw_class.item()
    if isinstance(raw_class, float) and raw_class.is_integer():
        raw_class = int(raw_class)
    return raw_class


def get_diet_name(raw_class):
    """Return CNT/STV/WD for a raw diet class label when the mapping is known."""
    raw_class = normalize_class_value(raw_class)
    if raw_class in DIET_CLASS_NAME_MAP:
        return DIET_CLASS_NAME_MAP[raw_class]
    raw_text = str(raw_class)
    return DIET_CLASS_NAME_MAP.get(raw_text, f"Diet label {raw_text}")


def get_hepatocyte_category_name(raw_class):
    """Return an unambiguous name for a hepatocyte category label."""
    raw_class = normalize_class_value(raw_class)
    custom_name = HEPATOCYTE_CATEGORY_NAME_MAP.get(raw_class)
    if custom_name:
        return f"Category {raw_class} - {custom_name}"
    return f"Category {raw_class}"


def filename_token(value):
    """Make a readable label safe to use inside a filename."""
    token = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(value)
    )
    return token.strip("_") or "unnamed"


def save_fig(filename):
    """Save current figure to output_dir and close it."""
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

#%%
# read the data
data = pd.read_csv(r"...add path...\figS21_S22_Experimental_Groups_Matrix.csv")
# FILTER: keep only rows where cell_id_linked != 0 (this is when motile can't find cells for organelles)
data = data[data["cell_id_linked"] != 0].reset_index(drop=True)

print("Remaining cells after filtering:", len(data))

# isolate labels
nutrient_labels = np.asarray(data["group"])
catagory_labels = np.asarray(data["Prediction"])

#%%
# select which columns to use in training
feature_columns = [
    # Mitochondria: general features
    "mito_density",
    "mito_avg_area",
    "mito_aspect_ratio",
    "mito_perimeter",
    "mito_percent_total_area",
    "mito_solidity",
    "mito_circularity",
    "mito_distance_from_edge",
    # Mitochondria: subtype 1
    "type_1_mito_density",
    "type_1_mito_avg_area",
    "type_1_mito_avg_aspect_ratio",
    "type_1_mito_perimeter",
    "type_1_mito_percent_total_area",
    "type_1_mito_avg_solidity",
    "type_1_mito_avg_circularity",
    "type_1_mito_dist_from_edge",
    "percent_type_1_mito",
    # Mitochondria: subtype 2
    "type_2_mito_density",
    "type_2_mito_avg_area",
    "type_2_mito_avg_aspect_ratio",
    "type_2_mito_perimeter",
    "type_2_mito_percent_total_area",
    "type_2_mito_avg_solidity",
    "type_2_mito_avg_circularity",
    "type_2_mito_dist_from_edge",
    "percent_type_2_mito",
    # Mitochondria: subtype 3
    "type_3_mito_density",
    "type_3_mito_avg_area",
    "type_3_mito_avg_aspect_ratio",
    "type_3_mito_perimeter",
    "type_3_mito_percent_total_area",
    "type_3_mito_avg_solidity",
    "type_3_mito_avg_circularity",
    "type_3_mito_dist_from_edge",
    "percent_type_3_mito",
    # Peroxisomes: general features
    "peroxisome_density",
    "peroxisome_avg_area",
    "peroxisome_aspect_ratio",
    "peroxisome_perimeter",
    "peroxisome_percent_total_area",
    "peroxisome_solidity",
    "peroxisome_circularity",
    "peroxisome_distance_from_edge",
    # Peroxisomes: subtype 1
    "type_1_peroxisome_density",
    "type_1_peroxisome_avg_area",
    "type_1_peroxisome_avg_aspect_ratio",
    "type_1_peroxisome_perimeter",
    "type_1_peroxisome_percent_total_area",
    "type_1_peroxisome_avg_solidity",
    "type_1_peroxisome_avg_circularity",
    "type_1_peroxisome_dist_from_edge",
    "percent_type_1_peroxisome",
    # Peroxisomes: subtype 2
    "type_2_peroxisome_density",
    "type_2_peroxisome_avg_area",
    "type_2_peroxisome_avg_aspect_ratio",
    "type_2_peroxisome_perimeter",
    "type_2_peroxisome_percent_total_area",
    "type_2_peroxisome_avg_solidity",
    "type_2_peroxisome_avg_circularity",
    "type_2_peroxisome_dist_from_edge",
    "percent_type_2_peroxisome",
    # Peroxisomes: subtype 3
    "type_3_peroxisome_density",
    "type_3_peroxisome_avg_area",
    "type_3_peroxisome_avg_aspect_ratio",
    "type_3_peroxisome_perimeter",
    "type_3_peroxisome_percent_total_area",
    "type_3_peroxisome_avg_solidity",
    "type_3_peroxisome_avg_circularity",
    "type_3_peroxisome_dist_from_edge",
    "percent_type_3_peroxisome",
    # Lipid droplets: general features
    "ld_density",
    "ld_avg_area",
    "ld_perimeter",
    "ld_percent_total_area",
    "ld_solidity",
    "ld_circularity",
    "ld_distance_from_edge",
    # Lipid droplets: subtype 1
    "type_1_ld_density",
    "type_1_ld_avg_area",
    "type_1_ld_perimeter",
    "type_1_ld_percent_total_area",
    "type_1_ld_avg_solidity",
    "type_1_ld_avg_circularity",
    "type_1_ld_dist_from_edge",
    "percent_type_1_ld",
    # Lipid droplets: subtype 2
    "type_2_ld_density",
    "type_2_ld_avg_area",
    "type_2_ld_perimeter",
    "type_2_ld_percent_total_area",
    "type_2_ld_avg_solidity",
    "type_2_ld_avg_circularity",
    "type_2_ld_dist_from_edge",
    "percent_type_2_ld",
    # Lipid droplets: subtype 3
    "type_3_ld_density",
    "type_3_ld_avg_area",
    "type_3_ld_perimeter",
    "type_3_ld_percent_total_area",
    "type_3_ld_avg_solidity",
    "type_3_ld_avg_circularity",
    "type_3_ld_dist_from_edge",
    "percent_type_3_ld",
    # Lipid droplets: subtype 4
    "type_4_ld_density",
    "type_4_ld_avg_area",
    "type_4_ld_perimeter",
    "type_4_ld_percent_total_area",
    "type_4_ld_avg_solidity",
    "type_4_ld_avg_circularity",
    "type_4_ld_dist_from_edge",
    "percent_type_4_ld",
]

input_data = np.asarray(data[feature_columns])

# normalize the input data so that all values are between 0 and 1
denom = np.max(input_data, axis=0) - np.min(input_data, axis=0)
denom[denom == 0] = 1  # avoid divide-by-zero if any column is constant
norm_data = (input_data - np.min(input_data, axis=0)) / denom

#%%
# 80/20 train/val split
indicies = np.arange(norm_data.shape[0])
np.random.shuffle(indicies)

split_idx = int(0.8 * len(indicies))

train_indicies = indicies[:split_idx]
val_indicies = indicies[split_idx:]

X_train = norm_data[train_indicies]
X_val = norm_data[val_indicies]

y_train_nutrient = nutrient_labels[train_indicies]
y_val_nutrient = nutrient_labels[val_indicies]

y_train_catagory = catagory_labels[train_indicies]
y_val_catagory = catagory_labels[val_indicies]

# DataFrame version of X_val for SHAP (preserves feature names)
X_val_df = pd.DataFrame(X_val, columns=feature_columns)

#%%
# train the nutrient random forest
print("Training Diet model...")
nutrient_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
nutrient_rf.fit(X_train, y_train_nutrient)

#%%
# train the category random forest
print("Training Hepatocyte Category model...")
catagory_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
catagory_rf.fit(X_train, y_train_catagory)

# Readable class names used only in plot text and image filenames.
diet_classes = nutrient_rf.classes_
diet_plot_names = [get_diet_name(cls) for cls in diet_classes]
category_classes = catagory_rf.classes_
category_plot_names = [get_hepatocyte_category_name(cls) for cls in category_classes]

print("Diet class mapping used in plots:")
for raw_class, readable_name in zip(diet_classes, diet_plot_names):
    print(f"  model label {raw_class} -> {readable_name}")

#%%
# make predictions
predicted_diet = nutrient_rf.predict(X_val)
true_diet = y_val_nutrient

predicted_catagory = catagory_rf.predict(X_val)
true_catagory = y_val_catagory

#%%
# =============================================================================
# CONFUSION MATRICES
# =============================================================================

# row-wise percentage confusion matrices
cat_confusion_matrix_raw = confusion_matrix(true_catagory, predicted_catagory)
nutrient_confusion_matrix_raw = confusion_matrix(true_diet, predicted_diet)

cat_cm_percent = cat_confusion_matrix_raw.astype(float) / np.maximum(
    cat_confusion_matrix_raw.sum(axis=1, keepdims=True), 1) * 100

nut_cm_percent = nutrient_confusion_matrix_raw.astype(float) / np.maximum(
    nutrient_confusion_matrix_raw.sum(axis=1, keepdims=True), 1) * 100

# ---------- CATEGORY confusion matrix ----------
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cat_cm_percent, cmap="Greys", vmin=0, vmax=100)
ax.set_xticks(np.arange(cat_cm_percent.shape[1]))
ax.set_yticks(np.arange(cat_cm_percent.shape[0]))
ax.set_xticklabels(category_plot_names, rotation=45, ha="right")
ax.set_yticklabels(category_plot_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Hepatocyte Category Confusion Matrix (%)")
ax.set_xticks(np.arange(-0.5, cat_cm_percent.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, cat_cm_percent.shape[0], 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
for i in range(cat_cm_percent.shape[0]):
    for j in range(cat_cm_percent.shape[1]):
        val = cat_cm_percent[i, j]
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                color="white" if val > 50 else "black")
plt.tight_layout()
save_fig("confusion_matrix_hepatocyte_category.png")

# ---------- DIET confusion matrix ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(nut_cm_percent, cmap="Greys", vmin=0, vmax=100)
diet_labels = diet_plot_names
ax.set_xticks(np.arange(len(diet_labels)))
ax.set_yticks(np.arange(len(diet_labels)))
ax.set_xticklabels(diet_labels)
ax.set_yticklabels(diet_labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Diet Confusion Matrix (%)")
ax.set_xticks(np.arange(-0.5, nut_cm_percent.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, nut_cm_percent.shape[0], 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
for i in range(nut_cm_percent.shape[0]):
    for j in range(nut_cm_percent.shape[1]):
        val = nut_cm_percent[i, j]
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                color="white" if val > 50 else "black")
plt.tight_layout()
save_fig("confusion_matrix_diet.png")

#%%
# =============================================================================
# FEATURE IMPORTANCE PLOTS (sklearn MDI)
# =============================================================================

nutrient_importance = nutrient_rf.feature_importances_
catagory_importance = catagory_rf.feature_importances_

plt.figure(figsize=(8, 10))
order = np.argsort(nutrient_importance)
plt.barh(np.array(feature_columns)[order], nutrient_importance[order])
plt.xlabel("MDI feature importance (mean decrease in impurity)")
plt.title("Diet Random Forest - MDI Feature Importance")
plt.tight_layout()
save_fig("MDI_feature_importance_diet.png")

plt.figure(figsize=(8, 10))
order = np.argsort(catagory_importance)
plt.barh(np.array(feature_columns)[order], catagory_importance[order])
plt.xlabel("MDI feature importance (mean decrease in impurity)")
plt.title("Hepatocyte Category Random Forest - MDI Feature Importance")
plt.tight_layout()
save_fig("MDI_feature_importance_hepatocyte_category.png")

#%%
# accuracy
nutrient_accuracy = np.sum(true_diet == predicted_diet) / true_diet.shape[0]
catagory_accuracy = np.sum(true_catagory == predicted_catagory) / true_catagory.shape[0]

print("Diet accuracy:", nutrient_accuracy)
print("Hepatocyte category accuracy:", catagory_accuracy)
print("Diet accuracy (sklearn):", accuracy_score(true_diet, predicted_diet))
print("Category accuracy (sklearn):", accuracy_score(true_catagory, predicted_catagory))


# =============================================================================
# SHAP VALUE ANALYSIS
# =============================================================================

#%%
# -------------------------------------------------------------------------
# SHAP — Diet (Nutrient) Model
# -------------------------------------------------------------------------
print("\nCalculating SHAP values for Diet model (this may take a few minutes)...")
 
explainer_nutrient = shap.TreeExplainer(nutrient_rf)
shap_values_nutrient = explainer_nutrient.shap_values(X_val_df)
diet_classes = nutrient_rf.classes_

# Handle both old (list of 2D) and new (single 3D array) SHAP output formats
# Correct slicing for Diet model
if isinstance(shap_values_nutrient, np.ndarray) and shap_values_nutrient.ndim == 3:
    shap_nutrient_per_class = [shap_values_nutrient[:, :, i] for i in range(shap_values_nutrient.shape[2])]
else:
    shap_nutrient_per_class = shap_values_nutrient


# --- All-class stacked bar ---
plt.figure()

shap.summary_plot(
    shap_nutrient_per_class,
    X_val_df.values,
    feature_names=feature_columns,
    class_names=diet_plot_names,
    plot_type="bar",
    max_display=len(feature_columns),
    show=False
)
plt.title("Diet SHAP - Mean |SHAP| per feature (CNT, STV, and WD)")
plt.xlabel("Mean |SHAP value|")
plt.gcf().set_size_inches(10, 18)
plt.tight_layout()
# Keep the diet legend in the readable CNT, STV, WD order.
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
diet_legend_order = {name: index for index, name in enumerate(diet_plot_names)}
sorted_pairs = sorted(
    zip(labels, handles),
    key=lambda pair: diet_legend_order.get(pair[0], len(diet_legend_order)),
)
sorted_labels, sorted_handles = zip(*sorted_pairs)
ax.legend(sorted_handles, sorted_labels, title="Diet group", loc="lower right")
save_fig("shap_diet_all_groups_mean_abs_bar.png")
# --- Save Diet legend as separate figure ---
fig_legend = plt.figure(figsize=(2, 2))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis("off")
ax_legend.legend(
    sorted_handles,
    sorted_labels,
    title="Diet group",
    loc="center",
    frameon=True
)
plt.tight_layout()
save_fig("shap_diet_legend_CNT_STV_WD.png")
plt.close()

# --- Per-class beeswarm plots ---
for i, cls in enumerate(diet_classes):
    diet_name = get_diet_name(cls)
    diet_token = filename_token(diet_name)
    plt.figure()
    shap.summary_plot(
        shap_values_nutrient[:, :, i],    # slice directly from original 3D array
        X_val_df.values,
        feature_names=feature_columns,
        show=False
    )
    plt.title(f"Diet SHAP Beeswarm - {diet_name}")
    plt.tight_layout()
    save_fig(f"shap_diet_beeswarm_{diet_token}.png")

# --- Per-class top-all mean |SHAP| bar chart ---
for i, cls in enumerate(diet_classes):
    diet_name = get_diet_name(cls)
    diet_token = filename_token(diet_name)
    mean_abs_shap = np.abs(shap_values_nutrient[:, :, i]).mean(axis=0)
    order = np.argsort(mean_abs_shap)
    fig, ax = plt.subplots(figsize=(8, 18))
    ax.barh(np.array(feature_columns)[order], mean_abs_shap[order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Diet SHAP Mean |SHAP| - {diet_name} (all features)")
    plt.tight_layout()
    save_fig(f"shap_diet_mean_abs_bar_{diet_token}.png")

# --- Export Diet SHAP values to CSV ---
for i, cls in enumerate(diet_classes):
    df_shap = pd.DataFrame(shap_values_nutrient[:, :, i], columns=feature_columns)
    df_shap.to_csv(os.path.join(output_dir, f"shap_diet_class_{cls}.csv"), index=False)
    print(f"Saved: shap_diet_class_{cls}.csv")


#%%
# -------------------------------------------------------------------------
# SHAP — Hepatocyte Category Model
# -------------------------------------------------------------------------
print("\nCalculating SHAP values for Hepatocyte Category model (this may take a few minutes)...")

explainer_catagory = shap.TreeExplainer(catagory_rf)
shap_values_catagory = explainer_catagory.shap_values(X_val_df)
category_classes = catagory_rf.classes_

# Handle both old (list of 2D) and new (single 3D array) SHAP output formats
if isinstance(shap_values_catagory, np.ndarray) and shap_values_catagory.ndim == 3:
    shap_category_per_class = [shap_values_catagory[:, :, i] for i in range(shap_values_catagory.shape[2])]
else:
    shap_category_per_class = shap_values_catagory

# --- All-class stacked bar (Hepatocyte Category) ---
import matplotlib.cm as cm

# Generate 11 visually distinct colors
n_classes = len(category_classes)
cmap = cm.get_cmap("tab20", n_classes)

plt.figure()
shap.summary_plot(
    shap_category_per_class,
    X_val_df.values,
    feature_names=feature_columns,
    class_names=category_plot_names,
    plot_type="bar",
    color=cmap,                        # pass colormap object, not a list
    max_display=len(feature_columns),
    show=False
)
plt.title("Hepatocyte Category SHAP - Mean |SHAP| per feature (all categories)")
plt.xlabel("Mean |SHAP value|")

# Keep the legend in the model's hepatocyte-category order.
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
category_legend_order = {name: index for index, name in enumerate(category_plot_names)}
sorted_pairs = sorted(
    zip(labels, handles),
    key=lambda pair: category_legend_order.get(pair[0], len(category_legend_order)),
)
sorted_labels, sorted_handles = zip(*sorted_pairs)
ax.legend(sorted_handles, sorted_labels, title="Hepatocyte category", loc="lower right")

plt.gcf().set_size_inches(10, 18)
plt.tight_layout()
save_fig("shap_hepatocyte_category_all_categories_mean_abs_bar.png")

  # --- Save Category legend as separate figure ---
fig_legend = plt.figure(figsize=(2, 3))  # taller since 11 classes
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis("off")
ax_legend.legend(
    sorted_handles,
    sorted_labels,
    title="Hepatocyte category",
    loc="center",
    frameon=True
)
plt.tight_layout()
save_fig("shap_hepatocyte_category_legend.png")
plt.close()

# --- Per-class beeswarm plots ---
for i, cls in enumerate(category_classes):
    category_name = get_hepatocyte_category_name(cls)
    category_token = filename_token(category_name).lower()
    plt.figure()
    shap.summary_plot(
        shap_values_catagory[:, :, i],    # slice directly from original 3D array
        X_val_df.values,
        feature_names=feature_columns,
        show=False
    )
    plt.title(f"Hepatocyte Category SHAP Beeswarm - {category_name}")
    plt.tight_layout()
    save_fig(f"shap_hepatocyte_category_beeswarm_{category_token}.png")
    
  

# --- Per-class all features mean |SHAP| bar chart ---
for i, cls in enumerate(category_classes):
    category_name = get_hepatocyte_category_name(cls)
    category_token = filename_token(category_name).lower()
    mean_abs_shap = np.abs(shap_values_catagory[:, :, i]).mean(axis=0)
    order = np.argsort(mean_abs_shap)
    fig, ax = plt.subplots(figsize=(8, 18))
    ax.barh(np.array(feature_columns)[order], mean_abs_shap[order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Hepatocyte Category SHAP Mean |SHAP| - {category_name} (all features)")
    plt.tight_layout()
    save_fig(f"shap_hepatocyte_category_mean_abs_bar_{category_token}.png")

# --- Export Category SHAP values to CSV ---
for i, cls in enumerate(category_classes):
    df_shap = pd.DataFrame(shap_values_catagory[:, :, i], columns=feature_columns)
    df_shap.to_csv(os.path.join(output_dir, f"shap_category_class_{cls}.csv"), index=False)
    print(f"Saved: shap_category_class_{cls}.csv")

print(f"\nAll outputs saved to: {output_dir}")