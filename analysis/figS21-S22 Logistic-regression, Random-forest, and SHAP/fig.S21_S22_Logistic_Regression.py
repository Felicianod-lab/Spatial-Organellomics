# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:37:37 2026

@author: adhikarir
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:17:27 2026

@author: adhikarir
"""

#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#%%
# read the data
data = pd.read_csv(r"...add path...\figS21_S22_Experimental_Groups_Matrix.csv")
# FILTER: keep only rows where cell_id_linked != 0 (this is when motile can't find cells for organelles)
data = data[data["cell_id_linked"] != 0].reset_index(drop=True)

print("Remaining cells after filtering:", len(data))

# labels
nutrient_labels = np.asarray(data["group"])
catagory_labels = np.asarray(data["Prediction"])

#%%
# features
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

# normalize (IMPORTANT for logistic regression)
denom = np.max(input_data, axis=0) - np.min(input_data, axis=0)
denom[denom == 0] = 1
norm_data = (input_data - np.min(input_data, axis=0)) / denom

#%%
# split
# 80/20 split
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

#%%
# train models
nutrient_lr = LogisticRegression(max_iter=2000, multi_class='multinomial')
nutrient_lr.fit(X_train, y_train_nutrient)

catagory_lr = LogisticRegression(max_iter=2000, multi_class='multinomial')
catagory_lr.fit(X_train, y_train_catagory)


#%%

#%%
# Logistic Regression feature contribution tables
# uses mean absolute coefficient across classes

# --- Diet / nutrient Logistic Regression ---
nutrient_coef = nutrient_lr.coef_   # shape: [num_classes, num_features]
nutrient_importance = np.mean(np.abs(nutrient_coef), axis=0)

nutrient_lr_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_raw": nutrient_importance
})

nutrient_lr_df = nutrient_lr_df.sort_values(
    by="importance_raw",
    ascending=False
).reset_index(drop=True)

nutrient_lr_df["importance_fraction"] = (
    nutrient_lr_df["importance_raw"] / nutrient_lr_df["importance_raw"].sum()
)
nutrient_lr_df["importance_percent"] = (
    nutrient_lr_df["importance_fraction"] * 100
)

# --- Category Logistic Regression ---
catagory_coef = catagory_lr.coef_   # shape: [num_classes, num_features]
catagory_importance = np.mean(np.abs(catagory_coef), axis=0)

catagory_lr_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_raw": catagory_importance
})

catagory_lr_df = catagory_lr_df.sort_values(
    by="importance_raw",
    ascending=False
).reset_index(drop=True)

catagory_lr_df["importance_fraction"] = (
    catagory_lr_df["importance_raw"] / catagory_lr_df["importance_raw"].sum()
)
catagory_lr_df["importance_percent"] = (
    catagory_lr_df["importance_fraction"] * 100
)

# --- cleaner rounding ---
for df in [nutrient_lr_df, catagory_lr_df]:
    df["importance_raw"] = df["importance_raw"].round(6)
    df["importance_fraction"] = df["importance_fraction"].round(6)
    df["importance_percent"] = df["importance_percent"].round(2)

# --- save as CSV ---
nutrient_lr_df.to_csv("nutrient_logistic_regression_feature_contribution.csv", index=False)
catagory_lr_df.to_csv("catagory_logistic_regression_feature_contribution.csv", index=False)

# --- optional preview in console ---
print("\nTop 20 diet Logistic Regression features:")
print(nutrient_lr_df.head(20))

print("\nTop 20 category Logistic Regression features:")
print(catagory_lr_df.head(20))
# Logistic Regression "feature importance" (coefficients)

# nutrient model
nutrient_coef = nutrient_lr.coef_
nutrient_importance = np.mean(np.abs(nutrient_coef), axis=0)

plt.figure(figsize=(8,10))
order = np.argsort(nutrient_importance)
plt.barh(np.array(feature_columns)[order], nutrient_importance[order])
plt.xlabel("Coefficient magnitude")
plt.title("Logistic Regression Feature Influence (Diet)")

# category model
catagory_coef = catagory_lr.coef_
catagory_importance = np.mean(np.abs(catagory_coef), axis=0)

plt.figure(figsize=(8,10))
order = np.argsort(catagory_importance)
plt.barh(np.array(feature_columns)[order], catagory_importance[order])
plt.xlabel("Coefficient magnitude")
plt.title("Logistic Regression Feature Influence (Category)")
#%%
# predictions
predicted_diet = nutrient_lr.predict(X_val)
true_diet = y_val_nutrient

predicted_catagory = catagory_lr.predict(X_val)
true_catagory = y_val_catagory

#%%
# confusion matrices
cat_confusion_matrix = confusion_matrix(true_catagory, predicted_catagory)
nutrient_confusion_matrix = confusion_matrix(true_diet, predicted_diet)

# normalize to %
cat_cm_percent = cat_confusion_matrix.astype(float) / np.maximum(
    cat_confusion_matrix.sum(axis=1, keepdims=True), 1
) * 100

nut_cm_percent = nutrient_confusion_matrix.astype(float) / np.maximum(
    nutrient_confusion_matrix.sum(axis=1, keepdims=True), 1
) * 100

#%%
# function to plot table-style confusion matrix
def plot_cm(cm_percent, title, labels=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm_percent, cmap="Greys", vmin=0, vmax=100)

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # grid (table look)
    ax.set_xticks(np.arange(-0.5, cm_percent.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm_percent.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # values inside boxes (rounded %)
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            val = cm_percent[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.show()

#%%
# plot
plot_cm(cat_cm_percent, "Hepatocyte Catagory (%)")
plot_cm(nut_cm_percent, "Diet (%)", labels=['CNT','STV','WD'])

#%%
# accuracy
nutrient_accuracy = np.mean(true_diet == predicted_diet)
catagory_accuracy = np.mean(true_catagory == predicted_catagory)

print("Diet accuracy:", nutrient_accuracy)
print("Catagory accuracy:", catagory_accuracy)