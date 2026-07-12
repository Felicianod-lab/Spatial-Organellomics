# %%
import os

# Work around duplicate OpenMP library conflicts on some local installations.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset


# %%
# Configuration
DATA_PATH = (
    "...add path.../Fig6_Experimental_Groups_Organelle_Features_Matrix.csv"
)

DEVICE = torch.device("cpu")
VALIDATION_FRACTION = 0.20
BATCH_SIZE = 256
LEARNING_RATE = 1e-5




FEATURE_COLUMNS = [
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


# %%
class CellDataset(Dataset):
    """Pairs normalized cell measurements with one-hot class labels."""

    def __init__(self, features, labels):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class MLP(nn.Module):
    """Multilayer perceptron used for both classification tasks."""

    def __init__(self, input_channels, output_channels):
        super().__init__()

        hidden_size = 64
        dropout_fraction = 0.25

        self.network = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.Dropout(p=dropout_fraction),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout_fraction),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout_fraction),
            nn.ReLU(),
            nn.Linear(hidden_size, output_channels),
            nn.Softmax(dim=1),
        )

    def forward(self, features):
        return self.network(features)


# %%
def split_indices(number_of_rows, validation_fraction):
    """Randomly split row indices into training and validation groups."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    validation_size = int(number_of_rows * validation_fraction)
    if validation_size == 0 or validation_size >= number_of_rows:
        raise ValueError(
            "The dataset is too small for the requested training/validation split."
        )

    indices = np.arange(number_of_rows)
    np.random.shuffle(indices)
    validation_indices, training_indices = np.split(indices, [validation_size])
    return training_indices, validation_indices


def normalize_using_training_data(
    input_data,
    training_indices,
    validation_indices,
):
    """
    Min-max normalize using statistics calculated from the training set only.

    Training features are scaled to [0, 1]. Validation values can fall outside
    that interval when they are below the training minimum or above the
    training maximum; this is expected and prevents validation-data leakage.
    """
    training_data = input_data[training_indices]
    validation_data = input_data[validation_indices]

    training_minimum = training_data.min(axis=0)
    training_maximum = training_data.max(axis=0)
    training_range = training_maximum - training_minimum

    # A constant training feature has no scale. Using 1 avoids division by zero
    # while leaving every training value for that feature at zero.
    safe_training_range = np.where(training_range == 0, 1.0, training_range)

    normalized_training_data = (
        training_data - training_minimum
    ) / safe_training_range
    normalized_validation_data = (
        validation_data - training_minimum
    ) / safe_training_range

    return (
        normalized_training_data.astype(np.float32),
        normalized_validation_data.astype(np.float32),
    )


def make_data_loaders(
    training_features,
    validation_features,
    labels,
    training_indices,
    validation_indices,
):
    """Create training and validation data loaders for one label set."""
    training_dataset = CellDataset(
        training_features,
        labels[training_indices],
    )
    validation_dataset = CellDataset(
        validation_features,
        labels[validation_indices],
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    return training_loader, validation_loader


# %%
def train_epoch(training_loader, model, optimizer, loss_function):
    """Train for one complete pass over the training dataset."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for features, labels in training_loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def validate_epoch(validation_loader, model, loss_function):
    """Evaluate one complete validation pass with dropout disabled."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in validation_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(features)
            loss = loss_function(predictions, labels)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def fit_model(model, training_loader, validation_loader, epochs, description):
    """Train a model and return its per-epoch training/validation losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCELoss()

    training_losses = []
    validation_losses = []

    for _ in tqdm.tqdm(range(epochs), desc=description):
        training_losses.append(
            train_epoch(training_loader, model, optimizer, loss_function)
        )
        validation_losses.append(
            validate_epoch(validation_loader, model, loss_function)
        )

    return training_losses, validation_losses


def predict(loader, model):
    """Return predicted and true class indices for all rows in a loader."""
    model.eval()

    predicted_classes = []
    true_classes = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            predictions = model(features)

            predicted_classes.append(
                predictions.argmax(dim=1).cpu().numpy()
            )
            true_classes.append(labels.argmax(dim=1).cpu().numpy())

    return (
        np.concatenate(predicted_classes),
        np.concatenate(true_classes),
    )


# %%
def normalize_confusion_matrix(matrix):
    """Convert each true-class row of a confusion matrix to percentages."""
    samples_per_class = matrix.sum(axis=1, keepdims=True)
    return (matrix / samples_per_class) * 100


def plot_confusion_heatmap(matrix, title=None, class_labels=None):
    """Plot a row-normalized confusion matrix as integer percentages."""
    normalized_matrix = normalize_confusion_matrix(matrix)
    integer_matrix = np.round(normalized_matrix).astype(int)

    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(
        integer_matrix,
        cmap="bone_r",
        linewidth=0.05,
        linecolor="k",
        xticklabels=class_labels if class_labels is not None else "auto",
        yticklabels=class_labels if class_labels is not None else "auto",
        annot=True,
        fmt="d",
        annot_kws={"size": 7},
    )
    plt.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        labelbottom=False,
        bottom=False,
        top=True,
        labeltop=True,
    )

    if title is not None:
        plt.title(title)


def plot_loss_curves(training_losses, validation_losses, title=None):
    """Plot training and validation loss by epoch."""
    plt.figure()
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "val"])

    if title is not None:
        plt.title(title)


# %%
# Load data and construct labels.
data = pd.read_csv(DATA_PATH)

nutrient_labels = nn.functional.one_hot(
    torch.as_tensor(data["group"].to_numpy(), dtype=torch.long)
)[:, 1:]

category_labels = nn.functional.one_hot(
    torch.as_tensor(data["CAT"].to_numpy(), dtype=torch.long) #Prediction
)

input_data = data[FEATURE_COLUMNS].to_numpy()


# Split before normalization so validation values do not influence scaling.
training_indices, validation_indices = split_indices(
    number_of_rows=input_data.shape[0],
    validation_fraction=VALIDATION_FRACTION,
)

training_features, validation_features = normalize_using_training_data(
    input_data=input_data,
    training_indices=training_indices,
    validation_indices=validation_indices,
)


# %%
# Nutrient model
NUTRIENT_EPOCHS = 2_000
nutrient_training_loader, nutrient_validation_loader = make_data_loaders(
    training_features=training_features,
    validation_features=validation_features,
    labels=nutrient_labels,
    training_indices=training_indices,
    validation_indices=validation_indices,
)

nutrient_mlp = MLP(
    input_channels=len(FEATURE_COLUMNS),
    output_channels=3,
)

nutrient_training_loss, nutrient_validation_loss = fit_model(
    model=nutrient_mlp,
    training_loader=nutrient_training_loader,
    validation_loader=nutrient_validation_loader,
    epochs=NUTRIENT_EPOCHS,
    description="Training nutrient model",
)


# %%
# Hepatocyte-category model
CATEGORY_EPOCHS = 4_000
category_training_loader, category_validation_loader = make_data_loaders(
    training_features=training_features,
    validation_features=validation_features,
    labels=category_labels,
    training_indices=training_indices,
    validation_indices=validation_indices,
)

category_mlp = MLP(
    input_channels=len(FEATURE_COLUMNS),
    output_channels=11,
)

category_training_loss, category_validation_loss = fit_model(
    model=category_mlp,
    training_loader=category_training_loader,
    validation_loader=category_validation_loader,
    epochs=CATEGORY_EPOCHS,
    description="Training hepatocyte-category model",
)


# %%
# Predictions and confusion matrices
predicted_diet, true_diet = predict(
    nutrient_validation_loader,
    nutrient_mlp,
)
predicted_category, true_category = predict(
    category_validation_loader,
    category_mlp,
)

nutrient_confusion_matrix = confusion_matrix(true_diet, predicted_diet)
category_confusion_matrix = confusion_matrix(
    true_category,
    predicted_category,
)

plot_confusion_heatmap(
    category_confusion_matrix,
    title="Hepatocyte Category",
)
plot_confusion_heatmap(
    nutrient_confusion_matrix,
    class_labels=["CNT", "STV", "WD"],
)

plot_loss_curves(
    nutrient_training_loss,
    nutrient_validation_loss,
    title="Diet",
)
plot_loss_curves(
    category_training_loss,
    category_validation_loss,
)


# %%
# Validation accuracy
nutrient_accuracy = np.mean(true_diet == predicted_diet)
category_accuracy = np.mean(true_category == predicted_category)

print("The nutrient prediction accuracy is =", nutrient_accuracy)
print("The category prediction accuracy is =", category_accuracy)
