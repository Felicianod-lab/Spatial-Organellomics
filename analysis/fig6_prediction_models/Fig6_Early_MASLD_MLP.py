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
DATA_PATH = "...add path.../Fig6_Early_MASLD_Model_Data_Matrix.csv"
FILTERED_DATA_SAVE_PATH = "...add path.../all_included.csv"

DEVICE = torch.device("cpu")
VALIDATION_FRACTION = 0.20
BATCH_SIZE = 256
LEARNING_RATE = 1e-5

MINIMUM_MITO_COUNT = 10
NUMBER_OF_CLASSES = 6

# This preserves the original plot text. The source data pattern is
# "18days_", although the original heatmap displayed that class as "17 days".
TIMEPOINT_CLASS_LABELS = [
    "0 days",
    "7 days",
    "17 days", # same as 18 days
    "31 days",
    "42 days",
    "50 days",
]

# Preserve the original CSV-saving behavior, which includes the DataFrame index.
SAVE_DATAFRAME_INDEX = True



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
    """Pair normalized cell measurements with one-hot time-point labels."""

    def __init__(self, features, labels):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class MLP(nn.Module):
    """Six-class multilayer perceptron used by the original script."""

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
def assign_timepoint_types(data):
    """Create the original numeric Type classes from the labels column."""
    labeled_data = data.copy()
    source_labels = labeled_data["labels"].astype(str)

    conditions = [
        source_labels.str.contains("C_", regex=False),
        source_labels.str.contains("7days_", regex=False),
        source_labels.str.contains("18days_", regex=False),
        source_labels.str.contains("31days_", regex=False),
        source_labels.str.contains("42days_", regex=False),
        source_labels.str.contains("50days_", regex=False),
    ]
    class_values = [1, 2, 3, 4, 5, 6]

    labeled_data["Type"] = np.select(
        conditions,
        class_values,
        default=0,
    )
    return labeled_data


def load_and_filter_data(data_path):
    """Load, label, and filter the data using the original rules."""
    data = pd.read_csv(data_path, na_values="-")

    # Preserve the original missing/infinite-value handling.
    data.fillna(0, inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    data = assign_timepoint_types(data)

    filtered_data = data.loc[data["mito_aspect_ratio"] > 0].copy()
    mito_count = filtered_data["mito_density"] * filtered_data["area"]
    filtered_data = filtered_data.loc[
        mito_count >= MINIMUM_MITO_COUNT
    ].copy()

    return filtered_data


def make_one_hot_labels(filtered_data):
    """Convert Type values 1-6 to the original six-column one-hot labels."""
    type_values = torch.as_tensor(
        filtered_data["Type"].to_numpy(),
        dtype=torch.long,
    )

    # Include class 0 during encoding and then remove its column, matching the
    # original [:, 1:] behavior while always producing six output columns.
    return nn.functional.one_hot(
        type_values,
        num_classes=NUMBER_OF_CLASSES + 1,
    )[:, 1:]


# %%
def split_indices(number_of_rows, validation_fraction):
    """Randomly split row indices into 80% training and 20% validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    validation_size = int(number_of_rows * validation_fraction)
    if validation_size == 0 or validation_size >= number_of_rows:
        raise ValueError(
            "The dataset is too small for the requested training/validation split."
        )

    indices = np.arange(number_of_rows)
    np.random.shuffle(indices)
    validation_indices, training_indices = np.split(
        indices,
        [validation_size],
    )
    return training_indices, validation_indices


def normalize_using_training_data(
    input_data,
    training_indices,
    validation_indices,
):
    """
    Min-max normalize using statistics calculated from training rows only.

    This prevents validation information from leaking into preprocessing.
    Validation values can be below 0 or above 1 when they fall outside the
    range observed in the training set.
    """
    training_data = input_data[training_indices]
    validation_data = input_data[validation_indices]

    training_minimum = training_data.min(axis=0)
    training_maximum = training_data.max(axis=0)
    training_range = training_maximum - training_minimum

    # Avoid division by zero for a feature that is constant in training data.
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
    """Create training and validation loaders for the six-class model."""
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
    """Evaluate one validation pass with dropout disabled."""
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


def fit_model(model, training_loader, validation_loader):
    """Train the model and return per-epoch training/validation losses."""
    EPOCHS = 8_000
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    loss_function = nn.BCELoss()

    training_losses = []
    validation_losses = []

    for _ in tqdm.tqdm(range(EPOCHS), desc="Training time-point model"):
        training_losses.append(
            train_epoch(
                training_loader,
                model,
                optimizer,
                loss_function,
            )
        )
        validation_losses.append(
            validate_epoch(
                validation_loader,
                model,
                loss_function,
            )
        )

    return training_losses, validation_losses


def predict(loader, model):
    """Return predicted and true class indices for every validation row."""
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
            true_classes.append(
                labels.argmax(dim=1).cpu().numpy()
            )

    return (
        np.concatenate(predicted_classes),
        np.concatenate(true_classes),
    )


# %%
def normalize_confusion_matrix(matrix):
    """Convert each true-class row of a confusion matrix to percentages."""
    samples_per_class = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.zeros_like(matrix, dtype=float)

    np.divide(
        matrix,
        samples_per_class,
        out=normalized_matrix,
        where=samples_per_class != 0,
    )
    return normalized_matrix * 100


def plot_confusion_heatmap(matrix):
    """Plot the row-normalized time-point confusion matrix."""
    normalized_matrix = normalize_confusion_matrix(matrix)
    integer_matrix = np.round(normalized_matrix).astype(int)

    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(
        integer_matrix,
        cmap="bone_r",
        linewidth=0.05,
        linecolor="k",
        xticklabels=TIMEPOINT_CLASS_LABELS,
        yticklabels=TIMEPOINT_CLASS_LABELS,
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


def plot_loss_curves(training_losses, validation_losses):
    """Plot training and validation loss by epoch."""
    plt.figure()
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "validation"])
    plt.title("Trained Model")


# %%
def main():
    filtered_data = load_and_filter_data(DATA_PATH)
    filtered_data.to_csv(
        FILTERED_DATA_SAVE_PATH,
        index=SAVE_DATAFRAME_INDEX,
    )

    unmatched_rows = int((filtered_data["Type"] == 0).sum())
    if unmatched_rows:
        print(
            "Warning:",
            unmatched_rows,
            "filtered rows did not match one of the six time-point patterns",
            "and therefore retain the original all-zero target behavior.",
        )

    timepoint_labels = make_one_hot_labels(filtered_data)
    input_data = filtered_data[FEATURE_COLUMNS].to_numpy(dtype=float)

    # Split before normalization so validation values do not affect scaling.
    training_indices, validation_indices = split_indices(
        number_of_rows=input_data.shape[0],
        validation_fraction=VALIDATION_FRACTION,
    )
    training_features, validation_features = normalize_using_training_data(
        input_data=input_data,
        training_indices=training_indices,
        validation_indices=validation_indices,
    )

    print("Total filtered cells:", len(filtered_data))
    print("Training cells:", len(training_indices))
    print("Validation cells:", len(validation_indices))

    training_loader, validation_loader = make_data_loaders(
        training_features=training_features,
        validation_features=validation_features,
        labels=timepoint_labels,
        training_indices=training_indices,
        validation_indices=validation_indices,
    )

    timepoint_mlp = MLP(
        input_channels=len(FEATURE_COLUMNS),
        output_channels=NUMBER_OF_CLASSES,
    ).to(DEVICE)

    training_loss, validation_loss = fit_model(
        model=timepoint_mlp,
        training_loader=training_loader,
        validation_loader=validation_loader,
    )

    predicted_timepoints, true_timepoints = predict(
        validation_loader,
        timepoint_mlp,
    )

    timepoint_confusion_matrix = confusion_matrix(
        true_timepoints,
        predicted_timepoints,
        labels=np.arange(NUMBER_OF_CLASSES),
    )

    plot_confusion_heatmap(timepoint_confusion_matrix)
    plot_loss_curves(training_loss, validation_loss)

    timepoint_accuracy = np.mean(
        true_timepoints == predicted_timepoints
    )
    print("The time-point prediction accuracy is =", timepoint_accuracy)

    plt.show()


if __name__ == "__main__":
    main()
