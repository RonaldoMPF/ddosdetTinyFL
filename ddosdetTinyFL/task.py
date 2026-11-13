"""Definition of the Deep Learning Model for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices"""

import time
from functools import lru_cache

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score

dataset_name = "RonaldoMPF/Edge-IIoTset"
fds = None  # Cache FederatedDataset


@lru_cache(maxsize=1)
def get_label_encoder() -> LabelEncoder:
    """Return one label mapping shared by all clients and centralized evaluation.

    Fitting an encoder independently on each client (or on the test split) makes
    the same class map to different integer targets. Discover the complete set
    of labels once and reuse that mapping everywhere.
    """
    labels = []
    for split in ("train", "test"):
        split_labels = load_dataset(dataset_name, split=split)["Attack_type"]
        labels.extend(label for label in split_labels if not pd.isna(label))

    if not labels:
        raise ValueError(f"No non-null Attack_type labels found in {dataset_name!r}")

    return LabelEncoder().fit(labels)


class DDoSClassifier(nn.Module):
    """Architecture of the MLP model compatible with pruning and dynamic quantization."""

    def __init__(self, input_size=96, num_classes=None):
        super().__init__()
        if num_classes is None:
            num_classes = len(get_label_encoder().classes_)

        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, num_classes)

    def forward(self, x):
        # Functional ReLU is compatible with dynamic quantization of Linear layers.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load a client partition and return its training and validation loaders."""
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="Attack_type",
            alpha=1.0,
            seed=42,
        )
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    dataset = dataset.dropna()

    X = dataset.drop("Attack_type", axis=1)
    y = dataset["Attack_type"]

    # Split the data at each node: 70% training, 30% internal validation.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    encoder = get_label_encoder()
    y_train_encoded = encoder.transform(y_train)
    y_validation_encoded = encoder.transform(y_validation)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_encoded, dtype=torch.long),
    )
    validation_ds = TensorDataset(
        torch.tensor(X_validation, dtype=torch.float32),
        torch.tensor(y_validation_encoded, dtype=torch.long),
    )

    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(validation_ds, batch_size),
    )


def trainer(model, train_loader, num_epochs, lr):
    """Train the model in Float32 to maintain gradient stability."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    model.train()

    running_loss = 0.0
    start_time = time.time()

    for _ in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    duration = time.time() - start_time
    avg_trainloss = running_loss / (num_epochs * len(train_loader))
    return avg_trainloss, duration


def evaluator(model, test_loader):
    """Evaluate the model (compatible with FP32 and quantized INT8 models)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    all_preds = []
    all_targets = []

    # Quantized models must run on the CPU.
    device = torch.device("cpu")

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)

    avg_loss = loss / len(test_loader)
    return avg_loss, accuracy, {"f1_score": f1, "precision": precision, "recall": recall}


def load_centralized_dataset():
    """Load the centralized test set using the shared class-to-index mapping."""
    batch_size = 32
    dataset = load_dataset(dataset_name, split="test").with_format("pandas")[:]
    dataset = dataset.dropna()

    X = dataset.drop("Attack_type", axis=1)
    y = dataset["Attack_type"]

    # Keep the existing centralized feature preprocessing behavior.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    y_encoded = get_label_encoder().transform(y)
    test_ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long),
    )
    return DataLoader(test_ds, batch_size)
