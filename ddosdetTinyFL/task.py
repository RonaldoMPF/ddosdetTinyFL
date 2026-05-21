"""Definition of the Deep Learning Model for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from datasets import load_dataset

dataset_name = "RonaldoMPF/CIC-DDoS2019-15C"
fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load Train set, preprocess it, and return a DataLoader yielding (X, y) tuples."""

    global fds
    if fds is None:
        
        partitioner = DirichletPartitioner( # Considering Non IID Data
            num_partitions=num_partitions,
            partition_by=" Label", 
            alpha=1.0,
            seed=42,
        )
        
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    # Load the Dataset in Pandas format
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

    dataset.dropna(inplace=True)

    X = dataset.drop(" Label", axis=1)
    y = dataset[" Label"]

    # Divide Data on each Node: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train) 
    y_test_encoded = encoder.transform(y_test) 

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                             torch.tensor(y_train_encoded, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                            torch.tensor(y_test_encoded, dtype=torch.long))

    return DataLoader(train_ds, batch_size, shuffle=True), DataLoader(test_ds, batch_size)


class DDoSClassifier(nn.Module):
    """Defining the Architecture and Configurations of the Deep Learning Model to be used."""
    
    def __init__(self, input_size=14, num_classes=12):

        super(DDoSClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, num_classes)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

def trainer(model, train_loader, num_epochs, lr):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    model.train()

    running_loss = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        
        for X_batch, y_batch in train_loader:
            
            optimizer.zero_grad()
            
            # Compute Prediction Error
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

    # Calculation of Model Training Metrics
    end_time = time.time()
    duration = end_time - start_time
    avg_trainloss = running_loss / (num_epochs * len(train_loader))

    return avg_trainloss, duration

def evaluator(model, test_loader):

    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():

        for X_batch, y_batch in test_loader:
            
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(y_batch.numpy())
            
    # Calculation of Model Performance Metrics
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    avg_loss = loss / len(test_loader)
    
    # A dictionary to facilitate aggregation in Flower.
    return avg_loss, accuracy, {"f1_score": f1, "precision": precision}

def load_centralized_dataset():
    """Load Centralized Test set, preprocess it, and return a DataLoader yielding (X, y) tuples."""

    batch_size = 32
   
    # Load the Dataset (Configure Dataset Splitting in the Online Repository)
    dataset = load_dataset(dataset_name, split="test").with_format("pandas")[:]
    dataset.dropna(inplace=True)

    # Separate Features and Labels
    X = dataset.drop(" Label", axis=1)
    y = dataset[" Label"]

    # Preprocessing (Must match the logic in load_data)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Create TensorDataset so the DataLoader returns (X, y) tuples
    test_ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32), 
        torch.tensor(y_encoded, dtype=torch.long)
    )
    
    return DataLoader(test_ds, batch_size)