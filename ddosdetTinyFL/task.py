"""Definition of the Deep Learning Model for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices"""

import time
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

dataset_name = "RonaldoMPF/DDoSEdge-IIoTset"
fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load Train set, preprocess it, and return a DataLoader yielding (X, y) tuples."""
    
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(  # Considering Non-IID data
            num_partitions=num_partitions,
            partition_by="Label", 
            alpha=1.0,
            seed=42,
        )
        
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    # Loads the Dataset partition in Pandas format
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    dataset.dropna(inplace=True)

    X = dataset.drop("Label", axis=1)
    y = dataset["Label"]

    # Splits the data at each Node: 70% Training, 30% Internal Validation
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Preprocessing
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train) 
    y_validation_encoded = encoder.transform(y_validation) 

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                             torch.tensor(y_train_encoded, dtype=torch.long))
    validation_ds = TensorDataset(torch.tensor(X_validation, dtype=torch.float32), 
                            torch.tensor(y_validation_encoded, dtype=torch.long))

    return DataLoader(train_ds, batch_size, shuffle=True), DataLoader(validation_ds, batch_size)


class DDoSClassifier(nn.Module):
    """Architecture of the Deep Learning MLP Model compatible with Pruning and Dynamic Quantization."""
    
    def __init__(self, input_size=96, num_classes=4):
        super(DDoSClassifier, self).__init__()
        
        # Simple Linear Layers: Application of Pruning and Dynamic Quantization
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):

        # The use of functional F.relu is fully compatible with the Dynamic Quantization of Linear Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def trainer(model, train_loader, num_epochs, lr):
    """Performs Model Training in Float32 (maintains Gradient Stability)."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    model.train()

    running_loss = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

    end_time = time.time()
    duration = end_time - start_time
    avg_trainloss = running_loss / (num_epochs * len(train_loader))

    return avg_trainloss, duration


def evaluator(model, test_loader):
    """Evaluates the Model (compatible with both FP32 and Quantized INT8 Models)."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    all_preds = []
    all_targets = []
    
    # Quantized Models must run on the CPU
    device = torch.device("cpu") 

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            # Guarantee CPU dispatch before converting to NumPy format
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    # Calculation of Performance Metrics
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    avg_loss = loss / len(test_loader)
    
    return avg_loss, accuracy, {"f1_score": f1, "precision": precision, "recall": recall}


def load_centralized_dataset():
    """Load Centralized Test set for Global Server Evaluation."""
    batch_size = 32
   
    # Loads the centralized Test subset
    dataset = load_dataset(dataset_name, split="test").with_format("pandas")[:]
    dataset.dropna(inplace=True)

    X = dataset.drop("Label", axis=1)
    y = dataset["Label"]

    # Preprocessing aligned with local Client partitions
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    test_ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32), 
        torch.tensor(y_encoded, dtype=torch.long)
    )
    
    return DataLoader(test_ds, batch_size)