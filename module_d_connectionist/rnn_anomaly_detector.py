"""
Module D: Recurrent Neural Network (PyTorch LSTM) for Sequence Anomaly Prediction.
Given a sequence of the last 10 multivariate sensor readings:
    [vibration, motor_temperature, current_draw, wheel_speed_drift]
predicts whether an anomaly/failure is imminent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple, Dict, Any


class WarehouseLSTMAnomalyModel(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, num_layers: int = 1, num_classes: int = 2):
        super(WarehouseLSTMAnomalyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len=10, input_dim=4)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use last time-step hidden representation
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


def generate_synthetic_sensor_sequences(
    num_samples: int = 1000,
    seq_len: int = 10,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic multivariate time-series.
    Class 0 (Normal): Gaussian noise around nominal baseline [vibration: 0.2, temp: 45C, current: 3A, drift: 0.0]
    Class 1 (Imminent Anomaly): Exponential escalation in vibration and temperature drift across the 10 steps.
    """
    np.random.seed(random_seed)
    X = np.zeros((num_samples, seq_len, 4), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        is_anomaly = (i % 2 == 1)
        y[i] = 1 if is_anomaly else 0

        # Base noise
        vib = np.random.normal(0.2, 0.05, seq_len)
        temp = np.random.normal(45.0, 1.5, seq_len)
        curr = np.random.normal(3.0, 0.2, seq_len)
        drift = np.random.normal(0.0, 0.05, seq_len)

        if is_anomaly:
            # Inject deteriorating ramp across last 5 time steps
            ramp = np.linspace(0, 1.0, seq_len) ** 2
            vib += ramp * np.random.uniform(0.8, 1.5)
            temp += ramp * np.random.uniform(15.0, 25.0)
            curr += ramp * np.random.uniform(2.5, 4.0)
            drift += ramp * np.random.uniform(0.4, 0.8)

        # Standardize roughly
        X[i, :, 0] = vib
        X[i, :, 1] = (temp - 45.0) / 10.0
        X[i, :, 2] = (curr - 3.0) / 2.0
        X[i, :, 3] = drift

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class RNNAnomalyTrainer:
    def __init__(self, hidden_dim: int = 16, lr: float = 0.01):
        self.model = WarehouseLSTMAnomalyModel(input_dim=4, hidden_dim=hidden_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_and_evaluate(self, epochs: int = 15, batch_size: int = 32) -> Dict[str, Any]:
        """
        Trains model on 80% split and evaluates on 20% held-out test split.
        """
        X_all, y_all = generate_synthetic_sensor_sequences(num_samples=1000)
        
        # 80/20 train/test split
        split_idx = int(0.8 * len(X_all))
        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_test, y_test = X_all[split_idx:], y_all[split_idx:]

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

        # Evaluate on held-out test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = float(self.criterion(test_outputs, y_test))
            preds = torch.argmax(test_outputs, dim=1)
            correct = (preds == y_test).sum().item()
            accuracy = float(correct / len(y_test))

        return {
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "num_test_samples": len(y_test),
            "epochs": epochs
        }

    def predict_imminent_anomaly(self, sequence_10x4: np.ndarray) -> Tuple[bool, float]:
        """
        Predicts anomaly status on a single 10-step sequence.
        Returns: (is_anomaly_imminent: bool, probability: float)
        """
        self.model.eval()
        with torch.no_grad():
            tensor_seq = torch.tensor(sequence_10x4, dtype=torch.float32).unsqueeze(0)
            logits = self.model(tensor_seq)
            probs = torch.softmax(logits, dim=1)
            prob_anomaly = float(probs[0, 1])
            return (prob_anomaly >= 0.5), prob_anomaly
