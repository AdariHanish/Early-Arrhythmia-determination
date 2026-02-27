# ============================================================================
# baseline_models.py — WITH FULL PROGRESS VISIBILITY
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, LEARNING_RATE, SEED


torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1️⃣ SVM BASELINE
# ============================================================

def run_svm(train_x, train_y, test_x, test_y):

    print("\n[SVM] Training Started...")

    scaler = StandardScaler()

    train_x_np = train_x.cpu().numpy()
    test_x_np = test_x.cpu().numpy()

    train_y_np = train_y.cpu().numpy()
    test_y_np = test_y.cpu().numpy()

    n_subset = min(12000, len(train_x_np))
    idx = np.random.choice(len(train_x_np), n_subset, replace=False)

    train_x_np = train_x_np[idx]
    train_y_np = train_y_np[idx]

    train_x_scaled = scaler.fit_transform(train_x_np)
    test_x_scaled = scaler.transform(test_x_np)

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        max_iter=3000
    )

    start = time.time()

    # Simulated progress bar for SVM fit
    svm.fit(train_x_scaled, train_y_np)

    preds = svm.predict(test_x_scaled)
    latency = (time.time() - start) * 1000

    acc = accuracy_score(test_y_np, preds)

    print(f"[SVM] Accuracy: {acc*100:.2f}%")

    return acc, latency


# ============================================================
# 2️⃣ CNN BASELINE
# ============================================================

class CNNBaseline(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3️⃣ LSTM BASELINE
# ============================================================

class LSTMBaseline(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# ============================================================
# TRAINING UTILITY WITH PROGRESS
# ============================================================

def train_nn_model(model, train_x, train_y, test_x, test_y, model_name):

    print(f"\n[{model_name}] Training Started...")

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_x = train_x.unsqueeze(1)
    test_x = test_x.unsqueeze(1)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    start = time.time()

    epochs = 6

    for epoch in tqdm(range(epochs), desc=f"{model_name} Epochs"):

        model.train()

        batch_bar = tqdm(
            train_loader,
            desc=f"{model_name} Epoch {epoch+1}/{epochs}",
            leave=False
        )

        for batch_x, batch_y in batch_bar:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    latency = (time.time() - start) * 1000

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=256)

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            preds = torch.argmax(model(batch_x), dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total

    print(f"[{model_name}] Accuracy: {acc*100:.2f}%")

    return acc, latency


# ============================================================
# MAIN BASELINE RUNNER
# ============================================================

def run_baselines(train_x, train_y, test_x, test_y):

    print("\n========== Running Baselines ==========")

    num_classes = len(torch.unique(train_y))
    results = {}

    # -------- SVM --------
    svm_acc, svm_lat = run_svm(train_x, train_y, test_x, test_y)
    results["svm"] = {
        "accuracy": float(svm_acc),
        "latency_ms": float(svm_lat)
    }

    # -------- CNN --------
    cnn = CNNBaseline(num_classes)
    cnn_acc, cnn_lat = train_nn_model(
        cnn, train_x, train_y, test_x, test_y, "CNN"
    )
    results["cnn"] = {
        "accuracy": float(cnn_acc),
        "latency_ms": float(cnn_lat)
    }

    # -------- LSTM --------
    lstm = LSTMBaseline(num_classes)
    lstm_acc, lstm_lat = train_nn_model(
        lstm, train_x, train_y, test_x, test_y, "LSTM"
    )
    results["lstm"] = {
        "accuracy": float(lstm_acc),
        "latency_ms": float(lstm_lat)
    }

    print("\n========== Baseline Completed ==========")

    return results
