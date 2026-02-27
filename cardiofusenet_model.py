# ============================================================================
# cardiofusenet_model.py — FINAL OPTIMIZED VERSION
# ============================================================================

import torch
import torch.nn as nn
from config import (
    DEVICE,
    SPARSE_FEATURE_DIM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    DROPOUT_RATE
)


class CardioFuseNet(nn.Module):
    """
    Eq.(5)  φ = g(u(t))          → nonlinear projection
    Eq.(6)  ŷ = C(φ)             → classifier
    Eq.(7)  P = softmax(ŷ)       → probability output
    """

    def __init__(self, input_dim=SPARSE_FEATURE_DIM, num_classes=5):
        super().__init__()

        # -----------------------------
        # Projection function g(·)
        # -----------------------------
        self.projection = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_1),
            nn.BatchNorm1d(HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.BatchNorm1d(HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )

        # -----------------------------
        # Classifier C(·)
        # -----------------------------
        self.classifier = nn.Linear(HIDDEN_DIM_2, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.to(DEVICE)

    def forward(self, x):

        # Eq.(5)
        phi = self.projection(x)

        # Eq.(6)
        logits = self.classifier(phi)

        # Eq.(7)
        probs = self.softmax(logits)

        return logits, probs
