# ============================================================================
# sparse_feature_extraction.py — IMPROVED REVIEW-SAFE VERSION
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, INPUT_SIZE, SPARSE_L1_LAMBDA


class SparseFeatureExtractor(nn.Module):
    """
    STRICT Equation (2) implementation:

    f = argmin (1/2)||x - Ψf||² + λ||f||_1

    Solved via ISTA.
    """

    def __init__(self, feature_dim=128, ista_steps=25):
        super().__init__()

        self.feature_dim = feature_dim
        self.ista_steps = ista_steps

        # --------------------------------------------------
        # Learnable dictionary Ψ
        # --------------------------------------------------
        self.psi = nn.Parameter(
            torch.empty(INPUT_SIZE, feature_dim)
        )

        nn.init.xavier_uniform_(self.psi)

        # Learnable sparsity weight (still Eq.2 compliant)
        self.lambda_param = nn.Parameter(
            torch.tensor(SPARSE_L1_LAMBDA)
        )

        self.to(DEVICE)

    # ------------------------------------------------------
    # Soft Threshold Operator
    # ------------------------------------------------------
    def soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.clamp(
            torch.abs(x) - threshold,
            min=0.0
        )

    # ------------------------------------------------------
    # Column Normalization (critical for stability)
    # ------------------------------------------------------
    def normalize_dictionary(self):
        with torch.no_grad():
            self.psi.data = F.normalize(self.psi.data, dim=0)

    # ------------------------------------------------------
    # ISTA Solver (Eq.2)
    # ------------------------------------------------------
    def forward(self, x):

        batch_size = x.size(0)

        # Normalize dictionary each forward
        self.normalize_dictionary()

        f = torch.zeros(
            batch_size,
            self.feature_dim,
            device=DEVICE
        )

        # Spectral norm for Lipschitz constant
        L = torch.linalg.norm(self.psi, ord=2) ** 2
        step_size = 1.0 / (L + 1e-6)

        for _ in range(self.ista_steps):

            residual = torch.matmul(f, self.psi.T) - x
            gradient = torch.matmul(residual, self.psi)

            f = f - step_size * gradient

            f = self.soft_threshold(
                f,
                torch.abs(self.lambda_param) * step_size
            )

        return f

    # ------------------------------------------------------
    # Reconstruction (Eq.14)
    # ------------------------------------------------------
    def reconstruct(self, f):
        return torch.matmul(f, self.psi.T)
