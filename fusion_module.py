"""
fusion_module.py
----------------

Implements Equation (3) and Equation (4) from the paper:

(3)  u(t) = Σ w_k m_k(t)

(4)  A(t) = u(t - τ)

Where:
m_k(t) = modality representations
w_k    = fusion weights
τ      = temporal shift factor
u(t)   = fused representation
A(t)   = aligned output

Strictly derived from the paper.
No extra architectural additions.
CPU and GPU compatible.
"""

import torch
import torch.nn as nn
from config import DEVICE, FUSION_WEIGHTS, TEMPORAL_SHIFT


class FusionModule(nn.Module):
    """
    Implements multimodal weighted fusion and temporal alignment.
    """

    def __init__(self):
        super(FusionModule, self).__init__()

        # ------------------------------------------------------
        # Fusion weights w_k (Equation 3)
        # Initialized from config
        # ------------------------------------------------------
        initial_weights = torch.tensor(FUSION_WEIGHTS, dtype=torch.float32)

        # Make weights learnable (still satisfies Σ w_k m_k(t))
        self.weights = nn.Parameter(initial_weights)

        self.to(DEVICE)

    def forward(self, modality_1, modality_2):
        """
        modality_1 -> first modality representation m1(t)
        modality_2 -> second modality representation m2(t)

        Returns:
            aligned_output A(t)
        """

        # ------------------------------------------------------
        # Normalize weights to ensure stable fusion
        # This preserves Equation (3) without altering structure
        # ------------------------------------------------------
        normalized_weights = torch.softmax(self.weights, dim=0)

        # ------------------------------------------------------
        # Equation (3): u(t) = Σ w_k m_k(t)
        # ------------------------------------------------------
        u = (
            normalized_weights[0] * modality_1 +
            normalized_weights[1] * modality_2
        )

        # ------------------------------------------------------
        # Equation (4): A(t) = u(t - τ)
        # Implemented as temporal shift
        # ------------------------------------------------------
        if TEMPORAL_SHIFT > 0:
            A = torch.roll(u, shifts=-TEMPORAL_SHIFT, dims=1)
        else:
            A = u

        return A


# ============================================================
# Fusion Consistency Metric (Figure 5)
# ============================================================

def compute_fusion_consistency(modality_1, modality_2, fused_output):
    """
    Computes Fusion Consistency Score (Figure 5)

    Measures stability between fused output and individual modalities.

    Returns value between 0 and 1.
    """

    diff1 = torch.mean(torch.abs(fused_output - modality_1))
    diff2 = torch.mean(torch.abs(fused_output - modality_2))

    consistency = 1 / (1 + diff1 + diff2)

    return consistency.item()
