# ============================================================================
# config.py — FINAL STABLE RESEARCH CONFIG
# ============================================================================

import os
import torch


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


# ============================================================
# SIGNAL PARAMETERS
# ============================================================

INPUT_SIZE = 187
SIGNAL_LENGTH = 187

SPARSE_FEATURE_DIM = 128
SPARSE_L1_LAMBDA = 0.001


# ============================================================
# MODEL PARAMETERS
# ============================================================

HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
DROPOUT_RATE = 0.3


# ============================================================
# FUSION PARAMETERS
# ============================================================

FUSION_WEIGHTS = [0.5, 0.5]
TEMPORAL_SHIFT = 1


# ============================================================
# TRAINING PARAMETERS
# ============================================================

EPOCHS = 50
LEARNING_RATE = 0.001
THRESHOLD = 0.5
SMOOTHING_ALPHA = 0.7


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("[CONFIG] Required directories created/verified.")


# ============================================================
# DATASET FILE PATHS
# ============================================================

MITBIH_TRAIN_PATH = os.path.join(DATASET_DIR, "mitbih_train.csv")
MITBIH_TEST_PATH = os.path.join(DATASET_DIR, "mitbih_test.csv")

PTBDB_NORMAL_PATH = os.path.join(DATASET_DIR, "ptbdb_normal.csv")
PTBDB_ABNORMAL_PATH = os.path.join(DATASET_DIR, "ptbdb_abnormal.csv")


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def print_config():

    print("\n[CONFIGURATION]")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Smoothing Alpha: {SMOOTHING_ALPHA}")
    print(f"Sparse L1 Lambda: {SPARSE_L1_LAMBDA}")
    print(f"Fusion Weights: {FUSION_WEIGHTS}")
    print(f"Temporal Shift: {TEMPORAL_SHIFT}")
    print(f"Signal Length: {SIGNAL_LENGTH}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("-" * 40)
