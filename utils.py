# ============================================================================
# utils.py - PyTorch Utility Functions (Final Clean Version)
# ============================================================================

import os
import json
import time
import random
import numpy as np
import torch

from config import SEED, DEVICE, TRAINING_HISTORY_PATH


# ============================================================
# DEVICE SETUP (Main.py expects setup_gpu)
# ============================================================

def setup_gpu():
    """
    Print device information.
    Compatible with main.py
    """
    print(f"[UTILS] Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"[UTILS] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[UTILS] Running on CPU.")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds():
    """
    Set all seeds for reproducibility.
    """

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(SEED)

    print(f"[UTILS] All seeds set to: {SEED}")


# ============================================================
# TIMER CLASS
# ============================================================

class Timer:
    def __init__(self, description="Operation"):
        self.description = description
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000
        print(f"[TIMER] {self.description}: {self.elapsed:.2f} ms")


# ============================================================
# INFERENCE LATENCY (Paper Figure 4)
# ============================================================

def measure_inference_latency(model, sample_input, num_runs=50):

    model.eval()
    latencies = []

    sample_input = sample_input.to(DEVICE)

    with torch.no_grad():

        # Warm-up
        _ = model(sample_input)

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(sample_input)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

    return latencies, float(np.mean(latencies))


# ============================================================
# RELIABILITY METRIC (Eq. 13)
# ============================================================

def compute_reliability_metric(y_true, y_pred):

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    return float(np.mean(np.abs(y_pred - y_true)))


# ============================================================
# GLOBAL PERFORMANCE (Eq. 15)
# ============================================================

def compute_global_performance(accuracies):
    return float(np.sum(accuracies))


# ============================================================
# SNR COMPUTATION (Figure 3)
# ============================================================

def compute_snr(original_signal, processed_signal):

    original_signal = np.array(original_signal, dtype=np.float64)
    processed_signal = np.array(processed_signal, dtype=np.float64)

    signal_power = np.mean(processed_signal ** 2)
    noise = original_signal - processed_signal
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return 30.0

    snr_db = 10 * np.log10(signal_power / noise_power)

    if np.isnan(snr_db) or np.isinf(snr_db):
        return 30.0

    return float(snr_db)


# ============================================================
# FUSION CONSISTENCY (Figure 5)
# ============================================================

def compute_fusion_consistency(predictions_list):

    if len(predictions_list) < 2:
        return np.ones(len(predictions_list[0])), 1.0

    n_samples = len(predictions_list[0])
    consistency_scores = np.zeros(n_samples)

    for i in range(n_samples):
        preds = [float(p[i]) for p in predictions_list]
        variance = np.var(preds)
        consistency_scores[i] = 1.0 / (1.0 + variance)

    return consistency_scores, float(np.mean(consistency_scores))


# ============================================================
# TRAINING HISTORY SAVE / LOAD
# ============================================================

def save_training_history(history_dict):

    serializable = {}

    for key, values in history_dict.items():
        serializable[key] = [float(v) for v in values]

    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(serializable, f, indent=4)

    print(f"[UTILS] Training history saved to {TRAINING_HISTORY_PATH}")


def load_training_history():

    if os.path.exists(TRAINING_HISTORY_PATH):
        with open(TRAINING_HISTORY_PATH, "r") as f:
            return json.load(f)

    return None


# ============================================================
# SAFE RESULTS SAVING (Prevents Corruption)
# ============================================================

def save_results(results_dict, filename):
    """
    Saves results safely to results/ directory.
    Converts tensors automatically.
    """

    os.makedirs("results", exist_ok=True)

    serializable = {}

    for k, v in results_dict.items():

        if isinstance(v, torch.Tensor):
            serializable[k] = v.detach().cpu().tolist()

        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()

        elif isinstance(v, (int, float, str, list, dict)):
            serializable[k] = v

        else:
            try:
                serializable[k] = float(v)
            except:
                serializable[k] = str(v)

    filepath = os.path.join("results", filename)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=4)

    print(f"[UTILS] Results saved to {filepath}")
import matplotlib.pyplot as plt
import os

def plot_training_curves(history, save_path):

    epochs = range(1, len(history["test_acc"]) + 1)

    plt.figure()
    plt.plot(epochs, history["test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true, y_pred, save_path):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)
    plt.close()
