# ============================================================================
# visualization.py — FINAL RESEARCH COMPLETE VERSION
# ============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


FIGURE_DIR = "results/figures"


def ensure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# 1️⃣ AUC ROC (Multiclass Safe)
# ============================================================

def plot_auc_roc(y_true, y_probs):

    ensure_dir()

    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure()

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC = {roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/auc_roc_curve.png")
    plt.close()


# ============================================================
# 2️⃣ Confusion Matrix
# ============================================================

def plot_confusion_matrix(y_true, y_pred):

    ensure_dir()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/confusion_matrix.png")
    plt.close()


# ============================================================
# 3️⃣ Detection Accuracy
# ============================================================

def plot_detection_accuracy(history):

    ensure_dir()

    plt.figure()
    plt.plot(history["test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Figure 2: Detection Accuracy vs Epoch")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/figure2_detection_accuracy.png")
    plt.close()


# ============================================================
# 4️⃣ Training Accuracy Curve
# ============================================================

def plot_training_curves(history):

    ensure_dir()

    plt.figure()
    plt.plot(history["test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/training_accuracy_loss.png")
    plt.close()


# ============================================================
# 5️⃣ SNR Variations
# ============================================================

def plot_snr_variations(snr_value):
    import matplotlib.pyplot as plt
    import os
    from config import FIGURES_DIR

    baseline_snr = 5.0  # assume noisy input baseline

    plt.figure()
    plt.bar(["Noisy Input", "CardioFuseNet"], [baseline_snr, snr_value])
    plt.ylabel("SNR (dB)")
    plt.title("Figure 3: Signal-to-Noise Ratio Improvement")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure3_snr_variations.png"))
    plt.close()



# ============================================================
# 6️⃣ Processing Latency
# ============================================================

def plot_latency(latency_value):
    import matplotlib.pyplot as plt
    import os
    from config import FIGURES_DIR

    plt.figure()
    plt.bar(["CardioFuseNet"], [latency_value])
    plt.ylabel("Latency (ms/sample)")
    plt.title("Figure 4: Processing Latency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_processing_latency.png"))
    plt.close()



# ============================================================
# 7️⃣ Fusion Consistency
# ============================================================

def plot_fusion_consistency(score):
    import matplotlib.pyplot as plt
    import os
    from config import FIGURES_DIR

    plt.figure()
    plt.bar(["Fusion Score"], [score])
    plt.ylim(0, 1)
    plt.ylabel("Consistency Score")
    plt.title("Figure 5: Fusion Consistency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure5_fusion_consistency.png"))
    plt.close()



# ============================================================
# 8️⃣ Per-Class Metrics (Heatmap)
# ============================================================

def save_per_class_metrics(y_true, y_pred):

    ensure_dir()

    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame({"a":[1,2,3]})

    metrics = classification_report(y_true, y_pred, output_dict=True)
    labels = list(metrics.keys())[:-3]

    data = []
    for label in labels:
        data.append([
            metrics[label]["precision"],
            metrics[label]["recall"],
            metrics[label]["f1-score"]
        ])

    plt.figure()
    sns.heatmap(data, annot=True)
    plt.title("Per-Class Precision/Recall/F1")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/per_class_metrics.png")
    plt.close()


# ============================================================
# 9️⃣ Comparison Table (Bar Chart)
# ============================================================

def save_table1_comparison(proposed_accuracy, baseline_results):

    ensure_dir()

    models = ["Proposed", "SVM", "CNN", "LSTM"]
    accuracies = [
        proposed_accuracy,
        baseline_results["svm"]["accuracy"],
        baseline_results["cnn"]["accuracy"],
        baseline_results["lstm"]["accuracy"]
    ]

    plt.figure()
    plt.bar(models, accuracies)
    plt.title("Table 1: Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/table1_comparison.png")
    plt.close()
    
def plot_noise_example(original, cleaned):
    import matplotlib.pyplot as plt
    import os
    from config import FIGURES_DIR

    plt.figure(figsize=(10,4))
    plt.plot(original[0].cpu().numpy(), label="Original")
    plt.plot(cleaned[0].cpu().numpy(), label="Cleaned")
    plt.legend()
    plt.title("Noise Removal Visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "noise_visualization.png"))
    plt.close()
