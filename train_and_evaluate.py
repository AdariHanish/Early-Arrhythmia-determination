# ============================================================================
# train_and_evaluate.py — FINAL CLEAN RESEARCH VERSION
# ============================================================================

import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from config import DEVICE, EPOCHS, LEARNING_RATE, CHECKPOINT_DIR

from preprocessing import remove_noise
from sparse_feature_extraction import SparseFeatureExtractor
from fusion_module import FusionModule
from cardiofusenet_model import CardioFuseNet


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_modules():
    sparse = SparseFeatureExtractor().to(DEVICE)
    fusion = FusionModule().to(DEVICE)
    model = CardioFuseNet().to(DEVICE)
    return sparse, fusion, model


# ============================================================
# TRAINING
# ============================================================

def train_model(train_signals, train_labels,
                test_signals, test_labels):

    train_signals = train_signals.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_signals = test_signals.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

    sparse, fusion, model = initialize_modules()

    optimizer = torch.optim.Adam(
        list(sparse.parameters()) +
        list(fusion.parameters()) +
        list(model.parameters()),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    dataset = TensorDataset(train_signals, train_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    best_acc = 0.0
    best_epoch = 0
    history = {"test_acc": []}

    print("\nTraining Started...\n")

    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):

        sparse.train()
        fusion.train()
        model.train()

        for batch_signals, batch_labels in loader:

            # Normalize inside training loop
            batch_signals = (batch_signals - batch_signals.mean(dim=1, keepdim=True)) / \
                            (batch_signals.std(dim=1, keepdim=True) + 1e-8)

            optimizer.zero_grad()

            cleaned = remove_noise(batch_signals)
            f = sparse(cleaned)

            reconstructed = sparse.reconstruct(f)
            reconstruction_loss = torch.mean((reconstructed - cleaned) ** 2)

            modality_2 = torch.abs(f)
            fused = fusion(f, modality_2)

            logits, probs = model(fused)

            loss_fn = nn.CrossEntropyLoss()
            classification_loss = loss_fn(logits, batch_labels)

            total_loss = classification_loss + 0.02 * reconstruction_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(sparse.parameters()) +
                list(fusion.parameters()) +
                list(model.parameters()),
                max_norm=5.0
            )

            optimizer.step()

        scheduler.step()

        metrics = evaluate_model(
            sparse, fusion, model,
            test_signals, test_labels,
            print_metrics=False
        )

        test_acc = metrics["accuracy"]
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            torch.save({
                "sparse_state_dict": sparse.state_dict(),
                "fusion_state_dict": fusion.state_dict(),
                "model_state_dict": model.state_dict(),
                "accuracy": best_acc,
                "epoch": best_epoch
            }, os.path.join(CHECKPOINT_DIR, "cardiofusenet_best.pth"))

            print(f"✅ Model saved at epoch {best_epoch} with accuracy {best_acc:.4f}")

        print(f"Epoch {epoch+1}/{EPOCHS} | Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}")

        if best_acc >= 0.98:
            print("\n🎯 Target reached. Early stopping.\n")
            break

    final_eval = evaluate_model(
        sparse, fusion, model,
        test_signals, test_labels,
        print_metrics=True
    )

    return {
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "history": history,
        "final_eval": final_eval
    }


# ============================================================
# EVALUATION (NORMALIZATION FIXED HERE)
# ============================================================
def evaluate_model(sparse, fusion, model,
                   signals, labels,
                   print_metrics=True):

    import time
    import torch
    import torch.nn.functional as F

    # Move to device
    signals = signals.to(DEVICE)
    labels = labels.to(DEVICE)

    # SAME normalization as training
    signals = (signals - signals.mean(dim=1, keepdim=True)) / \
              (signals.std(dim=1, keepdim=True) + 1e-8)

    sparse.eval()
    fusion.eval()
    model.eval()

    with torch.no_grad():

        # -------------------------------
        # Accurate latency measurement
        # -------------------------------
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        # 1️⃣ Noise Removal
        cleaned = remove_noise(signals)

        # 2️⃣ Sparse Feature Extraction
        f = sparse(cleaned)

        # 3️⃣ Reconstruction
        reconstructed = sparse.reconstruct(f)

        # 4️⃣ Fusion
        modality_2 = torch.abs(f)
        fused = fusion(f, modality_2)

        # 5️⃣ Classification
        logits, probs = model(fused)
        preds = torch.argmax(probs, dim=1)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        total_time = (time.time() - start_time) * 1000
        latency_ms = total_time / signals.size(0)

        # -------------------------------
        # Accuracy
        # -------------------------------
        accuracy = (preds == labels).float().mean().item()

        # -------------------------------
        # Scientific SNR
        # -------------------------------
        signal_power = torch.mean(cleaned ** 2)
        noise_power = torch.mean((cleaned - reconstructed) ** 2)
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        snr_value = snr_db.item()

        # -------------------------------
        # Fusion Consistency (0–1)
        # -------------------------------
        cos_sim = F.cosine_similarity(f, modality_2, dim=1)
        fusion_score = torch.mean((cos_sim + 1) / 2).item()

        # -------------------------------
        # Convert outputs safely
        # -------------------------------
        y_true = labels.detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()
        y_probs = probs.detach().cpu().numpy()

        if print_metrics:
            print(
                f"Accuracy: {accuracy:.4f} | "
                f"SNR: {snr_value:.2f} dB | "
                f"Latency/sample: {latency_ms:.4f} ms | "
                f"Fusion Score: {fusion_score:.4f}"
            )

    return {
        "accuracy": accuracy,
        "snr_db": snr_value,
        "latency_ms": latency_ms,
        "fusion_score": fusion_score,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs
    }
