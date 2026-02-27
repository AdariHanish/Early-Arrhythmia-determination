# ============================================================================
# main.py — FINAL RESEARCH STABLE VERSION
# ============================================================================

import os
import sys
import warnings
import torch

from visualization import (
    plot_auc_roc,
    plot_confusion_matrix,
    plot_detection_accuracy,
    plot_training_curves,
    plot_snr_variations,
    plot_latency,
    plot_fusion_consistency,
    save_per_class_metrics,
    save_table1_comparison
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


def main():

    print("=" * 70)
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║              CardioFuseNet v1.0                          ║")
    print("  ║  A Sparse-Fusion ML Framework for Early Arrhythmia      ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print("=" * 70)

    SKIP_BASELINES = False

    # ============================================================
    # STEP 1: ENVIRONMENT
    # ============================================================

    print("\n[MAIN] Step 1: Setting up environment...\n")

    from utils import setup_gpu, set_seeds, save_results
    from config import (
        create_directories,
        print_config,
        MITBIH_TRAIN_PATH,
        MITBIH_TEST_PATH,
        CHECKPOINT_DIR,
        DEVICE
    )

    setup_gpu()
    set_seeds()
    create_directories()
    print_config()

    # ============================================================
    # STEP 2: VERIFY DATASET
    # ============================================================

    print("\n[MAIN] Step 2: Verifying dataset files...\n")

    required = [MITBIH_TRAIN_PATH, MITBIH_TEST_PATH]

    for path in required:
        if not os.path.exists(path):
            print(f"❌ Missing dataset file: {path}")
            sys.exit(1)
        else:
            print(f"✅ {path}")

    print("All dataset files verified.\n")

    # ============================================================
    # STEP 3: LOAD DATA
    # ============================================================

    from preprocessing import load_csv_dataset

    train_signals, train_labels = load_csv_dataset(MITBIH_TRAIN_PATH)
    test_signals, test_labels = load_csv_dataset(MITBIH_TEST_PATH)

    # ============================================================
    # STEP 4: TRAIN OR LOAD MODEL
    # ============================================================

    print("\n[MAIN] Step 3: Training / Loading CardioFuseNet...\n")

    from train_and_evaluate import (
        train_model,
        initialize_modules,
        evaluate_model
    )

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "cardiofusenet_best.pth")

    if os.path.exists(checkpoint_path):

        print("✅ Existing trained model found. Loading checkpoint...\n")

        sparse, fusion, model = initialize_modules()
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        sparse.load_state_dict(checkpoint["sparse_state_dict"])
        fusion.load_state_dict(checkpoint["fusion_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])

        final_eval = evaluate_model(
            sparse, fusion, model,
            test_signals, test_labels,
            print_metrics=True
        )

        training_output = {
            "best_accuracy": checkpoint["accuracy"],
            "best_epoch": checkpoint["epoch"],
            "history": {"test_acc": []},
            "final_eval": final_eval
        }

    else:

        training_output = train_model(
            train_signals,
            train_labels,
            test_signals,
            test_labels
        )

    # ============================================================
    # STEP 5: GENERATE RESEARCH FIGURES
    # ============================================================

    print("\n[MAIN] Step 4: Generating Research Figures...\n")

    eval_results = training_output["final_eval"]

    # Core Figures
    plot_auc_roc(eval_results["y_true"], eval_results["y_probs"])
    plot_confusion_matrix(eval_results["y_true"], eval_results["y_pred"])

    # Training curves (only if model just trained)
    if training_output["history"]["test_acc"]:
        plot_detection_accuracy(training_output["history"])
        plot_training_curves(training_output["history"])

    # Scientific metrics
    plot_snr_variations(eval_results["snr_db"])
    plot_latency(eval_results.get("latency_ms", 0.0))
    plot_fusion_consistency(eval_results.get("fusion_score", 0.0))

    save_per_class_metrics(
        eval_results["y_true"],
        eval_results["y_pred"]
    )

    # ============================================================
    # STEP 6: SAVE PROPOSED RESULTS
    # ============================================================

    proposed_results = {
        "model": "CardioFuseNet",
        "best_accuracy": float(training_output["best_accuracy"]),
        "best_epoch": int(training_output["best_epoch"])
    }

    save_results(proposed_results, "proposed_results.json")

    print("\nProposed Model Results:")
    print(proposed_results)

    # ============================================================
    # STEP 7: RUN BASELINES
    # ============================================================

    if not SKIP_BASELINES:

        from baseline_models import run_baselines

        print("\n[MAIN] Step 5: Running Baseline Models...\n")

        baseline_results = run_baselines(
            train_signals,
            train_labels,
            test_signals,
            test_labels
        )

        save_results(baseline_results, "baseline_results.json")

        save_table1_comparison(
            proposed_results["best_accuracy"],
            baseline_results
        )

    print("\n" + "=" * 70)
    print("✅ CardioFuseNet execution complete!")
    print("=" * 70)

    return training_output


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
