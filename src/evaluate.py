# ============================================================
#  src/evaluate.py — Test Set Evaluation
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS FILE DOES:
#  Loads each saved best_model.pth from the experiments/
#  folder, runs it on the held-out test set (1,600 images),
#  and computes all evaluation metrics:
#
#  - Overall accuracy
#  - Per-class precision, recall, F1
#  - Macro F1 (average F1 across all 4 classes)
#  - Macro AUC-ROC (one-vs-rest)
#  - Confusion matrix
#
#  WHY THE TEST SET?
#  During training we used the validation set to monitor
#  progress and save the best model. That means the val set
#  indirectly influenced training decisions. The test set
#  was never touched — it gives unbiased final numbers.
#
#  HOW TO USE:
#  from src.evaluate import evaluate_experiment, evaluate_all
#
#  Evaluate one experiment:
#    results = evaluate_experiment("baseline_vgg16")
#
#  Evaluate all experiments and save a comparison table:
#    evaluate_all()
# ============================================================

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import BrainTumorDataset, CLASS_NAMES
from src.transforms import get_transforms
from src.models import BrainTumorClassifier


# ──────────────────────────────────────────────
#  Auto-detect project root
# ──────────────────────────────────────────────
def _get_project_root() -> str:
    colab = "/content/drive/MyDrive/brain_tumor_classification"
    if os.path.exists(colab):
        return colab
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


PROJECT_ROOT = _get_project_root()


# ──────────────────────────────────────────────
#  load_model_from_experiment()
#
#  Rebuilds the model architecture from config.yaml
#  and loads the saved weights from best_model.pth.
#
#  WHY WE NEED config.yaml:
#  The .pth file only contains the weight tensors —
#  not the architecture. We need to know which backbone
#  was used so we can build the exact same structure
#  before loading weights into it.
# ──────────────────────────────────────────────
def load_model_from_experiment(
    experiment_name: str,
    device: torch.device,
    experiments_dir: str = None,
) -> tuple:
    """
    Loads a trained model from an experiment folder.

    Args:
        experiment_name : Folder name under experiments/
                          e.g. "baseline_vgg16"
        device          : torch.device to load model onto
        experiments_dir : Path to experiments/ folder.
                          Auto-detected if None.

    Returns:
        model      : Loaded BrainTumorClassifier (eval mode)
        exp_config : The config dict loaded from config.yaml

    Raises:
        FileNotFoundError if config.yaml or best_model.pth
        are missing from the experiment folder.
    """
    if experiments_dir is None:
        experiments_dir = os.path.join(PROJECT_ROOT, "experiments")

    exp_dir    = os.path.join(experiments_dir, experiment_name)
    config_path = os.path.join(exp_dir, "config.yaml")
    model_path  = os.path.join(exp_dir, "best_model.pth")

    # ── Check files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in: {exp_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pth not found in: {exp_dir}")

    # ── Load config
    with open(config_path) as f:
        exp_config = yaml.safe_load(f)

    backbone     = exp_config["model"]["backbone"]
    num_classes  = exp_config["model"]["num_classes"]
    dropout_rate = exp_config["regularization"]["dropout_rate"]

    # ── Rebuild the same model architecture
    model = BrainTumorClassifier(
        backbone_name = backbone,
        num_classes   = num_classes,
        dropout_rate  = dropout_rate,
        pretrained    = False,   # we load our own weights below
    )

    # ── Load the trained weights into the model
    # map_location ensures weights load correctly
    # regardless of whether they were saved on GPU or CPU
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # ── Set to evaluation mode
    # This disables dropout and uses running statistics
    # for batch normalization — essential for correct inference
    model = model.to(device)
    model.eval()

    return model, exp_config


# ──────────────────────────────────────────────
#  get_test_predictions()
#
#  Runs the test set through the model and collects
#  all predictions, true labels, and class probabilities.
#
#  WHY PROBABILITIES?
#  We need raw probability scores (not just class indices)
#  to compute AUC-ROC. Softmax converts the model's raw
#  logit scores into proper probabilities that sum to 1.
# ──────────────────────────────────────────────
def get_test_predictions(
    model:       BrainTumorClassifier,
    test_loader: DataLoader,
    device:      torch.device,
) -> tuple:
    """
    Runs the model on all test images.

    Args:
        model       : Trained BrainTumorClassifier in eval mode
        test_loader : DataLoader for test set
        device      : torch.device

    Returns:
        all_preds  : List of predicted class indices
        all_labels : List of true class indices
        all_probs  : np.array of shape [N, 4] — softmax probabilities
                     Used for AUC-ROC computation
    """
    all_preds  = []
    all_labels = []
    all_probs  = []

    # torch.no_grad() — no gradient computation during inference
    # This saves memory and makes evaluation faster
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass → raw scores (logits)
            logits = model(images)               # [batch, 4]

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1) # [batch, 4]

            # Predicted class = index with highest score
            preds = torch.argmax(logits, dim=1)  # [batch]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, np.array(all_probs)


# ──────────────────────────────────────────────
#  compute_metrics()
#
#  Given predictions and true labels, computes
#  all evaluation metrics for one experiment.
# ──────────────────────────────────────────────
def compute_metrics(
    preds:  list,
    labels: list,
    probs:  np.ndarray,
    class_names: list = CLASS_NAMES,
) -> dict:
    """
    Computes all evaluation metrics.

    Args:
        preds       : Predicted class indices [N]
        labels      : True class indices [N]
        probs       : Softmax probabilities [N, 4]
        class_names : List of class name strings

    Returns:
        Dictionary containing all metrics:
        - accuracy, macro_f1, macro_auc
        - per-class precision, recall, f1
        - full classification report string
    """
    # ── Overall accuracy
    accuracy = accuracy_score(labels, preds)

    # ── Macro F1 — average F1 across all 4 classes
    # zero_division=0 prevents warnings when a class has
    # no predictions (rare but can happen)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # ── Per-class F1 scores
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    # ── Macro AUC-ROC (one-vs-rest)
    # We binarize the labels: for each class, it becomes
    # a binary problem (this class vs all others)
    labels_bin = label_binarize(labels, classes=list(range(len(class_names))))
    try:
        macro_auc = roc_auc_score(
            labels_bin, probs, average="macro", multi_class="ovr"
        )
    except ValueError:
        # Can happen if a class has no samples in test set
        macro_auc = float("nan")

    # ── Full classification report
    # Contains precision, recall, F1 for each class
    # This is the text you paste directly into your paper
    report = classification_report(
        labels, preds,
        target_names = class_names,
        zero_division = 0,
    )

    # ── Build result dictionary
    result = {
        "accuracy":  round(accuracy * 100, 2),
        "macro_f1":  round(macro_f1, 4),
        "macro_auc": round(macro_auc, 4) if not np.isnan(macro_auc) else None,
        "report":    report,
    }

    # Add per-class F1 for each class
    for i, cls in enumerate(class_names):
        result[f"f1_{cls}"] = round(per_class_f1[i], 4)

    return result


# ──────────────────────────────────────────────
#  plot_confusion_matrix()
#
#  Creates and saves a heatmap of the confusion matrix.
#  Rows = true labels, Columns = predicted labels.
#  Diagonal = correct predictions (want these high).
#  Off-diagonal = mistakes (want these low).
# ──────────────────────────────────────────────
def plot_confusion_matrix(
    labels:      list,
    preds:       list,
    experiment_name: str,
    class_names: list  = CLASS_NAMES,
    outputs_dir: str   = None,
) -> str:
    """
    Plots and saves a confusion matrix heatmap.

    Args:
        labels          : True class indices
        preds           : Predicted class indices
        experiment_name : Used for the plot title and filename
        class_names     : Class label strings
        outputs_dir     : Where to save the PNG

    Returns:
        Path to the saved PNG file
    """
    if outputs_dir is None:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # ── Compute the confusion matrix
    cm = confusion_matrix(labels, preds)

    # ── Also compute row-normalized version (shows % per true class)
    # This makes it easier to compare across classes with
    # different sample counts
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ── Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: raw counts
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        xticklabels = [c.capitalize() for c in class_names],
        yticklabels = [c.capitalize() for c in class_names],
        ax         = axes[0],
        linewidths = 0.5,
    )
    axes[0].set_title(f"{experiment_name}\nRaw counts", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("True label", fontsize=11)
    axes[0].set_xlabel("Predicted label", fontsize=11)

    # ── Right: row-normalized (percentage per true class)
    sns.heatmap(
        cm_normalized,
        annot      = True,
        fmt        = ".2f",
        cmap       = "Blues",
        xticklabels = [c.capitalize() for c in class_names],
        yticklabels = [c.capitalize() for c in class_names],
        ax         = axes[1],
        linewidths = 0.5,
        vmin       = 0,
        vmax       = 1,
    )
    axes[1].set_title(f"{experiment_name}\nNormalized (row %)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("True label", fontsize=11)
    axes[1].set_xlabel("Predicted label", fontsize=11)

    plt.tight_layout()

    save_path = os.path.join(outputs_dir, f"confusion_matrix_{experiment_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    return save_path


# ──────────────────────────────────────────────
#  evaluate_experiment()
#
#  Main function — evaluates one experiment end-to-end.
#  Loads the model, runs the test set, computes metrics,
#  optionally plots confusion matrix.
# ──────────────────────────────────────────────
def evaluate_experiment(
    experiment_name:    str,
    device:             torch.device  = None,
    experiments_dir:    str           = None,
    data_dir:           str           = None,
    outputs_dir:        str           = None,
    plot_cm:            bool          = True,
    image_size:         int           = 224,
    batch_size:         int           = 32,
    verbose:            bool          = True,
) -> dict:
    """
    Full evaluation pipeline for one experiment.

    Args:
        experiment_name : Folder name under experiments/
        device          : torch.device (auto-detected if None)
        experiments_dir : Path to experiments/ folder
        data_dir        : Path to data/ folder
        outputs_dir     : Where to save confusion matrix PNG
        plot_cm         : Whether to plot confusion matrix
        image_size      : Must match training image size (224)
        batch_size      : Batch size for inference
        verbose         : Print results to console

    Returns:
        Dictionary with all metrics for this experiment

    Usage:
        results = evaluate_experiment("baseline_vgg16")
        print(results["macro_f1"])
        print(results["report"])
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if experiments_dir is None:
        experiments_dir = os.path.join(PROJECT_ROOT, "experiments")
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")
    if outputs_dir is None:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")

    if verbose:
        print(f"\n{'='*52}")
        print(f"  Evaluating: {experiment_name}")
        print(f"{'='*52}")

    # ── Step 1: Load model
    model, exp_config = load_model_from_experiment(
        experiment_name, device, experiments_dir
    )
    backbone = exp_config["model"]["backbone"]
    if verbose:
        print(f"  Backbone  : {backbone}")
        print(f"  Device    : {device}")

    # ── Step 2: Build test DataLoader
    # Use val/test transforms — NO augmentation
    test_transform = get_transforms(image_size=image_size, phase="test")
    test_dir       = os.path.join(data_dir, "Testing")

    test_dataset = BrainTumorDataset(
        root_dir  = test_dir,
        transform = test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )
    if verbose:
        print(f"  Test set  : {len(test_dataset)} images")

    # ── Step 3: Get predictions
    preds, labels, probs = get_test_predictions(model, test_loader, device)

    # ── Step 4: Compute metrics
    metrics = compute_metrics(preds, labels, probs)

    # ── Step 5: Print results
    if verbose:
        print(f"\n  Accuracy  : {metrics['accuracy']:.2f}%")
        print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
        print(f"  Macro AUC : {metrics['macro_auc']}")
        print(f"\n  Per-class F1:")
        for cls in CLASS_NAMES:
            print(f"    {cls:<15} {metrics[f'f1_{cls}']:.4f}")
        print(f"\n  Classification Report:")
        print(metrics["report"])

    # ── Step 6: Plot confusion matrix
    if plot_cm:
        cm_path = plot_confusion_matrix(
            labels, preds,
            experiment_name = experiment_name,
            outputs_dir     = outputs_dir,
        )
        if verbose:
            print(f"  Confusion matrix saved → {cm_path}")
        metrics["confusion_matrix_path"] = cm_path

    # Add experiment metadata to the result
    metrics["experiment"] = experiment_name
    metrics["backbone"]   = backbone
    metrics["dropout"]    = exp_config["regularization"]["dropout_rate"]
    metrics["label_smooth"] = exp_config["train"]["label_smoothing"]
    metrics["weight_decay"] = exp_config["train"]["weight_decay"]

    return metrics


# ──────────────────────────────────────────────
#  evaluate_all()
#
#  Evaluates all experiments in a list and builds
#  the complete comparison table for the paper.
# ──────────────────────────────────────────────
def evaluate_all(
    experiment_names: list = None,
    device:           torch.device = None,
    experiments_dir:  str = None,
    data_dir:         str = None,
    outputs_dir:      str = None,
    cm_experiments:   list = None,
) -> pd.DataFrame:
    """
    Evaluates all experiments and saves a comparison table.

    Args:
        experiment_names : List of experiment folder names.
                           If None, uses the default 8 experiments.
        device           : torch.device (auto-detected if None)
        experiments_dir  : Path to experiments/ folder
        data_dir         : Path to data/ folder
        outputs_dir      : Where to save outputs
        cm_experiments   : Which experiments to plot confusion
                           matrices for. If None, plots top 3.

    Returns:
        pandas DataFrame with all results — one row per experiment

    Usage:
        from src.evaluate import evaluate_all
        df = evaluate_all()
        print(df.to_string(index=False))
    """
    if experiments_dir is None:
        experiments_dir = os.path.join(PROJECT_ROOT, "experiments")
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")
    if outputs_dir is None:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default: evaluate all 8 Phase 3 experiments
    if experiment_names is None:
        experiment_names = [
            "baseline_efficientnet_b3",
            "baseline_resnet50",
            "baseline_densenet121",
            "baseline_vgg16",
            "ablation_dropout_0",
            "ablation_dropout_5",
            "ablation_no_label_smooth",
            "ablation_high_weight_decay",
        ]

    # Default: plot confusion matrix for top 3 backbone baselines
    if cm_experiments is None:
        cm_experiments = [
            "baseline_vgg16",
            "baseline_densenet121",
            "baseline_resnet50",
        ]

    all_results = []

    for exp_name in experiment_names:
        exp_dir = os.path.join(experiments_dir, exp_name)
        if not os.path.exists(exp_dir):
            print(f"  WARNING: {exp_name} not found — skipping")
            continue

        plot_cm = exp_name in cm_experiments

        try:
            metrics = evaluate_experiment(
                experiment_name = exp_name,
                device          = device,
                experiments_dir = experiments_dir,
                data_dir        = data_dir,
                outputs_dir     = outputs_dir,
                plot_cm         = plot_cm,
                verbose         = True,
            )
            all_results.append(metrics)
        except Exception as e:
            print(f"  ERROR evaluating {exp_name}: {e}")
            continue

    if not all_results:
        print("No experiments were evaluated successfully.")
        return pd.DataFrame()

    # ── Build comparison DataFrame
    rows = []
    for r in all_results:
        rows.append({
            "Experiment":    r["experiment"],
            "Backbone":      r["backbone"],
            "Dropout":       r["dropout"],
            "Label Smooth":  r["label_smooth"],
            "Weight Decay":  r["weight_decay"],
            "Accuracy (%)":  r["accuracy"],
            "Macro F1":      r["macro_f1"],
            "Macro AUC":     r["macro_auc"],
            "F1 Glioma":     r["f1_glioma"],
            "F1 Meningioma": r["f1_meningioma"],
            "F1 Notumor":    r["f1_notumor"],
            "F1 Pituitary":  r["f1_pituitary"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Macro F1", ascending=False).reset_index(drop=True)

    # ── Print final summary table
    print("\n" + "=" * 80)
    print("  PHASE 4 — Test Set Results (all experiments)")
    print("=" * 80)
    summary_cols = [
        "Experiment", "Backbone", "Accuracy (%)",
        "Macro F1", "Macro AUC",
        "F1 Glioma", "F1 Meningioma", "F1 Notumor", "F1 Pituitary"
    ]
    print(df[summary_cols].to_string(index=False))
    print("=" * 80)

    # ── Highlight the hardest class
    worst_class = df[["F1 Glioma","F1 Meningioma","F1 Notumor","F1 Pituitary"]].mean().idxmin()
    print(f"\n  Hardest class across all models: {worst_class}")
    print(f"  Average F1: {df[worst_class].mean():.4f}")

    # ── Save results
    os.makedirs(outputs_dir, exist_ok=True)
    csv_path = os.path.join(outputs_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    # ── Save full classification reports to text file
    report_path = os.path.join(outputs_dir, "classification_reports.txt")
    with open(report_path, "w") as f:
        for r in all_results:
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment: {r['experiment']}\n")
            f.write(f"Backbone  : {r['backbone']}\n")
            f.write(f"Accuracy  : {r['accuracy']:.2f}%\n")
            f.write(f"Macro F1  : {r['macro_f1']:.4f}\n")
            f.write(f"Macro AUC : {r['macro_auc']}\n")
            f.write(f"{'='*60}\n")
            f.write(r["report"])
    print(f"  Reports saved → {report_path}")
    print("\n  Phase 4 evaluation complete.")

    return df
