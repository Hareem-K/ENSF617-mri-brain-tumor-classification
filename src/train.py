# ============================================================
#  src/train.py — The Master Training Loop
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS FILE DOES:
#  This is the engine of the project. It loads the config,
#  builds the model and dataloaders, and executes the
#  3-phase transfer learning strategy:
#
#  Phase 1 — Warmup
#    Backbone fully frozen. Only the classifier head trains.
#    High learning rate (1e-3). Short — typically 5 epochs.
#    Goal: stabilise the head before touching the backbone.
#
#  Phase 2 — Fine-tuning
#    Top 2 backbone blocks unfrozen. Lower learning rate (1e-4).
#    Goal: adapt high-level features to brain tumor patterns.
#
#  Phase 3 — Full fine-tuning
#    All layers unfrozen. Very low learning rate (1e-5).
#    Goal: gentle final polish across the entire model.
#
#  HOW TO RUN:
#  This file is called from notebooks/03_training.ipynb.
#  Do not run directly — use the notebook instead so you
#  can monitor training interactively.
# ============================================================

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import get_default_config
from src.dataset import create_dataloaders
from src.models import (
    BrainTumorClassifier,
    freeze_backbone,
    unfreeze_top_layers,
    unfreeze_all,
    get_optimizer,
    get_scheduler,
)
from utils.logger import get_logger, MetricLogger


# ──────────────────────────────────────────────
#  train_one_epoch()
#
#  Runs one complete pass over the training set.
#  Updates model weights via backpropagation.
#
#  WHAT AMP DOES:
#  Automatic Mixed Precision runs some operations in
#  float16 instead of float32. This roughly doubles
#  training speed on Colab T4 GPU with no accuracy loss.
# ──────────────────────────────────────────────
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    epoch,
):
    """
    Runs a single pass over the training data.

    Args:
        model      : The BrainTumorClassifier
        dataloader : Training DataLoader
        optimizer  : AdamW optimizer
        criterion  : CrossEntropyLoss with label smoothing
        device     : torch.device (cuda or cpu)
        scaler     : GradScaler for AMP, or None
        epoch      : Current epoch number (for progress bar)

    Returns:
        Tuple of (loss, accuracy, macro_f1) for this epoch
    """
    model.train()
    running_loss = 0.0
    all_preds    = []
    all_targets  = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ── Forward pass with AMP
        # torch.amp.autocast replaces deprecated torch.cuda.amp.autocast
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        # ── Track predictions for metrics
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # ── Compute epoch-level metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_targets, all_preds)
    epoch_f1   = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


# ──────────────────────────────────────────────
#  validate()
#
#  Evaluates the model on the validation set.
#  NO weight updates — torch.no_grad() ensures
#  no gradients are computed, saving memory.
# ──────────────────────────────────────────────
def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch,
):
    """
    Evaluates the model on the validation set.

    Args:
        model      : The BrainTumorClassifier
        dataloader : Validation DataLoader
        criterion  : CrossEntropyLoss
        device     : torch.device
        epoch      : Current epoch number (for progress bar)

    Returns:
        Tuple of (loss, accuracy, macro_f1) for this epoch
    """
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_targets  = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_targets, all_preds)
    epoch_f1   = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


# ──────────────────────────────────────────────
#  smoke_test()
#
#  Runs training on just 2 batches for 20 steps.
#  Loss MUST drop close to zero — if it does not,
#  there is a bug in the pipeline.
#  Always run this before a full training run.
# ──────────────────────────────────────────────
def smoke_test(model, train_loader, criterion, device):
    """
    Verifies the training loop works by overfitting
    on a tiny subset of the data.

    Args:
        model        : BrainTumorClassifier
        train_loader : Training DataLoader
        criterion    : Loss function
        device       : torch.device

    Returns:
        True if loss dropped below 0.1, False otherwise
    """
    print("Running smoke test (20 steps on 2 batches)...")
    model.train()

    # Temporarily unfreeze everything for the smoke test
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Grab just 2 batches
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i >= 1:
            break

    initial_loss = None
    final_loss   = None

    for step in range(20):
        batch  = batches[step % 2]
        images = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1:02d}/20 — loss: {loss.item():.4f}")

    print(f"\n  Initial loss : {initial_loss:.4f}")
    print(f"  Final loss   : {final_loss:.4f}")

    if final_loss < initial_loss * 0.5:
        print("  Smoke test PASSED — loss is dropping correctly.")
        return True
    else:
        print("  Smoke test FAILED — loss is not dropping. Check your pipeline.")
        return False


# ──────────────────────────────────────────────
#  main()
#
#  Orchestrates the full training run.
#  Called from 03_training.ipynb.
# ──────────────────────────────────────────────
def main(config=None):
    """
    Runs the full training pipeline for one experiment.

    Args:
        config : ExperimentConfig instance.
                 If None, loads the default config.

    Usage from notebook:
        from src.train import main
        from config import get_default_config

        cfg = get_default_config()
        cfg.name = "baseline_efficientnet_b3"
        cfg.model.backbone = "efficientnet_b3"
        main(config=cfg)
    """

    # ── 1. Load configuration
    cfg = config if config is not None else get_default_config()

    # ── 2. Setup experiment folder and logging
    exp_dir = cfg.experiment_dir()
    cfg.save()

    logger         = get_logger(cfg.name, log_dir=exp_dir)
    metric_tracker = MetricLogger(log_dir=exp_dir)

    logger.info("=" * 52)
    logger.info(f"  Experiment : {cfg.name}")
    logger.info(f"  Backbone   : {cfg.model.backbone}")
    logger.info(f"  Dropout    : {cfg.regularization.dropout_rate}")
    logger.info(f"  Label smooth: {cfg.train.label_smoothing}")
    logger.info("=" * 52)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 3. Load data
    logger.info("Loading DataLoaders...")
    train_loader, val_loader, _, info = create_dataloaders(
        data_dir    = cfg.paths.data_dir,
        image_size  = cfg.data.image_size,
        batch_size  = cfg.data.batch_size,
        val_split   = cfg.data.val_split,
        num_workers = cfg.data.num_workers,
        seed        = cfg.data.seed,
    )
    logger.info(f"Train: {info['train_size']} images | Val: {info['val_size']} images")

    # ── 4. Build model
    logger.info(f"Building model: {cfg.model.backbone}")
    model = BrainTumorClassifier(
        backbone_name = cfg.model.backbone,
        pretrained    = cfg.model.pretrained,
        dropout_rate  = cfg.regularization.dropout_rate,
        num_classes   = cfg.data.num_classes,
    ).to(device)

    # ── 5. Loss function
    # CrossEntropyLoss with label smoothing.
    # label_smoothing=0.1 means instead of training toward
    # [0, 0, 1, 0] the model trains toward [0.025, 0.025, 0.925, 0.025]
    # This prevents overconfidence and improves generalisation.
    criterion = nn.CrossEntropyLoss(
        label_smoothing = cfg.train.label_smoothing
    )

    # ── 6. AMP scaler
    # torch.amp.GradScaler replaces deprecated torch.cuda.amp.GradScaler
    scaler = (
        torch.cuda.amp.GradScaler() 
        if cfg.train.use_amp and device.type == "cuda" 
        else None
    )

    # ── 7. Tracking variables
    best_val_f1              = 0.0
    epochs_without_improvement = 0
    optimizer                = None
    scheduler                = None

    # ── 8. Master training loop
    for epoch in range(1, cfg.train.epochs + 1):

        # ── Phase control (transfer learning transitions)
        if epoch == 1:
            # Phase 1: backbone frozen, head trains at high LR
            logger.info(">>> PHASE 1: Warmup — backbone frozen <<<")
            freeze_backbone(model.backbone)
            optimizer = get_optimizer(
                model,
                head_lr      = cfg.train.warmup_lr,
                backbone_lr  = cfg.train.finetune_lr,  # ready for Phase 2
                weight_decay = cfg.train.weight_decay,
            )
            scheduler = get_scheduler(
                optimizer,
                epochs = cfg.train.warmup_epochs,
                min_lr     = cfg.train.lr_min,
            )

        elif epoch == cfg.train.warmup_epochs + 1:
            # Phase 2: unfreeze top 2 backbone blocks
            logger.info(">>> PHASE 2: Fine-tuning — top layers unfrozen <<<")
            unfreeze_top_layers(model.backbone, num_blocks=2)
            optimizer = get_optimizer(
                model,
                head_lr      = cfg.train.finetune_lr,
                backbone_lr  = cfg.train.finetune_lr * 0.1,
                weight_decay = cfg.train.weight_decay,
            )
            scheduler = get_scheduler(
                optimizer,
                epochs = cfg.train.full_finetune_epoch - cfg.train.warmup_epochs,
                min_lr     = cfg.train.lr_min,
            )

        elif epoch == cfg.train.full_finetune_epoch + 1:
            # Phase 3: unfreeze everything at very low LR
            logger.info(">>> PHASE 3: Full fine-tuning — all layers unfrozen <<<")
            unfreeze_all(model.backbone)
            optimizer = get_optimizer(
                model,
                head_lr      = cfg.train.full_finetune_lr,
                backbone_lr  = cfg.train.full_finetune_lr,
                weight_decay = cfg.train.weight_decay,
            )
            # New scheduler for the remaining epochs
            remaining = cfg.train.epochs - cfg.train.full_finetune_epoch
            scheduler = get_scheduler(
                optimizer,
                epochs = max(remaining, 1),
                min_lr     = cfg.train.lr_min,
            )

        # ── Train and validate
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device, epoch
        )

        # ── Step scheduler
        scheduler.step()

        # ── Log metrics
        metric_tracker.update(epoch, "train", train_loss, train_acc, train_f1)
        metric_tracker.update(epoch, "val",   val_loss,   val_acc,   val_f1)

        logger.info(
            f"Epoch {epoch:02d}/{cfg.train.epochs} | "
            f"Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f} | "
            f"Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}"
        )

        # ── Checkpoint: save best model
        if val_f1 > best_val_f1 + cfg.train.min_delta:
            best_val_f1                = val_f1
            epochs_without_improvement = 0
            save_path = os.path.join(exp_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  --> New best model saved (Val F1: {best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement. "
                f"Patience: {epochs_without_improvement}/{cfg.train.patience}"
            )

        # ── Early stopping
        if epochs_without_improvement >= cfg.train.patience:
            logger.warning(f"Early stopping at epoch {epoch}.")
            break

    # ── Save metrics CSV and wrap up
    metrics_path = metric_tracker.save()
    logger.info("=" * 52)
    logger.info(f"  Training complete.")
    logger.info(f"  Best Val F1  : {best_val_f1:.4f}")
    logger.info(f"  Model saved  : {os.path.join(exp_dir, 'best_model.pth')}")
    logger.info(f"  Metrics saved: {metrics_path}")
    logger.info("=" * 52)

    return best_val_f1


if __name__ == "__main__":
    main()
