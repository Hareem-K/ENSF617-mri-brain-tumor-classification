# ============================================================
#  src/train.py — Training loop for brain tumor classification
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS FILE DOES:
#  1. Builds the model from config.py
#  2. Trains with train / validation DataLoaders
#  3. Saves best checkpoint and per-epoch metrics.csv
#  4. Supports a beginner-friendly warmup → fine-tune flow
#
#  Phase schedule used here:
#    - Warmup phase         : classifier head only
#    - Partial fine-tuning  : last few backbone blocks
#    - Full fine-tuning     : all layers
# ============================================================

from __future__ import annotations

import copy
import csv
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm.auto import tqdm

from src.models import build_model, config_to_dict


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _phase_name(cfg: Any, epoch: int) -> str:
    if epoch <= max(0, cfg.train.warmup_epochs):
        return 'warmup'
    if epoch < cfg.train.full_finetune_epoch:
        return 'partial_finetune'
    return 'full_finetune'


def _phase_lr(cfg: Any, phase: str) -> float:
    if phase == 'warmup':
        return cfg.train.warmup_lr
    if phase == 'partial_finetune':
        return cfg.train.finetune_lr
    return cfg.train.full_finetune_lr


def _configure_phase(model: nn.Module, cfg: Any, phase: str) -> None:
    if phase == 'warmup':
        model.freeze_backbone()
    elif phase == 'partial_finetune':
        model.unfreeze_top_blocks(n_blocks=2)
    else:
        model.unfreeze_backbone()


def _make_optimizer(cfg: Any, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.train.optimizer.lower()

    if name == 'adam':
        return Adam(params, lr=lr, weight_decay=cfg.train.weight_decay)
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
    if name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=cfg.train.weight_decay, nesterov=True)

    raise ValueError(f'Unsupported optimizer: {cfg.train.optimizer}')


def _make_scheduler(cfg: Any, optimizer: torch.optim.Optimizer, remaining_epochs: int):
    name = cfg.train.scheduler.lower()
    remaining_epochs = max(1, remaining_epochs)

    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=cfg.train.lr_min)
    if name == 'step':
        step_size = max(1, remaining_epochs // 2)
        return StepLR(optimizer, step_size=step_size, gamma=0.1)
    if name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    raise ValueError(f'Unsupported scheduler: {cfg.train.scheduler}')


def _phase_end_epoch(cfg: Any, phase: str) -> int:
    if phase == 'warmup':
        return min(cfg.train.epochs, max(1, cfg.train.warmup_epochs))
    if phase == 'partial_finetune':
        return min(cfg.train.epochs, max(cfg.train.warmup_epochs + 1, cfg.train.full_finetune_epoch - 1))
    return cfg.train.epochs


def _compute_metrics(loss_sum: float, total: int, targets: List[int], preds: List[int]) -> Dict[str, float]:
    avg_loss = loss_sum / max(1, total)
    acc = 100.0 * sum(int(p == t) for p, t in zip(preds, targets)) / max(1, total)
    f1 = 100.0 * f1_score(targets, preds, average='macro') if targets else 0.0
    return {
        'loss': avg_loss,
        'acc': acc,
        'f1': f1,
    }


def _run_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    train: bool = True,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    targets: List[int] = []
    preds: List[int] = []
    loss_sum = 0.0
    total = 0

    pbar = tqdm(loader, leave=False)
    pbar.set_description('Train' if train else 'Val')

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train and optimizer is not None:
                if scaler is not None and use_amp and device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_preds = outputs.argmax(dim=1)
        targets.extend(labels.detach().cpu().tolist())
        preds.extend(batch_preds.detach().cpu().tolist())
        loss_sum += loss.item() * batch_size
        total += batch_size

        current = _compute_metrics(loss_sum, total, targets, preds)
        pbar.set_postfix(loss=f"{current['loss']:.4f}", acc=f"{current['acc']:.2f}")

    return _compute_metrics(loss_sum, total, targets, preds)


def _save_metrics_csv(history: List[Dict[str, float]], csv_path: str) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _metric_value(metrics: Dict[str, float], monitor: str) -> float:
    if monitor not in metrics:
        raise KeyError(f"Monitor '{monitor}' not found in metrics: {list(metrics.keys())}")
    return metrics[monitor]


def _is_better(current: float, best: Optional[float], monitor: str, min_delta: float) -> bool:
    if best is None:
        return True
    if monitor == 'val_loss':
        return current < (best - min_delta)
    return current > (best + min_delta)


def train_model(
    cfg: Any,
    train_loader,
    val_loader,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, List[Dict[str, float]], str]:
    """
    Main training entry point.

    Returns:
        model         : model loaded with best checkpoint weights
        history       : list of per-epoch metric dictionaries
        best_ckpt     : path to saved best_model.pth
    """
    set_seed(cfg.data.seed)
    device = device or get_device()

    exp_dir = Path(cfg.experiment_dir())
    os.makedirs(exp_dir, exist_ok=True)
    cfg.save(exp_dir / 'config.yaml')

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.use_amp and device.type == 'cuda'))

    history: List[Dict[str, float]] = []
    best_metric: Optional[float] = None
    best_checkpoint_path = str(exp_dir / 'best_model.pth')
    metrics_csv_path = str(exp_dir / 'metrics.csv')
    best_state_dict = None
    patience_counter = 0

    optimizer = None
    scheduler = None
    current_phase = None

    print('=' * 70)
    print('Starting training')
    print(f'Experiment     : {cfg.name}')
    print(f'Backbone       : {cfg.model.backbone}')
    print(f'Device         : {device}')
    print(f'Monitor        : {cfg.train.monitor}')
    print(f'Experiment dir : {exp_dir}')
    print('=' * 70)

    for epoch in range(1, cfg.train.epochs + 1):
        phase = _phase_name(cfg, epoch)
        if phase != current_phase:
            current_phase = phase
            _configure_phase(model, cfg, current_phase)
            phase_lr = _phase_lr(cfg, current_phase)
            phase_end = _phase_end_epoch(cfg, current_phase)
            optimizer = _make_optimizer(cfg, model, lr=phase_lr)
            scheduler = _make_scheduler(cfg, optimizer, remaining_epochs=(phase_end - epoch + 1))
            print()
            print(f'[Phase change] epoch={epoch}  phase={current_phase}  lr={phase_lr}')
            print(f'Trainable params: {model.num_trainable_parameters():,} / {model.num_total_parameters():,}')

        start = time.time()
        train_stats = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=cfg.train.use_amp,
            scaler=scaler,
            train=True,
        )
        val_stats = _run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            use_amp=cfg.train.use_amp,
            scaler=None,
            train=False,
        )

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_stats['f1'])
            else:
                scheduler.step()

        epoch_metrics = {
            'epoch': epoch,
            'phase': current_phase,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': round(train_stats['loss'], 6),
            'train_acc': round(train_stats['acc'], 4),
            'train_f1': round(train_stats['f1'], 4),
            'val_loss': round(val_stats['loss'], 6),
            'val_acc': round(val_stats['acc'], 4),
            'val_f1': round(val_stats['f1'], 4),
            'epoch_time_sec': round(time.time() - start, 2),
        }
        history.append(epoch_metrics)
        _save_metrics_csv(history, metrics_csv_path)

        current_value = _metric_value(epoch_metrics, cfg.train.monitor)
        improved = _is_better(current_value, best_metric, cfg.train.monitor, cfg.train.min_delta)

        print(
            f"Epoch {epoch:02d}/{cfg.train.epochs:02d} | "
            f"train_loss={epoch_metrics['train_loss']:.4f} | "
            f"val_loss={epoch_metrics['val_loss']:.4f} | "
            f"val_acc={epoch_metrics['val_acc']:.2f}% | "
            f"val_f1={epoch_metrics['val_f1']:.2f}%"
        )

        if improved:
            best_metric = current_value
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'monitor': cfg.train.monitor,
                    'model_state_dict': best_state_dict,
                    'config': config_to_dict(cfg),
                },
                best_checkpoint_path,
            )
            print(f'  -> Best model updated and saved to {best_checkpoint_path}')
        else:
            patience_counter += 1
            print(f'  -> No improvement. Early-stop counter: {patience_counter}/{cfg.train.patience}')

        if patience_counter >= cfg.train.patience:
            print()
            print('Early stopping triggered.')
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print()
    print('Training complete.')
    print(f'Best checkpoint : {best_checkpoint_path}')
    print(f'Metrics CSV      : {metrics_csv_path}')
    return model, history, best_checkpoint_path


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate a trained model on any loader (val or test)."""
    device = device or get_device()
    model = model.to(device)
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    targets: List[int] = []
    preds: List[int] = []
    loss_sum = 0.0
    total = 0

    for images, labels in tqdm(loader, leave=False, desc='Eval'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        total += batch_size
        targets.extend(labels.detach().cpu().tolist())
        preds.extend(batch_preds.detach().cpu().tolist())

    metrics = _compute_metrics(loss_sum, total, targets, preds)
    return {
        'loss': round(metrics['loss'], 6),
        'acc': round(metrics['acc'], 4),
        'f1': round(metrics['f1'], 4),
    }


@torch.no_grad()
def load_checkpoint(model: nn.Module, checkpoint_path: str, device: Optional[torch.device] = None) -> nn.Module:
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
