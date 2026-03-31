# ============================================================
#  src/utils/logger.py — Experiment Logger
#  Brain Tumor MRI Classification Project
#
#  Provides consistent logging to both console and file
#  across all training scripts. Creates one log file per
#  experiment run, named with a timestamp.
# ============================================================

import os
import sys
import logging
from datetime import datetime
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates and returns a logger that writes to both console
    and a log file in the experiment directory.

    Args:
        name       : Logger name (usually the experiment name)
        log_dir    : Directory to save the .log file. If None,
                     logs to console only.
        level      : Logging level (default: INFO)

    Returns:
        Configured Logger instance

    Usage:
        logger = get_logger("baseline_resnet50", log_dir="experiments/run1")
        logger.info("Training started")
        logger.warning("Learning rate is very low")
        logger.error("CUDA out of memory")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (always active)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (only if log_dir is provided)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")

    return logger


class MetricLogger:
    """
    Tracks and logs training metrics (loss, accuracy, F1)
    across epochs. Saves a CSV for easy plotting later.

    Usage:
        tracker = MetricLogger(log_dir="experiments/run1")
        tracker.update(epoch=1, phase="train", loss=0.82, acc=0.71, f1=0.69)
        tracker.update(epoch=1, phase="val",   loss=0.74, acc=0.76, f1=0.74)
        tracker.save()       # Saves metrics.csv
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records = []

    def update(
        self,
        epoch: int,
        phase: str,
        loss: float,
        acc: float,
        f1: float,
        auc: Optional[float] = None,
    ) -> None:
        record = {
            "epoch": epoch,
            "phase": phase,
            "loss": round(loss, 6),
            "accuracy": round(acc, 6),
            "f1": round(f1, 6),
        }
        if auc is not None:
            record["auc"] = round(auc, 6)
        self.records.append(record)

    def save(self, filename: str = "metrics.csv") -> str:
        """Saves all tracked metrics to a CSV file."""
        import csv
        path = os.path.join(self.log_dir, filename)
        if not self.records:
            return path
        keys = self.records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        return path

    def get_best(self, metric: str = "f1", phase: str = "val") -> dict:
        """Returns the record with the best value for a given metric."""
        phase_records = [r for r in self.records if r["phase"] == phase]
        if not phase_records:
            return {}
        return max(phase_records, key=lambda r: r.get(metric, 0))

    def print_summary(self) -> None:
        """Prints a table of all recorded epochs."""
        if not self.records:
            print("No metrics recorded yet.")
            return

        header = f"{'Epoch':>6} | {'Phase':>6} | {'Loss':>8} | {'Accuracy':>8} | {'F1':>8}"
        print("\n" + "─" * len(header))
        print(header)
        print("─" * len(header))
        for r in self.records:
            auc_str = f" | AUC {r['auc']:.4f}" if "auc" in r else ""
            print(
                f"{r['epoch']:>6} | {r['phase']:>6} | "
                f"{r['loss']:>8.4f} | {r['accuracy']:>8.4f} | "
                f"{r['f1']:>8.4f}{auc_str}"
            )
        print("─" * len(header))
        best = self.get_best("f1", "val")
        if best:
            print(f"\n  Best val F1: {best['f1']:.4f} at epoch {best['epoch']}")


if __name__ == "__main__":
    # Quick test
    logger = get_logger("test_run", log_dir="/tmp/test_logs")
    logger.info("Logger is working correctly.")
    logger.warning("This is a warning.")

    tracker = MetricLogger(log_dir="/tmp/test_logs")
    for epoch in range(1, 4):
        tracker.update(epoch, "train", loss=1.0 - epoch * 0.1, acc=0.6 + epoch * 0.05, f1=0.58 + epoch * 0.05)
        tracker.update(epoch, "val",   loss=1.1 - epoch * 0.1, acc=0.58 + epoch * 0.05, f1=0.56 + epoch * 0.05)
    tracker.print_summary()
    path = tracker.save()
    print(f"\nMetrics saved to: {path}")
