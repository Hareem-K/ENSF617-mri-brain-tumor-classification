# ============================================================
#  config.py — Central Configuration
#  Brain Tumor MRI Classification Project
#
#  THIS IS THE SINGLE SOURCE OF TRUTH.
#  All paths, hyperparameters, and experiment settings
#  live here. To run a new experiment, you only ever
#  change values in this file — never inside train.py
#  or any other source file.
#
#  HOW TO USE:
#  from config import get_default_config, get_experiment_config
#
#  Default experiment:
#    cfg = get_default_config()
#
#  Custom experiment:
#    cfg = get_experiment_config(
#        name     = "baseline_resnet50",
#        backbone = "resnet50",
#        dropout  = 0.3,
#    )
# ============================================================

import os
import yaml
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
#  Project root detection
#
#  Automatically finds the project root whether
#  you are running on Colab or locally.
# ──────────────────────────────────────────────
def _find_project_root() -> str:
    """
    Detects the project root directory automatically.
    - On Colab: /content/drive/MyDrive/brain_tumor_classification
    - Locally:  the directory containing this config.py file
    """
    colab_path = "/content/drive/MyDrive/brain_tumor_classification"
    if os.path.exists(colab_path):
        return colab_path
    # Fallback: use the directory of this file
    return os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = _find_project_root()


# ──────────────────────────────────────────────
#  PathConfig
#  All folder locations in one place.
#  Never hardcode paths anywhere else.
# ──────────────────────────────────────────────
@dataclass
class PathConfig:
    # Project root (auto-detected above)
    root: str = PROJECT_ROOT

    # Dataset folders — match Kaggle download structure exactly
    data_dir:  str = os.path.join(PROJECT_ROOT, "data")
    train_dir: str = os.path.join(PROJECT_ROOT, "data", "Training")
    test_dir:  str = os.path.join(PROJECT_ROOT, "data", "Testing")

    # Where model checkpoints + metrics are saved
    # One subfolder per experiment is created automatically
    experiments_dir: str = os.path.join(PROJECT_ROOT, "experiments")

    # Where visualizations are saved (EDA charts, Grad-CAM maps)
    outputs_dir: str = os.path.join(PROJECT_ROOT, "outputs")

    # Training log files
    logs_dir: str = os.path.join(PROJECT_ROOT, "logs")

    def __post_init__(self):
        # Create all directories if they do not exist
        for d in [self.experiments_dir, self.outputs_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
#  DataConfig
#  Everything about how data is loaded.
# ──────────────────────────────────────────────
@dataclass
class DataConfig:
    # Class names — must match folder names exactly
    class_names: List[str] = field(default_factory=lambda: [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    ])

    num_classes: int = 4

    # Input image size fed to the model
    # 224 works for all 4 backbones
    image_size: int = 224

    # Fraction of Training/ set reserved for validation
    # 0.2 = 20% val, 80% train
    val_split: float = 0.2

    # DataLoader settings
    batch_size:  int  = 32
    num_workers: int  = 2      # 2 is stable on Colab
    pin_memory:  bool = True   # speeds up GPU data transfer

    # ImageNet normalization — required for pretrained models
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Fixed seed for reproducible train/val split
    seed: int = 42


# ──────────────────────────────────────────────
#  TrainConfig
#  All training hyperparameters.
#
#  PHASED FINE-TUNING TIMELINE:
#  Epochs 1  → warmup_epochs      : backbone frozen (Phase 1)
#  Epochs warmup+1 → full_finetune: top layers open (Phase 2)
#  Epochs full_finetune+1 → end   : all layers open (Phase 3)
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Total training epochs (early stopping may end it sooner)
    epochs: int = 30

    # ── Phase 1: Warmup
    # Backbone frozen. Only classifier head trains.
    warmup_epochs: int   = 5
    warmup_lr:     float = 1e-3

    # ── Phase 2: Fine-tuning
    # Top 2 backbone blocks unfrozen.
    finetune_lr: float = 1e-4

    # ── Phase 3: Full fine-tuning
    # All layers unfrozen. Very low LR.
    full_finetune_epoch: int   = 20
    full_finetune_lr:    float = 1e-5

    # ── Optimizer
    optimizer:    str   = "adamw"
    weight_decay: float = 1e-4    # L2 regularization

    # ── LR Scheduler (cosine annealing)
    scheduler: str   = "cosine"
    lr_min:    float = 1e-6       # LR never goes below this

    # ── Loss function
    label_smoothing: float = 0.1  # 0.0 = hard targets, 0.1 = soft

    # ── Early stopping
    # Stops training if val F1 does not improve for N epochs
    patience:  int   = 7
    min_delta: float = 1e-4       # minimum improvement to count

    # ── Mixed precision training
    # Roughly 2x faster on Colab T4 GPU — keep True
    use_amp: bool = True

    # ── What metric to monitor for best model saving
    monitor: str = "val_f1"


# ──────────────────────────────────────────────
#  RegularizationConfig
#  Isolated here so ablation experiments only
#  need to change this section.
# ──────────────────────────────────────────────
@dataclass
class RegularizationConfig:
    # Dropout applied before the final classifier layer
    # Ablation values: [0.0, 0.3, 0.5]
    dropout_rate: float = 0.3

    # MixUp augmentation (advanced — off by default)
    use_mixup:   bool  = False
    mixup_alpha: float = 0.2

    # CutMix augmentation (advanced — off by default)
    use_cutmix:   bool  = False
    cutmix_alpha: float = 1.0


# ──────────────────────────────────────────────
#  ModelConfig
#  Which backbone to use and its settings.
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Backbone architecture
    # Options: "efficientnet_b3", "resnet50",
    #          "densenet121", "vgg16"
    backbone: str = "efficientnet_b3"

    # Load ImageNet pretrained weights
    pretrained: bool = True

    # Number of output classes
    num_classes: int = 4


# ──────────────────────────────────────────────
#  ExperimentConfig
#  Ties everything together into one object.
#  One instance = one training run.
# ──────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    # Human-readable name for this run
    # This becomes the folder name under experiments/
    name: str = "baseline_efficientnet_b3"

    paths:          PathConfig          = field(default_factory=PathConfig)
    data:           DataConfig          = field(default_factory=DataConfig)
    train:          TrainConfig         = field(default_factory=TrainConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    model:          ModelConfig         = field(default_factory=ModelConfig)

    def experiment_dir(self) -> str:
        """
        Returns the directory for this specific run.
        Creates it if it does not exist.
        Example: experiments/baseline_efficientnet_b3/
        """
        run_dir = os.path.join(self.paths.experiments_dir, self.name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save(self, path: str = None) -> None:
        """
        Saves this config as a YAML file inside the
        experiment directory. Allows full reproducibility —
        you can always see exactly what settings were used.

        Args:
            path : Optional custom path. Defaults to
                   experiments/{name}/config.yaml
        """
        if path is None:
            path = os.path.join(self.experiment_dir(), "config.yaml")

        config_dict = {
            "name": self.name,
            "model": {
                "backbone":     self.model.backbone,
                "pretrained":   self.model.pretrained,
                "num_classes":  self.model.num_classes,
            },
            "regularization": {
                "dropout_rate": self.regularization.dropout_rate,
                "use_mixup":    self.regularization.use_mixup,
                "use_cutmix":   self.regularization.use_cutmix,
            },
            "data": {
                "image_size":  self.data.image_size,
                "batch_size":  self.data.batch_size,
                "val_split":   self.data.val_split,
                "seed":        self.data.seed,
                "num_workers": self.data.num_workers,
            },
            "train": {
                "epochs":            self.train.epochs,
                "warmup_epochs":     self.train.warmup_epochs,
                "warmup_lr":         self.train.warmup_lr,
                "finetune_lr":       self.train.finetune_lr,
                "full_finetune_lr":  self.train.full_finetune_lr,
                "full_finetune_epoch": self.train.full_finetune_epoch,
                "optimizer":         self.train.optimizer,
                "weight_decay":      self.train.weight_decay,
                "scheduler":         self.train.scheduler,
                "lr_min":            self.train.lr_min,
                "label_smoothing":   self.train.label_smoothing,
                "patience":          self.train.patience,
                "use_amp":           self.train.use_amp,
            },
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"  Config saved → {path}")


# ──────────────────────────────────────────────
#  Ready-made experiment configs
#  Use these directly in your notebook cells.
# ──────────────────────────────────────────────

def get_default_config() -> ExperimentConfig:
    """
    Returns the default config — EfficientNetB3 baseline.
    This is the first experiment you run.
    """
    return ExperimentConfig(
        name  = "baseline_efficientnet_b3",
        model = ModelConfig(backbone="efficientnet_b3"),
    )


def get_experiment_config(
    name:         str,
    backbone:     str   = "efficientnet_b3",
    dropout:      float = 0.3,
    weight_decay: float = 1e-4,
    label_smooth: float = 0.1,
    epochs:       int   = 30,
) -> ExperimentConfig:
    """
    Creates a custom experiment config.
    Use this for ablation experiments.

    Args:
        name         : Experiment name (becomes folder name)
        backbone     : One of efficientnet_b3, resnet50,
                       densenet121, vgg16
        dropout      : Dropout rate (0.0, 0.3, 0.5)
        weight_decay : L2 regularization (1e-4, 1e-3)
        label_smooth : Label smoothing (0.0, 0.1)
        epochs       : Max training epochs

    Returns:
        ExperimentConfig ready to pass to main()

    Usage:
        cfg = get_experiment_config(
            name     = "ablation_dropout_0",
            backbone = "efficientnet_b3",
            dropout  = 0.0,
        )
        from src.train import main
        main(config=cfg)
    """
    cfg = ExperimentConfig(name=name)
    cfg.model.backbone                  = backbone
    cfg.regularization.dropout_rate     = dropout
    cfg.train.weight_decay              = weight_decay
    cfg.train.label_smoothing           = label_smooth
    cfg.train.epochs                    = epochs
    return cfg


# ──────────────────────────────────────────────
#  All planned experiments for Phase 3
#  Import this list in your notebook and iterate.
# ──────────────────────────────────────────────
BASELINE_EXPERIMENTS = [
    {
        "name":     "baseline_efficientnet_b3",
        "backbone": "efficientnet_b3",
        "dropout":  0.3,
        "wd":       1e-4,
        "smooth":   0.1,
    },
    {
        "name":     "baseline_resnet50",
        "backbone": "resnet50",
        "dropout":  0.3,
        "wd":       1e-4,
        "smooth":   0.1,
    },
    {
        "name":     "baseline_densenet121",
        "backbone": "densenet121",
        "dropout":  0.3,
        "wd":       1e-4,
        "smooth":   0.1,
    },
    {
        "name":     "baseline_vgg16",
        "backbone": "vgg16",
        "dropout":  0.3,
        "wd":       1e-4,
        "smooth":   0.1,
    },
]

ABLATION_EXPERIMENTS = [
    # Dropout ablation (using best backbone — update after Part A)
    {
        "name":     "ablation_dropout_0",
        "backbone": "efficientnet_b3",  # update to best backbone
        "dropout":  0.0,
        "wd":       1e-4,
        "smooth":   0.1,
    },
    {
        "name":     "ablation_dropout_5",
        "backbone": "efficientnet_b3",
        "dropout":  0.5,
        "wd":       1e-4,
        "smooth":   0.1,
    },
    # Label smoothing ablation
    {
        "name":     "ablation_no_label_smooth",
        "backbone": "efficientnet_b3",
        "dropout":  0.3,
        "wd":       1e-4,
        "smooth":   0.0,
    },
    # Weight decay ablation
    {
        "name":     "ablation_high_weight_decay",
        "backbone": "efficientnet_b3",
        "dropout":  0.3,
        "wd":       1e-3,
        "smooth":   0.1,
    },
]


# ──────────────────────────────────────────────
#  Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing config.py...")
    print("-" * 48)

    cfg = get_default_config()
    print(f"  Name          : {cfg.name}")
    print(f"  Project root  : {cfg.paths.root}")
    print(f"  Data dir      : {cfg.paths.data_dir}")
    print(f"  Experiments   : {cfg.paths.experiments_dir}")
    print(f"  Backbone      : {cfg.model.backbone}")
    print(f"  Image size    : {cfg.data.image_size}")
    print(f"  Batch size    : {cfg.data.batch_size}")
    print(f"  Epochs        : {cfg.train.epochs}")
    print(f"  Dropout       : {cfg.regularization.dropout_rate}")
    print(f"  Label smooth  : {cfg.train.label_smoothing}")
    print(f"  Weight decay  : {cfg.train.weight_decay}")
    print()
    print(f"  Baseline experiments  : {len(BASELINE_EXPERIMENTS)}")
    print(f"  Ablation experiments  : {len(ABLATION_EXPERIMENTS)}")
    print()
    cfg.save()
    print("config.py is working correctly.")
