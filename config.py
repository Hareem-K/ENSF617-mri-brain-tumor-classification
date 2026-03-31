# ============================================================
#  config.py — Central Configuration
#  Brain Tumor MRI Classification Project
#
#  ALL paths, hyperparameters, and experiment settings live
#  here. To run a new experiment, change values in this file
#  only — no hunting through training scripts.
# ============================================================

import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
@dataclass
class PathConfig:
    # Root of the project (auto-detected, or override manually)
    root: str = os.path.dirname(os.path.abspath(__file__))

    # Dataset location (update this after downloading from Kaggle)
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # Subdirectories (should match Kaggle dataset structure)
    train_dir: str = os.path.join(data_dir, "Training")
    test_dir: str = os.path.join(data_dir, "Testing")

    # Where model checkpoints and logs are saved
    experiments_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

    # Grad-CAM output folder
    gradcam_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "gradcam")

    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.gradcam_dir, exist_ok=True)


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────
@dataclass
class DataConfig:
    # Class names — must match folder names in Training/ and Testing/
    class_names: List[str] = field(default_factory=lambda: [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    ])

    # Number of classes
    num_classes: int = 4

    # Image size fed to the model (H, W)
    image_size: int = 224          # Use 300 for EfficientNetB3

    # Fraction of Training set used for validation (stratified)
    val_split: float = 0.2

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 2           # 2 works well on Colab
    pin_memory: bool = True        # Speeds up GPU transfer

    # ImageNet normalization stats (required for pretrained models)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Random seed for reproducible train/val split
    seed: int = 42


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Number of full passes over the training data
    epochs: int = 30

    # Phase 1: freeze backbone, train head only
    warmup_epochs: int = 5
    warmup_lr: float = 1e-3

    # Phase 2: unfreeze top layers, fine-tune
    finetune_lr: float = 1e-4

    # Phase 3 (optional): full model fine-tune
    full_finetune_lr: float = 1e-5
    full_finetune_epoch: int = 20   # Epoch at which to unfreeze all layers

    # Optimizer
    optimizer: str = "adamw"       # Options: "adam", "adamw", "sgd"
    weight_decay: float = 1e-4     # L2 regularization strength

    # Learning rate scheduler
    scheduler: str = "cosine"      # Options: "cosine", "step", "plateau"
    lr_min: float = 1e-6           # Minimum LR for cosine annealing

    # Loss function
    label_smoothing: float = 0.1   # 0.0 = no smoothing

    # Early stopping
    patience: int = 7              # Stop if val loss doesn't improve for N epochs
    min_delta: float = 1e-4        # Minimum improvement to count as progress

    # Mixed precision (faster training on Colab GPU)
    use_amp: bool = True

    # Save best model based on this metric
    monitor: str = "val_f1"        # Options: "val_loss", "val_acc", "val_f1"


# ──────────────────────────────────────────────
#  Regularization (ablation study settings)
# ──────────────────────────────────────────────
@dataclass
class RegularizationConfig:
    # Dropout rate applied before the final classifier
    dropout_rate: float = 0.3      # Ablate: [0.0, 0.3, 0.5]

    # Weight decay already set in TrainConfig.weight_decay
    # Ablate: [0.0, 1e-4, 1e-3]

    # Label smoothing already set in TrainConfig.label_smoothing
    # Ablate: [0.0, 0.1]

    # Whether to use MixUp augmentation
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # Whether to use CutMix augmentation
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0


# ──────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Backbone architecture — choose one per experiment
    # Options: "resnet50", "vgg16", "efficientnet_b3",
    #          "densenet121", "inception_v3"
    backbone: str = "efficientnet_b3"

    # Whether to use ImageNet pretrained weights
    pretrained: bool = True

    # Dropout rate (pulled from RegularizationConfig)
    dropout_rate: float = 0.3

    # Number of output classes (pulled from DataConfig)
    num_classes: int = 4


# ──────────────────────────────────────────────
#  Experiment (ties everything together)
# ──────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    # Human-readable name for this run
    # Used to create the experiment folder under experiments/
    name: str = "baseline_efficientnet_b3"

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def experiment_dir(self) -> str:
        """Returns the directory for this specific run."""
        run_dir = os.path.join(self.paths.experiments_dir, self.name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save(self, path: Optional[str] = None) -> None:
        """Saves config as a YAML file alongside the experiment."""
        if path is None:
            path = os.path.join(self.experiment_dir(), "config.yaml")
        config_dict = {
            "name": self.name,
            "model": {
                "backbone": self.model.backbone,
                "pretrained": self.model.pretrained,
                "dropout_rate": self.regularization.dropout_rate,
                "num_classes": self.model.num_classes,
            },
            "data": {
                "image_size": self.data.image_size,
                "batch_size": self.data.batch_size,
                "val_split": self.data.val_split,
                "seed": self.data.seed,
            },
            "train": {
                "epochs": self.train.epochs,
                "warmup_epochs": self.train.warmup_epochs,
                "warmup_lr": self.train.warmup_lr,
                "finetune_lr": self.train.finetune_lr,
                "optimizer": self.train.optimizer,
                "weight_decay": self.train.weight_decay,
                "scheduler": self.train.scheduler,
                "label_smoothing": self.train.label_smoothing,
                "patience": self.train.patience,
                "use_amp": self.train.use_amp,
            },
            "regularization": {
                "dropout_rate": self.regularization.dropout_rate,
                "use_mixup": self.regularization.use_mixup,
                "use_cutmix": self.regularization.use_cutmix,
            },
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to: {path}")

    @classmethod
    def load(cls, yaml_path: str) -> "ExperimentConfig":
        """Loads a config from a YAML file (for reproducing a past experiment)."""
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
        cfg = cls(name=d.get("name", "loaded_experiment"))
        cfg.model.backbone = d["model"]["backbone"]
        cfg.model.pretrained = d["model"]["pretrained"]
        cfg.regularization.dropout_rate = d["model"]["dropout_rate"]
        cfg.data.image_size = d["data"]["image_size"]
        cfg.data.batch_size = d["data"]["batch_size"]
        cfg.data.val_split = d["data"]["val_split"]
        cfg.data.seed = d["data"]["seed"]
        cfg.train.epochs = d["train"]["epochs"]
        cfg.train.warmup_lr = d["train"]["warmup_lr"]
        cfg.train.finetune_lr = d["train"]["finetune_lr"]
        cfg.train.optimizer = d["train"]["optimizer"]
        cfg.train.weight_decay = d["train"]["weight_decay"]
        cfg.train.scheduler = d["train"]["scheduler"]
        cfg.train.label_smoothing = d["train"]["label_smoothing"]
        cfg.train.patience = d["train"]["patience"]
        cfg.regularization.use_mixup = d["regularization"]["use_mixup"]
        cfg.regularization.use_cutmix = d["regularization"]["use_cutmix"]
        return cfg


# ──────────────────────────────────────────────
#  Quick-access default config
# ──────────────────────────────────────────────
def get_default_config() -> ExperimentConfig:
    """Returns a default config ready for a first training run."""
    return ExperimentConfig()


if __name__ == "__main__":
    # Sanity check — print and save the default config
    cfg = get_default_config()
    print("=" * 50)
    print(f"  Experiment : {cfg.name}")
    print(f"  Backbone   : {cfg.model.backbone}")
    print(f"  Image size : {cfg.data.image_size}x{cfg.data.image_size}")
    print(f"  Batch size : {cfg.data.batch_size}")
    print(f"  Epochs     : {cfg.train.epochs}")
    print(f"  Dropout    : {cfg.regularization.dropout_rate}")
    print(f"  Label smooth: {cfg.train.label_smoothing}")
    print(f"  Data dir   : {cfg.paths.data_dir}")
    print(f"  Experiments: {cfg.paths.experiments_dir}")
    print("=" * 50)
    cfg.save()
