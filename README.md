# Brain Tumor MRI Classification
### Evaluating Transfer Learning and Regularization Strategies

A production-level deep learning project for classifying brain tumors
from MRI images into four classes: **Glioma**, **Meningioma**,
**Pituitary tumor**, and **No tumor**.

---

## Project structure

```
brain_tumor_classification/
│
├── data/
│   ├── Training/          # 5,600 images (1,400 per class)
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/           # 1,600 images (400 per class)
│       └── ...
│
├── src/
│   ├── utils/
│   │   └── logger.py      # Console + file logging
│   ├── dataset.py         # Dataset, DataLoader, train/val split
│   ├── transforms.py      # Preprocessing + augmentation
│   ├── models.py          # Backbone loader + classifier head
│   ├── train.py           # Training loop (phased fine-tuning)
│   ├── evaluate.py        # Metrics, confusion matrix, ROC curves
│   └── gradcam.py         # Grad-CAM saliency maps
│
├── experiments/           # One subfolder per run (auto-created)
│   └── baseline_efficientnet_b3/
│       ├── config.yaml    # Exact settings used for this run
│       ├── best_model.pth # Best checkpoint (by val F1)
│       └── metrics.csv    # Per-epoch train/val metrics
│
├── outputs/
│   └── gradcam/           # Saved Grad-CAM visualizations
│
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   └── 02_results.ipynb   # Results comparison across models
│
├── config.py              # Central config (ALL settings here)
├── setup_colab.py         # One-click Colab setup script
└── requirements.txt       # Pinned dependencies
```

---

## Quickstart — Google Colab

### 1. Get your Kaggle API key

1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
2. This downloads `kaggle.json`
3. Upload `kaggle.json` to the **root of your Google Drive** (not inside any folder)

### 2. Upload project files to Google Drive

Create a folder called `brain_tumor_classification` in your Google Drive
and upload all project files into it.

### 3. Run the setup cell

In a new Colab notebook, paste and run:

```python
# Mount Drive and run setup
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/brain_tumor_classification')

exec(open('/content/drive/MyDrive/brain_tumor_classification/setup_colab.py').read())
```

This will:
- Mount your Google Drive
- Create the full project folder structure
- Install all required packages
- Download and extract the dataset from Kaggle (~400 MB)
- Verify your GPU and library versions

### 4. Verify the config

```python
from config import get_default_config
cfg = get_default_config()
# Prints all settings and saves config.yaml
```

---

## Running experiments

### Change model or hyperparameters

Edit `config.py` — specifically the `ExperimentConfig` dataclass.
To run a new experiment with a different backbone:

```python
from config import ExperimentConfig, ModelConfig

cfg = ExperimentConfig(name="baseline_resnet50")
cfg.model.backbone = "resnet50"
cfg.regularization.dropout_rate = 0.5
cfg.train.weight_decay = 1e-3
cfg.save()  # Saves config.yaml in experiments/baseline_resnet50/
```

### Ablation study — regularization

The comparison table from the paper is reproduced by running:

| Experiment name                | Backbone       | Dropout | Weight decay | Label smooth |
|-------------------------------|----------------|---------|--------------|--------------|
| `baseline_efficientnet_b3`    | EfficientNetB3 | 0.3     | 1e-4         | 0.1          |
| `ablation_no_dropout`         | EfficientNetB3 | 0.0     | 1e-4         | 0.1          |
| `ablation_heavy_dropout`      | EfficientNetB3 | 0.5     | 1e-4         | 0.1          |
| `ablation_no_label_smooth`    | EfficientNetB3 | 0.3     | 1e-4         | 0.0          |
| `ablation_high_wd`            | EfficientNetB3 | 0.3     | 1e-3         | 0.1          |
| `backbone_resnet50`           | ResNet50       | 0.3     | 1e-4         | 0.1          |
| `backbone_densenet121`        | DenseNet121    | 0.3     | 1e-4         | 0.1          |
| `backbone_vgg16`              | VGG16          | 0.3     | 1e-4         | 0.1          |

---

## Dataset

**Source:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split    | Glioma | Meningioma | No tumor | Pituitary | Total |
|----------|--------|------------|----------|-----------|-------|
| Training | 1,321  | 1,339      | 1,595    | 1,457     | 5,712 |
| Testing  | 300    | 306        | 405      | 300       | 1,311 |

> Note: Actual counts may differ slightly from the advertised 1,400/400
> as duplicate removal was applied in dataset version 2.

---

## Key design decisions

- **Phased fine-tuning** — backbone frozen → top layers unfrozen → full model
- **Stratified validation split** — 80/20 from Training set (never from Test)
- **Mixed precision training** — faster on Colab T4 GPU with `torch.cuda.amp`
- **Early stopping** — monitors val F1, patience = 7 epochs
- **Reproducible** — every run saves its exact config alongside its weights

---

## Requirements

See `requirements.txt`. Key dependencies:

| Library        | Purpose                          |
|----------------|----------------------------------|
| PyTorch 2.x    | Core deep learning framework     |
| timm           | Pretrained backbone zoo          |
| albumentations | Fast image augmentation          |
| grad-cam       | Saliency map visualization       |
| scikit-learn   | Metrics, stratified split        |
| tensorboard    | Training curve visualization     |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{brain_tumor_classification_2025,
  title   = {Evaluating Transfer Learning and Regularization Strategies
             for Brain Tumor Classification from MRI Images},
  year    = {2025},
  note    = {Dataset: Masoud Nickparvar, Kaggle Brain Tumor MRI Dataset}
}
```
