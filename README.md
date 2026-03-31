# Brain Tumor MRI Classification

**Evaluating Transfer Learning and Regularization Strategies for Brain Tumor Classification from MRI Images**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project File System](#2-project-file-system)
3. [Dataset](#3-dataset)
4. [Platform Support](#4-platform-support)
5. [Phase 1 — Environment Setup](#5-phase-1--environment-setup)
   - [Option A: Google Colab](#option-a-google-colab-recommended)
   - [Option B: Windows](#option-b-windows)
   - [Option C: macOS](#option-c-macos)
   - [Option D: Linux](#option-d-linux)
6. [Phase 2 — Data Pipeline](#6-phase-2--data-pipeline)
7. [Verify Your Setup](#7-verify-your-setup)
8. [Continuing to Phase 3 and Beyond](#8-continuing-to-phase-3-and-beyond)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Project Overview

This project builds a deep learning pipeline to classify brain tumor MRI images into four categories:

| Class | Description |
|---|---|
| `glioma` | Malignant brain tumor originating in glial cells |
| `meningioma` | Tumor arising from the meninges (brain lining) |
| `pituitary` | Tumor in the pituitary gland |
| `notumor` | Healthy brain scan with no tumor present |

**Research goals:**
- Compare multiple pretrained CNN backbones (VGG16, ResNet50, EfficientNetB3, DenseNet121)
- Evaluate regularization strategies (dropout, weight decay, label smoothing)
- Propose a novel attention-enhanced model (CBAM) to address meningioma classification difficulty
- Produce Grad-CAM saliency maps for interpretability

**Tech stack:** Python 3.10+, PyTorch 2.x, timm, albumentations, scikit-learn, grad-cam

---

## 2. Project File System

Understanding where every file lives is essential before you begin.

```
brain_tumor_classification/               ← Project root on Google Drive
│
├── data/                                 ← Raw dataset (auto-downloaded from Kaggle)
│   ├── Training/
│   │   ├── glioma/          (1,400 images)
│   │   ├── meningioma/      (1,400 images)
│   │   ├── notumor/         (1,400 images)
│   │   └── pituitary/       (1,400 images)
│   └── Testing/
│       ├── glioma/          (400 images)
│       ├── meningioma/      (400 images)
│       ├── notumor/         (400 images)
│       └── pituitary/       (400 images)
│
├── src/                                  ← All reusable Python source files
│   ├── transforms.py                     ← Image preprocessing + augmentation
│   ├── dataset.py                        ← Dataset class, DataLoaders, splits
│   ├── models.py                         ← (Phase 3) Backbone loader + classifier head
│   ├── train.py                          ← (Phase 3) Training loop
│   ├── evaluate.py                       ← (Phase 4) Metrics + confusion matrix
│   ├── gradcam.py                        ← (Phase 4) Grad-CAM saliency maps
│   ├── cbam.py                           ← (Phase 5) Proposed attention module
│   └── utils/
│       └── logger.py                     ← Logging utility
│
├── notebooks/                            ← Jupyter/Colab notebooks (one per phase)
│   ├── main.ipynb                        ← MASTER notebook — runs everything
│   ├── 00_setup.ipynb                    ← Phase 1: Environment setup
│   ├── 01_eda.ipynb                      ← Phase 1: Exploratory data analysis
│   ├── 02_data_pipeline.ipynb            ← Phase 2: Data pipeline verification
│   ├── 03_training.ipynb                 ← (Phase 3) Training experiments
│   └── 04_results.ipynb                  ← (Phase 4) Results + Grad-CAM
│
├── experiments/                          ← Auto-created during training
│   └── baseline_efficientnet_b3/         ← One folder per experiment run
│       ├── config.yaml                   ← Exact settings used for this run
│       ├── best_model.pth                ← Best model checkpoint
│       └── metrics.csv                   ← Per-epoch train/val metrics
│
├── outputs/                              ← Auto-created charts and visualizations
│   ├── eda_class_distribution.png
│   ├── eda_sample_images.png
│   ├── pipeline_augmentation.png
│   └── gradcam/                          ← Grad-CAM saliency maps
│
├── logs/                                 ← Auto-created training logs
│
├── config.py                             ← Central configuration (all settings)
├── setup_colab.py                        ← One-click Colab setup script
├── requirements.txt                      ← All Python dependencies
└── README.md                             ← This file
```

### File system rules — read before touching anything

| Rule | Reason |
|---|---|
| Never put notebooks inside `src/` | `src/` is for importable code only |
| Never put source code inside `notebooks/` | Notebooks call `src/` files — they don't contain logic |
| Never modify files inside `data/` | Raw data must stay untouched |
| All settings go in `config.py` | Never hardcode paths or values in training scripts |
| All experiment outputs go in `experiments/` | Keeps results reproducible and traceable |

---

## 3. Dataset

**Source:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
**Author:** Masoud Nickparvar
**License:** Attribution 4.0 International (CC BY 4.0)

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---|---|---|---|---|
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |
| **Total** | **1,800** | **1,800** | **1,800** | **1,800** | **7,200** |

The dataset is fully balanced. Images vary in size — all preprocessing steps resize to 224×224.

---

## 4. Platform Support

This project runs on all major platforms. Choose the setup guide for your system.

| Platform | GPU Support | Recommended For |
|---|---|---|
| **Google Colab** | Yes (free T4 GPU) | Anyone — easiest setup |
| **Windows** | Yes (NVIDIA only) | Local development |
| **macOS** | MPS (Apple Silicon) / CPU | Local development |
| **Linux** | Yes (NVIDIA) | Servers / HPC clusters |

---

## 5. Phase 1 — Environment Setup

---

### Option A: Google Colab (Recommended)

Google Colab gives you a free GPU and a pre-configured Python environment. This is the recommended option for most users.

#### Prerequisites

1. A Google account with Google Drive
2. A Kaggle account — [create one free here](https://www.kaggle.com)

#### Step 1 — Get your Kaggle API key

1. Log in to [kaggle.com](https://www.kaggle.com)
2. Click your profile photo → **Settings** → **API** → **Create New Token**
3. Make sure you copy the API Token
4. This downloads a file called `kaggle.json` to your computer

If the `kaggle.json` doesn't download after "Creating new Token", please follow the below steps:

1. Open a new blank document and save it as `kaggle.json`
2. Open the saved `kaggle.json` file and place the following information and save it.

```json
{
  "username": "", (place your kaggle username)
  "key": "", (place your generated token from kaggle)
}
```

#### Step 2 — Upload files to Google Drive

1. Open [Google Drive](https://drive.google.com)
2. Create a folder called exactly: `brain_tumor_classification`
3. Upload all project files into this folder:
   - `config.py`
   - `setup_colab.py`
   - `requirements.txt`
   - `README.md`
   - The entire `src/` folder (with `transforms.py`, `dataset.py`, etc.)
   - The entire `notebooks/` folder
4. Upload `kaggle.json` to the **root of your Drive** (not inside any folder)

Your Drive structure should look like this:
```
My Drive/
├── kaggle.json                           ← API key (root level)
└── brain_tumor_classification/
    ├── config.py
    ├── setup_colab.py
    ├── requirements.txt
    └── src/
        ├── transforms.py
        └── dataset.py
```

#### Step 3 — Open Google Colab and enable GPU

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New notebook**
3. Go to **Runtime → Change runtime type → T4 GPU → Save**

#### Step 4 — Run the setup cell

Paste this into the first cell and press the play button:

```python
from google.colab import drive
drive.mount('/content/drive')

exec(open('/content/drive/MyDrive/brain_tumor_classification/setup_colab.py').read())
```

Click **Allow** when Google asks for Drive permission. The script will:
- Create the full folder structure
- Install all dependencies from `requirements.txt`
- Download and extract the dataset from Kaggle (~400 MB)
- Verify all image counts

Wait for **"Setup Complete"** to appear at the bottom.

#### Step 5 — Restart the runtime

Go to **Runtime → Restart session**

This is required. Libraries only load correctly after a restart.

#### Step 6 — Verify the installation

After the restart, run this cell:

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/brain_tumor_classification')

import torch, timm, albumentations, cv2, sklearn, matplotlib, numpy

print("=" * 45)
print(f"  PyTorch        : {torch.__version__}")
print(f"  CUDA           : {torch.cuda.is_available()}")
print(f"  GPU            : {torch.cuda.get_device_name(0)}")
print(f"  NumPy          : {numpy.__version__}")
print(f"  OpenCV         : {cv2.__version__}")
print(f"  timm           : {timm.__version__}")
print(f"  albumentations : {albumentations.__version__}")
print(f"  scikit-learn   : {sklearn.__version__}")
print("=" * 45)
print("Phase 1 complete!")
```

All libraries should print version numbers. Phase 1 is done.

---

### Option B: Windows

#### Prerequisites

- Windows 10 or 11 (64-bit)
- Python 3.10 or 3.11 — [download here](https://www.python.org/downloads/)
- Git — [download here](https://git-scm.com/download/win)
- NVIDIA GPU recommended (for CUDA support)

#### Step 1 — Check Python is installed

Open **Command Prompt** and run:

```cmd
python --version
```

You should see `Python 3.10.x` or `Python 3.11.x`. If not, install Python from the link above and make sure you tick **"Add Python to PATH"** during installation.

#### Step 2 — Create the project folder

```cmd
mkdir C:\projects\brain_tumor_classification
cd C:\projects\brain_tumor_classification
```

Copy all project files into this folder.

#### Step 3 — Create a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command prompt. This means the virtual environment is active.

#### Step 4 — Install PyTorch with CUDA

If you have an NVIDIA GPU (check by running `nvidia-smi` in Command Prompt):

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you have no GPU (CPU only):

```cmd
pip install torch torchvision torchaudio
```

#### Step 5 — Install remaining dependencies

```cmd
pip install -r requirements.txt
```

#### Step 6 — Set up Kaggle and download dataset

```cmd
pip install kaggle
```

Place your `kaggle.json` file in:
```
C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
```

Then download the dataset:

```cmd
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

#### Step 7 — Verify installation

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import timm; print('timm:', timm.__version__)"
python -c "import albumentations; print('albumentations:', albumentations.__version__)"
```

#### Every time you return to this project

```cmd
cd C:\projects\brain_tumor_classification
venv\Scripts\activate
```

---

### Option C: macOS

#### Prerequisites

- macOS 11 (Big Sur) or later
- Python 3.10 or 3.11 — install via [Homebrew](https://brew.sh)

#### Step 1 — Install Homebrew (if not installed)

Open **Terminal** and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2 — Install Python

```bash
brew install python@3.11
```

Verify:

```bash
python3 --version
```

#### Step 3 — Create the project folder

```bash
mkdir -p ~/projects/brain_tumor_classification
cd ~/projects/brain_tumor_classification
```

Copy all project files into this folder.

#### Step 4 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

#### Step 5 — Install PyTorch

For **Apple Silicon (M1/M2/M3)** — uses MPS acceleration:

```bash
pip install torch torchvision torchaudio
```

For **Intel Mac** — CPU only:

```bash
pip install torch torchvision torchaudio
```

#### Step 6 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

#### Step 7 — Set up Kaggle and download dataset

```bash
pip install kaggle
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

#### Step 8 — Verify installation

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import timm; print('timm:', timm.__version__)"
python3 -c "import albumentations; print('albumentations:', albumentations.__version__)"
```

> **Note for Apple Silicon users:** CUDA is not available on Mac.
> PyTorch will use MPS (Metal Performance Shaders) for GPU acceleration.
> Training will be slower than a CUDA GPU. Google Colab is recommended
> for full training runs.

#### Every time you return to this project

```bash
cd ~/projects/brain_tumor_classification
source venv/bin/activate
```

---

### Option D: Linux (Ubuntu/Debian)

#### Prerequisites

- Ubuntu 20.04 / 22.04 / 24.04 or Debian equivalent
- Python 3.10 or 3.11
- NVIDIA GPU + CUDA drivers recommended

#### Step 1 — Install Python and dependencies

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
python3 --version
```

#### Step 2 — Create the project folder

```bash
mkdir -p ~/projects/brain_tumor_classification
cd ~/projects/brain_tumor_classification
```

Copy all project files into this folder.

#### Step 3 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 4 — Check CUDA availability

```bash
nvidia-smi
```

If this shows your GPU model and driver version, CUDA is available.

#### Step 5 — Install PyTorch

With NVIDIA GPU (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 6 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

#### Step 7 — Set up Kaggle and download dataset

```bash
pip install kaggle
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

#### Step 8 — Verify installation

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python3 -c "import timm; print('timm:', timm.__version__)"
python3 -c "import albumentations; print('albumentations:', albumentations.__version__)"
```

#### Every time you return to this project

```bash
cd ~/projects/brain_tumor_classification
source venv/bin/activate
```

---

## 6. Phase 2 — Data Pipeline

Once Phase 1 is complete on any platform, the data pipeline steps are identical everywhere.

### What Phase 2 does

Transforms raw MRI images into batches of tensors that a neural network can process:

```
Raw image (various sizes) → Resize 224×224 → Normalize → Augment → Tensor [3, 224, 224]
```

### Files involved

| File | Location | Purpose |
|---|---|---|
| `transforms.py` | `src/` | Defines preprocessing and augmentation pipelines |
| `dataset.py` | `src/` | Loads images, creates 80/20 stratified train/val split |
| `02_data_pipeline.ipynb` | `notebooks/` | Visualizes and verifies the pipeline |

### Step 1 — Verify transforms.py

Run this in your notebook or terminal:

```python
import sys
sys.path.insert(0, '/path/to/brain_tumor_classification')  # adjust for your platform

import numpy as np
from src.transforms import get_transforms, denormalize

dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

for phase in ['train', 'val']:
    transform = get_transforms(image_size=224, phase=phase)
    result    = transform(image=dummy)
    tensor    = result['image']
    print(f"Phase: {phase}")
    print(f"  Output shape : {tensor.shape}")    # expect torch.Size([3, 224, 224])
    print(f"  Value range  : [{tensor.min():.2f}, {tensor.max():.2f}]")
```

Expected output:
```
Phase: train
  Output shape : torch.Size([3, 224, 224])
  Value range  : [-2.12, 2.64]
Phase: val
  Output shape : torch.Size([3, 224, 224])
  Value range  : [-2.05, 2.57]
```

> The negative values are correct — they are the result of ImageNet normalization.

### Step 2 — Verify dataset.py

```python
from src.dataset import create_dataloaders, get_dataset_info

DATA_DIR = '/path/to/brain_tumor_classification/data'  # adjust for your platform

train_loader, val_loader, test_loader, info = create_dataloaders(
    data_dir   = DATA_DIR,
    image_size = 224,
    batch_size = 32,
)

get_dataset_info(info)

images, labels = next(iter(train_loader))
print(f"Batch shape : {images.shape}")    # expect torch.Size([32, 3, 224, 224])
print(f"Labels      : {labels[:8].tolist()}")
```

Expected output:
```
==================================================
  Dataset Summary
==================================================
  Train set    :  4480 images  (140 batches)
  Val set      :  1120 images  (35 batches)
  Test set     :  1600 images  (50 batches)
  Total        :  7200 images

Batch shape : torch.Size([32, 3, 224, 224])
```

### Step 3 — Run 02_data_pipeline.ipynb

Open `notebooks/02_data_pipeline.ipynb` in Colab (or Jupyter locally) and run all cells.

You will see 4 visualizations saved to `outputs/`:

| Visualization | File saved | What it shows |
|---|---|---|
| Split distribution | `pipeline_split_distribution.png` | Image counts per class per split |
| Sample images | `pipeline_sample_images.png` | Raw MRI images from each class |
| Augmentation effect | `pipeline_augmentation.png` | Same image shown 8 different ways |
| Batch sample | `pipeline_batch_sample.png` | One real training batch |

When all 4 charts display correctly, **Phase 2 is complete.**

---

## 7. Verify Your Setup

Run this complete verification script on any platform to confirm everything is working:

```python
import sys
sys.path.insert(0, '/path/to/brain_tumor_classification')  # change to your path

print("=" * 50)
print("  Full setup verification")
print("=" * 50)

# 1. Libraries
import torch, timm, albumentations, cv2, sklearn, numpy, matplotlib
print(f"  PyTorch        : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  timm           : {timm.__version__}")
print(f"  albumentations : {albumentations.__version__}")
print(f"  OpenCV         : {cv2.__version__}")
print(f"  NumPy          : {numpy.__version__}")

# 2. Transforms
import numpy as np
from src.transforms import get_transforms
dummy     = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
transform = get_transforms(224, 'train')
tensor    = transform(image=dummy)['image']
assert tensor.shape == torch.Size([3, 224, 224]), "Transform shape wrong"
print(f"\n  transforms.py  : OK — shape {tensor.shape}")

# 3. DataLoaders
from src.dataset import create_dataloaders
DATA_DIR = '/path/to/brain_tumor_classification/data'  # change to your path
train_loader, val_loader, test_loader, info = create_dataloaders(DATA_DIR)
images, labels = next(iter(train_loader))
assert images.shape == torch.Size([32, 3, 224, 224]), "Batch shape wrong"
print(f"  dataset.py     : OK — batch shape {images.shape}")

# 4. Config
from config import get_default_config
cfg = get_default_config()
print(f"  config.py      : OK — backbone = {cfg.model.backbone}")

print("\n" + "=" * 50)
print("  All checks passed. Ready for Phase 3.")
print("=" * 50)
```

---

## 8. Continuing to Phase 3 and Beyond

### Path adjustments by platform

When reading any notebook or source file, replace the data path with your platform's path:

| Platform | Data path |
|---|---|
| Google Colab | `/content/drive/MyDrive/brain_tumor_classification/data` |
| Windows | `C:\projects\brain_tumor_classification\data` |
| macOS | `~/projects/brain_tumor_classification/data` |
| Linux | `~/projects/brain_tumor_classification/data` |

### How to use main.ipynb

`main.ipynb` is the master notebook. It calls each `src/` file in order. Every time you start a new session:

1. Open `main.ipynb`
2. Run the setup cells (mount Drive / activate venv)
3. Continue from wherever you left off

### Phases still to complete

| Phase | What you will build | Files |
|---|---|---|
| Phase 3 | Pretrained backbones + training loop + experiments | `src/models.py`, `src/train.py` |
| Phase 4 | Metrics, confusion matrix, Grad-CAM | `src/evaluate.py`, `src/gradcam.py` |
| Phase 5 | Novel CBAM attention model | `src/cbam.py` |
| Phase 6 | Academic paper | Paper document |

### The golden rule for all future phases

> **Source code goes in `src/`.
> Notebooks go in `notebooks/`.
> Outputs go in `outputs/`.
> Experiment results go in `experiments/`.
> Settings go in `config.py`.
> Never mix these.**

---

## 9. Troubleshooting

### "Module not found: src.transforms"

You need to add the project root to Python's path before importing:

```python
import sys
sys.path.insert(0, '/path/to/brain_tumor_classification')
```

### "CUDA not available" on Windows/Linux

1. Check your NVIDIA driver: run `nvidia-smi` in terminal
2. Make sure you installed the CUDA version of PyTorch (see Step 4 above)
3. Reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org)

### "AttributeError: _ARRAY_API not found" (Colab only)

This is a NumPy/OpenCV version conflict. Fix it by running:

```python
!pip install -q --force-reinstall "opencv-python-headless>=4.10.0.84"
```

Then restart the runtime.

### "No images found" after dataset download

Check that your folder structure matches exactly:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/        ← note: "notumor" not "no_tumor"
│   └── pituitary/
└── Testing/
    └── ...
```

### "pip install" conflicts (Colab only)

Colab has many pre-installed packages. If you see dependency conflict warnings, they are usually safe to ignore as long as all libraries in the verification cell import correctly.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{brain_tumor_classification_2025,
  title  = {Evaluating Transfer Learning and Regularization Strategies
            for Brain Tumor Classification from MRI Images},
  year   = {2025},
  note   = {Dataset: Masoud Nickparvar,
            Kaggle Brain Tumor MRI Dataset,
            CC BY 4.0}
}
```

---

*README covers Phase 1 (Environment + EDA) and Phase 2 (Data Pipeline).
Phases 3–6 will be documented as the project progresses.*
