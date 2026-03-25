# ENSF617-mri-brain-tumor-classification

# Brain Tumor Classification using Transfer Learning

## Overview

This project investigates the use of convolutional neural networks (CNNs) and transfer learning techniques to classify brain MRI images into tumor categories. The focus is on comparing models trained from scratch with pretrained models, and analyzing the effects of regularization techniques on performance and generalization.

---

## Dataset

We use the **Brain Tumor MRI Dataset** from Kaggle.

The dataset consists of MRI images categorized into four classes:

* Glioma
* Meningioma
* Pituitary tumor
* No tumor

The dataset is downloaded programmatically using the Kaggle API (not stored in this repository).

---

## Project Structure

```
brain-tumor-classification-transfer-learning/
│
├── README.md                  # Project overview and instructions
├── requirements.txt          # Python dependencies
├── .gitignore                # Files/folders ignored by Git
│
├── data/
│   ├── raw/                  # Raw dataset downloaded from Kaggle (ignored)
│   ├── processed/            # Preprocessed data (optional)
│
├── notebooks/
│   ├── 01_data_loading.ipynb         # Dataset download, loading, and visualization
│   ├── 02_cnn_baseline.ipynb         # Baseline CNN model (from scratch)
│   ├── 03_transfer_learning.ipynb    # Transfer learning experiments (ResNet/VGG)
│   ├── 04_experiments.ipynb          # Final experiments and comparisons
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # Data loading, transforms, and train/val/test splits
│   │
│   ├── models/
│   │   ├── cnn.py            # Custom CNN architecture
│   │   ├── resnet.py         # Pretrained model setup (transfer learning)
│   │
│   ├── training/
│   │   ├── train.py          # Training loop implementation
│   │   ├── evaluate.py       # Model evaluation logic
│   │
│   ├── utils/
│   │   ├── metrics.py        # Accuracy, precision, recall, F1-score
│   │   ├── plots.py          # Visualization (loss curves, confusion matrices)
│
├── results/
│   ├── plots/                # Training curves and performance graphs
│   ├── confusion_matrices/   # Confusion matrix outputs
│   ├── logs/                 # Experiment logs
│
├── models/
│   ├── saved_models/         # Trained model weights (.pth files)
│
└── report/
    ├── report.docx           # Final project report
```

---

## Setup Instructions

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Download Dataset (Kaggle API)

```
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/raw/
```

### 3. Run Notebooks

Start with:

```
notebooks/01_data_loading.ipynb
```

---

## Methodology

* Baseline CNN trained from scratch
* Transfer learning using pretrained models (ResNet, VGG)
* Fine-tuning strategies (frozen, partial, full)
* Regularization techniques:

  * Data augmentation
  * Dropout
  * L2 regularization

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Results

(To be updated when experiments are completed)

---

## Authors

* Hareem Khan
* Rakin Sad Aftab
* Angus Cheang
* Md. Labib Hasan
