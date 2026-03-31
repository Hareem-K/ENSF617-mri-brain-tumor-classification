# ============================================================
#  src/dataset.py — Dataset and DataLoader
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS FILE DOES:
#  Handles everything related to loading images from disk
#  and preparing them in batches for the model.
#
#  Three things this file creates:
#
#  1. BrainTumorDataset
#     A custom PyTorch Dataset class. Knows how to read
#     one MRI image from disk, apply transforms, and
#     return it with its label.
#
#  2. create_dataloaders()
#     Splits Training/ into train (80%) and val (20%),
#     then creates three DataLoaders:
#       - train_loader   → shuffled, augmented
#       - val_loader     → ordered, no augmentation
#       - test_loader    → ordered, no augmentation
#
#  3. get_dataset_info()
#     Prints a summary of image counts and class mapping.
#
#  WHERE THIS FILE IS USED:
#  main.ipynb and 02_data_pipeline.ipynb import
#  create_dataloaders() to get the three loaders.
# ============================================================

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# Import our transforms from the same src/ folder
from src.transforms import get_transforms


# ──────────────────────────────────────────────
#  Class label mapping
#
#  The model outputs numbers (0, 1, 2, 3) not words.
#  This dictionary maps folder names to numbers
#  and back again.
#
#  WHY SORTED?
#  Sorting alphabetically ensures the mapping is
#  always identical regardless of OS or filesystem.
#  This means glioma=0, meningioma=1, notumor=2,
#  pituitary=3 — always.
# ──────────────────────────────────────────────
CLASS_NAMES = sorted(["glioma", "meningioma", "notumor", "pituitary"])
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


# ──────────────────────────────────────────────
#  BrainTumorDataset
#
#  This is a PyTorch Dataset. Think of it as a
#  smart list of (image_path, label) pairs.
#
#  When you ask for item number 42, it:
#    1. Finds the path for image 42
#    2. Opens the image from disk
#    3. Converts it to RGB (handles grayscale MRIs)
#    4. Applies the transform pipeline
#    5. Returns the tensor and its label number
# ──────────────────────────────────────────────
class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI images.

    Args:
        root_dir   : Path to the split folder
                     e.g. /content/drive/.../data/Training
        transform  : Albumentations transform pipeline
        class_names: List of class folder names

    Usage:
        from src.dataset import BrainTumorDataset
        from src.transforms import get_transforms

        dataset = BrainTumorDataset(
            root_dir  = "/content/drive/MyDrive/brain_tumor_classification/data/Training",
            transform = get_transforms(224, "train"),
        )
        image, label = dataset[0]
        print(image.shape)  # torch.Size([3, 224, 224])
        print(label)        # e.g. 0 (glioma)
    """

    def __init__(
        self,
        root_dir:    str,
        transform=   None,
        class_names: List[str] = CLASS_NAMES,
    ):
        self.root_dir    = Path(root_dir)
        self.transform   = transform
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        # ── Build the master list of (image_path, label) pairs
        # Walk through each class folder and collect all image paths
        self.samples: List[Tuple[Path, int]] = []

        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {cls_dir}\n"
                    f"Expected folder structure:\n"
                    f"  {self.root_dir}/\n"
                    f"    glioma/\n"
                    f"    meningioma/\n"
                    f"    notumor/\n"
                    f"    pituitary/"
                )
            label = self.class_to_idx[cls]
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_file, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self) -> int:
        """Returns total number of images in this dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads and returns one (image_tensor, label) pair.

        Args:
            idx : Index of the sample to load

        Returns:
            image : Tensor of shape [3, H, W]
            label : Integer class index (0–3)
        """
        img_path, label = self.samples[idx]

        # ── Open the image
        # PIL handles all common formats (JPEG, PNG, etc.)
        try:
            image = Image.open(img_path)
        except Exception as e:
            raise RuntimeError(f"Could not open image: {img_path}\nError: {e}")

        # ── Convert to RGB
        # Some MRI images are grayscale (1 channel).
        # Pretrained models expect 3 channels (RGB).
        # .convert("RGB") duplicates the channel 3 times.
        image = image.convert("RGB")

        # ── Convert to NumPy array
        # Albumentations expects numpy arrays, not PIL images.
        image = np.array(image, dtype=np.uint8)

        # ── Apply transforms (resize, augment, normalize, tensorize)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image     = augmented["image"]  # now a torch.Tensor [3, H, W]

        return image, label

    def get_labels(self) -> List[int]:
        """
        Returns all labels as a list.
        Used by StratifiedShuffleSplit to ensure
        each class is proportionally represented
        in both train and val splits.
        """
        return [label for _, label in self.samples]

    def class_counts(self) -> Dict[str, int]:
        """Returns a dict of {class_name: image_count}."""
        counts = {cls: 0 for cls in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


# ──────────────────────────────────────────────
#  create_dataloaders()
#
#  This is the main function you call from notebooks.
#  It does three things:
#
#  1. Creates the full Training dataset
#  2. Splits it into train (80%) and val (20%)
#     using StratifiedShuffleSplit — this ensures
#     each class keeps its proportion in both splits
#  3. Creates three DataLoaders ready for training
# ──────────────────────────────────────────────
def create_dataloaders(
    data_dir:   str,
    image_size: int   = 224,
    batch_size: int   = 32,
    val_split:  float = 0.2,
    num_workers: int  = 2,
    seed:       int   = 42,
    pin_memory: bool  = torch.cuda.is_available(),
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Creates train, validation, and test DataLoaders.

    Args:
        data_dir    : Path to the data/ folder
                      e.g. /content/drive/.../data
        image_size  : Target image size (default 224)
        batch_size  : Number of images per batch (default 32)
        val_split   : Fraction of training data for validation
                      (default 0.2 = 20%)
        num_workers : CPU workers for loading (2 works on Colab)
        seed        : Random seed for reproducible splits
        pin_memory  : Speeds up GPU data transfer

    Returns:
        train_loader : DataLoader for training
        val_loader   : DataLoader for validation
        test_loader  : DataLoader for testing
        info         : Dictionary with split sizes and class info

    Usage:
        from src.dataset import create_dataloaders

        train_loader, val_loader, test_loader, info = create_dataloaders(
            data_dir   = "/content/drive/MyDrive/brain_tumor_classification/data",
            image_size = 224,
            batch_size = 32,
        )
        print(info)
    """

    train_dir = os.path.join(data_dir, "Training")
    test_dir  = os.path.join(data_dir, "Testing")

    # ── Step 1: Get transforms for each phase
    train_transform    = get_transforms(image_size=image_size, phase="train")
    val_test_transform = get_transforms(image_size=image_size, phase="val")

    # ── Step 2: Create the full training dataset
    # We load it TWICE with different transforms:
    # - full_train_aug: used for actual training (with augmentation)
    # - full_train_clean: used to extract val samples (no augmentation)
    full_train_aug   = BrainTumorDataset(train_dir, transform=train_transform)
    full_train_clean = BrainTumorDataset(train_dir, transform=val_test_transform)

    # ── Step 3: Stratified train / val split
    # StratifiedShuffleSplit guarantees that each class
    # appears in the same proportion in both splits.
    # Example: if glioma is 25% of training, it will be
    # ~25% of both train and val splits.
    all_labels = full_train_aug.get_labels()

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=seed,
    )

    # Get the indices for train and val
    train_indices, val_indices = next(
        splitter.split(X=np.zeros(len(all_labels)), y=all_labels)
    )

    # ── Step 4: Create Subset datasets
    # Subset wraps the full dataset and only exposes
    # the images at the specified indices.
    train_dataset = Subset(full_train_aug,   train_indices)
    val_dataset   = Subset(full_train_clean, val_indices)

    # ── Step 5: Create the test dataset
    test_dataset = BrainTumorDataset(test_dir, transform=val_test_transform)

    # ── Step 6: Wrap in DataLoaders
    # DataLoader handles batching, shuffling, and
    # parallel loading via num_workers.

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,        # Shuffle each epoch for better training
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,        # Drop incomplete final batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,       # No shuffle — consistent evaluation
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    # ── Step 7: Collect useful info about the splits
    info = {
        "class_names":   CLASS_NAMES,
        "class_to_idx":  CLASS_TO_IDX,
        "idx_to_class":  IDX_TO_CLASS,
        "num_classes":   len(CLASS_NAMES),
        "train_size":    len(train_dataset),
        "val_size":      len(val_dataset),
        "test_size":     len(test_dataset),
        "total_train":   len(full_train_aug),
        "batch_size":    batch_size,
        "image_size":    image_size,
        "train_batches": len(train_loader),
        "val_batches":   len(val_loader),
        "test_batches":  len(test_loader),
    }

    return train_loader, val_loader, test_loader, info


# ──────────────────────────────────────────────
#  get_dataset_info()
#
#  Prints a clean summary of the dataset.
#  Call this after create_dataloaders() to
#  confirm everything looks correct.
# ──────────────────────────────────────────────
def get_dataset_info(info: Dict) -> None:
    """
    Prints a formatted summary of dataset splits.

    Args:
        info : The info dict returned by create_dataloaders()

    Usage:
        train_loader, val_loader, test_loader, info = create_dataloaders(...)
        get_dataset_info(info)
    """
    total = info["train_size"] + info["val_size"] + info["test_size"]
    print("=" * 50)
    print("  Dataset Summary")
    print("=" * 50)
    print(f"  Classes      : {', '.join(info['class_names'])}")
    print(f"  Image size   : {info['image_size']}x{info['image_size']}")
    print(f"  Batch size   : {info['batch_size']}")
    print()
    print(f"  Train set    : {info['train_size']:>5} images  ({info['train_batches']} batches)")
    print(f"  Val set      : {info['val_size']:>5} images  ({info['val_batches']} batches)")
    print(f"  Test set     : {info['test_size']:>5} images  ({info['test_batches']} batches)")
    print(f"  Total        : {total:>5} images")
    print()
    print("  Class → Index mapping:")
    for cls, idx in info["class_to_idx"].items():
        print(f"    {idx} → {cls}")
    print("=" * 50)


# ──────────────────────────────────────────────
#  Quick self-test
#  Run: python src/dataset.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Adjust this path if testing locally
    DATA_DIR = "/content/drive/MyDrive/brain_tumor_classification/data"

    print("Testing dataset.py...")
    print("-" * 40)

    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_dir   = DATA_DIR,
        image_size = 224,
        batch_size = 32,
    )

    get_dataset_info(info)

    # Load one batch and confirm shapes
    images, labels = next(iter(train_loader))
    print(f"\n  One batch from train_loader:")
    print(f"  images shape : {images.shape}")   # [32, 3, 224, 224]
    print(f"  labels shape : {labels.shape}")   # [32]
    print(f"  labels sample: {labels[:8].tolist()}")
    print()
    print("dataset.py working correctly.")
