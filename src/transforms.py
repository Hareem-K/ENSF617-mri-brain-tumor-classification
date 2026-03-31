# ============================================================
#  src/transforms.py — Image Preprocessing & Augmentation
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS FILE DOES:
#  Defines how raw MRI images are prepared before being
#  fed into a neural network. Two sets of transforms:
#
#  1. train_transforms  — for training images
#     Preprocessing + augmentation (random flips, rotation etc.)
#     Goal: make the model see slightly different versions
#     of each image every epoch so it generalises better.
#
#  2. val_test_transforms — for validation and test images
#     Preprocessing only — NO augmentation.
#     Goal: evaluate on clean, consistent images every time
#     so results are reproducible and fair.
#
#  WHERE THIS FILE IS USED:
#  src/dataset.py imports get_transforms() from here and
#  passes the correct transform to each split.
# ============================================================

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Tuple


# ──────────────────────────────────────────────
#  ImageNet normalization constants
#
#  WHY IMAGENET STATS?
#  Our pretrained models (ResNet, EfficientNet etc.)
#  were originally trained on ImageNet. They learned
#  to expect pixel values in a specific range.
#  If we feed them differently scaled pixels, the
#  pretrained weights become less useful.
#
#  These values subtract the mean and divide by std
#  so every image has roughly similar pixel ranges.
# ──────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(
    image_size: int = 224,
    phase: str = "train",
) -> A.Compose:
    """
    Returns the correct set of image transforms for a
    given phase (train / val / test).

    Args:
        image_size : Target height and width in pixels.
                     Default 224 works for most backbones.
                     Use 300 for EfficientNetB3.
        phase      : One of "train", "val", or "test".
                     "val" and "test" get identical transforms.

    Returns:
        An albumentations Compose pipeline.

    Usage example:
        from src.transforms import get_transforms
        transform = get_transforms(image_size=224, phase="train")
        augmented = transform(image=numpy_image)
        tensor    = augmented["image"]   # shape: [3, 224, 224]
    """

    if phase == "train":
        return _train_transforms(image_size)
    else:
        # val and test both use the same clean transforms
        return _val_test_transforms(image_size)


# ──────────────────────────────────────────────
#  Training transforms
#  Applied to TRAINING images only.
#
#  Each augmentation is explained below.
#  All augmentations are RANDOM — each image looks
#  slightly different every time it is loaded.
# ──────────────────────────────────────────────
def _train_transforms(image_size: int) -> A.Compose:
    return A.Compose([

        # ── Step 1: Resize
        # All images are resized to the same square size.
        # Our EDA showed images vary widely in size —
        # the model needs a fixed input size.
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
        ),

        # ── Step 2: Random horizontal flip
        # 50% chance the image is mirrored left-to-right.
        # Brain tumors can appear on either side of the
        # brain, so flipping creates realistic variation.
        A.HorizontalFlip(p=0.5),

        # ── Step 3: Random vertical flip
        # 20% chance the image is flipped top-to-bottom.
        # Less common than horizontal flip since MRI scans
        # usually have a consistent orientation.
        A.VerticalFlip(p=0.2),

        # ── Step 4: Random rotation
        # Rotates the image by up to ±15 degrees.
        # MRI scans are sometimes slightly tilted —
        # this teaches the model to handle that.
        A.Rotate(
            limit=15,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),

        # ── Step 5: Random zoom (scale)
        # Randomly zooms in or out slightly.
        # This simulates different distances/magnifications
        # in MRI acquisition.
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),   # zoom between 85% and 100%
            ratio=(0.9, 1.1),    # slight aspect ratio variation
            p=0.4,
        ),

        # ── Step 6: Brightness and contrast adjustment
        # Randomly changes how bright or contrasty the image is.
        # MRI scans from different machines or settings can
        # look brighter or darker — this teaches robustness.
        A.RandomBrightnessContrast(
            brightness_limit=0.2,   # ±20% brightness change
            contrast_limit=0.2,     # ±20% contrast change
            p=0.5,
        ),

        # ── Step 7: Gaussian noise
        # Adds a tiny amount of random pixel noise.
        # Real MRI images contain scanner noise —
        # this makes the model more tolerant of it.
        A.GaussNoise(
            p=0.3,
        ),

        # ── Step 8: Gaussian blur (mild)
        # Very lightly blurs the image occasionally.
        # Simulates slight focus variation between scans.
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2,
        ),

        # ── Step 9: Shift, scale, rotate (fine)
        # Small random shifts and slight scaling.
        # Teaches the model that the tumor can be
        # positioned anywhere in the frame.
        A.Affine(
            translate_percent=(-0.05, 0.05), # Replaces shift_limit (±5%)
            scale=(0.95, 1.05),              # Replaces scale_limit (1.0 ± 0.05)
            rotate=(-15, 15),                # Replaces rotate_limit (±15 degrees)
            p=0.4,
            border_mode=cv2.BORDER_CONSTANT,        
            fill=0,
        ),

        # ── Step 10: Normalize
        # Subtracts ImageNet mean, divides by ImageNet std.
        # This is REQUIRED for pretrained models — see note above.
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        # ── Step 11: Convert to PyTorch tensor
        # Changes shape from [H, W, 3] (numpy) to
        # [3, H, W] (PyTorch) and converts to float32.
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────
#  Validation / Test transforms
#  Applied to VALIDATION and TEST images.
#
#  NO augmentation here — only resize + normalize.
#  We want evaluation to be consistent and fair
#  every time, not random.
# ──────────────────────────────────────────────
def _val_test_transforms(image_size: int) -> A.Compose:
    return A.Compose([

        # ── Step 1: Resize (same as training)
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
        ),

        # ── Step 2: Normalize (same constants as training)
        # IMPORTANT: must use the same mean/std as training.
        # Using different values would break the model.
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        # ── Step 3: Convert to tensor
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────
#  Denormalization helper
#
#  WHAT THIS IS FOR:
#  After normalization, pixel values are no longer
#  in the 0-255 range — they are roughly between -2 and +2.
#  If you try to display a normalized image it will
#  look completely wrong (mostly black or grey).
#
#  This function reverses the normalization so you
#  can display the image correctly in matplotlib.
#  Used in visualization notebooks only.
# ──────────────────────────────────────────────
def denormalize(
    tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std:  Tuple[float, ...] = IMAGENET_STD,
) -> np.ndarray:
    """
    Reverses ImageNet normalization on a tensor for display.

    Args:
        tensor : PyTorch tensor of shape [3, H, W] or [B, 3, H, W]
        mean   : Mean used during normalization
        std    : Std used during normalization

    Returns:
        NumPy array of shape [H, W, 3] with values in [0, 1]

    Usage:
        img_display = denormalize(tensor)
        plt.imshow(img_display)
    """
    import torch

    # Handle both single image [3,H,W] and batch [B,3,H,W]
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Clone so we don't modify the original tensor
    img = tensor.clone().cpu().float()

    # Reverse: multiply by std, add mean
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t

    # Clamp to valid range and convert to [H, W, 3] numpy
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()

    return img


# ──────────────────────────────────────────────
#  Quick test — run this file directly to verify
#  the transforms work correctly.
#
#  Usage:
#    python src/transforms.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("Testing transforms.py...")
    print("-" * 40)

    # Create a fake MRI image: 256x256 grayscale converted to RGB
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    for phase in ["train", "val", "test"]:
        transform = get_transforms(image_size=224, phase=phase)
        result    = transform(image=dummy_image)
        tensor    = result["image"]

        print(f"  Phase : {phase}")
        print(f"  Input shape  : {dummy_image.shape}")
        print(f"  Output shape : {tensor.shape}")
        print(f"  Output dtype : {tensor.dtype}")
        print(f"  Value range  : [{tensor.min():.3f}, {tensor.max():.3f}]")
        print()

    # Test denormalize
    dummy_tensor = torch.randn(3, 224, 224)
    display_img  = denormalize(dummy_tensor)
    print(f"  Denormalize output shape : {display_img.shape}")
    print(f"  Denormalize value range  : [{display_img.min():.3f}, {display_img.max():.3f}]")
    print()
    print("All tests passed. transforms.py is working correctly.")
