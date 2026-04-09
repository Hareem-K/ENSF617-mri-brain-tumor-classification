# ============================================================
#  src/gradcam.py — Grad-CAM Saliency Maps
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS FILE DOES:
#  Generates Gradient-weighted Class Activation Maps (Grad-CAM)
#  for selected MRI images. These heatmaps show WHICH REGIONS
#  of the image the model focused on when making a prediction.
#
#  WHY THIS MATTERS FOR YOUR PAPER:
#  - Correct predictions with focused attention on tumor =
#    model is learning the right features
#  - Wrong predictions with attention on skull/background =
#    model is attending to irrelevant regions → motivates CBAM
#  - VGG16 vs EfficientNetB3 side-by-side comparison =
#    explains the performance gap visually
#
#  HOW GRAD-CAM WORKS (simplified):
#  1. Run forward pass → get prediction
#  2. Backpropagate gradient of predicted class score
#     back to the last convolutional layer
#  3. Average gradients across spatial dimensions
#     → importance weight per feature channel
#  4. Weighted sum of feature maps → raw heatmap
#  5. ReLU → resize → overlay on original image
#
#  OUTPUTS SAVED TO:
#  outputs/gradcam/
#    gradcam_correct_{model}.png     ← correctly predicted cases
#    gradcam_incorrect_{model}.png   ← misclassified cases
#    gradcam_comparison.png          ← VGG16 vs EfficientNetB3
# ============================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# pytorch-grad-cam library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import load_model_from_experiment, get_test_predictions
from src.dataset import BrainTumorDataset, CLASS_NAMES
from src.transforms import get_transforms


# ──────────────────────────────────────────────
#  Auto-detect project root
# ──────────────────────────────────────────────
def _get_project_root() -> str:
    colab = "/content/drive/MyDrive/brain_tumor_classification"
    if os.path.exists(colab):
        return colab
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


PROJECT_ROOT = _get_project_root()


# ──────────────────────────────────────────────
#  get_target_layer()
#
#  Returns the correct last convolutional layer
#  for each backbone architecture.
#
#  WHY THE LAST CONV LAYER?
#  It has the richest spatial information — earlier
#  layers have lower-level features (edges, textures),
#  later layers are too abstract. The last conv layer
#  balances spatial resolution with semantic meaning.
# ──────────────────────────────────────────────
def get_target_layer(model, backbone_name: str):
    """
    Returns the target convolutional layer for Grad-CAM.

    Args:
        model         : BrainTumorClassifier instance
        backbone_name : Architecture name string

    Returns:
        The target layer module for Grad-CAM to hook into

    Raises:
        ValueError if backbone not supported
    """
    backbone = model.backbone

    if backbone_name == "vgg16":
        # Last conv layer before adaptive pooling
        return backbone.features[-1]

    elif backbone_name == "efficientnet_b3":
        # Last block in the EfficientNet block sequence
        return backbone.blocks[-1]

    elif backbone_name == "densenet121":
        # Last dense block
        return backbone.features.denseblock4

    elif backbone_name == "resnet50":
        # Last residual block in layer4
        return backbone.layer4[-1]

    else:
        raise ValueError(
            f"Backbone '{backbone_name}' not supported for Grad-CAM.\n"
            f"Supported: vgg16, efficientnet_b3, densenet121, resnet50"
        )


# ──────────────────────────────────────────────
#  denormalize_image()
#
#  Reverses ImageNet normalization so the image
#  can be displayed correctly by matplotlib.
#  Normalized images look nearly black — you must
#  denormalize before overlaying the heatmap.
# ──────────────────────────────────────────────
def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized tensor back to a displayable image.

    Args:
        tensor : Image tensor [3, H, W] with ImageNet normalization

    Returns:
        NumPy array [H, W, 3] with values in [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.clone().cpu().float().numpy()
    img = img.transpose(1, 2, 0)      # [3, H, W] → [H, W, 3]
    img = std * img + mean             # reverse normalization
    img = np.clip(img, 0, 1)          # clamp to valid range

    return img.astype(np.float32)


# ──────────────────────────────────────────────
#  generate_gradcam()
#
#  Core function — generates one Grad-CAM heatmap
#  for one image.
# ──────────────────────────────────────────────
def generate_gradcam(
    model,
    target_layer,
    image_tensor: torch.Tensor,
    target_class: int = None,
    device: torch.device = None,
) -> tuple:
    """
    Generates a Grad-CAM heatmap for one image.

    Args:
        model         : BrainTumorClassifier in eval mode
        target_layer  : The conv layer to compute CAM for
        image_tensor  : Single image tensor [3, H, W] (normalized)
        target_class  : Class index to explain.
                        If None, uses the predicted class.
        device        : torch.device

    Returns:
        cam_image  : np.array [H, W, 3] — original image with
                     heatmap overlaid, values in [0, 1]
        pred_class : int — predicted class index
        pred_prob  : float — confidence of predicted class
        grayscale_cam : np.array [H, W] — raw heatmap values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add batch dimension: [3, H, W] → [1, 3, H, W]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # Get prediction first
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob  = probs[0, pred_class].item()

    # If no target class specified, explain the predicted class
    if target_class is None:
        target_class = pred_class

    # ── Initialize GradCAM
    # use_cuda=True speeds up computation on GPU
    cam = GradCAM(
        model        = model,
        target_layers = [target_layer],
    )

    # ── Define what class to explain
    # None means "explain the highest-scoring class"
    targets = None  # explains predicted class automatically

    # ── Generate the CAM
    # grayscale_cam shape: [1, H, W] — values 0 to 1
    grayscale_cam = cam(
        input_tensor  = input_tensor,
        targets       = targets,
    )
    grayscale_cam = grayscale_cam[0]  # remove batch dim → [H, W]

    # ── Get the original image for overlay
    original_image = denormalize_image(image_tensor)

    # ── Overlay heatmap on original image
    # show_cam_on_image blends the jet colormap with the image
    cam_image = show_cam_on_image(
        original_image,
        grayscale_cam,
        use_rgb   = True,
        colormap     = 2,  # cv2.COLORMAP_JET
        image_weight = 0.5,  # 50% original, 50% heatmap
    )
    # Convert to [0, 1] float for matplotlib
    cam_image = cam_image.astype(np.float32) / 255.0

    return cam_image, pred_class, pred_prob, grayscale_cam


# ──────────────────────────────────────────────
#  collect_samples()
#
#  Finds correct and incorrect predictions from
#  the test set for a given model.
#  Used to select interesting cases for Grad-CAM.
# ──────────────────────────────────────────────
def collect_samples(
    model,
    test_loader: DataLoader,
    device: torch.device,
    n_correct: int = 4,
    n_incorrect: int = 4,
) -> tuple:
    """
    Collects correctly and incorrectly classified
    test images for Grad-CAM visualization.

    Args:
        model        : Trained model in eval mode
        test_loader  : Test DataLoader
        device       : torch.device
        n_correct    : Number of correct samples to collect
        n_incorrect  : Number of incorrect samples to collect

    Returns:
        correct_samples   : list of (image_tensor, true_label, pred_label)
        incorrect_samples : list of (image_tensor, true_label, pred_label)
    """
    correct_samples   = []
    incorrect_samples = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds  = torch.argmax(logits, dim=1)

            for i in range(images.size(0)):
                img   = images[i].cpu()
                true  = labels[i].item()
                pred  = preds[i].item()

                if pred == true and len(correct_samples) < n_correct:
                    correct_samples.append((img, true, pred))

                elif pred != true and len(incorrect_samples) < n_incorrect:
                    incorrect_samples.append((img, true, pred))

            # Stop once we have enough samples
            if (len(correct_samples) >= n_correct and
                    len(incorrect_samples) >= n_incorrect):
                break

    return correct_samples, incorrect_samples


# ──────────────────────────────────────────────
#  plot_gradcam_grid()
#
#  Plots a grid of Grad-CAM maps.
#  Each row shows: original | heatmap overlay
#  Title shows true and predicted class.
#  Green title = correct, Red title = incorrect.
# ──────────────────────────────────────────────
def plot_gradcam_grid(
    model,
    target_layer,
    samples: list,
    title: str,
    device: torch.device,
    save_path: str = None,
    class_names: list = CLASS_NAMES,
) -> None:
    """
    Creates a Grad-CAM visualization grid.

    Args:
        model        : BrainTumorClassifier in eval mode
        target_layer : Target conv layer for Grad-CAM
        samples      : List of (image_tensor, true_label, pred_label)
        title        : Figure title
        device       : torch.device
        save_path    : Where to save the figure. Shows inline if None.
        class_names  : List of class name strings
    """
    n       = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.5))
    if n == 1:
        axes = axes[None, :]  # ensure 2D axes array

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for row, (img_tensor, true_label, pred_label) in enumerate(samples):
        # ── Generate Grad-CAM
        cam_image, pred_class, pred_prob, _ = generate_gradcam(
            model, target_layer, img_tensor, device=device
        )

        # ── Original image (denormalized)
        original = denormalize_image(img_tensor)

        # ── Color: green if correct, red if wrong
        color = "#2ca02c" if true_label == pred_label else "#d62728"
        true_name = class_names[true_label].capitalize()
        pred_name = class_names[pred_class].capitalize()

        # ── Left column: original image
        axes[row, 0].imshow(original, cmap="gray")
        axes[row, 0].set_title(
            f"True: {true_name}",
            fontsize=10, fontweight="bold", color="#444"
        )
        axes[row, 0].axis("off")

        # ── Right column: Grad-CAM overlay
        axes[row, 1].imshow(cam_image)
        axes[row, 1].set_title(
            f"Pred: {pred_name} ({pred_prob:.0%})",
            fontsize=10, fontweight="bold", color=color
        )
        axes[row, 1].axis("off")

        # Column labels on first row only
        if row == 0:
            axes[row, 0].set_title(
                "Original MRI\n" + f"True: {true_name}",
                fontsize=10, fontweight="bold", color="#444"
            )
            axes[row, 1].set_title(
                "Grad-CAM heatmap\n" + f"Pred: {pred_name} ({pred_prob:.0%})",
                fontsize=10, fontweight="bold", color=color
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────
#  plot_model_comparison()
#
#  Shows Grad-CAM maps for the same set of images
#  from two different models side by side.
#  Used to visually compare VGG16 vs EfficientNetB3.
#
#  Layout per image row:
#  [Original] | [VGG16 CAM] | [EfficientNet CAM]
# ──────────────────────────────────────────────
def plot_model_comparison(
    model_a,
    layer_a,
    model_b,
    layer_b,
    samples: list,
    label_a: str,
    label_b: str,
    device: torch.device,
    save_path: str = None,
    class_names: list = CLASS_NAMES,
) -> None:
    """
    Side-by-side Grad-CAM comparison between two models.

    Args:
        model_a   : First model (e.g. VGG16)
        layer_a   : Target layer for model_a
        model_b   : Second model (e.g. EfficientNetB3)
        layer_b   : Target layer for model_b
        samples   : List of (image_tensor, true_label, pred_label)
        label_a   : Name for model_a (used in plot title)
        label_b   : Name for model_b
        device    : torch.device
        save_path : Where to save the figure
        class_names : List of class name strings
    """
    n     = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
    if n == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"Grad-CAM comparison: {label_a} vs {label_b}",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Column headers
    col_titles = ["Original MRI", f"{label_a}\nGrad-CAM", f"{label_b}\nGrad-CAM"]

    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=11, fontweight="bold", pad=10)

    for row, (img_tensor, true_label, _) in enumerate(samples):
        true_name = class_names[true_label].capitalize()

        # ── Original image
        original = denormalize_image(img_tensor)
        axes[row, 0].imshow(original, cmap="gray")
        axes[row, 0].set_ylabel(
            f"True: {true_name}", fontsize=10,
            fontweight="bold", rotation=0,
            labelpad=70, va="center"
        )
        axes[row, 0].axis("off")

        # ── Model A Grad-CAM
        cam_a, pred_a, prob_a, _ = generate_gradcam(
            model_a, layer_a, img_tensor, device=device
        )
        color_a = "#2ca02c" if pred_a == true_label else "#d62728"
        axes[row, 1].imshow(cam_a)
        axes[row, 1].set_title(
            f"Pred: {class_names[pred_a].capitalize()} ({prob_a:.0%})",
            fontsize=9, color=color_a
        )
        axes[row, 1].axis("off")

        # ── Model B Grad-CAM
        cam_b, pred_b, prob_b, _ = generate_gradcam(
            model_b, layer_b, img_tensor, device=device
        )
        color_b = "#2ca02c" if pred_b == true_label else "#d62728"
        axes[row, 2].imshow(cam_b)
        axes[row, 2].set_title(
            f"Pred: {class_names[pred_b].capitalize()} ({prob_b:.0%})",
            fontsize=9, color=color_b
        )
        axes[row, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────
#  run_gradcam_analysis()
#
#  Master function — generates all Grad-CAM
#  figures needed for the paper.
#
#  Produces:
#  1. Correct + incorrect predictions for VGG16
#  2. Correct + incorrect predictions for EfficientNetB3
#  3. Side-by-side VGG16 vs EfficientNetB3 comparison
# ──────────────────────────────────────────────
def run_gradcam_analysis(
    device:          torch.device = None,
    experiments_dir: str  = None,
    data_dir:        str  = None,
    outputs_dir:     str  = None,
    n_samples:       int  = 4,
    batch_size:      int  = 16,
    image_size:      int  = 224,
) -> None:
    """
    Runs the full Grad-CAM analysis for the paper.

    Args:
        device          : torch.device (auto-detected if None)
        experiments_dir : Path to experiments/ folder
        data_dir        : Path to data/ folder
        outputs_dir     : Where to save Grad-CAM images
        n_samples       : Number of correct/incorrect samples per figure
        batch_size      : DataLoader batch size
        image_size      : Input image size (must match training)

    Usage:
        from src.gradcam import run_gradcam_analysis
        run_gradcam_analysis()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if experiments_dir is None:
        experiments_dir = os.path.join(PROJECT_ROOT, "experiments")
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")
    if outputs_dir is None:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs", "gradcam")

    os.makedirs(outputs_dir, exist_ok=True)

    # ── Build test DataLoader
    test_transform = get_transforms(image_size=image_size, phase="test")
    test_dataset   = BrainTumorDataset(
        root_dir  = os.path.join(data_dir, "Testing"),
        transform = test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = True,   # shuffle so we get varied samples
        num_workers = 2,
    )

    # ──────────────────────────────────────────
    #  Figure 1 and 2: VGG16 correct + incorrect
    # ──────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Generating Grad-CAM for VGG16...")
    print("=" * 52)

    model_vgg, _ = load_model_from_experiment(
        "baseline_vgg16", device, experiments_dir
    )
    layer_vgg = get_target_layer(model_vgg, "vgg16")

    correct_vgg, incorrect_vgg = collect_samples(
        model_vgg, test_loader, device,
        n_correct=n_samples, n_incorrect=n_samples
    )

    # Correctly classified cases
    plot_gradcam_grid(
        model        = model_vgg,
        target_layer = layer_vgg,
        samples      = correct_vgg,
        title        = "VGG16 — Correctly classified cases\n(Red = high attention, Blue = low attention)",
        device       = device,
        save_path    = os.path.join(outputs_dir, "gradcam_vgg16_correct.png"),
    )

    # Misclassified cases
    if incorrect_vgg:
        plot_gradcam_grid(
            model        = model_vgg,
            target_layer = layer_vgg,
            samples      = incorrect_vgg,
            title        = "VGG16 — Misclassified cases\n(Note where attention falls compared to correct cases)",
            device       = device,
            save_path    = os.path.join(outputs_dir, "gradcam_vgg16_incorrect.png"),
        )

    # ──────────────────────────────────────────
    #  Figure 3 and 4: EfficientNetB3 correct + incorrect
    # ──────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Generating Grad-CAM for EfficientNetB3...")
    print("=" * 52)

    model_eff, _ = load_model_from_experiment(
        "baseline_efficientnet_b3", device, experiments_dir
    )
    layer_eff = get_target_layer(model_eff, "efficientnet_b3")

    correct_eff, incorrect_eff = collect_samples(
        model_eff, test_loader, device,
        n_correct=n_samples, n_incorrect=n_samples
    )

    plot_gradcam_grid(
        model        = model_eff,
        target_layer = layer_eff,
        samples      = correct_eff,
        title        = "EfficientNetB3 — Correctly classified cases",
        device       = device,
        save_path    = os.path.join(outputs_dir, "gradcam_efficientnet_correct.png"),
    )

    if incorrect_eff:
        plot_gradcam_grid(
            model        = model_eff,
            target_layer = layer_eff,
            samples      = incorrect_eff,
            title        = "EfficientNetB3 — Misclassified cases",
            device       = device,
            save_path    = os.path.join(outputs_dir, "gradcam_efficientnet_incorrect.png"),
        )

    # ──────────────────────────────────────────
    #  Figure 5: Side-by-side model comparison
    #  Use the same images through both models
    # ──────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Generating VGG16 vs EfficientNetB3 comparison...")
    print("=" * 52)

    # Use the incorrect EfficientNet samples for comparison
    # (most interesting — shows where attention differs)
    comparison_samples = (incorrect_eff if incorrect_eff
                          else correct_eff)[:n_samples]

    plot_model_comparison(
        model_a   = model_vgg,
        layer_a   = layer_vgg,
        model_b   = model_eff,
        layer_b   = layer_eff,
        samples   = comparison_samples,
        label_a   = "VGG16",
        label_b   = "EfficientNetB3",
        device    = device,
        save_path = os.path.join(outputs_dir, "gradcam_comparison_vgg_vs_effnet.png"),
    )

    print("\n" + "=" * 52)
    print("  Grad-CAM analysis complete.")
    print(f"  All figures saved to: {outputs_dir}")
    print("=" * 52)
    print("\n  Files generated:")
    for f in os.listdir(outputs_dir):
        print(f"    {f}")
