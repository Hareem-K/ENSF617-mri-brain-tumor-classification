# ============================================================
#  src/cbam.py — Convolutional Block Attention Module
#  Brain Tumor MRI Classification Project
#
#  PAPER REFERENCE:
#  Woo et al., "CBAM: Convolutional Block Attention Module"
#  ECCV 2018. https://arxiv.org/abs/1807.06521
#
#  CLASSES IN THIS FILE:
#  1. ChannelAttention   — which feature channels matter
#  2. SpatialAttention   — which spatial locations matter
#  3. CBAM               — combines both in sequence
#  4. CBAMEfficientNetB3 — EfficientNetB3 + CBAM (first proposal)
#  5. CBAMVgg16          — VGG16 + CBAM (main proposal)
#
#  OPTIMIZERS:
#  get_cbam_optimizer()     — for CBAMEfficientNetB3
#  get_vgg_cbam_optimizer() — for CBAMVgg16
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ──────────────────────────────────────────────
#  1. Channel Attention Module
#
#  Answers: which FEATURE CHANNELS matter?
#  Suppresses irrelevant channels (background
#  textures) and amplifies relevant ones
#  (tumor-specific patterns).
# ──────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.

    Args:
        in_channels : Number of input feature channels (C)
        reduction   : Bottleneck ratio for shared MLP (default 16)

    Input  : Feature map [B, C, H, W]
    Output : Channel-refined feature map [B, C, H, W]
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        hidden = max(in_channels // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        avg_pool = F.adaptive_avg_pool2d(x, 1)   # [B, C, 1, 1]
        max_pool = F.adaptive_max_pool2d(x, 1)   # [B, C, 1, 1]

        avg_out  = self.mlp(avg_pool)             # [B, C]
        max_out  = self.mlp(max_pool)             # [B, C]

        weights  = self.sigmoid(avg_out + max_out)
        weights  = weights.view(B, C, 1, 1)

        return x * weights


# ──────────────────────────────────────────────
#  2. Spatial Attention Module
#
#  Answers: WHERE in the image should we look?
#  Suppresses background regions and amplifies
#  the tumor area spatial locations.
# ──────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.

    Args:
        kernel_size : Convolution kernel size (7 recommended)

    Input  : Channel-refined feature map [B, C, H, W]
    Output : Spatially-refined feature map [B, C, H, W]
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        padding  = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels  = 2,
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = padding,
            bias         = False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)     # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)   # [B, 2, H, W]
        attention_map = self.sigmoid(self.conv(combined)) # [B, 1, H, W]
        return x * attention_map


# ──────────────────────────────────────────────
#  3. Full CBAM Module
#
#  Applies Channel Attention then Spatial Attention
#  in sequence. Channel first, spatial second —
#  this ordering was validated in the original paper.
# ──────────────────────────────────────────────
class CBAM(nn.Module):
    """
    Full Convolutional Block Attention Module.

    Args:
        in_channels    : Number of input channels
        reduction      : Channel reduction ratio (default 16)
        spatial_kernel : Kernel size for spatial attention (default 7)

    Input  : Feature map [B, C, H, W]
    Output : Attention-refined feature map [B, C, H, W]
    """

    def __init__(
        self,
        in_channels:    int,
        reduction:      int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels = in_channels,
            reduction   = reduction,
        )
        self.spatial_attention = SpatialAttention(
            kernel_size = spatial_kernel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ──────────────────────────────────────────────
#  4. CBAMEfficientNetB3
#
#  EfficientNetB3 + CBAM inserted after the last
#  convolutional block, before global avg pooling.
#  First proposed model — used in early experiments.
# ──────────────────────────────────────────────
class CBAMEfficientNetB3(nn.Module):
    """
    EfficientNetB3 backbone with CBAM attention.

    Args:
        num_classes  : Number of output classes (default 4)
        dropout_rate : Dropout in classifier head (default 0.3)
        pretrained   : Use ImageNet pretrained weights (default True)
        reduction    : CBAM channel reduction ratio (default 16)
    """

    def __init__(
        self,
        num_classes:  int   = 4,
        dropout_rate: float = 0.3,
        pretrained:   bool  = True,
        reduction:    int   = 16,
    ):
        super().__init__()

        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained   = pretrained,
            num_classes  = 0,
            global_pool  = "",
        )

        num_features = 1536

        self.cbam = CBAM(
            in_channels = num_features,
            reduction   = reduction,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.cbam(features)
        features = self.global_pool(features)
        features = features.flatten(1)
        logits   = self.classifier(features)
        return logits

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        self._freeze_backbone()
        print("Backbone frozen. CBAM + classifier training.")
        print(f"Trainable params: {self.count_trainable_params():,}")

    def unfreeze_top_layers(self, num_blocks: int = 2):
        self._freeze_backbone()
        children = [
            (n, m) for n, m in self.backbone.named_children()
            if sum(p.numel() for p in m.parameters()) > 0
        ]
        for name, module in children[-num_blocks:]:
            for param in module.parameters():
                param.requires_grad = True
            params = sum(p.numel() for p in module.parameters())
            print(f"  Unfrozen: {name} ({params:,} params)")
        print(f"Trainable params: {self.count_trainable_params():,}")

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Full model unfrozen.")
        print(f"Trainable params: {self.count_trainable_params():,}")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_model_info(self):
        total      = self.count_total_params()
        trainable  = self.count_trainable_params()
        cbam_params = sum(p.numel() for p in self.cbam.parameters())
        print("=" * 52)
        print("  CBAMEfficientNetB3 — Model Summary")
        print("=" * 52)
        print(f"  Backbone       : EfficientNetB3 (pretrained)")
        print(f"  CBAM params    : {cbam_params:,}")
        print(f"  Total params   : {total:,}")
        print(f"  Trainable now  : {trainable:,}")
        print(f"  Dropout        : {self.dropout_rate}")
        print(f"  Output classes : {self.num_classes}")
        print("=" * 52)


# ──────────────────────────────────────────────
#  5. CBAMVgg16
#
#  VGG16 + CBAM inserted after the last conv block,
#  before global average pooling.
#
#  WHY VGG16?
#  VGG16 achieved the highest test accuracy (94.31%)
#  in the baseline study. Grad-CAM analysis showed
#  it still misclassifies glioma cases due to spatial
#  attention deficiencies. CBAM directly addresses
#  this with only 0.01% parameter overhead.
#
#  VGG16 last conv block outputs 512 channels.
# ──────────────────────────────────────────────
class CBAMVgg16(nn.Module):
    """
    Proposed model: VGG16 + CBAM attention module.

    Inserts CBAM after the last convolutional block
    of VGG16, before global average pooling.

    Args:
        num_classes  : Number of output classes (default 4)
        dropout_rate : Dropout in classifier head (default 0.3)
        pretrained   : Use ImageNet pretrained weights (default True)
        reduction    : CBAM channel reduction ratio (default 16)
    """

    def __init__(
        self,
        num_classes:  int   = 4,
        dropout_rate: float = 0.3,
        pretrained:   bool  = True,
        reduction:    int   = 16,
    ):
        super().__init__()

        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

        self.backbone = timm.create_model(
            "vgg16",
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = "",
        )

        # VGG16 last conv block outputs 512 channels
        num_features = 512

        self.cbam = CBAM(
            in_channels = num_features,
            reduction   = reduction,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)  # [B, 512, 7, 7] — conv layers only
        features = self.cbam(features)         # [B, 512, H, W]
        features = self.global_pool(features)  # [B, 512, 1, 1]
        features = features.flatten(1)         # [B, 512]
        logits   = self.classifier(features)   # [B, num_classes]
        return logits

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        self._freeze_backbone()
        print("Backbone frozen. CBAM + classifier training.")
        print(f"Trainable params: {self.count_trainable_params():,}")

    def unfreeze_top_layers(self, num_blocks: int = 2):
        self._freeze_backbone()
        children = [
            (n, m) for n, m in self.backbone.named_children()
            if sum(p.numel() for p in m.parameters()) > 0
        ]
        for name, module in children[-num_blocks:]:
            for param in module.parameters():
                param.requires_grad = True
            params = sum(p.numel() for p in module.parameters())
            print(f"  Unfrozen: {name} ({params:,} params)")
        print(f"Trainable params: {self.count_trainable_params():,}")

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Full model unfrozen.")
        print(f"Trainable params: {self.count_trainable_params():,}")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_model_info(self):
        total       = self.count_total_params()
        trainable   = self.count_trainable_params()
        cbam_params = sum(p.numel() for p in self.cbam.parameters())
        print("=" * 52)
        print("  CBAMVgg16 — Model Summary")
        print("=" * 52)
        print(f"  Backbone       : VGG16 (pretrained)")
        print(f"  CBAM params    : {cbam_params:,}")
        print(f"  Total params   : {total:,}")
        print(f"  Trainable now  : {trainable:,}")
        print(f"  Dropout        : {self.dropout_rate}")
        print(f"  Output classes : {self.num_classes}")
        print("=" * 52)


# ──────────────────────────────────────────────
#  Optimizers
# ──────────────────────────────────────────────
def get_cbam_optimizer(
    model:        CBAMEfficientNetB3,
    head_lr:      float = 1e-3,
    backbone_lr:  float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """AdamW optimizer for CBAMEfficientNetB3."""
    param_groups = [
        {
            "params": list(model.cbam.parameters()) +
                      list(model.classifier.parameters()),
            "lr":   head_lr,
            "name": "cbam_and_head",
        },
        {
            "params": model.backbone.parameters(),
            "lr":   backbone_lr,
            "name": "backbone",
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def get_vgg_cbam_optimizer(
    model:        CBAMVgg16,
    head_lr:      float = 1e-3,
    backbone_lr:  float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """AdamW optimizer for CBAMVgg16."""
    param_groups = [
        {
            "params": list(model.cbam.parameters()) +
                      list(model.classifier.parameters()),
            "lr":   head_lr,
            "name": "cbam_and_head",
        },
        {
            "params": model.backbone.parameters(),
            "lr":   backbone_lr,
            "name": "backbone",
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ──────────────────────────────────────────────
#  Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing cbam.py on: {device}\n")

    # Test individual modules
    print("Testing ChannelAttention...")
    ca  = ChannelAttention(in_channels=64).to(device)
    x   = torch.randn(2, 64, 14, 14).to(device)
    out = ca(x)
    assert out.shape == x.shape
    print(f"  {x.shape} → {out.shape}  PASSED")

    print("Testing SpatialAttention...")
    sa  = SpatialAttention().to(device)
    out = sa(x)
    assert out.shape == x.shape
    print(f"  {x.shape} → {out.shape}  PASSED")

    print("Testing CBAM...")
    cbam = CBAM(in_channels=64).to(device)
    out  = cbam(x)
    assert out.shape == x.shape
    print(f"  {x.shape} → {out.shape}  PASSED")

    print("\nTesting CBAMEfficientNetB3...")
    model_eff = CBAMEfficientNetB3(pretrained=False).to(device)
    model_eff.print_model_info()
    dummy  = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model_eff(dummy)
    assert out.shape == (2, 4)
    print(f"  Output: {out.shape}  PASSED")

    print("\nTesting CBAMVgg16...")
    model_vgg = CBAMVgg16(pretrained=False).to(device)
    model_vgg.print_model_info()
    with torch.no_grad():
        out = model_vgg(dummy)
    assert out.shape == (2, 4)
    print(f"  Output: {out.shape}  PASSED")

    print("\ncbam.py — all tests passed.")
