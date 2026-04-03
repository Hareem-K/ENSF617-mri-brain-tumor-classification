# ============================================================
#  src/models.py — Model builder for pretrained backbones
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS FILE DOES:
#  1. Builds a pretrained timm backbone (ResNet, VGG, etc.)
#  2. Replaces the classifier head for 4 brain-tumor classes
#  3. Provides helper methods for freezing / unfreezing layers
#
#  Your team can reuse this file for multiple backbones.
#  For your part, set cfg.model.backbone = "resnet50".
# ============================================================

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import timm


class BrainTumorClassifier(nn.Module):
    """
    Generic image classifier built on top of a pretrained timm backbone.

    Example:
        model = BrainTumorClassifier(
            backbone_name='resnet50',
            num_classes=4,
            pretrained=True,
            dropout_rate=0.3,
        )
    """

    def __init__(
        self,
        backbone_name: str = 'resnet50',
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        # num_classes=0 removes timm's default classifier.
        # global_pool='avg' gives one feature vector per image.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
        )

        num_features = getattr(self.backbone, 'num_features', None)
        if num_features is None:
            raise AttributeError(
                f"Backbone '{backbone_name}' does not expose 'num_features'."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters. Classifier stays trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze the entire backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, n_blocks: int = 2) -> None:
        """
        Keep most of the backbone frozen, but unfreeze the last few top-level blocks.
        This works reasonably well across several timm backbones.
        """
        self.freeze_backbone()
        children = list(self.backbone.children())
        if not children:
            self.unfreeze_backbone()
            return

        n_blocks = max(1, min(n_blocks, len(children)))
        for module in children[-n_blocks:]:
            for param in module.parameters():
                param.requires_grad = True

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: Any) -> BrainTumorClassifier:
    """Build model from ExperimentConfig."""
    model = BrainTumorClassifier(
        backbone_name=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        dropout_rate=cfg.regularization.dropout_rate,
    )
    return model


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    """Safely convert config/dataclass objects into plain dictionaries."""
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, '__dict__'):
        return dict(cfg.__dict__)
    return {'config_repr': repr(cfg)}


if __name__ == '__main__':
    model = BrainTumorClassifier(backbone_name='resnet50')
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print('=' * 50)
    print(f'Backbone      : {model.backbone_name}')
    print(f'Output shape  : {tuple(out.shape)}')
    print(f'Trainable     : {model.num_trainable_parameters():,}')
    print(f'Total params  : {model.num_total_parameters():,}')
    print('=' * 50)
