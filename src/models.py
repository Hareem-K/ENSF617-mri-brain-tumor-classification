import torch
import torch.nn as nn
from torchvision import models

class BrainTumorVGG(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3) -> None:
        super().__init__()
        backbone = models.vgg16(weights='IMAGENET1K_V1')

        self.features = backbone.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits
    
def build_vgg(num_classes: int = 4, dropout_rate: float = 0.3) -> BrainTumorVGG:
    model = BrainTumorVGG(num_classes=num_classes, dropout_rate=dropout_rate)
    return model