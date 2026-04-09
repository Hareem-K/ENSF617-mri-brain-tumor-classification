# ============================================================
#  src/models.py — Step 3 of 8
#  Brain Tumor MRI Classification Project
#
#  WHAT THIS VERSION DOES:
#  Loads a single pretrained backbone (EfficientNetB3)
#  from ImageNet using the timm library.
#  Removes its original 1000-class head.
#  Runs a dummy image through it to confirm the output.
#
#  WHAT "num_classes=0" MEANS:
#  Normally EfficientNetB3 outputs 1000 class scores.
#  Setting num_classes=0 removes that final layer and
#  gives us the raw feature vector instead — a list of
#  1536 numbers that describe what the model "sees"
#  in the image. Our own head will classify from these.
#
#  WHAT "global_pool='avg'" MEANS:
#  After all the convolutional layers, the feature map
#  is still a 3D tensor [channels, height, width].
#  Global average pooling collapses the height and width
#  dimensions by taking the average — giving us a flat
#  1D vector of [1536] per image. This is what our
#  classifier head will work with.
# ============================================================

import torch
import timm


def load_backbone(
    backbone_name: str  = "efficientnet_b3",
    pretrained:    bool = True,
) -> torch.nn.Module:
    """
    Loads a pretrained backbone from timm with the
    classifier head removed.

    Args:
        backbone_name : Name of the architecture in timm
        pretrained    : Use ImageNet pretrained weights

    Returns:
        backbone model with num_features attribute

    Usage:
        backbone = load_backbone("efficientnet_b3")
        features = backbone(images)  # shape: [B, 1536]
    """
    backbone = timm.create_model(
        backbone_name,
        pretrained  = pretrained,
        num_classes = 0,      # removes original head
        global_pool = "avg",  # collapses spatial dims
    )

    return backbone


# ============================================================
#  Step 4 — Freeze and Unfreeze the Backbone
#
#  WHAT FREEZING DOES:
#  Sets requires_grad=False on every backbone parameter.
#  PyTorch will not compute gradients for frozen params —
#  they will not update during training. This protects
#  the pretrained ImageNet weights from being overwritten
#  too early.
#
#  WHY WE FREEZE FIRST:
#  In Phase 1 (warmup) we want ONLY our new classifier
#  head to learn. The backbone already knows how to extract
#  visual features — we just need the head to learn how to
#  use those features for our 4 tumor classes.
#
#  PHASED FINE-TUNING PLAN:
#  Phase 1 (epochs 1-5)  : backbone fully frozen
#  Phase 2 (epochs 6-20) : top 2 blocks unfrozen
#  Phase 3 (epochs 21+)  : entire backbone unfrozen
# ============================================================

def count_parameters(model: torch.nn.Module) -> dict:
    """
    Counts total and trainable parameters in a model.

    Args:
        model : Any PyTorch model

    Returns:
        dict with 'total' and 'trainable' counts

    Usage:
        counts = count_parameters(backbone)
        print(counts['trainable'])
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return {"total": total, "trainable": trainable}


def freeze_backbone(backbone: torch.nn.Module) -> None:
    """
    Freezes ALL backbone parameters.
    After calling this, only the classifier head will train.

    Args:
        backbone : The pretrained backbone module

    Usage:
        freeze_backbone(backbone)
        # now backbone params will not update during training
    """
    for param in backbone.parameters():
        param.requires_grad = False


def unfreeze_top_layers(
    backbone:   torch.nn.Module,
    num_blocks: int = 2,
) -> None:
    """
    Unfreezes the top N child blocks of the backbone
    that actually contain trainable parameters.
    Used in Phase 2 (fine-tuning) after the head has
    stabilised during warmup.

    WHY TOP LAYERS?
    The top (last) layers of a backbone are responsible
    for high-level feature detection — things like
    "this looks like a circular mass" or "this texture
    is typical of tumor tissue". These are the layers
    most relevant to our specific task, so we unfreeze
    them first.

    WHY FILTER BY PARAMS > 0?
    Some blocks (like global_pool, classifier) have zero
    parameters after we remove the head with num_classes=0.
    Unfreezing them does nothing — we skip them and only
    count blocks that actually have weights to train.

    Args:
        backbone   : The pretrained backbone module
        num_blocks : How many top-level blocks to unfreeze

    Usage:
        unfreeze_top_layers(backbone, num_blocks=2)
    """
    # First freeze everything
    freeze_backbone(backbone)

    # Get only children that have actual parameters
    children_with_params = [
        (name, module)
        for name, module in backbone.named_children()
        if sum(p.numel() for p in module.parameters()) > 0
    ]

    # Unfreeze the last num_blocks of those
    for name, module in children_with_params[-num_blocks:]:
        for param in module.parameters():
            param.requires_grad = True
        params = sum(p.numel() for p in module.parameters())
        print(f"  Unfrozen block: {name:<20} ({params:,} params)")


def unfreeze_all(backbone: torch.nn.Module) -> None:
    """
    Unfreezes ALL backbone parameters.
    Used in Phase 3 (full fine-tuning) with a very
    small learning rate (1e-5) to gently polish the
    entire model.

    Args:
        backbone : The pretrained backbone module

    Usage:
        unfreeze_all(backbone)
    """
    for param in backbone.parameters():
        param.requires_grad = True


class BrainTumorClassifier(torch.nn.Module):
    """
    Complete model combining a pretrained backbone and a custom classifier head.
    Supports EfficientNetB3, ResNet50, DenseNet121, and VGG16.
    """
    def __init__(
        self, 
        backbone_name: str = "efficientnet_b3", 
        pretrained: bool = True, 
        dropout_rate: float = 0.3, 
        num_classes: int = 4
    ):
        super().__init__()
        
        # 1. Load the backbone
        self.backbone = load_backbone(backbone_name, pretrained)
        
        # 2. Get the output feature size dynamically (The Bulletproof Method)
        # We pass a dummy tensor through the backbone to see exactly 
        # how many features it outputs, avoiding VGG architecture quirks.
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, 3, 224, 224)
            dummy_out = self.backbone(dummy_tensor)
            in_features = dummy_out.shape[1]
        
        # 3. Build the custom head
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def get_optimizer(
    model: torch.nn.Module, 
    backbone_lr: float, 
    head_lr: float, 
    weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    """
    Creates an AdamW optimizer with different learning rates.
    """
    param_groups = [
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

def get_scheduler(
    optimizer: torch.optim.Optimizer, 
    epochs: int, 
    min_lr: float = 1e-6
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Creates a Cosine Annealing learning rate scheduler.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=min_lr
    )


# ──────────────────────────────────────────────
#  Quick self-test
#  Run this file directly to verify Steps 3+4.
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("-" * 45)

    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test all 4 required backbones
    backbones_to_test = ["efficientnet_b3", "resnet50", "densenet121", "vgg16"]
    
    for b_name in backbones_to_test:
        print(f"\nTesting architecture: {b_name}")
        print("-" * 30)
        
        try:
            # Initialize full classifier
            model = BrainTumorClassifier(backbone_name=b_name, pretrained=False).to(device)
            
            # Run forward pass
            with torch.no_grad():
                logits = model(dummy_input)
                
            counts = count_parameters(model)
            
            print(f"  Input shape      : {dummy_input.shape}")
            print(f"  Output shape     : {logits.shape}  ← (should be [4, 4])")
            print(f"  Total params     : {counts['total']:>12,}")
            
            # Quick test of the optimizer structure
            opt = get_optimizer(model, backbone_lr=1e-5, head_lr=1e-3)
            print(f"  Optimizer groups : {len(opt.param_groups)} (Backbone + Head)")
            
        except Exception as e:
            print(f"  [ERROR] {b_name} failed: {e}")

    print("\n" + "=" * 45)
    print("Complete.")
    print("=" * 45)
