"""Backbone factory: one interface for every classification model we try.

All backbones come from ``timm`` so DenseNet-121, ConvNeXt, and anything else we
sweep over share the same construction path and a 14-logit head. We also expose
the final convolutional stage, which stage 4's Grad-CAM needs as its target
layer.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from ..data.constants import NUM_CLASSES

# Backbones we actually train. Keys are the short names used in configs; values
# are the timm model identifiers. Adding a row is all it takes to sweep a new one.
BACKBONES: dict[str, str] = {
    "densenet121": "densenet121",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "resnet50": "resnet50",  # cheap baseline for the "6-point gain" comparison
}


def create_model(
    backbone: str = "densenet121",
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Create a multi-label classification backbone with a ``num_classes`` head.

    Returns raw logits (no sigmoid); loss and metrics apply the activation. Use
    ``backbone`` keys from :data:`BACKBONES`, or any valid timm model name.
    """
    timm_name = BACKBONES.get(backbone, backbone)
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the last spatial feature layer for Grad-CAM.

    ``timm`` exposes the feature extractor under different attribute names per
    family; ``forward_features`` always yields the pre-pool feature map, but
    pytorch-grad-cam needs an actual module. We locate the last Conv2d /
    LayerNorm-bearing block by walking the module tree.
    """
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found; cannot pick a Grad-CAM target.")
    return last_conv


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count (for logging / sanity checks)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
