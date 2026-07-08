"""Grad-CAM heatmaps for the chest X-ray classifier.

Thin wrapper over the ``grad-cam`` (pytorch-grad-cam) library. Given a trained
model and an input image, it produces a per-class heatmap over the last
convolutional feature map, which stage 4 compares against ground-truth boxes and
the serving app overlays on the X-ray to flag regions of concern.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data.constants import LABEL_TO_INDEX
from ..models.factory import gradcam_target_layer


class GradCAMExplainer:
    """Compute Grad-CAM heatmaps for one or more target classes.

    Parameters
    ----------
    model:        a trained classifier returning ``(B, num_classes)`` logits.
    target_layer: module to hook; defaults to the model's last Conv2d.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        # Imported lazily so the core package doesn't require the interpret extra.
        from pytorch_grad_cam import GradCAM

        self.model = model.eval()
        self.target_layer = target_layer or gradcam_target_layer(model)
        self._cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def heatmap(self, image: torch.Tensor, class_name: str) -> np.ndarray:
        """Return a normalized (H, W) heatmap for ``class_name`` on one image.

        ``image`` is a (3, H, W) or (1, 3, H, W) tensor already normalized the way
        the model expects. Output is float32 in [0, 1] at the input resolution.
        """
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if class_name not in LABEL_TO_INDEX:
            raise KeyError(f"Unknown pathology: {class_name!r}")
        if image.dim() == 3:
            image = image.unsqueeze(0)

        targets = [ClassifierOutputTarget(LABEL_TO_INDEX[class_name])]
        cam = self._cam(input_tensor=image, targets=targets)  # (B, H, W) in [0,1]
        return cam[0].astype(np.float32)

    def heatmaps(self, image: torch.Tensor, class_names: list[str]) -> dict[str, np.ndarray]:
        """Heatmaps for several classes on the same image."""
        return {name: self.heatmap(image, name) for name in class_names}
