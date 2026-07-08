"""Grad-CAM wrapper produces a valid heatmap on a real (small) model."""

import numpy as np
import pytest
import torch

from radarmd.data.constants import NUM_CLASSES
from radarmd.models.factory import create_model

pytest.importorskip("pytorch_grad_cam")

from radarmd.interpret.gradcam import GradCAMExplainer  # noqa: E402


def test_heatmap_shape_and_range():
    model = create_model("resnet50", pretrained=False)
    explainer = GradCAMExplainer(model)
    img = torch.randn(3, 64, 64)
    cam = explainer.heatmap(img, "Mass")
    assert cam.shape == (64, 64)
    assert cam.dtype == np.float32
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5


def test_unknown_class_raises():
    model = create_model("resnet50", pretrained=False)
    explainer = GradCAMExplainer(model)
    with pytest.raises(KeyError):
        explainer.heatmap(torch.randn(3, 64, 64), "NotAPathology")


def test_heatmaps_for_multiple_classes():
    model = create_model("densenet121", pretrained=False)
    explainer = GradCAMExplainer(model)
    out = explainer.heatmaps(torch.randn(1, 3, 64, 64), ["Mass", "Effusion"])
    assert set(out) == {"Mass", "Effusion"}
    assert all(v.shape == (64, 64) for v in out.values())
    assert NUM_CLASSES == 14  # guard: class indexing assumes the 14-class head
