"""Model factory: shapes, head size, and Grad-CAM target discovery."""

import torch

from radarmd.data.constants import NUM_CLASSES
from radarmd.models.factory import create_model, gradcam_target_layer


def test_forward_shape_densenet():
    model = create_model("densenet121", pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_forward_shape_convnext():
    model = create_model("convnext_tiny", pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_gradcam_target_is_conv():
    model = create_model("densenet121", pretrained=False)
    layer = gradcam_target_layer(model)
    assert isinstance(layer, torch.nn.Conv2d)


def test_custom_num_classes():
    model = create_model("resnet50", pretrained=False, num_classes=5)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 5)
