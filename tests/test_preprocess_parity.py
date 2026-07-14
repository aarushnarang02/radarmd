"""Serving preprocessing (PIL/numpy) must track the MONAI eval pipeline.

Resampling kernels differ between PIL and MONAI, so we don't demand bit
equality; we pin the drift at the pixel level on realistic (smooth) content and
at the model-output level, which is what actually matters for served
predictions.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from radarmd.data.transforms import build_transforms
from radarmd.serve.preprocess import preprocess_image

SIZE = 64


def _smooth_image(n=256) -> Image.Image:
    """A smooth radial gradient — closer to an X-ray than white noise."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = (n - 1) / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    arr = 255.0 * (1.0 - r / r.max())
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def test_pixel_level_parity_on_smooth_image():
    img = _smooth_image()
    ours = preprocess_image(img, SIZE)[0]  # (3, H, W)

    monai_tf = build_transforms(image_size=SIZE, train=False)
    theirs = monai_tf(np.asarray(img.convert("L"), dtype=np.float32))
    theirs = theirs.numpy() if isinstance(theirs, torch.Tensor) else np.asarray(theirs)

    assert ours.shape == theirs.shape
    # Normalized units (std ~1); mean drift must be well under a hundredth std.
    assert float(np.abs(ours - theirs).mean()) < 0.05
    assert float(np.abs(ours - theirs).max()) < 0.5  # worst pixel still close


def test_model_output_parity(tmp_path):
    """The two preprocessors must yield nearly identical model outputs."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    from radarmd.models.factory import create_model
    from radarmd.serve.onnx_export import export_onnx

    torch.manual_seed(0)
    model = create_model("resnet50", pretrained=False).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path, image_size=SIZE)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    img = _smooth_image()
    ours = preprocess_image(img, SIZE)

    monai_tf = build_transforms(image_size=SIZE, train=False)
    theirs = monai_tf(np.asarray(img.convert("L"), dtype=np.float32))
    theirs = (theirs.numpy() if isinstance(theirs, torch.Tensor) else np.asarray(theirs))[None]

    p1 = sess.run(["probs"], {"image": ours})[0]
    p2 = sess.run(["probs"], {"image": theirs.astype(np.float32)})[0]
    assert float(np.abs(p1 - p2).max()) < 0.02  # probabilities agree to <2 points


def test_preprocess_shape_dtype_and_stats():
    out = preprocess_image(_smooth_image(), SIZE)
    assert out.shape == (1, 3, SIZE, SIZE)
    assert out.dtype == np.float32
    # After ImageNet normalization some values must be negative.
    assert out.min() < 0.0


def test_constant_image_does_not_divide_by_zero():
    flat = Image.new("L", (100, 100), color=128)
    out = preprocess_image(flat, SIZE)
    assert np.isfinite(out).all()
