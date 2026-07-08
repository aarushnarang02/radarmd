"""Export a trained classifier to ONNX and verify numerical parity.

ONNX Runtime gives ~4x faster CPU inference than eager PyTorch for this model,
which is what makes the ~120 ms/image serving target reachable on Cloud Run's
CPU instances. We export with a dynamic batch axis and check that ORT and
PyTorch agree to a tight tolerance before trusting the exported graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.constants import NUM_CLASSES

DEFAULT_OPSET = 17


def export_onnx(
    model: nn.Module,
    path: str | Path,
    image_size: int = 320,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export ``model`` (returning logits) to ONNX at ``path``.

    A sigmoid is appended so the ONNX graph emits probabilities directly, which
    keeps the serving code trivial and identical across PyTorch/ORT. Batch and
    spatial dims are dynamic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapped = _WithSigmoid(model).eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        wrapped,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["probs"],
        opset_version=opset,
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "probs": {0: "batch"},
        },
        # Use the stable TorchScript exporter; the new dynamo path pulls in
        # onnxscript and changes the dynamic-shape API.
        dynamo=False,
    )
    return path


def quantize_onnx(onnx_path: str | Path, quant_path: str | Path | None = None) -> Path:
    """INT8 dynamic-quantize an ONNX model for faster CPU inference.

    Dynamic quantization casts weights to int8 (activations quantized on the fly),
    which is what delivers the large CPU speedup on x86 serving hardware. It needs
    no calibration data. Returns the path to the quantized model.

    Note: the speedup is hardware dependent. On x86 (Cloud Run) int8 typically
    runs several times faster than fp32 eager PyTorch; on Apple Silicon the gap is
    smaller because PyTorch's ARM kernels are already highly optimized.
    """
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    onnx_path = Path(onnx_path)
    if quant_path is None:
        quant_path = onnx_path.with_name(onnx_path.stem + ".int8.onnx")
    quant_path = Path(quant_path)

    # Pre-process (shape inference + graph opt) so conv/matmul weights are proper
    # initializers; quantize_dynamic fails on some graphs without this.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        prepped = tmp.name
    # Full pre-process (symbolic + ONNX shape inference + graph opt); skipping
    # symbolic shape inference leaves some Conv weights as non-initializers.
    quant_pre_process(str(onnx_path), prepped)
    quantize_dynamic(prepped, str(quant_path), weight_type=QuantType.QInt8)
    Path(prepped).unlink(missing_ok=True)
    return quant_path


class _WithSigmoid(nn.Module):
    """Wrap a logits model so its ONNX output is probabilities."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def verify_parity(
    model: nn.Module,
    onnx_path: str | Path,
    image_size: int = 320,
    n: int = 4,
    atol: float = 1e-4,
) -> float:
    """Compare ORT vs PyTorch(+sigmoid) outputs; return the max abs difference.

    Raises AssertionError if they diverge by more than ``atol``.
    """
    import onnxruntime as ort

    wrapped = _WithSigmoid(model).eval()
    x = torch.randn(n, 3, image_size, image_size)
    with torch.no_grad():
        torch_out = wrapped(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["probs"], {"image": x.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    assert max_diff <= atol, f"ONNX parity failed: max|Δ|={max_diff:.2e} > {atol:.1e}"
    assert ort_out.shape == (n, NUM_CLASSES)
    return max_diff
