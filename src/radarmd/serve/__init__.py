"""Serving: ONNX export, ONNX-Runtime predictor, FastAPI app, Gradio UI.

Only torch-free modules are imported eagerly; the export helpers (which need
torch) load lazily so the slim serving container never imports torch.
"""

from typing import Any

from .overlay import overlay_heatmap
from .predictor import Finding, OnnxPredictor
from .preprocess import preprocess_image

__all__ = [
    "export_onnx",
    "verify_parity",
    "quantize_onnx",
    "OnnxPredictor",
    "Finding",
    "overlay_heatmap",
    "preprocess_image",
]

_EXPORT_HELPERS = {"export_onnx", "verify_parity", "quantize_onnx"}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_HELPERS:  # lazy: these pull in torch
        from . import onnx_export

        return getattr(onnx_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
