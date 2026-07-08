"""ONNX-Runtime predictor: preprocess an image, run the model, format results.

This is the inference core shared by the FastAPI endpoint and the Gradio UI. It
loads the exported ONNX graph once and, optionally, per-class operating
thresholds so it can mark which findings cross their decision boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.constants import CRITICAL_FINDINGS, NUM_CLASSES, PATHOLOGIES
from ..data.transforms import build_transforms


@dataclass
class Finding:
    pathology: str
    probability: float
    flagged: bool  # probability >= its operating threshold
    critical: bool


class OnnxPredictor:
    """Run chest X-ray inference from an exported ONNX model."""

    def __init__(
        self,
        onnx_path: str | Path,
        image_size: int = 320,
        thresholds: dict[str, float] | str | Path | None = None,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.image_size = image_size
        self.transform = build_transforms(image_size=image_size, train=False)
        self.thresholds = self._load_thresholds(thresholds)

    @staticmethod
    def _load_thresholds(thresholds) -> np.ndarray:
        vec = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
        if thresholds is None:
            return vec
        if isinstance(thresholds, (str, Path)):
            with open(thresholds) as fh:
                thresholds = json.load(fh)
        for i, name in enumerate(PATHOLOGIES):
            if name in thresholds:
                vec[i] = float(thresholds[name])
        return vec

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        tensor = self.transform(arr)  # (3, H, W)
        return tensor.numpy()[None, ...]  # add batch dim

    def predict_probs(self, image: Image.Image) -> np.ndarray:
        """Return the (14,) probability vector for one PIL image."""
        x = self._preprocess(image)
        probs = self.session.run(["probs"], {"image": x})[0]
        return probs[0].astype(np.float32)

    def predict(self, image: Image.Image) -> list[Finding]:
        """Return findings sorted by probability (highest first)."""
        probs = self.predict_probs(image)
        critical_set = set(CRITICAL_FINDINGS)
        findings = [
            Finding(
                pathology=name,
                probability=float(probs[i]),
                flagged=bool(probs[i] >= self.thresholds[i]),
                critical=name in critical_set,
            )
            for i, name in enumerate(PATHOLOGIES)
        ]
        findings.sort(key=lambda f: f.probability, reverse=True)
        return findings
