"""Aggregate Grad-CAM localization scoring against the 880 GT boxes.

Given a trained model and the bbox CSV, this computes, per pathology and overall:
  - pointing-game accuracy (peak inside GT box),
  - IoU@0.5 / @0.25 / @0.1 localization accuracy (weakly supervised localization
    on ChestX-ray14 is hard, so lower IoU thresholds are reported alongside 0.5).

Kept model-agnostic and side-effect free so it can be unit-tested with a stub
"explainer" that returns canned heatmaps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .localization import iou, localize_hit, pointing_game_hit


class Explainer(Protocol):
    """Anything that maps (image, class_name) -> normalized (H, W) heatmap."""

    def heatmap(self, image, class_name: str) -> np.ndarray: ...


@dataclass
class LocalizationResult:
    iou_thresholds: tuple[float, ...]
    pointing: dict[str, float] = field(default_factory=dict)
    iou_acc: dict[float, dict[str, float]] = field(default_factory=dict)
    n_boxes: dict[str, int] = field(default_factory=dict)

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for cls in sorted(self.n_boxes):
            row = {"pathology": cls, "n": self.n_boxes[cls], "pointing": self.pointing[cls]}
            for t in self.iou_thresholds:
                row[f"iou@{t}"] = self.iou_acc[t][cls]
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate_localization(
    explainer: Explainer,
    bboxes: pd.DataFrame,
    load_image,
    iou_thresholds: tuple[float, ...] = (0.1, 0.25, 0.5),
    cam_threshold: float = 0.5,
) -> LocalizationResult:
    """Score localization for every GT box.

    Parameters
    ----------
    explainer:      produces a heatmap for a given image and class.
    bboxes:         normalized GT boxes (from ``load_bboxes``), one row per box.
    load_image:     callable ``image_filename -> model-ready image tensor``.
    iou_thresholds: IoU cutoffs at which to report localization accuracy.
    """
    pointing_hits: dict[str, list[bool]] = defaultdict(list)
    iou_hits: dict[float, dict[str, list[bool]]] = {t: defaultdict(list) for t in iou_thresholds}

    # Cache heatmaps per (image, class) so repeated boxes don't recompute.
    cache: dict[tuple[str, str], np.ndarray] = {}

    for _, box in bboxes.iterrows():
        cls, img_name = box["label"], box["image"]
        key = (img_name, cls)
        if key not in cache:
            cache[key] = explainer.heatmap(load_image(img_name), cls)
        cam = cache[key]
        gt = (box["x0"], box["y0"], box["x1"], box["y1"])

        pointing_hits[cls].append(pointing_game_hit(cam, gt))
        for t in iou_thresholds:
            iou_hits[t][cls].append(localize_hit(cam, gt, iou_threshold=t, cam_threshold=cam_threshold))

    def _rate(d: dict[str, list[bool]]) -> dict[str, float]:
        return {k: float(np.mean(v)) if v else 0.0 for k, v in d.items()}

    result = LocalizationResult(iou_thresholds=iou_thresholds)
    result.pointing = _rate(pointing_hits)
    result.iou_acc = {t: _rate(iou_hits[t]) for t in iou_thresholds}
    result.n_boxes = {k: len(v) for k, v in pointing_hits.items()}
    return result


# Re-exported for callers that want the raw IoU without importing localization.
__all__ = ["Explainer", "LocalizationResult", "evaluate_localization", "iou"]
