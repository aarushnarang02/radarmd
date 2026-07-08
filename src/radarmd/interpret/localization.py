"""Localization metrics for Grad-CAM heatmaps vs ground-truth boxes.

Stage 4 validates that the model looks in the right place. NIH provides 880
hand-drawn boxes over 8 pathologies; for each we turn the class heatmap into a
predicted region and score it two ways:

  - **Pointing game**: does the heatmap's peak fall inside the GT box? A lenient,
    threshold-free check of "is the hottest point on target".
  - **IoU@k**: threshold the heatmap, take the bounding box of the largest hot
    region, and compare to the GT box by intersection-over-union. A hit is
    IoU >= k (k=0.5 by convention, though 0.1/0.25 are common for weakly
    supervised localization on this dataset).

All coordinates here are normalized to [0, 1] so heatmap resolution doesn't
matter; GT boxes come pre-normalized from :mod:`radarmd.data.bboxes`.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]  # (x0, y0, x1, y1) normalized


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two normalized (x0, y0, x1, y1) boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def peak_point(cam: np.ndarray) -> tuple[float, float]:
    """Normalized (x, y) of the heatmap's maximum."""
    iy, ix = np.unravel_index(int(np.argmax(cam)), cam.shape)
    h, w = cam.shape
    return (ix + 0.5) / w, (iy + 0.5) / h


def pointing_game_hit(cam: np.ndarray, gt_box: Box) -> bool:
    """True if the heatmap peak lies inside the ground-truth box."""
    x, y = peak_point(cam)
    x0, y0, x1, y1 = gt_box
    return x0 <= x <= x1 and y0 <= y <= y1


def cam_to_box(cam: np.ndarray, threshold: float = 0.5) -> Box | None:
    """Bounding box of the largest connected hot region of a heatmap.

    The heatmap is min-max normalized, thresholded at ``threshold`` (fraction of
    its own range), and the connected component with the greatest total
    activation is boxed. Returns None if nothing exceeds the threshold.
    """
    cam = np.asarray(cam, dtype=np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    if hi <= lo:
        return None
    norm = (cam - lo) / (hi - lo)
    mask = norm >= threshold
    if not mask.any():
        return None

    labels = _connected_components(mask)
    best_label, best_score = 0, -1.0
    for lbl in range(1, labels.max() + 1):
        score = float(norm[labels == lbl].sum())
        if score > best_score:
            best_label, best_score = lbl, score

    ys, xs = np.where(labels == best_label)
    h, w = cam.shape
    # +1 on the far edge so a single-pixel region has non-zero area.
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def localize_hit(
    cam: np.ndarray, gt_box: Box, iou_threshold: float = 0.5, cam_threshold: float = 0.5
) -> bool:
    """True if the CAM-derived box overlaps the GT box by >= ``iou_threshold``."""
    pred = cam_to_box(cam, threshold=cam_threshold)
    if pred is None:
        return False
    return iou(pred, gt_box) >= iou_threshold


def _connected_components(mask: np.ndarray) -> np.ndarray:
    """4-connectivity labeling via iterative flood fill (no scipy dependency)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            current += 1
            stack = [(sy, sx)]
            labels[sy, sx] = current
            while stack:
                y, x = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels
