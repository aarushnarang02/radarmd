"""Blend a Grad-CAM heatmap onto the original X-ray for display.

Kept dependency-light (numpy + PIL only) so the serving image stays small: no
matplotlib/opencv at request time. Produces an RGB image with a red-hot heatmap
overlaid at the requested opacity.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _apply_colormap(cam: np.ndarray) -> np.ndarray:
    """Map a [0,1] heatmap to an RGB 'hot'-style colormap, returning uint8."""
    cam = np.clip(cam, 0.0, 1.0)
    # Simple, monotonic hot colormap: black -> red -> yellow -> white.
    r = np.clip(cam * 3.0, 0, 1)
    g = np.clip(cam * 3.0 - 1.0, 0, 1)
    b = np.clip(cam * 3.0 - 2.0, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Overlay a normalized heatmap ``cam`` on ``image``.

    ``cam`` is any 2D array; it is min-max normalized and resized to the image.
    Returns an RGB PIL image. ``alpha`` is the heatmap opacity.
    """
    base = image.convert("RGB")
    w, h = base.size

    cam = np.asarray(cam, dtype=np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    cam = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)

    heat = Image.fromarray(_apply_colormap(cam)).resize((w, h), Image.BILINEAR)
    blended = Image.blend(base, heat, alpha=alpha)
    return blended
