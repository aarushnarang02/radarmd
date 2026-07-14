"""Torch-free preprocessing for the serving path.

The training/eval pipeline uses MONAI (which needs torch). Pulling torch+MONAI
into the serving container would triple its size and slow Cloud Run cold starts,
so serving re-implements the *eval* transform with PIL + numpy only:

    grayscale -> scale to [0,1] -> resize -> repeat to 3ch -> ImageNet normalize

Resampling libraries differ slightly (PIL's antialiased bilinear vs MONAI's
``area`` interpolation), so outputs are not bit-identical to the MONAI pipeline;
``tests/test_preprocess_parity.py`` pins the difference to a small tolerance at
both the pixel and the model-output level.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..data.constants import IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    """PIL image -> normalized float32 array of shape (1, 3, H, W)."""
    gray = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)

    # Scale to [0,1] by the image's own range, matching MONAI's ScaleIntensity
    # (min-max normalization, not a fixed /255).
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    # (H, W) -> (3, H, W), then ImageNet per-channel normalization.
    chw = np.repeat(arr[None, :, :], 3, axis=0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    chw = (chw - mean) / std

    return chw[None, ...].astype(np.float32)
