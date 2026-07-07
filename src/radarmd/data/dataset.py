"""PyTorch Dataset for ChestX-ray14 images.

Images live in a flat directory (the sample dataset) or across the NIH
``images_001..012`` folders (full dataset). We build a filename -> path index
once so the Dataset works regardless of directory layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import PATHOLOGIES
from .transforms import build_transforms

# Image extensions we accept. The raw dataset ships PNGs; our resized Colab
# shards extract to JPEGs. We key the index by filename *stem* so a metadata row
# referencing ``00000001_000.png`` matches whichever format is on disk.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def image_key(name: str) -> str:
    """Normalize an image filename to its extension-less lookup key."""
    return Path(name).stem


def index_images(root: str | Path) -> dict[str, Path]:
    """Map every image *stem* under ``root`` to its full path.

    Handles the flat sample layout, the full dataset's nested
    ``images_XXX/images/`` folders, and extracted JPEG shards. Later duplicates
    do not overwrite earlier ones (filenames are unique across NIH folders).
    """
    root = Path(root)
    index: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in _IMAGE_EXTS:
            index.setdefault(p.stem, p)
    return index


class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset returning ``(image_tensor, label_vec)``."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_index: dict[str, Path],
        image_size: int = 224,
        train: bool = True,
    ) -> None:
        # Keep only rows whose image file we actually have on disk (matched by
        # stem). This lets a split computed on full metadata run against the
        # sample image set, or against resized JPEG shards.
        keys = df["image"].map(image_key)
        available = keys.isin(image_index.keys())
        self.df = df[available].reset_index(drop=True)
        self.image_index = image_index
        self.transform = build_transforms(image_size=image_size, train=train)
        self._labels = self.df[list(PATHOLOGIES)].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.image_index[image_key(row["image"])]
        # Some NIH PNGs are RGBA/palette; force single-channel grayscale.
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        tensor = self.transform(img)
        # .copy() so the collated batch owns writable memory (the label matrix
        # is a read-only view otherwise).
        label = torch.from_numpy(self._labels[idx].copy())
        return tensor, label
