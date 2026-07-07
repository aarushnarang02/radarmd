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


def index_images(root: str | Path) -> dict[str, Path]:
    """Map every ``*.png`` filename under ``root`` to its full path.

    Handles both the flat sample layout and the full dataset's nested
    ``images_XXX/images/`` folders. Later duplicates do not overwrite earlier
    ones (filenames are unique across NIH folders anyway).
    """
    root = Path(root)
    index: dict[str, Path] = {}
    for p in root.rglob("*.png"):
        index.setdefault(p.name, p)
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
        # Keep only rows whose image file we actually have on disk. This lets a
        # split computed on full metadata run against the sample image set.
        available = df["image"].isin(image_index.keys())
        self.df = df[available].reset_index(drop=True)
        self.image_index = image_index
        self.transform = build_transforms(image_size=image_size, train=train)
        self._labels = self.df[list(PATHOLOGIES)].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.image_index[row["image"]]
        # Some NIH PNGs are RGBA/palette; force single-channel grayscale.
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        tensor = self.transform(img)
        # .copy() so the collated batch owns writable memory (the label matrix
        # is a read-only view otherwise).
        label = torch.from_numpy(self._labels[idx].copy())
        return tensor, label
