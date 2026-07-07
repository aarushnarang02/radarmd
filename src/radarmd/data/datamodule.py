"""Lightning DataModule tying metadata, splits, and datasets together."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .constants import PATHOLOGIES
from .dataset import ChestXrayDataset, index_images
from .labels import load_metadata
from .splits import official_split, random_patient_split


class ChestXrayDataModule(LightningDataModule):
    """Loads NIH ChestX-ray14 with leak-free patient-level splits.

    Parameters
    ----------
    data_dir:      directory containing the image files (searched recursively).
    metadata_csv:  path to ``Data_Entry_2017.csv``.
    split:         "random" (CheXNet-style) or "official" (NIH lists).
    official_lists: (train_val_list, test_list) paths, required when
                   ``split == "official"``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        metadata_csv: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        split: str = "random",
        official_lists: tuple[str | Path, str | Path] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["official_lists"])
        self.data_dir = Path(data_dir)
        self.metadata_csv = Path(metadata_csv)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.official_lists = official_lists
        self.seed = seed
        self._image_index: dict[str, Path] = {}
        self._splits: dict[str, pd.DataFrame] = {}
        self.datasets: dict[str, ChestXrayDataset] = {}

    def prepare_data(self) -> None:
        # No downloads here (handled by scripts/prepare_data.py). Just validate.
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

    def setup(self, stage: str | None = None) -> None:
        df = load_metadata(self.metadata_csv)
        if self.split == "random":
            self._splits = random_patient_split(df, seed=self.seed)
        elif self.split == "official":
            if self.official_lists is None:
                raise ValueError("official split requires official_lists=(train_val, test)")
            self._splits = official_split(df, *self.official_lists, seed=self.seed)
        else:
            raise ValueError(f"Unknown split: {self.split!r}")

        self._image_index = index_images(self.data_dir)
        for name, part in self._splits.items():
            self.datasets[name] = ChestXrayDataset(
                part,
                self._image_index,
                image_size=self.image_size,
                train=(name == "train"),
            )

    def pos_weight(self) -> torch.Tensor:
        """BCE ``pos_weight`` = (#neg / #pos) per class, from the train split.

        Compensates for heavy class imbalance (most findings are rare). Clamped
        to avoid exploding weights on very rare classes like Hernia.
        """
        train = self._splits["train"]
        pos = train[list(PATHOLOGIES)].to_numpy(dtype=np.float32).sum(axis=0)
        n = len(train)
        neg = n - pos
        weight = np.divide(neg, np.maximum(pos, 1.0))
        weight = np.clip(weight, 1.0, 50.0)
        return torch.tensor(weight, dtype=torch.float32)

    def _loader(self, name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[name],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)
