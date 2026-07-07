"""Render a labeled grid from one training batch to prove the pipeline works.

Loads real metadata + images through the DataModule, pulls one batch, un-does
ImageNet normalization for display, and writes a PNG grid with the positive
pathology labels under each image.

Usage:
    uv run python scripts/inspect_batch.py \
        --data-dir data --metadata data/Data_Entry_2017.csv --out outputs/batch.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from radarmd.data.constants import PATHOLOGIES
from radarmd.data.datamodule import ChestXrayDataModule
from radarmd.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (img * std + mean).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def labels_for(vec: np.ndarray) -> str:
    names = [PATHOLOGIES[i] for i in np.nonzero(vec)[0]]
    return "\n".join(names) if names else "No Finding"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    ap.add_argument("--out", default="outputs/batch.png")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--n", type=int, default=12, help="images to show")
    args = ap.parse_args()

    dm = ChestXrayDataModule(
        data_dir=args.data_dir,
        metadata_csv=args.metadata,
        image_size=args.image_size,
        batch_size=args.n,
        num_workers=0,
        split="random",
    )
    dm.prepare_data()
    dm.setup()
    print(
        f"Split sizes -> train: {len(dm.datasets['train'])}, "
        f"val: {len(dm.datasets['val'])}, test: {len(dm.datasets['test'])}"
    )
    print("pos_weight per class:")
    for name, w in zip(PATHOLOGIES, dm.pos_weight().tolist(), strict=True):
        print(f"  {name:20s} {w:6.2f}")

    imgs, labels = next(iter(dm.train_dataloader()))
    n = min(args.n, imgs.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(denormalize(imgs[i]))
            ax.set_title(labels_for(labels[i].numpy()), fontsize=8)
        ax.axis("off")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
