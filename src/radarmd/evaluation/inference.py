"""Run a trained model over a dataloader to collect probabilities and labels."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module, dataloader: DataLoader, device: str | torch.device = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, labels)`` as (N, 14) numpy arrays.

    Applies sigmoid to logits so ``probs`` are calibrated to [0, 1]. Labels are
    the ground-truth multi-hot vectors from the dataset.
    """
    model = model.to(device).eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, labels in dataloader:
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)
