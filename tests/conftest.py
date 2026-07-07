"""Shared fixtures: synthetic ChestX-ray14-shaped metadata and images.

Tests must run in CI without the 42GB dataset, so we synthesize small CSVs and
PNGs that match the real schema exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from radarmd.data.constants import LOCALIZED_PATHOLOGIES, PATHOLOGIES

rng = np.random.default_rng(0)


def _make_finding(labels: list[str]) -> str:
    return "|".join(labels) if labels else "No Finding"


@pytest.fixture
def metadata_df() -> pd.DataFrame:
    """~200 images across 40 patients, multi-label, schema-identical to NIH."""
    rows = []
    n_patients = 40
    for pid in range(1, n_patients + 1):
        n_studies = rng.integers(1, 8)  # patients contribute multiple studies
        for follow in range(n_studies):
            k = rng.integers(0, 3)  # 0-2 findings per image
            labels = list(rng.choice(PATHOLOGIES, size=k, replace=False)) if k else []
            rows.append(
                {
                    "Image Index": f"{pid:08d}_{follow:03d}.png",
                    "Finding Labels": _make_finding(labels),
                    "Patient ID": pid,
                    "Follow-up #": follow,
                    "Patient Age": int(rng.integers(20, 90)),
                    "Patient Gender": rng.choice(["M", "F"]),
                    "View Position": rng.choice(["PA", "AP"]),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def metadata_csv(tmp_path, metadata_df) -> str:
    path = tmp_path / "Data_Entry_2017.csv"
    metadata_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def image_dir(tmp_path, metadata_df) -> str:
    """Write a tiny grayscale PNG for every image in the metadata."""
    d = tmp_path / "images"
    d.mkdir()
    for name in metadata_df["Image Index"]:
        arr = (rng.random((64, 64)) * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(d / name)
    return str(d)


@pytest.fixture
def bbox_csv(tmp_path, metadata_df) -> str:
    """A handful of ground-truth boxes with the messy real-world header."""
    imgs = metadata_df["Image Index"].tolist()[:10]
    rows = []
    for img in imgs:
        rows.append(
            {
                "Image Index": img,
                "Finding Label": rng.choice(LOCALIZED_PATHOLOGIES),
                "Bbox [x": float(rng.integers(0, 512)),
                "y": float(rng.integers(0, 512)),
                "w": float(rng.integers(50, 300)),
                "h]": float(rng.integers(50, 300)),
                "Unnamed: 6": np.nan,
            }
        )
    path = tmp_path / "BBox_List_2017.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)
