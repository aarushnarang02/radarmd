"""Parse NIH ChestX-ray14 metadata into a tidy, model-ready DataFrame.

``Data_Entry_2017.csv`` has one row per image with a pipe-delimited
``Finding Labels`` column. This module loads that file, encodes the labels into
a 14-dim multi-hot matrix, and normalizes the handful of columns we care about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABEL_TO_INDEX, NO_FINDING, NUM_CLASSES, PATHOLOGIES


def encode_labels(finding_labels: str) -> np.ndarray:
    """Convert a pipe-delimited label string to a float32 multi-hot vector.

    ``"Cardiomegaly|Effusion"`` -> vector with 1.0 at those two indices.
    ``"No Finding"`` -> all zeros. Unknown tokens are ignored (defensive; the
    dataset is clean but we don't want a typo to crash a training run).
    """
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    if not finding_labels or finding_labels == NO_FINDING:
        return vec
    for token in finding_labels.split("|"):
        idx = LABEL_TO_INDEX.get(token.strip())
        if idx is not None:
            vec[idx] = 1.0
    return vec


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load ``Data_Entry_2017.csv`` into a normalized DataFrame.

    Returns one row per image with columns:
      - ``image``:      image filename (e.g. ``00000001_000.png``)
      - ``patient_id``: int patient identifier (for leak-free splitting)
      - ``follow_up``:  follow-up number for the study
      - ``age``, ``gender``, ``view``: demographics / projection
      - ``labels``:     python list of positive pathology names
      - ``<pathology>``: one float32 column per class (the multi-hot encoding)
    """
    df = pd.read_csv(csv_path)

    # The CSV column names have shifted slightly across dataset mirrors; map the
    # ones we rely on defensively.
    rename = {
        "Image Index": "image",
        "Finding Labels": "finding_labels",
        "Patient ID": "patient_id",
        "Follow-up #": "follow_up",
        "Patient Age": "age",
        "Patient Gender": "gender",
        "View Position": "view",
    }
    present = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=present)

    missing = {"image", "finding_labels", "patient_id"} - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {sorted(missing)}")

    df["patient_id"] = df["patient_id"].astype(int)

    encoded = np.stack([encode_labels(s) for s in df["finding_labels"]])
    label_frame = pd.DataFrame(encoded, columns=list(PATHOLOGIES), index=df.index)

    df["labels"] = [
        [PATHOLOGIES[i] for i in np.nonzero(row)[0]] for row in encoded
    ]

    keep = [c for c in ("image", "patient_id", "follow_up", "age", "gender", "view") if c in df.columns]
    return pd.concat([df[keep], df[["labels"]], label_frame], axis=1)


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the (N, 14) float32 multi-hot matrix from a metadata frame."""
    return df[list(PATHOLOGIES)].to_numpy(dtype=np.float32)


def positive_counts(df: pd.DataFrame) -> pd.Series:
    """Per-class positive counts, useful for pos_weight and EDA."""
    return df[list(PATHOLOGIES)].sum().astype(int)
