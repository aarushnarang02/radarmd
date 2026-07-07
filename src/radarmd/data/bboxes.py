"""Parse ``BBox_List_2017.csv``, the 880 hand-drawn ground-truth boxes.

These boxes cover 8 of the 14 pathologies and are used in stage 4 to validate
Grad-CAM localization (IoU@0.5 and the pointing game). Each row is one box on
one image, in original pixel coordinates of the 1024x1024 source image.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import LOCALIZED_PATHOLOGIES

# Native resolution of the source PNGs the boxes were drawn on.
SOURCE_SIZE = 1024


def load_bboxes(csv_path: str | Path) -> pd.DataFrame:
    """Load the bounding-box CSV into a normalized DataFrame.

    Returns columns: ``image``, ``label``, ``x``, ``y``, ``w``, ``h``
    (top-left corner + width/height, in source pixels), plus normalized
    ``x0,y0,x1,y1`` in [0,1] for resolution-independent comparison.
    """
    df = pd.read_csv(csv_path)

    # Column headers in this file are notoriously messy (trailing spaces, stray
    # unnamed columns). Normalize by stripping and matching known prefixes.
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]

    # The real header is literally ``Image Index,Finding Label,Bbox [x,y,w,h]``,
    # so coordinate columns carry stray bracket characters. Strip everything but
    # letters before matching.
    def _clean(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalpha())

    colmap = {}
    for c in df.columns:
        cl = _clean(c)
        if cl.startswith("image"):
            colmap[c] = "image"
        elif "finding" in cl or cl == "label":
            colmap[c] = "label"
        elif cl in ("bboxx", "x"):
            colmap[c] = "x"
        elif cl == "y":
            colmap[c] = "y"
        elif cl in ("w", "width"):
            colmap[c] = "w"
        elif cl in ("h", "height"):
            colmap[c] = "h"
    df = df.rename(columns=colmap)

    required = {"image", "label", "x", "y", "w", "h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BBox CSV missing columns after normalization: {sorted(missing)}")

    df = df[["image", "label", "x", "y", "w", "h"]].copy()
    for c in ("x", "y", "w", "h"):
        df[c] = df[c].astype(float)

    df["x0"] = df["x"] / SOURCE_SIZE
    df["y0"] = df["y"] / SOURCE_SIZE
    df["x1"] = (df["x"] + df["w"]) / SOURCE_SIZE
    df["y1"] = (df["y"] + df["h"]) / SOURCE_SIZE

    return df.reset_index(drop=True)


def boxes_for_image(bboxes: pd.DataFrame, image: str) -> pd.DataFrame:
    """All ground-truth boxes for one image filename."""
    return bboxes[bboxes["image"] == image].reset_index(drop=True)


def validate_bbox_labels(bboxes: pd.DataFrame) -> None:
    """Raise if any box carries a label outside the 8 localized pathologies."""
    unknown = set(bboxes["label"].unique()) - set(LOCALIZED_PATHOLOGIES)
    if unknown:
        raise ValueError(f"Unexpected bbox labels: {sorted(unknown)}")
