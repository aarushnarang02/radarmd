"""Canonical label definitions for NIH ChestX-ray14.

The dataset annotates 14 thoracic pathologies plus an implicit "No Finding"
class. Labels arrive as a pipe-delimited string in ``Data_Entry_2017.csv``
(e.g. ``"Cardiomegaly|Effusion"``); we always encode them into a fixed-order
14-dim multi-hot vector so downstream code never depends on CSV ordering.
"""

from __future__ import annotations

# Fixed order. NEVER reorder: model checkpoints and ONNX outputs are indexed by
# this list. Append-only if the label set ever changes.
PATHOLOGIES: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

NUM_CLASSES: int = len(PATHOLOGIES)

# The string the dataset uses for a negative (healthy) study. It is not one of
# the 14 classes; a "No Finding" row maps to an all-zero multi-hot vector.
NO_FINDING = "No Finding"

# Index lookup for O(1) string -> position mapping.
LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(PATHOLOGIES)}

# The 8 pathologies that have hand-drawn ground-truth bounding boxes in
# ``BBox_List_2017.csv`` (used for Grad-CAM localization validation in stage 4).
LOCALIZED_PATHOLOGIES: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
)

# Findings where a missed diagnosis is most dangerous. Stage 5 enforces a
# sensitivity floor (low false-negative rate) on these when picking operating
# thresholds, rather than optimizing raw accuracy.
CRITICAL_FINDINGS: tuple[str, ...] = (
    "Pneumothorax",
    "Mass",
    "Nodule",
    "Effusion",
    "Consolidation",
    "Pneumonia",
    "Edema",
)
