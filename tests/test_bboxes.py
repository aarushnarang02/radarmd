"""Bounding-box parsing, including the messy real-world CSV header."""

from radarmd.data.bboxes import load_bboxes, validate_bbox_labels
from radarmd.data.constants import LOCALIZED_PATHOLOGIES


def test_load_bboxes_normalizes_messy_header(bbox_csv):
    df = load_bboxes(bbox_csv)
    for col in ("image", "label", "x", "y", "w", "h", "x0", "y0", "x1", "y1"):
        assert col in df.columns


def test_normalized_coords_in_unit_range(bbox_csv):
    df = load_bboxes(bbox_csv)
    for col in ("x0", "y0", "x1", "y1"):
        assert (df[col] >= 0).all()
        assert (df[col] <= 1.0).all()
    assert (df["x1"] >= df["x0"]).all()
    assert (df["y1"] >= df["y0"]).all()


def test_labels_within_localized_set(bbox_csv):
    df = load_bboxes(bbox_csv)
    validate_bbox_labels(df)  # raises if any label is outside the 8
    assert set(df["label"]).issubset(set(LOCALIZED_PATHOLOGIES))
