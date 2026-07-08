"""Localization metrics: IoU, CAM-to-box, pointing game, aggregation."""

import numpy as np
import pandas as pd

from radarmd.interpret.evaluate import evaluate_localization
from radarmd.interpret.localization import (
    cam_to_box,
    iou,
    localize_hit,
    peak_point,
    pointing_game_hit,
)


def test_iou_identical_boxes():
    b = (0.1, 0.1, 0.5, 0.5)
    assert iou(b, b) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.9, 0.9)) == 0.0


def test_iou_half_overlap():
    # Two unit-ish boxes overlapping in half their area.
    a = (0.0, 0.0, 0.4, 0.4)
    b = (0.2, 0.0, 0.6, 0.4)
    # intersection 0.2x0.4=0.08; union = 0.16+0.16-0.08=0.24 -> 1/3
    assert abs(iou(a, b) - (0.08 / 0.24)) < 1e-6


def test_peak_point_center_of_hotspot():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[7, 3] = 1.0
    x, y = peak_point(cam)
    assert abs(x - 0.35) < 1e-6  # (3 + 0.5) / 10
    assert abs(y - 0.75) < 1e-6  # (7 + 0.5) / 10


def test_pointing_game_hit_and_miss():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[2, 8] = 1.0  # peak at x=0.85, y=0.25
    assert pointing_game_hit(cam, (0.7, 0.1, 1.0, 0.4))
    assert not pointing_game_hit(cam, (0.0, 0.0, 0.3, 0.3))


def test_cam_to_box_wraps_hot_region():
    cam = np.zeros((20, 20), dtype=np.float32)
    cam[5:10, 4:8] = 1.0  # a clear rectangular blob
    box = cam_to_box(cam, threshold=0.5)
    assert box is not None
    x0, y0, x1, y1 = box
    assert abs(x0 - 4 / 20) < 1e-6 and abs(x1 - 8 / 20) < 1e-6
    assert abs(y0 - 5 / 20) < 1e-6 and abs(y1 - 10 / 20) < 1e-6


def test_cam_to_box_none_when_flat():
    assert cam_to_box(np.zeros((8, 8), dtype=np.float32)) is None


def test_cam_to_box_picks_largest_component():
    cam = np.zeros((20, 20), dtype=np.float32)
    cam[0:2, 0:2] = 0.9  # small hot spot
    cam[10:18, 10:18] = 1.0  # large hot spot -> should win
    box = cam_to_box(cam, threshold=0.5)
    x0, y0, x1, y1 = box
    assert x0 >= 10 / 20 and y0 >= 10 / 20


def test_localize_hit_true_when_boxes_align():
    cam = np.zeros((20, 20), dtype=np.float32)
    cam[5:15, 5:15] = 1.0
    gt = (5 / 20, 5 / 20, 15 / 20, 15 / 20)
    assert localize_hit(cam, gt, iou_threshold=0.5)


def test_localize_hit_false_when_off_target():
    cam = np.zeros((20, 20), dtype=np.float32)
    cam[0:4, 0:4] = 1.0
    gt = (0.7, 0.7, 0.95, 0.95)
    assert not localize_hit(cam, gt, iou_threshold=0.5)


class _StubExplainer:
    """Returns a heatmap peaked at the center of each box's class for a hit."""

    def __init__(self, boxes: pd.DataFrame):
        self.boxes = boxes

    def heatmap(self, image, class_name):
        # image is (cx, cy) of the intended hot spot in [0,1].
        cx, cy = image
        cam = np.zeros((20, 20), dtype=np.float32)
        gy, gx = int(cy * 20), int(cx * 20)
        cam[max(0, gy - 3): gy + 3, max(0, gx - 3): gx + 3] = 1.0
        return cam


def test_evaluate_localization_perfect_and_reports_per_class():
    # One box per class, and the stub always peaks at the box center -> all hits.
    rows = [
        {"image": "a.png", "label": "Mass", "x0": 0.3, "y0": 0.3, "x1": 0.6, "y1": 0.6},
        {"image": "b.png", "label": "Effusion", "x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.4},
    ]
    boxes = pd.DataFrame(rows)

    def load_image(name):
        row = boxes[boxes["image"] == name].iloc[0]
        return ((row["x0"] + row["x1"]) / 2, (row["y0"] + row["y1"]) / 2)

    res = evaluate_localization(_StubExplainer(boxes), boxes, load_image, iou_thresholds=(0.25, 0.5))
    assert res.pointing["Mass"] == 1.0
    assert res.pointing["Effusion"] == 1.0
    assert res.n_boxes == {"Mass": 1, "Effusion": 1}
    frame = res.summary_frame()
    assert set(frame["pathology"]) == {"Mass", "Effusion"}
    assert "iou@0.5" in frame.columns
