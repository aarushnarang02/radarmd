"""Grad-CAM heatmaps and localization metrics vs ground-truth boxes."""

from .localization import (
    Box,
    cam_to_box,
    iou,
    localize_hit,
    peak_point,
    pointing_game_hit,
)

__all__ = [
    "Box",
    "iou",
    "peak_point",
    "pointing_game_hit",
    "cam_to_box",
    "localize_hit",
]
