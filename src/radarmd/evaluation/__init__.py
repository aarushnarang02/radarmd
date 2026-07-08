"""Operating-threshold selection, per-class metrics, and report generation."""

from .metrics import critical_fnr_violations, mean_auroc, per_class_report
from .report import format_report, meets_sensitivity_target
from .thresholds import (
    DEFAULT_SENSITIVITY_FLOOR,
    OperatingPoint,
    select_operating_points,
    thresholds_vector,
)

__all__ = [
    "DEFAULT_SENSITIVITY_FLOOR",
    "OperatingPoint",
    "select_operating_points",
    "thresholds_vector",
    "per_class_report",
    "mean_auroc",
    "critical_fnr_violations",
    "format_report",
    "meets_sensitivity_target",
]
