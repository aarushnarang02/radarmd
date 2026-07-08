"""Pick a per-class operating threshold for the multi-label classifier.

A single 0.5 cutoff is wrong here: the classes are imbalanced and, clinically,
missing a pneumothorax matters far more than a false alarm. So we tune one
threshold per class on the validation set, using a **sensitivity floor** for the
serious findings (guarantee recall >= floor, i.e. false-negative rate below
1-floor) and Youden's J for the rest.

Thresholds are chosen on validation probabilities and then applied unchanged to
the test set (never tune on test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve

from ..data.constants import CRITICAL_FINDINGS, PATHOLOGIES

# Default recall floor for critical findings: recall >= 0.92 => FNR < 8%, the
# "missed diagnoses under 8%" target.
DEFAULT_SENSITIVITY_FLOOR = 0.92


@dataclass
class OperatingPoint:
    threshold: float
    strategy: str
    achieved_sensitivity: float
    achieved_specificity: float
    support_pos: int


def _roc(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # roc_curve prepends a threshold of inf; clip so we never emit an
    # unreachable cutoff.
    thr = np.clip(thr, 0.0, 1.0)
    return fpr, tpr, thr


def threshold_sensitivity_floor(
    y_true: np.ndarray, y_score: np.ndarray, floor: float = DEFAULT_SENSITIVITY_FLOOR
) -> OperatingPoint:
    """Highest threshold that still achieves sensitivity >= ``floor``.

    Higher thresholds give better specificity but lower recall; we take the most
    specific operating point that still meets the recall floor.
    """
    fpr, tpr, thr = _roc(y_true, y_score)
    ok = tpr >= floor
    if not ok.any():
        # Cannot meet the floor at any cutoff; fall back to the most sensitive
        # (lowest) threshold so we err toward catching positives.
        idx = int(np.argmax(tpr))
    else:
        # roc_curve thresholds are descending; among those meeting the floor,
        # the first (largest threshold) has the smallest FPR = best specificity.
        idx = int(np.argmax(ok))
    return OperatingPoint(
        threshold=float(thr[idx]),
        strategy="sensitivity_floor",
        achieved_sensitivity=float(tpr[idx]),
        achieved_specificity=float(1.0 - fpr[idx]),
        support_pos=int(y_true.sum()),
    )


def threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> OperatingPoint:
    """Threshold maximizing Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thr = _roc(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return OperatingPoint(
        threshold=float(thr[idx]),
        strategy="youden",
        achieved_sensitivity=float(tpr[idx]),
        achieved_specificity=float(1.0 - fpr[idx]),
        support_pos=int(y_true.sum()),
    )


def select_operating_points(
    probs: np.ndarray,
    labels: np.ndarray,
    critical: tuple[str, ...] = CRITICAL_FINDINGS,
    floor: float = DEFAULT_SENSITIVITY_FLOOR,
    default_threshold: float = 0.5,
) -> dict[str, OperatingPoint]:
    """Choose an operating point per class from validation probs/labels.

    ``probs`` and ``labels`` are (N, 14) arrays in the canonical class order.
    Critical findings use the sensitivity floor; the rest use Youden's J. A class
    with no positive (or no negative) examples can't be tuned and falls back to
    ``default_threshold``.
    """
    if probs.shape != labels.shape:
        raise ValueError(f"probs {probs.shape} and labels {labels.shape} must match")

    points: dict[str, OperatingPoint] = {}
    critical_set = set(critical)
    for i, name in enumerate(PATHOLOGIES):
        y_true = labels[:, i].astype(int)
        y_score = probs[:, i]
        pos, neg = int(y_true.sum()), int((y_true == 0).sum())
        if pos == 0 or neg == 0:
            points[name] = OperatingPoint(default_threshold, "fallback", float("nan"), float("nan"), pos)
            continue
        if name in critical_set:
            points[name] = threshold_sensitivity_floor(y_true, y_score, floor)
        else:
            points[name] = threshold_youden(y_true, y_score)
    return points


def thresholds_vector(points: dict[str, OperatingPoint]) -> np.ndarray:
    """Pack per-class thresholds into a (14,) array in canonical order.

    float64 on purpose: a threshold from ``roc_curve`` equals an actual score,
    and rounding it to float32 can nudge it just above that score, flipping true
    positives to negatives at the decision boundary.
    """
    return np.array([points[name].threshold for name in PATHOLOGIES], dtype=np.float64)
