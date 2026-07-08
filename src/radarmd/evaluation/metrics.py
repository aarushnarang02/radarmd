"""Per-class and aggregate metrics for the multi-label classifier.

Threshold-free ranking metrics (AUROC, average precision) plus threshold-applied
operating metrics (sensitivity, specificity, F1, false-negative rate) computed
at the per-class operating points chosen in :mod:`.thresholds`. Built on
scikit-learn so the numbers match what reviewers expect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from ..data.constants import CRITICAL_FINDINGS, PATHOLOGIES


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Undefined when only one class present in y_true.
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def per_class_report(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Build a per-class metrics table.

    Columns: pathology, critical flag, support, threshold, auroc, ap,
    sensitivity, specificity, f1, fnr (false-negative rate).
    """
    if not (probs.shape == labels.shape and probs.shape[1] == len(PATHOLOGIES)):
        raise ValueError("probs/labels must be (N, 14) and equal shape")

    rows = []
    critical_set = set(CRITICAL_FINDINGS)
    for i, name in enumerate(PATHOLOGIES):
        y_true = labels[:, i].astype(int)
        y_score = probs[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)

        tn, fp, fn, tp = _confusion(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (2 * prec * sens / (prec + sens)) if prec and sens and not np.isnan(prec) and not np.isnan(sens) else 0.0
        fnr = fn / (tp + fn) if (tp + fn) else float("nan")

        rows.append(
            {
                "pathology": name,
                "critical": name in critical_set,
                "support": int(y_true.sum()),
                "threshold": float(thresholds[i]),
                "auroc": _safe_auroc(y_true, y_score),
                "ap": _safe_ap(y_true, y_score),
                "sensitivity": sens,
                "specificity": spec,
                "f1": f1,
                "fnr": fnr,
            }
        )
    return pd.DataFrame(rows)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def mean_auroc(report: pd.DataFrame) -> float:
    """Macro mean AUROC across classes (ignoring undefined ones)."""
    return float(np.nanmean(report["auroc"].to_numpy()))


def critical_fnr_violations(
    report: pd.DataFrame, max_fnr: float
) -> pd.DataFrame:
    """Rows for critical findings whose false-negative rate exceeds ``max_fnr``."""
    crit = report[report["critical"]]
    return crit[crit["fnr"] > max_fnr + 1e-9]
