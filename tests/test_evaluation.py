"""Threshold selection, metrics, and the sensitivity-floor guarantee."""

import numpy as np
import pytest

from radarmd.data.constants import CRITICAL_FINDINGS, NUM_CLASSES, PATHOLOGIES
from radarmd.evaluation.metrics import (
    critical_fnr_violations,
    mean_auroc,
    per_class_report,
)
from radarmd.evaluation.report import format_report, meets_sensitivity_target
from radarmd.evaluation.thresholds import (
    select_operating_points,
    threshold_sensitivity_floor,
    threshold_youden,
    thresholds_vector,
)

rng = np.random.default_rng(0)


def _separable(n=400, sep=2.0):
    """A single class where positives score higher than negatives."""
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.normal(0, 1, n) + sep * y
    # squash to (0,1)
    score = 1 / (1 + np.exp(-score))
    return y, score


def test_sensitivity_floor_is_met():
    y, s = _separable()
    op = threshold_sensitivity_floor(y, s, floor=0.92)
    # Recompute recall at the chosen threshold to confirm the guarantee.
    pred = (s >= op.threshold).astype(int)
    recall = pred[y == 1].sum() / max((y == 1).sum(), 1)
    assert recall >= 0.92 - 1e-9


def test_sensitivity_floor_prefers_higher_specificity_than_trivial():
    y, s = _separable(sep=3.0)
    op = threshold_sensitivity_floor(y, s, floor=0.90)
    # A separable problem should hit the floor without predicting everything
    # positive, so specificity is meaningfully above zero.
    assert op.achieved_specificity > 0.2
    assert op.threshold > 0.0


def test_youden_threshold_reasonable():
    y, s = _separable(sep=2.5)
    op = threshold_youden(y, s)
    assert 0.0 <= op.threshold <= 1.0
    assert op.achieved_sensitivity > 0.5


def test_select_operating_points_strategy_by_criticality():
    n = 500
    probs = rng.random((n, NUM_CLASSES))
    labels = (rng.random((n, NUM_CLASSES)) < 0.25).astype(float)
    # make each column separable-ish so ROC is defined
    probs = np.clip(probs * 0.5 + labels * 0.5, 0, 1)
    points = select_operating_points(probs, labels, floor=0.9)
    assert set(points) == set(PATHOLOGIES)
    for name, op in points.items():
        if name in set(CRITICAL_FINDINGS):
            assert op.strategy in ("sensitivity_floor", "fallback")
        else:
            assert op.strategy in ("youden", "fallback")


def test_fallback_when_class_all_negative():
    n = 100
    probs = rng.random((n, NUM_CLASSES))
    labels = np.zeros((n, NUM_CLASSES), dtype=float)  # no positives anywhere
    points = select_operating_points(probs, labels)
    assert all(op.strategy == "fallback" for op in points.values())
    assert all(op.threshold == 0.5 for op in points.values())


def test_thresholds_vector_order_and_shape():
    n = 300
    probs = rng.random((n, NUM_CLASSES))
    labels = (rng.random((n, NUM_CLASSES)) < 0.3).astype(float)
    points = select_operating_points(probs, labels)
    vec = thresholds_vector(points)
    assert vec.shape == (NUM_CLASSES,)
    assert vec[0] == points[PATHOLOGIES[0]].threshold


def test_per_class_report_shape_and_columns():
    n = 300
    probs = rng.random((n, NUM_CLASSES))
    labels = (rng.random((n, NUM_CLASSES)) < 0.3).astype(float)
    thr = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    rep = per_class_report(probs, labels, thr)
    assert len(rep) == NUM_CLASSES
    for col in ("auroc", "ap", "sensitivity", "specificity", "f1", "fnr", "critical"):
        assert col in rep.columns
    assert rep["critical"].sum() == len(CRITICAL_FINDINGS)


def test_perfect_scores_give_auroc_one_and_no_violations():
    n = 200
    labels = (rng.random((n, NUM_CLASSES)) < 0.4).astype(float)
    probs = labels * 0.99 + 0.005  # essentially perfect ranking
    # tune on the same data (fine for this determinism test)
    points = select_operating_points(probs, labels, floor=0.92)
    thr = thresholds_vector(points)
    rep = per_class_report(probs, labels, thr)
    assert mean_auroc(rep) > 0.99
    assert meets_sensitivity_target(rep, sensitivity_floor=0.92)
    assert critical_fnr_violations(rep, 0.08).empty


def test_format_report_contains_verdict():
    n = 200
    labels = (rng.random((n, NUM_CLASSES)) < 0.4).astype(float)
    probs = labels * 0.99 + 0.005
    thr = thresholds_vector(select_operating_points(probs, labels))
    rep = per_class_report(probs, labels, thr)
    text = format_report(rep, split_name="test")
    assert "Mean AUROC" in text
    assert ("PASS" in text) or ("FAIL" in text)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        per_class_report(np.zeros((5, NUM_CLASSES)), np.zeros((5, 3)), np.zeros(NUM_CLASSES))
