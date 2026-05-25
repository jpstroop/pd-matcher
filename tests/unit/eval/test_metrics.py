"""Tests for :mod:`pd_matcher.eval.metrics`."""

from math import isclose

from pytest import raises

from pd_matcher.eval.metrics import ThresholdPoint
from pd_matcher.eval.metrics import average_precision
from pd_matcher.eval.metrics import roc_auc
from pd_matcher.eval.metrics import threshold_sweep


def test_roc_auc_perfect_ranking() -> None:
    """All positives above all negatives -> AUC = 1.0."""
    scored = [(0.9, 1), (0.8, 1), (0.4, 0), (0.1, 0)]
    assert roc_auc(scored) == 1.0


def test_roc_auc_inverse_ranking() -> None:
    """All negatives above all positives -> AUC = 0.0."""
    scored = [(0.9, 0), (0.8, 0), (0.4, 1), (0.1, 1)]
    assert roc_auc(scored) == 0.0


def test_roc_auc_random_ranking_known_value() -> None:
    """Hand-computed mid-ranking -> known AUC value."""
    scored = [(0.9, 1), (0.8, 0), (0.4, 1), (0.1, 0)]
    assert isclose(roc_auc(scored), 0.75)


def test_roc_auc_handles_ties_with_midranks() -> None:
    """Ties across classes are scored with mid-ranks (no bias)."""
    scored = [(0.5, 1), (0.5, 0), (0.4, 0), (0.6, 1)]
    assert isclose(roc_auc(scored), 0.875)


def test_roc_auc_single_class_returns_default() -> None:
    """Only positives -> AUC undefined; function returns 0.5."""
    assert roc_auc([(0.9, 1), (0.8, 1), (0.4, 1)]) == 0.5


def test_roc_auc_single_class_negatives_only_returns_default() -> None:
    """Only negatives -> AUC undefined; function returns 0.5."""
    assert roc_auc([(0.9, 0), (0.8, 0), (0.4, 0)]) == 0.5


def test_roc_auc_empty_raises() -> None:
    """Empty input is a programming error, not a 0.0."""
    with raises(ValueError, match="at least one entry"):
        roc_auc([])


def test_roc_auc_rejects_bad_label() -> None:
    """Labels other than 0/1 are rejected up front."""
    with raises(ValueError, match="label must be 0 or 1"):
        roc_auc([(0.5, 2)])


def test_average_precision_perfect_ranking() -> None:
    """All positives ranked first -> AP = 1.0."""
    scored = [(0.9, 1), (0.8, 1), (0.4, 0), (0.1, 0)]
    assert average_precision(scored) == 1.0


def test_average_precision_imperfect_ranking_known_value() -> None:
    """Hand-computed alternating ranking -> known AP value."""
    scored = [(0.9, 1), (0.8, 0), (0.4, 1), (0.1, 0)]
    # rank 1 (pos): precision 1/1 = 1.0
    # rank 3 (pos): precision 2/3 ~= 0.6667
    # AP = (1.0 + 2/3) / 2 = 0.8333...
    assert isclose(average_precision(scored), (1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_no_positives_returns_zero() -> None:
    """No positives -> AP undefined; function returns 0.0."""
    assert average_precision([(0.9, 0), (0.1, 0)]) == 0.0


def test_average_precision_all_positives_returns_one() -> None:
    """Every entry is a positive -> precision 1.0 at every rank -> AP = 1.0."""
    assert average_precision([(0.9, 1), (0.8, 1), (0.4, 1)]) == 1.0


def test_average_precision_empty_raises() -> None:
    with raises(ValueError, match="at least one entry"):
        average_precision([])


def test_average_precision_rejects_bad_label() -> None:
    with raises(ValueError, match="label must be 0 or 1"):
        average_precision([(0.5, -1)])


def test_threshold_sweep_default_grid_has_twenty_one_points() -> None:
    """Default 0.0..1.0 step 0.05 -> 21 inclusive grid points."""
    scored = [(0.9, 1), (0.4, 0)]
    sweep = threshold_sweep(scored)
    assert len(sweep) == 21
    assert sweep[0].threshold == 0.0
    assert isclose(sweep[-1].threshold, 1.0)


def test_threshold_sweep_zero_threshold_catches_all() -> None:
    """``threshold = 0.0`` predicts positive for everyone -> recall = 1.0."""
    scored = [(0.9, 1), (0.4, 0), (0.1, 1)]
    sweep = threshold_sweep(scored, start=0.0, stop=1.0, step=0.5)
    bottom = sweep[0]
    assert bottom.true_positives == 2
    assert bottom.false_positives == 1
    assert bottom.false_negatives == 0
    assert bottom.recall == 1.0
    assert isclose(bottom.precision, 2.0 / 3.0)


def test_threshold_sweep_one_threshold_catches_nothing() -> None:
    """Above-everyone threshold -> no predictions -> all positives become FN."""
    scored = [(0.9, 1), (0.4, 0), (0.1, 1)]
    sweep = threshold_sweep(scored, start=2.0, stop=2.0, step=1.0)
    point = sweep[0]
    assert point.true_positives == 0
    assert point.false_positives == 0
    assert point.false_negatives == 2
    assert point.precision == 0.0
    assert point.recall == 0.0
    assert point.f1 == 0.0


def test_threshold_sweep_returns_threshold_points() -> None:
    """Sweep results are :class:`ThresholdPoint` instances with derived f1."""
    scored = [(0.9, 1), (0.1, 0)]
    sweep = threshold_sweep(scored, start=0.5, stop=0.5, step=1.0)
    point = sweep[0]
    assert isinstance(point, ThresholdPoint)
    assert point.true_positives == 1
    assert point.false_positives == 0
    assert point.false_negatives == 0
    assert point.precision == 1.0
    assert point.recall == 1.0
    assert point.f1 == 1.0


def test_threshold_sweep_f1_zero_when_both_precision_and_recall_zero() -> None:
    """Both precision and recall zero -> f1 = 0.0 (no NaN)."""
    scored = [(0.1, 1), (0.1, 0)]
    sweep = threshold_sweep(scored, start=0.5, stop=0.5, step=1.0)
    point = sweep[0]
    assert point.precision == 0.0
    assert point.recall == 0.0
    assert point.f1 == 0.0


def test_threshold_sweep_empty_raises() -> None:
    with raises(ValueError, match="at least one entry"):
        threshold_sweep([])


def test_threshold_sweep_rejects_bad_label() -> None:
    with raises(ValueError, match="label must be 0 or 1"):
        threshold_sweep([(0.5, 5)])


def test_threshold_sweep_rejects_non_positive_step() -> None:
    scored = [(0.5, 1)]
    with raises(ValueError, match="step must be > 0"):
        threshold_sweep(scored, start=0.0, stop=1.0, step=0.0)


def test_threshold_sweep_rejects_stop_before_start() -> None:
    scored = [(0.5, 1)]
    with raises(ValueError, match="stop must be >= start"):
        threshold_sweep(scored, start=1.0, stop=0.0, step=0.1)
