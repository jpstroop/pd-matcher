"""Tests for :mod:`pd_matcher.match.combiners.calibrator`."""

from pathlib import Path

from pytest import raises

from pd_matcher.match.combiners.calibrator import calibrate
from pd_matcher.match.combiners.calibrator import load_calibrator
from pd_matcher.match.combiners.calibrator import save_calibrator
from pd_matcher.match.combiners.calibrator import train_calibrator


def test_train_calibrator_separates_clear_classes() -> None:
    """With strong separation the calibrator should put positives above 0.5."""
    positives = [95.0, 90.0, 85.0, 88.0, 92.0, 87.0, 96.0, 91.0]
    negatives = [10.0, 15.0, 12.0, 8.0, 5.0, 11.0, 9.0, 14.0]
    calibrator = train_calibrator(positives, negatives)
    for raw in positives:
        assert calibrate(raw, calibrator) > 0.5
    for raw in negatives:
        assert calibrate(raw, calibrator) < 0.5


def test_train_calibrator_is_monotonic_in_raw_score() -> None:
    """Higher raw scores must produce non-decreasing calibrated probabilities."""
    positives = [95.0, 90.0, 88.0]
    negatives = [10.0, 12.0, 14.0]
    calibrator = train_calibrator(positives, negatives)
    previous = -1.0
    for raw in range(0, 101, 10):
        current = calibrate(float(raw), calibrator)
        assert current >= previous
        previous = current


def test_train_calibrator_rejects_empty_inputs() -> None:
    """Empty positives or negatives must raise."""
    with raises(ValueError, match="non-empty"):
        train_calibrator([], [1.0])
    with raises(ValueError, match="non-empty"):
        train_calibrator([1.0], [])


def test_calibrator_roundtrip_via_msgpack(tmp_path: Path) -> None:
    """Save + load reproduces the calibrator exactly."""
    calibrator = train_calibrator([95.0, 92.0], [5.0, 10.0])
    path = tmp_path / "calibrator.msgpack"
    save_calibrator(calibrator, path)
    assert path.exists()
    loaded = load_calibrator(path)
    assert loaded == calibrator


def test_calibrator_records_class_counts() -> None:
    """The fitted struct records how many examples it trained on."""
    calibrator = train_calibrator([95.0, 92.0, 90.0], [5.0, 10.0])
    assert calibrator.n_positive == 3
    assert calibrator.n_negative == 2
    assert calibrator.trained_at.endswith("+00:00")


def test_calibrator_respects_max_iterations() -> None:
    """``max_iterations=1`` forces the loop to exit by exhaustion."""
    calibrator = train_calibrator([95.0, 92.0], [5.0, 10.0], max_iterations=1)
    assert calibrator.n_positive == 2
    assert calibrator.n_negative == 2


def test_calibrator_converges_quickly_on_overlapping_classes() -> None:
    """Even with overlap the optimiser must return a finite calibrator."""
    positives = [60.0, 65.0, 70.0, 72.0]
    negatives = [55.0, 62.0, 68.0, 60.0]
    calibrator = train_calibrator(positives, negatives)
    for raw in (0.0, 50.0, 100.0):
        probability = calibrate(raw, calibrator)
        assert 0.0 <= probability <= 1.0
