"""Tests for :mod:`pd_matcher.match.combiners.learned_calibrator`."""

from pathlib import Path

from pd_matcher.match.combiners.learned_calibrator import IsotonicCalibrator
from pd_matcher.match.combiners.learned_calibrator import apply_isotonic
from pd_matcher.match.combiners.learned_calibrator import load_learned_calibrator
from pd_matcher.match.combiners.learned_calibrator import save_learned_calibrator


def _calibrator() -> IsotonicCalibrator:
    """A three-segment monotone calibrator with one flat segment."""
    return IsotonicCalibrator(
        xs=(0.0, 0.5, 0.5, 1.0),
        ys=(0.0, 0.2, 0.2, 0.9),
        trained_at="2026-07-11T00:00:00+00:00",
        n_positive=100,
        n_negative=100,
    )


def test_apply_clamps_below_range() -> None:
    """Inputs at or below the first breakpoint return the first output."""
    cal = _calibrator()
    assert apply_isotonic(-1.0, cal) == 0.0
    assert apply_isotonic(0.0, cal) == 0.0


def test_apply_clamps_above_range() -> None:
    """Inputs at or above the last breakpoint return the last output."""
    cal = _calibrator()
    assert apply_isotonic(2.0, cal) == 0.9
    assert apply_isotonic(1.0, cal) == 0.9


def test_apply_interpolates_between_breakpoints() -> None:
    """A midpoint interpolates linearly between neighbouring breakpoints."""
    cal = _calibrator()
    # halfway across the [0.5, 1.0] -> [0.2, 0.9] segment
    assert apply_isotonic(0.75, cal) == 0.55


def test_apply_handles_duplicated_breakpoint() -> None:
    """A duplicated breakpoint (flat step) interpolates from the right segment.

    ``bisect_right`` lands past both equal breakpoints, so the interpolation
    always uses a strictly-positive-width segment (no division by zero).
    """
    cal = IsotonicCalibrator(
        xs=(0.0, 0.5, 0.5, 1.0),
        ys=(0.0, 0.3, 0.7, 1.0),
        trained_at="2026-07-11T00:00:00+00:00",
        n_positive=1,
        n_negative=1,
    )
    assert apply_isotonic(0.5, cal) == 0.7


def test_apply_is_monotone_non_decreasing() -> None:
    """Calibrated output never decreases as the raw probability rises."""
    cal = _calibrator()
    previous = -1.0
    for i in range(101):
        current = apply_isotonic(i / 100.0, cal)
        assert current >= previous
        previous = current


def test_save_load_round_trip(tmp_path: Path) -> None:
    """A saved calibrator reloads to an identical struct."""
    cal = _calibrator()
    path = tmp_path / "learned_calibrator.msgpack"
    save_learned_calibrator(cal, path)
    assert load_learned_calibrator(path) == cal
