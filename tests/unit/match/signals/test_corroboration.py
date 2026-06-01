"""Tests for :mod:`pd_matcher.match.signals.corroboration`."""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.signals.corroboration import has_no_corroboration


def _evidence(
    *,
    score: float = 0.0,
    skipped: bool = False,
    scorer: str = "test.scorer",
) -> Evidence:
    return Evidence(
        scorer=scorer,
        score=score,
        max=100.0,
        skipped=skipped,
        decisive=False,
        features=(),
    )


def test_returns_true_when_all_others_skipped() -> None:
    """All scorers skipped → no corroboration."""
    others = (_evidence(skipped=True), _evidence(skipped=True))
    assert has_no_corroboration(others, threshold=50.0) is True


def test_returns_true_when_all_others_below_threshold() -> None:
    """Every scorer fires but stays below threshold → no corroboration."""
    others = (_evidence(score=10.0), _evidence(score=49.9), _evidence(score=0.0))
    assert has_no_corroboration(others, threshold=50.0) is True


def test_returns_false_when_one_other_meets_threshold() -> None:
    """Any single Evidence at or above threshold corroborates."""
    others = (_evidence(score=10.0), _evidence(score=50.0), _evidence(score=0.0))
    assert has_no_corroboration(others, threshold=50.0) is False


def test_returns_false_when_one_other_well_above_threshold() -> None:
    others = (_evidence(score=0.0), _evidence(score=100.0))
    assert has_no_corroboration(others, threshold=50.0) is False


def test_skipped_evidence_at_high_score_does_not_corroborate() -> None:
    """A skipped Evidence carries no signal regardless of its score."""
    others = (_evidence(score=100.0, skipped=True),)
    assert has_no_corroboration(others, threshold=50.0) is True


def test_empty_others_returns_true() -> None:
    """An empty corroboration set is the most extreme isolation."""
    assert has_no_corroboration((), threshold=50.0) is True


def test_threshold_is_inclusive_at_lower_bound() -> None:
    """A score exactly at the threshold counts as corroborating."""
    others = (_evidence(score=50.0),)
    assert has_no_corroboration(others, threshold=50.0) is False
