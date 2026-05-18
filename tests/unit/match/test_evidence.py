"""Tests for :mod:`pd_matcher.match.evidence`."""

from pytest import raises

from pd_matcher.match.evidence import Evidence


def _evidence(
    *,
    scorer: str = "title.token_set",
    score: float = 75.0,
    max_score: float = 100.0,
    skipped: bool = False,
    decisive: bool = False,
    features: tuple[tuple[str, float], ...] = (),
) -> Evidence:
    return Evidence(
        scorer=scorer,
        score=score,
        max=max_score,
        skipped=skipped,
        decisive=decisive,
        features=features,
    )


def test_evidence_normalized_returns_score_over_max() -> None:
    """The normalized property divides ``score`` by ``max``."""
    ev = _evidence(score=80.0, max_score=100.0)
    assert ev.normalized == 0.8


def test_evidence_normalized_returns_zero_when_skipped() -> None:
    """Skipped Evidence has a normalized score of zero."""
    ev = _evidence(skipped=True, score=42.0)
    assert ev.normalized == 0.0


def test_evidence_normalized_returns_zero_when_max_zero() -> None:
    """A zero ``max`` short-circuits to zero rather than dividing by zero."""
    ev = _evidence(score=0.0, max_score=0.0)
    assert ev.normalized == 0.0


def test_evidence_is_frozen() -> None:
    """The struct must reject attribute mutation."""
    ev = _evidence()
    with raises(AttributeError):
        setattr(ev, "score", 99.0)
