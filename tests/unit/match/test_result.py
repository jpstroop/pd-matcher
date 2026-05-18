"""Tests for :mod:`pd_matcher.match.result`."""

from pytest import raises

from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult


def test_candidate_match_is_frozen() -> None:
    """CandidateMatch must reject attribute mutation."""
    candidate = CandidateMatch(
        nypl_uuid="u",
        nypl_year=1940,
        combined=CombinedScore(raw=80.0, calibrated=0.8),
        evidence=(),
        losing_evidence=(),
    )
    with raises(AttributeError):
        setattr(candidate, "nypl_uuid", "other")


def test_match_result_is_frozen() -> None:
    """MatchResult must reject attribute mutation."""
    result = MatchResult(
        marc_control_id="m",
        best=None,
        alternates=(),
        candidates_considered=0,
    )
    with raises(AttributeError):
        setattr(result, "marc_control_id", "other")
