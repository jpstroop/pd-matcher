"""Tests for :mod:`pd_matcher.workers.messages`."""

from datetime import date

from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import decode_worker_output
from pd_matcher.workers.messages import encode_worker_output


def test_worker_output_roundtrip_with_full_match() -> None:
    marc = MarcRecord(control_id="m", title="t", title_main="t", publication_year=1940)
    nypl = IndexedNyplRegRecord(
        uuid="UUID-1",
        title="t",
        was_renewed=False,
        reg_date=date(1940, 1, 1),
        reg_year=1940,
    )
    match = MatchResult(
        marc_control_id="m",
        best=CandidateMatch(
            nypl_uuid="UUID-1",
            nypl_year=1940,
            combined=CombinedScore(raw=80.0, calibrated=0.8),
            evidence=(
                Evidence(
                    scorer="title.token_set",
                    score=80.0,
                    max=100.0,
                    skipped=False,
                    decisive=False,
                    features=(("overlap", 1.0),),
                ),
            ),
            losing_evidence=(),
        ),
        alternates=(),
        candidates_considered=1,
    )
    assessment = CopyrightAssessment(
        status=CopyrightStatus.PD_REGISTERED_NOT_RENEWED,
        matched_rule_name="rule",
        explanation="ok",
        assumptions=("a",),
    )
    payload = WorkerOutput(marc=marc, match=match, assessment=assessment, matched_nypl=nypl)
    decoded = decode_worker_output(encode_worker_output(payload))
    assert decoded == payload


def test_worker_output_roundtrip_with_empty_match() -> None:
    """A MatchResult with ``best=None`` is the empty-match signal on the wire."""
    marc = MarcRecord(control_id="m", title="t", title_main="t")
    empty_match = MatchResult(
        marc_control_id="m",
        best=None,
        alternates=(),
        candidates_considered=0,
    )
    assessment = CopyrightAssessment(
        status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
        matched_rule_name=None,
        explanation="",
        assumptions=(),
    )
    payload = WorkerOutput(marc=marc, match=empty_match, assessment=assessment, matched_nypl=None)
    decoded = decode_worker_output(encode_worker_output(payload))
    assert decoded == payload
