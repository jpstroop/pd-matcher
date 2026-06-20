"""Unit tests for active-learning dual-scoring + disagreement (issue #81).

Covers the disagreement signal, bucketing, and the per-record dual-scoring pass
in isolation. A fake learned combiner and an in-memory candidate scorer drive
:func:`score_record` so no LMDB index or real learned artifact is needed.
"""

from collections.abc import Iterator
from collections.abc import Sequence

from pd_groundtruth.active_score import BUCKET_AGREE_HIGH
from pd_groundtruth.active_score import BUCKET_AGREE_LOW
from pd_groundtruth.active_score import BUCKET_INFORMATIVE
from pd_groundtruth.active_score import CandidateScorer
from pd_groundtruth.active_score import TopCandidate
from pd_groundtruth.active_score import bucket_of
from pd_groundtruth.active_score import disagreement_magnitude
from pd_groundtruth.active_score import score_record
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


class _LearnedFromMap:
    """A combiner returning a per-uuid calibrated score keyed off Evidence.

    The fake encodes the candidate's uuid in a single ``Evidence.scorer`` tag so
    the learned arm can return a different probability per candidate without a
    real model.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:
        uuid = evidence[0].scorer
        score = self._scores[uuid]
        return CombinedScore(raw=score * 100.0, calibrated=score)


def _marc(control_id: str = "ctrl-1") -> MarcRecord:
    return MarcRecord(
        control_id=control_id, title="A Title", title_main="A Title", publication_year=1953
    )


def _cce(uuid: str) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid,
        title="CCE Title",
        was_renewed=True,
        regnum="R1",
        reg_year=1953,
    )


def _candidate(uuid: str, weighted: float) -> CandidateMatch:
    evidence = Evidence(
        scorer=uuid, score=weighted, max=1.0, skipped=False, decisive=False, features=()
    )
    return CandidateMatch(
        nypl_uuid=uuid,
        nypl_year=1953,
        combined=CombinedScore(raw=weighted * 100.0, calibrated=weighted),
        evidence=(evidence,),
        losing_evidence=(),
        evidence_sources=(("245", "title"),),
    )


def _scorer(candidates: list[tuple[IndexedNyplRegRecord, CandidateMatch]]) -> CandidateScorer:
    def scorer(_marc: MarcRecord) -> Iterator[tuple[IndexedNyplRegRecord, CandidateMatch]]:
        yield from candidates

    return scorer


def test_disagreement_zero_when_no_candidates() -> None:
    top = TopCandidate(uuid=None, score=0.0)
    assert disagreement_magnitude(top, top) == 0.0


def test_disagreement_different_pick_outranks_any_gap() -> None:
    weighted = TopCandidate(uuid="a", score=0.9)
    learned = TopCandidate(uuid="b", score=0.1)
    magnitude = disagreement_magnitude(weighted, learned)
    assert magnitude == 1.0 + 0.8
    assert magnitude > 1.0


def test_disagreement_same_pick_is_score_gap() -> None:
    weighted = TopCandidate(uuid="a", score=0.9)
    learned = TopCandidate(uuid="a", score=0.4)
    assert abs(disagreement_magnitude(weighted, learned) - 0.5) < 1e-9


def test_bucket_agree_high_requires_same_pick_and_both_confident() -> None:
    weighted = TopCandidate(uuid="a", score=0.9)
    learned = TopCandidate(uuid="a", score=0.85)
    assert bucket_of(weighted, learned) == BUCKET_AGREE_HIGH


def test_bucket_agree_low_when_both_reject() -> None:
    weighted = TopCandidate(uuid="a", score=0.2)
    learned = TopCandidate(uuid="a", score=0.1)
    assert bucket_of(weighted, learned) == BUCKET_AGREE_LOW


def test_bucket_agree_low_when_no_candidates() -> None:
    top = TopCandidate(uuid=None, score=0.0)
    assert bucket_of(top, top) == BUCKET_AGREE_LOW


def test_bucket_informative_on_committee_split() -> None:
    weighted = TopCandidate(uuid="a", score=0.9)
    learned = TopCandidate(uuid="b", score=0.9)
    assert bucket_of(weighted, learned) == BUCKET_INFORMATIVE


def test_bucket_informative_on_same_pick_but_split_confidence() -> None:
    weighted = TopCandidate(uuid="a", score=0.9)
    learned = TopCandidate(uuid="a", score=0.2)
    assert bucket_of(weighted, learned) == BUCKET_INFORMATIVE


def test_bucket_informative_when_both_uncertain() -> None:
    weighted = TopCandidate(uuid="a", score=0.5)
    learned = TopCandidate(uuid="a", score=0.5)
    assert bucket_of(weighted, learned) == BUCKET_INFORMATIVE


def test_score_record_no_candidates_is_agree_low_with_no_cce() -> None:
    record = score_record(
        _marc(),
        candidate_scorer=_scorer([]),
        learned=_LearnedFromMap({}),
    )
    assert record.bucket == BUCKET_AGREE_LOW
    assert record.cce is None
    assert record.weighted.uuid is None
    assert record.disagreement == 0.0
    assert record.evidence == ()


def test_score_record_agreeing_match_keeps_weighted_top_cce() -> None:
    candidates = [(_cce("a"), _candidate("a", 0.9))]
    record = score_record(
        _marc(),
        candidate_scorer=_scorer(candidates),
        learned=_LearnedFromMap({"a": 0.88}),
    )
    assert record.bucket == BUCKET_AGREE_HIGH
    assert record.cce is not None
    assert record.cce.uuid == "a"
    assert record.weighted.uuid == "a"
    assert record.learned.uuid == "a"


def test_score_record_committee_split_is_informative() -> None:
    candidates = [
        (_cce("a"), _candidate("a", 0.9)),
        (_cce("b"), _candidate("b", 0.4)),
    ]
    record = score_record(
        _marc(),
        candidate_scorer=_scorer(candidates),
        learned=_LearnedFromMap({"a": 0.2, "b": 0.95}),
    )
    assert record.bucket == BUCKET_INFORMATIVE
    assert record.weighted.uuid == "a"
    assert record.learned.uuid == "b"
    assert record.cce is not None
    assert record.cce.uuid == "a"
    assert record.disagreement > 1.0
    assert record.evidence_sources == (("245", "title"),)
