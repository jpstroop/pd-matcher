"""Dual-scoring + disagreement signal for active-learning selection (issue #81).

Turns one selected :class:`~pd_matcher.models.MarcRecord` into a
:class:`ScoredRecord` carrying BOTH combiners' verdict on its best CCE
candidate, then classifies the record into an active-learning bucket by how
much the two matchers disagree.

The Evidence for each candidate is computed ONCE (via the matcher's per-pair
routine forced onto the weighted-mean combiner — the per-scorer Evidence is
combiner-independent, exactly as
:func:`pd_matcher.match.combiners.train._scoring_config` relies on). The
weighted-mean combiner's calibrated output and the learned combiner's
probability are both read off that single Evidence stream, so the two scores
are strictly comparable and the learned model never re-retrieves.

A record's disagreement is the union of three signals (issue #81):

* the two combiners pick a DIFFERENT top-1 candidate (query-by-committee), or
* they pick the SAME top-1 but their calibrated scores differ by a large gap,
  or
* both are low-confidence on their respective top-1 (a both-uncertain pair).

Bucketing collapses to three labels: ``agree-high`` (both confidently match —
likely a real match), ``agree-low`` (both confidently non-match), and
``informative`` (everything else — the disagreements and the both-middle
pairs). Only ``informative`` records are queued for human review.

The functions here are pure over their inputs (a candidate scorer callable, the
two combiners, plain score floats), so the whole signal is unit-testable
without an LMDB index or the real learned artifact.
"""

from collections.abc import Callable
from collections.abc import Iterator
from logging import getLogger

from msgspec import Struct

from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)

BUCKET_AGREE_HIGH: str = "agree-high"
BUCKET_AGREE_LOW: str = "agree-low"
BUCKET_INFORMATIVE: str = "informative"

BUCKET_ORDER: tuple[str, ...] = (BUCKET_INFORMATIVE, BUCKET_AGREE_HIGH, BUCKET_AGREE_LOW)

# Calibrated-probability zones, shared by both combiners (the learned model's
# output is already a calibrated probability; the weighted combiner's
# ``calibrated`` is the Platt-scaled probability). A pair is confident-match at
# ``>= _HIGH``, confident-no at ``<= _LOW``, and uncertain in between.
_HIGH: float = 0.70
_LOW: float = 0.30

# Candidate retrieval for one MARC: yields ``(cce_record, weighted_scored)``
# pairs so the full CCE registration travels alongside the score for the queue
# write. Injected so the scoring policy is testable without an LMDB index (the
# orchestration binds the real ``candidates_for`` + per-pair scorer).
CandidateScorer = Callable[[MarcRecord], Iterator[tuple[IndexedNyplRegRecord, CandidateMatch]]]


class TopCandidate(Struct, frozen=True, forbid_unknown_fields=True):
    """One combiner's best CCE candidate for a MARC record.

    ``uuid`` is :data:`None` only when the MARC retrieved no candidate at all
    (no in-window, token-sharing CCE), in which case ``score`` is ``0.0``.
    """

    uuid: str | None
    score: float


class ScoredRecord(Struct, frozen=True, forbid_unknown_fields=True):
    """One selected MARC dual-scored by both combiners plus its bucket.

    ``evidence`` / ``evidence_sources`` belong to the weighted combiner's
    top-1 candidate (the pair that gets written to the review queue when the
    record is ``informative``). ``cce`` is that same top-1 CCE registration, or
    :data:`None` when no candidate was retrieved.
    """

    marc: MarcRecord
    cce: IndexedNyplRegRecord | None
    weighted: TopCandidate
    learned: TopCandidate
    evidence: tuple[Evidence, ...]
    evidence_sources: tuple[tuple[str, str], ...]
    disagreement: float
    bucket: str


class _Scored(Struct, frozen=True, forbid_unknown_fields=True):
    """One candidate carried with both combiners' scores during ranking."""

    cce: IndexedNyplRegRecord
    candidate: CandidateMatch
    learned_score: float


def _top_by(scored: list[_Scored], score_of: Callable[[_Scored], float]) -> _Scored | None:
    """Return the :class:`_Scored` item maximizing ``score_of`` (``None`` if empty)."""
    if not scored:
        return None
    best = scored[0]
    best_value = score_of(best)
    for item in scored[1:]:
        value = score_of(item)
        if value > best_value:
            best_value = value
            best = item
    return best


def disagreement_magnitude(weighted: TopCandidate, learned: TopCandidate) -> float:
    """Return how strongly the two combiners disagree on a record.

    The magnitude ranks records for review (higher = more informative):

    * Different top-1 candidates → ``1.0 + |weighted.score - learned.score|``,
      so any committee split outranks every same-pick gap.
    * Same top-1 candidate → the absolute calibrated-score gap on that pick.
    * No candidate retrieved on either side → ``0.0`` (nothing to disagree on).
    """
    if weighted.uuid is None and learned.uuid is None:
        return 0.0
    if weighted.uuid != learned.uuid:
        return 1.0 + abs(weighted.score - learned.score)
    return abs(weighted.score - learned.score)


def bucket_of(weighted: TopCandidate, learned: TopCandidate) -> str:
    """Classify a dual-scored record into its active-learning bucket.

    ``agree-high`` requires both combiners to confidently match the SAME top-1
    candidate; ``agree-low`` requires both to confidently reject (top-1 score
    ``<= _LOW``, or no candidate retrieved). Everything else — a committee
    split, a large same-pick gap, or a both-uncertain pair — is
    ``informative``.
    """
    same_pick = weighted.uuid == learned.uuid and weighted.uuid is not None
    both_high = weighted.score >= _HIGH and learned.score >= _HIGH
    if same_pick and both_high:
        return BUCKET_AGREE_HIGH
    both_low = weighted.score <= _LOW and learned.score <= _LOW
    if both_low:
        return BUCKET_AGREE_LOW
    return BUCKET_INFORMATIVE


def score_record(
    marc: MarcRecord,
    *,
    candidate_scorer: CandidateScorer,
    learned: Combiner,
) -> ScoredRecord:
    """Dual-score one MARC and classify it for active learning.

    ``candidate_scorer`` yields ``(cce_record, weighted_scored)`` pairs already
    scored by the WEIGHTED combiner (each :class:`CandidateMatch` carries the
    shared Evidence and the weighted calibrated score). The learned probability
    for every candidate is read off that same Evidence via ``learned.combine``,
    so no re-retrieval happens. The weighted top-1 (by ``combined.calibrated``)
    and the learned top-1 (by learned probability) are compared to produce the
    record's disagreement magnitude and bucket. The CCE registration written to
    the queue is the WEIGHTED top-1's (the pair the production matcher proposes).
    """
    scored: list[_Scored] = [
        _Scored(
            cce=cce,
            candidate=candidate,
            learned_score=learned.combine(candidate.evidence).calibrated,
        )
        for cce, candidate in candidate_scorer(marc)
    ]
    weighted_best = _top_by(scored, lambda item: item.candidate.combined.calibrated)
    learned_best = _top_by(scored, lambda item: item.learned_score)
    if weighted_best is None or learned_best is None:
        weighted_top = TopCandidate(uuid=None, score=0.0)
        learned_top = TopCandidate(uuid=None, score=0.0)
        magnitude = disagreement_magnitude(weighted_top, learned_top)
        return ScoredRecord(
            marc=marc,
            cce=None,
            weighted=weighted_top,
            learned=learned_top,
            evidence=(),
            evidence_sources=(),
            disagreement=magnitude,
            bucket=bucket_of(weighted_top, learned_top),
        )
    weighted_top = TopCandidate(
        uuid=weighted_best.candidate.nypl_uuid,
        score=weighted_best.candidate.combined.calibrated,
    )
    learned_top = TopCandidate(
        uuid=learned_best.candidate.nypl_uuid, score=learned_best.learned_score
    )
    return ScoredRecord(
        marc=marc,
        cce=weighted_best.cce,
        weighted=weighted_top,
        learned=learned_top,
        evidence=weighted_best.candidate.evidence,
        evidence_sources=weighted_best.candidate.evidence_sources,
        disagreement=disagreement_magnitude(weighted_top, learned_top),
        bucket=bucket_of(weighted_top, learned_top),
    )


__all__ = [
    "BUCKET_AGREE_HIGH",
    "BUCKET_AGREE_LOW",
    "BUCKET_INFORMATIVE",
    "BUCKET_ORDER",
    "CandidateScorer",
    "ScoredRecord",
    "TopCandidate",
    "bucket_of",
    "disagreement_magnitude",
    "score_record",
]
