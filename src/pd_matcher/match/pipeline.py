"""Public entry point for the Phase 4 matching pipeline.

:func:`match_record` is the single function the rest of the codebase calls
to match a :class:`MarcRecord` against the indexed NYPL corpus. The flow
is intentionally small:

1. Retrieve candidates from the lookup: registrations that share both the
   year window and at least one title/author/publisher token with the MARC
   record (cheap inverted-index retrieval, not a full year-bucket scan).
2. Build one :class:`ScorerContext` for the record (one stopword/stemmer
   resolution per record, not per candidate).
3. For each candidate run all scorers, keep the best Evidence per scorer
   from the bounded field-pair permutations, and combine.
4. Sort by calibrated score, apply the configured floor, and return the
   top result plus up to ``top_k - 1`` runners-up.
"""

from collections.abc import Callable
from collections.abc import Sequence

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import calibrate
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairing
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.edition import score_edition
from pd_matcher.match.scorers.isbn import score_isbn
from pd_matcher.match.scorers.lccn import score_lccn
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.match.scorers.title import score_title
from pd_matcher.match.scorers.year import score_year
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.stemming import stemmer_for
from pd_matcher.normalize.stopwords import load_stopwords

_DEFAULT_LANGUAGE: str = "eng"


def _build_context(marc: MarcRecord, idf: IdfTable, config: MatchingConfig) -> ScorerContext:
    language = marc.language_code or _DEFAULT_LANGUAGE
    return ScorerContext(
        language=language,
        stopwords=load_stopwords(language),
        stemmer=stemmer_for(language),
        idf=idf,
        config=config,
    )


def _select_best(evidences: Sequence[Evidence]) -> tuple[Evidence, tuple[Evidence, ...]]:
    """Return the highest-scoring Evidence and the losers in input order."""
    best_index = 0
    best_score = evidences[0].score if not evidences[0].skipped else -1.0
    for index in range(1, len(evidences)):
        current = evidences[index]
        current_score = current.score if not current.skipped else -1.0
        if current_score > best_score:
            best_score = current_score
            best_index = index
    losers = tuple(ev for index, ev in enumerate(evidences) if index != best_index)
    return evidences[best_index], losers


_GroupScorer = Callable[[str | None, str | None, ScorerContext], Evidence]

_GROUP_SCORERS: dict[str, _GroupScorer] = {
    "title": score_title,
    "author": score_author,
    "publisher": score_publisher,
}


def _score_group(
    pairings: tuple[CompiledPairing, ...],
    marc: MarcRecord,
    candidate: IndexedNyplRegRecord,
    ctx: ScorerContext,
    winning: list[Evidence],
    losing: list[Evidence],
) -> None:
    """Score every pairing in one group and append best/losers to the lists.

    The combiner keys on exactly one Evidence per scorer tag, so the best
    Evidence (the highest-scoring pairing) is appended to ``winning`` and
    the rest to ``losing`` for audit.
    """
    if not pairings:
        return
    scorer = _GROUP_SCORERS[pairings[0].group]
    evidences = tuple(
        scorer(pairing.marc_accessor(marc), pairing.cce_accessor(candidate), ctx)
        for pairing in pairings
    )
    best, losers = _select_best(evidences)
    winning.append(best)
    losing.extend(losers)


def _score_candidate(
    marc: MarcRecord,
    candidate: IndexedNyplRegRecord,
    ctx: ScorerContext,
    combiner: Combiner,
    calibrator: PlattCalibrator | None,
    pairings: CompiledPairings,
) -> CandidateMatch:
    winning: list[Evidence] = []
    losing: list[Evidence] = []

    winning.append(score_lccn(marc.lccn, candidate, ctx))
    winning.append(score_isbn(marc.isbns, candidate, ctx))

    _score_group(pairings.title, marc, candidate, ctx, winning, losing)
    _score_group(pairings.author, marc, candidate, ctx, winning, losing)
    _score_group(pairings.publisher, marc, candidate, ctx, winning, losing)

    winning.append(score_year(marc.publication_year, candidate.reg_year, ctx))
    winning.append(score_edition(marc.edition, candidate.edition, ctx))

    combined = combiner.combine(tuple(winning))
    if calibrator is not None:
        calibrated = calibrate(combined.raw, calibrator)
        combined = CombinedScore(raw=combined.raw, calibrated=calibrated)
    return CandidateMatch(
        nypl_uuid=candidate.uuid,
        nypl_year=candidate.reg_year,
        combined=combined,
        evidence=tuple(winning),
        losing_evidence=tuple(losing),
    )


def match_record(
    marc: MarcRecord,
    *,
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    combiner: Combiner,
    pairings: CompiledPairings,
    top_k: int = 3,
) -> MatchResult:
    """Match a MARC record against the indexed NYPL corpus.

    Args:
        marc: The MARC record to match.
        lookup: Open read-only LMDB lookup.
        config: Active :class:`MatchingConfig`.
        idf: Pre-built :class:`IdfTable`.
        calibrator: Optional Platt calibrator. When supplied,
            ``combined.calibrated`` is set to ``P(true match)``; when
            ``None``, ``calibrated = raw / 100``.
        combiner: Concrete :class:`Combiner` (Phase 4 default is
            :class:`WeightedMeanCombiner`).
        pairings: Compiled field pairings driving the title/author/
            publisher scorer groups.
        top_k: Total number of candidates to retain (best + alternates).

    Returns:
        A :class:`MatchResult` describing the verdict.
    """
    if marc.publication_year is None:
        return MatchResult(
            marc_control_id=marc.control_id,
            best=None,
            alternates=(),
            candidates_considered=0,
        )
    candidates = list(lookup.candidates_for(marc, config.year_window))
    if not candidates:
        return MatchResult(
            marc_control_id=marc.control_id,
            best=None,
            alternates=(),
            candidates_considered=0,
        )
    ctx = _build_context(marc, idf, config)
    scored = [
        _score_candidate(marc, candidate, ctx, combiner, calibrator, pairings)
        for candidate in candidates
    ]
    scored.sort(key=lambda match: match.combined.calibrated, reverse=True)
    floor = config.min_combined_score / 100.0
    qualifying = [match for match in scored if match.combined.calibrated >= floor]
    if not qualifying:
        return MatchResult(
            marc_control_id=marc.control_id,
            best=None,
            alternates=(),
            candidates_considered=len(candidates),
        )
    best = qualifying[0]
    alternates = tuple(qualifying[1:top_k])
    return MatchResult(
        marc_control_id=marc.control_id,
        best=best,
        alternates=alternates,
        candidates_considered=len(candidates),
    )


__all__ = [
    "match_record",
]
