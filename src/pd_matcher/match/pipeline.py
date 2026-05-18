"""Public entry point for the Phase 4 matching pipeline.

:func:`match_record` is the single function the rest of the codebase calls
to match a :class:`MarcRecord` against the indexed NYPL corpus. The flow
is intentionally small:

1. Retrieve year-bucketed candidates from the lookup.
2. Build one :class:`ScorerContext` for the record (one stopword/stemmer
   resolution per record, not per candidate).
3. For each candidate run all scorers, keep the best Evidence per scorer
   from the bounded field-pair permutations, and combine.
4. Sort by calibrated score, apply the configured floor, and return the
   top result plus up to ``top_k - 1`` runners-up.
"""

from collections.abc import Sequence

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import calibrate
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairings import publisher_pairings
from pd_matcher.match.pairings import title_pairings
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


def _score_candidate(
    marc: MarcRecord,
    candidate: IndexedNyplRegRecord,
    ctx: ScorerContext,
    combiner: Combiner,
    calibrator: PlattCalibrator | None,
) -> CandidateMatch:
    winning: list[Evidence] = []
    losing: list[Evidence] = []

    winning.append(score_lccn(marc.lccn, candidate, ctx))
    winning.append(score_isbn(marc.isbns, candidate, ctx))

    title_evidences = tuple(
        score_title(marc_value, nypl_value, ctx)
        for marc_value, nypl_value in title_pairings(marc, candidate)
    )
    title_best, title_losers = _select_best(title_evidences)
    winning.append(title_best)
    losing.extend(title_losers)

    winning.append(score_author(marc.main_author, candidate.author_name, ctx))

    publisher_evidences = tuple(
        score_publisher(marc_value, nypl_value, ctx)
        for marc_value, nypl_value in publisher_pairings(marc, candidate)
    )
    publisher_best, publisher_losers = _select_best(publisher_evidences)
    winning.append(publisher_best)
    losing.extend(publisher_losers)

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
    candidates = list(lookup.candidates_for_year(marc.publication_year, config.year_window))
    if not candidates:
        return MatchResult(
            marc_control_id=marc.control_id,
            best=None,
            alternates=(),
            candidates_considered=0,
        )
    ctx = _build_context(marc, idf, config)
    scored = [
        _score_candidate(marc, candidate, ctx, combiner, calibrator) for candidate in candidates
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
