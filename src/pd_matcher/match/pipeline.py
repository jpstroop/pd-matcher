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

from msgspec import Struct

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.claimant_routing import AUTHOR_SCORER
from pd_matcher.match.claimant_routing import PUBLISHER_SCORER
from pd_matcher.match.claimant_routing import compute_routing
from pd_matcher.match.claimant_routing import is_blank_publisher
from pd_matcher.match.claimant_routing import value_key
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
from pd_matcher.match.scorers.extent import score_extent
from pd_matcher.match.scorers.isbn import score_isbn
from pd_matcher.match.scorers.lccn import score_lccn
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.match.scorers.organization import looks_like_organization
from pd_matcher.match.scorers.title import prepare_cross_field_stems
from pd_matcher.match.scorers.title import score_title
from pd_matcher.match.scorers.volume import score_volume
from pd_matcher.match.scorers.year import score_year
from pd_matcher.match.signals.corroboration import has_no_corroboration
from pd_matcher.match.signals.translation import is_translation_signal
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.publishers import get_default_alias_index
from pd_matcher.normalize.stemming import stemmer_for
from pd_matcher.normalize.stopwords import load_stopwords

_DEFAULT_LANGUAGE: str = "eng"
_FIXED_SOURCE: tuple[str, str] = ("", "")
_TRANSLATION_AUTHOR_MULTIPLIER: float = 0.5
_CORROBORATION_THRESHOLD: float = 50.0
_TITLE_ISOLATION_MULTIPLIER: float = 0.3
_NON_CORROBORATING_SCORERS: frozenset[str] = frozenset(
    {
        "year.delta",  # many records share a year; year alone is too weak a corroboration signal
    }
)
_MAX_NAME_SCORE: float = 100.0
_GENUINE_PUBLISHER_CCE: str = "publisher_names"
_ROUTED_TO_AUTHOR_SOURCE: tuple[str, str] = ("", "routed → name.author")
_ROUTED_TO_PUBLISHER_SOURCE: tuple[str, str] = ("", "routed → name.publisher")
_BLANK_PUBLISHER_SOURCE: tuple[str, str] = ("", "skipped: no CCE publisher")


def _build_context(
    marc: MarcRecord,
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
    config: MatchingConfig,
) -> ScorerContext:
    language = marc.language_code or _DEFAULT_LANGUAGE
    stopwords = load_stopwords(language)
    stemmer = stemmer_for(language)
    cross_field_values = tuple(
        value
        for value in (marc.publisher, marc.publication_place, marc.statement_of_responsibility)
        if value
    )
    return ScorerContext(
        language=language,
        stopwords=stopwords,
        stemmer=stemmer,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        config=config,
        publisher_alias_index=get_default_alias_index(),
        cross_field_title_stems=prepare_cross_field_stems(
            cross_field_values, language, stopwords.title, stemmer
        ),
    )


def _argmax_by_score(evidences: Sequence[Evidence]) -> int:
    """Return the index of the highest-scoring Evidence.

    A ``skipped`` Evidence scores ``-1.0`` so a real (even zero) score always
    wins over a skipped one, and ties keep the first (lowest-index) entry.
    """
    best_index = 0
    best_score = evidences[0].score if not evidences[0].skipped else -1.0
    for index in range(1, len(evidences)):
        current = evidences[index]
        current_score = current.score if not current.skipped else -1.0
        if current_score > best_score:
            best_score = current_score
            best_index = index
    return best_index


def _select_best(evidences: Sequence[Evidence]) -> tuple[int, Evidence, tuple[Evidence, ...]]:
    """Return the highest-scoring Evidence's index plus the losers in input order."""
    best_index = _argmax_by_score(evidences)
    losers = tuple(ev for index, ev in enumerate(evidences) if index != best_index)
    return best_index, evidences[best_index], losers


_GroupScorer = Callable[[str | None, str | None, ScorerContext], Evidence]


def _best_title_element_evidence(
    marc_value: str | None,
    cce_values: tuple[str, ...],
    candidate: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> Evidence:
    """Best-of-element title scoring that threads the candidate's title script.

    Like :func:`_best_element_evidence` but specialised for the title scorer:
    each element that equals the candidate's canonical title reuses its
    precomputed :attr:`~pd_matcher.models.IndexedNyplRegRecord.title_script`,
    so the script-mismatch guard does not re-derive the dominant script per
    candidate.
    """
    if not cce_values:
        return score_title(marc_value, None, ctx)
    scored = tuple(
        score_title(
            marc_value,
            cce_value,
            ctx,
            nypl_title_script=(candidate.title_script if cce_value == candidate.title else None),
        )
        for cce_value in cce_values
    )
    return scored[_argmax_by_score(scored)]


def _apply_title_isolation_multiplier(winning: list[Evidence], title_index: int) -> None:
    """Set the title scorer's ``weight_multiplier`` to the isolation value when
    a strong title match is the only meaningful signal.

    Mutates ``winning`` in place. The multiplier fires only when:

    * A title Evidence was emitted and is not ``skipped``.
    * The title's own score reaches ``_CORROBORATION_THRESHOLD`` — a weak
      title doesn't justify a downweight, and downweighting it can spuriously
      raise the combined average when the title is scoring below the
      non-title scorers' mean.
    * No non-title scorer outside :data:`_NON_CORROBORATING_SCORERS` reaches
      the same threshold. The year scorer is excluded because many records
      share a year and a year-only match doesn't distinguish a real pair
      from a coincidence.
    """
    if title_index >= len(winning):
        return
    title_evidence = winning[title_index]
    if title_evidence.skipped:
        return
    if title_evidence.score < _CORROBORATION_THRESHOLD:
        return
    other_evidences = tuple(
        ev
        for index, ev in enumerate(winning)
        if index != title_index and ev.scorer not in _NON_CORROBORATING_SCORERS
    )
    if has_no_corroboration(other_evidences, _CORROBORATION_THRESHOLD):
        winning[title_index] = _with_multiplier(title_evidence, _TITLE_ISOLATION_MULTIPLIER)


def _with_multiplier(evidence: Evidence, multiplier: float) -> Evidence:
    """Return a copy of ``evidence`` with ``weight_multiplier`` set."""
    return Evidence(
        scorer=evidence.scorer,
        score=evidence.score,
        max=evidence.max,
        skipped=evidence.skipped,
        decisive=evidence.decisive,
        features=evidence.features,
        weight_multiplier=multiplier,
    )


class _PairingResult(Struct, frozen=True, forbid_unknown_fields=True):
    """One name pairing's best-of-element outcome, retained for routing.

    ``cce_value`` is the winning CCE element string (``None`` when the pairing
    had no CCE values and scored ``skipped``); ``cce_key`` is its normalized
    token-set key, matched against a :class:`RoutingDecision`'s routed-away key
    sets. ``evidence`` is the best Evidence across the pairing's elements.
    """

    pairing: CompiledPairing
    cce_value: str | None
    cce_key: frozenset[str]
    evidence: Evidence


def _skipped_name_evidence(scorer_name: str) -> Evidence:
    """Return a skipped Evidence for a name group emptied by routing / gating."""
    return Evidence(
        scorer=scorer_name,
        score=0.0,
        max=_MAX_NAME_SCORE,
        skipped=True,
        decisive=False,
        features=(),
    )


def _best_pairing_result(
    scorer: _GroupScorer,
    marc_value: str | None,
    cce_values: tuple[str, ...],
    ctx: ScorerContext,
    pairing: CompiledPairing,
) -> _PairingResult:
    """Score ``marc_value`` against each CCE element and keep the best result.

    A ``best``-combined CCE field surfaces one element per co-claimant, so the
    correct name in a "Putnam James D. Horan" list is scored on its own instead
    of diluted into one blob. An empty tuple (the CCE field is absent) scores
    once against ``None`` to emit the pairing's skipped Evidence with no winning
    value. A scalar or ``first``/``join`` field yields a 1-tuple, so the common
    case makes exactly one scorer call, matching the pre-routing cost.
    """
    if not cce_values:
        return _PairingResult(
            pairing=pairing,
            cce_value=None,
            cce_key=frozenset(),
            evidence=scorer(marc_value, None, ctx),
        )
    scored = tuple((value, scorer(marc_value, value, ctx)) for value in cce_values)
    best_value, best_evidence = scored[_argmax_by_score(tuple(ev for _, ev in scored))]
    return _PairingResult(
        pairing=pairing,
        cce_value=best_value,
        cce_key=value_key(best_value),
        evidence=best_evidence,
    )


def _name_pairing_results(
    pairings: tuple[CompiledPairing, ...],
    marc: MarcRecord,
    candidate: IndexedNyplRegRecord,
    ctx: ScorerContext,
    scorer: _GroupScorer,
) -> tuple[_PairingResult, ...]:
    """Return the best-of-element result for every pairing in a name group."""
    return tuple(
        _best_pairing_result(
            scorer, pairing.marc_accessor(marc), pairing.cce_accessor(candidate), ctx, pairing
        )
        for pairing in pairings
    )


def _apply_blank_publisher_gate(
    results: tuple[_PairingResult, ...],
    candidate: IndexedNyplRegRecord,
) -> tuple[tuple[_PairingResult, ...], bool]:
    """Skip publisher pairings that compare a person against an empty publisher.

    When the CCE record carries no publisher of its own (issue #86, direction
    A2), the only publisher pairings left are cross-field fallbacks against the
    ``author_name`` / ``claimants`` slots. A person comparand there fabricates a
    ``name.publisher = 0.0`` that penalizes a genuine match, so it is forced to
    ``skipped`` (dropped from the combiner) unless the comparand
    :func:`~pd_matcher.match.scorers.organization.looks_like_organization` — a
    corporate self-publisher (``Knopf``) is legitimately the publisher and is
    kept. The genuine ``publisher_names`` pairing is never gated. Returns the
    (possibly transformed) results and whether the gate fired.
    """
    if not is_blank_publisher(candidate.publisher_names):
        return results, False
    gated: list[_PairingResult] = []
    fired = False
    for result in results:
        if (
            result.cce_value is not None
            and result.pairing.cce_name != _GENUINE_PUBLISHER_CCE
            and not looks_like_organization(result.cce_value)
        ):
            gated.append(
                _PairingResult(
                    pairing=result.pairing,
                    cce_value=result.cce_value,
                    cce_key=result.cce_key,
                    evidence=_skipped_name_evidence(PUBLISHER_SCORER),
                )
            )
            fired = True
        else:
            gated.append(result)
    return tuple(gated), fired


def _finalize_name_group(
    results: tuple[_PairingResult, ...],
    winning: list[Evidence],
    losing: list[Evidence],
    winning_sources: list[tuple[str, str]],
    *,
    dropped_keys: frozenset[frozenset[str]],
    routed_source: tuple[str, str],
    gate_source: tuple[str, str],
    gate_fired: bool,
    skipped_scorer: str,
) -> None:
    """Keep the best routed-eligible Evidence for one name group.

    Pairings whose winning CCE value was routed to the *other* group (its key
    is in ``dropped_keys``) are excluded so a shared ``publisher==claimant``
    value contributes evidence at most once. The best remaining non-skipped
    Evidence represents the group; when routing or the blank-publisher gate
    leaves nothing, a skipped Evidence is emitted (the group goes missing) and
    the recorded source names why — ``routed_source`` when a routing drop caused
    it, ``gate_source`` when the blank-publisher gate did, else the empty
    sentinel. An empty ``results`` (the group has no pairings) appends nothing,
    matching the pre-routing early return.
    """
    if not results:
        return
    kept = [result for result in results if not (result.cce_key and result.cce_key in dropped_keys)]
    eligible = [result for result in kept if not result.evidence.skipped]
    if eligible:
        best = eligible[_argmax_by_score(tuple(result.evidence for result in eligible))]
        winning.append(best.evidence)
        winning_sources.append((best.pairing.marc_name, best.pairing.cce_name))
        losing.extend(result.evidence for result in results if result is not best)
        return
    winning.append(_skipped_name_evidence(skipped_scorer))
    losing.extend(result.evidence for result in results)
    if any(result.cce_key and result.cce_key in dropped_keys for result in results):
        winning_sources.append(routed_source)
    elif gate_fired:
        winning_sources.append(gate_source)
    else:
        winning_sources.append(_FIXED_SOURCE)


def _score_name_groups(
    pairings: CompiledPairings,
    marc: MarcRecord,
    candidate: IndexedNyplRegRecord,
    ctx: ScorerContext,
    winning: list[Evidence],
    losing: list[Evidence],
    winning_sources: list[tuple[str, str]],
) -> None:
    """Score the author and publisher groups with issue #86 claimant routing.

    Both groups are scored per pairing, then a shared ``publisher==claimant``
    value that clears the floor is routed to the better-matching group and
    dropped from the other (direction A1); when the CCE publisher slot is empty,
    person comparands in the publisher group are gated out (direction A2). The
    author Evidence is appended before the publisher Evidence, preserving the
    combiner's field order.
    """
    author_results = _name_pairing_results(pairings.author, marc, candidate, ctx, score_author)
    publisher_results = _name_pairing_results(
        pairings.publisher, marc, candidate, ctx, score_publisher
    )
    publisher_results, gate_fired = _apply_blank_publisher_gate(publisher_results, candidate)
    routing = compute_routing(marc, candidate.publisher_names, candidate.claimants, ctx)
    _finalize_name_group(
        author_results,
        winning,
        losing,
        winning_sources,
        dropped_keys=routing.publisher_routed,
        routed_source=_ROUTED_TO_PUBLISHER_SOURCE,
        gate_source=_FIXED_SOURCE,
        gate_fired=False,
        skipped_scorer=AUTHOR_SCORER,
    )
    _finalize_name_group(
        publisher_results,
        winning,
        losing,
        winning_sources,
        dropped_keys=routing.author_routed,
        routed_source=_ROUTED_TO_AUTHOR_SOURCE,
        gate_source=_BLANK_PUBLISHER_SOURCE,
        gate_fired=gate_fired,
        skipped_scorer=PUBLISHER_SCORER,
    )


def _finalize_group(
    pairings: tuple[CompiledPairing, ...],
    evidences: tuple[Evidence, ...],
    winning: list[Evidence],
    losing: list[Evidence],
    winning_sources: list[tuple[str, str]],
) -> None:
    """Keep the best Evidence for one scorer group and record its source.

    The combiner keys on exactly one Evidence per scorer tag, so the best
    Evidence (the highest-scoring pairing) is appended to ``winning`` and the
    rest to ``losing`` for audit. The winning pairing's ``(marc_name,
    cce_name)`` is appended to ``winning_sources`` so callers can surface which
    composed-field pair produced the kept Evidence (vital for diagnosing
    cross-pairings that score non-zero against fuzzy noise).
    """
    best_index, best, losers = _select_best(evidences)
    winning.append(best)
    losing.extend(losers)
    winning_sources.append((pairings[best_index].marc_name, pairings[best_index].cce_name))


def _score_title_group(
    pairings: tuple[CompiledPairing, ...],
    marc: MarcRecord,
    candidate: IndexedNyplRegRecord,
    ctx: ScorerContext,
    winning: list[Evidence],
    losing: list[Evidence],
    winning_sources: list[tuple[str, str]],
) -> None:
    """Score the title group, feeding the candidate's precomputed title script.

    Like :func:`_score_group` but specialised for the title scorer: when a
    pairing's CCE comparand is the candidate's canonical title, its precomputed
    :attr:`~pd_matcher.models.IndexedNyplRegRecord.title_script` is threaded
    into :func:`score_title` so the script-mismatch guard does not re-derive the
    dominant script per candidate. For any other CCE comparand (e.g. the renewal
    title) the script is left for the scorer to derive, so the score is
    byte-identical to the unspecialised path.
    """
    if not pairings:
        return
    evidences = tuple(
        _best_title_element_evidence(
            pairing.marc_accessor(marc), pairing.cce_accessor(candidate), candidate, ctx
        )
        for pairing in pairings
    )
    _finalize_group(pairings, evidences, winning, losing, winning_sources)


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
    sources: list[tuple[str, str]] = []

    winning.append(score_lccn(marc.lccn, candidate, ctx))
    sources.append(_FIXED_SOURCE)
    winning.append(score_isbn(marc.isbns, candidate, ctx))
    sources.append(_FIXED_SOURCE)

    title_index = len(winning)
    _score_title_group(pairings.title, marc, candidate, ctx, winning, losing, sources)
    author_index = len(winning)
    _score_name_groups(pairings, marc, candidate, ctx, winning, losing, sources)
    if author_index < len(winning) and is_translation_signal(candidate):
        winning[author_index] = _with_multiplier(
            winning[author_index], _TRANSLATION_AUTHOR_MULTIPLIER
        )

    winning.append(score_year(marc.publication_year, candidate.reg_year, ctx))
    sources.append(_FIXED_SOURCE)
    winning.append(score_edition(marc.edition, candidate.edition, ctx))
    sources.append(_FIXED_SOURCE)
    winning.append(score_extent(marc.extent, candidate.desc, ctx))
    sources.append(_FIXED_SOURCE)
    winning.append(score_volume(marc, candidate, ctx))
    sources.append(_FIXED_SOURCE)

    _apply_title_isolation_multiplier(winning, title_index)

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
        evidence_sources=tuple(sources),
    )


def match_record(
    marc: MarcRecord,
    *,
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
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
        idf: Pre-built title :class:`IdfTable`.
        author_idf: Pre-built author-name :class:`IdfTable`.
        publisher_idf: Pre-built publisher-name :class:`IdfTable`.
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
    ctx = _build_context(marc, idf, author_idf, publisher_idf, config)
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
