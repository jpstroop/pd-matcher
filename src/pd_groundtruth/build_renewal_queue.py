"""Renewal-first review-queue builder for scenario-4 (renewal-only) candidates.

The registration queue (:mod:`pd_groundtruth.build_queue`) proposes
``(MARC, registration)`` pairs; this builder is its renewal-side sibling. It is
*renewal-first*: the cheap renewal search runs before the expensive
registration check, so the registration matcher only ever touches the small
fraction of pool books that actually have a renewal.

For every in-scope MARC record in the pool:

1. **Renewal search (cheap, first):** renewal candidates are retrieved with
   :meth:`pd_matcher.index.lookup.NyplIndexLookup.candidates_for_renewal`
   (``odat``-year bucketed) and scored with the production title / author /
   claimants / year scorers and the weighted-mean combiner. The single best
   renewal ``R`` is kept; if none clears ``min_score`` the MARC is *not* a
   renewal-haver and is skipped immediately. This filter is what makes the
   build fast.
2. **Registration presence check, limited to ``R``'s ``odat`` year:**
   registration candidates are retrieved for ``R``'s original-registration year
   (``R.odat`` — *not* the MARC's publication year) via
   :meth:`pd_matcher.index.lookup.NyplIndexLookup.candidates_in_year` and scored
   with the learned scorer. If any registration scores at or above
   ``reg_min_score`` a registration exists, so the book is excluded (it is
   scenario 2 or 3 — scenario 3's verified renewal link is sourced from the
   vault, not here).
3. **Scenario 4 (renewal-only):** when no registration clears the floor in the
   ``odat`` year the renewal ``R`` is emitted as a ``pairing_type="renewal"``
   pair carrying a scenario-4 ``audit_note``. This is the labeling candidate.

A full-corpus join analysis confirmed the design: a renewal's normalized
``oreg`` + ``odat`` points to at most one registration in 99.4%+ of cases, so
the matcher's content scoring resolves the rare many-to-one via title without
special-casing. The registration check here only *routes* a book into scenario
4 versus excludes it; the renewal-match labels the queue collects are
human-verified, never seeded from unverified matcher output.

The registration arm uses the learned scorer by default, so its model artifact
must be present beside the index; its absence fails the build loudly via
:func:`pd_matcher.match.combiners.build_combiner`. The renewal arm stays on the
weighted-mean combiner because the renewal pathway is untrained.

A renewal pair's ``nypl_uuid`` column carries the renewal record's ``entry_id``
rather than a registration UUID — the column is polymorphic by
``pairing_type``. The renewal record's fields populate both the denormalized
``cce_*`` columns (so the labels table and search keep working) and the
``cce_renewal_*`` columns (so the renewal card renders the renewal directly).

MARC records already present in the target review DB — registration or renewal
— are skipped, so this command can append a renewal pass onto an existing
registration queue without duplicating a MARC.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

from msgspec import Struct
from msgspec.json import encode as json_encode
from msgspec.structs import replace

from pd_groundtruth.build_queue import _decade_of
from pd_groundtruth.build_queue import _iter_language_dirs
from pd_groundtruth.build_queue import _language_of
from pd_groundtruth.review_db import PAIRING_RENEWAL
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import band_of
from pd_groundtruth.vault_pair_resolver import (
    AUTHOR_IDF_CACHE_NAME as _SHARED_AUTHOR_IDF_CACHE_NAME,
)
from pd_groundtruth.vault_pair_resolver import IDF_CACHE_NAME as _SHARED_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import (
    PUBLISHER_IDF_CACHE_NAME as _SHARED_PUBLISHER_IDF_CACHE_NAME,
)
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import calibrate
from pd_matcher.match.combiners.calibrator import load_calibrator
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import load_or_build_author_idf
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.idf import load_or_build_publisher_idf
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.pipeline import _score_candidate
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.match.scorers.title import score_title
from pd_matcher.match.scorers.year import score_year
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.parsers.marc import iter_marc_records

_LOGGER = getLogger(__name__)

_CALIBRATOR_NAME: str = "calibrator.msgpack"
_FILL_LOG_INTERVAL: int = 250
_SCANNED_LOG_INTERVAL: int = 5000
_REG_YEAR_WINDOW: int = 0

SOURCE_RENEWAL: str = "renewal"

_LEARNED_SCORER: str = "learned"

_SCORER_TITLE: str = "title"
_SCORER_AUTHOR: str = "author"
_SCORER_CLAIMANTS: str = "claimants"
_SCORER_YEAR: str = "year"


class RenewalScore(Struct, frozen=True, forbid_unknown_fields=True):
    """The calibrated confidence and per-scorer evidence for one renewal pairing.

    ``calibrated`` is the weighted-mean combiner's calibrated score in
    ``[0, 1]``; ``evidence`` maps each fired scorer (``title`` / ``author`` /
    ``claimants`` / ``year``) to its normalized ``[0, 1]`` reading, omitting
    skipped scorers so the card renders only the bars that contributed.
    """

    calibrated: float
    evidence: dict[str, float]


RenewalScoreFn = Callable[[MarcRecord, NyplRenRecord], RenewalScore]

RegPresentFn = Callable[[MarcRecord, int | None], bool]


def _scenario_4_note(odat_year: int | None) -> str:
    """Return the scenario-4 ``audit_note`` naming the unchecked ``odat`` year."""
    return f"scenario 4: renewal-only (no registration in odat year {odat_year})"


def _best_evidence(candidates: tuple[Evidence, ...]) -> Evidence:
    """Return the highest-scoring non-skipped Evidence, or the first if all skip."""
    best = candidates[0]
    best_score = best.score if not best.skipped else -1.0
    for evidence in candidates[1:]:
        score = evidence.score if not evidence.skipped else -1.0
        if score > best_score:
            best_score = score
            best = evidence
    return best


def score_renewal(
    marc: MarcRecord,
    renewal: NyplRenRecord,
    ctx: ScorerContext,
    combiner: Combiner,
    calibrator: PlattCalibrator | None,
) -> RenewalScore:
    """Score one MARC↔renewal pairing with the production scorers and combiner.

    Mirrors ``scripts/renewal_gate_measure.py``: the renewal's title / author
    are scored against the MARC's title and author fields (best pairing kept
    per group), the renewal's claimants against the MARC publisher, and the
    renewal's original-registration year (``odat``) against the MARC year. The
    weighted-mean combiner is applied to the four Evidence readings and the
    calibrator (when present) maps the raw score exactly as the registration
    pipeline does.
    """
    title_evidence = _best_evidence(
        tuple(
            score_title(marc_value, renewal.title, ctx)
            for marc_value in (marc.title, marc.title_main)
            if marc_value
        )
        or (score_title(marc.title, renewal.title, ctx),)
    )
    author_evidence = _best_evidence(
        tuple(
            score_author(marc_value, renewal.author, ctx)
            for marc_value in (marc.main_author, marc.statement_of_responsibility)
            if marc_value
        )
        or (score_author(marc.main_author, renewal.author, ctx),)
    )
    claimants_evidence = score_publisher(marc.publisher, renewal.claimants, ctx)
    renewal_year = renewal.odat.year if renewal.odat is not None else None
    year_evidence = score_year(marc.publication_year, renewal_year, ctx)
    combined = combiner.combine(
        (title_evidence, author_evidence, claimants_evidence, year_evidence)
    )
    if calibrator is not None:
        combined = CombinedScore(raw=combined.raw, calibrated=calibrate(combined.raw, calibrator))
    payload: dict[str, float] = {}
    for name, evidence in (
        (_SCORER_TITLE, title_evidence),
        (_SCORER_AUTHOR, author_evidence),
        (_SCORER_CLAIMANTS, claimants_evidence),
        (_SCORER_YEAR, year_evidence),
    ):
        if not evidence.skipped:
            payload[name] = evidence.normalized
    return RenewalScore(calibrated=combined.calibrated, evidence=payload)


def best_renewal(
    marc: MarcRecord,
    candidates: Iterable[NyplRenRecord],
    score_fn: RenewalScoreFn,
) -> tuple[NyplRenRecord, RenewalScore] | None:
    """Return the highest-calibrated ``(renewal, score)`` pair, or ``None`` if none.

    Scores every candidate via ``score_fn`` and keeps the one with the largest
    calibrated score. ``None`` is returned when ``candidates`` is empty.
    """
    best: tuple[NyplRenRecord, RenewalScore] | None = None
    for renewal in candidates:
        score = score_fn(marc, renewal)
        if best is None or score.calibrated > best[1].calibrated:
            best = (renewal, score)
    return best


def _build_renewal_pair_insert(
    marc: MarcRecord,
    renewal: NyplRenRecord,
    score: RenewalScore,
    *,
    language: str,
    band: str,
    audit_note: str,
) -> PairInsert:
    """Assemble a ``pairing_type="renewal"`` :class:`PairInsert` for one pairing.

    The renewal record is the CCE side: its fields populate both the
    denormalized ``cce_*`` columns and the ``cce_renewal_*`` columns, and the
    renewal ``entry_id`` is stored in ``nypl_uuid``. ``cce_was_renewed`` is
    ``True`` by construction (a renewal record exists); ``cce_regnum`` /
    ``cce_publishers`` are ``None`` because no registration was matched.
    """
    odat_year = renewal.odat.year if renewal.odat is not None else None
    rdat_iso = renewal.rdat.isoformat() if renewal.rdat is not None else None
    return PairInsert(
        language=language,
        decade=_decade_of(marc.publication_year),
        score=score.calibrated,
        band=band,
        source=SOURCE_RENEWAL,
        pairing_type=PAIRING_RENEWAL,
        marc_control_id=marc.control_id,
        marc_json=json_encode(marc).decode("utf-8"),
        marc_title=marc.title,
        marc_author=marc.main_author or marc.statement_of_responsibility,
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        nypl_uuid=renewal.entry_id,
        cce_title=renewal.title,
        cce_author=renewal.author,
        cce_publishers=None,
        cce_claimants=renewal.claimants,
        cce_reg_year=odat_year,
        cce_was_renewed=True,
        cce_regnum=None,
        evidence_json=json_encode(score.evidence).decode("utf-8"),
        cce_renewal_id=renewal.id,
        cce_renewal_oreg=renewal.oreg,
        cce_renewal_rdat=rdat_iso,
        cce_renewal_author=renewal.author,
        cce_renewal_title=renewal.title,
        cce_renewal_claimants=renewal.claimants,
        cce_renewal_new_matter=renewal.new_matter,
        audit_note=audit_note,
    )


class RenewalBuildSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Result of one :func:`build_renewal_queue` invocation.

    ``records_scanned`` counts distinct pool MARCs considered (after skipping
    those already in the target DB); ``renewal_havers`` counts those whose best
    renewal cleared ``min_score`` and therefore reached the registration check;
    ``reg_excluded`` counts renewal-havers dropped because a registration was
    found in the renewal's ``odat`` year; ``scenario4_written`` counts the
    renewal-only pairs emitted (``renewal_havers == reg_excluded +
    scenario4_written``).
    """

    records_scanned: int
    renewal_havers: int
    reg_excluded: int
    scenario4_written: int


def _load_calibrator(parent: Path) -> PlattCalibrator | None:
    """Load a Platt calibrator from ``<parent>/calibrator.msgpack`` if present."""
    candidate = parent / _CALIBRATOR_NAME
    if not candidate.exists():
        return None
    return load_calibrator(candidate)


def _make_score_fn(
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
    config: MatchingConfig,
    combiner: Combiner,
    calibrator: PlattCalibrator | None,
) -> RenewalScoreFn:
    """Build a ``(marc, renewal) -> RenewalScore`` closure.

    The per-MARC :class:`ScorerContext` is expensive (stopwords + stemmer), so
    it is cached and reused across the candidates of one MARC; the cache holds
    only the most recent MARC since candidates are scored consecutively.
    """
    cache: dict[str, ScorerContext] = {}

    def score_fn(marc: MarcRecord, renewal: NyplRenRecord) -> RenewalScore:
        ctx = cache.get(marc.control_id)
        if ctx is None:
            ctx = _build_context(marc, idf, author_idf, publisher_idf, config)
            cache.clear()
            cache[marc.control_id] = ctx
        return score_renewal(marc, renewal, ctx, combiner, calibrator)

    return score_fn


def _make_reg_present_fn(
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
    calibrator: PlattCalibrator | None,
    combiner: Combiner,
    pairings: CompiledPairings,
    reg_min_calibrated: float,
) -> RegPresentFn:
    """Build a ``(marc, year) -> bool`` registration-presence closure.

    The returned predicate retrieves registration candidates for the EXPLICIT
    ``year`` (a renewal's ``odat`` year) via
    :meth:`pd_matcher.index.lookup.NyplIndexLookup.candidates_in_year` — never
    ``marc.publication_year`` — scores each with the learned scorer, and reports
    whether any clears ``reg_min_calibrated``. A ``None`` year (a renewal
    without an ``odat``) cannot be checked and reports ``False``. The per-MARC
    :class:`ScorerContext` is cached exactly like :func:`_make_score_fn`.
    """
    cache: dict[str, ScorerContext] = {}

    def reg_present(marc: MarcRecord, year: int | None) -> bool:
        if year is None:
            return False
        ctx = cache.get(marc.control_id)
        if ctx is None:
            ctx = _build_context(marc, idf, author_idf, publisher_idf, config)
            cache.clear()
            cache[marc.control_id] = ctx
        for candidate in lookup.candidates_in_year(marc, year, _REG_YEAR_WINDOW):
            match = _score_candidate(marc, candidate, ctx, combiner, calibrator, pairings)
            if match.combined.calibrated >= reg_min_calibrated:
                return True
        return False

    return reg_present


def _iter_pool_records(pool: Path) -> Iterator[MarcRecord]:
    """Yield every MARC record from the ``<pool>/<lang>/*.xml`` shards."""
    for _language, language_dir in _iter_language_dirs(pool):
        for shard in sorted(language_dir.glob("*.xml")):
            yield from iter_marc_records(shard)


def build_renewal_queue(
    *,
    pool: Path,
    index_path: Path,
    out_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    min_score: float,
    reg_min_score: float,
    reg_scorer: str,
) -> RenewalBuildSummary:
    """Build (or append) scenario-4 renewal-only pairs into ``out_path``.

    Loads the IDF caches and calibrator beside ``index_path``, opens the CCE
    index and the review DB, and runs the renewal-first pipeline over every pool
    MARC not already queued: the cheap renewal search filters to renewal-havers,
    then the registration check (``reg_scorer`` with a ``reg_min_score`` floor,
    limited to the best renewal's ``odat`` year) excludes books that have a
    registration. Survivors are emitted as ``pairing_type="renewal"`` pairs with
    a scenario-4 ``audit_note``.

    Args:
        pool: Root directory whose ``<lang>/*.xml`` shards form the pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination SQLite review DB; renewal pairs are appended and
            MARCs already present (any pairing type) are skipped.
        matching_config: Active matcher config; supplies the year window used
            for renewal retrieval and the scoring weights.
        pairing_config: Field-pairing config driving the registration matcher's
            title/author/publisher scorer groups.
        min_score: Renewal-arm score floor on the 0-100 scale; a MARC whose best
            renewal scores below it is not a renewal-haver and is skipped before
            the registration check ever runs.
        reg_min_score: Registration-arm score floor on the 0-100 scale; a
            registration at or above it in the renewal's ``odat`` year excludes
            the book (a registration exists).
        reg_scorer: Combiner used by the registration arm
            (``learned`` | ``weighted_mean``); the learned model artifact must
            exist beside ``index_path`` when ``learned`` is selected.

    Returns:
        A populated :class:`RenewalBuildSummary`.
    """
    idf = load_or_build_idf(
        index_path.parent / _SHARED_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    author_idf = load_or_build_author_idf(
        index_path.parent / _SHARED_AUTHOR_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    publisher_idf = load_or_build_publisher_idf(
        index_path.parent / _SHARED_PUBLISHER_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    calibrator = _load_calibrator(index_path.parent)
    renewal_combiner = build_combiner(matching_config, learned_model_dir=None)
    reg_config = replace(matching_config, scorer=reg_scorer)
    learned_model_dir = index_path.parent if reg_scorer == _LEARNED_SCORER else None
    reg_combiner = build_combiner(reg_config, learned_model_dir=learned_model_dir)
    pairings = compile_pairings(pairing_config)
    window = matching_config.year_window
    min_calibrated = min_score / 100.0
    reg_min_calibrated = reg_min_score / 100.0
    score_fn = _make_score_fn(
        idf, author_idf, publisher_idf, matching_config, renewal_combiner, calibrator
    )
    _LOGGER.info(
        "renewal queue start: pool=%s window=%d min_score=%.1f reg_min_score=%.1f "
        "reg_scorer=%s calibrator=%s",
        pool,
        window,
        min_score,
        reg_min_score,
        reg_scorer,
        "yes" if calibrator is not None else "no",
    )
    scanned = 0
    renewal_havers = 0
    reg_excluded = 0
    scenario4_written = 0
    with NyplIndexLookup(index_path) as lookup, ReviewDb.connect(out_path) as db:
        reg_present_fn = _make_reg_present_fn(
            lookup,
            matching_config,
            idf,
            author_idf,
            publisher_idf,
            calibrator,
            reg_combiner,
            pairings,
            reg_min_calibrated,
        )
        seen = {marc_id for marc_id, _uuid in db.pair_keys()}
        # Commit incrementally so an interrupted long build keeps its progress:
        # insert_pair does not commit, and ReviewDb.__exit__ only commits on a
        # clean exit, so without this a Ctrl-C (KeyboardInterrupt) would discard
        # every inserted row. The finally clause flushes the final partial batch.
        try:
            for marc in _iter_pool_records(pool):
                if marc.control_id in seen:
                    continue
                seen.add(marc.control_id)
                scanned += 1
                if scanned % _SCANNED_LOG_INTERVAL == 0:
                    db.commit()
                    _LOGGER.info(
                        "renewal queue: scanned=%d renewal_havers=%d "
                        "reg_excluded=%d scenario4_written=%d",
                        scanned,
                        renewal_havers,
                        reg_excluded,
                        scenario4_written,
                    )
                best = best_renewal(marc, lookup.candidates_for_renewal(marc, window), score_fn)
                if best is None:
                    continue
                renewal, score = best
                if score.calibrated < min_calibrated:
                    continue
                renewal_havers += 1
                odat_year = renewal.odat.year if renewal.odat is not None else None
                if reg_present_fn(marc, odat_year):
                    reg_excluded += 1
                    continue
                pair = _build_renewal_pair_insert(
                    marc,
                    renewal,
                    score,
                    language=_language_of(marc),
                    band=band_of(score.calibrated),
                    audit_note=_scenario_4_note(odat_year),
                )
                db.insert_pair(pair)
                scenario4_written += 1
                if scenario4_written % _FILL_LOG_INTERVAL == 0:
                    db.commit()
        finally:
            db.commit()
    _LOGGER.info(
        "renewal queue complete: scanned=%d renewal_havers=%d reg_excluded=%d scenario4_written=%d",
        scanned,
        renewal_havers,
        reg_excluded,
        scenario4_written,
    )
    return RenewalBuildSummary(
        records_scanned=scanned,
        renewal_havers=renewal_havers,
        reg_excluded=reg_excluded,
        scenario4_written=scenario4_written,
    )


__all__ = [
    "SOURCE_RENEWAL",
    "RegPresentFn",
    "RenewalBuildSummary",
    "RenewalScore",
    "RenewalScoreFn",
    "best_renewal",
    "build_renewal_queue",
    "score_renewal",
]
