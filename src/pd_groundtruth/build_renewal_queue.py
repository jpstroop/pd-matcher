"""Renewal-first review-queue builder for scenario-4 (renewal-only) candidates.

The registration queue (:mod:`pd_groundtruth.build_queue`) proposes
``(MARC, registration)`` pairs; this builder is its renewal-side sibling. It is
*renewal-first*: the cheap renewal search runs before the cheap join check, so
the build only ever emits books whose best renewal is not already joined to a
registration we hold.

A scenario-4 candidate is a renewal that is **not joined to any registration in
our index** — its underlying work has no determined registration, so the
renewal alone needs human review. A renewal whose original-registration the
index already carries is *joined*: its work is already determined and it must
never reach this queue.

The reliable join signal is baked into the index at build time. Each
:class:`pd_matcher.models.IndexedNyplRegRecord` carries ``was_renewed`` and, when
renewed, the ``renewal_id`` of the :class:`pd_matcher.models.NyplRenRecord` it
joined. The set of joined renewal ids is therefore::

    {reg.renewal_id for reg in registrations if reg.was_renewed and reg.renewal_id is not None}

This set is computed once at queue startup by scanning the registration store
(:meth:`pd_matcher.index.lookup.NyplIndexLookup.iter_registrations`) and held as
a ``frozenset[str]``; it requires no index schema change or rebuild.

For every in-scope MARC record in the pool:

1. **Renewal search (cheap):** renewal candidates are retrieved with
   :meth:`pd_matcher.index.lookup.NyplIndexLookup.candidates_for_renewal`
   (``odat``-year bucketed) and scored with the production title / author /
   claimants / year scorers and the weighted-mean combiner. The single best
   renewal ``R`` is kept; if none clears ``min_score`` the MARC is *not* a
   renewal-haver and is skipped immediately (the join set is never consulted for
   it). This filter is what makes the build fast.
2. **Join check (O(1)):** the best renewal ``R`` is emitted only when ``R.id``
   is *not* in the joined-renewal-id set; a joined ``R`` is skipped because its
   work is already determined by a registration we hold.
3. **Scenario 4 (renewal-only):** the surviving unjoined renewal ``R`` is
   emitted as a ``pairing_type="renewal"`` pair carrying a scenario-4
   ``audit_note``. This is the labeling candidate.

The renewal arm uses the weighted-mean combiner because the renewal pathway is
untrained. The labels the queue collects are human-verified, never seeded from
unverified matcher output.

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
from pd_matcher.match.pipeline import _build_context
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

SOURCE_RENEWAL: str = "renewal"

_SCENARIO_4_NOTE: str = "scenario 4: renewal-only (unjoined renewal)"

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
    renewal cleared ``min_score`` and therefore reached the join check;
    ``joined_excluded`` counts renewal-havers dropped because the best renewal is
    joined to a registration in our index; ``scenario4_written`` counts the
    renewal-only pairs emitted (``renewal_havers == joined_excluded +
    scenario4_written``).
    """

    records_scanned: int
    renewal_havers: int
    joined_excluded: int
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


def _joined_renewal_ids(lookup: NyplIndexLookup) -> frozenset[str]:
    """Return every renewal id that a registration in the index joins to.

    Scans the registration store once and collects the ``renewal_id`` of every
    :class:`pd_matcher.models.IndexedNyplRegRecord` that ``was_renewed`` and
    carries a non-``None`` ``renewal_id``. A renewal whose id is in this set is
    *joined* — a registration we hold already links to it — so it is not a
    scenario-4 (renewal-only) candidate. This is computed at runtime from data
    already in the index; no schema change or rebuild is required.
    """
    return frozenset(
        reg.renewal_id
        for reg in lookup.iter_registrations()
        if reg.was_renewed and reg.renewal_id is not None
    )


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
    min_score: float,
) -> RenewalBuildSummary:
    """Build (or append) scenario-4 renewal-only pairs into ``out_path``.

    Loads the IDF caches and calibrator beside ``index_path``, opens the CCE
    index and the review DB, computes the joined-renewal-id set once, and runs
    the renewal-first pipeline over every pool MARC not already queued: the cheap
    renewal search filters to renewal-havers, then the O(1) join check excludes
    any book whose best renewal is already joined to a registration in the index.
    Survivors are emitted as ``pairing_type="renewal"`` pairs with a scenario-4
    ``audit_note``.

    Args:
        pool: Root directory whose ``<lang>/*.xml`` shards form the pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination SQLite review DB; renewal pairs are appended and
            MARCs already present (any pairing type) are skipped.
        matching_config: Active matcher config; supplies the year window used
            for renewal retrieval and the scoring weights.
        min_score: Renewal-arm score floor on the 0-100 scale; a MARC whose best
            renewal scores below it is not a renewal-haver and is skipped before
            the join check ever runs.

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
    window = matching_config.year_window
    min_calibrated = min_score / 100.0
    score_fn = _make_score_fn(
        idf, author_idf, publisher_idf, matching_config, renewal_combiner, calibrator
    )
    _LOGGER.info(
        "renewal queue start: pool=%s window=%d min_score=%.1f calibrator=%s",
        pool,
        window,
        min_score,
        "yes" if calibrator is not None else "no",
    )
    scanned = 0
    renewal_havers = 0
    joined_excluded = 0
    scenario4_written = 0
    with NyplIndexLookup(index_path) as lookup, ReviewDb.connect(out_path) as db:
        joined_ids = _joined_renewal_ids(lookup)
        _LOGGER.info("renewal queue: %d joined renewal ids in index", len(joined_ids))
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
                        "joined_excluded=%d scenario4_written=%d",
                        scanned,
                        renewal_havers,
                        joined_excluded,
                        scenario4_written,
                    )
                best = best_renewal(marc, lookup.candidates_for_renewal(marc, window), score_fn)
                if best is None:
                    continue
                renewal, score = best
                if score.calibrated < min_calibrated:
                    continue
                renewal_havers += 1
                if renewal.id in joined_ids:
                    joined_excluded += 1
                    continue
                pair = _build_renewal_pair_insert(
                    marc,
                    renewal,
                    score,
                    language=_language_of(marc),
                    band=band_of(score.calibrated),
                    audit_note=_SCENARIO_4_NOTE,
                )
                db.insert_pair(pair)
                scenario4_written += 1
                if scenario4_written % _FILL_LOG_INTERVAL == 0:
                    db.commit()
        finally:
            db.commit()
    _LOGGER.info(
        "renewal queue complete: scanned=%d renewal_havers=%d "
        "joined_excluded=%d scenario4_written=%d",
        scanned,
        renewal_havers,
        joined_excluded,
        scenario4_written,
    )
    return RenewalBuildSummary(
        records_scanned=scanned,
        renewal_havers=renewal_havers,
        joined_excluded=joined_excluded,
        scenario4_written=scenario4_written,
    )


__all__ = [
    "SOURCE_RENEWAL",
    "RenewalBuildSummary",
    "RenewalScore",
    "RenewalScoreFn",
    "best_renewal",
    "build_renewal_queue",
    "score_renewal",
]
