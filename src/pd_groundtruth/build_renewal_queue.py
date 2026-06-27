"""Renewal-pair review-queue builder.

The registration queue (:mod:`pd_groundtruth.build_queue`) proposes
``(MARC, registration)`` pairs; this builder is its renewal-side sibling. For
every in-scope MARC record in the pool it retrieves renewal candidates with
:meth:`pd_matcher.index.lookup.NyplIndexLookup.candidates_for_renewal`, scores
each MARC↔renewal pairing with the production title / author / claimants / year
scorers and the weighted-mean combiner (mirroring
``scripts/renewal_gate_measure.py``), keeps the single best candidate above the
score floor, and writes it into the review DB as a ``pairing_type="renewal"``
pair so the same UI can label it.

Before the renewal arm runs, each MARC is first put through the production
*registration* matcher (the learned scorer with a high ``reg_min_score`` floor)
to decide whether a renewal pair is worth labeling at all:

* **Scenario 2** — a registration matches above the floor and that
  registration is already joined to a renewal (``was_renewed=True``). The
  copyright story is already settled, so the MARC is skipped entirely; no
  renewal pair is emitted.
* **Scenario 3** — a registration matches above the floor but is *not* joined
  to a renewal (``was_renewed=False``). A renewal may exist unlinked, so the
  renewal arm runs and any emitted pair records the scenario on its
  ``audit_note``.
* **Scenario 4** — no registration clears the floor. The renewal arm runs as a
  renewal-only candidate and the emitted pair records the scenario.

The registration arm uses the learned scorer by construction, so its model
artifact must be present beside the index; its absence fails the build loudly
via :func:`pd_matcher.match.combiners.build_combiner`. The renewal arm stays on
the weighted-mean combiner because the renewal pathway is untrained.

A renewal pair's ``nypl_uuid`` column carries the renewal record's
``entry_id`` rather than a registration UUID — the column is polymorphic by
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
from pd_matcher.match.pipeline import match_record
from pd_matcher.match.result import MatchResult
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

SOURCE_RENEWAL: str = "renewal"

_LEARNED_SCORER: str = "learned"

SCENARIO_ALREADY_RENEWED: int = 2
SCENARIO_REG_UNLINKED: int = 3
SCENARIO_NO_REG_MATCH: int = 4

_SCENARIO_AUDIT_NOTES: dict[int, str] = {
    SCENARIO_REG_UNLINKED: "scenario 3: registration matched (was_renewed=False)",
    SCENARIO_NO_REG_MATCH: "scenario 4: no registration match",
}

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

RegScenarioFn = Callable[[MarcRecord], int]


def _scenario_for_result(result: MatchResult, lookup: NyplIndexLookup) -> int:
    """Classify ``result`` into a renewal-queue scenario (2, 3, or 4).

    Returns :data:`SCENARIO_NO_REG_MATCH` when no registration cleared the
    floor, :data:`SCENARIO_ALREADY_RENEWED` when the best registration is
    already joined to a renewal (``was_renewed=True``), and
    :data:`SCENARIO_REG_UNLINKED` otherwise. A best registration whose record
    can no longer be fetched is treated as unlinked.
    """
    if result.best is None:
        return SCENARIO_NO_REG_MATCH
    registration = lookup.get_registration(result.best.nypl_uuid)
    if registration is not None and registration.was_renewed:
        return SCENARIO_ALREADY_RENEWED
    return SCENARIO_REG_UNLINKED


def _make_reg_scenario_fn(
    lookup: NyplIndexLookup,
    reg_config: MatchingConfig,
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
    calibrator: PlattCalibrator | None,
    combiner: Combiner,
    pairings: CompiledPairings,
) -> RegScenarioFn:
    """Build a ``marc -> scenario`` closure over the registration matcher.

    Each call runs the production :func:`pd_matcher.match.pipeline.match_record`
    with the learned scorer and the registration floor baked into
    ``reg_config``, then maps the verdict to a scenario via
    :func:`_scenario_for_result`.
    """

    def scenario_fn(marc: MarcRecord) -> int:
        result = match_record(
            marc,
            lookup=lookup,
            config=reg_config,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            calibrator=calibrator,
            combiner=combiner,
            pairings=pairings,
        )
        return _scenario_for_result(result, lookup)

    return scenario_fn


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
    calibrator (when present) maps the raw score exactly as
    :func:`pd_matcher.match.pipeline.match_record` does.
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


def renewal_pair_for(
    marc: MarcRecord,
    candidates: Iterable[NyplRenRecord],
    *,
    score_fn: RenewalScoreFn,
    min_calibrated: float,
    audit_note: str,
) -> PairInsert | None:
    """Return the best renewal :class:`PairInsert` for ``marc``, or ``None``.

    Selects the highest-calibrated candidate via :func:`best_renewal` and
    returns a renewal pair only when its calibrated score is at or above
    ``min_calibrated``; otherwise ``None`` (no candidate, or all below floor).
    ``audit_note`` is stamped onto the emitted pair so the labeler sees which
    scenario produced it.
    """
    best = best_renewal(marc, candidates, score_fn)
    if best is None:
        return None
    renewal, score = best
    if score.calibrated < min_calibrated:
        return None
    return _build_renewal_pair_insert(
        marc,
        renewal,
        score,
        language=_language_of(marc),
        band=band_of(score.calibrated),
        audit_note=audit_note,
    )


class RenewalBuildSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Result of one :func:`build_renewal_queue` invocation.

    ``records_scanned`` counts distinct pool MARCs considered (after skipping
    those already in the target DB); ``pairs_written`` counts renewal pairs
    inserted. ``scenario2_skipped`` counts MARCs dropped because their
    confident registration was already renewed; ``scenario3_written`` and
    ``scenario4_written`` count the renewal pairs emitted under each remaining
    scenario (their sum equals ``pairs_written``).
    """

    records_scanned: int
    pairs_written: int
    scenario2_skipped: int
    scenario3_written: int
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
    """Build (or append) scenario-filtered renewal pairs into ``out_path``.

    Loads the IDF caches and calibrator beside ``index_path``, opens the CCE
    index and the review DB, and for every pool MARC not already queued first
    runs the production registration matcher (``reg_scorer`` with a
    ``reg_min_score`` floor) to classify the MARC. MARCs whose confident
    registration is already renewed (scenario 2) are skipped; the rest run the
    renewal arm and the best renewal candidate scoring at or above ``min_score``
    is inserted as a ``pairing_type="renewal"`` pair carrying a scenario
    ``audit_note``.

    Args:
        pool: Root directory whose ``<lang>/*.xml`` shards form the pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination SQLite review DB; renewal pairs are appended and
            MARCs already present (any pairing type) are skipped.
        matching_config: Active matcher config; supplies the year window used
            for renewal retrieval and the scoring weights.
        pairing_config: Field-pairing config driving the registration matcher's
            title/author/publisher scorer groups.
        min_score: Renewal-arm score floor on the 0-100 scale; only the best
            renewal candidate at or above it is queued.
        reg_min_score: Registration-arm score floor on the 0-100 scale; the
            production criterion separating scenario 2/3 from scenario 4.
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
    combiner = build_combiner(matching_config, learned_model_dir=None)
    reg_config = replace(matching_config, scorer=reg_scorer, min_combined_score=reg_min_score)
    learned_model_dir = index_path.parent if reg_scorer == _LEARNED_SCORER else None
    reg_combiner = build_combiner(reg_config, learned_model_dir=learned_model_dir)
    pairings = compile_pairings(pairing_config)
    window = matching_config.year_window
    min_calibrated = min_score / 100.0
    score_fn = _make_score_fn(idf, author_idf, publisher_idf, matching_config, combiner, calibrator)
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
    written = 0
    scenario2_skipped = 0
    scenario3_written = 0
    scenario4_written = 0
    with NyplIndexLookup(index_path) as lookup, ReviewDb.connect(out_path) as db:
        reg_scenario_fn = _make_reg_scenario_fn(
            lookup, reg_config, idf, author_idf, publisher_idf, calibrator, reg_combiner, pairings
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
                scenario = reg_scenario_fn(marc)
                if scenario == SCENARIO_ALREADY_RENEWED:
                    scenario2_skipped += 1
                    continue
                pair = renewal_pair_for(
                    marc,
                    lookup.candidates_for_renewal(marc, window),
                    score_fn=score_fn,
                    min_calibrated=min_calibrated,
                    audit_note=_SCENARIO_AUDIT_NOTES[scenario],
                )
                if pair is None:
                    continue
                db.insert_pair(pair)
                written += 1
                if scenario == SCENARIO_REG_UNLINKED:
                    scenario3_written += 1
                else:
                    scenario4_written += 1
                if written % _FILL_LOG_INTERVAL == 0:
                    db.commit()
                    _LOGGER.info(
                        "renewal queue: scanned=%d written=%d "
                        "scenario2_skipped=%d scenario3_written=%d scenario4_written=%d",
                        scanned,
                        written,
                        scenario2_skipped,
                        scenario3_written,
                        scenario4_written,
                    )
        finally:
            db.commit()
    _LOGGER.info(
        "renewal queue complete: scanned=%d written=%d "
        "scenario2_skipped=%d scenario3_written=%d scenario4_written=%d",
        scanned,
        written,
        scenario2_skipped,
        scenario3_written,
        scenario4_written,
    )
    return RenewalBuildSummary(
        records_scanned=scanned,
        pairs_written=written,
        scenario2_skipped=scenario2_skipped,
        scenario3_written=scenario3_written,
        scenario4_written=scenario4_written,
    )


__all__ = [
    "SCENARIO_ALREADY_RENEWED",
    "SCENARIO_NO_REG_MATCH",
    "SCENARIO_REG_UNLINKED",
    "SOURCE_RENEWAL",
    "RegScenarioFn",
    "RenewalBuildSummary",
    "RenewalScore",
    "RenewalScoreFn",
    "best_renewal",
    "build_renewal_queue",
    "renewal_pair_for",
    "score_renewal",
]
