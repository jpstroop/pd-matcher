"""Stratified review-queue builder orchestration.

Runs the main app's matcher (:mod:`pd_matcher`) over a stratified sample of
the full-MARC candidate pool and writes a self-contained SQLite review DB of
proposed ``(MARC, CCE-candidate)`` pairs for Phase 2b hand-labeling.

Parallelism mirrors :mod:`pd_matcher.eval.ground_truth`: a ``spawn`` pool
whose workers each open one read-only LMDB lookup and load the IDF table
once. To avoid ``lmdb.Error: already open`` (opening the same env twice in a
process), the IDF cache is refreshed exactly once in the *main* process via a
transient lookup that closes before any worker spawns; workers then hit the
warm cache (a fast load, no rebuild) and open their own single matching
lookup.

The matcher runs with ``min_combined_score = 0.0`` so ``best`` is always the
top candidate regardless of score; ``year_window`` keeps its default. Workers
emit serializable :class:`WorkerOutcome` structs (no live LMDB handles cross
the process boundary); the main process streams them into a
:class:`pd_groundtruth.sampling.Stratifier`, then re-opens one lookup to
snapshot CCE fields for the accepted pairs and persists via
:mod:`pd_groundtruth.review_db`.
"""

from collections.abc import Iterator
from logging import getLogger
from multiprocessing import get_context
from pathlib import Path
from time import monotonic
from typing import Protocol

from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode
from msgspec.structs import replace
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.parsers.marc import iter_marc_records

from pd_groundtruth.progress import ProgressReporter
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import BAND_BELOW
from pd_groundtruth.sampling import AcceptedPair
from pd_groundtruth.sampling import BudgetModel
from pd_groundtruth.sampling import Stratifier
from pd_groundtruth.sampling import StratumOutcome
from pd_groundtruth.sampling import band_of
from pd_groundtruth.sampling import reservoir_sample

_LOGGER = getLogger(__name__)

_IDF_CACHE_NAME: str = "idf.msgpack"
_CHUNK_DIVISOR: int = 8

_MARC_DECODER = json_decode
_MARC_ENCODER = json_encode


class WorkerOutcome(Struct, frozen=True, forbid_unknown_fields=True):
    """One matched record returned across the spawn-pool boundary.

    Carries only serializable values — the lossless MARC JSON, the matcher's
    top score, the matched NYPL uuid, and the per-field evidence JSON — so no
    live LMDB handle ever crosses processes.
    """

    language: str
    marc_control_id: str
    marc_json: bytes
    score: float
    nypl_uuid: str
    evidence_json: bytes


class MatcherState(Protocol):
    """The matcher resources :func:`_match_one` reads.

    A Protocol (rather than the concrete :class:`_WorkerState`) so the pure
    matching step can be unit-tested with a lightweight fabricated state and
    a monkeypatched ``match_record`` — no LMDB handle required. The members
    are read-only properties so concrete implementations may narrow the
    attribute types (e.g. a specific :class:`Combiner` subtype).
    """

    @property
    def lookup(self) -> NyplIndexLookup: ...

    @property
    def idf(self) -> IdfTable: ...

    @property
    def combiner(self) -> Combiner: ...

    @property
    def matching_config(self) -> MatchingConfig: ...

    @property
    def pairings(self) -> CompiledPairings: ...


class _WorkerState:
    """Per-process matcher resources opened once in the pool initializer."""

    __slots__ = ("combiner", "idf", "lookup", "matching_config", "pairings")

    def __init__(
        self,
        *,
        index_path: Path,
        idf_cache_path: Path,
        matching_config: MatchingConfig,
        pairing_config: PairingConfig,
    ) -> None:
        self.idf: IdfTable = load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index_path))
        self.lookup = NyplIndexLookup(index_path)
        self.combiner = WeightedMeanCombiner(config=matching_config)
        self.matching_config = matching_config
        self.pairings = compile_pairings(pairing_config)


_WORKER_STATE: _WorkerState | None = None


def _evidence_payload(evidence: tuple[tuple[str, float], ...]) -> dict[str, float]:
    """Return a ``scorer -> normalized score`` mapping for the 2b card."""
    return dict(evidence)


def _match_one(language: str, marc: MarcRecord, state: MatcherState) -> WorkerOutcome | None:
    """Match one MARC record and return a serializable outcome (or ``None``).

    ``None`` means the matcher found no candidate at all (e.g. no
    publication year, or an empty year bucket) — such records carry no
    proposable pair and are dropped before stratification.
    """
    result = match_record(
        marc,
        lookup=state.lookup,
        config=state.matching_config,
        idf=state.idf,
        calibrator=None,
        combiner=state.combiner,
        pairings=state.pairings,
    )
    best = result.best
    if best is None:
        return None
    evidence = tuple((ev.scorer, ev.normalized) for ev in best.evidence if not ev.skipped)
    return WorkerOutcome(
        language=language,
        marc_control_id=marc.control_id,
        marc_json=_MARC_ENCODER(marc),
        score=best.combined.calibrated,
        nypl_uuid=best.nypl_uuid,
        evidence_json=_MARC_ENCODER(_evidence_payload(evidence)),
    )


class _Task(Struct, frozen=True, forbid_unknown_fields=True):
    """One MARC record handed to a worker, paired with its language."""

    language: str
    marc_json: bytes


def _pool_initializer(
    index_path: Path,
    idf_cache_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
) -> None:
    """Spawn-pool initializer: build the per-worker :class:`_WorkerState`."""
    global _WORKER_STATE
    _WORKER_STATE = _WorkerState(
        index_path=index_path,
        idf_cache_path=idf_cache_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
    )


def _pool_match(task: _Task) -> WorkerOutcome | None:
    """Spawn-pool worker function: match one task using process-local state."""
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("_pool_match called before _pool_initializer ran")
    marc = _MARC_DECODER(task.marc_json, type=MarcRecord)
    return _match_one(task.language, marc, state)


def _iter_language_dirs(pool: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(language, dir)`` pairs for each language subdir under ``pool``."""
    for child in sorted(pool.iterdir()):
        if child.is_dir():
            yield child.name, child


def _sample_language(
    language_dir: Path,
    *,
    sample_per_lang: int,
    seed: int,
) -> list[MarcRecord]:
    """Reservoir-sample up to ``sample_per_lang`` MARC records from one dir."""

    def _records() -> Iterator[MarcRecord]:
        for shard in sorted(language_dir.glob("*.xml")):
            yield from iter_marc_records(shard)

    return reservoir_sample(_records(), sample_per_lang, seed)


def _build_tasks(language: str, records: list[MarcRecord]) -> list[_Task]:
    """Encode sampled records into worker tasks."""
    return [_Task(language=language, marc_json=_MARC_ENCODER(marc)) for marc in records]


def _refresh_idf_cache(index_path: Path, idf_cache_path: Path) -> None:
    """Build/refresh the IDF cache ONCE in the main process.

    Uses a transient lookup that ``load_or_build_idf`` opens and closes, so
    the main process never holds a second open env while the workers later
    open theirs. After this returns, every worker's ``load_or_build_idf``
    hits the warm cache (fast load, no rebuild).
    """
    load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index_path))


def _run_pool(
    tasks: list[_Task],
    *,
    index_path: Path,
    idf_cache_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    workers: int,
) -> Iterator[WorkerOutcome]:
    """Fan ``tasks`` across a spawn pool and yield non-``None`` outcomes."""
    if not tasks:
        return
    chunksize = max(1, len(tasks) // (workers * _CHUNK_DIVISOR))
    ctx = get_context("spawn")
    init_args = (index_path, idf_cache_path, matching_config, pairing_config)
    with ctx.Pool(
        processes=workers,
        initializer=_pool_initializer,
        initargs=init_args,
    ) as pool:
        for outcome in pool.imap_unordered(_pool_match, tasks, chunksize=chunksize):
            if outcome is not None:
                yield outcome


def _decade_of(year: int | None) -> int | None:
    """Return the decade bucket for ``year`` (e.g. 1953 -> 1950)."""
    if year is None:
        return None
    return (year // 10) * 10


def _join(values: tuple[str, ...]) -> str | None:
    """Join a tuple of strings with ``" | "`` or return ``None`` when empty."""
    return " | ".join(values) if values else None


def _pair_insert(
    accepted: AcceptedPair,
    outcome: WorkerOutcome,
    cce: IndexedNyplRegRecord | None,
) -> PairInsert:
    """Build a :class:`PairInsert` from an accepted outcome and CCE snapshot."""
    marc = _MARC_DECODER(outcome.marc_json, type=MarcRecord)
    return PairInsert(
        language=accepted.language,
        decade=_decade_of(marc.publication_year),
        score=accepted.score,
        band=accepted.band,
        source=accepted.source,
        marc_control_id=marc.control_id,
        marc_json=outcome.marc_json.decode("utf-8"),
        marc_title=marc.title,
        marc_author=marc.main_author or marc.statement_of_responsibility,
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        nypl_uuid=outcome.nypl_uuid,
        cce_title=None if cce is None else cce.title,
        cce_author=None if cce is None else cce.author_name,
        cce_publishers=None if cce is None else _join(cce.publisher_names),
        cce_claimants=None if cce is None else _join(cce.claimants),
        cce_reg_year=None if cce is None else cce.reg_year,
        cce_was_renewed=None if cce is None else cce.was_renewed,
        cce_regnum=None if cce is None else cce.regnum,
        evidence_json=outcome.evidence_json.decode("utf-8"),
    )


class BuildSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Result of one :func:`build_queue` invocation."""

    records_sampled: int
    records_matched: int
    pairs_written: int
    stratum_counts: dict[str, int]


def _stratum_label(language: str, band: str) -> str:
    """Render a ``(language, band)`` key as a flat ``lang/band`` string."""
    return f"{language}/{band}"


def _tally_kept(
    kept_by_stratum: dict[tuple[str, str], int],
    outcome: WorkerOutcome,
    accepted: AcceptedPair | None,
    budget: BudgetModel,
) -> None:
    """Update the live kept-per-stratum tally for the progress readout.

    Banded acceptances increment their exact ``(language, band)``. Below-0.7
    outcomes are buffered by the :class:`Stratifier` (its reservoir draws at
    finalize), so their running count is tracked here and clamped to the
    ``below`` cap to mirror the eventual accepted count.
    """
    if accepted is not None:
        key = (accepted.language, accepted.band)
        kept_by_stratum[key] = kept_by_stratum.get(key, 0) + 1
        return
    if band_of(outcome.score) == BAND_BELOW:
        key = (outcome.language, BAND_BELOW)
        cap = budget.cap_for(outcome.language, BAND_BELOW)
        kept_by_stratum[key] = min(cap, kept_by_stratum.get(key, 0) + 1)


def build_queue(
    *,
    pool: Path,
    index_path: Path,
    out_path: Path,
    budget: BudgetModel,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    seed: int,
    workers: int,
    sample_per_lang: int,
) -> BuildSummary:
    """Build the stratified review queue and persist it to ``out_path``.

    Args:
        pool: Root directory whose ``<lang>/*.xml`` shards form the
            candidate pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination SQLite review database.
        budget: Per-(language, band) caps.
        matching_config: Active config; the score floor is forced to ``0.0``
            internally so ``best`` is always the top candidate.
        pairing_config: Active field-pairing config.
        seed: Seed for the reservoir samplers (record selection and the
            below-0.7 draw).
        workers: Number of spawn-pool worker processes (``>= 1``).
        sample_per_lang: Reservoir size per language directory.

    Returns:
        A populated :class:`BuildSummary`.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1 (got {workers!r})")
    floored_config = replace(matching_config, min_combined_score=0.0)
    idf_cache_path = index_path.parent / _IDF_CACHE_NAME
    _LOGGER.info("refreshing IDF cache once in main process")
    _refresh_idf_cache(index_path, idf_cache_path)

    tasks: list[_Task] = []
    for language, language_dir in _iter_language_dirs(pool):
        records = _sample_language(language_dir, sample_per_lang=sample_per_lang, seed=seed)
        _LOGGER.info("sampled %d records for language=%s", len(records), language)
        tasks.extend(_build_tasks(language, records))

    stratifier = Stratifier(budget, seed=seed)
    outcomes_by_key: dict[str, WorkerOutcome] = {}
    kept_by_stratum: dict[tuple[str, str], int] = {}
    records_matched = 0
    _LOGGER.info("matching start: total=%d workers=%d", len(tasks), workers)
    reporter = ProgressReporter(
        logger=_LOGGER,
        total=len(tasks),
        budget=budget,
        clock=monotonic,
    )
    for outcome in _run_pool(
        tasks,
        index_path=index_path,
        idf_cache_path=idf_cache_path,
        matching_config=floored_config,
        pairing_config=pairing_config,
        workers=workers,
    ):
        records_matched += 1
        outcomes_by_key[outcome.marc_control_id] = outcome
        accepted_pair = stratifier.offer(
            StratumOutcome(
                key=outcome.marc_control_id,
                language=outcome.language,
                score=outcome.score,
            )
        )
        _tally_kept(kept_by_stratum, outcome, accepted_pair, budget)
        reporter.update(records_matched, kept_by_stratum)

    accepted = stratifier.finalize()
    pairs_written = _persist(out_path, accepted, outcomes_by_key, index_path)
    counts = {_stratum_label(lang, band): n for (lang, band), n in stratifier.counts().items()}
    for label, n in sorted(counts.items()):
        _LOGGER.info("stratum %s filled=%d", label, n)
    _LOGGER.info(
        "build complete: sampled=%d matched=%d written=%d",
        len(tasks),
        records_matched,
        pairs_written,
    )
    return BuildSummary(
        records_sampled=len(tasks),
        records_matched=records_matched,
        pairs_written=pairs_written,
        stratum_counts=counts,
    )


def _persist(
    out_path: Path,
    accepted: list[AcceptedPair],
    outcomes_by_key: dict[str, WorkerOutcome],
    index_path: Path,
) -> int:
    """Snapshot CCE fields for accepted pairs and write them to the DB."""
    written = 0
    with NyplIndexLookup(index_path) as lookup, ReviewDb.connect(out_path) as db:
        for pair in accepted:
            outcome = outcomes_by_key[pair.key]
            cce = lookup.get_registration(outcome.nypl_uuid)
            db.insert_pair(_pair_insert(pair, outcome, cce))
            written += 1
    return written


__all__ = [
    "BuildSummary",
    "MatcherState",
    "WorkerOutcome",
    "build_queue",
]
