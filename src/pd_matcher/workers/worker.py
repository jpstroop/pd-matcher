"""Worker process body for the Phase 6 spawn pool.

Each spawned worker opens the LMDB index read-only on init, loads the
IDF table and Platt calibrator (small msgpack files), and then runs a
loop that dequeues encoded MARC batches, matches each record, runs the
copyright rule engine, and pushes a :class:`WorkerOutput` blob onto the
output queue.

LMDB read-only mode is the design's payoff: the mmap'd index file is
shared across every worker via the OS page cache regardless of fork vs.
spawn semantics, so no special handle-passing trick is required. Each
worker gets its own LMDB env handle but they all point at the same
backing pages.
"""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from time import monotonic

from structlog import get_logger
from structlog.contextvars import bind_contextvars
from structlog.contextvars import unbind_contextvars

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.copyright.facts import build_facts
from pd_matcher.copyright.rules import assess
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import encode_stats_event
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import encode_worker_output
from pd_matcher.workers.producer import decode_batch

_LOGGER = get_logger(__name__)

_WORKER_LOG_EVERY_N: int = 1000


def _worker_rate(processed: int, started_at: float, now: float) -> float:
    """Return per-worker throughput, falling back to ``0.0`` when idle."""
    elapsed = now - started_at
    if elapsed <= 0.0:
        return 0.0
    return processed / elapsed


def _process_record(
    marc: MarcRecord,
    *,
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    combiner: WeightedMeanCombiner,
    pairings: CompiledPairings,
    ruleset: CopyrightRuleSet,
    assessment_config: CopyrightAssessmentConfig,
) -> WorkerOutput:
    """Run match + copyright rules over one record and return the wire payload."""
    match = match_record(
        marc,
        lookup=lookup,
        config=config,
        idf=idf,
        calibrator=calibrator,
        combiner=combiner,
        pairings=pairings,
    )
    matched_nypl = None
    if match.best is not None:
        matched_nypl = lookup.get_registration(match.best.nypl_uuid)
    as_of_year = (
        assessment_config.as_of_year
        if assessment_config.as_of_year is not None
        else date.today().year
    )
    facts = build_facts(marc, match, as_of_year=as_of_year, matched_nypl=matched_nypl)
    assessment = assess(
        facts,
        ruleset,
        enable_assumptions=assessment_config.enable_assumptions,
    )
    return WorkerOutput(
        marc=marc,
        match=match,
        assessment=assessment,
        matched_nypl=matched_nypl,
    )


def _stats_event_for(output: WorkerOutput) -> bytes:
    """Build the stats payload for one :class:`WorkerOutput`."""
    confidence = 0.0
    if output.match.best is not None:
        confidence = output.match.best.combined.calibrated
    return encode_stats_event(
        RecordProcessed(
            status=output.assessment.status,
            confidence=confidence,
            candidates_considered=output.match.candidates_considered,
        )
    )


def _log_hit(worker_id: int, output: WorkerOutput) -> None:
    """Emit a per-record ``-vv`` line describing one match outcome."""
    best = output.match.best
    reg = "none" if best is None else best.nypl_uuid
    score = 0.0 if best is None else best.combined.calibrated
    _LOGGER.info(
        "worker.hit",
        worker=worker_id,
        marc=output.marc.control_id,
        reg=reg,
        score=round(score, 4),
        status=output.assessment.status.value,
    )


def run_worker_loop(
    *,
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    pairings: CompiledPairings,
    ruleset: CopyrightRuleSet,
    assessment_config: CopyrightAssessmentConfig,
    input_get: Callable[[], bytes | None],
    output_put: Callable[[bytes], None],
    stats_put: Callable[[bytes], None],
    is_shutdown: Callable[[], bool],
    worker_id: int = 0,
    verbosity: int = 0,
    clock: Callable[[], float] = monotonic,
) -> int:
    """Drain the input queue, score records, and forward results.

    Args:
        lookup: Open read-only :class:`NyplIndexLookup`.
        config: Active :class:`MatchingConfig`.
        idf: Pre-built :class:`IdfTable`.
        calibrator: Optional Platt calibrator.
        pairings: Compiled field pairings shared across all records.
        ruleset: Loaded :class:`CopyrightRuleSet`.
        assessment_config: Runtime configuration for the rule engine.
        input_get: Zero-arg callable returning the next encoded batch
            blob, or ``None`` to signal the worker should stop. Usually
            ``input_queue.get``.
        output_put: Callable that pushes one :class:`WorkerOutput` blob
            onto the writer queue.
        stats_put: Callable that pushes one stats event onto the stats
            queue.
        is_shutdown: Zero-arg callable returning ``True`` when shutdown
            has been requested. Checked between batches and between
            records.
        worker_id: Stable ordinal for this worker, used in per-worker logs.
        verbosity: ``0`` aggregate-only (no per-worker logs); ``1`` logs a
            ``worker=<id> processed=<n> rate=<r>/s`` line on start/finish and
            every :data:`_WORKER_LOG_EVERY_N` records; ``2`` additionally logs
            one line per record processed.
        clock: Monotonic clock; injected so per-worker rate is testable.

    Returns:
        The number of records processed by this worker.
    """
    combiner = WeightedMeanCombiner(config=config)
    started_at = clock()
    if verbosity >= 1:
        _LOGGER.info("worker.start", worker=worker_id)
    processed = 0
    while True:
        if is_shutdown():
            break
        blob = input_get()
        if blob is None:
            break
        batch = decode_batch(blob)
        for marc in batch:
            if is_shutdown():
                if verbosity >= 1:
                    _LOGGER.info(
                        "worker.finish",
                        worker=worker_id,
                        processed=processed,
                        rate=round(_worker_rate(processed, started_at, clock()), 1),
                    )
                return processed
            bind_contextvars(marc_id=marc.control_id)
            try:
                output = _process_record(
                    marc,
                    lookup=lookup,
                    config=config,
                    idf=idf,
                    calibrator=calibrator,
                    combiner=combiner,
                    pairings=pairings,
                    ruleset=ruleset,
                    assessment_config=assessment_config,
                )
                output_put(encode_worker_output(output))
                stats_put(_stats_event_for(output))
                processed += 1
                if verbosity >= 2:
                    _log_hit(worker_id, output)
                if verbosity >= 1 and processed % _WORKER_LOG_EVERY_N == 0:
                    _LOGGER.info(
                        "worker.progress",
                        worker=worker_id,
                        processed=processed,
                        rate=round(_worker_rate(processed, started_at, clock()), 1),
                    )
            finally:
                unbind_contextvars("marc_id")
    if verbosity >= 1:
        _LOGGER.info(
            "worker.finish",
            worker=worker_id,
            processed=processed,
            rate=round(_worker_rate(processed, started_at, clock()), 1),
        )
    return processed


def worker_main(
    *,
    index_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    input_get: Callable[[], bytes | None],
    output_put: Callable[[bytes], None],
    stats_put: Callable[[bytes], None],
    is_shutdown: Callable[[], bool],
    worker_id: int = 0,
    verbosity: int = 0,
) -> int:
    """Top-level worker entry point.

    Opens the LMDB lookup, compiles the pairing config once, and runs the
    consume loop until exhaustion or shutdown. ``worker_id`` and ``verbosity``
    flow straight through to :func:`run_worker_loop`'s per-worker logging.
    Returns the count of processed records.
    """
    pairings = compile_pairings(pairing_config)
    with NyplIndexLookup(index_path) as lookup:
        return run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=idf,
            calibrator=calibrator,
            pairings=pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
            input_get=input_get,
            output_put=output_put,
            stats_put=stats_put,
            is_shutdown=is_shutdown,
            worker_id=worker_id,
            verbosity=verbosity,
        )


__all__ = [
    "run_worker_loop",
    "worker_main",
]
