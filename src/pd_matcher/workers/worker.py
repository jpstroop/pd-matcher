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

from structlog import get_logger
from structlog.contextvars import bind_contextvars
from structlog.contextvars import unbind_contextvars

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.copyright.facts import build_facts
from pd_matcher.copyright.rules import assess
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import encode_stats_event
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import encode_worker_output
from pd_matcher.workers.producer import decode_batch

_LOGGER = get_logger(__name__)


def _process_record(
    marc: MarcRecord,
    *,
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    combiner: WeightedMeanCombiner,
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


def run_worker_loop(
    *,
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    ruleset: CopyrightRuleSet,
    assessment_config: CopyrightAssessmentConfig,
    input_get: Callable[[], bytes | None],
    output_put: Callable[[bytes], None],
    stats_put: Callable[[bytes], None],
    is_shutdown: Callable[[], bool],
) -> int:
    """Drain the input queue, score records, and forward results.

    Args:
        lookup: Open read-only :class:`NyplIndexLookup`.
        config: Active :class:`MatchingConfig`.
        idf: Pre-built :class:`IdfTable`.
        calibrator: Optional Platt calibrator.
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

    Returns:
        The number of records processed by this worker.
    """
    combiner = WeightedMeanCombiner(config=config)
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
                    ruleset=ruleset,
                    assessment_config=assessment_config,
                )
                output_put(encode_worker_output(output))
                stats_put(_stats_event_for(output))
                processed += 1
            finally:
                unbind_contextvars("marc_id")
    return processed


def worker_main(
    *,
    index_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    input_get: Callable[[], bytes | None],
    output_put: Callable[[bytes], None],
    stats_put: Callable[[bytes], None],
    is_shutdown: Callable[[], bool],
) -> int:
    """Top-level worker entry point.

    Opens the LMDB lookup and runs the consume loop until exhaustion or
    shutdown. Returns the count of processed records.
    """
    with NyplIndexLookup(index_path) as lookup:
        return run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=idf,
            calibrator=calibrator,
            ruleset=ruleset,
            assessment_config=copyright_config,
            input_get=input_get,
            output_put=output_put,
            stats_put=stats_put,
            is_shutdown=is_shutdown,
        )


__all__ = [
    "run_worker_loop",
    "worker_main",
]
