"""Public entry point for Phase 6's spawn-based worker pool.

``run_match`` is the single function the CLI calls to match every MARC
record in a file against the indexed CCE corpus and emit one CSV row per
record. It owns the orchestration: it spawns workers and one writer,
runs the producer inline in main, drives a reporter thread, hands a
shared :class:`multiprocessing.Event` to every party so SIGINT drains
the pool cleanly, and finally returns a :class:`RunReport` summarising
the run.

Spawn (not fork) is mandatory: LMDB's mmap'd index file is shared via
the OS page cache regardless of start method, so we lose nothing by
sticking to ``spawn`` and we gain identical behaviour on macOS and Linux.
"""

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as EventType
from os import cpu_count
from pathlib import Path
from time import monotonic

from msgspec import Struct
from structlog import get_logger

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.output.csv_writer import CsvResultWriter
from pd_matcher.output.csv_writer import ResultWriter
from pd_matcher.parsers.marc import iter_marc_records
from pd_matcher.workers.events import ShutdownEvent
from pd_matcher.workers.events import encode_stats_event
from pd_matcher.workers.producer import run_producer
from pd_matcher.workers.reporter import Reporter
from pd_matcher.workers.shutdown import ShutdownCoordinator
from pd_matcher.workers.worker import worker_main
from pd_matcher.workers.writer import WriterFactory
from pd_matcher.workers.writer import writer_main

_LOGGER = get_logger(__name__)
_DEFAULT_BATCH_SIZE: int = 32
_DEFAULT_REPORT_INTERVAL_SECONDS: float = 5.0
_WORKER_JOIN_TIMEOUT_SECONDS: float = 30.0


class RunReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of one :func:`run_match` invocation."""

    records_processed: int
    records_written: int
    records_enqueued: int
    duration_seconds: float
    by_status: dict[str, int]
    interrupted: bool


def _default_workers() -> int:
    """Return a sensible default worker count (``cpu_count - 1``, min 1)."""
    cpus = cpu_count() or 2
    return max(1, cpus - 1)


def _build_csv_writer(path: Path) -> ResultWriter:
    """Construct the default CSV writer."""
    return CsvResultWriter(path)


def _spawn_workers(
    ctx: SpawnContext,
    *,
    workers: int,
    index_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    input_queue: MpQueue[bytes | None],
    output_queue: MpQueue[bytes | None],
    stats_queue: MpQueue[bytes],
    shutdown_event: EventType,
) -> list[SpawnProcess]:
    """Spawn the configured number of worker processes and return them."""
    processes: list[SpawnProcess] = []
    for index in range(workers):
        process = ctx.Process(
            target=_worker_entry,
            name=f"pd_matcher.worker.{index}",
            kwargs={
                "index_path": index_path,
                "matching_config": matching_config,
                "copyright_config": copyright_config,
                "ruleset": ruleset,
                "pairing_config": pairing_config,
                "idf": idf,
                "calibrator": calibrator,
                "input_queue": input_queue,
                "output_queue": output_queue,
                "stats_queue": stats_queue,
                "shutdown_event": shutdown_event,
            },
        )
        process.start()
        processes.append(process)
    return processes


def _worker_entry(
    *,
    index_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    input_queue: MpQueue[bytes | None],
    output_queue: MpQueue[bytes | None],
    stats_queue: MpQueue[bytes],
    shutdown_event: EventType,
) -> None:
    """Top-level callable executed inside each spawned worker process.

    Keeping this thin (it just unpacks queues into get/put callables and
    defers to :func:`worker_main`) means the heavy lifting stays in a
    module that is also reachable from in-process tests.
    """
    is_shutdown = _shutdown_predicate(shutdown_event)
    worker_main(
        index_path=index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        ruleset=ruleset,
        pairing_config=pairing_config,
        idf=idf,
        calibrator=calibrator,
        input_get=input_queue.get,
        output_put=output_queue.put,
        stats_put=stats_queue.put,
        is_shutdown=is_shutdown,
    )


def _writer_entry(
    *,
    output_path: Path,
    writer_factory: WriterFactory,
    output_queue: MpQueue[bytes | None],
    stats_queue: MpQueue[bytes],
    shutdown_event: EventType,
) -> None:
    """Top-level callable executed inside the writer process."""
    is_shutdown = _shutdown_predicate(shutdown_event)
    writer_main(
        output_path=output_path,
        writer_factory=writer_factory,
        output_get=output_queue.get,
        stats_put=stats_queue.put,
        is_shutdown=is_shutdown,
    )


def _shutdown_predicate(event: EventType) -> Callable[[], bool]:
    """Return a callable that reports whether ``event.is_set()``."""

    def predicate() -> bool:
        return event.is_set()

    return predicate


def run_match(
    marc_path: Path,
    *,
    index_path: Path,
    output_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None = None,
    workers: int | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    queue_maxsize: int | None = None,
    writer_factory: WriterFactory = _build_csv_writer,
    report_interval_seconds: float = _DEFAULT_REPORT_INTERVAL_SECONDS,
) -> RunReport:
    """Run the full match pipeline over ``marc_path`` and return a summary.

    Args:
        marc_path: MARCXML file to ingest.
        index_path: LMDB env directory produced by ``pd-matcher index build``.
        output_path: Destination CSV path.
        matching_config: Loaded :class:`MatchingConfig`.
        copyright_config: Loaded :class:`CopyrightAssessmentConfig`.
        ruleset: Loaded :class:`CopyrightRuleSet`.
        pairing_config: Loaded :class:`PairingConfig`; each worker compiles
            it into :class:`CompiledPairings` once at init.
        idf: Pre-built :class:`IdfTable` (workers load it once at init).
        calibrator: Optional Platt calibrator.
        workers: Number of worker processes. ``None`` uses ``cpu_count - 1``.
        batch_size: MARC records per IPC batch.
        queue_maxsize: Bound on the input queue. ``None`` derives ``workers * 4``.
        writer_factory: Factory producing the per-row CSV writer. Defaults
            to :class:`CsvResultWriter`.
        report_interval_seconds: Reporter cadence.

    Returns:
        A :class:`RunReport` describing the run.
    """
    worker_count = workers if workers is not None else _default_workers()
    if worker_count < 1:
        raise ValueError(f"workers must be >= 1 (got {worker_count!r})")
    input_capacity = queue_maxsize if queue_maxsize is not None else worker_count * 4
    ctx = get_context("spawn")
    input_queue: MpQueue[bytes | None] = ctx.Queue(maxsize=input_capacity)
    output_queue: MpQueue[bytes | None] = ctx.Queue()
    stats_queue: MpQueue[bytes] = ctx.Queue()
    started_at = monotonic()
    with (
        ShutdownCoordinator(ctx.Event()) as coord,
        Reporter(
            queue=stats_queue,
            report_interval_seconds=report_interval_seconds,
        ) as reporter,
    ):
        worker_processes = _spawn_workers(
            ctx,
            workers=worker_count,
            index_path=index_path,
            matching_config=matching_config,
            copyright_config=copyright_config,
            ruleset=ruleset,
            pairing_config=pairing_config,
            idf=idf,
            calibrator=calibrator,
            input_queue=input_queue,
            output_queue=output_queue,
            stats_queue=stats_queue,
            shutdown_event=coord.event,
        )
        writer_process = ctx.Process(
            target=_writer_entry,
            name="pd_matcher.writer",
            kwargs={
                "output_path": output_path,
                "writer_factory": writer_factory,
                "output_queue": output_queue,
                "stats_queue": stats_queue,
                "shutdown_event": coord.event,
            },
        )
        writer_process.start()
        _LOGGER.info(
            "match.pool.start",
            workers=worker_count,
            batch_size=batch_size,
            queue_maxsize=input_capacity,
            output_path=str(output_path),
        )
        records_enqueued = run_producer(
            iter_marc_records(marc_path),
            input_put=input_queue.put,
            stats_put=stats_queue.put,
            is_shutdown=coord.event.is_set,
            batch_size=batch_size,
        )
        for _ in range(worker_count):
            input_queue.put(None)
        for process in worker_processes:
            process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
        output_queue.put(None)
        writer_process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
        stats_queue.put(encode_stats_event(ShutdownEvent(reason="completed")))
    snapshot = reporter.totals.snapshot()
    duration = monotonic() - started_at
    interrupted = coord.is_set
    _LOGGER.info(
        "match.pool.complete",
        records_processed=snapshot.records_processed,
        records_written=snapshot.records_written,
        records_enqueued=records_enqueued,
        duration_seconds=duration,
        interrupted=interrupted,
    )
    return RunReport(
        records_processed=snapshot.records_processed,
        records_written=snapshot.records_written,
        records_enqueued=records_enqueued,
        duration_seconds=duration,
        by_status=snapshot.by_status,
        interrupted=interrupted,
    )


__all__ = [
    "RunReport",
    "run_match",
]
