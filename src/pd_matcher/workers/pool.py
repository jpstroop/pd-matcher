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
from collections.abc import Iterator
from multiprocessing import get_context
from multiprocessing.context import SpawnContext
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as EventType
from os import cpu_count
from pathlib import Path
from time import monotonic
from typing import Protocol

from msgspec import Struct
from structlog import get_logger

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.logging_config import configure_logging
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.prepare import iter_prepared_records
from pd_matcher.models import MarcRecord
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
    interrupted: bool


class _TerminableProcess(Protocol):
    """Subset of :class:`SpawnProcess` used by :func:`_terminate_if_alive`."""

    def is_alive(self) -> bool: ...  # pragma: no cover
    def terminate(self) -> None: ...  # pragma: no cover
    def kill(self) -> None: ...  # pragma: no cover
    def join(self, timeout: float | None = ...) -> None: ...  # pragma: no cover


def _terminate_if_alive(process: _TerminableProcess, timeout: float) -> None:
    """Escalate shutdown so atexit has no live children to block on.

    A bounded ``join`` in the happy path can fall through with the child
    still alive (e.g. a writer stuck on its ``QueueFeederThread`` because
    nothing is reading the stats pipe any more). When ``run_match``
    returns, ``multiprocessing.util._exit_function`` joins every surviving
    child *without a timeout* in atexit — which is the hang reported in
    issue #69. Escalating here (terminate → join → kill → join) ensures
    every child is reaped before the function returns.
    """
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=timeout)
    if not process.is_alive():
        return
    process.kill()
    process.join()


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
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    learned_model_dir: Path | None,
    input_queue: MpQueue[bytes | None],
    output_queue: MpQueue[bytes | None],
    stats_queue: MpQueue[bytes],
    shutdown_event: EventType,
    verbosity: int,
    log_level: str,
    json_logs: bool,
    log_file: Path | None,
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
                "pairing_config": pairing_config,
                "idf": idf,
                "calibrator": calibrator,
                "learned_model_dir": learned_model_dir,
                "input_queue": input_queue,
                "output_queue": output_queue,
                "stats_queue": stats_queue,
                "shutdown_event": shutdown_event,
                "worker_id": index,
                "verbosity": verbosity,
                "log_level": log_level,
                "json_logs": json_logs,
                "log_file": log_file,
            },
        )
        process.start()
        processes.append(process)
    return processes


def _worker_entry(
    *,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    learned_model_dir: Path | None,
    input_queue: MpQueue[bytes | None],
    output_queue: MpQueue[bytes | None],
    stats_queue: MpQueue[bytes],
    shutdown_event: EventType,
    worker_id: int = 0,
    verbosity: int = 0,
    log_level: str = "INFO",
    json_logs: bool = False,
    log_file: Path | None = None,
) -> None:
    """Top-level callable executed inside each spawned worker process.

    Spawned workers start with a pristine interpreter and therefore do NOT
    inherit the parent's ``structlog`` configuration, so logging is
    reconfigured here before the consume loop runs. Otherwise this stays
    thin (it just unpacks queues into get/put callables and defers to
    :func:`worker_main`) so the heavy lifting remains reachable from
    in-process tests, which configure logging themselves.
    """
    configure_logging(level=log_level, json_output=json_logs, log_file=log_file)
    is_shutdown = _shutdown_predicate(shutdown_event)
    worker_main(
        index_path=index_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
        idf=idf,
        calibrator=calibrator,
        learned_model_dir=learned_model_dir,
        input_get=input_queue.get,
        output_put=output_queue.put,
        stats_put=stats_queue.put,
        is_shutdown=is_shutdown,
        worker_id=worker_id,
        verbosity=verbosity,
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


def _resolve_source(
    marc_path: Path | None,
    prepared_dir: Path | None,
) -> Iterator[MarcRecord]:
    """Return the record iterator for exactly one of the two input modes.

    Args:
        marc_path: MARCXML file to stream-parse, or ``None``.
        prepared_dir: Prepared-chunk directory to replay, or ``None``.

    Raises:
        ValueError: If neither or both inputs are supplied.
    """
    if (marc_path is None) == (prepared_dir is None):
        raise ValueError("exactly one of marc_path or prepared_dir is required")
    if prepared_dir is not None:
        return iter_prepared_records(prepared_dir)
    assert marc_path is not None
    return iter_marc_records(marc_path)


def run_match(
    marc_path: Path | None = None,
    *,
    prepared_dir: Path | None = None,
    expected_total: int | None = None,
    index_path: Path,
    output_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None = None,
    learned_model_dir: Path | None = None,
    workers: int | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    queue_maxsize: int | None = None,
    writer_factory: WriterFactory = _build_csv_writer,
    report_interval_seconds: float = _DEFAULT_REPORT_INTERVAL_SECONDS,
    verbosity: int = 0,
    log_level: str = "INFO",
    json_logs: bool = False,
    log_file: Path | None = None,
) -> RunReport:
    """Run the full match pipeline over one input source and return a summary.

    Exactly one of ``marc_path`` (stream-parse MARCXML) or ``prepared_dir``
    (replay pickled chunks) must be supplied. The prepared path is the
    re-runnable production source; pass the manifest's record count as
    ``expected_total`` so the reporter can show percent-complete and ETA.

    Args:
        marc_path: MARCXML file to ingest, or ``None`` when using
            ``prepared_dir``.
        prepared_dir: Prepared-chunk directory (see
            :mod:`pd_matcher.match.prepare`), or ``None`` when using
            ``marc_path``.
        expected_total: Total records the run will process when known.
            Threaded into the reporter to enable percent/ETA; ``None`` omits
            both gracefully (the typical ``--marc`` case).
        index_path: LMDB env directory produced by ``pd-matcher index build``.
        output_path: Destination CSV path.
        matching_config: Loaded :class:`MatchingConfig`.
        pairing_config: Loaded :class:`PairingConfig`; each worker compiles
            it into :class:`CompiledPairings` once at init.
        idf: Pre-built :class:`IdfTable` (workers load it once at init).
        calibrator: Optional Platt calibrator.
        learned_model_dir: Directory holding the learned-model artifact,
            forwarded to each worker so the learned combiner loads its model
            once per process; ``None`` on the default weighted-mean path.
        workers: Number of worker processes. ``None`` uses ``cpu_count - 1``.
        batch_size: MARC records per IPC batch.
        queue_maxsize: Bound on the input queue. ``None`` derives ``workers * 4``.
        writer_factory: Factory producing the per-row CSV writer. Defaults
            to :class:`CsvResultWriter`.
        report_interval_seconds: Reporter cadence.
        verbosity: ``0`` aggregate-only; ``1`` adds per-worker heartbeats;
            ``2`` adds per-record hit lines. Forwarded to each worker.
        log_level: Log level reconfigured inside each spawned worker (which
            do not inherit the parent's logging config).
        json_logs: Whether spawned workers emit JSON logs.
        log_file: Optional log file path reopened in append mode inside each
            spawned worker so worker output is co-located with the parent's.

    Returns:
        A :class:`RunReport` describing the run.
    """
    records = _resolve_source(marc_path, prepared_dir)
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
            expected_total=expected_total,
        ) as reporter,
    ):
        worker_processes = _spawn_workers(
            ctx,
            workers=worker_count,
            index_path=index_path,
            matching_config=matching_config,
            pairing_config=pairing_config,
            idf=idf,
            calibrator=calibrator,
            learned_model_dir=learned_model_dir,
            input_queue=input_queue,
            output_queue=output_queue,
            stats_queue=stats_queue,
            shutdown_event=coord.event,
            verbosity=verbosity,
            log_level=log_level,
            json_logs=json_logs,
            log_file=log_file,
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
            records,
            input_put=input_queue.put,
            stats_put=stats_queue.put,
            is_shutdown=coord.event.is_set,
            batch_size=batch_size,
        )
        for _ in range(worker_count):
            input_queue.put(None)
        for process in worker_processes:
            process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
            _terminate_if_alive(process, _WORKER_JOIN_TIMEOUT_SECONDS)
        output_queue.put(None)
        writer_process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
        _terminate_if_alive(writer_process, _WORKER_JOIN_TIMEOUT_SECONDS)
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
        interrupted=interrupted,
    )


__all__ = [
    "RunReport",
    "run_match",
]
