"""Writer process body for the Phase 6 pool.

A single dedicated writer process drains the output queue, decodes each
:class:`WorkerOutput`, and emits one record through the supplied
:class:`pd_matcher.output.jsonl_writer.ResultWriter`. Serialising the write
side eliminates inter-process contention on the destination file: workers
queue, the writer flushes.

The writer terminates when it dequeues the ``None`` poison pill placed
on the output queue by the orchestrator after every worker has joined.
A final :class:`WriterHeartbeat` is emitted before exit so the reporter
always sees the canonical record count.

The writer is currently tied to :class:`pd_matcher.output.jsonl_writer.ResultWriter`'s
three-argument signature ``(marc, match, matched_nypl)`` — the linkage JSONL.
The groundtruth queue builder swaps in its own writer factory.
"""

from collections.abc import Callable
from pathlib import Path
from time import monotonic

from structlog import get_logger

from pd_matcher.output.jsonl_writer import ResultWriter
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.events import encode_stats_event
from pd_matcher.workers.messages import decode_worker_output

_LOGGER = get_logger(__name__)

type WriterFactory = Callable[[Path], ResultWriter]


def run_writer_loop(
    *,
    writer: ResultWriter,
    output_get: Callable[[], bytes | None],
    stats_put: Callable[[bytes], None],
    is_shutdown: Callable[[], bool],
    heartbeat_interval_seconds: float = 5.0,
    clock: Callable[[], float] = monotonic,
) -> int:
    """Drain the output queue and forward rows to ``writer``.

    Args:
        writer: An already-entered :class:`ResultWriter` (the caller owns
            the lifecycle).
        output_get: Zero-arg callable returning the next blob or ``None``
            to terminate. Usually ``output_queue.get``.
        stats_put: Callable that pushes a stats event blob.
        is_shutdown: Zero-arg callable returning ``True`` when shutdown is
            requested. Checked after every dequeue.
        heartbeat_interval_seconds: Minimum wall-clock seconds between
            :class:`WriterHeartbeat` emissions.
        clock: Monotonic clock for deterministic unit tests.

    Returns:
        The number of rows written.
    """
    written = 0
    last_heartbeat = clock()
    while True:
        blob = output_get()
        if blob is None:
            break
        payload = decode_worker_output(blob)
        writer.write(
            payload.marc,
            payload.match,
            payload.matched_nypl,
        )
        written += 1
        now = clock()
        if now - last_heartbeat >= heartbeat_interval_seconds:
            stats_put(encode_stats_event(WriterHeartbeat(records_written=written)))
            last_heartbeat = now
        if is_shutdown():
            break
    stats_put(encode_stats_event(WriterHeartbeat(records_written=written)))
    return written


def writer_main(
    *,
    output_path: Path,
    writer_factory: WriterFactory,
    output_get: Callable[[], bytes | None],
    stats_put: Callable[[bytes], None],
    is_shutdown: Callable[[], bool],
) -> int:
    """Top-level writer entry point.

    Constructs a :class:`ResultWriter` from ``writer_factory(output_path)``,
    enters it as a context manager, and runs :func:`run_writer_loop` to
    completion.
    """
    writer = writer_factory(output_path)
    with writer as active:
        written = run_writer_loop(
            writer=active,
            output_get=output_get,
            stats_put=stats_put,
            is_shutdown=is_shutdown,
        )
    _LOGGER.info("writer.complete", records_written=written)
    return written


__all__ = [
    "WriterFactory",
    "run_writer_loop",
    "writer_main",
]
