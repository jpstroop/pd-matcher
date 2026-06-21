"""Reporter thread + :class:`RunningTotals` aggregator for Phase 6 progress.

The reporter is intentionally a thread inside the main process (not a
separate process): it consumes lightweight stats events off a queue,
keeps a running aggregate, and logs a human-readable progress line every
N seconds. Aggregating in-process saves one IPC hop and means the final
:class:`pd_matcher.workers.pool.RunReport` can be assembled by simply
reading the thread's totals after it joins.

The reporter terminates when it dequeues a :class:`ShutdownEvent`; the
orchestrator emits exactly one such event after the writer process has
joined so the last :class:`WriterHeartbeat` is guaranteed to be in the
queue ahead of it.
"""

from collections.abc import Callable
from queue import Empty
from threading import Thread
from time import monotonic
from types import TracebackType
from typing import Protocol
from typing import Self

from msgspec import Struct
from structlog import get_logger

from pd_matcher.progress import ProgressSnapshot
from pd_matcher.workers.events import ProducerHeartbeat
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import RecordSkipped
from pd_matcher.workers.events import StatsEvent
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.events import decode_stats_event

_LOGGER = get_logger(__name__)
_POLL_TIMEOUT_SECONDS: float = 0.1
_SECONDS_PER_MINUTE: int = 60


class EventQueue(Protocol):
    """Structural type for any queue exposing ``get(block=..., timeout=...)``.

    Both :class:`queue.Queue` and :class:`multiprocessing.queues.Queue`
    satisfy this Protocol; the reporter only needs the timed ``get``
    method and is agnostic to which backing implementation is in use.
    """

    def get(  # pragma: no cover
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> bytes:
        """Block up to ``timeout`` seconds for the next event blob."""
        ...


class RunningTotals:
    """Mutable per-thread aggregate of every event the reporter has seen."""

    __slots__ = (
        "records_enqueued",
        "records_processed",
        "records_skipped",
        "records_written",
        "started_at",
        "stop_reason",
    )

    def __init__(self, *, started_at: float) -> None:
        """Initialize all counters to zero and record the start time."""
        self.records_processed: int = 0
        self.records_skipped: int = 0
        self.records_written: int = 0
        self.records_enqueued: int = 0
        self.started_at: float = started_at
        self.stop_reason: str = "running"

    @property
    def records_done(self) -> int:
        """Records that have finished, whether scored or skipped on error."""
        return self.records_processed + self.records_skipped

    def apply(self, event: StatsEvent) -> bool:
        """Fold ``event`` into the running totals.

        Returns:
            ``True`` when the event is a :class:`ShutdownEvent` (the
            reporter should stop polling); ``False`` otherwise.
        """
        if isinstance(event, RecordProcessed):
            self.records_processed += 1
            return False
        if isinstance(event, RecordSkipped):
            self.records_skipped += 1
            return False
        if isinstance(event, ProducerHeartbeat):
            self.records_enqueued = event.records_enqueued
            return False
        if isinstance(event, WriterHeartbeat):
            self.records_written = event.records_written
            return False
        self.stop_reason = event.reason
        return True

    def throughput_per_sec(self, now: float) -> float:
        """Return finished-record throughput, falling back to ``0.0`` early."""
        elapsed = now - self.started_at
        if elapsed <= 0.0:
            return 0.0
        return self.records_done / elapsed

    def eta_seconds(self, total_expected: int | None, now: float) -> float | None:
        """Return remaining-seconds estimate, or ``None`` when unknown."""
        if total_expected is None or total_expected <= self.records_done:
            return None
        rate = self.throughput_per_sec(now)
        if rate <= 0.0:
            return None
        remaining = total_expected - self.records_done
        return remaining / rate

    def snapshot(self) -> TotalsSnapshot:
        """Return an immutable snapshot of the current totals."""
        return TotalsSnapshot(
            records_processed=self.records_processed,
            records_skipped=self.records_skipped,
            records_written=self.records_written,
            records_enqueued=self.records_enqueued,
            duration_seconds=monotonic() - self.started_at,
            stop_reason=self.stop_reason,
        )


class TotalsSnapshot(Struct, frozen=True, forbid_unknown_fields=True):
    """Immutable view of the reporter's aggregate state.

    Exists so the orchestrator can read totals without a lock once the
    reporter has joined; in practice the join happens before the call,
    so the snapshot is just a clean transport object.
    """

    records_processed: int
    records_skipped: int
    records_written: int
    records_enqueued: int
    duration_seconds: float
    stop_reason: str


def _format_detail(totals: RunningTotals) -> str:
    """Render the domain-specific suffix appended to every progress line."""
    return (
        f"written={totals.records_written} "
        f"skipped={totals.records_skipped} "
        f"enqueued={totals.records_enqueued}"
    )


def _format_progress_line(
    totals: RunningTotals,
    now: float,
    expected_total: int | None,
) -> str:
    """Render one progress line, omitting percent/ETA when total is unknown.

    With a known ``expected_total`` the line reuses
    :meth:`ProgressSnapshot.render` (percent + ETA). Without one, a terse
    variant reuses the same rate/elapsed math but prints neither a
    misleading ``0/0 (0%)`` nor a bogus ETA.
    """
    detail = _format_detail(totals)
    if expected_total is not None:
        snapshot = ProgressSnapshot(
            done=totals.records_done,
            total=expected_total,
            elapsed_seconds=now - totals.started_at,
        )
        return f"{snapshot.render()}  {detail}"
    elapsed = now - totals.started_at
    rate = totals.throughput_per_sec(now)
    return (
        f"progress  {totals.records_done:,} done  "
        f"agg {rate:.1f} rec/s ({rate * _SECONDS_PER_MINUTE:.0f}/min)  "
        f"elapsed {_format_mmss(elapsed)}  {detail}"
    )


def _format_mmss(seconds: float) -> str:
    """Render a non-negative duration in seconds as ``mm:ss`` (no 60-wrap)."""
    total = int(seconds)
    minutes, secs = divmod(total, _SECONDS_PER_MINUTE)
    return f"{minutes:02d}:{secs:02d}"


class Reporter:
    """Thread wrapper that consumes events and aggregates totals.

    Use as a context manager — the thread is started on ``__enter__`` and
    joined on ``__exit__``. After exit, :attr:`totals` holds the final
    snapshot (``records_processed``, ``records_written``, etc.).
    """

    __slots__ = ("_expected_total", "_interval", "_logger", "_queue", "_thread", "_totals")

    def __init__(
        self,
        queue: EventQueue,
        *,
        report_interval_seconds: float,
        expected_total: int | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Construct a reporter bound to ``queue``.

        Args:
            queue: A queue that yields msgpack-encoded :data:`StatsEvent`
                blobs. A thread-safe object (e.g. :class:`queue.Queue`
                or a :class:`multiprocessing.Queue`) is required.
            report_interval_seconds: Minimum wall-clock seconds between
                progress log lines.
            expected_total: Total records the run will process when known
                (e.g. a prepared-chunk manifest's count). Enables percent
                and ETA in the progress line; ``None`` omits both gracefully.
            clock: Monotonic clock for unit tests that need to advance
                time deterministically.
        """
        self._queue = queue
        self._interval = report_interval_seconds
        self._expected_total = expected_total
        self._totals = RunningTotals(started_at=clock())
        self._thread = Thread(target=self._run, name="pd_matcher.reporter", daemon=True)
        self._logger = _LOGGER

    @property
    def totals(self) -> RunningTotals:
        """Return the underlying :class:`RunningTotals`."""
        return self._totals

    def __enter__(self) -> Self:
        """Start the consumer thread."""
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Wait for the consumer thread to exit."""
        self._thread.join()

    def _run(self) -> None:
        """Main loop: drain the queue, aggregate, log on a fixed cadence."""
        last_log = monotonic()
        while True:
            try:
                blob = self._queue.get(timeout=_POLL_TIMEOUT_SECONDS)
            except Empty:
                if monotonic() - last_log >= self._interval:
                    self._logger.info(
                        "match.progress",
                        progress=_format_progress_line(
                            self._totals, monotonic(), self._expected_total
                        ),
                    )
                    last_log = monotonic()
                continue
            event = decode_stats_event(blob)
            should_stop = self._totals.apply(event)
            now = monotonic()
            if now - last_log >= self._interval:
                self._logger.info(
                    "match.progress",
                    progress=_format_progress_line(self._totals, now, self._expected_total),
                )
                last_log = now
            if should_stop:
                self._logger.info(
                    "match.progress.final",
                    progress=_format_progress_line(self._totals, now, self._expected_total),
                )
                break


__all__ = [
    "EventQueue",
    "Reporter",
    "RunningTotals",
    "TotalsSnapshot",
]
