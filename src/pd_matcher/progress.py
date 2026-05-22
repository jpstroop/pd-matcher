"""Generic throughput/ETA math and a cadence-gated progress reporter.

A long matching or queue-building run can stream tens of thousands of records
with nothing logged between start and the final summary, leaving no way to
judge progress or remaining time. This module supplies a reusable, fully
deterministic progress readout shared by the production matcher and the
groundtruth queue builder.

The *computation* of progress (rate, ETA, percent, ``mm:ss`` formatting — all
pure given inputs, in :class:`ProgressSnapshot`) is separated from the
*emission* of it (the every-N-records or every-few-seconds gating in
:class:`ProgressReporter`). Domain-specific detail is appended by callers via
the ``detail`` hook on :meth:`ProgressReporter.update`, so this module never
depends on any particular domain model.
"""

from collections.abc import Callable
from logging import Logger

from msgspec import Struct

_SECONDS_PER_MINUTE: int = 60
_DEFAULT_EVERY_N: int = 500
_DEFAULT_EVERY_SECONDS: float = 5.0
_NO_ETA: str = "--"


def _format_mmss(seconds: float) -> str:
    """Render a non-negative duration in seconds as ``mm:ss``.

    Minutes are not wrapped at 60, so a 75-second value renders ``01:15``
    and a 3,725-second value renders ``62:05``.
    """
    total = int(seconds)
    minutes, secs = divmod(total, _SECONDS_PER_MINUTE)
    return f"{minutes:02d}:{secs:02d}"


class ProgressSnapshot(Struct, frozen=True, forbid_unknown_fields=True):
    """An immutable view of progress at one instant.

    All derived figures (rate, ETA, percent) are pure functions of the two
    counters plus elapsed wall time, so this struct is fully unit-testable
    without running a workload.
    """

    done: int
    total: int
    elapsed_seconds: float

    @property
    def records_per_second(self) -> float:
        """Aggregate throughput since work started (``0.0`` if idle)."""
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.done / self.elapsed_seconds

    @property
    def records_per_minute(self) -> float:
        """Aggregate throughput expressed per minute."""
        return self.records_per_second * _SECONDS_PER_MINUTE

    @property
    def percent(self) -> int:
        """Completion percentage rounded down (``0`` when ``total`` is 0)."""
        if self.total <= 0:
            return 0
        return int(self.done * 100 // self.total)

    @property
    def eta_seconds(self) -> float | None:
        """Estimated seconds remaining, or ``None`` when not computable.

        ``None`` is returned when the aggregate rate is zero (nothing done
        yet) or when ``done`` already meets or exceeds ``total``.
        """
        rate = self.records_per_second
        if rate <= 0.0 or self.done >= self.total:
            return None
        return (self.total - self.done) / rate

    def render(self) -> str:
        """Render the generic one-line progress readout (see module docstring)."""
        eta = self.eta_seconds
        eta_text = _NO_ETA if eta is None else _format_mmss(eta)
        return (
            f"progress  {self.done:,}/{self.total:,} ({self.percent}%)  "
            f"agg {self.records_per_second:.1f} rec/s ({self.records_per_minute:.0f}/min)  "
            f"elapsed {_format_mmss(self.elapsed_seconds)}  ETA {eta_text}"
        )


class ProgressReporter:
    """Emit a progress log line on a record-count or wall-time cadence.

    A line is logged when either ``every_n`` records have completed since the
    last emission OR ``every_seconds`` have elapsed since it, whichever comes
    first. The clock is injected so the gating logic is testable with a
    fabricated stream and a controllable time source.
    """

    __slots__ = (
        "_clock",
        "_every_n",
        "_every_seconds",
        "_last_emit_count",
        "_last_emit_time",
        "_logger",
        "_start_time",
        "_total",
    )

    def __init__(
        self,
        *,
        logger: Logger,
        total: int,
        clock: Callable[[], float],
        every_n: int = _DEFAULT_EVERY_N,
        every_seconds: float = _DEFAULT_EVERY_SECONDS,
    ) -> None:
        self._logger = logger
        self._total = total
        self._clock = clock
        self._every_n = every_n
        self._every_seconds = every_seconds
        self._start_time = clock()
        self._last_emit_time = self._start_time
        self._last_emit_count = 0

    def _emit(self, done: int, detail: str) -> None:
        """Log one progress line for the current counters."""
        now = self._clock()
        snapshot = ProgressSnapshot(
            done=done,
            total=self._total,
            elapsed_seconds=now - self._start_time,
        )
        line = snapshot.render()
        self._logger.info("%s", f"{line} {detail}" if detail else line)
        self._last_emit_time = now
        self._last_emit_count = done

    def update(self, done: int, *, detail: str = "") -> bool:
        """Maybe emit a progress line; return ``True`` when one was logged.

        Args:
            done: Total records processed so far.
            detail: Optional domain-specific text appended after the generic
                readout (e.g. per-stratum kept counts or by-status totals).
        """
        since_count = done - self._last_emit_count
        since_time = self._clock() - self._last_emit_time
        if since_count >= self._every_n or since_time >= self._every_seconds:
            self._emit(done, detail)
            return True
        return False


__all__ = [
    "ProgressReporter",
    "ProgressSnapshot",
]
