"""Pure throughput/ETA math and a cadence-gated progress reporter.

The matching phase of :mod:`pd_groundtruth.build_queue` streams
:class:`~pd_groundtruth.build_queue.WorkerOutcome` results from a spawn pool
back into a :class:`~pd_groundtruth.sampling.Stratifier`. A long run (~30
minutes) previously emitted nothing between the sampling logs and the final
summary, so there was no way to judge progress or remaining time.

This module separates the *computation* of progress (rate, ETA, percent,
``mm:ss`` formatting — all pure and deterministic given inputs, in
:class:`ProgressSnapshot`) from the *emission* of it (the every-N-records or
every-few-seconds gating in :class:`ProgressReporter`). The former is unit
tested without a pool; the latter is exercised with fabricated outcome
streams and an injected clock.
"""

from collections.abc import Callable
from logging import Logger

from msgspec import Struct

from pd_groundtruth.sampling import BAND_BELOW
from pd_groundtruth.sampling import BudgetModel
from pd_groundtruth.sampling import iter_capped_bands

_SECONDS_PER_MINUTE: int = 60
_DEFAULT_EVERY_N: int = 500
_DEFAULT_EVERY_SECONDS: float = 5.0
_NO_ETA: str = "--"
_ALL_BANDS: tuple[str, ...] = (*iter_capped_bands(), BAND_BELOW)


def _format_mmss(seconds: float) -> str:
    """Render a non-negative duration in seconds as ``mm:ss``.

    Minutes are not wrapped at 60, so a 75-second value renders ``01:15``
    and a 3,725-second value renders ``62:05``.
    """
    total = int(seconds)
    minutes, secs = divmod(total, _SECONDS_PER_MINUTE)
    return f"{minutes:02d}:{secs:02d}"


class ProgressSnapshot(Struct, frozen=True, forbid_unknown_fields=True):
    """An immutable view of matching progress at one instant.

    All derived figures (rate, ETA, percent) are pure functions of the three
    counters plus elapsed wall time, so this struct is fully unit-testable
    without running a pool.
    """

    done: int
    total: int
    elapsed_seconds: float
    kept_by_stratum: dict[tuple[str, str], int]

    @property
    def records_per_second(self) -> float:
        """Aggregate throughput since matching started (``0.0`` if idle)."""
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

    def _render_kept(self, budget: BudgetModel) -> str:
        """Render kept-per-stratum counts compactly, English broken out.

        The first configured language gets a per-band breakdown with caps
        (``eng[ge90 412/500 ...]``); every other language collapses to a
        single running total (``fre 140``).
        """
        languages = budget.languages()
        if not languages:
            return ""
        lead = languages[0]
        bands = " ".join(
            f"{band} {self.kept_by_stratum.get((lead, band), 0)}/{budget.cap_for(lead, band)}"
            for band in _ALL_BANDS
        )
        parts = [f"{lead}[{bands}]"]
        for language in languages[1:]:
            total = sum(self.kept_by_stratum.get((language, band), 0) for band in _ALL_BANDS)
            parts.append(f"{language} {total}")
        return " ".join(parts)

    def render(self, budget: BudgetModel) -> str:
        """Render the one-line progress readout (see module docstring)."""
        eta = self.eta_seconds
        eta_text = _NO_ETA if eta is None else _format_mmss(eta)
        return (
            f"progress  {self.done:,}/{self.total:,} ({self.percent}%)  "
            f"agg {self.records_per_second:.1f} rec/s ({self.records_per_minute:.0f}/min)  "
            f"elapsed {_format_mmss(self.elapsed_seconds)}  ETA {eta_text}  "
            f"kept: {self._render_kept(budget)}"
        )


def format_progress(
    done: int,
    total: int,
    elapsed_seconds: float,
    kept_by_stratum: dict[tuple[str, str], int],
    budget: BudgetModel,
) -> str:
    """Build and render a :class:`ProgressSnapshot` in one call."""
    return ProgressSnapshot(
        done=done,
        total=total,
        elapsed_seconds=elapsed_seconds,
        kept_by_stratum=kept_by_stratum,
    ).render(budget)


class ProgressReporter:
    """Emit a progress log line on a record-count or wall-time cadence.

    A line is logged when either ``every_n`` records have completed since the
    last emission OR ``every_seconds`` have elapsed since it, whichever comes
    first. The clock is injected so the gating logic is testable with a
    fabricated stream and a controllable time source.
    """

    __slots__ = (
        "_budget",
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
        budget: BudgetModel,
        clock: Callable[[], float],
        every_n: int = _DEFAULT_EVERY_N,
        every_seconds: float = _DEFAULT_EVERY_SECONDS,
    ) -> None:
        self._logger = logger
        self._total = total
        self._budget = budget
        self._clock = clock
        self._every_n = every_n
        self._every_seconds = every_seconds
        self._start_time = clock()
        self._last_emit_time = self._start_time
        self._last_emit_count = 0

    def _emit(self, done: int, kept_by_stratum: dict[tuple[str, str], int]) -> None:
        """Log one progress line for the current counters."""
        now = self._clock()
        snapshot = ProgressSnapshot(
            done=done,
            total=self._total,
            elapsed_seconds=now - self._start_time,
            kept_by_stratum=kept_by_stratum,
        )
        self._logger.info("%s", snapshot.render(self._budget))
        self._last_emit_time = now
        self._last_emit_count = done

    def update(self, done: int, kept_by_stratum: dict[tuple[str, str], int]) -> bool:
        """Maybe emit a progress line; return ``True`` when one was logged.

        Args:
            done: Total records matched so far.
            kept_by_stratum: Running accepted counts per ``(language, band)``.
        """
        since_count = done - self._last_emit_count
        since_time = self._clock() - self._last_emit_time
        if since_count >= self._every_n or since_time >= self._every_seconds:
            self._emit(done, kept_by_stratum)
            return True
        return False


__all__ = [
    "ProgressReporter",
    "ProgressSnapshot",
    "format_progress",
]
