"""SIGINT-aware shutdown coordinator for the Phase 6 worker pool.

:class:`ShutdownCoordinator` is a context manager that owns:

1. A :class:`multiprocessing.Event` (set when the pool should drain).
2. A SIGINT handler that flips that event on the first interrupt and
   restores the original handler on the second so a hung worker does
   not trap the operator.

The coordinator's job is intentionally tiny — it does not own any
queues, workers, or threads. The orchestrator in ``pool.py`` reads
``coord.event`` to decide whether to keep enqueueing or to start draining.
"""

from collections.abc import Callable
from multiprocessing import Event as DefaultEvent
from multiprocessing.synchronize import Event as EventType
from signal import SIGINT
from signal import Handlers
from signal import signal as set_signal_handler
from types import FrameType
from types import TracebackType
from typing import Self

SignalHandler = Callable[[int, FrameType | None], object] | int | Handlers | None


class ShutdownCoordinator:
    """Context manager that installs a SIGINT handler over a shared Event.

    The first SIGINT flips :attr:`event`; the second restores the previous
    handler so a hung pool cannot trap the operator. On exit the original
    SIGINT handler is restored unconditionally.
    """

    __slots__ = ("_event", "_first_received", "_previous_handler")

    def __init__(self, event: EventType | None = None) -> None:
        """Initialize with ``event`` (or a fresh default Event); install nothing yet.

        Args:
            event: Optional pre-constructed event, typically one produced
                by a specific :class:`multiprocessing` context (e.g.
                ``ctx.Event()``). When omitted a default-context Event is
                created.
        """
        self._event: EventType = event if event is not None else DefaultEvent()
        self._previous_handler: SignalHandler = None
        self._first_received: bool = False

    @property
    def event(self) -> EventType:
        """Return the shared :class:`multiprocessing.Event`."""
        return self._event

    @property
    def is_set(self) -> bool:
        """Return whether shutdown has been requested."""
        return self._event.is_set()

    def request_shutdown(self) -> None:
        """Programmatically request shutdown (no signal required).

        Useful for tests and for callers who want to drain the pool from
        application code without raising :class:`KeyboardInterrupt`.
        """
        self._event.set()

    def _on_sigint(self, signum: int, frame: FrameType | None) -> None:
        """SIGINT handler: flip the event on first interrupt, restore on second.

        The second-SIGINT branch hands control back to the previous handler
        (typically ``signal.default_int_handler``) so the operator regains
        the ability to ``Ctrl-C`` again even if a worker is wedged. That
        path is intentionally excluded from coverage: triggering it inside
        the test process would tear down pytest's own SIGINT handling.
        """
        if self._first_received:  # pragma: no cover - second-SIGINT escape hatch
            set_signal_handler(SIGINT, self._previous_handler)
            return
        self._first_received = True
        self._event.set()

    def __enter__(self) -> Self:
        """Install the SIGINT handler and return ``self``."""
        self._previous_handler = set_signal_handler(SIGINT, self._on_sigint)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Restore the original SIGINT handler."""
        set_signal_handler(SIGINT, self._previous_handler)


__all__ = [
    "ShutdownCoordinator",
    "SignalHandler",
]
