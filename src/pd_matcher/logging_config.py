"""Structlog configuration for pd_matcher.

Provides one public entry point, :func:`configure_logging`, that wires up
``structlog`` and the standard library ``logging`` module so that every log
record carries a timestamp, level, logger name, and any context bound via
``structlog.contextvars`` (e.g. ``marc_id`` and ``worker_id``).
"""

import sys
from logging import INFO
from logging import WARNING
from logging import Logger
from logging import StreamHandler
from logging import getLevelNamesMapping
from logging import getLogger

from structlog import configure
from structlog import make_filtering_bound_logger
from structlog.contextvars import merge_contextvars
from structlog.dev import ConsoleRenderer
from structlog.processors import EventRenamer
from structlog.processors import JSONRenderer
from structlog.processors import StackInfoRenderer
from structlog.processors import TimeStamper
from structlog.processors import add_log_level
from structlog.processors import format_exc_info
from structlog.stdlib import BoundLogger
from structlog.stdlib import add_logger_name
from structlog.types import Processor


def _resolve_level(level: str) -> int:
    """Translate a string log level into the corresponding ``logging`` integer.

    Args:
        level: Case-insensitive level name (e.g. ``"INFO"``, ``"DEBUG"``).

    Returns:
        The numeric logging level.

    Raises:
        ValueError: If ``level`` is not a recognized level name.
    """
    mapping = getLevelNamesMapping()
    upper = level.upper()
    if upper not in mapping:
        raise ValueError(f"Unknown log level: {level!r}")
    return mapping[upper]


def configure_logging(level: str, json_output: bool) -> None:
    """Configure ``structlog`` and standard ``logging`` for the process.

    Args:
        level: Minimum log level as a string (e.g. ``"INFO"`` or ``"DEBUG"``).
        json_output: When ``True`` emit JSON one-record-per-line; when
            ``False`` emit a human-readable, colorized console rendering.
    """
    numeric_level = _resolve_level(level)

    root: Logger = getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = StreamHandler(stream=sys.stderr)
    handler.setLevel(numeric_level)
    root.addHandler(handler)
    root.setLevel(numeric_level)

    shared_processors: list[Processor] = [
        merge_contextvars,
        add_log_level,
        add_logger_name,
        TimeStamper(fmt="iso", utc=True),
        StackInfoRenderer(),
        format_exc_info,
    ]

    renderer: Processor
    if json_output:
        renderer = JSONRenderer()
        processors: list[Processor] = [
            *shared_processors,
            EventRenamer("message"),
            renderer,
        ]
    else:
        renderer = ConsoleRenderer(colors=sys.stderr.isatty())
        processors = [*shared_processors, renderer]

    configure(
        processors=processors,
        wrapper_class=make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=_StderrLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    getLogger("asyncio").setLevel(WARNING)
    if numeric_level > INFO:
        getLogger("pd_matcher").setLevel(numeric_level)


class _StderrLoggerFactory:
    """Logger factory that writes to ``sys.stderr`` via a print-like callable.

    structlog's default ``PrintLoggerFactory`` keeps a reference to the stream
    at construction time, which interferes with ``capsys``/``capfd`` in tests.
    This factory resolves ``sys.stderr`` on every emission so test capture
    works without monkeypatching internals.
    """

    __slots__ = ()

    def __call__(self, name: str | None = None) -> _StderrLogger:
        return _StderrLogger(name)


class _StderrLogger:
    """Minimal logger surface required by structlog's filtering wrapper."""

    __slots__ = ("name",)

    def __init__(self, name: str | None) -> None:
        self.name = name

    def msg(self, message: str) -> None:
        """Write ``message`` followed by a newline to the current stderr."""
        stream = sys.stderr
        stream.write(message + "\n")
        stream.flush()

    debug = msg
    info = msg
    warning = msg
    error = msg
    critical = msg
    log = msg


__all__ = ["BoundLogger", "configure_logging"]
