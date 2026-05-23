"""Structlog configuration for pd_matcher.

Provides one public entry point, :func:`configure_logging`, that wires up
``structlog`` and the standard library ``logging`` module so that every log
record carries a timestamp, level, logger name, and any context bound via
``structlog.contextvars`` (e.g. ``marc_id`` and ``worker_id``).

In addition to the console sink (``sys.stderr``, byte-identical to today),
:func:`configure_logging` accepts an optional ``log_file`` path. When
supplied, every emitted record is also written to that file in plain text
(ANSI colors stripped from the console rendering, or JSON one-record-per-line
when ``json_output`` is true). The directory is created if missing and the
file is opened in append mode so spawned workers can share one path.
"""

import sys
from io import TextIOBase
from logging import INFO
from logging import WARNING
from logging import Logger
from logging import StreamHandler
from logging import getLevelNamesMapping
from logging import getLogger
from pathlib import Path
from re import compile as re_compile

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

_ANSI_PATTERN = re_compile(r"\x1b\[[0-9;]*[A-Za-z]")


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


def configure_logging(
    level: str,
    json_output: bool,
    log_file: Path | None = None,
) -> Path | None:
    """Configure ``structlog`` and standard ``logging`` for the process.

    Args:
        level: Minimum log level as a string (e.g. ``"INFO"`` or ``"DEBUG"``).
        json_output: When ``True`` emit JSON one-record-per-line; when
            ``False`` emit a human-readable, colorized console rendering.
        log_file: Optional path that, when supplied, receives every emitted
            record in addition to ``sys.stderr``. The parent directory is
            created if missing and the file is opened in append mode so
            spawned workers can share the same file.

    Returns:
        The resolved ``log_file`` path (unchanged) or ``None`` when no file
        sink was requested.
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

    file_handle: TextIOBase | None = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handle = log_file.open("a", encoding="utf-8", buffering=1)

    configure(
        processors=processors,
        wrapper_class=make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=_DualSinkLoggerFactory(
            file_handle=file_handle,
            strip_ansi=not json_output,
        ),
        cache_logger_on_first_use=False,
    )

    getLogger("asyncio").setLevel(WARNING)
    if numeric_level > INFO:
        getLogger("pd_matcher").setLevel(numeric_level)
    return log_file


class _DualSinkLoggerFactory:
    """Logger factory that writes to ``sys.stderr`` and optionally a file.

    structlog's default ``PrintLoggerFactory`` keeps a reference to the stream
    at construction time, which interferes with ``capsys``/``capfd`` in tests.
    This factory resolves ``sys.stderr`` on every emission so test capture
    works without monkeypatching internals, and additionally fans the
    rendered message to ``file_handle`` (with ANSI escapes stripped when
    ``strip_ansi`` is true).
    """

    __slots__ = ("_file_handle", "_strip_ansi")

    def __init__(self, *, file_handle: TextIOBase | None, strip_ansi: bool) -> None:
        self._file_handle = file_handle
        self._strip_ansi = strip_ansi

    def __call__(self, name: str | None = None) -> _DualSinkLogger:
        return _DualSinkLogger(
            name=name,
            file_handle=self._file_handle,
            strip_ansi=self._strip_ansi,
        )


class _DualSinkLogger:
    """Minimal logger surface required by structlog's filtering wrapper."""

    __slots__ = ("_file_handle", "_strip_ansi", "name")

    def __init__(
        self,
        *,
        name: str | None,
        file_handle: TextIOBase | None,
        strip_ansi: bool,
    ) -> None:
        self.name = name
        self._file_handle = file_handle
        self._strip_ansi = strip_ansi

    def msg(self, message: str) -> None:
        """Write ``message`` to stderr (always) and the file sink (if open)."""
        stream = sys.stderr
        stream.write(message + "\n")
        stream.flush()
        handle = self._file_handle
        if handle is not None:
            text = _ANSI_PATTERN.sub("", message) if self._strip_ansi else message
            handle.write(text + "\n")
            handle.flush()

    debug = msg
    info = msg
    warning = msg
    error = msg
    critical = msg
    log = msg


__all__ = ["BoundLogger", "configure_logging"]
