"""Structlog configuration for pd_matcher.

Provides one public entry point, :func:`configure_logging`, that wires up
``structlog`` and the standard library ``logging`` module so that every log
record carries a timestamp, level, logger name, and any context bound via
``structlog.contextvars`` (e.g. ``marc_id`` and ``worker_id``).

Terminal and file sinks render the same event independently:

* Terminal (``sys.stderr``) uses a compact, human-readable format with a
  ``HH:MM:SS`` timestamp, ``[level]`` markers (no padding), no event-name
  padding, and a logger label with the redundant ``pd_matcher.`` prefix
  stripped. ANSI colors are emitted when ``sys.stderr`` is a TTY.
* File sink (``log_file``) uses :class:`structlog.processors.LogfmtRenderer`
  by default (no padding, full ISO timestamp, full logger name), which is a
  tight grep- and tool-friendly representation. When ``json_output`` is
  ``True`` both sinks emit JSON one-record-per-line instead.

The file's directory is created if missing and the file is opened in append
mode so spawned workers can share one path.
"""

import sys
from collections.abc import Callable
from io import TextIOBase
from logging import INFO
from logging import WARNING
from logging import Logger
from logging import StreamHandler
from logging import getLevelNamesMapping
from logging import getLogger
from pathlib import Path
from typing import cast

from structlog import configure
from structlog import make_filtering_bound_logger
from structlog.contextvars import merge_contextvars
from structlog.dev import ConsoleRenderer
from structlog.processors import EventRenamer
from structlog.processors import JSONRenderer
from structlog.processors import LogfmtRenderer
from structlog.processors import StackInfoRenderer
from structlog.processors import TimeStamper
from structlog.processors import add_log_level
from structlog.processors import format_exc_info
from structlog.stdlib import BoundLogger
from structlog.stdlib import add_logger_name
from structlog.types import EventDict
from structlog.types import Processor
from structlog.types import WrappedLogger

type _Transformer = Callable[[WrappedLogger, str, EventDict], EventDict]
type _StringRenderer = Callable[[WrappedLogger, str, EventDict], str]

_JSON_RENDERER = cast("_StringRenderer", JSONRenderer())

_LOGGER_PREFIX = "pd_matcher."


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


def _passthrough_renderer(
    _logger: WrappedLogger, _method: str, event_dict: EventDict
) -> tuple[tuple[EventDict], dict[str, object]]:
    """Terminal processor that hands the event_dict to the sink unrendered.

    Returning ``((event_dict,), {})`` instructs structlog to call the
    bound logger method as ``method(event_dict)`` so the per-sink renderers
    in :class:`_DualSinkLogger` can each format the same event independently.
    """
    return ((event_dict,), {})


def _shorten_logger(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Strip the redundant ``pd_matcher.`` prefix from the terminal logger label."""
    name = event_dict.get("logger")
    if isinstance(name, str) and name.startswith(_LOGGER_PREFIX):
        event_dict["logger"] = name[len(_LOGGER_PREFIX) :]
    return event_dict


def configure_logging(
    level: str,
    json_output: bool,
    log_file: Path | None = None,
) -> Path | None:
    """Configure ``structlog`` and standard ``logging`` for the process.

    Args:
        level: Minimum log level as a string (e.g. ``"INFO"`` or ``"DEBUG"``).
        json_output: When ``True`` emit JSON one-record-per-line on both
            sinks; when ``False`` the terminal gets a compact human-readable
            rendering and the file gets logfmt.
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
        StackInfoRenderer(),
        format_exc_info,
    ]

    file_handle: TextIOBase | None = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handle = log_file.open("a", encoding="utf-8", buffering=1)

    terminal_render = _build_terminal_renderer(json_output)
    file_render = _build_file_renderer(json_output)

    configure(
        processors=[*shared_processors, _passthrough_renderer],
        wrapper_class=make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=_DualSinkLoggerFactory(
            file_handle=file_handle,
            terminal_render=terminal_render,
            file_render=file_render,
        ),
        cache_logger_on_first_use=False,
    )

    getLogger("asyncio").setLevel(WARNING)
    if numeric_level > INFO:
        getLogger("pd_matcher").setLevel(numeric_level)
    return log_file


def _build_terminal_renderer(json_output: bool) -> _SinkRenderer:
    """Return the renderer used for ``sys.stderr`` output."""
    if json_output:
        return _SinkRenderer(
            transformers=[TimeStamper(fmt="iso", utc=True), EventRenamer("message")],
            renderer=_JSON_RENDERER,
        )
    return _SinkRenderer(
        transformers=[TimeStamper(fmt="%H:%M:%S", utc=True), _shorten_logger],
        renderer=ConsoleRenderer(
            colors=sys.stderr.isatty(),
            pad_event_to=0,
            pad_level=False,
        ),
    )


def _build_file_renderer(json_output: bool) -> _SinkRenderer:
    """Return the renderer used for the file sink."""
    if json_output:
        return _SinkRenderer(
            transformers=[TimeStamper(fmt="iso", utc=True), EventRenamer("message")],
            renderer=_JSON_RENDERER,
        )
    return _SinkRenderer(
        transformers=[TimeStamper(fmt="iso", utc=True)],
        renderer=LogfmtRenderer(
            key_order=["timestamp", "level", "logger", "event"],
            sort_keys=True,
        ),
    )


class _SinkRenderer:
    """Run a per-sink processor chain producing a single rendered string."""

    __slots__ = ("_renderer", "_transformers")

    def __init__(
        self,
        *,
        transformers: list[_Transformer],
        renderer: _StringRenderer,
    ) -> None:
        self._transformers = transformers
        self._renderer = renderer

    def __call__(self, name: str | None, method: str, event_dict: EventDict) -> str:
        ev: EventDict = dict(event_dict)
        for transformer in self._transformers:
            ev = transformer(name, method, ev)
        return self._renderer(name, method, ev)


class _DualSinkLoggerFactory:
    """Logger factory that writes to ``sys.stderr`` and optionally a file.

    structlog's default ``PrintLoggerFactory`` keeps a reference to the stream
    at construction time, which interferes with ``capsys``/``capfd`` in tests.
    This factory resolves ``sys.stderr`` on every emission so test capture
    works without monkeypatching internals, and additionally fans the event
    to ``file_handle`` using a separate, tighter file-format renderer.
    """

    __slots__ = ("_file_handle", "_file_render", "_terminal_render")

    def __init__(
        self,
        *,
        file_handle: TextIOBase | None,
        terminal_render: _SinkRenderer,
        file_render: _SinkRenderer,
    ) -> None:
        self._file_handle = file_handle
        self._terminal_render = terminal_render
        self._file_render = file_render

    def __call__(self, name: str | None = None) -> _DualSinkLogger:
        return _DualSinkLogger(
            name=name,
            file_handle=self._file_handle,
            terminal_render=self._terminal_render,
            file_render=self._file_render,
        )


class _DualSinkLogger:
    """Minimal logger surface required by structlog's filtering wrapper."""

    __slots__ = ("_file_handle", "_file_render", "_terminal_render", "name")

    def __init__(
        self,
        *,
        name: str | None,
        file_handle: TextIOBase | None,
        terminal_render: _SinkRenderer,
        file_render: _SinkRenderer,
    ) -> None:
        self.name = name
        self._file_handle = file_handle
        self._terminal_render = terminal_render
        self._file_render = file_render

    def msg(self, event_dict: EventDict) -> None:
        """Render ``event_dict`` for stderr (always) and the file sink (if open)."""
        terminal_line = self._terminal_render(self.name, "info", event_dict)
        stream = sys.stderr
        stream.write(terminal_line + "\n")
        stream.flush()
        handle = self._file_handle
        if handle is not None:
            file_line = self._file_render(self.name, "info", event_dict)
            handle.write(file_line + "\n")
            handle.flush()

    debug = msg
    info = msg
    warning = msg
    error = msg
    critical = msg
    log = msg


__all__ = ["BoundLogger", "configure_logging"]
