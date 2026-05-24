"""Tests for :mod:`pd_matcher.logging_config`."""

from json import loads
from pathlib import Path
from re import fullmatch
from re import search

from _pytest.capture import CaptureFixture
from pytest import mark
from pytest import raises
from structlog import get_logger
from structlog.contextvars import bind_contextvars
from structlog.contextvars import clear_contextvars

from pd_matcher.logging_config import configure_logging

_ANSI_RE = r"\x1b\["
_HHMMSS_AT_START = r"^\d{2}:\d{2}:\d{2}\b"
_ISO_TIMESTAMP_AT_START = r"^timestamp=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\b"


@mark.parametrize("level", ["DEBUG", "info", "Warning", "ERROR"])
def test_configure_logging_accepts_case_insensitive_levels(level: str) -> None:
    """``configure_logging`` should accept any case for known level names."""
    configure_logging(level=level, json_output=False)


def test_configure_logging_rejects_unknown_level() -> None:
    """An unknown level name must raise ``ValueError``."""
    with raises(ValueError, match="Unknown log level"):
        configure_logging(level="LOUD", json_output=False)


def test_console_renderer_emits_human_readable_output(capsys: CaptureFixture[str]) -> None:
    """Non-JSON mode should emit a human-readable line containing the event."""
    clear_contextvars()
    configure_logging(level="INFO", json_output=False)
    get_logger("pd_matcher.test").info("hello", foo="bar")
    captured = capsys.readouterr()
    assert "hello" in captured.err
    assert "foo" in captured.err
    assert "bar" in captured.err


def test_json_renderer_emits_parseable_json_with_context(
    capsys: CaptureFixture[str],
) -> None:
    """JSON mode should emit one JSON object containing bound context vars."""
    clear_contextvars()
    configure_logging(level="DEBUG", json_output=True)
    bind_contextvars(marc_id="abc123", worker_id=7)
    try:
        get_logger("pd_matcher.worker").info("processed")
    finally:
        clear_contextvars()
    captured = capsys.readouterr()
    payload: dict[str, object] = loads(captured.err.strip().splitlines()[-1])
    assert payload["message"] == "processed"
    assert payload["marc_id"] == "abc123"
    assert payload["worker_id"] == 7
    assert payload["level"] == "info"
    assert payload["logger"] == "pd_matcher.worker"
    assert "timestamp" in payload


def test_filtering_drops_messages_below_level(capsys: CaptureFixture[str]) -> None:
    """Messages below the configured level should not be rendered."""
    clear_contextvars()
    configure_logging(level="ERROR", json_output=False)
    get_logger("pd_matcher.filter").info("should not appear")
    captured = capsys.readouterr()
    assert "should not appear" not in captured.err


def test_reconfiguration_clears_previous_handlers(capsys: CaptureFixture[str]) -> None:
    """Calling configure_logging twice must not duplicate handler output."""
    clear_contextvars()
    configure_logging(level="INFO", json_output=False)
    configure_logging(level="INFO", json_output=True)
    get_logger("pd_matcher.reconfig").info("once")
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1
    payload: dict[str, object] = loads(lines[0])
    assert payload["message"] == "once"


def test_configure_logging_returns_log_file_path(tmp_path: Path) -> None:
    """The function returns the ``log_file`` argument unchanged."""
    clear_contextvars()
    target = tmp_path / "sub" / "run.log"
    returned = configure_logging(level="INFO", json_output=False, log_file=target)
    assert returned == target


def test_configure_logging_returns_none_without_file() -> None:
    """When no log file is supplied the function returns ``None``."""
    clear_contextvars()
    returned = configure_logging(level="INFO", json_output=False)
    assert returned is None


def test_log_file_uses_logfmt_without_ansi(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """File sink renders logfmt with full ISO timestamp and no ANSI escapes."""
    clear_contextvars()
    target = tmp_path / "run.log"
    configure_logging(level="INFO", json_output=False, log_file=target)
    get_logger("pd_matcher.test").info("hello", foo="bar")
    capsys.readouterr()
    line = target.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert search(_ANSI_RE, line) is None
    assert search(_ISO_TIMESTAMP_AT_START, line) is not None
    assert "level=info" in line
    assert "event=hello" in line
    assert "logger=pd_matcher.test" in line
    assert "foo=bar" in line


def test_log_file_receives_json_lines_when_json_output(
    capsys: CaptureFixture[str], tmp_path: Path
) -> None:
    """File sink captures JSON-rendered lines when ``json_output`` is true."""
    clear_contextvars()
    target = tmp_path / "run.log"
    configure_logging(level="INFO", json_output=True, log_file=target)
    get_logger("pd_matcher.test").info("payload", marc_id="abc")
    capsys.readouterr()
    line = target.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload: dict[str, object] = loads(line)
    assert payload["message"] == "payload"
    assert payload["marc_id"] == "abc"


def test_log_file_creates_parent_directory(tmp_path: Path) -> None:
    """A missing parent directory for ``log_file`` is created on demand."""
    clear_contextvars()
    target = tmp_path / "new" / "nested" / "run.log"
    configure_logging(level="INFO", json_output=False, log_file=target)
    get_logger("pd_matcher.test").info("hello")
    assert target.exists()


def test_log_file_appends_across_reconfigurations(tmp_path: Path) -> None:
    """Reopening the same path appends rather than truncating prior content."""
    clear_contextvars()
    target = tmp_path / "run.log"
    configure_logging(level="INFO", json_output=False, log_file=target)
    get_logger("pd_matcher.test").info("first")
    configure_logging(level="INFO", json_output=False, log_file=target)
    get_logger("pd_matcher.test").info("second")
    contents = target.read_text(encoding="utf-8")
    assert "first" in contents
    assert "second" in contents


def test_terminal_uses_short_timestamp_and_unpadded_event(
    capsys: CaptureFixture[str],
) -> None:
    """Terminal output starts with HH:MM:SS and has no event-name padding."""
    clear_contextvars()
    configure_logging(level="INFO", json_output=False)
    get_logger("pd_matcher.short").info("evt", k="v")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    assert fullmatch(_HHMMSS_AT_START + r".*", line) is not None
    assert "evt  " not in line
    assert "evt " in line or line.endswith("evt") or "evt]" in line or " evt " in line


def test_terminal_level_has_no_padding(capsys: CaptureFixture[str]) -> None:
    """Level markers are wrapped in brackets but not padded to a fixed width."""
    clear_contextvars()
    configure_logging(level="INFO", json_output=False)
    get_logger("pd_matcher.level").info("x")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    assert "[info]" in line
    assert "[info " not in line


def test_terminal_strips_pd_matcher_prefix_from_logger(
    capsys: CaptureFixture[str],
) -> None:
    """The ``pd_matcher.`` prefix is removed from the terminal logger label."""
    clear_contextvars()
    configure_logging(level="INFO", json_output=False)
    get_logger("pd_matcher.index.builder").info("evt")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    assert "[index.builder]" in line
    assert "pd_matcher.index.builder" not in line


def test_terminal_leaves_non_pd_matcher_logger_unchanged(
    capsys: CaptureFixture[str],
) -> None:
    """Logger names that lack the ``pd_matcher.`` prefix are not modified."""
    clear_contextvars()
    configure_logging(level="INFO", json_output=False)
    get_logger("uvicorn.error").info("evt")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    assert "[uvicorn.error]" in line


def test_both_sinks_render_same_event_independently(
    capsys: CaptureFixture[str], tmp_path: Path
) -> None:
    """A single log call produces one terminal line and one file line."""
    clear_contextvars()
    target = tmp_path / "run.log"
    configure_logging(level="INFO", json_output=False, log_file=target)
    get_logger("pd_matcher.dual").info("dual_event", k="v")
    captured = capsys.readouterr()
    terminal_lines = [line for line in captured.err.splitlines() if line.strip()]
    file_lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(terminal_lines) == 1
    assert len(file_lines) == 1
    assert search(_HHMMSS_AT_START, terminal_lines[0]) is not None
    assert "[dual]" in terminal_lines[0]
    assert "dual_event" in terminal_lines[0]
    assert file_lines[0].startswith("timestamp=")
    assert "logger=pd_matcher.dual" in file_lines[0]
    assert "event=dual_event" in file_lines[0]
