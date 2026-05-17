"""Tests for :mod:`pd_matcher.logging_config`."""

from json import loads

from _pytest.capture import CaptureFixture
from pytest import mark
from pytest import raises
from structlog import get_logger
from structlog.contextvars import bind_contextvars
from structlog.contextvars import clear_contextvars

from pd_matcher.logging_config import configure_logging


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
