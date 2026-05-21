"""Unit tests for the typer CLI wiring (acquisition mocked)."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from pd_groundtruth.acquire import AcquireReport
from pd_groundtruth.cli import app

_RUNNER = CliRunner()


def _report() -> AcquireReport:
    return AcquireReport(
        dumps_processed=1,
        records_scanned=10,
        kept_by_language_decade={
            "eng": {1930: 0, 1940: 0, 1950: 4, 1960: 0, 1970: 0},
            "fre": {1930: 0, 1940: 1, 1950: 0, 1960: 0, 1970: 0},
            "ger": {1930: 0, 1940: 0, 1950: 0, 1960: 0, 1970: 0},
            "spa": {1930: 0, 1940: 0, 1950: 0, 1960: 0, 1970: 0},
            "ita": {1930: 0, 1940: 0, 1950: 0, 1960: 0, 1970: 0},
        },
        kept_by_language={"eng": 4, "fre": 1, "ger": 0, "spa": 0, "ita": 0},
        shards_written=2,
        stopped_reason="max_dumps",
    )


def test_acquire_command_passes_arguments_through(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.acquire", return_value=_report()) as mock_acquire:
        result = _RUNNER.invoke(
            app,
            [
                "acquire",
                "--out-dir",
                str(tmp_path / "out"),
                "--manifest-url",
                "https://example.test/m.json",
                "--per-decade-cap",
                "100",
                "--min-year",
                "1931",
                "--max-dumps",
                "1",
            ],
        )

    assert result.exit_code == 0
    mock_acquire.assert_called_once_with(
        out_dir=tmp_path / "out",
        per_decade_cap=100,
        min_year=1931,
        manifest_url="https://example.test/m.json",
        max_dumps=1,
    )
    assert "eng=4" in result.stdout
    assert "stopped_reason=max_dumps" in result.stdout


def test_acquire_command_defaults(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.acquire", return_value=_report()) as mock_acquire:
        result = _RUNNER.invoke(app, ["acquire", "--out-dir", str(tmp_path / "out")])

    assert result.exit_code == 0
    _, kwargs = mock_acquire.call_args
    assert kwargs["per_decade_cap"] == 20000
    assert kwargs["min_year"] == date.today().year - 95
    assert kwargs["max_dumps"] is None
