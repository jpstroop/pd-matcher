"""Unit tests for the typer CLI wiring (acquisition mocked)."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from pd_groundtruth.acquire import AcquireReport
from pd_groundtruth.cli import app

_RUNNER = CliRunner()


def test_acquire_command_passes_arguments_through(tmp_path: Path) -> None:
    report = AcquireReport(
        dumps_processed=1,
        records_scanned=10,
        kept_by_language={"eng": 4, "fre": 1, "ger": 0, "spa": 0, "ita": 0},
        shards_written=2,
        stopped_reason="max_dumps",
    )
    with patch("pd_groundtruth.cli.acquire", return_value=report) as mock_acquire:
        result = _RUNNER.invoke(
            app,
            [
                "acquire",
                "--out-dir",
                str(tmp_path / "out"),
                "--manifest-url",
                "https://example.test/m.json",
                "--cap-eng",
                "100",
                "--cap-fre",
                "20",
                "--max-dumps",
                "1",
            ],
        )

    assert result.exit_code == 0
    mock_acquire.assert_called_once_with(
        out_dir=tmp_path / "out",
        caps={"eng": 100, "fre": 20, "ger": 2500, "spa": 2500, "ita": 2500},
        manifest_url="https://example.test/m.json",
        max_dumps=1,
    )
    assert "eng=4" in result.stdout
    assert "stopped_reason=max_dumps" in result.stdout


def test_acquire_command_default_caps(tmp_path: Path) -> None:
    report = AcquireReport(
        dumps_processed=0,
        records_scanned=0,
        kept_by_language={"eng": 0, "fre": 0, "ger": 0, "spa": 0, "ita": 0},
        shards_written=0,
        stopped_reason="dumps_exhausted",
    )
    with patch("pd_groundtruth.cli.acquire", return_value=report) as mock_acquire:
        result = _RUNNER.invoke(app, ["acquire", "--out-dir", str(tmp_path / "out")])

    assert result.exit_code == 0
    _, kwargs = mock_acquire.call_args
    assert kwargs["caps"] == {"eng": 40000, "fre": 2500, "ger": 2500, "spa": 2500, "ita": 2500}
