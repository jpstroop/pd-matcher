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
        records_kept=4,
        shards_written=1,
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
                "--max-records",
                "100",
                "--max-dumps",
                "1",
            ],
        )

    assert result.exit_code == 0
    mock_acquire.assert_called_once_with(
        out_dir=tmp_path / "out",
        manifest_url="https://example.test/m.json",
        max_records=100,
        max_dumps=1,
    )
    assert "records_kept=4" in result.stdout
    assert "stopped_reason=max_dumps" in result.stdout
