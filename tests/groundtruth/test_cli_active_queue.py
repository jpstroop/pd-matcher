"""Unit tests for the ``build-active-queue`` CLI command (issue #81).

The heavy ``run_active_learning`` is mocked; these tests pin the option
parsing, the overwrite/rebuild guards, the dry-run wiring, and the summary
rendering — never touching real data or the vault.
"""

from pathlib import Path
from unittest.mock import patch

from pytest import raises
from typer import BadParameter
from typer.testing import CliRunner

from pd_groundtruth.active_learning import ActiveLearningSummary
from pd_groundtruth.active_learning import BucketStats
from pd_groundtruth.active_score import BUCKET_AGREE_HIGH
from pd_groundtruth.active_score import BUCKET_AGREE_LOW
from pd_groundtruth.active_score import BUCKET_INFORMATIVE
from pd_groundtruth.active_select import DEFAULT_LANGUAGE_WEIGHTS
from pd_groundtruth.active_select import LanguagePlan
from pd_groundtruth.cli import _parse_language_weights
from pd_groundtruth.cli import app

_RUNNER = CliRunner()


def _summary(*, written: int, dry_run: bool) -> ActiveLearningSummary:
    return ActiveLearningSummary(
        selected=3,
        excluded=1,
        out_of_scope=2,
        scored=3,
        buckets=(
            BucketStats(
                bucket=BUCKET_INFORMATIVE,
                count=1,
                min_disagreement=0.5,
                max_disagreement=1.8,
                mean_disagreement=1.1,
            ),
            BucketStats(
                bucket=BUCKET_AGREE_HIGH,
                count=1,
                min_disagreement=0.0,
                max_disagreement=0.0,
                mean_disagreement=0.0,
            ),
            BucketStats(
                bucket=BUCKET_AGREE_LOW,
                count=1,
                min_disagreement=0.0,
                max_disagreement=0.0,
                mean_disagreement=0.0,
            ),
        ),
        written=written,
        dry_run=dry_run,
        language_plans=(LanguagePlan(language="eng", target=3, selected=3),),
    )


def test_parse_language_weights_default_is_english_heavy() -> None:
    assert _parse_language_weights(None) == dict(DEFAULT_LANGUAGE_WEIGHTS)
    assert _parse_language_weights([]) == dict(DEFAULT_LANGUAGE_WEIGHTS)


def test_parse_language_weights_parses_tokens() -> None:
    assert _parse_language_weights(["eng=0.6", "fre=0.4"]) == {"eng": 0.6, "fre": 0.4}


def test_parse_language_weights_last_value_wins() -> None:
    assert _parse_language_weights(["eng=0.6", "eng=0.9"]) == {"eng": 0.9}


def test_parse_language_weights_rejects_missing_separator() -> None:
    with raises(BadParameter, match="lang=weight"):
        _parse_language_weights(["eng"])


def test_parse_language_weights_rejects_empty_language() -> None:
    with raises(BadParameter, match="lang=weight"):
        _parse_language_weights(["=0.5"])


def test_parse_language_weights_rejects_non_numeric() -> None:
    with raises(BadParameter, match="not a number"):
        _parse_language_weights(["eng=high"])


def test_parse_language_weights_rejects_non_positive() -> None:
    with raises(BadParameter, match="must be positive"):
        _parse_language_weights(["eng=0"])


def test_command_writes_and_reports(tmp_path: Path) -> None:
    out = tmp_path / "active.db"
    with patch(
        "pd_groundtruth.cli.run_active_learning",
        return_value=_summary(written=1, dry_run=False),
    ) as mock_run:
        result = _RUNNER.invoke(
            app,
            [
                "build-active-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
                "--vault",
                str(tmp_path / "vault.jsonl"),
                "--target",
                "5",
                "--weight",
                "eng=1.0",
            ],
        )
    assert result.exit_code == 0
    _, kwargs = mock_run.call_args
    assert kwargs["target"] == 5
    assert kwargs["weights"] == {"eng": 1.0}
    assert kwargs["dry_run"] is False
    assert "wrote 1 informative pairs" in result.stdout
    assert "informative" in result.stdout


def test_command_dry_run_passes_flag_and_reports_preview(tmp_path: Path) -> None:
    out = tmp_path / "active.db"
    with patch(
        "pd_groundtruth.cli.run_active_learning",
        return_value=_summary(written=0, dry_run=True),
    ) as mock_run:
        result = _RUNNER.invoke(
            app,
            ["build-active-queue", "--out", str(out), "--dry-run"],
        )
    assert result.exit_code == 0
    _, kwargs = mock_run.call_args
    assert kwargs["dry_run"] is True
    assert kwargs["weights"] == dict(DEFAULT_LANGUAGE_WEIGHTS)
    assert "(dry-run)" in result.stdout
    assert "would be written" in result.stdout


def test_command_refuses_to_overwrite_existing_db(tmp_path: Path) -> None:
    out = tmp_path / "active.db"
    out.write_text("existing", encoding="utf-8")
    with patch("pd_groundtruth.cli.run_active_learning") as mock_run:
        result = _RUNNER.invoke(app, ["build-active-queue", "--out", str(out)])
    assert result.exit_code == 2
    assert "already exists" in result.stderr
    mock_run.assert_not_called()


def test_command_rebuild_drops_existing_db(tmp_path: Path) -> None:
    out = tmp_path / "active.db"
    out.write_text("existing", encoding="utf-8")
    with patch(
        "pd_groundtruth.cli.run_active_learning",
        return_value=_summary(written=1, dry_run=False),
    ):
        result = _RUNNER.invoke(app, ["build-active-queue", "--out", str(out), "--rebuild"])
    assert result.exit_code == 0


def test_command_dry_run_ignores_existing_db(tmp_path: Path) -> None:
    out = tmp_path / "active.db"
    out.write_text("existing", encoding="utf-8")
    with patch(
        "pd_groundtruth.cli.run_active_learning",
        return_value=_summary(written=0, dry_run=True),
    ):
        result = _RUNNER.invoke(app, ["build-active-queue", "--out", str(out), "--dry-run"])
    assert result.exit_code == 0
