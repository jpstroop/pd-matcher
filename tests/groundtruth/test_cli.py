"""Unit tests for the typer CLI wiring (acquisition mocked)."""

from datetime import date
from logging import getLogger
from os import chdir
from os import getcwd
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from pd_groundtruth.acquire import AcquireReport
from pd_groundtruth.build_corpus import CorpusReport
from pd_groundtruth.build_queue import BuildSummary
from pd_groundtruth.cli import _configure_logging
from pd_groundtruth.cli import app
from pd_groundtruth.disk_guard import InsufficientDiskSpaceError
from pd_groundtruth.filter import FilterReport
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import default_budget

_RUNNER = CliRunner()


def _build_summary() -> BuildSummary:
    return BuildSummary(
        records_sampled=10,
        records_matched=8,
        pairs_written=5,
        stratum_counts={"eng/ge90": 3, "eng/below": 2},
    )


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
        min_free_space_mb=2048,
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
    assert kwargs["min_free_space_mb"] == 2048


def test_build_queue_command_defaults(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(tmp_path / "review.db"),
            ],
        )

    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["seed"] == 42
    assert kwargs["workers"] == 8
    assert kwargs["sample_per_lang"] == 1500
    assert kwargs["budget"].caps == default_budget().caps
    assert kwargs["vault_path"] == Path("data/training/label_vault.jsonl")
    assert "pairs_written=5" in result.stdout
    assert "eng/ge90=3" in result.stdout


def test_configure_logging_with_explicit_path_writes_to_it(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "explicit.log"
    resolved = _configure_logging("acquire", target)
    assert resolved == target
    getLogger("pd_groundtruth.test").info("hello vault")
    contents = target.read_text(encoding="utf-8")
    assert "hello vault" in contents
    assert "pd_groundtruth.test" in contents


def test_configure_logging_auto_path_lands_under_logs_dir(tmp_path: Path) -> None:
    original = getcwd()
    chdir(tmp_path)
    try:
        resolved = _configure_logging("build-queue", None)
    finally:
        chdir(original)
    assert resolved.parent == Path("logs")
    assert resolved.name.startswith("build-queue_")
    assert resolved.name.endswith(".log")
    assert (tmp_path / resolved).exists()


def test_acquire_command_auto_creates_log_file(tmp_path: Path) -> None:
    original = getcwd()
    chdir(tmp_path)
    try:
        with patch("pd_groundtruth.cli.acquire", return_value=_report()):
            result = _RUNNER.invoke(app, ["acquire", "--out-dir", str(tmp_path / "out")])
    finally:
        chdir(original)
    assert result.exit_code == 0
    log_dir = tmp_path / "logs"
    assert log_dir.is_dir()
    log_files = list(log_dir.glob("acquire_*.log"))
    assert len(log_files) == 1
    contents = log_files[0].read_text(encoding="utf-8")
    assert "INFO" in contents or "WARNING" in contents or contents == ""


def test_acquire_command_honors_explicit_log_file(tmp_path: Path) -> None:
    target = tmp_path / "explicit" / "run.log"
    with patch("pd_groundtruth.cli.acquire", return_value=_report()):
        result = _RUNNER.invoke(
            app,
            [
                "acquire",
                "--out-dir",
                str(tmp_path / "out"),
                "--log-file",
                str(target),
            ],
        )
    assert result.exit_code == 0
    assert target.exists()


def test_build_queue_command_threads_log_file_to_build_queue(tmp_path: Path) -> None:
    target = tmp_path / "queue.log"
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(tmp_path / "review.db"),
                "--log-file",
                str(target),
            ],
        )
    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["log_file"] == target


def _seed_existing_db(path: Path) -> None:
    pair = PairInsert(
        language="eng",
        decade=1950,
        score=0.95,
        band="ge90",
        source="banded",
        marc_control_id="ctrl-x",
        marc_json='{"control_id": "ctrl-x"}',
        marc_title="t",
        marc_author="a",
        marc_publisher="p",
        marc_year=1953,
        nypl_uuid="u-x",
        cce_title="t",
        cce_author="a",
        cce_publishers=None,
        cce_claimants=None,
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json="{}",
    )
    with ReviewDb.connect(path) as db:
        pair_id = db.insert_pair(pair)
        db.add_label(pair_id, VERDICT_MATCH)


def test_build_queue_refuses_to_silently_append_to_non_empty_db(tmp_path: Path) -> None:
    out = tmp_path / "review.db"
    _seed_existing_db(out)
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
            ],
        )
    assert result.exit_code == 2
    assert "already contains 1 pairs" in result.stderr
    assert "--rebuild" in result.stderr
    assert "--append" in result.stderr
    mock_build.assert_not_called()
    assert out.exists()


def test_build_queue_rebuild_drops_existing_db_and_runs(tmp_path: Path) -> None:
    out = tmp_path / "review.db"
    _seed_existing_db(out)
    assert out.exists()
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
                "--rebuild",
            ],
        )
    assert result.exit_code == 0
    mock_build.assert_called_once()
    _, kwargs = mock_build.call_args
    assert kwargs["out_path"] == out


def test_build_queue_append_proceeds_against_non_empty_db(tmp_path: Path) -> None:
    out = tmp_path / "review.db"
    _seed_existing_db(out)
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
                "--append",
            ],
        )
    assert result.exit_code == 0
    mock_build.assert_called_once()
    assert out.exists()


def test_build_queue_succeeds_on_missing_db_with_no_flag(tmp_path: Path) -> None:
    out = tmp_path / "review.db"
    assert not out.exists()
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
            ],
        )
    assert result.exit_code == 0
    mock_build.assert_called_once()


def test_build_queue_succeeds_on_empty_schema_only_db_with_no_flag(tmp_path: Path) -> None:
    out = tmp_path / "review.db"
    with ReviewDb.connect(out):
        pass
    assert out.exists()
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
            ],
        )
    assert result.exit_code == 0
    mock_build.assert_called_once()


def test_build_queue_rejects_rebuild_and_append_together(tmp_path: Path) -> None:
    out = tmp_path / "review.db"
    _seed_existing_db(out)
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
                "--rebuild",
                "--append",
            ],
        )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
    mock_build.assert_not_called()
    assert out.exists()


def test_build_queue_treats_non_empty_db_without_review_pair_table_as_empty(
    tmp_path: Path,
) -> None:
    from sqlite3 import connect as sqlite_connect

    out = tmp_path / "review.db"
    connection = sqlite_connect(out)
    try:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO unrelated DEFAULT VALUES")
        connection.commit()
    finally:
        connection.close()
    assert out.stat().st_size > 0
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(out),
            ],
        )
    assert result.exit_code == 0
    mock_build.assert_called_once()


def test_review_command_invokes_serve_with_cli_args(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    vault_path = tmp_path / "vault.jsonl"
    with patch("pd_groundtruth.cli.serve") as mock_serve:
        result = _RUNNER.invoke(
            app,
            [
                "review",
                "--db",
                str(db_path),
                "--vault",
                str(vault_path),
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
            ],
        )
    assert result.exit_code == 0
    mock_serve.assert_called_once_with(db_path, vault_path, host="0.0.0.0", port=9000)
    assert "serving review UI" in result.stdout


def test_build_queue_command_scales_budget(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(tmp_path / "review.db"),
                "--vault",
                str(tmp_path / "vault.jsonl"),
                "--budget",
                "200",
                "--seed",
                "7",
                "--workers",
                "4",
                "--sample-per-lang",
                "500",
            ],
        )

    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["seed"] == 7
    assert kwargs["workers"] == 4
    assert kwargs["sample_per_lang"] == 500
    assert kwargs["vault_path"] == tmp_path / "vault.jsonl"
    assert kwargs["budget"].total() < default_budget().total()


def test_build_queue_command_without_requeue_passes_empty_frozenset(tmp_path: Path) -> None:
    """Default behavior: no ``--requeue`` flag -> empty frozenset."""
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(tmp_path / "review.db"),
            ],
        )
    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["requeue_verdicts"] == frozenset()


def test_build_queue_command_collects_requeue_flags(tmp_path: Path) -> None:
    """Repeated ``--requeue`` flags collapse into a frozenset of verdicts."""
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(tmp_path / "review.db"),
                "--requeue",
                "unsure",
                "--requeue",
                "no_match",
            ],
        )
    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["requeue_verdicts"] == frozenset({"unsure", "no_match"})


def test_build_queue_command_rejects_invalid_requeue_value(tmp_path: Path) -> None:
    """``--requeue invalid`` fails fast before build_queue is invoked."""
    with patch("pd_groundtruth.cli.build_queue", return_value=_build_summary()) as mock_build:
        result = _RUNNER.invoke(
            app,
            [
                "build-queue",
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "cce.lmdb"),
                "--out",
                str(tmp_path / "review.db"),
                "--requeue",
                "invalid",
            ],
        )
    assert result.exit_code != 0
    assert "invalid" in result.stderr
    mock_build.assert_not_called()


def _filter_report() -> FilterReport:
    return FilterReport(
        scanned=10,
        kept=6,
        dropped=4,
        dropped_by_reason={"not_a_book": 3, "year_out_of_range": 1},
    )


def test_filter_command_passes_arguments_through(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.filter_marcxml", return_value=_filter_report()) as mock_filter:
        result = _RUNNER.invoke(
            app,
            [
                "filter",
                "--input",
                str(tmp_path / "in.marcxml"),
                "--output",
                str(tmp_path / "out.marcxml"),
                "--min-year",
                "1931",
                "--languages",
                "eng, fre",
                "--log-file",
                str(tmp_path / "filter.log"),
            ],
        )

    assert result.exit_code == 0
    mock_filter.assert_called_once_with(
        input_path=tmp_path / "in.marcxml",
        output_path=tmp_path / "out.marcxml",
        min_year=1931,
        languages=frozenset({"eng", "fre"}),
    )
    assert "scanned=10 kept=6 dropped=4" in result.stdout
    assert "not_a_book=3" in result.stdout
    assert "year_out_of_range=1" in result.stdout


def test_filter_command_defaults_to_moving_wall_and_all_languages(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.filter_marcxml", return_value=_filter_report()) as mock_filter:
        result = _RUNNER.invoke(
            app,
            [
                "filter",
                "--input",
                str(tmp_path / "in.marcxml"),
                "--output",
                str(tmp_path / "out.marcxml"),
                "--log-file",
                str(tmp_path / "filter.log"),
            ],
        )

    assert result.exit_code == 0
    _, kwargs = mock_filter.call_args
    assert kwargs["min_year"] == date.today().year - 95
    assert kwargs["languages"] is None


def test_filter_command_rejects_empty_languages(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.filter_marcxml", return_value=_filter_report()) as mock_filter:
        result = _RUNNER.invoke(
            app,
            [
                "filter",
                "--input",
                str(tmp_path / "in.marcxml"),
                "--output",
                str(tmp_path / "out.marcxml"),
                "--languages",
                " , ",
                "--log-file",
                str(tmp_path / "filter.log"),
            ],
        )

    assert result.exit_code != 0
    mock_filter.assert_not_called()


def _corpus_report() -> CorpusReport:
    return CorpusReport(
        dumps_processed=2,
        records_scanned=100,
        kept=60,
        dropped=40,
        dropped_by_reason={"not_a_book": 30, "year_out_of_range": 10},
    )


def test_build_corpus_command_passes_arguments_through(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.build_corpus", return_value=_corpus_report()) as mock_corpus:
        result = _RUNNER.invoke(
            app,
            [
                "build-corpus",
                "--output",
                str(tmp_path / "corpus.marcxml"),
                "--min-year",
                "1931",
                "--languages",
                "eng, fre",
                "--manifest-url",
                "https://example.test/m.json",
                "--max-dumps",
                "3",
                "--log-file",
                str(tmp_path / "corpus.log"),
            ],
        )

    assert result.exit_code == 0
    mock_corpus.assert_called_once_with(
        output_path=tmp_path / "corpus.marcxml",
        min_year=1931,
        languages=frozenset({"eng", "fre"}),
        manifest_url="https://example.test/m.json",
        max_dumps=3,
        min_free_space_mb=2048,
    )
    assert "dumps_processed=2 records_scanned=100 kept=60 dropped=40" in result.stdout
    assert "not_a_book=30" in result.stdout
    assert "year_out_of_range=10" in result.stdout


def test_build_corpus_command_defaults_to_moving_wall_and_all_languages(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.build_corpus", return_value=_corpus_report()) as mock_corpus:
        result = _RUNNER.invoke(
            app,
            [
                "build-corpus",
                "--output",
                str(tmp_path / "corpus.marcxml"),
                "--log-file",
                str(tmp_path / "corpus.log"),
            ],
        )

    assert result.exit_code == 0
    _, kwargs = mock_corpus.call_args
    assert kwargs["min_year"] == date.today().year - 95
    assert kwargs["languages"] is None
    assert kwargs["max_dumps"] is None
    assert kwargs["manifest_url"] == DEFAULT_MANIFEST_URL
    assert kwargs["min_free_space_mb"] == 2048


def test_build_corpus_command_rejects_empty_languages(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.build_corpus", return_value=_corpus_report()) as mock_corpus:
        result = _RUNNER.invoke(
            app,
            [
                "build-corpus",
                "--output",
                str(tmp_path / "corpus.marcxml"),
                "--languages",
                " , ",
                "--log-file",
                str(tmp_path / "corpus.log"),
            ],
        )

    assert result.exit_code != 0
    mock_corpus.assert_not_called()


def test_build_corpus_command_threads_min_free_space(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.build_corpus", return_value=_corpus_report()) as mock_corpus:
        result = _RUNNER.invoke(
            app,
            [
                "build-corpus",
                "--output",
                str(tmp_path / "corpus.marcxml"),
                "--min-free-space-mb",
                "512",
                "--log-file",
                str(tmp_path / "corpus.log"),
            ],
        )

    assert result.exit_code == 0
    _, kwargs = mock_corpus.call_args
    assert kwargs["min_free_space_mb"] == 512


def test_acquire_command_threads_min_free_space(tmp_path: Path) -> None:
    with patch("pd_groundtruth.cli.acquire", return_value=_report()) as mock_acquire:
        result = _RUNNER.invoke(
            app,
            [
                "acquire",
                "--out-dir",
                str(tmp_path / "out"),
                "--min-free-space-mb",
                "256",
            ],
        )

    assert result.exit_code == 0
    _, kwargs = mock_acquire.call_args
    assert kwargs["min_free_space_mb"] == 256


def test_build_corpus_command_exits_nonzero_on_disk_space_error(tmp_path: Path) -> None:
    error = InsufficientDiskSpaceError("insufficient disk space at /tmp: 1.00 MB free")
    error.records_written = 42
    error.dumps_written = 3
    with patch("pd_groundtruth.cli.build_corpus", side_effect=error):
        result = _RUNNER.invoke(
            app,
            [
                "build-corpus",
                "--output",
                str(tmp_path / "corpus.marcxml"),
                "--min-free-space-mb",
                "2048",
                "--log-file",
                str(tmp_path / "corpus.log"),
            ],
        )

    assert result.exit_code == 1
    assert "wrote 42 records across 3 dumps" in result.stderr
    assert "threshold was 2048 MB" in result.stderr
    assert "insufficient disk space" in result.stderr


def test_acquire_command_exits_nonzero_on_disk_space_error(tmp_path: Path) -> None:
    error = InsufficientDiskSpaceError("insufficient disk space at /tmp: 1.00 MB free")
    error.records_written = 17
    error.dumps_written = 2
    out_dir = tmp_path / "out"
    with patch("pd_groundtruth.cli.acquire", side_effect=error):
        result = _RUNNER.invoke(
            app,
            [
                "acquire",
                "--out-dir",
                str(out_dir),
                "--min-free-space-mb",
                "2048",
                "--log-file",
                str(tmp_path / "acquire.log"),
            ],
        )

    assert result.exit_code == 1
    assert "wrote 17 records across 2 dumps" in result.stderr
    assert f"to {out_dir}" in result.stderr
    assert "threshold was 2048 MB" in result.stderr
    assert "insufficient disk space" in result.stderr
