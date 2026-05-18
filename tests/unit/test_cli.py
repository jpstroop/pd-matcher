"""Tests for :mod:`pd_matcher.cli`."""

from typer.testing import CliRunner

from pd_matcher.cli import app

_runner: CliRunner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})


def test_root_help_lists_subcommands() -> None:
    """The top-level ``--help`` should list every registered command."""
    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.stdout
    assert "match" in result.stdout
    assert "eval" in result.stdout


def test_index_help_lists_build_and_info() -> None:
    """The ``index`` sub-app help should list both subcommands."""
    result = _runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "build" in result.stdout
    assert "info" in result.stdout


def test_index_build_help() -> None:
    """``index build --help`` must succeed and mention each option."""
    result = _runner.invoke(app, ["index", "build", "--help"])
    assert result.exit_code == 0
    assert "--reg-dir" in result.stdout
    assert "--ren-dir" in result.stdout
    assert "--out" in result.stdout


def test_index_info_help() -> None:
    """``index info --help`` must succeed and mention its option."""
    result = _runner.invoke(app, ["index", "info", "--help"])
    assert result.exit_code == 0
    assert "--lmdb-path" in result.stdout


def test_match_help_lists_options() -> None:
    """``match --help`` must mention every public option."""
    result = _runner.invoke(app, ["match", "--help"])
    assert result.exit_code == 0
    for flag in ("--marc", "--index", "--out", "--workers", "--year-window", "--min-score"):
        assert flag in result.stdout


def test_eval_help_lists_options() -> None:
    """``eval --help`` must mention every public option."""
    result = _runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    for flag in ("--ground-truth", "--index", "--report"):
        assert flag in result.stdout


def test_index_build_exits_with_not_implemented_code() -> None:
    """``index build`` must parse args and exit 2 (not yet implemented)."""
    result = _runner.invoke(
        app,
        [
            "index",
            "build",
            "--reg-dir",
            "/tmp/reg",
            "--ren-dir",
            "/tmp/ren",
            "--out",
            "/tmp/idx",
        ],
    )
    assert result.exit_code == 2
    assert "not yet implemented" in result.output


def test_index_info_exits_with_not_implemented_code() -> None:
    """``index info`` must parse args and exit 2."""
    result = _runner.invoke(app, ["index", "info", "--lmdb-path", "/tmp/idx"])
    assert result.exit_code == 2
    assert "not yet implemented" in result.output


def test_match_exits_with_not_implemented_code() -> None:
    """``match`` must parse args and exit 2."""
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            "/tmp/a.marcxml",
            "--index",
            "/tmp/idx",
            "--out",
            "/tmp/out.csv",
            "--workers",
            "4",
            "--year-window",
            "3",
            "--min-score",
            "80.0",
        ],
    )
    assert result.exit_code == 2
    assert "not yet implemented" in result.output


def test_eval_exits_with_not_implemented_code() -> None:
    """``eval`` must parse args (with and without ``--report``) and exit 2."""
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            "/tmp/gt.csv",
            "--index",
            "/tmp/idx",
            "--report",
            "/tmp/r.json",
        ],
    )
    assert result.exit_code == 2
    result_no_report = _runner.invoke(
        app,
        ["eval", "--ground-truth", "/tmp/gt.csv", "--index", "/tmp/idx"],
    )
    assert result_no_report.exit_code == 2


def test_root_callback_accepts_json_logs_flag() -> None:
    """The root callback's ``--json-logs`` flag must be wired through."""
    result = _runner.invoke(
        app,
        ["--log-level", "DEBUG", "--json-logs", "index", "info", "--lmdb-path", "/tmp/x"],
    )
    assert result.exit_code == 2
