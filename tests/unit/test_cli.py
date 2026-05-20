"""Tests for :mod:`pd_matcher.cli`."""

from csv import DictReader
from json import loads
from pathlib import Path
from typing import Self

from msgspec.msgpack import Encoder
from pytest import MonkeyPatch
from pytest import raises
from typer import BadParameter
from typer.testing import CliRunner

from pd_matcher.cli import _eval_workers_upper_bound
from pd_matcher.cli import _parse_as_of
from pd_matcher.cli import _validate_eval_workers
from pd_matcher.cli import _validate_sample
from pd_matcher.cli import _validate_year_window
from pd_matcher.cli import app
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.builder import build_index
from pd_matcher.match.combiners.calibrator import PlattCalibrator

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_runner: CliRunner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})


def _stage_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the tiny reg/ren fixtures into ``tmp_path`` and return their dirs."""
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    return reg_dir, ren_dir


def _build_index(tmp_path: Path) -> Path:
    """Build a tiny LMDB index in ``tmp_path`` and return its path."""
    reg_dir, ren_dir = _stage_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def _write_ground_truth(path: Path, as_of_year: int) -> None:
    """Write a small ground-truth CSV using fields known to the tiny index."""
    header = (
        "marc_id,marc_title_original,marc_title_normalized,marc_title_stemmed,"
        "marc_author_original,marc_author_normalized,marc_author_stemmed,"
        "marc_main_author_original,marc_main_author_normalized,marc_main_author_stemmed,"
        "marc_publisher_original,marc_publisher_normalized,marc_publisher_stemmed,"
        "marc_year,marc_lccn,marc_lccn_normalized,marc_country_code,marc_language_code,"
        "match_type,match_title,match_title_normalized,match_author,match_author_normalized,"
        "match_publisher,match_publisher_normalized,match_year,match_source_id,match_date,"
        "title_score,author_score,publisher_score,combined_score,year_difference,"
        "copyright_status"
    )
    row = (
        f"marc-aaa,A study of widgets,a study of widgets,studi widget,"
        f"by Smith,by smith,smith,Smith John,smith john,smith john,"
        f"Acme Press,acme press,acme press,"
        f"{as_of_year - 80},,,xxu,eng,"
        f"registration,A study of widgets,a study of widgets,smith john,smith john,"
        f"acme press,acme press,{as_of_year - 80},UUID-0001,1940,100,80,90,90.0,0,"
        f"PD_REGISTERED_NOT_RENEWED\n"
    )
    bogus = (
        "marc-bbb,Unrelated Title,unrelated title,unrelat titl,,"
        ",,,,,,,,1955,,,xxu,eng,"
        ",,,,,,,,,,,,,,GIBBERISH_LABEL\n"
    )
    path.write_text(header + "\n" + row + bogus, encoding="utf-8")


def test_root_help_lists_subcommands() -> None:
    """The top-level ``--help`` should list every registered command."""
    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for token in ("index", "match", "eval", "train-scorer"):
        assert token in result.stdout


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
    for flag in ("--reg-dir", "--ren-dir", "--out", "--force"):
        assert flag in result.stdout


def test_index_info_help() -> None:
    """``index info --help`` must succeed and mention its option."""
    result = _runner.invoke(app, ["index", "info", "--help"])
    assert result.exit_code == 0
    assert "--lmdb-path" in result.stdout


def test_match_help_lists_options() -> None:
    """``match --help`` must mention every public option."""
    result = _runner.invoke(app, ["match", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--marc",
        "--index",
        "--out",
        "--workers",
        "--year-window",
        "--min-score",
        "--as-of",
    ):
        assert flag in result.stdout


def test_eval_help_lists_options() -> None:
    """``eval --help`` must mention every public option."""
    result = _runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--ground-truth",
        "--index",
        "--report",
        "--as-of",
        "--limit",
        "--sample",
        "--seed",
        "--year-window",
    ):
        assert flag in result.stdout


def test_train_scorer_help_lists_phase_9_note() -> None:
    """``train-scorer --help`` must succeed (and mention the Phase 9 placeholder)."""
    result = _runner.invoke(app, ["train-scorer", "--help"])
    assert result.exit_code == 0
    assert "Phase 9" in result.stdout


def test_index_build_succeeds_on_tiny_fixtures(tmp_path: Path) -> None:
    """``index build`` against the tiny fixtures returns code 0 and prints counts."""
    reg_dir, ren_dir = _stage_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    result = _runner.invoke(
        app,
        [
            "index",
            "build",
            "--reg-dir",
            str(reg_dir),
            "--ren-dir",
            str(ren_dir),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "registrations:" in result.stdout
    assert out_path.exists()


def test_index_build_force_rebuilds(tmp_path: Path) -> None:
    """``--force`` triggers a rebuild even when the index is already current."""
    reg_dir, ren_dir = _stage_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    result = _runner.invoke(
        app,
        [
            "index",
            "build",
            "--reg-dir",
            str(reg_dir),
            "--ren-dir",
            str(ren_dir),
            "--out",
            str(out_path),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "skipped: no" in result.stdout


def test_index_build_rejects_missing_reg_dir(tmp_path: Path) -> None:
    """``index build`` fails fast with exit 1 when ``--reg-dir`` is absent."""
    result = _runner.invoke(
        app,
        [
            "index",
            "build",
            "--reg-dir",
            str(tmp_path / "missing"),
            "--ren-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "idx.lmdb"),
        ],
    )
    assert result.exit_code == 1
    assert "--reg-dir" in result.output


def test_index_build_rejects_missing_ren_dir(tmp_path: Path) -> None:
    """``index build`` fails fast with exit 1 when ``--ren-dir`` is absent."""
    reg_dir, _ = _stage_sources(tmp_path)
    result = _runner.invoke(
        app,
        [
            "index",
            "build",
            "--reg-dir",
            str(reg_dir),
            "--ren-dir",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "idx.lmdb"),
        ],
    )
    assert result.exit_code == 1
    assert "--ren-dir" in result.output


def test_index_build_reports_oserror(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """``index build`` surfaces OSError as a runtime failure with exit 1."""
    reg_dir, ren_dir = _stage_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    def _raise(**_kwargs: object) -> object:
        raise OSError("disk full")

    monkeypatch.setattr("pd_matcher.cli.build_index", _raise)
    result = _runner.invoke(
        app,
        [
            "index",
            "build",
            "--reg-dir",
            str(reg_dir),
            "--ren-dir",
            str(ren_dir),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 1
    assert "disk full" in result.output


def test_index_info_succeeds(tmp_path: Path) -> None:
    """``index info`` prints the populated counts for an existing env."""
    index_path = _build_index(tmp_path)
    result = _runner.invoke(app, ["index", "info", "--lmdb-path", str(index_path)])
    assert result.exit_code == 0, result.output
    assert "schema_version:" in result.stdout
    assert "registrations:" in result.stdout


def test_index_info_rejects_missing_path(tmp_path: Path) -> None:
    """``index info`` fails with exit 1 when the env directory is missing."""
    result = _runner.invoke(app, ["index", "info", "--lmdb-path", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_index_info_reports_corrupt_env(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """``index info`` surfaces RuntimeError from incomplete metadata."""
    index_path = _build_index(tmp_path)

    class _BrokenLookup:
        def __init__(self, _path: Path) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def stats(self) -> object:
            raise RuntimeError("meta missing")

    monkeypatch.setattr("pd_matcher.cli.NyplIndexLookup", _BrokenLookup)
    result = _runner.invoke(app, ["index", "info", "--lmdb-path", str(index_path)])
    assert result.exit_code == 1
    assert "meta missing" in result.output


def test_match_runs_against_tiny_fixtures(tmp_path: Path) -> None:
    """``match`` end-to-end produces a CSV with one row per MARC record."""
    index_path = _build_index(tmp_path)
    out_csv = tmp_path / "out.csv"
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(out_csv),
            "--workers",
            "1",
            "--year-window",
            "2",
            "--min-score",
            "30.0",
            "--as-of",
            "2026",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_csv.exists()
    with out_csv.open(encoding="utf-8") as fp:
        rows = list(DictReader(fp))
    assert rows
    assert (tmp_path / "idf.msgpack").exists()


def test_match_loads_calibrator_when_present(tmp_path: Path) -> None:
    """A calibrator at ``<index_parent>/calibrator.msgpack`` is honored."""
    index_path = _build_index(tmp_path)
    encoder = Encoder()
    cal = PlattCalibrator(
        a=-1.0,
        b=0.0,
        trained_at="2026-05-18T00:00:00+00:00",
        n_positive=10,
        n_negative=20,
    )
    (tmp_path / "calibrator.msgpack").write_bytes(encoder.encode(cal))
    out_csv = tmp_path / "out.csv"
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(out_csv),
            "--workers",
            "1",
            "--min-score",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_match_with_default_workers(tmp_path: Path) -> None:
    """``match`` without ``--workers`` uses the worker default (cpu_count - 1)."""
    index_path = _build_index(tmp_path)
    out_csv = tmp_path / "out.csv"
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(out_csv),
            "--min-score",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_match_rejects_bad_as_of(tmp_path: Path) -> None:
    """``match --as-of`` rejects non-integer values with exit 2 on stderr."""
    index_path = _build_index(tmp_path)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
            "--as-of",
            "not-a-year",
        ],
    )
    assert result.exit_code == 2
    assert "four-digit year" in result.output


def test_match_rejects_missing_marc(tmp_path: Path) -> None:
    """``match`` exits 1 when the MARC file is absent."""
    index_path = _build_index(tmp_path)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(tmp_path / "missing.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "--marc" in result.output


def test_match_rejects_missing_index(tmp_path: Path) -> None:
    """``match`` exits 1 when the index directory is absent."""
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(tmp_path / "missing.lmdb"),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "--index" in result.output


def test_match_reports_interrupted(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """When ``run_match`` reports ``interrupted=True`` the CLI exits 130."""
    index_path = _build_index(tmp_path)
    out_csv = tmp_path / "out.csv"

    from pd_matcher.workers.pool import RunReport

    def _fake_run_match(**_kwargs: object) -> RunReport:
        return RunReport(
            records_processed=2,
            records_written=1,
            records_enqueued=3,
            duration_seconds=0.01,
            by_status={},
            interrupted=True,
        )

    monkeypatch.setattr("pd_matcher.cli.run_match", _fake_run_match)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(out_csv),
        ],
    )
    assert result.exit_code == 130
    assert "interrupted" in result.output


def test_match_surfaces_oserror_from_run_match(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """An OSError from ``run_match`` surfaces as exit 1."""
    index_path = _build_index(tmp_path)

    def _raise(**_kwargs: object) -> object:
        raise OSError("pipe broken")

    monkeypatch.setattr("pd_matcher.cli.run_match", _raise)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "pipe broken" in result.output


def test_match_surfaces_idf_oserror(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """An OSError from ``load_or_build_idf`` surfaces as exit 1."""
    index_path = _build_index(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("cannot write idf")

    monkeypatch.setattr("pd_matcher.cli.load_or_build_idf", _raise)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "cannot write idf" in result.output


def test_match_surfaces_matching_config_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A ConfigError during matching-defaults load surfaces as exit 1."""
    index_path = _build_index(tmp_path)
    from pd_matcher.config.loader import ConfigError

    def _raise() -> object:
        raise ConfigError("bad yaml")

    monkeypatch.setattr("pd_matcher.cli._load_default_matching_config", _raise)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "matching defaults" in result.output


def test_match_surfaces_ruleset_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A ConfigError during ruleset load surfaces as exit 1."""
    index_path = _build_index(tmp_path)
    from pd_matcher.config.loader import ConfigError

    def _raise(_path: Path) -> object:
        raise ConfigError("ruleset corrupt")

    monkeypatch.setattr("pd_matcher.cli.load_copyright_rules", _raise)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "copyright rules" in result.output


def test_match_surfaces_pairing_config_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A ConfigError during pairing-defaults load surfaces as exit 1."""
    index_path = _build_index(tmp_path)
    from pd_matcher.config.loader import ConfigError

    def _raise() -> object:
        raise ConfigError("pairings corrupt")

    monkeypatch.setattr("pd_matcher.cli._load_default_pairing_config", _raise)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "pairing defaults" in result.output


def test_eval_runs_against_tiny_index(tmp_path: Path) -> None:
    """``eval`` succeeds against the tiny index + a synthetic GT CSV."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    report_path = tmp_path / "report.json"
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--report",
            str(report_path),
            "--as-of",
            "2026",
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Eval report:" in result.stdout
    assert report_path.exists()
    payload = loads(report_path.read_text(encoding="utf-8"))
    assert payload["rows_evaluated"] == 2


def test_eval_without_report_does_not_write_file(tmp_path: Path) -> None:
    """When ``--report`` is omitted no JSON file is written."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_rejects_bad_as_of(tmp_path: Path) -> None:
    """``eval --as-of`` rejects non-integer values with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--as-of",
            "bogus",
        ],
    )
    assert result.exit_code == 2


def test_eval_rejects_missing_ground_truth(tmp_path: Path) -> None:
    """``eval`` exits 1 when ``--ground-truth`` does not exist."""
    index_path = _build_index(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(tmp_path / "nope.csv"),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 1
    assert "--ground-truth" in result.output


def test_eval_rejects_missing_index(tmp_path: Path) -> None:
    """``eval`` exits 1 when ``--index`` does not exist."""
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(tmp_path / "missing.lmdb"),
        ],
    )
    assert result.exit_code == 1
    assert "--index" in result.output


def test_eval_surfaces_matching_config_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A ConfigError during matching-defaults load surfaces as exit 1."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    from pd_matcher.config.loader import ConfigError

    def _raise() -> object:
        raise ConfigError("bad yaml")

    monkeypatch.setattr("pd_matcher.cli._load_default_matching_config", _raise)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 1
    assert "matching defaults" in result.output


def test_eval_surfaces_pairing_config_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A ConfigError during pairing-defaults load surfaces as exit 1."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    from pd_matcher.config.loader import ConfigError

    def _raise() -> object:
        raise ConfigError("pairings corrupt")

    monkeypatch.setattr("pd_matcher.cli._load_default_pairing_config", _raise)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 1
    assert "pairing defaults" in result.output


def test_eval_surfaces_run_eval_oserror(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """An OSError from ``run_eval`` surfaces as exit 1."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)

    def _raise(**_kwargs: object) -> object:
        raise OSError("io error")

    monkeypatch.setattr("pd_matcher.cli.run_eval", _raise)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 1
    assert "io error" in result.output


def test_eval_year_window_override_threads_through(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """``--year-window`` rebuilds the MatchingConfig before invoking ``run_eval``.

    We capture the ``matching_config`` that the CLI passes to ``run_eval`` and
    assert its ``year_window`` reflects the override (not the default).
    """
    from pd_matcher.eval.ground_truth import EvalReport

    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    captured: dict[str, int] = {}

    def _fake_run_eval(**kwargs: object) -> EvalReport:
        config = kwargs["matching_config"]
        assert isinstance(config, MatchingConfig)
        captured["year_window"] = config.year_window
        return EvalReport(
            rows_evaluated=0,
            rows_with_predicted_match=0,
            rows_with_ground_truth_match=0,
            rows_agreeing=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status_confusion={},
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr("pd_matcher.cli.run_eval", _fake_run_eval)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--year-window",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["year_window"] == 7


def test_eval_accepts_year_window_zero(tmp_path: Path) -> None:
    """``eval --year-window 0`` is the lower bound and is accepted."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--year-window",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_accepts_year_window_five(tmp_path: Path) -> None:
    """``eval --year-window 5`` (typical override) is accepted."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--year-window",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_accepts_year_window_upper_bound(tmp_path: Path) -> None:
    """``eval --year-window 100`` is the upper bound and is accepted."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--year-window",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_rejects_year_window_below_zero(tmp_path: Path) -> None:
    """``eval --year-window -1`` is rejected with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--year-window",
            "-1",
        ],
    )
    assert result.exit_code == 2
    assert "year-window" in result.output


def test_eval_rejects_year_window_above_upper_bound(tmp_path: Path) -> None:
    """``eval --year-window 101`` is rejected with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--year-window",
            "101",
        ],
    )
    assert result.exit_code == 2
    assert "year-window" in result.output


def test_eval_rejects_year_window_non_integer(tmp_path: Path) -> None:
    """``eval --year-window abc`` is rejected by typer's parser with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--year-window",
            "abc",
        ],
    )
    assert result.exit_code == 2


def test_eval_accepts_sample(tmp_path: Path) -> None:
    """``eval --sample 100`` is accepted and runs to completion."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_rejects_sample_zero(tmp_path: Path) -> None:
    """``eval --sample 0`` is rejected with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "sample" in result.output


def test_eval_rejects_sample_negative(tmp_path: Path) -> None:
    """``eval --sample -1`` is rejected with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "-1",
        ],
    )
    assert result.exit_code == 2
    assert "sample" in result.output


def test_eval_rejects_sample_non_integer(tmp_path: Path) -> None:
    """``eval --sample abc`` is rejected by typer's parser with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "abc",
        ],
    )
    assert result.exit_code == 2


def test_eval_accepts_seed(tmp_path: Path) -> None:
    """``eval --sample 100 --seed 42`` is accepted."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "100",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_accepts_default_seed(tmp_path: Path) -> None:
    """Omitting ``--seed`` is fine; the default is used."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_rejects_negative_seed(tmp_path: Path) -> None:
    """``--seed -1`` is rejected with exit 2 (non-negative integers only)."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "100",
            "--seed",
            "-1",
        ],
    )
    assert result.exit_code == 2
    assert "seed" in result.output


def test_eval_rejects_sample_and_limit_together(tmp_path: Path) -> None:
    """``--sample`` and ``--limit`` are mutually exclusive."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--sample",
            "100",
            "--limit",
            "50",
        ],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_validate_year_window_passes_none() -> None:
    """``_validate_year_window(None)`` returns ``None`` (flag omitted)."""
    assert _validate_year_window(None) is None


def test_validate_year_window_accepts_within_range() -> None:
    """``_validate_year_window`` accepts values in ``[0, 100]``."""
    assert _validate_year_window(0) == 0
    assert _validate_year_window(50) == 50
    assert _validate_year_window(100) == 100


def test_validate_year_window_rejects_below_zero() -> None:
    """``_validate_year_window(-1)`` raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="year-window"):
        _validate_year_window(-1)


def test_validate_year_window_rejects_above_upper_bound() -> None:
    """``_validate_year_window(101)`` raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="year-window"):
        _validate_year_window(101)


def test_validate_sample_passes_none() -> None:
    """``_validate_sample(None)`` returns ``None`` (flag omitted)."""
    assert _validate_sample(None) is None


def test_validate_sample_accepts_positive() -> None:
    """``_validate_sample`` accepts any value ``>= 1``."""
    assert _validate_sample(1) == 1
    assert _validate_sample(10_000) == 10_000


def test_validate_sample_rejects_zero() -> None:
    """``_validate_sample(0)`` raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="sample"):
        _validate_sample(0)


def test_validate_sample_rejects_negative() -> None:
    """``_validate_sample(-1)`` raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="sample"):
        _validate_sample(-1)


def test_validate_eval_workers_accepts_default_one() -> None:
    """``_validate_eval_workers(1)`` is the lower bound and is accepted."""
    assert _validate_eval_workers(1) == 1


def test_validate_eval_workers_accepts_upper_bound() -> None:
    """``_validate_eval_workers`` accepts ``cpu_count() * 2`` (the upper bound)."""
    upper = _eval_workers_upper_bound()
    assert _validate_eval_workers(upper) == upper


def test_validate_eval_workers_rejects_zero() -> None:
    """``_validate_eval_workers(0)`` raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="workers"):
        _validate_eval_workers(0)


def test_validate_eval_workers_rejects_negative() -> None:
    """``_validate_eval_workers(-1)`` raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="workers"):
        _validate_eval_workers(-1)


def test_validate_eval_workers_rejects_above_upper_bound() -> None:
    """``_validate_eval_workers(cpu_count*2 + 1)`` raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="workers"):
        _validate_eval_workers(_eval_workers_upper_bound() + 1)


def test_eval_workers_upper_bound_falls_back_when_cpu_count_returns_none(
    monkeypatch: MonkeyPatch,
) -> None:
    """When ``os.cpu_count`` returns ``None`` the upper bound falls back to ``2``."""
    monkeypatch.setattr("pd_matcher.cli.cpu_count", lambda: None)
    assert _eval_workers_upper_bound() == 2


def test_eval_workers_flag_threads_through(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """``--workers N`` is forwarded to ``run_eval`` unchanged."""
    from pd_matcher.eval.ground_truth import EvalReport

    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    captured: dict[str, int] = {}

    def _fake_run_eval(**kwargs: object) -> EvalReport:
        workers = kwargs["workers"]
        assert isinstance(workers, int)
        captured["workers"] = workers
        return EvalReport(
            rows_evaluated=0,
            rows_with_predicted_match=0,
            rows_with_ground_truth_match=0,
            rows_agreeing=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status_confusion={},
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr("pd_matcher.cli.run_eval", _fake_run_eval)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["workers"] == 2


def test_eval_default_workers_is_one(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Omitting ``--workers`` defaults to ``1`` (single-process)."""
    from pd_matcher.eval.ground_truth import EvalReport

    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    captured: dict[str, int] = {}

    def _fake_run_eval(**kwargs: object) -> EvalReport:
        workers = kwargs["workers"]
        assert isinstance(workers, int)
        captured["workers"] = workers
        return EvalReport(
            rows_evaluated=0,
            rows_with_predicted_match=0,
            rows_with_ground_truth_match=0,
            rows_agreeing=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status_confusion={},
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr("pd_matcher.cli.run_eval", _fake_run_eval)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["workers"] == 1


def test_eval_rejects_workers_zero(tmp_path: Path) -> None:
    """``eval --workers 0`` is rejected with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--workers",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "workers" in result.output


def test_eval_rejects_workers_above_upper_bound(tmp_path: Path) -> None:
    """``eval --workers <cpu_count*2 + 1>`` is rejected with exit 2."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path, 2026)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--ground-truth",
            str(gt_path),
            "--index",
            str(index_path),
            "--workers",
            str(_eval_workers_upper_bound() + 1),
        ],
    )
    assert result.exit_code == 2
    assert "workers" in result.output


def test_train_scorer_returns_phase_9_stub() -> None:
    """``train-scorer`` exits 2 with the Phase 9 placeholder message."""
    result = _runner.invoke(app, ["train-scorer"])
    assert result.exit_code == 2
    assert "Phase 9" in result.output


def test_root_callback_accepts_log_flags() -> None:
    """The root callback honors ``--log-level``, ``--json-logs`` and ``--quiet``."""
    result = _runner.invoke(
        app,
        [
            "--log-level",
            "DEBUG",
            "--json-logs",
            "--quiet",
            "index",
            "info",
            "--lmdb-path",
            "/tmp/x",
        ],
    )
    assert result.exit_code == 1


def test_as_of_past_year_affects_moving_wall(tmp_path: Path) -> None:
    """``--as-of`` in the deep past short-circuits the moving wall.

    The moving wall (Phase 5) fires when ``pub_year < as_of_year - 95``.
    Setting ``--as-of 1950`` makes the cutoff 1855 — no record in the
    tiny fixtures qualifies — so no row carries
    ``PD_BY_AGE_PRE_95_YEARS``. With ``--as-of 2100`` the cutoff is
    2005, the parseable-year records all qualify, and at least one row
    carries that status.
    """
    index_path = _build_index(tmp_path)
    out_early = tmp_path / "early.csv"
    early = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(out_early),
            "--workers",
            "1",
            "--min-score",
            "1.0",
            "--as-of",
            "1950",
        ],
    )
    assert early.exit_code == 0, early.output
    with out_early.open(encoding="utf-8") as fp:
        early_statuses = {row["copyright_status"] for row in DictReader(fp)}
    assert "PD_BY_AGE_PRE_95_YEARS" not in early_statuses

    out_late = tmp_path / "late.csv"
    late = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--index",
            str(index_path),
            "--out",
            str(out_late),
            "--workers",
            "1",
            "--min-score",
            "1.0",
            "--as-of",
            "2100",
        ],
    )
    assert late.exit_code == 0, late.output
    with out_late.open(encoding="utf-8") as fp:
        late_statuses = {row["copyright_status"] for row in DictReader(fp)}
    assert "PD_BY_AGE_PRE_95_YEARS" in late_statuses


def test_parse_as_of_none_returns_current_year() -> None:
    """``_parse_as_of(None)`` returns the current calendar year."""
    from datetime import date as _date

    assert _parse_as_of(None) == _date.today().year


def test_parse_as_of_accepts_valid_year() -> None:
    """``_parse_as_of('2026')`` returns the int ``2026``."""
    assert _parse_as_of("2026") == 2026


def test_parse_as_of_rejects_non_integer() -> None:
    """Non-integer input raises :class:`typer.BadParameter`."""
    with raises(BadParameter, match="four-digit year"):
        _parse_as_of("not-a-number")


def test_parse_as_of_rejects_below_lower_bound() -> None:
    """Year below 1923 is rejected."""
    with raises(BadParameter, match="between 1923 and 2100"):
        _parse_as_of("1922")


def test_parse_as_of_rejects_above_upper_bound() -> None:
    """Year above 2100 is rejected."""
    with raises(BadParameter, match="between 1923 and 2100"):
        _parse_as_of("2101")


def test_parse_as_of_rejects_typo_out_of_range() -> None:
    """A five-digit typo like ``20270`` is rejected by the upper-bound check."""
    with raises(BadParameter, match="between 1923 and 2100"):
        _parse_as_of("20270")
