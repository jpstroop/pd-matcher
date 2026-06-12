"""Tests for :mod:`pd_matcher.cli`."""

from csv import DictReader
from json import loads
from pathlib import Path
from typing import Literal
from typing import Self

from msgspec.msgpack import Encoder
from pytest import MonkeyPatch
from typer.testing import CliRunner

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import upsert_entry
from pd_matcher.cli import _learned_model_dir
from pd_matcher.cli import _resolve_log_file
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


_MARCXML_PROLOG = (
    "<?xml version='1.0' encoding='UTF-8'?>\n<collection xmlns='http://www.loc.gov/MARC21/slim'>\n"
)
_MARCXML_EPILOG = "</collection>\n"


def _marc_record_xml(
    control_id: str,
    title: str,
    author: str,
    publisher: str,
    year: str,
    *,
    language: str = "eng",
) -> str:
    """Return one MARCXML <record> element synthesized for tests."""
    return (
        "  <record>\n"
        f"    <controlfield tag='001'>{control_id}</controlfield>\n"
        f"    <controlfield tag='008'>200718s{year}    xxu           000 0 {language}  </controlfield>\n"  # noqa: E501
        f"    <datafield ind1='1' ind2=' ' tag='100'><subfield code='a'>{author}</subfield></datafield>\n"  # noqa: E501
        f"    <datafield ind1='0' ind2='0' tag='245'><subfield code='a'>{title}</subfield></datafield>\n"  # noqa: E501
        f"    <datafield ind1=' ' ind2=' ' tag='260'><subfield code='a'>New York :</subfield><subfield code='b'>{publisher},</subfield><subfield code='c'>{year}.</subfield></datafield>\n"  # noqa: E501
        "  </record>\n"
    )


def _write_pool(pool_root: Path) -> None:
    """Write a tiny eng/widgets.xml shard mirroring UUID-0001 in the tiny index."""
    eng = pool_root / "eng"
    eng.mkdir(parents=True)
    body = _marc_record_xml(
        control_id="marc-aaa",
        title="A study of widgets",
        author="Smith, John",
        publisher="Acme Press",
        year="1940",
    )
    (eng / "shard.xml").write_text(
        _MARCXML_PROLOG + body + _MARCXML_EPILOG,
        encoding="utf-8",
    )


def _write_vault(vault_path: Path) -> None:
    """Append two entries to ``vault_path``: one match + one no_match."""
    match_entry = VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id="marc-aaa",
        nypl_uuid="UUID-0001",
        verdict="match",
        note=None,
        labeled_at="2026-05-22T10:00:00+00:00",
        labeler="test",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )
    no_match_entry = VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id="marc-aaa",
        nypl_uuid="UUID-0002",
        verdict="no_match",
        note=None,
        labeled_at="2026-05-22T10:05:00+00:00",
        labeler="test",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )
    upsert_entry(vault_path, match_entry)
    upsert_entry(vault_path, no_match_entry)


def _prepare_vault_and_pool(tmp_path: Path) -> tuple[Path, Path]:
    """Stand up a vault JSONL + pool dir; return ``(vault, pool)``."""
    vault_path = tmp_path / "vault.jsonl"
    pool_path = tmp_path / "pool"
    _write_vault(vault_path)
    _write_pool(pool_path)
    return vault_path, pool_path


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
        "--prepared",
        "--verbose",
        "--index",
        "--out",
        "--workers",
        "--year-window",
        "--min-score",
    ):
        assert flag in result.stdout


def test_eval_help_lists_options() -> None:
    """``eval --help`` must mention every public option."""
    result = _runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--vault",
        "--pool",
        "--index",
        "--report",
        "--year-window",
    ):
        assert flag in result.stdout


def test_train_scorer_help_lists_phase_9_note() -> None:
    """``train-scorer --help`` must succeed (and mention the Phase 9 placeholder)."""
    result = _runner.invoke(app, ["train-scorer", "--help"])
    assert result.exit_code == 0
    assert "Phase 9" in result.stdout


def _matching_config(scorer: Literal["weighted_mean", "learned"]) -> MatchingConfig:
    """A valid matching config (weights sum to 1.0) with ``scorer`` set."""
    return MatchingConfig(
        title_weight=0.35,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.0,
        extent_weight=0.05,
        volume_weight=0.05,
        year_window=0,
        min_combined_score=0.0,
        scorer=scorer,
    )


def test_learned_model_dir_none_on_default_scorer(tmp_path: Path) -> None:
    """The weighted-mean path never resolves a learned-model directory."""
    assert _learned_model_dir(tmp_path, _matching_config("weighted_mean")) is None


def test_learned_model_dir_returns_parent_for_learned_scorer(tmp_path: Path) -> None:
    """The learned scorer resolves the artifact directory to the index parent."""
    assert _learned_model_dir(tmp_path, _matching_config("learned")) == tmp_path


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


def test_prepare_marc_help_lists_options() -> None:
    """``prepare-marc --help`` must mention every public option."""
    result = _runner.invoke(app, ["prepare-marc", "--help"])
    assert result.exit_code == 0
    for flag in ("--marc", "--out", "--chunk-size", "--force"):
        assert flag in result.stdout


def test_prepare_marc_runs_and_is_idempotent(tmp_path: Path) -> None:
    """``prepare-marc`` writes chunks once, then no-ops on a clean re-run."""
    out_dir = tmp_path / "prepared"
    first = _runner.invoke(
        app,
        [
            "prepare-marc",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--out",
            str(out_dir),
            "--chunk-size",
            "5",
        ],
    )
    assert first.exit_code == 0, first.output
    assert "records:" in first.stdout
    assert "skipped: no" in first.stdout
    assert sorted(p.name for p in out_dir.glob("chunk_*.pkl"))
    second = _runner.invoke(
        app,
        [
            "prepare-marc",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--out",
            str(out_dir),
            "--chunk-size",
            "5",
        ],
    )
    assert second.exit_code == 0, second.output
    assert "skipped: yes" in second.stdout


def test_prepare_marc_rejects_missing_source(tmp_path: Path) -> None:
    result = _runner.invoke(
        app,
        [
            "prepare-marc",
            "--marc",
            str(tmp_path / "nope.xml"),
            "--out",
            str(tmp_path / "prepared"),
        ],
    )
    assert result.exit_code == 1
    assert "--marc" in result.output


def test_prepare_marc_rejects_zero_chunk_size(tmp_path: Path) -> None:
    result = _runner.invoke(
        app,
        [
            "prepare-marc",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--out",
            str(tmp_path / "prepared"),
            "--chunk-size",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "chunk-size" in result.output


def test_prepare_marc_surfaces_oserror(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("disk full")

    monkeypatch.setattr("pd_matcher.cli.prepare_marc", _raise)
    result = _runner.invoke(
        app,
        [
            "prepare-marc",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--out",
            str(tmp_path / "prepared"),
        ],
    )
    assert result.exit_code == 1
    assert "disk full" in result.output


def test_match_consumes_prepared_directory(tmp_path: Path) -> None:
    """``match --prepared`` runs against a prepared cache and writes a CSV."""
    index_path = _build_index(tmp_path)
    prepared = tmp_path / "prepared"
    prep = _runner.invoke(
        app,
        [
            "prepare-marc",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--out",
            str(prepared),
        ],
    )
    assert prep.exit_code == 0, prep.output
    out_csv = tmp_path / "out.csv"
    result = _runner.invoke(
        app,
        [
            "match",
            "--prepared",
            str(prepared),
            "--index",
            str(index_path),
            "--out",
            str(out_csv),
            "--workers",
            "1",
            "--min-score",
            "1.0",
            "-v",
        ],
    )
    assert result.exit_code == 0, result.output
    with out_csv.open(encoding="utf-8") as fp:
        assert list(DictReader(fp))


def test_match_rejects_both_marc_and_prepared(tmp_path: Path) -> None:
    """Supplying both ``--marc`` and ``--prepared`` exits 2."""
    index_path = _build_index(tmp_path)
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(_FIXTURES / "tiny.marcxml"),
            "--prepared",
            str(tmp_path / "prepared"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 2
    assert "exactly one of" in result.output


def test_match_rejects_neither_marc_nor_prepared(tmp_path: Path) -> None:
    """Supplying neither ``--marc`` nor ``--prepared`` exits 2."""
    index_path = _build_index(tmp_path)
    result = _runner.invoke(
        app,
        [
            "match",
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 2
    assert "exactly one of" in result.output


def test_match_rejects_prepared_directory_missing(tmp_path: Path) -> None:
    """A ``--prepared`` path that is not a directory exits 1."""
    index_path = _build_index(tmp_path)
    result = _runner.invoke(
        app,
        [
            "match",
            "--prepared",
            str(tmp_path / "nope"),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "--prepared" in result.output


def test_match_rejects_prepared_directory_without_manifest(tmp_path: Path) -> None:
    """A ``--prepared`` directory lacking a manifest exits 1."""
    index_path = _build_index(tmp_path)
    empty = tmp_path / "prepared"
    empty.mkdir()
    result = _runner.invoke(
        app,
        [
            "match",
            "--prepared",
            str(empty),
            "--index",
            str(index_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
    assert "manifest" in result.output


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
    """``eval`` succeeds against the tiny index + vault + pool fixtures."""
    index_path = _build_index(tmp_path)
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    report_path = tmp_path / "report.json"
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
            "--index",
            str(index_path),
            "--report",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Eval report:" in result.stdout
    assert "auc_roc" in result.stdout
    assert report_path.exists()
    payload = loads(report_path.read_text(encoding="utf-8"))
    assert "auc_roc" in payload
    assert "average_precision" in payload
    assert payload["pairs_evaluated"] == 2


def test_eval_without_report_does_not_write_file(tmp_path: Path) -> None:
    """When ``--report`` is omitted no JSON file is written."""
    index_path = _build_index(tmp_path)
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_rejects_missing_vault(tmp_path: Path) -> None:
    """``eval`` exits 1 when ``--vault`` does not exist."""
    index_path = _build_index(tmp_path)
    _, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(tmp_path / "missing.jsonl"),
            "--pool",
            str(pool_path),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 1
    assert "--vault" in result.output


def test_eval_rejects_missing_pool(tmp_path: Path) -> None:
    """``eval`` exits 1 when ``--pool`` is not a directory."""
    index_path = _build_index(tmp_path)
    vault_path, _ = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(tmp_path / "missing-pool"),
            "--index",
            str(index_path),
        ],
    )
    assert result.exit_code == 1
    assert "--pool" in result.output


def test_eval_rejects_missing_index(tmp_path: Path) -> None:
    """``eval`` exits 1 when ``--index`` does not exist."""
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    from pd_matcher.config.loader import ConfigError

    def _raise() -> object:
        raise ConfigError("bad yaml")

    monkeypatch.setattr("pd_matcher.cli._load_default_matching_config", _raise)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    from pd_matcher.config.loader import ConfigError

    def _raise() -> object:
        raise ConfigError("pairings corrupt")

    monkeypatch.setattr("pd_matcher.cli._load_default_pairing_config", _raise)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)

    def _raise(**_kwargs: object) -> object:
        raise OSError("io error")

    monkeypatch.setattr("pd_matcher.cli.run_eval", _raise)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    """``--year-window`` rebuilds the MatchingConfig before invoking ``run_eval``."""
    from pd_matcher.eval.ground_truth import EvalReport

    index_path = _build_index(tmp_path)
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    captured: dict[str, int] = {}

    def _fake_run_eval(**kwargs: object) -> EvalReport:
        config = kwargs["matching_config"]
        assert isinstance(config, MatchingConfig)
        captured["year_window"] = config.year_window
        return EvalReport(
            pairs_evaluated=0,
            pairs_positive=0,
            pairs_negative=0,
            pairs_unsure_excluded=0,
            marcs_evaluated=0,
            marcs_with_matcher_top=0,
            marcs_with_correct_top=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            auc_roc=0.0,
            average_precision=0.0,
            threshold_sweep=(),
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr("pd_matcher.cli.run_eval", _fake_run_eval)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
            "--index",
            str(index_path),
            "--year-window",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_accepts_year_window_upper_bound(tmp_path: Path) -> None:
    """``eval --year-window 100`` is the upper bound and is accepted."""
    index_path = _build_index(tmp_path)
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
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
    vault_path, pool_path = _prepare_vault_and_pool(tmp_path)
    result = _runner.invoke(
        app,
        [
            "eval",
            "--vault",
            str(vault_path),
            "--pool",
            str(pool_path),
            "--index",
            str(index_path),
            "--year-window",
            "abc",
        ],
    )
    assert result.exit_code == 2


def test_format_eval_report_empty_sweep_says_no_pairs() -> None:
    """``_format_eval_report`` includes a 'no scored pairs' marker on empty sweeps."""
    from pd_matcher.cli import _format_eval_report
    from pd_matcher.eval.ground_truth import EvalReport

    report = EvalReport(
        pairs_evaluated=0,
        pairs_positive=0,
        pairs_negative=0,
        pairs_unsure_excluded=0,
        marcs_evaluated=0,
        marcs_with_matcher_top=0,
        marcs_with_correct_top=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        auc_roc=0.0,
        average_precision=0.0,
        threshold_sweep=(),
        elapsed_seconds=0.0,
    )
    rendered = _format_eval_report(report)
    assert "no scored pairs" in rendered


def test_format_sweep_appends_tail_when_grid_does_not_align() -> None:
    """When the previewed slice misses the last grid point, the tail is appended."""
    from pd_matcher.cli import _format_sweep
    from pd_matcher.eval.metrics import ThresholdPoint

    sweep = tuple(
        ThresholdPoint(
            threshold=index * 0.1,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
        )
        for index in range(11)
    )
    rendered = _format_sweep(sweep)
    assert "t=1.00" in rendered


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


def test_resolve_log_file_returns_override_when_supplied(tmp_path: Path) -> None:
    """``--log-file`` overrides the auto-generated path verbatim."""
    target = tmp_path / "custom.log"
    assert _resolve_log_file("match", target) == target


def test_resolve_log_file_auto_generates_under_logs_dir() -> None:
    """Without an override the path is ``logs/{command}_*.log``."""
    resolved = _resolve_log_file("match", None)
    assert resolved.parent == Path("logs")
    assert resolved.name.startswith("match_")
    assert resolved.name.endswith(".log")


def test_match_writes_log_file_with_explicit_path(tmp_path: Path) -> None:
    """``match --log-file`` lands log lines at the requested path."""
    index_path = _build_index(tmp_path)
    target = tmp_path / "match-run.log"
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
            "--log-file",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    contents = target.read_text(encoding="utf-8")
    assert "match.pool.start" in contents or "match.pool.complete" in contents
