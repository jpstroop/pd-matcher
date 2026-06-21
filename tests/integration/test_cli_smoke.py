"""End-to-end smoke test for the ``pd-matcher match`` CLI command.

Builds a tiny LMDB index in ``tmp_path``, invokes ``pd-matcher match``
through :class:`typer.testing.CliRunner` against the shared MARC
fixture, and asserts the destination JSONL has one record per MARC
record. This catches CLI-only regressions (argument wiring, default
lookup paths, IDF cache placement) that the in-process Phase 6 smoke
test cannot.
"""

from json import loads
from pathlib import Path

from typer.testing import CliRunner

from pd_matcher.cli import app
from pd_matcher.index.builder import build_index
from pd_matcher.parsers.marc import iter_marc_records

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_runner: CliRunner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})


def _build_index(tmp_path: Path) -> Path:
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def test_match_cli_produces_one_row_per_record(tmp_path: Path) -> None:
    """``pd-matcher match`` end-to-end writes one JSONL record per MARC record."""
    index_path = _build_index(tmp_path)
    marc_path = _FIXTURES / "tiny.marcxml"
    expected = sum(1 for _ in iter_marc_records(marc_path))
    out_jsonl = tmp_path / "results.jsonl"
    result = _runner.invoke(
        app,
        [
            "match",
            "--marc",
            str(marc_path),
            "--index",
            str(index_path),
            "--out",
            str(out_jsonl),
            "--workers",
            "1",
            "--min-score",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output
    records = [loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == expected
