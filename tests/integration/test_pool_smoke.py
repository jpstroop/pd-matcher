"""End-to-end smoke test for :func:`pd_matcher.workers.pool.run_match`.

Stands up the tiny LMDB index, runs the producer/worker/writer pipeline
under a real spawn context with two worker processes, and asserts that
the resulting CSV has one row per MARC record in the fixture.
"""

from csv import DictReader
from pathlib import Path

from pd_matcher.config.loader import load_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import build_idf_table
from pd_matcher.parsers.marc import iter_marc_records
from pd_matcher.workers import run_match

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_PAIRINGS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pd_matcher"
    / "config"
    / "defaults"
    / "field_pairings.yaml"
)


def _build_index_and_idf(tmp_path: Path) -> Path:
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def test_run_match_emits_one_row_per_input_record(tmp_path: Path) -> None:
    index_path = _build_index_and_idf(tmp_path)
    with NyplIndexLookup(index_path) as lookup:
        idf = build_idf_table(lookup)
    marc_path = _FIXTURES / "tiny.marcxml"
    expected_records = sum(1 for _ in iter_marc_records(marc_path))
    output_path = tmp_path / "results.csv"
    config = MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        year_window=2,
        min_combined_score=30.0,
        scorer="weighted_mean",
    )
    pairing_config = load_pairing_config(_PAIRINGS)
    report = run_match(
        marc_path=marc_path,
        index_path=index_path,
        output_path=output_path,
        matching_config=config,
        pairing_config=pairing_config,
        idf=idf,
        workers=2,
        batch_size=2,
        report_interval_seconds=0.05,
    )
    with output_path.open(encoding="utf-8") as fp:
        rows = list(DictReader(fp))
    assert report.records_processed == expected_records
    assert report.records_written == expected_records
    assert report.records_enqueued == expected_records
    assert len(rows) == expected_records
    assert report.interrupted is False
