"""Tests for :mod:`pd_matcher.eval.ground_truth`."""

from datetime import date
from pathlib import Path

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.eval.ground_truth import UNRECOGNIZED_GT_STATUS
from pd_matcher.eval.ground_truth import EvalReport
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.index.builder import build_index

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _build_index(tmp_path: Path) -> Path:
    """Stand up a tiny LMDB env from the shared fixtures."""
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def _matching_config() -> MatchingConfig:
    """A permissive :class:`MatchingConfig` so the tiny corpus produces matches."""
    return MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        year_window=2,
        min_combined_score=1.0,
        scorer="weighted_mean",
    )


_HEADER_FIELDS: tuple[str, ...] = (
    "marc_id",
    "marc_title_original",
    "marc_title_normalized",
    "marc_title_stemmed",
    "marc_author_original",
    "marc_author_normalized",
    "marc_author_stemmed",
    "marc_main_author_original",
    "marc_main_author_normalized",
    "marc_main_author_stemmed",
    "marc_publisher_original",
    "marc_publisher_normalized",
    "marc_publisher_stemmed",
    "marc_year",
    "marc_lccn",
    "marc_lccn_normalized",
    "marc_country_code",
    "marc_language_code",
    "match_type",
    "match_title",
    "match_title_normalized",
    "match_author",
    "match_author_normalized",
    "match_publisher",
    "match_publisher_normalized",
    "match_year",
    "match_source_id",
    "match_date",
    "title_score",
    "author_score",
    "publisher_score",
    "combined_score",
    "year_difference",
    "copyright_status",
)


def _row(overrides: dict[str, str]) -> str:
    """Emit a CSV row with ``overrides`` filling the named columns."""
    values = [overrides.get(field, "") for field in _HEADER_FIELDS]
    return ",".join(values) + "\n"


def _write_ground_truth(path: Path) -> None:
    """Emit a 3-row CSV: agreement, disagreement, no-prediction."""
    header = ",".join(_HEADER_FIELDS) + "\n"
    agree = _row(
        {
            "marc_id": "marc-aaa",
            "marc_title_original": "A study of widgets",
            "marc_main_author_original": "Smith John",
            "marc_publisher_original": "Acme Press",
            "marc_year": "1940",
            "marc_country_code": "xxu",
            "marc_language_code": "eng",
            "match_source_id": "UUID-0001",
            "copyright_status": "PD_REGISTERED_NOT_RENEWED",
        }
    )
    disagree = _row(
        {
            "marc_id": "marc-bbb",
            "marc_title_original": "Le petit livre",
            "marc_main_author_original": "Dubois David",
            "marc_publisher_original": "Editions Beta",
            "marc_year": "1955",
            "marc_country_code": "fr",
            "marc_language_code": "fre",
            "match_source_id": "UUID-0999",
            "copyright_status": "IN_COPYRIGHT_REGISTERED_AND_RENEWED",
        }
    )
    no_match = _row(
        {
            "marc_id": "marc-ccc",
            "marc_title_original": "Unrelated Title",
            "marc_year": "1700",
            "marc_country_code": "xxu",
            "marc_language_code": "eng",
            "copyright_status": "UNRECOGNIZED_LABEL",
        }
    )
    path.write_text(header + agree + disagree + no_match, encoding="utf-8")


def test_run_eval_returns_expected_aggregates(tmp_path: Path) -> None:
    """Three rows: 1 agreeing prediction, 1 disagreeing, 1 unmatched."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path)
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        today=date(2026, 5, 18),
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(today=date(2026, 5, 18)),
    )
    assert isinstance(report, EvalReport)
    assert report.rows_evaluated == 3
    assert report.rows_with_ground_truth_match == 2
    assert report.rows_with_predicted_match >= 1
    assert report.rows_agreeing >= 1
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.f1 <= 1.0
    assert report.elapsed_seconds >= 0.0
    flattened: list[str] = []
    for gt_buckets in report.status_confusion.values():
        flattened.extend(gt_buckets.keys())
    assert UNRECOGNIZED_GT_STATUS in flattened


def test_run_eval_respects_limit(tmp_path: Path) -> None:
    """``limit`` caps the row count regardless of CSV size."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path)
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        today=date(2026, 5, 18),
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(today=date(2026, 5, 18)),
        limit=1,
    )
    assert report.rows_evaluated == 1


def test_run_eval_zero_rows_yields_zero_metrics(tmp_path: Path) -> None:
    """An empty CSV (header only) yields zero metrics, not a divide-by-zero."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("marc_id,marc_year,match_source_id,copyright_status\n", encoding="utf-8")
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        today=date(2026, 5, 18),
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(today=date(2026, 5, 18)),
    )
    assert report.rows_evaluated == 0
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.status_confusion == {}


def test_run_eval_handles_unparseable_year(tmp_path: Path) -> None:
    """Rows with malformed ``marc_year`` still evaluate (no match possible)."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text(
        "marc_id,marc_year,match_source_id,copyright_status\n"
        "marc-zzz,not-a-year,UUID-0001,PD_REGISTERED_NOT_RENEWED\n",
        encoding="utf-8",
    )
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        today=date(2026, 5, 18),
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(today=date(2026, 5, 18)),
    )
    assert report.rows_evaluated == 1
    assert report.rows_with_predicted_match == 0


def test_run_eval_handles_empty_year(tmp_path: Path) -> None:
    """Rows with an empty ``marc_year`` evaluate (no match possible)."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text(
        "marc_id,marc_year,match_source_id,copyright_status\n"
        "marc-yyy,,UUID-0001,PD_REGISTERED_NOT_RENEWED\n",
        encoding="utf-8",
    )
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        today=date(2026, 5, 18),
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(today=date(2026, 5, 18)),
    )
    assert report.rows_evaluated == 1
    assert report.rows_with_predicted_match == 0
