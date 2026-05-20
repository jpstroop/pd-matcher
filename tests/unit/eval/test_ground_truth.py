"""Tests for :mod:`pd_matcher.eval.ground_truth`."""

from pathlib import Path

from pytest import raises

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
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
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
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
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
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
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
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
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
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
    )
    assert report.rows_evaluated == 1
    assert report.rows_with_predicted_match == 0


def _write_many_distinct_rows(path: Path, count: int) -> None:
    """Emit ``count`` distinct rows so a random sampler can shuffle them."""
    header = ",".join(_HEADER_FIELDS) + "\n"
    rows = "".join(
        _row(
            {
                "marc_id": f"marc-{i:04d}",
                "marc_title_original": f"Title {i}",
                "marc_year": "1940",
                "marc_country_code": "xxu",
                "marc_language_code": "eng",
                "copyright_status": "UNKNOWN_INSUFFICIENT_DATA",
            }
        )
        for i in range(count)
    )
    path.write_text(header + rows, encoding="utf-8")


def test_run_eval_sample_caps_row_count(tmp_path: Path) -> None:
    """``sample=5`` reduces the evaluated row count to 5."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_many_distinct_rows(gt_path, 20)
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        sample=5,
        seed=42,
    )
    assert report.rows_evaluated == 5


def test_run_eval_sample_larger_than_corpus_evaluates_all(tmp_path: Path) -> None:
    """``sample > len(rows)`` evaluates every row, no error raised."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_many_distinct_rows(gt_path, 4)
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        sample=100,
        seed=0,
    )
    assert report.rows_evaluated == 4


def test_run_eval_same_seed_picks_same_rows(tmp_path: Path) -> None:
    """Two runs with the same seed select identical rows (deterministic)."""
    from pd_matcher.eval.ground_truth import _load_rows

    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_many_distinct_rows(gt_path, 50)
    first = _load_rows(gt_path, sample=10, seed=7)
    second = _load_rows(gt_path, sample=10, seed=7)
    assert [row["marc_id"] for row in first] == [row["marc_id"] for row in second]
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        sample=10,
        seed=7,
    )
    assert report.rows_evaluated == 10


def test_run_eval_different_seeds_pick_different_rows(tmp_path: Path) -> None:
    """Different seeds yield different selections on a 50-row corpus."""
    from pd_matcher.eval.ground_truth import _load_rows

    gt_path = tmp_path / "gt.csv"
    _write_many_distinct_rows(gt_path, 50)
    first = _load_rows(gt_path, sample=25, seed=1)
    second = _load_rows(gt_path, sample=25, seed=2)
    assert [row["marc_id"] for row in first] != [row["marc_id"] for row in second]


def test_load_rows_without_sample_preserves_file_order(tmp_path: Path) -> None:
    """``sample=None`` returns rows in file order — no shuffling, no surprises."""
    from pd_matcher.eval.ground_truth import _load_rows

    gt_path = tmp_path / "gt.csv"
    _write_many_distinct_rows(gt_path, 5)
    rows = _load_rows(gt_path, sample=None, seed=99)
    assert [row["marc_id"] for row in rows] == [f"marc-{i:04d}" for i in range(5)]


def _write_year_drift_row(path: Path) -> None:
    """Emit a single GT row whose MARC year is ``5`` off from the indexed reg year.

    The tiny fixtures register UUID-0001 in 1940; this row claims 1945.
    With ``year_window=0`` no candidate is returned (year_diff exceeds the
    window); with ``year_window=5`` the registration is in-bucket.
    """
    header = ",".join(_HEADER_FIELDS) + "\n"
    body = _row(
        {
            "marc_id": "marc-drift",
            "marc_title_original": "A study of widgets",
            "marc_main_author_original": "Smith John",
            "marc_publisher_original": "Acme Press",
            "marc_year": "1945",
            "marc_country_code": "xxu",
            "marc_language_code": "eng",
            "match_source_id": "UUID-0001",
            "copyright_status": "PD_REGISTERED_NOT_RENEWED",
        }
    )
    path.write_text(header + body, encoding="utf-8")


def test_run_eval_year_window_zero_yields_no_predicted_match(tmp_path: Path) -> None:
    """``year_window=0`` blocks the year-drifted candidate from matching."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_year_drift_row(gt_path)
    narrow = MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        year_window=0,
        min_combined_score=1.0,
        scorer="weighted_mean",
    )
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=narrow,
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
    )
    assert report.rows_with_predicted_match == 0


def test_run_eval_year_window_five_admits_year_drifted_match(tmp_path: Path) -> None:
    """``year_window=5`` admits the same drifted candidate as a match."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_year_drift_row(gt_path)
    wide = MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        year_window=5,
        min_combined_score=1.0,
        scorer="weighted_mean",
    )
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=wide,
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
    )
    assert report.rows_with_predicted_match == 1


def test_run_eval_rejects_workers_below_one(tmp_path: Path) -> None:
    """``workers=0`` is rejected by the public API with :class:`ValueError`."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path)
    with raises(ValueError, match="workers must be >= 1"):
        run_eval(
            ground_truth_path=gt_path,
            index_path=index_path,
            as_of_year=2026,
            matching_config=_matching_config(),
            copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
            workers=0,
        )


def test_run_eval_workers_one_matches_default_path(tmp_path: Path) -> None:
    """``workers=1`` produces the same report as the default invocation."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path)
    default_report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
    )
    explicit_report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        workers=1,
    )
    assert default_report.rows_evaluated == explicit_report.rows_evaluated
    assert default_report.rows_with_predicted_match == explicit_report.rows_with_predicted_match
    assert (
        default_report.rows_with_ground_truth_match == explicit_report.rows_with_ground_truth_match
    )
    assert default_report.rows_agreeing == explicit_report.rows_agreeing
    assert default_report.status_confusion == explicit_report.status_confusion


def test_run_eval_workers_two_matches_workers_one(tmp_path: Path) -> None:
    """``workers=2`` (spawn pool) yields the identical aggregate as ``workers=1``."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_ground_truth(gt_path)
    serial_report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        workers=1,
    )
    parallel_report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        workers=2,
    )
    assert parallel_report.rows_evaluated == serial_report.rows_evaluated
    assert parallel_report.rows_with_predicted_match == serial_report.rows_with_predicted_match
    assert (
        parallel_report.rows_with_ground_truth_match == serial_report.rows_with_ground_truth_match
    )
    assert parallel_report.rows_agreeing == serial_report.rows_agreeing
    assert parallel_report.precision == serial_report.precision
    assert parallel_report.recall == serial_report.recall
    assert parallel_report.f1 == serial_report.f1
    assert parallel_report.status_confusion == serial_report.status_confusion


def test_run_eval_workers_two_respects_limit(tmp_path: Path) -> None:
    """``--limit`` truncates the row list before fanning out to the pool."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    _write_many_distinct_rows(gt_path, 12)
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        limit=4,
        workers=2,
    )
    assert report.rows_evaluated == 4


def test_run_eval_workers_two_empty_corpus_yields_zero_metrics(tmp_path: Path) -> None:
    """An empty CSV under ``workers=2`` short-circuits before spawning a pool."""
    index_path = _build_index(tmp_path)
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("marc_id,marc_year,match_source_id,copyright_status\n", encoding="utf-8")
    report = run_eval(
        ground_truth_path=gt_path,
        index_path=index_path,
        as_of_year=2026,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        workers=2,
    )
    assert report.rows_evaluated == 0
    assert report.status_confusion == {}


def test_eval_one_row_is_a_pure_function(tmp_path: Path) -> None:
    """``_eval_one_row`` is callable directly with a constructed ``_WorkerState``.

    Exercising the per-row helper without spawning processes keeps the unit
    fast and lets us assert the structured outcome shape without IPC.
    """
    from pd_matcher.eval.ground_truth import _eval_one_row
    from pd_matcher.eval.ground_truth import _WorkerState

    index_path = _build_index(tmp_path)
    state = _WorkerState(
        index_path=index_path,
        matching_config=_matching_config(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=2026),
        as_of_year=2026,
    )
    try:
        row = {
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
        outcome = _eval_one_row(row, state=state)
    finally:
        state.lookup.close()
    assert outcome.has_predicted_match is True
    assert outcome.has_ground_truth_match is True
    assert outcome.agrees is True


def test_pool_eval_row_raises_when_initializer_skipped() -> None:
    """``_pool_eval_row`` fails fast when the module-global state is unset.

    The initializer always runs inside the spawn pool, so this guard
    only fires when a caller misuses the worker function directly; the
    explicit ``RuntimeError`` is preferable to an opaque ``AttributeError``.
    """
    from pd_matcher.eval import ground_truth as gt_module

    saved = gt_module._WORKER_STATE
    gt_module._WORKER_STATE = None
    try:
        with raises(RuntimeError, match="_pool_initializer"):
            gt_module._pool_eval_row({"marc_id": "x"})
    finally:
        gt_module._WORKER_STATE = saved


def test_pool_initializer_and_pool_eval_row_round_trip(tmp_path: Path) -> None:
    """``_pool_initializer`` populates the module global so ``_pool_eval_row`` works.

    Exercises the spawn-pool entry points in-process so coverage reaches
    the lines that ``multiprocessing.Pool`` would normally execute
    inside a separate interpreter (where pytest-cov can't see).
    """
    from pd_matcher.eval import ground_truth as gt_module

    index_path = _build_index(tmp_path)
    saved = gt_module._WORKER_STATE
    try:
        gt_module._pool_initializer(
            index_path,
            _matching_config(),
            CopyrightAssessmentConfig(as_of_year=2026),
            2026,
        )
        assert gt_module._WORKER_STATE is not None
        outcome = gt_module._pool_eval_row(
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
        assert outcome.has_predicted_match is True
        assert outcome.agrees is True
    finally:
        if gt_module._WORKER_STATE is not None:
            gt_module._WORKER_STATE.lookup.close()
        gt_module._WORKER_STATE = saved
