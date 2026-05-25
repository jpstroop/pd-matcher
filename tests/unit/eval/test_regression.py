"""Tests for :mod:`pd_matcher.eval.regression` (fast, no index needed)."""

from pathlib import Path

from pytest import raises

from pd_matcher.eval.ground_truth import EvalReport
from pd_matcher.eval.regression import Baseline
from pd_matcher.eval.regression import BaselineCounts
from pd_matcher.eval.regression import BaselineMetrics
from pd_matcher.eval.regression import BaselineParams
from pd_matcher.eval.regression import RegressionError
from pd_matcher.eval.regression import baseline_from_report
from pd_matcher.eval.regression import compare
from pd_matcher.eval.regression import load_baseline


def _params() -> BaselineParams:
    return BaselineParams(
        sample=1000,
        seed=42,
        year_window=0,
        ground_truth="combined_ground_truth.csv",
    )


def _baseline(
    *, precision: float = 0.85, recall: float = 0.78, tolerance: float = 0.02
) -> Baseline:
    return Baseline(
        params=_params(),
        metrics=BaselineMetrics(precision=precision, recall=recall, f1=0.81),
        counts=BaselineCounts(
            rows_evaluated=1000,
            rows_with_predicted_match=914,
            rows_with_ground_truth_match=1000,
            rows_agreeing=777,
        ),
        tolerance=tolerance,
        notes="test baseline",
    )


def _report(*, precision: float, recall: float, f1: float = 0.0) -> EvalReport:
    return EvalReport(
        rows_evaluated=1000,
        rows_with_predicted_match=914,
        rows_with_ground_truth_match=1000,
        rows_agreeing=777,
        precision=precision,
        recall=recall,
        f1=f1,
        elapsed_seconds=1.0,
    )


def test_compare_passes_when_metrics_match() -> None:
    baseline = _baseline(precision=0.85, recall=0.78)
    result = compare(baseline, _report(precision=0.85, recall=0.78))
    assert result.passed is True
    assert result.precision_delta == 0.0
    assert result.recall_delta == 0.0
    assert len(result.messages) == 2
    assert all("PASS" in message for message in result.messages)


def test_compare_fails_on_precision_regression() -> None:
    baseline = _baseline(precision=0.85, recall=0.78, tolerance=0.02)
    result = compare(baseline, _report(precision=0.80, recall=0.78))
    assert result.passed is False
    assert result.precision_delta < 0.0
    assert "precision: REGRESS" in result.messages[0]
    assert "recall: PASS" in result.messages[1]


def test_compare_fails_on_recall_regression() -> None:
    baseline = _baseline(precision=0.85, recall=0.78, tolerance=0.02)
    result = compare(baseline, _report(precision=0.85, recall=0.70))
    assert result.passed is False
    assert result.recall_delta < 0.0
    assert "precision: PASS" in result.messages[0]
    assert "recall: REGRESS" in result.messages[1]


def test_compare_passes_on_improvement() -> None:
    baseline = _baseline(precision=0.85, recall=0.78)
    result = compare(baseline, _report(precision=0.90, recall=0.85))
    assert result.passed is True
    assert result.precision_delta > 0.0
    assert result.recall_delta > 0.0


def test_compare_passes_at_exact_tolerance_boundary() -> None:
    baseline = _baseline(precision=0.02, recall=0.02, tolerance=0.02)
    result = compare(baseline, _report(precision=0.0, recall=0.0))
    assert result.precision_delta == -0.02
    assert result.recall_delta == -0.02
    assert result.passed is True


def test_compare_fails_just_past_tolerance_boundary() -> None:
    baseline = _baseline(precision=0.02, recall=0.02, tolerance=0.02)
    result = compare(baseline, _report(precision=-0.0000001, recall=0.02))
    assert result.precision_delta < -0.02
    assert result.passed is False


def test_load_baseline_happy(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(
        """
        {
          "params": {"sample": 1000, "seed": 42, "year_window": 0,
                     "ground_truth": "combined_ground_truth.csv"},
          "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
          "counts": {"rows_evaluated": 1000, "rows_with_predicted_match": 914,
                     "rows_with_ground_truth_match": 1000, "rows_agreeing": 777},
          "tolerance": 0.02,
          "notes": "ok"
        }
        """,
        encoding="utf-8",
    )
    baseline = load_baseline(path)
    assert baseline.params.sample == 1000
    assert baseline.metrics.precision == 0.85
    assert baseline.tolerance == 0.02


def test_load_baseline_missing_file(tmp_path: Path) -> None:
    with raises(RegressionError, match="Cannot read baseline file"):
        load_baseline(tmp_path / "does_not_exist.json")


def test_load_baseline_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with raises(RegressionError, match="Invalid baseline JSON"):
        load_baseline(path)


def test_load_baseline_schema_violation(tmp_path: Path) -> None:
    path = tmp_path / "wrong_schema.json"
    path.write_text('{"params": {"sample": 1000}}', encoding="utf-8")
    with raises(RegressionError, match="Invalid baseline JSON"):
        load_baseline(path)


def test_baseline_from_report_round_trip() -> None:
    report = _report(precision=0.85, recall=0.78, f1=0.81)
    baseline = baseline_from_report(
        report,
        params=_params(),
        tolerance=0.02,
        notes="round trip",
    )
    assert baseline.metrics.precision == report.precision
    assert baseline.metrics.recall == report.recall
    assert baseline.metrics.f1 == report.f1
    assert baseline.counts.rows_evaluated == report.rows_evaluated
    assert baseline.counts.rows_with_predicted_match == report.rows_with_predicted_match
    assert baseline.counts.rows_with_ground_truth_match == report.rows_with_ground_truth_match
    assert baseline.counts.rows_agreeing == report.rows_agreeing
    assert baseline.tolerance == 0.02
    assert baseline.notes == "round trip"
    assert baseline.params == _params()
    assert compare(baseline, report).passed is True
