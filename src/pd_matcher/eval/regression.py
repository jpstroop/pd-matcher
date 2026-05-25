"""Regression baseline schema and comparison for the eval gate.

This module is a pure library: it defines the checked-in baseline schema
(:class:`Baseline` and its parts), loads it from JSON, and compares a
fresh :class:`~pd_matcher.eval.ground_truth.EvalReport` against it
(:func:`compare`). None of it touches the LMDB index, so every function
is unit-testable by fabricating :class:`EvalReport` instances directly.

The slow, index-dependent end-to-end gate (``pdm run regression``) lives
under ``tests/regression`` and is the only place that actually runs the
1000-row eval; it consumes the helpers defined here.

The gate locks two metrics, precision and recall, against the baseline
with a symmetric tolerance: a drop of more than ``tolerance`` (default
2 percentage points) below the baseline fails; improvements never fail.
F1 is reported for context but is not itself gated, since it is fully
determined by precision and recall.
"""

from pathlib import Path

from msgspec import DecodeError
from msgspec import Struct
from msgspec import ValidationError
from msgspec.json import decode

from pd_matcher.eval.ground_truth import EvalReport


class RegressionError(Exception):
    """Raised when a baseline JSON file is missing, malformed, or invalid."""


class BaselineParams(Struct, frozen=True, forbid_unknown_fields=True):
    """The eval invocation parameters that the baseline was measured under."""

    sample: int
    seed: int
    year_window: int
    ground_truth: str


class BaselineMetrics(Struct, frozen=True, forbid_unknown_fields=True):
    """The locked precision/recall/F1 numbers for the baseline run."""

    precision: float
    recall: float
    f1: float


class BaselineCounts(Struct, frozen=True, forbid_unknown_fields=True):
    """The locked row counts for the baseline run (context, not gated)."""

    rows_evaluated: int
    rows_with_predicted_match: int
    rows_with_ground_truth_match: int
    rows_agreeing: int


class Baseline(Struct, frozen=True, forbid_unknown_fields=True):
    """A checked-in regression baseline: params, metrics, counts, tolerance."""

    params: BaselineParams
    metrics: BaselineMetrics
    counts: BaselineCounts
    tolerance: float
    notes: str


class RegressionResult(Struct, frozen=True, forbid_unknown_fields=True):
    """Outcome of comparing a fresh report against a baseline.

    ``precision_delta`` and ``recall_delta`` are ``report - baseline``, so
    a negative value is a regression and a positive value is an
    improvement.
    """

    passed: bool
    precision_delta: float
    recall_delta: float
    messages: tuple[str, ...]


def _metric_message(name: str, baseline_value: float, report_value: float, tolerance: float) -> str:
    """Build a human-readable pass/regress line for one gated metric."""
    delta = report_value - baseline_value
    verdict = "PASS" if delta >= -tolerance else "REGRESS"
    return (
        f"{name}: {verdict} "
        f"(baseline={baseline_value:.6f} report={report_value:.6f} "
        f"delta={delta:+.6f} tolerance={tolerance:.6f})"
    )


def compare(baseline: Baseline, report: EvalReport) -> RegressionResult:
    """Compare a fresh :class:`EvalReport` against a :class:`Baseline`.

    The gate passes when neither precision nor recall has dropped more
    than ``baseline.tolerance`` below the baseline value. Improvements,
    and drops within tolerance, both pass.

    Args:
        baseline: The checked-in baseline to compare against.
        report: A fresh report produced by
            :func:`~pd_matcher.eval.ground_truth.run_eval`.

    Returns:
        A :class:`RegressionResult` carrying the pass/fail verdict, the
        precision and recall deltas (report minus baseline), and one
        human-readable message per gated metric.
    """
    tolerance = baseline.tolerance
    precision_delta = report.precision - baseline.metrics.precision
    recall_delta = report.recall - baseline.metrics.recall
    passed = precision_delta >= -tolerance and recall_delta >= -tolerance
    messages = (
        _metric_message("precision", baseline.metrics.precision, report.precision, tolerance),
        _metric_message("recall", baseline.metrics.recall, report.recall, tolerance),
    )
    return RegressionResult(
        passed=passed,
        precision_delta=precision_delta,
        recall_delta=recall_delta,
        messages=messages,
    )


def load_baseline(path: Path) -> Baseline:
    """Load and validate a :class:`Baseline` from a JSON file.

    Args:
        path: Filesystem path to a baseline JSON document matching the
            :class:`Baseline` schema.

    Returns:
        A validated :class:`Baseline`.

    Raises:
        RegressionError: If the file cannot be read, is not valid JSON,
            or does not match the :class:`Baseline` schema.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RegressionError(f"Cannot read baseline file {path}: {exc}") from exc
    try:
        return decode(raw, type=Baseline)
    except (DecodeError, ValidationError) as exc:
        raise RegressionError(f"Invalid baseline JSON in {path}: {exc}") from exc


def baseline_from_report(
    report: EvalReport,
    *,
    params: BaselineParams,
    tolerance: float,
    notes: str,
) -> Baseline:
    """Build a :class:`Baseline` from a fresh :class:`EvalReport`.

    Used by the refresh script (``pdm run regression-baseline``) to
    regenerate the checked-in baseline after an intentional change to the
    matching or assessment pipeline.

    Args:
        report: A fresh report produced by
            :func:`~pd_matcher.eval.ground_truth.run_eval`.
        params: The invocation parameters the report was measured under.
        tolerance: The symmetric drop tolerance to lock into the baseline.
        notes: Human-readable provenance and caveats.

    Returns:
        A :class:`Baseline` snapshotting the report's metrics and counts.
    """
    return Baseline(
        params=params,
        metrics=BaselineMetrics(
            precision=report.precision,
            recall=report.recall,
            f1=report.f1,
        ),
        counts=BaselineCounts(
            rows_evaluated=report.rows_evaluated,
            rows_with_predicted_match=report.rows_with_predicted_match,
            rows_with_ground_truth_match=report.rows_with_ground_truth_match,
            rows_agreeing=report.rows_agreeing,
        ),
        tolerance=tolerance,
        notes=notes,
    )


__all__ = [
    "Baseline",
    "BaselineCounts",
    "BaselineMetrics",
    "BaselineParams",
    "RegressionError",
    "RegressionResult",
    "baseline_from_report",
    "compare",
    "load_baseline",
]
