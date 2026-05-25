"""Regression baseline schema and comparison for the eval gate.

This module is a pure library: it defines the checked-in baseline schema
(:class:`Baseline` and its parts), loads it from JSON, and compares a
fresh :class:`~pd_matcher.eval.ground_truth.EvalReport` against it
(:func:`compare`). None of it touches the LMDB index, so every function
is unit-testable by fabricating :class:`EvalReport` instances directly.

The slow, index-dependent end-to-end gate (``pdm run regression``) lives
under ``tests/regression`` and is the only place that actually runs the
vault-driven eval; it consumes the helpers defined here.

The gate locks two metrics, precision and recall, against the baseline
with a symmetric tolerance: a drop of more than ``tolerance`` (default
2 percentage points) below the baseline fails; improvements never fail.
F1, AUC, and AP are reported for context but are not themselves gated.
F1 is a function of the gated metrics; AUC and AP gate-in once the
vault stabilises enough to lock them (separate ticket).
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

    vault: str
    pool: str
    year_window: int


class BaselineMetrics(Struct, frozen=True, forbid_unknown_fields=True):
    """The locked numbers for the baseline run.

    ``precision`` and ``recall`` are gated by :func:`compare`. ``f1``,
    ``auc_roc``, and ``average_precision`` are reported but not gated
    against the baseline yet — they exist so a future ticket can lock
    them once we have a stable vault to compare against.
    """

    precision: float
    recall: float
    f1: float
    auc_roc: float
    average_precision: float


class BaselineCounts(Struct, frozen=True, forbid_unknown_fields=True):
    """The locked vault counts for the baseline run (context, not gated)."""

    pairs_evaluated: int
    pairs_positive: int
    pairs_negative: int
    pairs_unsure_excluded: int
    marcs_evaluated: int
    marcs_with_matcher_top: int
    marcs_with_correct_top: int


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


def _reported_message(name: str, baseline_value: float, report_value: float) -> str:
    """Build a human-readable report-only line for an ungated metric."""
    delta = report_value - baseline_value
    return (
        f"{name}: REPORT "
        f"(baseline={baseline_value:.6f} report={report_value:.6f} delta={delta:+.6f})"
    )


def compare(baseline: Baseline, report: EvalReport) -> RegressionResult:
    """Compare a fresh :class:`EvalReport` against a :class:`Baseline`.

    The gate passes when neither precision nor recall has dropped more
    than ``baseline.tolerance`` below the baseline value. Improvements,
    and drops within tolerance, both pass. AUC and AP are formatted into
    the messages tuple alongside the gated lines so the regression
    output surfaces both signals at once, but they are not used in the
    pass/fail decision.

    Args:
        baseline: The checked-in baseline to compare against.
        report: A fresh report produced by
            :func:`~pd_matcher.eval.ground_truth.run_eval`.

    Returns:
        A :class:`RegressionResult` carrying the pass/fail verdict, the
        precision and recall deltas (report minus baseline), and one
        human-readable message per gated and reported metric.
    """
    tolerance = baseline.tolerance
    precision_delta = report.precision - baseline.metrics.precision
    recall_delta = report.recall - baseline.metrics.recall
    passed = precision_delta >= -tolerance and recall_delta >= -tolerance
    messages = (
        _metric_message("precision", baseline.metrics.precision, report.precision, tolerance),
        _metric_message("recall", baseline.metrics.recall, report.recall, tolerance),
        _reported_message("auc_roc", baseline.metrics.auc_roc, report.auc_roc),
        _reported_message(
            "average_precision",
            baseline.metrics.average_precision,
            report.average_precision,
        ),
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
    regenerate the checked-in baseline after an intentional change to
    the matching or assessment pipeline.

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
            auc_roc=report.auc_roc,
            average_precision=report.average_precision,
        ),
        counts=BaselineCounts(
            pairs_evaluated=report.pairs_evaluated,
            pairs_positive=report.pairs_positive,
            pairs_negative=report.pairs_negative,
            pairs_unsure_excluded=report.pairs_unsure_excluded,
            marcs_evaluated=report.marcs_evaluated,
            marcs_with_matcher_top=report.marcs_with_matcher_top,
            marcs_with_correct_top=report.marcs_with_correct_top,
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
