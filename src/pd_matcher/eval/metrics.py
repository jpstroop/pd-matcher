"""Threshold-independent and threshold-swept metrics over scored pairs.

Pure functions over ``(score, label)`` sequences — no I/O, no fixtures,
no sklearn. Hand-coded so the gates stay light and the math is
inspectable: AUC is the rank-sum (Mann-Whitney U) closed form, average
precision is the mean of precision-at-each-positive-rank, and the sweep
walks an inclusive grid of thresholds turning each into a contingency
table the caller can plot or table directly.

``label`` is ``1`` for the positive class and ``0`` for the negative
class; anything else is rejected (see :func:`_validate_labels`). The
vault feeds ``match`` -> ``1`` and ``no_match`` -> ``0``; ``unsure``
entries are filtered out by the caller before metrics see them.
"""

from collections.abc import Sequence
from logging import getLogger

from msgspec import Struct

_LOGGER = getLogger(__name__)

_POSITIVE_LABEL: int = 1
_NEGATIVE_LABEL: int = 0
_DEGENERATE_AUC: float = 0.5
_DEFAULT_SWEEP_START: float = 0.0
_DEFAULT_SWEEP_STOP: float = 1.0
_DEFAULT_SWEEP_STEP: float = 0.05


class ThresholdPoint(Struct, frozen=True, forbid_unknown_fields=True):
    """One row of a precision/recall sweep at a fixed score threshold.

    ``true_positives`` and ``false_positives`` partition the predicted-
    positive set (score >= threshold); ``false_negatives`` is the
    positive labels that fell below the threshold. Precision, recall,
    and F1 are derived; they are stored on the struct so consumers do
    not have to recompute them per row.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _validate_labels(scored_labels: Sequence[tuple[float, int]]) -> None:
    """Reject empty input and any label that is not ``0`` or ``1``."""
    if not scored_labels:
        raise ValueError("scored_labels must contain at least one entry")
    for _score, label in scored_labels:
        if label not in (_POSITIVE_LABEL, _NEGATIVE_LABEL):
            raise ValueError(f"label must be 0 or 1 (got {label!r})")


def _count_classes(scored_labels: Sequence[tuple[float, int]]) -> tuple[int, int]:
    """Return ``(positives, negatives)`` in ``scored_labels``."""
    positives = sum(1 for _score, label in scored_labels if label == _POSITIVE_LABEL)
    negatives = len(scored_labels) - positives
    return positives, negatives


def _safe_division(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` or ``0.0`` when the denominator is zero."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; ``0.0`` when both are zero."""
    total = precision + recall
    if total <= 0.0:
        return 0.0
    return 2.0 * precision * recall / total


def roc_auc(scored_labels: Sequence[tuple[float, int]]) -> float:
    """Return the area under the ROC curve using the rank-sum closed form.

    The formula is
    ``(sum_of_positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)``
    where ranks are assigned in ascending score order with mid-ranks for
    ties (so identical scores across the two classes do not bias the
    estimate). This is the Mann-Whitney U statistic divided by the
    product of class sizes — equivalent to integrating the ROC curve.

    Args:
        scored_labels: Sequence of ``(score, label)`` tuples; ``label``
            must be ``0`` or ``1``.

    Returns:
        The AUC in ``[0.0, 1.0]``. When only one class is present the
        AUC is undefined; this function returns ``0.5`` with a logged
        warning so callers can plot the report without special-casing.

    Raises:
        ValueError: If ``scored_labels`` is empty or contains a label
            outside ``{0, 1}``.
    """
    _validate_labels(scored_labels)
    n_pos, n_neg = _count_classes(scored_labels)
    if n_pos == 0 or n_neg == 0:
        _LOGGER.warning(
            "metrics.roc_auc.single_class positives=%d negatives=%d (returning 0.5)",
            n_pos,
            n_neg,
        )
        return _DEGENERATE_AUC
    ordered = sorted(scored_labels, key=lambda item: item[0])
    sum_positive_ranks = 0.0
    index = 0
    total = len(ordered)
    while index < total:
        tie_end = index + 1
        while tie_end < total and ordered[tie_end][0] == ordered[index][0]:
            tie_end += 1
        avg_rank = (index + 1 + tie_end) / 2.0
        for tied_index in range(index, tie_end):
            if ordered[tied_index][1] == _POSITIVE_LABEL:
                sum_positive_ranks += avg_rank
        index = tie_end
    return (sum_positive_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scored_labels: Sequence[tuple[float, int]]) -> float:
    """Return the mean precision evaluated at each positive's rank.

    Sorts pairs by score descending; for each positive in that order
    computes precision at its rank (positives_seen / pairs_seen) and
    averages the result over the total positive count. This matches
    sklearn's ``average_precision_score`` for the unique-score case and
    is a standard estimator of the area under the precision-recall
    curve.

    Args:
        scored_labels: Sequence of ``(score, label)`` tuples; ``label``
            must be ``0`` or ``1``.

    Returns:
        The average precision in ``[0.0, 1.0]``. When no positives are
        present the score is undefined; this function returns ``0.0``
        with a logged warning so callers can plot the report without
        special-casing.

    Raises:
        ValueError: If ``scored_labels`` is empty or contains a label
            outside ``{0, 1}``.
    """
    _validate_labels(scored_labels)
    n_pos, _ = _count_classes(scored_labels)
    if n_pos == 0:
        _LOGGER.warning("metrics.average_precision.no_positives (returning 0.0)")
        return 0.0
    ordered = sorted(scored_labels, key=lambda item: item[0], reverse=True)
    positives_seen = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label == _POSITIVE_LABEL:
            positives_seen += 1
            precision_sum += positives_seen / rank
    return precision_sum / n_pos


def _sweep_thresholds(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Return an inclusive grid of thresholds in ``[start, stop]`` spaced by ``step``."""
    if step <= 0.0:
        raise ValueError(f"step must be > 0 (got {step!r})")
    if stop < start:
        raise ValueError(f"stop must be >= start (got start={start!r} stop={stop!r})")
    points = round((stop - start) / step) + 1
    return tuple(start + step * index for index in range(points))


def threshold_sweep(
    scored_labels: Sequence[tuple[float, int]],
    *,
    start: float = _DEFAULT_SWEEP_START,
    stop: float = _DEFAULT_SWEEP_STOP,
    step: float = _DEFAULT_SWEEP_STEP,
) -> tuple[ThresholdPoint, ...]:
    """Sweep ``[start, stop]`` (inclusive) computing P/R/F1 at each threshold.

    At each threshold ``t``, pairs with ``score >= t`` are predicted
    positive; everything else predicted negative. The four-cell
    contingency table (TP, FP, FN, TN) is then projected onto
    precision/recall/F1.

    Args:
        scored_labels: Sequence of ``(score, label)`` tuples; ``label``
            must be ``0`` or ``1``.
        start: Inclusive lower endpoint of the sweep.
        stop: Inclusive upper endpoint of the sweep.
        step: Grid spacing; must be positive.

    Returns:
        One :class:`ThresholdPoint` per grid point, in ascending
        threshold order.

    Raises:
        ValueError: If ``scored_labels`` is empty or contains a label
            outside ``{0, 1}``; or if the sweep grid is malformed.
    """
    _validate_labels(scored_labels)
    thresholds = _sweep_thresholds(start, stop, step)
    n_pos, _ = _count_classes(scored_labels)
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        for score, label in scored_labels:
            if score >= threshold:
                if label == _POSITIVE_LABEL:
                    true_positives += 1
                else:
                    false_positives += 1
        false_negatives = n_pos - true_positives
        predicted_positive = true_positives + false_positives
        precision = _safe_division(true_positives, predicted_positive)
        recall = _safe_division(true_positives, n_pos)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
            )
        )
    return tuple(points)


__all__ = [
    "ThresholdPoint",
    "average_precision",
    "roc_auc",
    "threshold_sweep",
]
