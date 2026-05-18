"""Platt scaling for the weighted-mean combiner.

Platt scaling fits a logistic regression ``P(y=1 | x) = 1 / (1 + exp(a*x + b))``
to the ``(raw_score, is_true_match)`` pairs from the ground-truth corpus.
The training routine implements a small handwritten Newton-Raphson with
backtracking; we deliberately avoid pulling scikit-learn into Phase 4
(that dependency is reserved for the Phase 9 learned combiner).

Run a trained calibrator once via :func:`calibrate` to map a raw score in
``[0, 100]`` to a probability in ``[0, 1]``. Persist it via msgpack with
:func:`save_calibrator` / :func:`load_calibrator`.
"""

from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from math import exp
from math import log
from pathlib import Path

from msgspec import Struct
from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder

_MAX_NEWTON_ITERATIONS: int = 100
_GRADIENT_TOLERANCE: float = 1e-7
_HESSIAN_RIDGE: float = 1e-12


class PlattCalibrator(Struct, frozen=True, forbid_unknown_fields=True):
    """Frozen calibrator parameters ``(a, b)`` plus provenance metadata.

    Attributes:
        a: Slope coefficient (negative for monotone-increasing calibrators).
        b: Intercept coefficient.
        trained_at: ISO-8601 UTC timestamp of training.
        n_positive: Count of positive examples used in training.
        n_negative: Count of negative examples used in training.
    """

    a: float
    b: float
    trained_at: str
    n_positive: int
    n_negative: int


_ENCODER: Encoder = Encoder()
_DECODER: Decoder[PlattCalibrator] = Decoder(PlattCalibrator)


def _sigmoid(value: float) -> float:
    """Stable logistic sigmoid for the calibrator's link function."""
    if value >= 0:
        ez = exp(-value)
        return 1.0 / (1.0 + ez)
    ez = exp(value)
    return ez / (1.0 + ez)


def calibrate(raw_score: float, calibrator: PlattCalibrator) -> float:
    """Return ``P(true match)`` for ``raw_score`` (in ``[0, 100]``)."""
    z = calibrator.a * raw_score + calibrator.b
    return _sigmoid(-z)


def _log_likelihood(
    scores: Sequence[float],
    targets: Sequence[float],
    a: float,
    b: float,
) -> float:
    total = 0.0
    for score, target in zip(scores, targets, strict=True):
        z = a * score + b
        # Stable log(1+exp(z)) via numpy-style trick.
        log1pez = z + log(1.0 + exp(-z)) if z >= 0 else log(1.0 + exp(z))
        total += target * z + log1pez
    return total


def train_calibrator(
    positives: Sequence[float],
    negatives: Sequence[float],
    *,
    max_iterations: int = _MAX_NEWTON_ITERATIONS,
) -> PlattCalibrator:
    """Fit a :class:`PlattCalibrator` via Newton-Raphson with backtracking.

    Targets follow the standard Platt prior smoothing: positives use
    ``(N+ + 1) / (N+ + 2)`` and negatives use ``1 / (N- + 2)``. The fit
    minimises the regularised log-loss using a few Newton iterations; the
    objective is convex so the line-search-free Newton converges quickly
    even on tiny training sets, which is the only scale Phase 4's tests
    exercise.

    Args:
        positives: Raw scores in ``[0, 100]`` for true matches.
        negatives: Raw scores in ``[0, 100]`` for non-matches.

    Returns:
        A fitted :class:`PlattCalibrator`.

    Raises:
        ValueError: If either set is empty.
    """
    if not positives or not negatives:
        raise ValueError("train_calibrator requires non-empty positives and negatives")
    n_pos = len(positives)
    n_neg = len(negatives)
    pos_target = (n_pos + 1.0) / (n_pos + 2.0)
    neg_target = 1.0 / (n_neg + 2.0)
    scores: list[float] = [*list(positives), *list(negatives)]
    targets: list[float] = [pos_target] * n_pos + [neg_target] * n_neg
    a = 0.0
    b = log((n_neg + 1.0) / (n_pos + 1.0))
    prev_loss = _log_likelihood(scores, targets, a, b)
    for _ in range(max_iterations):
        # Compute gradient and Hessian.
        h11 = _HESSIAN_RIDGE
        h22 = _HESSIAN_RIDGE
        h12 = 0.0
        g1 = 0.0
        g2 = 0.0
        for score, target in zip(scores, targets, strict=True):
            z = a * score + b
            p = _sigmoid(z)
            diff = p - (1.0 - target)
            g1 += score * diff
            g2 += diff
            d = p * (1.0 - p)
            h11 += score * score * d
            h12 += score * d
            h22 += d
        det = h11 * h22 - h12 * h12
        delta_a = (h22 * g1 - h12 * g2) / det
        delta_b = (-h12 * g1 + h11 * g2) / det
        step = 1.0
        new_a = a - step * delta_a
        new_b = b - step * delta_b
        new_loss = _log_likelihood(scores, targets, new_a, new_b)
        # Backtracking line search to guarantee progress.
        while new_loss > prev_loss and step > 1e-10:
            step *= 0.5
            new_a = a - step * delta_a
            new_b = b - step * delta_b
            new_loss = _log_likelihood(scores, targets, new_a, new_b)
        a, b = new_a, new_b
        if abs(prev_loss - new_loss) < _GRADIENT_TOLERANCE:
            prev_loss = new_loss
            break
        prev_loss = new_loss
    timestamp = datetime.now(tz=UTC).isoformat()
    return PlattCalibrator(
        a=a,
        b=b,
        trained_at=timestamp,
        n_positive=n_pos,
        n_negative=n_neg,
    )


def save_calibrator(calibrator: PlattCalibrator, path: Path) -> None:
    """Persist ``calibrator`` to ``path`` via msgspec msgpack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ENCODER.encode(calibrator))


def load_calibrator(path: Path) -> PlattCalibrator:
    """Load a :class:`PlattCalibrator` previously persisted via :func:`save_calibrator`."""
    return _DECODER.decode(path.read_bytes())


__all__ = [
    "PlattCalibrator",
    "calibrate",
    "load_calibrator",
    "save_calibrator",
    "train_calibrator",
]
