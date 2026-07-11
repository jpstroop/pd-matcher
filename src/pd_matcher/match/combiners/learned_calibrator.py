"""Isotonic post-hoc calibration for the learned (LightGBM) arm (issue #130).

The learned arm reports its LightGBM match probability directly as the
calibrated score. On the honest (out-of-fold) vault that probability is
already well-calibrated, but the *deployed* model — trained on the whole
vault — is overconfident on the pairs it memorised, and the accept pile is
score-compressed near ``1.0``. This module fits a monotone isotonic mapping
``raw probability -> honest probability`` on out-of-fold predictions so the
learned arm's ``[0, 1]`` score means the same thing the weighted arm's Platt
output does, without changing the ranking (isotonic regression is monotone,
so top-1 selection is untouched).

The artifact is a single ``learned_calibrator.msgpack`` living next to the
learned-model artifact, loaded automatically by
:func:`pd_matcher.match.combiners.learned.load_learned_model` when present —
the exact lifecycle the weighted arm's ``calibrator.msgpack`` follows.
"""

from bisect import bisect_right
from pathlib import Path

from msgspec import Struct
from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder

CALIBRATOR_FILENAME: str = "learned_calibrator.msgpack"


class IsotonicCalibrator(Struct, frozen=True, forbid_unknown_fields=True):
    """Piecewise-linear monotone calibrator plus provenance.

    Attributes:
        xs: Strictly increasing raw-probability breakpoints in ``[0, 1]``.
        ys: Non-decreasing calibrated probabilities in ``[0, 1]``, one per
            breakpoint (the isotonic fit's step values).
        trained_at: ISO-8601 UTC timestamp of the fit.
        n_positive: Positive examples used in the out-of-fold fit.
        n_negative: Negative examples used in the out-of-fold fit.
    """

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    trained_at: str
    n_positive: int
    n_negative: int


_ENCODER: Encoder = Encoder()
_DECODER: Decoder[IsotonicCalibrator] = Decoder(IsotonicCalibrator)


def apply_isotonic(probability: float, calibrator: IsotonicCalibrator) -> float:
    """Map a raw learned probability to its calibrated probability.

    Clamps to the fitted range at both ends and linearly interpolates between
    breakpoints. Monotone non-decreasing in ``probability`` by construction, so
    applying it never reorders a set of candidates.
    """
    xs = calibrator.xs
    ys = calibrator.ys
    if probability <= xs[0]:
        return ys[0]
    if probability >= xs[-1]:
        return ys[-1]
    upper = bisect_right(xs, probability)
    x0 = xs[upper - 1]
    x1 = xs[upper]
    y0 = ys[upper - 1]
    y1 = ys[upper]
    return y0 + (y1 - y0) * (probability - x0) / (x1 - x0)


def save_learned_calibrator(calibrator: IsotonicCalibrator, path: Path) -> None:
    """Persist ``calibrator`` to ``path`` via msgspec msgpack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ENCODER.encode(calibrator))


def load_learned_calibrator(path: Path) -> IsotonicCalibrator:
    """Load an :class:`IsotonicCalibrator` previously persisted to ``path``."""
    return _DECODER.decode(path.read_bytes())


__all__ = [
    "CALIBRATOR_FILENAME",
    "IsotonicCalibrator",
    "apply_isotonic",
    "load_learned_calibrator",
    "save_learned_calibrator",
]
