"""LightGBM learned combiner plus its on-disk model artifact.

The learned combiner replaces the weighted mean's hand-tuned linear blend
with a gradient-boosted classifier trained on the label vault (issue #4).
It consumes the same per-scorer :class:`pd_matcher.match.evidence.Evidence`
stream as :class:`~pd_matcher.match.combiners.weighted_mean.WeightedMeanCombiner`,
projects it through the canonical
:func:`pd_matcher.match.combiners.features.feature_row`, and returns the
model's raw probability as the calibrated score (the tightening round found
raw LightGBM probabilities are already the best-calibrated variant, so there
is NO separate calibration layer).

The artifact is two files in the index's cache directory:

* ``learned_scorer.txt`` — the LightGBM Booster text dump.
* ``learned_scorer.msgpack`` — a :class:`LearnedModelMeta` carrying the
  feature-name contract and training provenance.

``lightgbm`` is imported lazily inside the functions that need it (never at
module level) so the rest of the matcher — and the default weighted-mean
path — never pays the import cost or requires the optional ``ml`` dependency
group to be installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from msgspec import Struct
from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder
from numpy import asarray
from numpy import float64
from numpy.typing import NDArray

from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.features import feature_names
from pd_matcher.match.combiners.features import feature_row
from pd_matcher.match.evidence import Evidence

if TYPE_CHECKING:  # pragma: no cover
    from lightgbm import Booster

_RAW_MAX: float = 100.0

MODEL_FILENAME: str = "learned_scorer.txt"
META_FILENAME: str = "learned_scorer.msgpack"

_TRAIN_COMMAND: str = "pdm run pd-matcher train-scorer"


class LearnedModelMeta(Struct, frozen=True, forbid_unknown_fields=True):
    """Feature-name contract and provenance for a trained learned model.

    The hyperparameters are stored as explicit typed fields (not a dict) so
    the artifact is self-describing and strictly typed. ``feature_names`` is
    the canonical column order the model was trained on; inference refuses to
    run a model whose names disagree with the current
    :func:`pd_matcher.match.combiners.features.feature_names`.
    """

    feature_names: tuple[str, ...]
    trained_at: str
    n_positive: int
    n_negative: int
    lightgbm_version: str
    max_depth: int
    num_leaves: int
    min_data_in_leaf: int
    lambda_l2: float
    n_estimators: int
    class_weight: str


_META_ENCODER: Encoder = Encoder()
_META_DECODER: Decoder[LearnedModelMeta] = Decoder(LearnedModelMeta)


def _require_lightgbm() -> type[Booster]:
    """Import and return :class:`lightgbm.Booster`, or raise a clear error.

    Isolated in one helper so the lazy import has a single test seam and so
    the missing-dependency message is identical everywhere lightgbm is
    needed.
    """
    try:
        from lightgbm import Booster
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for the learned scorer; install the optional "
            "'ml' dependency group (pdm install -G ml)"
        ) from exc
    return Booster


def save_learned_model(
    booster: Booster,
    meta: LearnedModelMeta,
    directory: Path,
) -> None:
    """Persist a trained Booster and its metadata under ``directory``.

    Writes ``learned_scorer.txt`` (the Booster dump) and
    ``learned_scorer.msgpack`` (the :class:`LearnedModelMeta`). The directory
    is created if absent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(directory / MODEL_FILENAME)
    (directory / META_FILENAME).write_bytes(_META_ENCODER.encode(meta))


def load_learned_model(directory: Path) -> LearnedCombiner:
    """Load a trained learned model from ``directory`` into a combiner.

    Reads both artifact files, validates that the stored feature-name
    contract matches the current :func:`feature_names`, and returns a ready
    :class:`LearnedCombiner`.

    Raises:
        ImportError: When the optional lightgbm dependency is not installed.
        ValueError: When the stored feature names disagree with the current
            canonical order (the model is stale; retrain it).
    """
    booster_cls = _require_lightgbm()
    meta = _META_DECODER.decode((directory / META_FILENAME).read_bytes())
    current = feature_names()
    if meta.feature_names != current:
        raise ValueError(
            "learned model feature names are stale: the model was trained on a "
            f"different feature set ({len(meta.feature_names)} columns) than the "
            f"current pipeline emits ({len(current)} columns). Retrain with "
            f"`{_TRAIN_COMMAND}`."
        )
    booster = booster_cls(model_file=str(directory / MODEL_FILENAME))
    return LearnedCombiner(booster=booster, names=current)


@dataclass(frozen=True, slots=True)
class LearnedCombiner:
    """Combiner backed by a trained LightGBM Booster.

    Holds the Booster and the feature-name order it was trained on. A plain
    frozen slotted dataclass (not a msgspec Struct) because the Booster is an
    opaque foreign object msgspec cannot validate or encode.
    """

    booster: Booster
    names: tuple[str, ...]

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:
        """Score one pair's Evidence via the Booster's match probability.

        Projects ``evidence`` through the canonical
        :func:`feature_row`, predicts a single probability, and returns it as
        both the calibrated score (``[0, 1]``) and the raw score
        (``probability * 100``).
        """
        row: NDArray[float64] = asarray([feature_row(evidence)], dtype=float64)
        predicted: NDArray[float64] = asarray(self.booster.predict(row), dtype=float64)
        probability = float(predicted[0])
        return CombinedScore(raw=probability * _RAW_MAX, calibrated=probability)


def model_metadata(
    booster: Booster,
    *,
    n_positive: int,
    n_negative: int,
    max_depth: int,
    num_leaves: int,
    min_data_in_leaf: int,
    lambda_l2: float,
    n_estimators: int,
    class_weight: str,
) -> LearnedModelMeta:
    """Build a :class:`LearnedModelMeta` for ``booster`` at the current time.

    The feature-name contract is taken from the canonical
    :func:`feature_names`; ``train-scorer`` is responsible for having trained
    the Booster on exactly that column order. ``lightgbm_version`` is read
    from the installed package so a stale artifact is traceable.
    """
    del booster
    from lightgbm import __version__ as lightgbm_version

    return LearnedModelMeta(
        feature_names=feature_names(),
        trained_at=datetime.now(tz=UTC).isoformat(),
        n_positive=n_positive,
        n_negative=n_negative,
        lightgbm_version=lightgbm_version,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        lambda_l2=lambda_l2,
        n_estimators=n_estimators,
        class_weight=class_weight,
    )


__all__ = [
    "META_FILENAME",
    "MODEL_FILENAME",
    "LearnedCombiner",
    "LearnedModelMeta",
    "load_learned_model",
    "model_metadata",
    "save_learned_model",
]
