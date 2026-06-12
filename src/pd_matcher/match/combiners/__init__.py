"""Evidence combiners producing the final :class:`CombinedScore`.

:func:`build_combiner` is the single dispatch point that turns a
:class:`~pd_matcher.config.schemas.MatchingConfig` into a concrete
:class:`~pd_matcher.match.combiners.base.Combiner`. The default
``weighted_mean`` scorer needs no artifact; the ``learned`` scorer loads its
LightGBM model from ``learned_model_dir`` and fails loudly (naming the
``train-scorer`` command) when that directory is missing or has no artifact.
"""

from pathlib import Path

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.learned import META_FILENAME
from pd_matcher.match.combiners.learned import load_learned_model
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner

_LEARNED_SCORER: str = "learned"
_TRAIN_COMMAND: str = "pdm run pd-matcher train-scorer"


def build_combiner(
    config: MatchingConfig,
    *,
    learned_model_dir: Path | None,
) -> Combiner:
    """Return the combiner selected by ``config.scorer``.

    Args:
        config: Active matching config; ``config.scorer`` selects the
            implementation (``"weighted_mean"`` or ``"learned"``).
        learned_model_dir: Directory holding the learned-model artifact
            (``learned_scorer.txt`` + ``learned_scorer.msgpack``). Required
            when ``config.scorer == "learned"``; ignored otherwise.

    Returns:
        A :class:`Combiner`.

    Raises:
        ValueError: When ``config.scorer == "learned"`` but
            ``learned_model_dir`` is ``None`` or has no artifact.
    """
    if config.scorer != _LEARNED_SCORER:
        return WeightedMeanCombiner(config=config)
    if learned_model_dir is None or not (learned_model_dir / META_FILENAME).is_file():
        raise ValueError(
            "scorer is 'learned' but no learned-model artifact was found"
            + (f" in {learned_model_dir}" if learned_model_dir is not None else "")
            + f". Train one with `{_TRAIN_COMMAND}`."
        )
    return load_learned_model(learned_model_dir)


__all__ = [
    "build_combiner",
]
