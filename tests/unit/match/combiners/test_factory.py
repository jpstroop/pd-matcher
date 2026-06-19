"""Tests for :func:`pd_matcher.match.combiners.build_combiner`."""

from pathlib import Path
from typing import Literal

from lightgbm import LGBMClassifier
from numpy import asarray
from numpy import float64
from numpy import int64
from pytest import raises

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.learned import LearnedCombiner
from pd_matcher.match.combiners.learned import model_metadata
from pd_matcher.match.combiners.learned import save_learned_model
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner

_N_FEATURES: int = 53


def _config(scorer: Literal["weighted_mean", "learned"]) -> MatchingConfig:
    """Return a valid matching config (weights sum to 1.0) with ``scorer`` set."""
    return MatchingConfig(
        title_weight=0.45,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.0,
        extent_weight=0.05,
        volume_weight=0.05,
        year_window=0,
        min_combined_score=0.0,
        scorer=scorer,
    )


def _write_model(directory: Path) -> None:
    """Train and persist a tiny learned-model artifact under ``directory``."""
    rows = [[0.9 if i % 2 == 0 else 0.1] + [0.0] * (_N_FEATURES - 1) for i in range(20)]
    labels = [1 if i % 2 == 0 else 0 for i in range(20)]
    model = LGBMClassifier(
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        n_estimators=10,
        random_state=0,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(asarray(rows, dtype=float64), asarray(labels, dtype=int64).astype(float64))
    meta = model_metadata(
        model.booster_,
        n_positive=10,
        n_negative=10,
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        lambda_l2=1.0,
        n_estimators=10,
        class_weight="balanced",
    )
    save_learned_model(model.booster_, meta, directory)


def test_build_combiner_weighted_mean_default() -> None:
    """The default scorer yields a weighted-mean combiner and ignores the dir."""
    combiner = build_combiner(_config("weighted_mean"), learned_model_dir=None)
    assert isinstance(combiner, WeightedMeanCombiner)


def test_build_combiner_learned_loads_artifact(tmp_path: Path) -> None:
    """The learned scorer loads the artifact into a LearnedCombiner."""
    _write_model(tmp_path)
    combiner = build_combiner(_config("learned"), learned_model_dir=tmp_path)
    assert isinstance(combiner, LearnedCombiner)


def test_build_combiner_learned_without_dir_raises() -> None:
    """A learned scorer with no directory fails loudly and names train-scorer."""
    with raises(ValueError, match="train-scorer"):
        build_combiner(_config("learned"), learned_model_dir=None)


def test_build_combiner_learned_missing_artifact_raises(tmp_path: Path) -> None:
    """A learned scorer pointed at an empty directory fails loudly."""
    with raises(ValueError, match="train-scorer"):
        build_combiner(_config("learned"), learned_model_dir=tmp_path)
