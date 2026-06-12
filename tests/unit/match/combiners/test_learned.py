"""Tests for :mod:`pd_matcher.match.combiners.learned`."""

from collections.abc import Sequence
from pathlib import Path

from lightgbm import Booster
from lightgbm import LGBMClassifier
from numpy import asarray
from numpy import float64
from numpy import int64
from numpy.typing import NDArray
from pytest import MonkeyPatch
from pytest import fixture
from pytest import raises

from pd_matcher.match.combiners import learned as learned_module
from pd_matcher.match.combiners.features import feature_names
from pd_matcher.match.combiners.learned import LearnedCombiner
from pd_matcher.match.combiners.learned import LearnedModelMeta
from pd_matcher.match.combiners.learned import load_learned_model
from pd_matcher.match.combiners.learned import model_metadata
from pd_matcher.match.combiners.learned import save_learned_model
from pd_matcher.match.evidence import Evidence

_N_FEATURES: int = 53
_N_ROWS: int = 30


def _synthetic_matrix() -> tuple[NDArray[float64], NDArray[int64]]:
    """Build a tiny separable matrix: first feature carries the label signal."""
    rows: list[list[float]] = []
    labels: list[int] = []
    for index in range(_N_ROWS):
        positive = index % 2 == 0
        row = [0.0] * _N_FEATURES
        row[0] = 0.9 if positive else 0.1
        row[1] = 0.8 if positive else 0.2
        rows.append(row)
        labels.append(1 if positive else 0)
    return asarray(rows, dtype=float64), asarray(labels, dtype=int64)


def _train_booster() -> Booster:
    """Fit a small deterministic LightGBM and return its Booster."""
    x, y = _synthetic_matrix()
    model = LGBMClassifier(
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        n_estimators=20,
        random_state=0,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(x, y.astype(float64))
    return model.booster_


def _meta(booster: Booster) -> LearnedModelMeta:
    """Build matching metadata for ``booster``."""
    return model_metadata(
        booster,
        n_positive=_N_ROWS // 2,
        n_negative=_N_ROWS // 2,
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        lambda_l2=1.0,
        n_estimators=20,
        class_weight="balanced",
    )


def _full_evidence() -> Sequence[Evidence]:
    """A title-heavy Evidence set so feature_row has a non-zero leading value."""
    return (
        Evidence(
            scorer="title.token_set",
            score=90.0,
            max=100.0,
            skipped=False,
            decisive=False,
            features=(
                ("token_overlap", 4.0),
                ("marc_token_len", 5.0),
                ("nypl_token_len", 5.0),
            ),
        ),
        Evidence(
            scorer="name.author",
            score=80.0,
            max=100.0,
            skipped=False,
            decisive=False,
            features=(("token_overlap", 2.0),),
        ),
    )


@fixture
def trained_booster() -> Booster:
    """A freshly trained tiny Booster for the round-trip tests."""
    return _train_booster()


def test_combine_returns_probability_score(trained_booster: Booster) -> None:
    """combine() yields calibrated in [0,1] and raw == 100 * calibrated."""
    combiner = LearnedCombiner(booster=trained_booster, names=feature_names())
    result = combiner.combine(_full_evidence())
    assert 0.0 <= result.calibrated <= 1.0
    assert result.raw == result.calibrated * 100.0


def test_save_load_round_trip_predictions_match(
    trained_booster: Booster,
    tmp_path: Path,
) -> None:
    """A saved-then-loaded model predicts identically to the in-memory one."""
    save_learned_model(trained_booster, _meta(trained_booster), tmp_path)
    loaded = load_learned_model(tmp_path)
    original = LearnedCombiner(booster=trained_booster, names=feature_names())
    evidence = _full_evidence()
    assert loaded.combine(evidence).calibrated == original.combine(evidence).calibrated


def test_save_writes_both_artifact_files(
    trained_booster: Booster,
    tmp_path: Path,
) -> None:
    """Both the Booster text dump and the msgpack metadata land on disk."""
    save_learned_model(trained_booster, _meta(trained_booster), tmp_path)
    assert (tmp_path / learned_module.MODEL_FILENAME).is_file()
    assert (tmp_path / learned_module.META_FILENAME).is_file()


def test_load_rejects_stale_feature_names(
    trained_booster: Booster,
    tmp_path: Path,
) -> None:
    """A feature-name mismatch raises and names the retrain command."""
    stale = LearnedModelMeta(
        feature_names=("only", "two"),
        trained_at="2026-06-12T00:00:00+00:00",
        n_positive=1,
        n_negative=1,
        lightgbm_version="4.6.0",
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        lambda_l2=1.0,
        n_estimators=20,
        class_weight="balanced",
    )
    save_learned_model(trained_booster, stale, tmp_path)
    with raises(ValueError, match="train-scorer"):
        load_learned_model(tmp_path)


def test_require_lightgbm_reraises_as_install_hint(monkeypatch: MonkeyPatch) -> None:
    """The lazy import helper surfaces a clear install message on ImportError.

    Setting ``sys.modules['lightgbm'] = None`` makes the next
    ``from lightgbm import Booster`` raise ``ImportError`` without uninstalling
    the package, which exercises the lazy-import failure branch. monkeypatch
    restores the real module afterwards.
    """
    from sys import modules

    monkeypatch.setitem(modules, "lightgbm", None)
    with raises(ImportError, match=r"ml.*dependency group"):
        learned_module._require_lightgbm()


def test_metadata_records_feature_contract(trained_booster: Booster) -> None:
    """model_metadata stamps the canonical feature names and the lgbm version."""
    meta = _meta(trained_booster)
    assert meta.feature_names == feature_names()
    assert meta.lightgbm_version
    assert meta.n_positive == _N_ROWS // 2
