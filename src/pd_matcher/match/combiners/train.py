"""Training pipeline for the LightGBM learned combiner (issue #4).

``train-scorer`` calls :func:`train_learned_model`: it resolves every
non-``unsure`` vault pair against the candidate pool and the CCE index, scores
each pair through the *production* matching pipeline to obtain the winning
Evidence, projects that Evidence through the canonical
:func:`pd_matcher.match.combiners.features.feature_row`, fits a LightGBM
classifier with the locked issue #4 hyperparameters, and reports a 5-fold
out-of-fold AUC sanity number. The caller persists the returned Booster and
metadata via :func:`pd_matcher.match.combiners.learned.save_learned_model`.

``lightgbm`` and ``scikit-learn`` are imported lazily inside this module's
functions (never at import time) so the optional ``ml`` dependency group is
only required when training actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final

from numpy import asarray
from numpy import float64
from numpy import int64
from numpy.typing import NDArray

from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.features import feature_row
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings

if TYPE_CHECKING:  # pragma: no cover
    from lightgbm import Booster
    from lightgbm import LGBMClassifier

_LOGGER = getLogger(__name__)

_VERDICT_MATCH: Final[str] = "match"
_VERDICT_UNSURE: Final[str] = "unsure"

# Locked recipe from docs/findings/learned_scorer_tightening_2026-06-12.md.
MAX_DEPTH: Final[int] = 3
NUM_LEAVES: Final[int] = 8
MIN_DATA_IN_LEAF: Final[int] = 10
LAMBDA_L2: Final[float] = 1.0
N_ESTIMATORS: Final[int] = 200
CLASS_WEIGHT: Final[str] = "balanced"

_RANDOM_STATE: Final[int] = 20260612
_N_SPLITS: Final[int] = 5


def _scoring_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` with the weighted-mean scorer forced on.

    Training extracts per-scorer Evidence, which is identical no matter which
    combiner runs; the combined score is discarded. Forcing weighted-mean
    avoids the chicken-and-egg of needing a learned artifact to produce the
    very data that trains it.
    """
    if config.scorer == "weighted_mean":
        return config
    return MatchingConfig(
        title_weight=config.title_weight,
        author_weight=config.author_weight,
        publisher_weight=config.publisher_weight,
        edition_weight=config.edition_weight,
        lccn_weight=config.lccn_weight,
        isbn_weight=config.isbn_weight,
        extent_weight=config.extent_weight,
        volume_weight=config.volume_weight,
        year_window=config.year_window,
        min_combined_score=config.min_combined_score,
        scorer="weighted_mean",
    )


@dataclass(frozen=True, slots=True)
class TrainingMatrix:
    """The resolved training data plus the skip totals for reporting."""

    x: NDArray[float64]
    y: NDArray[int64]
    n_positive: int
    n_negative: int
    missing_in_pool: int
    missing_in_index: int


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A fitted Booster plus its OOF sanity AUC and class counts."""

    booster: Booster
    oof_auc: float
    n_positive: int
    n_negative: int


def build_training_matrix(
    *,
    vault_path: Path,
    pool_path: Path,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
) -> TrainingMatrix:
    """Resolve and score every trainable vault pair into a feature matrix.

    Reuses the same resolution machinery as the eval pass A
    (:func:`make_pair_scorer`, :func:`build_marc_index`) so the rows carry the
    exact Evidence the production matcher would emit. ``unsure`` verdicts are
    excluded. Pairs whose MARC is absent from the pool or whose CCE is absent
    from the index are logged and skipped.
    """
    raw = current_entries(vault_path)
    kept = [entry for entry in raw.values() if entry.verdict != _VERDICT_UNSURE]
    needed_marc_ids = {entry.marc_control_id for entry in kept}
    marc_by_id = build_marc_index(pool_path, needed_marc_ids)
    pairings = compile_pairings(pairing_config)
    scoring_config = _scoring_config(matching_config)

    feature_rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    missing_in_pool = 0
    missing_in_index = 0
    with NyplIndexLookup(index_path) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
        score_pair = make_pair_scorer(
            matching_config=scoring_config,
            pairings=pairings,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            calibrator=None,
        )
        for entry in kept:
            marc = marc_by_id.get(entry.marc_control_id)
            if marc is None:
                missing_in_pool += 1
                _LOGGER.warning(
                    "train.vault.marc_not_in_pool marc_control_id=%s nypl_uuid=%s",
                    entry.marc_control_id,
                    entry.nypl_uuid,
                )
                continue
            cce = lookup.get_registration(entry.nypl_uuid)
            if cce is None:
                missing_in_index += 1
                _LOGGER.warning(
                    "train.vault.cce_not_in_index marc_control_id=%s nypl_uuid=%s",
                    entry.marc_control_id,
                    entry.nypl_uuid,
                )
                continue
            candidate = score_pair(marc, cce)
            feature_rows.append(feature_row(candidate.evidence))
            labels.append(1 if entry.verdict == _VERDICT_MATCH else 0)

    x = asarray(feature_rows, dtype=float64)
    y = asarray(labels, dtype=int64)
    return TrainingMatrix(
        x=x,
        y=y,
        n_positive=sum(labels),
        n_negative=len(labels) - sum(labels),
        missing_in_pool=missing_in_pool,
        missing_in_index=missing_in_index,
    )


def _new_classifier() -> LGBMClassifier:
    """Construct an LGBMClassifier with the locked issue #4 hyperparameters."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        max_depth=MAX_DEPTH,
        num_leaves=NUM_LEAVES,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        reg_lambda=LAMBDA_L2,
        n_estimators=N_ESTIMATORS,
        class_weight=CLASS_WEIGHT,
        objective="binary",
        verbose=-1,
        random_state=_RANDOM_STATE,
        n_jobs=1,
    )


def _oof_auc(x: NDArray[float64], y: NDArray[int64]) -> float:
    """Return the 5-fold out-of-fold ROC-AUC as a deterministic sanity number."""
    from numpy import zeros
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    oof: NDArray[float64] = zeros(x.shape[0], dtype=float64)
    splitter = StratifiedKFold(
        n_splits=_N_SPLITS,
        shuffle=True,
        random_state=_RANDOM_STATE,
    )
    y_float = y.astype(float64)
    for train_idx, test_idx in splitter.split(x, y):
        model = _new_classifier()
        model.fit(x[train_idx], y_float[train_idx])
        probabilities = asarray(model.predict_proba(x[test_idx]), dtype=float64)[:, 1]
        oof[test_idx] = probabilities
    return float(roc_auc_score(y, oof))


def train_learned_model(matrix: TrainingMatrix) -> TrainedModel:
    """Fit the final Booster on all trainable rows after an OOF AUC sanity pass.

    Raises:
        ValueError: When either class is empty (a single-class corpus cannot
            train a binary classifier).
    """
    if matrix.n_positive == 0 or matrix.n_negative == 0:
        raise ValueError(
            "train-scorer needs both match and no_match labels; got "
            f"{matrix.n_positive} positive / {matrix.n_negative} negative"
        )
    oof_auc = _oof_auc(matrix.x, matrix.y)
    model = _new_classifier()
    model.fit(matrix.x, matrix.y.astype(float64))
    return TrainedModel(
        booster=model.booster_,
        oof_auc=oof_auc,
        n_positive=matrix.n_positive,
        n_negative=matrix.n_negative,
    )


__all__ = [
    "CLASS_WEIGHT",
    "LAMBDA_L2",
    "MAX_DEPTH",
    "MIN_DATA_IN_LEAF",
    "NUM_LEAVES",
    "N_ESTIMATORS",
    "TrainedModel",
    "TrainingMatrix",
    "build_training_matrix",
    "train_learned_model",
]
