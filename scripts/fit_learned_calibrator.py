"""Fit the learned arm's isotonic calibrator on out-of-fold predictions (#130).

Throwaway research instrument, mirroring ``scripts/fit_calibrator.py`` for the
weighted arm. NOT shipped; ``scripts/`` is excluded from the coverage source
allowlist.

Resolves every non-``unsure`` vault pair, projects each through the canonical
:func:`feature_row`, generates leakage-free out-of-fold LightGBM probabilities
(``GroupKFold`` by ``marc_control_id``, matching ``scripts/learned_scorer_heldout``),
fits :class:`sklearn.isotonic.IsotonicRegression` on the OOF pairs, and persists
the breakpoints as an :class:`IsotonicCalibrator` beside the learned model. The
isotonic fit is monotone, so it never changes ranking or top-1 selection.

Usage:
    pdm run python scripts/fit_learned_calibrator.py \\
        > docs/findings/fit_learned_calibrator_<date>.md
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Final

from numpy import asarray
from numpy import float64
from numpy import zeros
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import build_marc_index_from_collection
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.features import feature_row
from pd_matcher.match.combiners.learned_calibrator import CALIBRATOR_FILENAME
from pd_matcher.match.combiners.learned_calibrator import IsotonicCalibrator
from pd_matcher.match.combiners.learned_calibrator import save_learned_calibrator
from pd_matcher.match.combiners.train import CLASS_WEIGHT
from pd_matcher.match.combiners.train import LAMBDA_L2
from pd_matcher.match.combiners.train import MAX_DEPTH
from pd_matcher.match.combiners.train import MIN_DATA_IN_LEAF
from pd_matcher.match.combiners.train import N_ESTIMATORS
from pd_matcher.match.combiners.train import NUM_LEAVES
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings

_VAULT_PATH: Final[Path] = Path("data/training/label_vault.jsonl")
_MARC_COLLECTION: Final[Path] = Path("data/training/marc.xml")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_OUT_PATH: Final[Path] = Path("caches") / CALIBRATOR_FILENAME
_N_SPLITS: Final[int] = 5
_VERDICT_MATCH: Final[str] = "match"
_VERDICT_UNSURE: Final[str] = "unsure"
_RANDOM_STATE: Final[int] = 20260612
_LGB_PARAMS: Final[dict[str, object]] = {
    "max_depth": MAX_DEPTH, "num_leaves": NUM_LEAVES, "min_data_in_leaf": MIN_DATA_IN_LEAF,
    "reg_lambda": LAMBDA_L2, "n_estimators": N_ESTIMATORS, "class_weight": CLASS_WEIGHT,
    "objective": "binary", "verbose": -1, "random_state": _RANDOM_STATE, "n_jobs": 1,
}


def _collect() -> tuple[list[tuple[float, ...]], list[int], list[str]]:
    """Return (feature rows, labels, marc ids) for every resolvable vault pair."""
    matching_config = _load_default_matching_config()
    pairings = compile_pairings(_load_default_pairing_config())
    entries = [e for e in current_entries(_VAULT_PATH).values() if e.verdict != _VERDICT_UNSURE]
    needed = {e.marc_control_id for e in entries}
    marc_by_id = build_marc_index_from_collection(_MARC_COLLECTION, needed)
    remaining = needed - set(marc_by_id)
    if remaining:
        marc_by_id |= build_marc_index(_POOL_PATH, remaining)
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    marc_ids: list[str] = []
    with NyplIndexLookup(_INDEX_PATH) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
        score_pair = make_pair_scorer(
            matching_config=matching_config, pairings=pairings, idf=idf,
            author_idf=author_idf, publisher_idf=publisher_idf,
            calibrator=None, learned_model_dir=None,
        )
        for entry in entries:
            marc = marc_by_id.get(entry.marc_control_id)
            if marc is None:
                continue
            cce = lookup.get_registration(entry.nypl_uuid)
            if cce is None:
                continue
            rows.append(feature_row(score_pair(marc, cce).evidence))
            labels.append(1 if entry.verdict == _VERDICT_MATCH else 0)
            marc_ids.append(entry.marc_control_id)
    return rows, labels, marc_ids


def main() -> None:
    from lightgbm import LGBMClassifier

    rows, labels, marc_ids = _collect()
    x = asarray(rows, dtype=float64)
    y = asarray(labels, dtype=float64)
    groups = asarray(marc_ids)
    oof = zeros(len(rows), dtype=float64)
    for train_idx, test_idx in GroupKFold(n_splits=_N_SPLITS).split(x, y, groups):
        model = LGBMClassifier(**_LGB_PARAMS)
        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof, y)
    xs = tuple(float(v) for v in iso.X_thresholds_)
    ys = tuple(float(v) for v in iso.y_thresholds_)
    n_pos = int(y.sum())
    calibrator = IsotonicCalibrator(
        xs=xs, ys=ys, trained_at=datetime.now(tz=UTC).isoformat(),
        n_positive=n_pos, n_negative=len(labels) - n_pos,
    )
    save_learned_calibrator(calibrator, _OUT_PATH)

    today = datetime.now(UTC).date().isoformat()
    print(f"# Learned isotonic calibrator fit — {today}\n")
    print(f"- Pairs: {len(rows)} ({n_pos} match / {len(labels) - n_pos} no_match)")
    print(f"- Breakpoints: {len(xs)}")
    print(f"- Persisted to: `{_OUT_PATH}`\n")
    print("| raw prob | calibrated |")
    print("|---:|---:|")
    for raw in (0.5, 0.8, 0.9, 0.95, 0.99, 0.998, 1.0):
        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - raw))
        print(f"| {raw} | {ys[idx]:.4f} |")


if __name__ == "__main__":
    main()
