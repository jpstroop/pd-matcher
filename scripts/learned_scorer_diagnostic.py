"""LightGBM diagnostic study over the labeled vault.

Throwaway one-off measurement script. NOT shipped; ``scripts/`` is gitignored
from the published package via the ``[tool.coverage.run].source`` allowlist.

Trains a small, heavily-regularized LightGBM model with 5-fold stratified
cross-validation against the existing labeled vault, then prints a markdown
report covering:

1. Experimental setup (corpus size, hyperparameters, fold metrics).
2. Feature importance ranking (LightGBM ``gain`` averaged across folds), with
   the production combiner weights printed alongside for direct comparison.
3. Per-feature SHAP-style contribution statistics across all out-of-fold
   predictions (mean abs, std, direction).
4. Top-30 disagreement pairs: ``|lgbm_pred - combined_score|`` desc with
   provenance for spot-checking.
5. Learning curve over stratified subsamples — is the corpus large enough?
6. Head-to-head: LightGBM OOF vs the weighted-mean combiner on the same pairs.
7. Category-sliced accuracy and per-direction disagreement tables.
8. Decision gate for issue #4 adoption.

The script is one-shot research; it intentionally does not deploy or update
the production combiner, calibrator, or matching configuration. It is the
prelude to issue #4 (learned scorer) and is meant to be re-run at each ~500
label increment.

Usage:
    pdm run python scripts/learned_scorer_diagnostic.py \\
        > docs/findings/learned_scorer_diagnostic_2026-06-12.md
"""

from __future__ import annotations

from dataclasses import dataclass
from json import dumps as json_dumps
from pathlib import Path
from statistics import mean
from statistics import pstdev
from sys import stderr
from typing import Final

from lightgbm import Booster
from lightgbm import LGBMClassifier
from numpy import abs as np_abs
from numpy import argsort
from numpy import asarray
from numpy import float64
from numpy import int64
from numpy import ndarray
from numpy import unique
from numpy import zeros
from numpy.random import default_rng
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pd_groundtruth.label_vault import CategoryKey
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.eval.feature_matrix import FeatureMatrixRow
from pd_matcher.eval.feature_matrix import extract_feature_matrix

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")

_RANDOM_STATE: Final[int] = 20260529
_N_SPLITS: Final[int] = 5
_TOP_DISAGREEMENT: Final[int] = 30

_LEARNING_CURVE_SIZES: Final[tuple[int, ...]] = (500, 750, 1000, 1250)
_THRESHOLD_STEP: Final[float] = 0.05
_DISAGREEMENT_CAP: Final[int] = 25
_PROGRESS_LOG: Final[Path] = Path("/tmp/agent-progress.log")
_PRIOR_FOLD_AUC_STD: Final[float] = 0.0026
_ADOPTION_F1_BAR: Final[float] = 0.02
_CATEGORY_KEYS: Final[tuple[CategoryKey, ...]] = (
    "marc_whole_cce_part",
    "cce_whole_marc_part",
    "translation",
    "different_edition",
    "ocr_confusion",
    "same_title_different_work",
    "generic_title",
)


def _progress(message: str) -> None:
    """Emit a one-line milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")

_LGBM_PARAMS: Final[dict[str, object]] = {
    "max_depth": 3,
    "num_leaves": 8,
    "min_data_in_leaf": 10,
    "lambda_l2": 1.0,
    "n_estimators": 100,
    "class_weight": "balanced",
    "objective": "binary",
    "verbose": -1,
    "random_state": _RANDOM_STATE,
}

_CURRENT_WEIGHTS: Final[dict[str, float]] = {
    "title.token_set": 0.35,
    "name.author": 0.20,
    "name.publisher": 0.10,
    "year.delta": 0.10,
    "edition.compat": 0.05,
    "lccn.exact": 0.10,
    "isbn.exact": 0.00,
    "extent.page_count": 0.05,
    "volume.compat": 0.05,
}


def _fit_classifier(
    x_train: NDArray[float64],
    y_train: NDArray[float64],
) -> LGBMClassifier:
    """Fit one LightGBM classifier with the diagnostic hyperparameters."""
    model = LGBMClassifier(**_LGBM_PARAMS)
    model.fit(x_train, y_train)
    return model


def _shap_contributions(model: LGBMClassifier, x: NDArray[float64]) -> ndarray:
    """Return per-row, per-feature SHAP contributions (excluding the bias column).

    ``predict(pred_contrib=True)`` returns one extra column for the model's
    expected value; dropping it leaves a ``(n_rows, n_features)`` matrix of
    raw per-feature contributions, which is what the aggregation needs.
    """
    booster: Booster = model.booster_
    contrib = booster.predict(x, pred_contrib=True)
    contrib_array = asarray(contrib, dtype=float64)
    return contrib_array[:, :-1]


def _cross_validate(
    x: NDArray[float64],
    y: NDArray[float64],
    feature_names: list[str],
    *,
    quiet: bool = False,
) -> tuple[
    NDArray[float64],
    ndarray,
    NDArray[float64],
    list[float],
    list[float],
]:
    """Run stratified 5-fold CV; return OOF preds, SHAP, importance, fold AUCs.

    Args:
        x: Feature matrix, shape ``(n, k)``.
        y: Binary labels, shape ``(n,)``.
        feature_names: Column order, length ``k``.
        quiet: When ``True``, suppress the per-fold ``<!-- fold ... -->``
            stdout breadcrumb (used by the learning-curve sweep so the
            markdown report stays clean).

    Returns:
        Tuple of:
            * Out-of-fold predicted probabilities, shape ``(n,)``.
            * Per-row per-feature SHAP contributions, shape
              ``(n, n_features)``.
            * Per-feature average ``gain`` importance across folds,
              shape ``(n_features,)``.
            * Per-fold ROC-AUC scores.
            * Per-fold average-precision (PR-AUC) scores.
    """
    n_rows = x.shape[0]
    n_features = len(feature_names)
    oof_predictions: NDArray[float64] = zeros((n_rows,), dtype=float64)
    shap_contributions: ndarray = zeros((n_rows, n_features), dtype=float64)
    importance_accumulator: NDArray[float64] = zeros((n_features,), dtype=float64)
    fold_roc_auc: list[float] = []
    fold_pr_auc: list[float] = []
    splitter = StratifiedKFold(
        n_splits=_N_SPLITS,
        shuffle=True,
        random_state=_RANDOM_STATE,
    )
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        model = _fit_classifier(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        oof_predictions[test_idx] = probabilities
        shap_contributions[test_idx] = _shap_contributions(model, x_test)
        fold_importance = asarray(
            model.booster_.feature_importance(importance_type="gain"),
            dtype=float64,
        )
        importance_accumulator += fold_importance
        fold_roc_auc.append(float(roc_auc_score(y_test, probabilities)))
        fold_pr_auc.append(float(average_precision_score(y_test, probabilities)))
        if not quiet:
            print(
                f"<!-- fold {fold_index}: "
                f"roc_auc={fold_roc_auc[-1]:.4f} pr_auc={fold_pr_auc[-1]:.4f} -->"
            )
    importance_average = importance_accumulator / float(_N_SPLITS)
    return oof_predictions, shap_contributions, importance_average, fold_roc_auc, fold_pr_auc


def _print_experimental_setup(
    rows: tuple[FeatureMatrixRow, ...],
    fold_roc_auc: list[float],
    fold_pr_auc: list[float],
) -> None:
    """Section 1: experimental setup + fold-level metrics."""
    positives = sum(1 for row in rows if row.verdict == "match")
    negatives = sum(1 for row in rows if row.verdict == "no_match")
    print("# Learned-scorer diagnostic — 2026-06-12\n")
    print("## 1. Experimental setup\n")
    print(f"- **Pairs scored**: {len(rows)} ({positives} match / {negatives} no_match)")
    print(f"- **Cross-validation**: {_N_SPLITS}-fold stratified, random_state={_RANDOM_STATE}")
    print("- **Model**: LightGBM binary classifier, hyperparameters:")
    for key, value in _LGBM_PARAMS.items():
        print(f"    - `{key}`: `{value}`")
    print()
    print(
        f"- **ROC-AUC** across folds: mean={mean(fold_roc_auc):.4f} std={pstdev(fold_roc_auc):.4f}"
    )
    print(
        f"- **PR-AUC** across folds: mean={mean(fold_pr_auc):.4f} std={pstdev(fold_pr_auc):.4f}\n"
    )
    print(
        "> The negative class is the binding constraint at this corpus size; "
        "fold-level numbers are directional, not deployable. "
        "Recompute at ~1500 labels.\n"
    )


def _print_feature_importance(
    feature_names: list[str],
    importance_average: NDArray[float64],
) -> None:
    """Section 2: feature importance ranking + comparison to current weights."""
    print("## 2. Feature importance ranking\n")
    print(
        "LightGBM `gain` importance averaged across the five folds, normalized "
        "to sum to 1.0 across all features. The `current_weight` column shows "
        "the matching.yaml weight for that scorer (skipped-flag features have "
        "no direct weight analogue and are marked `--`).\n"
    )
    total = float(importance_average.sum())
    if total <= 0.0:
        total = 1.0
    normalized = importance_average / total
    order = list(argsort(-normalized))
    print("| rank | feature | lgbm_importance | current_weight |")
    print("|---:|:---|---:|---:|")
    for rank_index, feature_index in enumerate(order, start=1):
        feature = feature_names[feature_index]
        weight = _CURRENT_WEIGHTS.get(feature)
        weight_str = f"{weight:.3f}" if weight is not None else "--"
        print(f"| {rank_index} | `{feature}` | {normalized[feature_index]:.4f} | {weight_str} |")
    print()


def _shap_direction(values: ndarray) -> str:
    """Classify a feature's SHAP contributions as positive, negative, or bidirectional."""
    positive = float((values > 0.0).sum())
    negative = float((values < 0.0).sum())
    if positive == 0.0 and negative == 0.0:
        return "inert"
    ratio = positive / max(positive + negative, 1.0)
    if ratio >= 0.85:
        return "positive"
    if ratio <= 0.15:
        return "negative"
    return "bidirectional"


def _print_shap_distributions(
    feature_names: list[str],
    shap_contributions: ndarray,
) -> None:
    """Section 3: per-feature SHAP distribution stats."""
    print("## 3. Per-feature SHAP contribution distributions\n")
    print(
        "Mean absolute contribution, standard deviation, and a coarse direction "
        "label across all out-of-fold predictions. High `std` relative to "
        "`mean_abs` signals interaction effects: the feature pushes predictions "
        "in different directions in different contexts.\n"
    )
    mean_abs = asarray(np_abs(shap_contributions).mean(axis=0), dtype=float64)
    std_values = asarray(shap_contributions.std(axis=0), dtype=float64)
    order = list(argsort(-mean_abs))
    print("| feature | mean_abs | std | direction |")
    print("|:---|---:|---:|:---|")
    for feature_index in order:
        feature = feature_names[feature_index]
        direction = _shap_direction(shap_contributions[:, feature_index])
        print(
            f"| `{feature}` | {mean_abs[feature_index]:.4f} | "
            f"{std_values[feature_index]:.4f} | {direction} |"
        )
    print()


def _print_disagreement_table(
    rows: tuple[FeatureMatrixRow, ...],
    oof_predictions: NDArray[float64],
) -> None:
    """Section 4: top-K disagreement pairs."""
    print(f"## 4. Top-{_TOP_DISAGREEMENT} disagreement pairs\n")
    print(
        "Pairs sorted by `|lgbm_pred - combined_score|` (descending). "
        "`verdict` is the human label. Use this as the worktable for the next "
        "round of weight inspection.\n"
    )
    combined = asarray([row.combined_score for row in rows], dtype=float64)
    deltas = asarray(np_abs(oof_predictions - combined), dtype=float64)
    order = list(argsort(-deltas))[:_TOP_DISAGREEMENT]
    print(
        "| rank | pair_id | marc_control_id | nypl_uuid | verdict | "
        "combined | lgbm | |delta| | marc_title | cce_title |"
    )
    print("|---:|---:|:---|:---|:---|---:|---:|---:|:---|:---|")
    for rank_index, row_index in enumerate(order, start=1):
        row = rows[row_index]
        marc_title = _truncate(row.marc_title, 60)
        cce_title = _truncate(row.cce_title, 60)
        print(
            f"| {rank_index} | {row.pair_id} | `{row.marc_control_id}` | "
            f"`{row.nypl_uuid}` | {row.verdict} | "
            f"{row.combined_score:.3f} | {oof_predictions[row_index]:.3f} | "
            f"{deltas[row_index]:.3f} | {marc_title} | {cce_title} |"
        )
    print()


def _truncate(text: str, max_length: int) -> str:
    """Truncate ``text`` for the markdown table, escaping pipe characters."""
    cleaned = text.replace("|", "\\|").replace("\n", " ").strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1] + "…"


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Best-F1 decision threshold for a probability/score vector."""

    best_threshold: float
    best_f1: float


def _best_threshold(y: NDArray[int64], scores: NDArray[float64]) -> ThresholdResult:
    """Sweep thresholds in :data:`_THRESHOLD_STEP` steps; return the best-F1 point.

    A pair is predicted positive when ``score >= threshold``. The sweep spans
    ``[0.0, 1.0]`` inclusive so a degenerate all-positive or all-negative
    decision is reachable. Ties on F1 keep the lowest threshold (the first
    seen), which is the more permissive operating point.
    """
    best_f1 = -1.0
    best_threshold = 0.0
    steps = int(round(1.0 / _THRESHOLD_STEP)) + 1
    for step in range(steps):
        threshold = step * _THRESHOLD_STEP
        predictions = (scores >= threshold).astype(int64)
        score = float(f1_score(y, predictions, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return ThresholdResult(best_threshold=best_threshold, best_f1=best_f1)


def _stratified_subsample(
    y: NDArray[int64],
    target_n: int,
) -> NDArray[int64]:
    """Return row indices for a label-stratified subsample of size ~``target_n``.

    Each class contributes a share proportional to its prevalence in ``y``.
    Selection is deterministic: indices within a class are drawn in a fixed
    pseudo-random order seeded by :data:`_RANDOM_STATE` so the curve is
    reproducible across runs. The realized size may differ from ``target_n``
    by a couple of rows due to per-class rounding.
    """
    rng_order = _seeded_permutation(y.shape[0])
    selected: list[int] = []
    total = y.shape[0]
    for label in unique(y):
        class_positions = [int(i) for i in rng_order if int(y[i]) == int(label)]
        class_total = len(class_positions)
        take = int(round(target_n * class_total / total))
        take = max(1, min(take, class_total))
        selected.extend(class_positions[:take])
    selected.sort()
    return asarray(selected, dtype=int64)


def _seeded_permutation(n: int) -> list[int]:
    """Return a deterministic permutation of ``range(n)`` seeded reproducibly."""
    rng = default_rng(_RANDOM_STATE)
    permuted = rng.permutation(n)
    return [int(value) for value in permuted]


@dataclass(frozen=True, slots=True)
class LearningCurvePoint:
    """One ``n`` on the learning curve: realized size and CV metrics."""

    requested_n: int
    realized_n: int
    mean_auc: float
    std_auc: float
    mean_pr_auc: float


def _print_learning_curve(
    x: NDArray[float64],
    y: NDArray[int64],
    feature_names: list[str],
) -> tuple[float, bool]:
    """Section 5: learning curve over stratified subsamples.

    Returns ``(full_n_fold_std, plateau)``: the AUC fold-std at full ``n`` (so
    the Decision section can gate it against the prior run's fold variance) and
    whether the curve has plateaued (last-two-point AUC delta within one
    fold-std of the larger point).
    """
    print("## 5. Learning curve\n")
    print(
        "Stratified-by-label subsamples drawn with a fixed seed "
        f"(`random_state={_RANDOM_STATE}`); the existing {_N_SPLITS}-fold CV "
        "and hyperparameters run at each size. `mean_auc` is the mean "
        "out-of-fold ROC-AUC across folds, `std` its fold standard deviation, "
        "`pr_auc` the mean fold PR-AUC.\n"
    )
    y_float = y.astype(float64)
    points: list[LearningCurvePoint] = []
    requested_sizes = list(_LEARNING_CURVE_SIZES) + [x.shape[0]]
    for requested_n in requested_sizes:
        if requested_n >= x.shape[0]:
            indices = asarray(range(x.shape[0]), dtype=int64)
        else:
            indices = _stratified_subsample(y, requested_n)
        _, _, _, fold_roc_auc, fold_pr_auc = _cross_validate(
            x[indices],
            y_float[indices],
            feature_names,
            quiet=True,
        )
        point = LearningCurvePoint(
            requested_n=requested_n,
            realized_n=len(indices),
            mean_auc=mean(fold_roc_auc),
            std_auc=pstdev(fold_roc_auc),
            mean_pr_auc=mean(fold_pr_auc),
        )
        points.append(point)
        _progress(
            f"learning curve n={point.realized_n} "
            f"auc={point.mean_auc:.4f} std={point.std_auc:.4f}"
        )
    print("| requested_n | realized_n | mean_auc | std | pr_auc |")
    print("|---:|---:|---:|---:|---:|")
    for point in points:
        label = "all" if point.requested_n >= x.shape[0] else str(point.requested_n)
        print(
            f"| {label} | {point.realized_n} | {point.mean_auc:.4f} | "
            f"{point.std_auc:.4f} | {point.mean_pr_auc:.4f} |"
        )
    print()
    last = points[-1]
    penultimate = points[-2]
    delta = abs(last.mean_auc - penultimate.mean_auc)
    tolerance = max(last.std_auc, penultimate.std_auc)
    plateau = delta <= tolerance
    verdict = "PLATEAU" if plateau else "STILL CLIMBING"
    print(
        f"**Verdict: {verdict}** — AUC delta between the last two points is "
        f"{delta:.4f}; one fold-std of the larger point is {tolerance:.4f}.\n"
    )
    return last.std_auc, plateau


@dataclass(frozen=True, slots=True)
class CombinerComparison:
    """AUC/PR-AUC/best-F1 for the weighted mean and the LightGBM OOF model."""

    weighted_auc: float
    weighted_pr_auc: float
    weighted_threshold: ThresholdResult
    lgbm_auc: float
    lgbm_pr_auc: float
    lgbm_threshold: ThresholdResult


def _print_head_to_head(
    rows: tuple[FeatureMatrixRow, ...],
    y: NDArray[int64],
    oof_predictions: NDArray[float64],
) -> CombinerComparison:
    """Section 6: LightGBM OOF vs the weighted-mean combiner on the same pairs."""
    print("## 6. Head-to-head: LightGBM OOF vs weighted-mean combiner\n")
    print(
        "Identical pair set, identical Evidence. The weighted-mean score is "
        "the production combiner's calibrated output (`combined_score`, "
        "deterministic — no CV); the LightGBM column is the out-of-fold "
        f"probability from the {_N_SPLITS}-fold CV, so every pair is scored by "
        "a model that never saw it.\n"
    )
    weighted = asarray([row.combined_score for row in rows], dtype=float64)
    weighted_auc = float(roc_auc_score(y, weighted))
    weighted_pr = float(average_precision_score(y, weighted))
    weighted_threshold = _best_threshold(y, weighted)
    lgbm_auc = float(roc_auc_score(y, oof_predictions))
    lgbm_pr = float(average_precision_score(y, oof_predictions))
    lgbm_threshold = _best_threshold(y, oof_predictions)
    print("| scorer | AUC | PR-AUC | best_F1 | at_threshold |")
    print("|:---|---:|---:|---:|---:|")
    print(
        f"| weighted-mean | {weighted_auc:.4f} | {weighted_pr:.4f} | "
        f"{weighted_threshold.best_f1:.4f} | {weighted_threshold.best_threshold:.2f} |"
    )
    print(
        f"| LightGBM OOF | {lgbm_auc:.4f} | {lgbm_pr:.4f} | "
        f"{lgbm_threshold.best_f1:.4f} | {lgbm_threshold.best_threshold:.2f} |"
    )
    print()
    f1_delta = lgbm_threshold.best_f1 - weighted_threshold.best_f1
    print(
        f"**Best-F1 delta (LightGBM − weighted-mean): {f1_delta:+.4f}.** "
        f"Issue #4's informal adoption bar is ~{_ADOPTION_F1_BAR:.2f} F1 points. "
        "Note: the authoritative bar is top-1 linkage F1 on the regression "
        "eval (`pass-B`), which requires wiring the learned combiner into the "
        "matching pipeline — a separate, next-phase task. The OOF best-F1 here "
        "is a per-pair classification proxy, not the linkage metric.\n"
    )
    return CombinerComparison(
        weighted_auc=weighted_auc,
        weighted_pr_auc=weighted_pr,
        weighted_threshold=weighted_threshold,
        lgbm_auc=lgbm_auc,
        lgbm_pr_auc=lgbm_pr,
        lgbm_threshold=lgbm_threshold,
    )


def _entries_by_pair(vault_path: Path) -> dict[tuple[str, str], VaultEntry]:
    """Return the vault keyed by ``(marc_control_id, nypl_uuid)``."""
    return current_entries(vault_path)


def _print_category_method() -> None:
    """Section 7 preamble: why categories are not model features."""
    print("## 7. Category-sliced analysis\n")
    print(
        "Categories are deliberately not model inputs. They are assigned by "
        "the labeler at verdict time and do not exist for unlabeled pairs, so "
        "a combiner trained on them could never run at inference; worse, they "
        "encode the verdict itself (e.g. `marc_whole_cce_part` is 85% "
        "`no_match`), so training on them is label leakage. They are used here "
        "only to slice evaluation results.\n"
    )


def _print_category_slices(
    rows: tuple[FeatureMatrixRow, ...],
    y: NDArray[int64],
    oof_predictions: NDArray[float64],
    comparison: CombinerComparison,
    entries: dict[tuple[str, str], VaultEntry],
) -> None:
    """Section 7a: per-category accuracy of both scorers at their best threshold."""
    weighted = asarray([row.combined_score for row in rows], dtype=float64)
    weighted_threshold = comparison.weighted_threshold.best_threshold
    lgbm_threshold = comparison.lgbm_threshold.best_threshold
    weighted_correct = ((weighted >= weighted_threshold).astype(int64) == y)
    lgbm_correct = ((oof_predictions >= lgbm_threshold).astype(int64) == y)

    tagged_total = 0
    print(
        "Per-category accuracy at each scorer's own best-F1 threshold "
        f"(weighted-mean @ {weighted_threshold:.2f}, "
        f"LightGBM @ {lgbm_threshold:.2f}). `unsure` verdicts are already "
        "excluded by the feature matrix; only `match`/`no_match` pairs appear.\n"
    )
    print("| category | n | weighted_acc | lgbm_acc | delta |")
    print("|:---|---:|---:|---:|---:|")
    for key in _CATEGORY_KEYS:
        indices = [
            i
            for i, row in enumerate(rows)
            if key in _categories_for_row(row, entries)
        ]
        if not indices:
            print(f"| `{key}` | 0 | -- | -- | -- |")
            continue
        tagged_total += len(indices)
        idx = asarray(indices, dtype=int64)
        weighted_acc = float(weighted_correct[idx].mean())
        lgbm_acc = float(lgbm_correct[idx].mean())
        delta = lgbm_acc - weighted_acc
        print(
            f"| `{key}` | {len(indices)} | {weighted_acc:.3f} | "
            f"{lgbm_acc:.3f} | {delta:+.3f} |"
        )
    print()
    print(
        f"> {tagged_total} category-tag occurrences fall on scored "
        "(non-`unsure`) pairs; a pair tagged with multiple categories is "
        "counted under each. `unsure`-verdict pairs carrying tags are dropped "
        "by the feature matrix and absent above.\n"
    )


def _categories_for_row(
    row: FeatureMatrixRow,
    entries: dict[tuple[str, str], VaultEntry],
) -> tuple[CategoryKey, ...]:
    """Return the vault categories for a feature-matrix row, or ``()`` if absent."""
    entry = entries.get((row.marc_control_id, row.nypl_uuid))
    if entry is None:
        return ()
    return entry.categories


def _print_disagreement_directions(
    rows: tuple[FeatureMatrixRow, ...],
    y: NDArray[int64],
    oof_predictions: NDArray[float64],
    comparison: CombinerComparison,
    entries: dict[tuple[str, str], VaultEntry],
) -> None:
    """Section 7b: pairs one scorer gets right and the other wrong."""
    weighted = asarray([row.combined_score for row in rows], dtype=float64)
    weighted_threshold = comparison.weighted_threshold.best_threshold
    lgbm_threshold = comparison.lgbm_threshold.best_threshold
    weighted_pred = (weighted >= weighted_threshold).astype(int64)
    lgbm_pred = (oof_predictions >= lgbm_threshold).astype(int64)
    weighted_correct = weighted_pred == y
    lgbm_correct = lgbm_pred == y

    lgbm_wins = [
        i for i in range(len(rows)) if lgbm_correct[i] and not weighted_correct[i]
    ]
    weighted_wins = [
        i for i in range(len(rows)) if weighted_correct[i] and not lgbm_correct[i]
    ]
    _print_direction_table(
        "LightGBM right, weighted-mean wrong",
        lgbm_wins,
        rows,
        weighted,
        oof_predictions,
        entries,
    )
    _print_direction_table(
        "Weighted-mean right, LightGBM wrong",
        weighted_wins,
        rows,
        weighted,
        oof_predictions,
        entries,
    )
    _dump_disagreements_sidecar(
        lgbm_wins,
        weighted_wins,
        rows,
        weighted,
        oof_predictions,
        entries,
    )


def _dump_disagreements_sidecar(
    lgbm_wins: list[int],
    weighted_wins: list[int],
    rows: tuple[FeatureMatrixRow, ...],
    weighted: NDArray[float64],
    oof_predictions: NDArray[float64],
    entries: dict[tuple[str, str], VaultEntry],
) -> None:
    """Write every disagreement pair (uncapped) as JSONL to a sidecar file.

    Machine-readable companion to the capped markdown tables, so follow-up
    analysis (pair-by-pair review, link generation) does not require a
    re-run. Path is fixed; the file is overwritten on every run.
    """
    sidecar = Path("/tmp/learned_scorer_disagreements.jsonl")
    with sidecar.open("w", encoding="utf-8") as handle:
        for direction, indices in (
            ("lgbm_right_weighted_wrong", lgbm_wins),
            ("weighted_right_lgbm_wrong", weighted_wins),
        ):
            for row_index in indices:
                row = rows[row_index]
                entry = entries.get((row.marc_control_id, row.nypl_uuid))
                record = {
                    "direction": direction,
                    "marc_control_id": row.marc_control_id,
                    "nypl_uuid": row.nypl_uuid,
                    "truth": row.verdict,
                    "weighted": round(float(weighted[row_index]), 4),
                    "lgbm_oof": round(float(oof_predictions[row_index]), 4),
                    "categories": list(entry.categories) if entry else [],
                    "note": entry.note if entry else None,
                }
                handle.write(json_dumps(record) + "\n")
    print(
        f"_Full uncapped disagreement dump: `{sidecar}` "
        f"({len(lgbm_wins) + len(weighted_wins)} pairs)._\n"
    )


def _print_direction_table(
    heading: str,
    indices: list[int],
    rows: tuple[FeatureMatrixRow, ...],
    weighted: NDArray[float64],
    oof_predictions: NDArray[float64],
    entries: dict[tuple[str, str], VaultEntry],
) -> None:
    """Render one capped disagreement-direction table."""
    print(f"### {heading} ({len(indices)} pairs, showing up to {_DISAGREEMENT_CAP})\n")
    if not indices:
        print("_None._\n")
        return
    print(
        "| marc_control_id | truth | weighted | lgbm | categories | note? |"
    )
    print("|:---|:---|---:|---:|:---|:---|")
    for row_index in indices[:_DISAGREEMENT_CAP]:
        row = rows[row_index]
        entry = entries.get((row.marc_control_id, row.nypl_uuid))
        categories = ", ".join(entry.categories) if entry and entry.categories else "--"
        has_note = "yes" if entry is not None and entry.note else "no"
        print(
            f"| `{row.marc_control_id}` | {row.verdict} | "
            f"{weighted[row_index]:.3f} | {oof_predictions[row_index]:.3f} | "
            f"{categories} | {has_note} |"
        )
    print()


def _print_decision(
    learning_curve_plateau: bool,
    comparison: CombinerComparison,
    full_n_std: float,
    vault_label_count: int,
) -> None:
    """Section 8: the programmatic adoption gate for issue #4."""
    print("## 8. Decision\n")
    f1_delta = comparison.lgbm_threshold.best_f1 - comparison.weighted_threshold.best_f1
    f1_beats = f1_delta >= _ADOPTION_F1_BAR
    variance_ok = full_n_std <= _PRIOR_FOLD_AUC_STD
    proceed = (learning_curve_plateau or f1_beats) and variance_ok
    print("Gate inputs:\n")
    print(
        f"- Learning curve: **{'PLATEAU' if learning_curve_plateau else 'STILL CLIMBING'}**"
    )
    print(
        f"- OOF best-F1 beats weighted-mean by ≥ {_ADOPTION_F1_BAR:.2f}: "
        f"**{f1_beats}** (delta {f1_delta:+.4f})"
    )
    print(
        f"- Full-n fold AUC std ≤ {_PRIOR_FOLD_AUC_STD:.4f} (2026-05-31 run): "
        f"**{variance_ok}** (std {full_n_std:.4f})\n"
    )
    if proceed:
        print(
            "**PROCEED.** The gate is met: the corpus has plateaued (or the "
            "learned model clears the F1 bar) and fold variance is no worse "
            "than the prior run. The next phase is wiring a learned combiner "
            "into the matching pipeline and measuring top-1 linkage F1 on the "
            "regression eval.\n"
        )
    else:
        retrigger = _next_increment(vault_label_count)
        print(
            f"**HOLD.** The gate is not met. Re-run this diagnostic at the next "
            f"500-label increment (~{retrigger} labels) and re-evaluate.\n"
        )


def _next_increment(current: int) -> int:
    """Round ``current`` up to the next 500-label vault increment."""
    return ((current // 500) + 1) * 500


def main() -> None:
    """Run the diagnostic study and print the markdown report to stdout."""
    matching_config = _load_default_matching_config()
    pairing_config = _load_default_pairing_config()
    _progress("script extended")
    x, y, feature_names, rows = extract_feature_matrix(
        vault_path=_VAULT_PATH,
        pool_path=_POOL_PATH,
        index_path=_INDEX_PATH,
        matching_config=matching_config,
        pairing_config=pairing_config,
    )
    if x.shape[0] == 0:
        print("# Learned-scorer diagnostic\n")
        print("No labeled pairs available after vault/pool/index resolution; aborting.")
        return
    _progress("pair scoring done")
    vault_label_count = len(_entries_by_pair(_VAULT_PATH))
    entries = _entries_by_pair(_VAULT_PATH)
    y_float = y.astype(float64)
    (
        oof_predictions,
        shap_contributions,
        importance_average,
        fold_roc_auc,
        fold_pr_auc,
    ) = _cross_validate(x, y_float, feature_names)
    _print_experimental_setup(rows, fold_roc_auc, fold_pr_auc)
    _print_feature_importance(feature_names, importance_average)
    _print_shap_distributions(feature_names, shap_contributions)
    _print_disagreement_table(rows, oof_predictions)
    full_n_std, learning_curve_plateau = _print_learning_curve(x, y, feature_names)
    _progress("learning curve done")
    comparison = _print_head_to_head(rows, y, oof_predictions)
    _print_category_method()
    _print_category_slices(rows, y, oof_predictions, comparison, entries)
    _print_disagreement_directions(rows, y, oof_predictions, comparison, entries)
    _print_decision(learning_curve_plateau, comparison, full_n_std, vault_label_count)
    _progress("report written")


if __name__ == "__main__":
    main()
