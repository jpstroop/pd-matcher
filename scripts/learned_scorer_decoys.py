"""Decoy-negatives experiment for issue #77 (the #4 original design), round 2.

Throwaway one-off measurement script. NOT shipped; ``scripts/`` is gitignored
from the published package via the ``[tool.coverage.run].source`` allowlist.

The production A/B (``docs/findings/learned_scorer_ab_2026-06-12.md``) showed
the learned combiner FAILS the top-1 linkage gate (911 vs the weighted mean's
915 correct top picks) despite memorizing the labeled pairs. Root cause: a
train/inference distribution mismatch — the model trains only on labeled pairs
(the matcher's reviewed top picks) but at inference must rank the true match
above thousands of unlabeled same-year, token-sharing decoys it has never seen.

Round 1 (``docs/findings/learned_scorer_decoys_2026-06-12.md``) augmented the
training set with ``MatchResult.alternates``: runners-up that clear the floor.
Under ``year_window: 0`` only 207/922 MARCs had any above-floor runner-up, so
the harvest was ~10x thinner than the ticket anticipated (267 decoys), lifting
top-1 from 911 to 913 — IMPROVED-BUT-SHORT vs the weighted mean's 915 — while
nudging the grouped-OOF labeled-rows slice down (−0.0076 best-F1).

Round 2 pulls the two levers the round-1 findings named:

1. **Below-floor harvest.** For each labeled-MATCH MARC, ``match_record`` is
   run with ``min_combined_score`` forced to 0.0 and a large ``top_k``, so
   ``best + alternates`` is the FULL ranked candidate set (no floor cull). The
   top ``_DECOY_TOP_K`` non-true candidates by combined score become decoys,
   yielding decoy mass across virtually all 922 MARCs rather than 207. The
   existing exclusion is kept: any candidate whose ``nypl_uuid`` equals the
   labeled true match or ANY labeled pair for that MARC is dropped.

2. **Decoy down-weighting sweep.** The augmented model is trained with explicit
   per-row ``sample_weight`` (labeled rows 1.0, decoy rows ``w``) across
   ``w ∈ {0.25, 0.5, 1.0}``. Each config is scored by an OOF RANKING PROXY:
   the fraction of labeled-MATCH MARCs whose true-pair grouped-OOF probability
   is strictly greater than the max grouped-OOF probability over that MARC's
   decoys. This proxies the top-1 gate without the ~18-minute production eval.
   The winning ``w`` is picked by ``rank_proxy``, tie-broken by the
   labeled-rows-slice best-F1 (Gate 2 health); only the winner's Booster is
   persisted.

It builds the baseline (labeled-only) matrix and each augmented matrix through
the *production* feature projection
(:func:`pd_matcher.match.combiners.features.feature_row`), so the persisted
artifact's feature-name contract matches inference exactly. It then reports
GroupKFold-by-``marc_control_id`` OOF AUC / PR-AUC / best-F1 and the rank proxy
for every sweep config, and fits the FINAL Booster on ALL augmented rows under
the winning ``w`` with the locked hyperparameters, persisting it via the
production :func:`pd_matcher.match.combiners.learned.save_learned_model`,
OVERWRITING ``caches/learned_scorer.{txt,msgpack}`` (gitignored, re-derivable
via ``pdm run pd-matcher train-scorer``).

It does NOT modify anything under ``src/``. It writes nothing under ``data/``.
The vault is read-only.

Usage:
    pdm run python scripts/learned_scorer_decoys.py \\
        > docs/findings/learned_scorer_decoys_2026-06-12.md

    # Smoke run capping the labeled-MATCH MARCs processed (report is flagged):
    pdm run python scripts/learned_scorer_decoys.py --limit 20 \\
        > /tmp/decoys_smoke.md
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from sys import stderr
from typing import Final

from lightgbm import LGBMClassifier
from msgspec import structs
from numpy import asarray
from numpy import float64
from numpy import int64
from numpy import ones
from numpy import zeros
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.features import feature_row
from pd_matcher.match.combiners.learned import model_metadata
from pd_matcher.match.combiners.learned import save_learned_model
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_ARTIFACT_DIR: Final[Path] = Path("caches")
_PROGRESS_LOG: Final[Path] = Path("/tmp/agent-progress.log")

_VERDICT_MATCH: Final[str] = "match"
_VERDICT_UNSURE: Final[str] = "unsure"

# Locked recipe from src/pd_matcher/match/combiners/train.py.
_MAX_DEPTH: Final[int] = 3
_NUM_LEAVES: Final[int] = 8
_MIN_DATA_IN_LEAF: Final[int] = 10
_LAMBDA_L2: Final[float] = 1.0
_N_ESTIMATORS: Final[int] = 200
_CLASS_WEIGHT: Final[str] = "balanced"

_RANDOM_STATE: Final[int] = 20260612
_N_SPLITS: Final[int] = 5
_THRESHOLD_STEP: Final[float] = 0.05

# Round 2: harvest the top-k non-true candidates per MARC from the FULL ranked
# set (floor forced to 0), and sweep the decoy sample-weight.
_DECOY_TOP_K: Final[int] = 5
# top_k passed to match_record must surface k decoys PLUS any labeled candidates
# we exclude PLUS the true match at rank 1, with headroom; the floor is 0 so
# best + alternates is the full ranked set up to this cap.
_HARVEST_TOP_K: Final[int] = _DECOY_TOP_K + 16
_DECOY_FLOOR: Final[float] = 0.0
_DECOY_WEIGHT_SWEEP: Final[tuple[float, ...]] = (0.25, 0.5, 1.0)

_WEIGHTED_F1_REFERENCE: Final[float] = 0.99349
_WEIGHTED_CORRECT_TOP: Final[int] = 915
_LABELED_ONLY_CORRECT_TOP: Final[int] = 911
_ROUND1_CORRECT_TOP: Final[int] = 913


def _progress(message: str) -> None:
    """Emit a one-line milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _classifier_params() -> dict[str, object]:
    """Return the locked LightGBM hyperparameters as a parameter dict."""
    return {
        "max_depth": _MAX_DEPTH,
        "num_leaves": _NUM_LEAVES,
        "min_data_in_leaf": _MIN_DATA_IN_LEAF,
        "reg_lambda": _LAMBDA_L2,
        "n_estimators": _N_ESTIMATORS,
        "class_weight": _CLASS_WEIGHT,
        "objective": "binary",
        "verbose": -1,
        "random_state": _RANDOM_STATE,
        "n_jobs": 1,
    }


def _scoring_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` forced onto the weighted-mean scorer.

    Mirrors :func:`pd_matcher.match.combiners.train._scoring_config`: the
    per-scorer Evidence is identical regardless of the combiner, and forcing
    weighted-mean avoids needing a learned artifact to produce the very data
    that trains it.
    """
    if config.scorer == "weighted_mean":
        return config
    return MatchingConfig(
        title_weight=config.title_weight,
        author_weight=config.author_weight,
        publisher_weight=config.publisher_weight,
        year_weight=config.year_weight,
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
class HarvestStats:
    """Counts describing the below-floor decoy-harvest pass."""

    marcs_with_match: int
    marcs_matched_in_pool: int
    decoys_harvested: int
    dropped_true_match: int
    dropped_labeled_pair: int
    marcs_yielding_decoys: int
    limit: int | None


@dataclass(frozen=True, slots=True)
class Dataset:
    """Augmented feature matrix with parallel group ids and a labeled mask.

    ``true_row`` and ``decoy_rows_by_group`` index, per MARC group id, the row
    holding that MARC's true (labeled-MATCH) pair and the rows holding its
    decoys; the OOF ranking proxy reads these to ask "did the true pair out
    -rank all of this MARC's decoys?" without re-deriving group membership.
    """

    x: NDArray[float64]
    y: NDArray[int64]
    groups: NDArray[int64]
    is_labeled: NDArray[int64]
    is_decoy: NDArray[int64]
    n_positive_labeled: int
    n_negative_labeled: int
    n_decoy: int
    true_row: dict[int, int]
    decoy_rows_by_group: dict[int, list[int]]
    harvest: HarvestStats


def _partition_entries(entries: dict[tuple[str, str], VaultEntry]) -> list[VaultEntry]:
    """Drop ``unsure`` entries; preserve insertion order for determinism."""
    return [e for e in entries.values() if e.verdict != _VERDICT_UNSURE]


def _labeled_uuids_by_marc(
    entries: dict[tuple[str, str], VaultEntry],
) -> dict[str, set[str]]:
    """Return ``marc_control_id -> {every labeled nypl_uuid}`` (all verdicts)."""
    by_marc: dict[str, set[str]] = {}
    for entry in entries.values():
        by_marc.setdefault(entry.marc_control_id, set()).add(entry.nypl_uuid)
    return by_marc


def _ground_truth_by_marc(entries: dict[tuple[str, str], VaultEntry]) -> dict[str, str]:
    """Return ``marc_control_id -> nypl_uuid`` for current ``match`` verdicts."""
    gt: dict[str, str] = {}
    for entry in entries.values():
        if entry.verdict == _VERDICT_MATCH:
            gt[entry.marc_control_id] = entry.nypl_uuid
    return gt


def _build_dataset(
    entries: dict[tuple[str, str], VaultEntry],
    limit: int | None,
) -> Dataset:
    """Score labeled pairs and harvest below-floor decoy negatives.

    Labeled rows reuse the eval pass-A resolution (``make_pair_scorer``) and
    the canonical :func:`feature_row`. Decoy rows come from running
    ``match_record`` for each labeled-MATCH MARC under a floor of 0.0 and a
    large ``top_k`` — so ``best + alternates`` is the FULL ranked candidate set
    — then taking the top :data:`_DECOY_TOP_K` non-true candidates by combined
    score. Each decoy's Evidence is projected through the same
    :func:`feature_row`. The group id is a per-MARC integer so GroupKFold keeps
    all rows of one MARC in one fold.

    ``limit`` (when not ``None``) caps the labeled-MATCH MARCs harvested for
    smoke runs; the labeled-row matrix is restricted to those same MARCs so the
    baseline and augmented populations stay coherent under the cap.
    """
    matching_config = _load_default_matching_config()
    pairing_config = _load_default_pairing_config()
    scoring_config = _scoring_config(matching_config)
    harvest_config = structs.replace(scoring_config, min_combined_score=_DECOY_FLOOR)
    ground_truth = _ground_truth_by_marc(entries)
    harvest_marc_ids = _limited_marc_ids(ground_truth, limit)
    # Full run: keep ALL non-unsure labeled rows (922 pos + 512 human-labeled
    # neg). Only under --limit do we restrict labeled rows to the harvested
    # match-MARCs, to keep the smoke baseline and augmented populations
    # coherent under the cap. The earlier unconditional restriction silently
    # dropped every labeled negative on a full run (no_match pairs live on
    # MARCs absent from the match-only harvest set).
    all_labeled = _partition_entries(entries)
    if limit is None:
        kept = all_labeled
    else:
        kept = [e for e in all_labeled if e.marc_control_id in harvest_marc_ids]
    labeled_by_marc = _labeled_uuids_by_marc(entries)
    needed_marc_ids = {e.marc_control_id for e in kept} | harvest_marc_ids
    marc_by_id = build_marc_index(_POOL_PATH, needed_marc_ids)
    pairings = compile_pairings(pairing_config)

    group_of: dict[str, int] = {}

    def group_id(marc_control_id: str) -> int:
        if marc_control_id not in group_of:
            group_of[marc_control_id] = len(group_of)
        return group_of[marc_control_id]

    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    groups: list[int] = []
    labeled_flags: list[int] = []
    decoy_flags: list[int] = []
    true_row: dict[int, int] = {}
    decoy_rows_by_group: dict[int, list[int]] = {}
    n_pos = 0
    n_neg = 0
    n_decoy = 0
    dropped_true = 0
    dropped_labeled = 0
    marcs_with_match = len(harvest_marc_ids)
    marcs_matched_in_pool = 0
    marcs_yielding_decoys = 0

    with NyplIndexLookup(_INDEX_PATH) as lookup:
        idf = build_idf_table(lookup)
        score_pair = make_pair_scorer(
            matching_config=scoring_config,
            pairings=pairings,
            idf=idf,
            calibrator=None,
        )
        combiner = build_combiner(scoring_config, learned_model_dir=None)

        for entry in kept:
            marc = marc_by_id.get(entry.marc_control_id)
            if marc is None:
                continue
            cce = lookup.get_registration(entry.nypl_uuid)
            if cce is None:
                continue
            candidate = score_pair(marc, cce)
            row_index = len(rows)
            rows.append(feature_row(candidate.evidence))
            label = 1 if entry.verdict == _VERDICT_MATCH else 0
            labels.append(label)
            gid = group_id(entry.marc_control_id)
            groups.append(gid)
            labeled_flags.append(1)
            decoy_flags.append(0)
            if label == 1:
                n_pos += 1
                if entry.nypl_uuid == ground_truth.get(entry.marc_control_id):
                    true_row[gid] = row_index
            else:
                n_neg += 1

        for marc_id in harvest_marc_ids:
            true_uuid = ground_truth[marc_id]
            marc = marc_by_id.get(marc_id)
            if marc is None:
                continue
            marcs_matched_in_pool += 1
            result = match_record(
                marc,
                lookup=lookup,
                config=harvest_config,
                idf=idf,
                calibrator=None,
                combiner=combiner,
                pairings=pairings,
                top_k=_HARVEST_TOP_K,
            )
            ranked = ([result.best] if result.best is not None else []) + list(result.alternates)
            labeled_uuids = labeled_by_marc.get(marc_id, set())
            gid = group_id(marc_id)
            taken = 0
            for candidate in ranked:
                if taken >= _DECOY_TOP_K:
                    break
                if candidate.nypl_uuid == true_uuid:
                    dropped_true += 1
                    continue
                if candidate.nypl_uuid in labeled_uuids:
                    dropped_labeled += 1
                    continue
                row_index = len(rows)
                rows.append(feature_row(candidate.evidence))
                labels.append(0)
                groups.append(gid)
                labeled_flags.append(0)
                decoy_flags.append(1)
                decoy_rows_by_group.setdefault(gid, []).append(row_index)
                n_decoy += 1
                taken += 1
            if taken > 0:
                marcs_yielding_decoys += 1

    harvest = HarvestStats(
        marcs_with_match=marcs_with_match,
        marcs_matched_in_pool=marcs_matched_in_pool,
        decoys_harvested=n_decoy,
        dropped_true_match=dropped_true,
        dropped_labeled_pair=dropped_labeled,
        marcs_yielding_decoys=marcs_yielding_decoys,
        limit=limit,
    )
    return Dataset(
        x=asarray(rows, dtype=float64),
        y=asarray(labels, dtype=int64),
        groups=asarray(groups, dtype=int64),
        is_labeled=asarray(labeled_flags, dtype=int64),
        is_decoy=asarray(decoy_flags, dtype=int64),
        n_positive_labeled=n_pos,
        n_negative_labeled=n_neg,
        n_decoy=n_decoy,
        true_row=true_row,
        decoy_rows_by_group=decoy_rows_by_group,
        harvest=harvest,
    )


def _limited_marc_ids(ground_truth: dict[str, str], limit: int | None) -> set[str]:
    """Return the labeled-MATCH MARC ids to harvest, capped at ``limit``.

    Insertion order of ``ground_truth`` (built from vault scan order) is the
    deterministic selection order, so a given ``--limit`` always picks the same
    MARCs across runs.
    """
    marc_ids = list(ground_truth)
    if limit is not None:
        marc_ids = marc_ids[:limit]
    return set(marc_ids)


@dataclass(frozen=True, slots=True)
class Metrics:
    """AUC / PR-AUC / best-F1 for one OOF prediction slice."""

    auc: float
    pr_auc: float
    best_f1: float
    best_threshold: float
    n_rows: int
    n_positive: int


def _best_f1(y: NDArray[int64], scores: NDArray[float64]) -> tuple[float, float]:
    """Sweep thresholds in :data:`_THRESHOLD_STEP` steps; return (best_f1, at)."""
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
    return best_f1, best_threshold


def _metrics(y: NDArray[int64], oof: NDArray[float64]) -> Metrics:
    """Compute AUC / PR-AUC / best-F1 for an OOF prediction slice."""
    best_f1, best_threshold = _best_f1(y, oof)
    return Metrics(
        auc=float(roc_auc_score(y, oof)),
        pr_auc=float(average_precision_score(y, oof)),
        best_f1=best_f1,
        best_threshold=best_threshold,
        n_rows=int(y.shape[0]),
        n_positive=int(y.sum()),
    )


def _grouped_oof(
    x: NDArray[float64],
    y: NDArray[int64],
    groups: NDArray[int64],
    sample_weight: NDArray[float64] | None = None,
) -> NDArray[float64]:
    """Return GroupKFold OOF match probabilities (groups never split folds).

    ``sample_weight`` (when supplied) is sliced per fold and passed to
    ``fit``; LightGBM MULTIPLIES it with the ``class_weight="balanced"`` factor,
    so a decoy weight ``w < 1`` down-weights decoys ON TOP of class balancing.
    """
    oof: NDArray[float64] = zeros(x.shape[0], dtype=float64)
    splitter = GroupKFold(n_splits=_N_SPLITS)
    y_float = y.astype(float64)
    params = _classifier_params()
    for train_idx, test_idx in splitter.split(x, y, groups):
        model = LGBMClassifier(**params)
        fold_weight = None if sample_weight is None else sample_weight[train_idx]
        model.fit(x[train_idx], y_float[train_idx], sample_weight=fold_weight)
        probabilities = asarray(model.predict_proba(x[test_idx]), dtype=float64)[:, 1]
        oof[test_idx] = probabilities
    return oof


def _decoy_sample_weight(dataset: Dataset, decoy_weight: float) -> NDArray[float64]:
    """Return per-row sample weights: 1.0 for labeled rows, ``w`` for decoys."""
    weights: NDArray[float64] = ones(dataset.x.shape[0], dtype=float64)
    weights[dataset.is_decoy == 1] = decoy_weight
    return weights


def _rank_proxy(dataset: Dataset, oof: NDArray[float64]) -> tuple[float, int, int]:
    """Fraction of labeled-MATCH MARCs whose true pair out-ranks all its decoys.

    For each MARC group with both a true-pair row and ≥1 decoy row, the proxy
    counts a win when ``oof[true] > max(oof[decoys])`` (strict). Returns
    ``(fraction, wins, evaluable_marcs)``; MARCs with no harvested decoy are
    excluded (the proxy cannot distinguish a top-1 win there).
    """
    wins = 0
    evaluable = 0
    for gid, true_idx in dataset.true_row.items():
        decoy_indices = dataset.decoy_rows_by_group.get(gid)
        if not decoy_indices:
            continue
        evaluable += 1
        true_prob = oof[true_idx]
        max_decoy_prob = max(oof[idx] for idx in decoy_indices)
        if true_prob > max_decoy_prob:
            wins += 1
    fraction = wins / evaluable if evaluable else 0.0
    return fraction, wins, evaluable


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One decoy-weight sweep config: OOF metrics + the rank proxy."""

    decoy_weight: float
    augmented_full: Metrics
    augmented_labeled: Metrics
    rank_proxy: float
    rank_wins: int
    rank_evaluable: int


def _run_sweep(dataset: Dataset, baseline_mask: NDArray[int64]) -> list[SweepResult]:
    """Grouped-OOF + rank proxy for every decoy-weight in the sweep."""
    results: list[SweepResult] = []
    labeled_idx = baseline_mask == 1
    for decoy_weight in _DECOY_WEIGHT_SWEEP:
        weights = _decoy_sample_weight(dataset, decoy_weight)
        oof = _grouped_oof(dataset.x, dataset.y, dataset.groups, weights)
        augmented_full = _metrics(dataset.y, oof)
        augmented_labeled = _metrics(dataset.y[labeled_idx], oof[labeled_idx])
        fraction, wins, evaluable = _rank_proxy(dataset, oof)
        results.append(
            SweepResult(
                decoy_weight=decoy_weight,
                augmented_full=augmented_full,
                augmented_labeled=augmented_labeled,
                rank_proxy=fraction,
                rank_wins=wins,
                rank_evaluable=evaluable,
            )
        )
        _progress(
            f"sweep w={decoy_weight}: rank_proxy={fraction:.4f} "
            f"({wins}/{evaluable}), labeled-F1={augmented_labeled.best_f1:.4f}"
        )
    return results


def _select_winner(results: list[SweepResult]) -> SweepResult:
    """Pick the sweep config with the highest rank proxy; tie-break on F1."""
    return max(results, key=lambda r: (r.rank_proxy, r.augmented_labeled.best_f1))


def _persist_artifact(dataset: Dataset, decoy_weight: float) -> None:
    """Fit the final Booster on ALL augmented rows and overwrite the artifact.

    The winning decoy ``sample_weight`` is applied so the persisted model is the
    exact one the sweep selected.
    """
    model = LGBMClassifier(**_classifier_params())
    model.fit(
        dataset.x,
        dataset.y.astype(float64),
        sample_weight=_decoy_sample_weight(dataset, decoy_weight),
    )
    booster = model.booster_
    n_positive = int((dataset.y == 1).sum())
    n_negative = int((dataset.y == 0).sum())
    meta = model_metadata(
        booster,
        n_positive=n_positive,
        n_negative=n_negative,
        max_depth=_MAX_DEPTH,
        num_leaves=_NUM_LEAVES,
        min_data_in_leaf=_MIN_DATA_IN_LEAF,
        lambda_l2=_LAMBDA_L2,
        n_estimators=_N_ESTIMATORS,
        class_weight=_CLASS_WEIGHT,
    )
    save_learned_model(booster, meta, _ARTIFACT_DIR)


def _print_limit_warning(limit: int | None) -> None:
    """Emit a prominent SMOKE banner when a ``--limit`` was active."""
    if limit is None:
        return
    print(
        f"> ⚠️ **SMOKE RUN — `--limit {limit}` was active.** Only {limit} "
        "labeled-MATCH MARCs were harvested; every count, sweep number, and the "
        "persisted artifact below are from a truncated dataset and are NOT real "
        "findings. Re-run WITHOUT `--limit` for the production report.\n"
    )


def _print_header(dataset: Dataset) -> None:
    """Emit the document title and the Method section."""
    h = dataset.harvest
    n_features = dataset.x.shape[1]
    n_labeled = dataset.n_positive_labeled + dataset.n_negative_labeled
    print("# Learned-scorer decoy-negatives experiment (round 2) — 2026-06-12\n")
    _print_limit_warning(h.limit)
    print(
        "Issue #77, the #4 ORIGINAL design the research rounds dropped: train "
        "with sampled same-year-bucket non-matches as negatives. The production "
        "A/B (`docs/findings/learned_scorer_ab_2026-06-12.md`) showed the "
        "labeled-only learned model FAILS top-1 linkage "
        f"({_LABELED_ONLY_CORRECT_TOP} vs the weighted mean's "
        f"{_WEIGHTED_CORRECT_TOP} correct top picks). Round 1 augmented training "
        "with above-floor `alternates` (only 267 decoys, 207/922 MARCs) and "
        f"reached {_ROUND1_CORRECT_TOP}/922 — IMPROVED-BUT-SHORT. Round 2 pulls "
        "the two levers the round-1 findings named: a BELOW-FLOOR harvest (~10x "
        "the decoy mass) and a decoy-weight sweep selected by an OOF ranking "
        "proxy.\n"
    )
    print("## Method\n")
    print(
        "**Below-floor decoy harvest.** For each labeled-MATCH MARC the "
        "production `match_record` is run under the weighted-mean scorer (eval "
        "pass A / `train-scorer` produce identical per-scorer Evidence — it is "
        "combiner-independent) with `min_combined_score` forced to "
        f"`{_DECOY_FLOOR}` and `top_k={_HARVEST_TOP_K}`, so `best + alternates` "
        "is the FULL ranked candidate set with no floor cull. The top "
        f"`{_DECOY_TOP_K}` non-true candidates by combined score become decoys; "
        "each carries full Evidence at zero extra scoring cost and is projected "
        "through the canonical `feature_row`. Realized counts:\n"
    )
    print(f"- MARCs with a current `match` verdict (harvested): **{h.marcs_with_match}**")
    print(f"- of those resolved in the candidate pool: **{h.marcs_matched_in_pool}**")
    print(f"- MARCs yielding ≥1 decoy: **{h.marcs_yielding_decoys}**")
    print(f"- decoy negatives harvested: **{h.decoys_harvested}**")
    print(
        f"- candidates dropped as the true match: **{h.dropped_true_match}**; "
        f"dropped as another labeled pair for that MARC: "
        f"**{h.dropped_labeled_pair}**\n"
    )
    print(
        "With the floor removed, the full retrieved candidate set is available "
        f"per MARC, so virtually every MARC yields up to {_DECOY_TOP_K} decoys "
        f"(ceiling {h.marcs_matched_in_pool}×{_DECOY_TOP_K}); the realized count "
        f"({h.decoys_harvested}) falls short only where a MARC retrieved fewer "
        "than that many non-true candidates.\n"
    )
    print(
        "**Decoy down-weighting sweep.** The augmented model trains with "
        "explicit per-row `sample_weight` (labeled rows 1.0, decoy rows `w`) "
        f"across `w ∈ {{{', '.join(str(w) for w in _DECOY_WEIGHT_SWEEP)}}}`, "
        "keeping `class_weight=balanced` (LightGBM MULTIPLIES the sample weight "
        "with the class-balance factor, so `w < 1` down-weights decoys on top of "
        "class balancing). Configs are selected by an **OOF ranking proxy**: for "
        "each labeled-MATCH MARC with ≥1 harvested decoy, using grouped-OOF "
        "predictions, does the true pair's probability strictly exceed the max "
        "over that MARC's decoys? `rank_proxy` is the fraction of such MARCs the "
        "true pair wins — a direct, eval-free proxy for the pass-A top-1 gate. "
        "The winner is the highest `rank_proxy`, tie-broken by labeled-rows-slice "
        "best-F1 (Gate 2 health).\n"
    )
    print(
        "**Label-noise caveat.** A harvested decoy is assumed a non-match, but "
        "an unlabeled candidate could be a true duplicate registration. Any "
        "candidate whose `nypl_uuid` equals the labeled true match OR ANY "
        "labeled pair for that MARC is dropped, so known labels never poison the "
        "decoy set; the residual risk is an unlabeled true duplicate, accepted "
        "per the ticket as small. The below-floor harvest enlarges this surface "
        "(low-scoring candidates are less likely true duplicates, but more "
        "numerous), which the sweep's down-weighting partly hedges.\n"
    )
    print(
        "**GroupKFold rationale.** Decoys and the positive from the same MARC "
        "share near-identical features; a random split would leak a MARC's "
        "positive into a fold that also holds its decoys, inflating OOF. "
        "GroupKFold by `marc_control_id` forces every row of one MARC into one "
        "fold, so the OOF numbers — and the rank proxy that reads them — are "
        "honest under the augmented population.\n"
    )
    print(
        f"- **Feature count**: {n_features} (production `feature_names()`; the "
        "persisted artifact's contract matches inference exactly)"
    )
    print(
        f"- **Labeled rows**: {n_labeled} "
        f"({dataset.n_positive_labeled} pos / {dataset.n_negative_labeled} neg); "
        f"**decoy rows**: {dataset.n_decoy}; **total**: {dataset.x.shape[0]}"
    )
    print(
        f"- **Cross-validation**: {_N_SPLITS}-fold GroupKFold by "
        "`marc_control_id`, locked hyperparameters "
        f"(max_depth={_MAX_DEPTH}, num_leaves={_NUM_LEAVES}, "
        f"min_data_in_leaf={_MIN_DATA_IN_LEAF}, lambda_l2={_LAMBDA_L2}, "
        f"n_estimators={_N_ESTIMATORS}, class_weight={_CLASS_WEIGHT}), "
        f"random_state={_RANDOM_STATE}, deterministic (`n_jobs=1`)\n"
    )


def _print_sweep_table(
    baseline_labeled: Metrics,
    results: list[SweepResult],
    winner: SweepResult,
) -> None:
    """Section: the decoy-weight sweep, scored by the OOF ranking proxy."""
    print("## Decoy-weight sweep (selected by the OOF ranking proxy)\n")
    print(
        "All configs share the augmented matrix, locked hyperparameters, seed, "
        "and 5-group GroupKFold folds; only the decoy `sample_weight` `w` "
        "differs. **rank_proxy** is the fraction of harvested MARCs whose true "
        "pair out-ranks all its decoys under grouped OOF — the eval-free top-1 "
        "proxy. **OOF AUC / best-F1 (labeled)** restrict the OOF predictions to "
        "the labeled rows, the apples-to-apples Gate-2 health check against the "
        "labeled-only baseline. The winner (★) maximizes rank_proxy, tie-broken "
        "by labeled best-F1.\n"
    )
    print(
        "| w | OOF AUC (full) | OOF AUC (labeled) | best-F1 (labeled) | "
        "rank_proxy | wins/evaluable |"
    )
    print("|:---|---:|---:|---:|---:|---:|")
    for result in results:
        mark = " ★" if result is winner else ""
        print(
            f"| {result.decoy_weight}{mark} | "
            f"{result.augmented_full.auc:.4f} | "
            f"{result.augmented_labeled.auc:.4f} | "
            f"{result.augmented_labeled.best_f1:.4f} | "
            f"{result.rank_proxy:.4f} | "
            f"{result.rank_wins}/{result.rank_evaluable} |"
        )
    print()
    auc_delta = winner.augmented_labeled.auc - baseline_labeled.auc
    f1_delta = winner.augmented_labeled.best_f1 - baseline_labeled.best_f1
    print(
        f"**Winner: w={winner.decoy_weight}** (rank_proxy "
        f"{winner.rank_proxy:.4f}, {winner.rank_wins}/{winner.rank_evaluable}). "
        "Labeled-only baseline grouped-OOF: AUC "
        f"{baseline_labeled.auc:.4f}, best-F1 {baseline_labeled.best_f1:.4f}. "
        "**Winner minus labeled-only on the labeled-rows slice:** "
        f"AUC {auc_delta:+.4f}, best-F1 {f1_delta:+.4f}.\n"
    )


def _parse_limit() -> int | None:
    """Parse the optional ``--limit N`` smoke flag from argv."""
    parser = ArgumentParser(description="Decoy-negatives experiment (round 2).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap the labeled-MATCH MARCs harvested (smoke runs; report flagged)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    limit: int | None = args.limit
    return limit


def main() -> None:
    """Run the decoy experiment and print the markdown report to stdout."""
    limit = _parse_limit()
    _progress(f"script started (limit={limit})")
    entries = current_entries(_VAULT_PATH)
    dataset = _build_dataset(entries, limit)
    _progress(f"harvest done: {dataset.n_decoy} decoys")

    baseline_mask = dataset.is_labeled
    labeled_idx = baseline_mask == 1
    baseline_oof = _grouped_oof(
        dataset.x[labeled_idx],
        dataset.y[labeled_idx],
        dataset.groups[labeled_idx],
    )
    baseline_labeled = _metrics(dataset.y[labeled_idx], baseline_oof)
    _progress("baseline OOF done")

    results = _run_sweep(dataset, baseline_mask)
    winner = _select_winner(results)
    _progress(f"sweep done; winner w={winner.decoy_weight}")

    _persist_artifact(dataset, winner.decoy_weight)
    _progress("artifact saved")

    _print_header(dataset)
    _print_sweep_table(baseline_labeled, results, winner)
    _emit_ab_placeholder(dataset, baseline_labeled, winner)
    _progress("report written")


def _emit_ab_placeholder(
    dataset: Dataset,
    baseline_labeled: Metrics,
    winner: SweepResult,
) -> None:
    """Emit the A/B + decision sections; the A/B numbers are filled post-eval.

    The learned-scorer eval runs as a separate ~18-minute CLI pass against the
    persisted artifact (`pdm run pd-matcher eval --scorer learned`); its top-1
    counts cannot be produced inside this script. This section states the gate,
    the reference numbers, and how to read the result; the agent driving the
    run fills the realized decoy-model top-1 row and the verdict from
    `/tmp/ab_learned_decoys.json`.
    """
    print("## A/B result — top-1 linkage (pass A)\n")
    _print_limit_warning(dataset.harvest.limit)
    print(
        "The weighted-mean reference is unchanged "
        f"(`/tmp/ab_weighted.json`): {_WEIGHTED_CORRECT_TOP}/922 correct top "
        f"picks, F1 {_WEIGHTED_F1_REFERENCE:.5f}. The labeled-only learned model "
        f"scored {_LABELED_ONLY_CORRECT_TOP}/922; round 1 (above-floor decoys "
        f"only) scored {_ROUND1_CORRECT_TOP}/922. The round-2 decoy model below "
        f"(below-floor harvest, winning decoy weight w={winner.decoy_weight}) is "
        "evaluated by `pd-matcher eval --scorer learned` against the "
        "freshly-persisted `caches/learned_scorer.*`.\n"
    )
    print(
        "_Caveat (unchanged from the prior A/Bs):_ pass A on labeled MARCs is "
        "still partially train-set-flavored for the labeled rows, and pass-B "
        "AUC near 1.0 on labeled pairs remains uninformative. The decoy "
        "population at rank-2..N is now in-distribution, but the gate metric is "
        "pass-A top-1 vs the weighted reference.\n"
    )
    print("| scorer | correct top / 922 | precision | recall | F1 |")
    print("|:---|---:|---:|---:|---:|")
    print(
        f"| weighted_mean (reference) | {_WEIGHTED_CORRECT_TOP} | 0.99457 | "
        f"0.99241 | {_WEIGHTED_F1_REFERENCE:.5f} |"
    )
    print(
        f"| learned (labeled-only, prior A/B) | {_LABELED_ONLY_CORRECT_TOP} | "
        "0.98914 | 0.98807 | 0.98861 |"
    )
    print(
        f"| learned (round 1, above-floor) | {_ROUND1_CORRECT_TOP} | 0.99564 | "
        "0.99024 | 0.99293 |"
    )
    print(
        f"| learned (round 2, below-floor, w={winner.decoy_weight}) | _FILL_ | "
        "_FILL_ | _FILL_ | _FILL_ |"
    )
    print()
    print("## Decision against the #77 gate\n")
    auc_delta = winner.augmented_labeled.auc - baseline_labeled.auc
    f1_delta = winner.augmented_labeled.best_f1 - baseline_labeled.best_f1
    gate2_pass = f1_delta >= 0.0 and auc_delta >= 0.0
    print(
        f"- **Gate 1 — top-1 F1 ≥ {_WEIGHTED_F1_REFERENCE:.5f} (weighted "
        "reference):** _FILL PASS/FAIL_ (decoy-model correct top _FILL_/922). "
        f"OOF rank proxy for the winner: {winner.rank_proxy:.4f} "
        f"({winner.rank_wins}/{winner.rank_evaluable}).\n"
    )
    print(
        "- **Gate 2 — grouped-OOF labeled-rows slice not degraded vs "
        f"labeled-only:** **{'PASS' if gate2_pass else 'FAIL'}** "
        f"(AUC {auc_delta:+.4f}, best-F1 {f1_delta:+.4f} on the labeled rows, "
        f"winning w={winner.decoy_weight}).\n"
    )
    print(
        "- **Gate 3 — throughput:** **PASS by construction.** Decoys change "
        "training only; inference is the identical per-candidate "
        "`Booster.predict`, so throughput is unchanged from the labeled-only "
        "learned model (85% of weighted mean in the prior A/B).\n"
    )
    print(
        "**Programmatic verdict:** _FILL_ — ADOPT-CANDIDATE if Gate 1 passes "
        "(recommend flipping the default in a follow-up), IMPROVED-BUT-SHORT if "
        f"the decoy model's correct-top exceeds {_ROUND1_CORRECT_TOP} but "
        f"stays below {_WEIGHTED_CORRECT_TOP}, NO-IMPROVEMENT otherwise.\n"
    )
    print("## Artifact + reproduction\n")
    print(
        "The final Booster is fit on ALL augmented rows with the locked "
        f"hyperparameters and the winning decoy weight (w={winner.decoy_weight}) "
        "and persisted via the production `save_learned_model` to "
        "`caches/learned_scorer.{txt,msgpack}`, OVERWRITING the labeled-only "
        "artifact. Both files are gitignored and re-derivable: "
        "`pdm run pd-matcher train-scorer` rebuilds the labeled-only model; "
        "rerunning this script rebuilds the decoy-augmented one.\n"
    )
    print("```")
    print("pdm run python scripts/learned_scorer_decoys.py \\")
    print("    > docs/findings/learned_scorer_decoys_2026-06-12.md")
    print(
        "pdm run pd-matcher eval --index caches/cce.lmdb --scorer learned \\"
    )
    print("    --report /tmp/ab_learned_decoys.json")
    print("```")


if __name__ == "__main__":
    main()
