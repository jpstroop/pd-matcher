"""K-fold held-out top-1 eval for the learned matcher (issue #80).

Throwaway one-off measurement script. NOT shipped; ``scripts/`` is gitignored
from the published package via the ``[tool.coverage.run].source`` allowlist. It
does NOT modify anything under ``src/``. It writes nothing under ``data/``. The
vault is read-only.

THE CONFOUND THIS RESOLVES. Every prior learned-vs-weighted top-1 comparison is
contaminated: production ``train-scorer`` fits on ALL trainable vault pairs and
``eval`` then scores top-1 over the WHOLE vault, so the learned model grades
itself on pairs it memorized. Round-2 pass-A 919/922
(``docs/findings/learned_scorer_decoys_round2_2026-06-13.md``) is an UPPER
BOUND; the only leakage-free signal there — a grouped-OOF rank PROXY at 911/922
— sits BELOW the weighted mean's honest 915/922. The proxy is a per-row ranking
question, not the real top-1 gate (it cannot see the full retrieved candidate
set, only the harvested decoys). We have never produced an honest top-1 number.

WHAT THIS SCRIPT DOES. It runs the GroupKFold-through-pipeline eval the round-2
findings named as "the only confound-free answer":

1. **GroupKFold by ``marc_control_id``, 5 folds**, over the labeled-MATCH MARCs.
   Every MARC (its labeled pairs AND any decoys harvested from it) lives in
   exactly ONE fold. Deterministic, fixed seed.

2. **Per held-out fold H**, a model is trained FROM SCRATCH on the OTHER four
   folds only:
   * labeled pairs (match + no_match, ``unsure`` excluded) for TRAIN-fold MARCs,
     via the canonical :func:`pd_matcher.match.combiners.features.feature_row`;
   * PLUS below-floor decoys harvested via ``match_record`` under a floor of
     ``0.0`` and a large ``top_k`` — the top ``_DECOY_TOP_K`` non-true
     candidates per TRAIN-fold match-MARC (never from held-out MARCs);
   * the round-2 best config: decoy ``sample_weight`` ``w=0.5`` and the locked
     hyperparameters (``max_depth=3, num_leaves=8, min_data_in_leaf=10,
     lambda_l2=1.0, n_estimators=200, class_weight=balanced``).
   The fold Booster is wrapped directly as the production
   :class:`pd_matcher.match.combiners.learned.LearnedCombiner`.

3. **Held-out top-1 eval for fold H.** For each labeled-MATCH MARC in H, the
   production ``match_record`` runs with the FOLD model as combiner over the
   MARC's FULL retrieved candidate set; top-1 is correct when the best pick's
   ``nypl_uuid`` equals the gold match uuid. The fold model never trained on
   this MARC, so the pick is honest.

4. **Aggregate over all 5 folds** → honest learned top-1 = correct / evaluated,
   reported as ``pd-matcher eval`` does it: precision = correct / had-a-top,
   recall = correct / evaluated, F1 the harmonic mean.

5. **Weighted-mean reference, inline.** The weighted mean is deterministic and
   needs no training, so its honest top-1 is its standard number. It is rerun
   inline over the EXACT same match-MARC set (same denominator), so the
   comparison is apples-to-apples.

Usage:
    pdm run python scripts/learned_scorer_heldout.py \\
        > docs/findings/learned_scorer_heldout_2026-06-13.md

    # Smoke run capping the match-MARCs across all folds (report is flagged):
    pdm run python scripts/learned_scorer_heldout.py --limit 25 \\
        > /tmp/heldout_smoke.md
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
from numpy import zeros
from numpy.typing import NDArray
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
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.features import feature_names
from pd_matcher.match.combiners.features import feature_row
from pd_matcher.match.combiners.learned import LearnedCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
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

_RANDOM_STATE: Final[int] = 20260613
_N_SPLITS: Final[int] = 5

# Round-2 best config: below-floor harvest, top-5 decoys/MARC, decoy weight 0.5.
_DECOY_TOP_K: Final[int] = 5
_HARVEST_TOP_K: Final[int] = _DECOY_TOP_K + 16
_DECOY_FLOOR: Final[float] = 0.0
_DECOY_WEIGHT: Final[float] = 0.5

# Confounded reference numbers carried forward for the context row.
_WEIGHTED_KNOWN_CORRECT_TOP: Final[int] = 915
_LABELED_ONLY_CORRECT_TOP: Final[int] = 911
_ROUND2_PASS_A_CORRECT_TOP: Final[int] = 919
_ROUND2_OOF_PROXY_CORRECT: Final[int] = 911
_PARITY_BAND_MARCS: Final[int] = 2


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
    per-scorer Evidence is identical regardless of the combiner, so forcing
    weighted-mean lets the harvest and the inline reference run without a
    learned artifact.
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


def _partition_labeled(entries: dict[tuple[str, str], VaultEntry]) -> list[VaultEntry]:
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


def _fold_assignment(match_marc_ids: list[str]) -> dict[str, int]:
    """Assign each match-MARC to one of ``_N_SPLITS`` folds via GroupKFold.

    Each MARC is its own group, so GroupKFold partitions the MARCs into folds
    with no MARC split across folds (degenerate-by-design: the "group" unit and
    the "row" unit coincide here, which is exactly the held-out partition we
    want). The assignment is deterministic for a fixed MARC order and seed.
    """
    n_marcs = len(match_marc_ids)
    n_splits = min(_N_SPLITS, n_marcs)
    dummy_x: NDArray[int64] = zeros((n_marcs, 1), dtype=int64)
    dummy_y: NDArray[int64] = zeros(n_marcs, dtype=int64)
    groups: NDArray[int64] = asarray(range(n_marcs), dtype=int64)
    splitter = GroupKFold(n_splits=n_splits)
    fold_of: dict[str, int] = {}
    for fold_index, (_, test_idx) in enumerate(splitter.split(dummy_x, dummy_y, groups)):
        for row in test_idx:
            fold_of[match_marc_ids[int(row)]] = fold_index
    return fold_of


@dataclass(frozen=True, slots=True)
class FoldHarvest:
    """The train-side feature matrix and held-out MARC set for one fold."""

    x: NDArray[float64]
    y: NDArray[int64]
    sample_weight: NDArray[float64]
    n_labeled: int
    n_decoy: int
    holdout_marc_ids: list[str]


@dataclass(frozen=True, slots=True)
class HarvestContext:
    """Shared, fold-independent resolution machinery and vault projections."""

    lookup: NyplIndexLookup
    idf: IdfTable
    author_idf: IdfTable
    publisher_idf: IdfTable
    pairings: CompiledPairings
    scoring_config: MatchingConfig
    harvest_config: MatchingConfig
    marc_by_id: dict[str, MarcRecord]
    ground_truth: dict[str, str]
    labeled_by_marc: dict[str, set[str]]
    labeled_entries: list[VaultEntry]


def _build_fold_matrix(
    ctx: HarvestContext,
    train_marc_ids: set[str],
    holdout_marc_ids: list[str],
) -> FoldHarvest:
    """Build one fold's train matrix from TRAIN-fold MARCs only.

    Labeled rows: every non-``unsure`` vault pair whose ``marc_control_id`` is
    in ``train_marc_ids`` (match + no_match). Decoy rows: the top
    ``_DECOY_TOP_K`` non-true candidates per TRAIN-fold match-MARC, harvested
    via ``match_record`` under a zero floor. Decoys are NEVER harvested from
    held-out MARCs, so the fold model is blind to fold H.
    """
    score_pair = make_pair_scorer(
        matching_config=ctx.scoring_config,
        pairings=ctx.pairings,
        idf=ctx.idf,
        author_idf=ctx.author_idf,
        publisher_idf=ctx.publisher_idf,
        calibrator=None,
    )
    weighted_combiner = build_combiner(ctx.scoring_config, learned_model_dir=None)

    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    weights: list[float] = []
    n_labeled = 0
    n_decoy = 0

    for entry in ctx.labeled_entries:
        if entry.marc_control_id not in train_marc_ids:
            continue
        marc = ctx.marc_by_id.get(entry.marc_control_id)
        if marc is None:
            continue
        cce = ctx.lookup.get_registration(entry.nypl_uuid)
        if cce is None:
            continue
        candidate = score_pair(marc, cce)
        rows.append(feature_row(candidate.evidence))
        labels.append(1 if entry.verdict == _VERDICT_MATCH else 0)
        weights.append(1.0)
        n_labeled += 1

    for marc_id in train_marc_ids:
        true_uuid = ctx.ground_truth.get(marc_id)
        if true_uuid is None:
            continue
        marc = ctx.marc_by_id.get(marc_id)
        if marc is None:
            continue
        result = match_record(
            marc,
            lookup=ctx.lookup,
            config=ctx.harvest_config,
            idf=ctx.idf,
            author_idf=ctx.author_idf,
            publisher_idf=ctx.publisher_idf,
            calibrator=None,
            combiner=weighted_combiner,
            pairings=ctx.pairings,
            top_k=_HARVEST_TOP_K,
        )
        ranked = ([result.best] if result.best is not None else []) + list(result.alternates)
        labeled_uuids = ctx.labeled_by_marc.get(marc_id, set())
        taken = 0
        for candidate in ranked:
            if taken >= _DECOY_TOP_K:
                break
            if candidate.nypl_uuid == true_uuid:
                continue
            if candidate.nypl_uuid in labeled_uuids:
                continue
            rows.append(feature_row(candidate.evidence))
            labels.append(0)
            weights.append(_DECOY_WEIGHT)
            n_decoy += 1
            taken += 1

    return FoldHarvest(
        x=asarray(rows, dtype=float64),
        y=asarray(labels, dtype=int64),
        sample_weight=asarray(weights, dtype=float64),
        n_labeled=n_labeled,
        n_decoy=n_decoy,
        holdout_marc_ids=holdout_marc_ids,
    )


def _train_fold_combiner(fold: FoldHarvest) -> LearnedCombiner:
    """Fit a LightGBM model on the fold's train matrix; wrap as a Combiner.

    Constructs the production :class:`LearnedCombiner` directly around the fitted
    Booster and the current canonical feature-name order — no disk round-trip,
    no artifact overwrite (the production ``caches/learned_scorer.*`` is left
    untouched).
    """
    model = LGBMClassifier(**_classifier_params())
    model.fit(fold.x, fold.y.astype(float64), sample_weight=fold.sample_weight)
    return LearnedCombiner(booster=model.booster_, names=feature_names())


@dataclass(frozen=True, slots=True)
class TopOneTally:
    """Top-1 counts for one scorer over a set of match-MARCs."""

    evaluated: int
    with_top: int
    correct_top: int

    @property
    def precision(self) -> float:
        """Correct top picks over MARCs that produced any top pick."""
        return self.correct_top / self.with_top if self.with_top else 0.0

    @property
    def recall(self) -> float:
        """Correct top picks over MARCs evaluated."""
        return self.correct_top / self.evaluated if self.evaluated else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 2.0 * self.precision * self.recall / total if total > 0.0 else 0.0


def _eval_top_one(
    ctx: HarvestContext,
    marc_ids: list[str],
    combiner: Combiner,
    config: MatchingConfig,
) -> tuple[int, int, int]:
    """Run per-MARC top-1 over ``marc_ids``; return ``(eval, with_top, correct)``.

    Mirrors ``pd_matcher.eval.ground_truth._run_pass_b``: full retrieved
    candidate set, the configured floor applied, top-1 correct when the best
    pick's ``nypl_uuid`` equals the gold uuid.
    """
    evaluated = 0
    with_top = 0
    correct = 0
    for marc_id in marc_ids:
        gt_uuid = ctx.ground_truth.get(marc_id)
        if gt_uuid is None:
            continue
        marc = ctx.marc_by_id.get(marc_id)
        if marc is None:
            continue
        evaluated += 1
        result = match_record(
            marc,
            lookup=ctx.lookup,
            config=config,
            idf=ctx.idf,
            author_idf=ctx.author_idf,
            publisher_idf=ctx.publisher_idf,
            calibrator=None,
            combiner=combiner,
            pairings=ctx.pairings,
        )
        if result.best is None:
            continue
        with_top += 1
        if result.best.nypl_uuid == gt_uuid:
            correct += 1
    return evaluated, with_top, correct


@dataclass(frozen=True, slots=True)
class HeldoutResult:
    """Aggregate held-out learned vs inline weighted top-1 over all folds."""

    learned: TopOneTally
    weighted: TopOneTally
    n_folds: int
    fold_sizes: list[int]
    total_decoys: int
    total_train_labeled: int
    feature_count: int
    limit: int | None


def _limited_marc_ids(ground_truth: dict[str, str], limit: int | None) -> list[str]:
    """Return the match-MARC ids to evaluate, capped at ``limit``.

    Insertion order of ``ground_truth`` (vault scan order) is the deterministic
    selection order, so a given ``--limit`` always picks the same MARCs.
    """
    marc_ids = list(ground_truth)
    if limit is not None:
        marc_ids = marc_ids[:limit]
    return marc_ids


def run_heldout(entries: dict[tuple[str, str], VaultEntry], limit: int | None) -> HeldoutResult:
    """Run the 5-fold held-out top-1 comparison and aggregate over folds."""
    matching_config = _load_default_matching_config()
    pairing_config = _load_default_pairing_config()
    scoring_config = _scoring_config(matching_config)
    harvest_config = structs.replace(scoring_config, min_combined_score=_DECOY_FLOOR)

    ground_truth = _ground_truth_by_marc(entries)
    eval_marc_ids = _limited_marc_ids(ground_truth, limit)
    fold_of = _fold_assignment(eval_marc_ids)
    n_folds = max(fold_of.values()) + 1 if fold_of else 0

    labeled_entries = _partition_labeled(entries)
    labeled_by_marc = _labeled_uuids_by_marc(entries)
    eval_set = set(eval_marc_ids)
    # Training MARCs span the whole non-unsure vault, restricted to the eval
    # MARC universe under --limit so train/holdout populations stay coherent.
    train_universe = (
        {e.marc_control_id for e in labeled_entries} | set(ground_truth)
        if limit is None
        else eval_set
    )
    needed_marc_ids = train_universe | eval_set
    marc_by_id = build_marc_index(_POOL_PATH, needed_marc_ids)
    pairings = compile_pairings(pairing_config)

    learned_eval = 0
    learned_with_top = 0
    learned_correct = 0
    weighted_eval = 0
    weighted_with_top = 0
    weighted_correct = 0
    fold_sizes: list[int] = []
    total_decoys = 0
    total_train_labeled = 0
    feature_count = len(feature_names())

    with NyplIndexLookup(_INDEX_PATH) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
        ctx = HarvestContext(
            lookup=lookup,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            pairings=pairings,
            scoring_config=scoring_config,
            harvest_config=harvest_config,
            marc_by_id=marc_by_id,
            ground_truth=ground_truth,
            labeled_by_marc=labeled_by_marc,
            labeled_entries=labeled_entries,
        )
        weighted_combiner = build_combiner(scoring_config, learned_model_dir=None)

        for fold_index in range(n_folds):
            holdout_marc_ids = [m for m in eval_marc_ids if fold_of[m] == fold_index]
            holdout_set = set(holdout_marc_ids)
            train_marc_ids = train_universe - holdout_set
            fold = _build_fold_matrix(ctx, train_marc_ids, holdout_marc_ids)
            total_decoys += fold.n_decoy
            total_train_labeled += fold.n_labeled
            fold_sizes.append(len(holdout_marc_ids))
            _progress(
                f"fold {fold_index}: train_labeled={fold.n_labeled} "
                f"decoys={fold.n_decoy} holdout={len(holdout_marc_ids)}"
            )
            combiner = _train_fold_combiner(fold)

            l_eval, l_top, l_correct = _eval_top_one(
                ctx, holdout_marc_ids, combiner, matching_config
            )
            learned_eval += l_eval
            learned_with_top += l_top
            learned_correct += l_correct

            w_eval, w_top, w_correct = _eval_top_one(
                ctx, holdout_marc_ids, weighted_combiner, scoring_config
            )
            weighted_eval += w_eval
            weighted_with_top += w_top
            weighted_correct += w_correct
            _progress(
                f"fold {fold_index} done: learned {l_correct}/{l_eval}, "
                f"weighted {w_correct}/{w_eval}"
            )

    return HeldoutResult(
        learned=TopOneTally(learned_eval, learned_with_top, learned_correct),
        weighted=TopOneTally(weighted_eval, weighted_with_top, weighted_correct),
        n_folds=n_folds,
        fold_sizes=fold_sizes,
        total_decoys=total_decoys,
        total_train_labeled=total_train_labeled,
        feature_count=feature_count,
        limit=limit,
    )


def _verdict(result: HeldoutResult) -> tuple[str, str]:
    """Return ``(label, gap_sentence)`` from the learned-minus-weighted gap."""
    gap = result.learned.correct_top - result.weighted.correct_top
    f1_gap = result.learned.f1 - result.weighted.f1
    sign = "+" if gap >= 0 else ""
    gap_sentence = (
        f"learned minus weighted = {sign}{gap} MARCs "
        f"({result.learned.correct_top} vs {result.weighted.correct_top}), "
        f"F1 {f1_gap:+.5f}"
    )
    if gap > _PARITY_BAND_MARCS:
        return "ADOPT-CANDIDATE", gap_sentence
    if gap < -_PARITY_BAND_MARCS:
        return "HONEST-LOSS", gap_sentence
    return "PARITY", gap_sentence


def _print_limit_warning(limit: int | None) -> None:
    """Emit a prominent SMOKE banner when a ``--limit`` was active."""
    if limit is None:
        return
    print(
        f"> ⚠️ **SMOKE RUN — `--limit {limit}` was active.** Only {limit} "
        "match-MARCs were evaluated across all folds; every count below is from "
        "a truncated set and is NOT a real finding. Re-run WITHOUT `--limit` for "
        "the production report.\n"
    )


def _print_report(result: HeldoutResult) -> None:
    """Emit the full markdown report to stdout."""
    print("# Learned-scorer held-out top-1 (k-fold through pipeline) — 2026-06-13\n")
    _print_limit_warning(result.limit)
    print(
        "Issue #80. The FIRST confound-free top-1 comparison of the learned "
        "matcher against the weighted mean. Production `train-scorer` fits on "
        "ALL trainable vault pairs and `eval` then scores top-1 over the WHOLE "
        "vault, so the learned model grades itself on memorized pairs — its "
        f"round-2 pass-A {_ROUND2_PASS_A_CORRECT_TOP}/922 "
        "(`docs/findings/learned_scorer_decoys_round2_2026-06-13.md`) is an "
        "UPPER BOUND, and the only leakage-free signal there (a grouped-OOF "
        f"rank PROXY at {_ROUND2_OOF_PROXY_CORRECT}/922) sits below the weighted "
        f"mean's honest {_WEIGHTED_KNOWN_CORRECT_TOP}/922. This script removes "
        "the confound by construction: every evaluated MARC is scored by a "
        "fold-model that never trained on it.\n"
    )
    print("## Method\n")
    print(
        f"**GroupKFold by `marc_control_id`, {result.n_folds} folds.** The "
        "labeled-MATCH MARCs are partitioned into folds with each MARC as its "
        "own group, so every MARC — its labeled pairs AND any decoys harvested "
        "from it — lives in exactly one fold. There is no path for a MARC's own "
        "rows to leak into the model that grades it. Deterministic for a fixed "
        f"MARC order and seed (`random_state={_RANDOM_STATE}`).\n"
    )
    print(
        "**Per held-out fold H, train from scratch on the other four folds.** "
        "The fold's training matrix is: every non-`unsure` vault pair (match + "
        "no_match) whose `marc_control_id` is a TRAIN-fold MARC, via the "
        "canonical `feature_row`; PLUS below-floor decoys — the top "
        f"`{_DECOY_TOP_K}` non-true candidates per TRAIN-fold match-MARC, "
        "harvested by `match_record` under `min_combined_score=0.0` and "
        f"`top_k={_HARVEST_TOP_K}` (the full ranked candidate set, no floor "
        "cull). Decoys are NEVER harvested from held-out MARCs. This reproduces "
        f"the round-2 best config: decoy `sample_weight` w={_DECOY_WEIGHT}, "
        f"locked hyperparameters (max_depth={_MAX_DEPTH}, num_leaves={_NUM_LEAVES}"
        f", min_data_in_leaf={_MIN_DATA_IN_LEAF}, lambda_l2={_LAMBDA_L2}, "
        f"n_estimators={_N_ESTIMATORS}, class_weight={_CLASS_WEIGHT}). The fold "
        "Booster is wrapped directly as the production `LearnedCombiner` (no "
        "disk round-trip; the production `caches/learned_scorer.*` artifact is "
        "left untouched).\n"
    )
    print(
        "**Held-out top-1 eval for fold H.** For each labeled-MATCH MARC in H, "
        "the production `match_record` runs with the FOLD model as combiner over "
        "the MARC's FULL retrieved candidate set (the configured floor applied "
        "exactly as production does). Top-1 is correct when the best pick's "
        "`nypl_uuid` equals the gold uuid. **The model-never-saw-the-evaluated-"
        "MARC guarantee is what makes this honest** — unlike pass-A eval, no "
        "evaluated pair was a training row.\n"
    )
    print(
        "**Weighted-mean reference, inline.** The weighted mean is deterministic "
        "and untrained, so its honest top-1 is its standard number. It is rerun "
        "inline over the EXACT same match-MARC set, fold by fold, so the "
        "denominator matches the learned column to the MARC.\n"
    )
    print(
        f"- **Feature count**: {result.feature_count} (production "
        "`feature_names()`)"
    )
    print(
        f"- **Folds**: {result.n_folds}; held-out MARCs per fold: "
        f"{', '.join(str(n) for n in result.fold_sizes)} "
        f"(total {sum(result.fold_sizes)})"
    )
    print(
        f"- **Train-side rows (summed over folds)**: labeled "
        f"{result.total_train_labeled}, decoys {result.total_decoys} "
        "(each fold trains on ~4/5 of the labeled vault plus its own decoy "
        "harvest)\n"
    )

    print("## Headline — honest held-out top-1\n")
    _print_limit_warning(result.limit)
    print(
        "Both columns are over the same match-MARC set; the weighted column is "
        "rerun inline so the denominators are identical.\n"
    )
    denom = result.learned.evaluated
    print(f"| scorer | correct top / {denom} | precision | recall | F1 |")
    print("|:---|---:|---:|---:|---:|")
    print(
        f"| weighted_mean (honest, inline) | {result.weighted.correct_top} | "
        f"{result.weighted.precision:.5f} | {result.weighted.recall:.5f} | "
        f"{result.weighted.f1:.5f} |"
    )
    print(
        f"| learned (honest, k-fold held-out) | {result.learned.correct_top} | "
        f"{result.learned.precision:.5f} | {result.learned.recall:.5f} | "
        f"{result.learned.f1:.5f} |"
    )
    print()

    print("## Context — prior CONFOUNDED learned numbers (over the whole vault)\n")
    print(
        "These were all measured with the learned model graded on pairs it "
        "trained on; they are upper bounds, shown for honest-vs-confounded "
        "contrast. The weighted mean has no training, so its 915/922 is already "
        "honest.\n"
    )
    print("| measurement | correct top / 922 | leakage-free? |")
    print("|:---|---:|:---|")
    print(
        f"| weighted_mean (known full-vault) | {_WEIGHTED_KNOWN_CORRECT_TOP} | "
        "yes (untrained) |"
    )
    print(
        f"| learned labeled-only (pass A) | {_LABELED_ONLY_CORRECT_TOP} | "
        "no (trained on eval pairs) |"
    )
    print(
        f"| learned round-2 decoys (pass A) | {_ROUND2_PASS_A_CORRECT_TOP} | "
        "no (trained on eval pairs) |"
    )
    print(
        f"| learned round-2 grouped-OOF rank PROXY | {_ROUND2_OOF_PROXY_CORRECT} "
        "| partial (per-row proxy, not full top-1) |"
    )
    print()

    label, gap_sentence = _verdict(result)
    print("## Decision\n")
    print(
        "- **ADOPT-CANDIDATE** if honest learned top-1 clears weighted by more "
        f"than {_PARITY_BAND_MARCS} MARCs (recommend flipping the default).\n"
        f"- **PARITY** if within ±{_PARITY_BAND_MARCS} MARCs (noise).\n"
        "- **HONEST-LOSS** if clearly below weighted.\n"
    )
    print(f"**Verdict: {label}.** The honest held-out gap is {gap_sentence}.\n")
    if label == "ADOPT-CANDIDATE":
        print(
            "The learned matcher wins on unseen MARCs, not just on memorized "
            "pairs. Recommend a follow-up to flip `scorer` to `learned` as the "
            "production default and (per #80 tier 2) make the holdout durable so "
            "every future retrain stays honest.\n"
        )
    elif label == "PARITY":
        print(
            "The learned matcher is at honest parity with the hand-tuned "
            "weighted mean on this Princeton-only vault — the confounded pass-A "
            "lead does NOT survive de-leaking. Consistent with the round-2 "
            "finding that the OOF proxy did not clear 915. No basis to flip the "
            "default on top-1 alone; the model's value remains the #76 audit "
            "queue (pair-level discrimination) and future cross-institution "
            "generalization.\n"
        )
    else:
        print(
            "The learned matcher loses honestly to the weighted mean on unseen "
            "MARCs — the pass-A lead was entirely memorization. The weighted "
            "mean stays the production default; the learned artifact's value is "
            "the #76 audit queue, not top-1 linkage.\n"
        )

    print("## Reproduction\n")
    print("```")
    print("pdm run python scripts/learned_scorer_heldout.py \\")
    print("    > docs/findings/learned_scorer_heldout_2026-06-13.md")
    print("```")
    print(
        "\nThis script touches nothing under `src/` or `data/`, overwrites no "
        "artifact, and is deterministic (fixed seed, `n_jobs=1`).\n"
    )


def _parse_limit() -> int | None:
    """Parse the optional ``--limit N`` smoke flag from argv."""
    parser = ArgumentParser(description="K-fold held-out top-1 eval (issue #80).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap the match-MARCs evaluated across all folds (smoke; flagged)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    limit: int | None = args.limit
    return limit


def main() -> None:
    """Run the held-out eval and print the markdown report to stdout."""
    limit = _parse_limit()
    _progress(f"script started (limit={limit})")
    entries = current_entries(_VAULT_PATH)
    result = run_heldout(entries, limit)
    _progress(
        f"folds done: learned {result.learned.correct_top}/{result.learned.evaluated}, "
        f"weighted {result.weighted.correct_top}/{result.weighted.evaluated}"
    )
    _print_report(result)
    _progress("report written")


if __name__ == "__main__":
    main()
