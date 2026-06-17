"""Wild-distribution SEPARATION test for the learned matcher (issue #84).

Throwaway one-off measurement script. NOT shipped; ``scripts/`` is gitignored
from the published package via the ``[tool.coverage.run].source`` allowlist. It
does NOT modify anything under ``src/``. It writes nothing under ``data/`` and
overwrites no ``caches/learned_scorer.*`` artifact (the trained Booster is
wrapped directly in memory). The vault is read-only.

THE QUESTION THIS RESOLVES. Every prior learned-vs-weighted separation number
is a grouped-OOF measurement over the labeled MATCH-biased vault: learned AUC
0.993, weighted 0.942. The reframe (`project_current_status`) is that the
matcher's scaling lever is SEPARATION — can a score threshold auto-decide a
pair without a human — not top-1 (which is saturated). The open worry is that
learned separation might be "worse on unknown" pairs: the vault is a
self-selected distribution, so an OOF split inside it is not the same as a
held-out tail labeled later from a different, middle-heavy sample.

WHAT THIS SCRIPT DOES. It splits the vault strictly by ``labeled_at``: every
pair labeled at or before the cutoff is TRAIN (~1500), every pair labeled after
is TEST (~500, the wild, stratified, middle-heavy held-out sample). A single
learned model trains on the TRAIN pairs ONLY (labeled-only, production
``train-scorer`` recipe — no decoys: this is pair-level separation, not top-1).
Each TEST pair is then scored once and graded by three combiner arms (learned,
weighted_mean, weighted_mean with year's weight zeroed) for pair-level
SEPARATION: ROC-AUC, average precision, a 0.00-1.00 threshold sweep, the
best-F1 threshold, a disagreement dump at the best learned threshold, and a
triage-viability readout (auto-accept / auto-reject tails vs the residual human
middle). The model never trains on a TEST pair, so the separation is honest.

Usage:
    pdm run python scripts/separation_wild_test.py \\
        > docs/findings/separation_wild_test_2026-06-17.md

    # Smoke run capping the TEST pairs scored (report is flagged):
    pdm run python scripts/separation_wild_test.py --limit 40 \\
        > /tmp/septest_smoke.md
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
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import ScorePairFn
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
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_PROGRESS_LOG: Final[Path] = Path("/tmp/agent-progress.log")

_CUTOFF: Final[str] = "2026-06-12T18:55:29Z"

_VERDICT_MATCH: Final[str] = "match"
_VERDICT_NO_MATCH: Final[str] = "no_match"
_VERDICT_UNSURE: Final[str] = "unsure"

# Locked recipe from src/pd_matcher/match/combiners/train.py.
_MAX_DEPTH: Final[int] = 3
_NUM_LEAVES: Final[int] = 8
_MIN_DATA_IN_LEAF: Final[int] = 10
_LAMBDA_L2: Final[float] = 1.0
_N_ESTIMATORS: Final[int] = 200
_CLASS_WEIGHT: Final[str] = "balanced"
_RANDOM_STATE: Final[int] = 20260613

# Approximate expected split sizes; the assertion is a loose sanity band.
_TRAIN_LOW: Final[int] = 1200
_TEST_LOW: Final[int] = 300
_TEST_HIGH: Final[int] = 800

# Threshold sweep grid (inclusive 0.00 .. 1.00 step 0.05).
_SWEEP_STEPS: Final[int] = 21
_SWEEP_STEP: Final[float] = 0.05

# Known labeled-vault grouped-OOF baselines, carried forward for contrast.
_LEARNED_OOF_AUC: Final[float] = 0.993
_WEIGHTED_OOF_AUC: Final[float] = 0.942
# An off-vault TEST AUC within this band of the OOF baseline "holds".
_HOLD_BAND: Final[float] = 0.03

# Disagreement dump cap per direction.
_DUMP_CAP: Final[int] = 25
_LOW_REGION: Final[float] = 0.50


def _progress(message: str) -> None:
    """Emit a one-line milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _build_classifier() -> LGBMClassifier:
    """Construct an :class:`LGBMClassifier` with the locked hyperparameters.

    Parameters are passed as explicit keyword arguments (not a splatted dict)
    so the call stays fully type-checked against the constructor signature.
    """
    return LGBMClassifier(
        max_depth=_MAX_DEPTH,
        num_leaves=_NUM_LEAVES,
        min_data_in_leaf=_MIN_DATA_IN_LEAF,
        reg_lambda=_LAMBDA_L2,
        n_estimators=_N_ESTIMATORS,
        class_weight=_CLASS_WEIGHT,
        objective="binary",
        verbose=-1,
        random_state=_RANDOM_STATE,
        n_jobs=1,
    )


def _scoring_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` forced onto the weighted-mean scorer.

    Mirrors :func:`pd_matcher.match.combiners.train._scoring_config`: the
    per-scorer Evidence is identical regardless of the combiner, so forcing
    weighted-mean lets every pair be scored once without a learned artifact.
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


def _zero_year_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` with ``year_weight`` zeroed and the rest renormalized.

    The schema requires the weight tuple to sum to 1.0, so year's mass is
    redistributed across the remaining scorers in proportion to their existing
    weights. This removes year's contribution to the combined score while
    preserving the relative balance of every other scorer.
    """
    remaining = (
        config.title_weight
        + config.author_weight
        + config.publisher_weight
        + config.edition_weight
        + config.lccn_weight
        + config.isbn_weight
        + config.extent_weight
        + config.volume_weight
    )
    factor = 1.0 / remaining if remaining > 0.0 else 0.0
    return structs.replace(
        config,
        title_weight=config.title_weight * factor,
        author_weight=config.author_weight * factor,
        publisher_weight=config.publisher_weight * factor,
        year_weight=0.0,
        edition_weight=config.edition_weight * factor,
        lccn_weight=config.lccn_weight * factor,
        isbn_weight=config.isbn_weight * factor,
        extent_weight=config.extent_weight * factor,
        volume_weight=config.volume_weight * factor,
    )


def _split_by_cutoff(
    entries: dict[tuple[str, str], VaultEntry],
) -> tuple[list[VaultEntry], list[VaultEntry]]:
    """Partition non-``unsure`` entries into (TRAIN, TEST) by ``labeled_at``.

    TRAIN holds every entry labeled at or before :data:`_CUTOFF`; TEST holds
    every entry labeled strictly after it. ``unsure`` verdicts are dropped from
    both sides. Insertion order is preserved for determinism.
    """
    train: list[VaultEntry] = []
    test: list[VaultEntry] = []
    for entry in entries.values():
        if entry.verdict == _VERDICT_UNSURE:
            continue
        if entry.labeled_at <= _CUTOFF:
            train.append(entry)
        else:
            test.append(entry)
    return train, test


@dataclass(frozen=True, slots=True)
class ResolvedPair:
    """A TEST vault entry resolved to its records and gold separation label."""

    entry: VaultEntry
    marc: MarcRecord
    cce: IndexedNyplRegRecord
    gold: int


@dataclass(frozen=True, slots=True)
class ArmScores:
    """One combiner arm's per-pair calibrated scores and gold labels."""

    name: str
    scores: NDArray[float64]
    gold: NDArray[int64]


@dataclass(frozen=True, slots=True)
class SweepRow:
    """Precision / recall / F1 at one threshold for one arm."""

    threshold: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class TriageReadout:
    """Auto-accept / auto-reject tails and residual human middle for an arm."""

    accept_threshold: float
    reject_threshold: float
    auto_accept: int
    auto_reject: int
    residual: int
    total: int


def _resolve_test_pairs(
    test: list[VaultEntry],
    marc_by_id: dict[str, MarcRecord],
    lookup: NyplIndexLookup,
    limit: int | None,
) -> tuple[list[ResolvedPair], int, int, int]:
    """Resolve TEST entries to records; return (pairs, no_marc, no_cce, dropped).

    ``dropped`` counts ``unsure`` defensively (already filtered upstream) and
    any verdict outside {match, no_match}. The ``--limit`` cap is applied to
    the resolved-and-graded TEST pairs in deterministic vault order.
    """
    pairs: list[ResolvedPair] = []
    no_marc = 0
    no_cce = 0
    dropped = 0
    for entry in test:
        if entry.verdict not in (_VERDICT_MATCH, _VERDICT_NO_MATCH):
            dropped += 1
            continue
        marc = marc_by_id.get(entry.marc_control_id)
        if marc is None:
            no_marc += 1
            continue
        cce = lookup.get_registration(entry.nypl_uuid)
        if cce is None:
            no_cce += 1
            continue
        gold = 1 if entry.verdict == _VERDICT_MATCH else 0
        pairs.append(ResolvedPair(entry=entry, marc=marc, cce=cce, gold=gold))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs, no_marc, no_cce, dropped


def _train_learned_combiner(
    train: list[VaultEntry],
    marc_by_id: dict[str, MarcRecord],
    lookup: NyplIndexLookup,
    score_pair: ScorePairFn,
) -> tuple[LearnedCombiner, int, int]:
    """Fit one labeled-only learned model on the TRAIN pairs; wrap in memory.

    Production ``train-scorer`` recipe with NO decoys: each resolvable TRAIN
    entry contributes one row (``feature_row`` of its scored evidence) labeled
    1 for ``match`` and 0 otherwise. Returns (combiner, n_positive, n_negative).
    The Booster is wrapped directly as the production :class:`LearnedCombiner`;
    the on-disk ``caches/learned_scorer.*`` artifact is never touched.
    """
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    for entry in train:
        if entry.verdict not in (_VERDICT_MATCH, _VERDICT_NO_MATCH):
            continue
        marc = marc_by_id.get(entry.marc_control_id)
        if marc is None:
            continue
        cce = lookup.get_registration(entry.nypl_uuid)
        if cce is None:
            continue
        candidate = score_pair(marc, cce)
        rows.append(feature_row(candidate.evidence))
        labels.append(1 if entry.verdict == _VERDICT_MATCH else 0)
    x: NDArray[float64] = asarray(rows, dtype=float64)
    y: NDArray[int64] = asarray(labels, dtype=int64)
    model = _build_classifier()
    model.fit(x, y.astype(float64))
    combiner = LearnedCombiner(booster=model.booster_, names=feature_names())
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    return combiner, n_positive, n_negative


def _score_arm(
    name: str,
    pairs: list[ResolvedPair],
    combiner: Combiner,
    score_pair: ScorePairFn,
) -> ArmScores:
    """Score every TEST pair with one combiner; return calibrated scores + gold.

    The per-scorer Evidence is rebuilt once per pair via ``score_pair`` and
    then handed to ``combiner.combine`` — identical Evidence flows to all arms,
    so the arms differ only in how they combine it.
    """
    scores: list[float] = []
    gold: list[int] = []
    for pair in pairs:
        candidate = score_pair(pair.marc, pair.cce)
        scores.append(combiner.combine(candidate.evidence).calibrated)
        gold.append(pair.gold)
    return ArmScores(
        name=name,
        scores=asarray(scores, dtype=float64),
        gold=asarray(gold, dtype=int64),
    )


def _sweep(arm: ArmScores) -> list[SweepRow]:
    """Return precision/recall/F1 at thresholds 0.00..1.00 step 0.05."""
    positives = int(arm.gold.sum())
    rows: list[SweepRow] = []
    for step in range(_SWEEP_STEPS):
        threshold = step * _SWEEP_STEP
        predicted_positive = 0
        true_positive = 0
        for score, gold in zip(arm.scores, arm.gold, strict=True):
            if float(score) >= threshold:
                predicted_positive += 1
                if int(gold) == 1:
                    true_positive += 1
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / positives if positives else 0.0
        denom = precision + recall
        f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0
        rows.append(SweepRow(threshold=threshold, precision=precision, recall=recall, f1=f1))
    return rows


def _best_f1_row(rows: list[SweepRow]) -> SweepRow:
    """Return the sweep row with the highest F1 (lowest threshold wins ties)."""
    best = rows[0]
    for row in rows[1:]:
        if row.f1 > best.f1:
            best = row
    return best


def _triage_readout(arm: ArmScores) -> TriageReadout:
    """Find auto-accept / auto-reject tails for ``arm`` and the residual middle.

    Auto-accept threshold T_hi: the LOWEST grid threshold at or above which the
    set of pairs scoring ``>= T_hi`` contains zero gold-no_match (zero
    false-accepts), so everything above it can be auto-ACCEPTED. Auto-reject
    threshold T_lo: the HIGHEST grid threshold at or below which the set of
    pairs scoring ``< T_lo`` contains zero gold-match (zero false-rejects), so
    everything below it can be auto-REJECTED. The residual is the middle band
    left for humans.
    """
    total = len(arm.gold)
    accept_threshold = 1.0 + _SWEEP_STEP
    for step in range(_SWEEP_STEPS):
        threshold = step * _SWEEP_STEP
        false_accept = sum(
            1
            for score, gold in zip(arm.scores, arm.gold, strict=True)
            if float(score) >= threshold and int(gold) == 0
        )
        if false_accept == 0:
            accept_threshold = threshold
            break
    reject_threshold = 0.0
    for step in range(_SWEEP_STEPS):
        threshold = (_SWEEP_STEPS - 1 - step) * _SWEEP_STEP
        false_reject = sum(
            1
            for score, gold in zip(arm.scores, arm.gold, strict=True)
            if float(score) < threshold and int(gold) == 1
        )
        if false_reject == 0:
            reject_threshold = threshold
            break
    auto_accept = sum(1 for score in arm.scores if float(score) >= accept_threshold)
    auto_reject = sum(1 for score in arm.scores if float(score) < reject_threshold)
    residual = total - auto_accept - auto_reject
    return TriageReadout(
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        auto_accept=auto_accept,
        auto_reject=auto_reject,
        residual=residual,
        total=total,
    )


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One false-accept or false-reject pair at the best learned threshold."""

    marc_control_id: str
    nypl_uuid: str
    score: float
    marc_title: str
    cce_title: str
    gold: int


def _disagreements(
    pairs: list[ResolvedPair],
    arm: ArmScores,
    threshold: float,
) -> tuple[list[Disagreement], list[Disagreement]]:
    """Return (false_accepts, false_rejects) at ``threshold`` for ``arm``.

    False-accept: ``score >= threshold`` but gold is no_match. False-reject:
    ``score < threshold`` but gold is match. Each list is sorted by score
    (false-accepts descending, false-rejects ascending) so the worst offenders
    surface first; the caller caps the dump.
    """
    false_accepts: list[Disagreement] = []
    false_rejects: list[Disagreement] = []
    for pair, score in zip(pairs, arm.scores, strict=True):
        value = float(score)
        record = Disagreement(
            marc_control_id=pair.entry.marc_control_id,
            nypl_uuid=pair.entry.nypl_uuid,
            score=value,
            marc_title=pair.marc.title,
            cce_title=pair.cce.title,
            gold=pair.gold,
        )
        if value >= threshold and pair.gold == 0:
            false_accepts.append(record)
        elif value < threshold and pair.gold == 1:
            false_rejects.append(record)
    false_accepts.sort(key=lambda d: d.score, reverse=True)
    false_rejects.sort(key=lambda d: d.score)
    return false_accepts, false_rejects


@dataclass(frozen=True, slots=True)
class SeparationResult:
    """Everything the report needs from one wild separation run."""

    n_train: int
    n_test_total: int
    n_test_scored: int
    n_train_positive: int
    n_train_negative: int
    no_marc: int
    no_cce: int
    dropped: int
    feature_count: int
    arms: list[ArmScores]
    sweeps: dict[str, list[SweepRow]]
    triage: TriageReadout
    false_accepts: list[Disagreement]
    false_rejects: list[Disagreement]
    best_learned: SweepRow
    limit: int | None


def run_separation(
    entries: dict[tuple[str, str], VaultEntry], limit: int | None
) -> SeparationResult:
    """Run the wild separation test and aggregate every reported metric."""
    matching_config = _load_default_matching_config()
    pairing_config = _load_default_pairing_config()
    scoring_config = _scoring_config(matching_config)
    no_year_config = _zero_year_config(scoring_config)

    train, test = _split_by_cutoff(entries)
    train_keys = {(e.marc_control_id, e.nypl_uuid) for e in train}
    test_keys = {(e.marc_control_id, e.nypl_uuid) for e in test}
    assert not (train_keys & test_keys), "TRAIN and TEST share a pair — split is leaky"
    assert len(train) >= _TRAIN_LOW, f"TRAIN too small: {len(train)}"
    assert _TEST_LOW <= len(test) <= _TEST_HIGH, f"TEST out of band: {len(test)}"
    _progress(f"split ok: train={len(train)} test={len(test)} (cutoff={_CUTOFF})")

    needed = {e.marc_control_id for e in train} | {e.marc_control_id for e in test}
    marc_by_id = build_marc_index(_POOL_PATH, needed)
    pairings: CompiledPairings = compile_pairings(pairing_config)

    with NyplIndexLookup(_INDEX_PATH) as lookup:
        idf: IdfTable = build_idf_table(lookup)
        author_idf: IdfTable = build_author_idf_table(lookup)
        publisher_idf: IdfTable = build_publisher_idf_table(lookup)
        score_pair = make_pair_scorer(
            matching_config=scoring_config,
            pairings=pairings,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            calibrator=None,
        )

        _progress("training learned model on TRAIN (labeled-only, no decoys)")
        learned, n_positive, n_negative = _train_learned_combiner(
            train, marc_by_id, lookup, score_pair
        )
        weighted = build_combiner(scoring_config, learned_model_dir=None)
        weighted_no_year = build_combiner(no_year_config, learned_model_dir=None)

        test_pairs, no_marc, no_cce, dropped = _resolve_test_pairs(
            test, marc_by_id, lookup, limit
        )
        _progress(
            f"resolved test pairs: scored={len(test_pairs)} no_marc={no_marc} "
            f"no_cce={no_cce} dropped={dropped}"
        )

        learned_arm = _score_arm("learned", test_pairs, learned, score_pair)
        weighted_arm = _score_arm("weighted", test_pairs, weighted, score_pair)
        no_year_arm = _score_arm(
            "weighted_minus_year", test_pairs, weighted_no_year, score_pair
        )
        _progress("all three arms scored")

    arms = [learned_arm, weighted_arm, no_year_arm]
    sweeps = {arm.name: _sweep(arm) for arm in arms}
    best_learned = _best_f1_row(sweeps["learned"])
    triage = _triage_readout(learned_arm)
    false_accepts, false_rejects = _disagreements(
        test_pairs, learned_arm, best_learned.threshold
    )

    return SeparationResult(
        n_train=len(train),
        n_test_total=len(test),
        n_test_scored=len(test_pairs),
        n_train_positive=n_positive,
        n_train_negative=n_negative,
        no_marc=no_marc,
        no_cce=no_cce,
        dropped=dropped,
        feature_count=len(feature_names()),
        arms=arms,
        sweeps=sweeps,
        triage=triage,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        best_learned=best_learned,
        limit=limit,
    )


def _auc(arm: ArmScores) -> float:
    """ROC-AUC of one arm; 0.0 when gold has a single class."""
    if int(arm.gold.sum()) in (0, len(arm.gold)):
        return 0.0
    return float(roc_auc_score(arm.gold, arm.scores))


def _ap(arm: ArmScores) -> float:
    """Average precision of one arm; 0.0 when gold has a single class."""
    if int(arm.gold.sum()) in (0, len(arm.gold)):
        return 0.0
    return float(average_precision_score(arm.gold, arm.scores))


def _print_limit_warning(limit: int | None) -> None:
    """Emit a prominent SMOKE banner when a ``--limit`` was active."""
    if limit is None:
        return
    print(
        f"> ⚠️ **SMOKE RUN — `--limit {limit}` was active.** Only {limit} TEST "
        "pairs were scored; every separation metric below is from a truncated "
        "set and is NOT a real finding. Re-run WITHOUT `--limit` for the "
        "production report.\n"
    )


def _print_sweep_table(rows: list[SweepRow]) -> None:
    """Print one arm's threshold sweep as a markdown table."""
    print("| threshold | precision | recall | F1 |")
    print("|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.threshold:.2f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} |"
        )
    print()


def _print_dump(title: str, items: list[Disagreement]) -> None:
    """Print up to :data:`_DUMP_CAP` disagreements as a markdown table."""
    print(f"### {title} ({len(items)} total)\n")
    if not items:
        print("_None._\n")
        return
    shown = items[:_DUMP_CAP]
    print("| score | low? | marc_control_id | nypl_uuid | MARC title | CCE title |")
    print("|---:|:--:|:---|:---|:---|:---|")
    for item in shown:
        low = "⚠️" if item.score < _LOW_REGION else ""
        marc_title = item.marc_title.replace("|", "\\|")
        cce_title = item.cce_title.replace("|", "\\|")
        print(
            f"| {item.score:.4f} | {low} | {item.marc_control_id} | "
            f"{item.nypl_uuid} | {marc_title} | {cce_title} |"
        )
    if len(items) > _DUMP_CAP:
        print(f"\n_Truncated: {len(items) - _DUMP_CAP} more not shown._")
    print()


def _print_report(result: SeparationResult) -> None:
    """Emit the full markdown report to stdout."""
    print("# Wild-distribution separation test — 2026-06-17\n")
    _print_limit_warning(result.limit)
    print(
        "Issue #84. The FIRST off-vault SEPARATION check of the learned matcher. "
        "Prior separation numbers (learned grouped-OOF AUC "
        f"{_LEARNED_OOF_AUC:.3f}, weighted {_WEIGHTED_OOF_AUC:.3f}) are measured "
        "INSIDE the labeled, MATCH-biased vault. The reframe is that the "
        "matcher's scaling lever is pair-level SEPARATION — can a score "
        "threshold auto-decide a pair without a human — not the saturated top-1 "
        "linkage. The open worry is that learned separation might be worse on "
        "unknown pairs. This script answers it with a strict held-out split.\n"
    )

    print("## Method\n")
    print(
        f"**Time split by `labeled_at` at `{_CUTOFF}`.** TRAIN = every "
        "non-`unsure` pair labeled at or before the cutoff; TEST = every "
        "non-`unsure` pair labeled strictly after it (the wild, stratified, "
        "middle-heavy sample). TRAIN ∩ TEST is asserted empty, so the model can "
        "never train on a TEST pair.\n"
    )
    print(
        "**One labeled-only learned model trains on TRAIN only** — production "
        "`train-scorer` recipe, NO decoys (decoys were a top-1 fix; this is "
        "pair-level separation, so labeled-only matches the OOF-0.993 "
        f"baseline). Locked hyperparameters (max_depth={_MAX_DEPTH}, "
        f"num_leaves={_NUM_LEAVES}, min_data_in_leaf={_MIN_DATA_IN_LEAF}, "
        f"lambda_l2={_LAMBDA_L2}, n_estimators={_N_ESTIMATORS}, "
        f"class_weight={_CLASS_WEIGHT}, random_state={_RANDOM_STATE}). The "
        "Booster is wrapped directly as the production `LearnedCombiner` — no "
        "disk round-trip, the production `caches/learned_scorer.*` is untouched.\n"
    )
    print(
        "**Each TEST pair is scored once** into per-scorer Evidence; three "
        "combiner arms grade that same Evidence: `learned`, `weighted_mean`, "
        "and `weighted_minus_year` (weighted mean with `year_weight=0.0`, the "
        "remaining weights renormalized to sum to 1.0, to remove year's "
        "constant uplift). Gold = 1 for `match`, 0 for `no_match`.\n"
    )
    print(
        f"- **Feature count**: {result.feature_count} (production `feature_names()`)\n"
        f"- **TRAIN pairs**: {result.n_train} "
        f"(positive {result.n_train_positive}, negative {result.n_train_negative})\n"
        f"- **TEST pairs**: {result.n_test_total} labeled; "
        f"{result.n_test_scored} resolved and scored "
        f"(unresolved: no_marc={result.no_marc}, no_cce={result.no_cce}, "
        f"dropped={result.dropped})\n"
    )

    print("## Headline — held-out separation (TEST set)\n")
    _print_limit_warning(result.limit)
    print("| arm | ROC-AUC | average precision |")
    print("|:---|---:|---:|")
    for arm in result.arms:
        print(f"| {arm.name} | {_auc(arm):.4f} | {_ap(arm):.4f} |")
    print()

    learned_auc = _auc(result.arms[0])
    print("## Does separation hold off-vault?\n")
    print(
        "Known labeled-vault grouped-OOF baselines for contrast: "
        f"**learned {_LEARNED_OOF_AUC:.3f}**, **weighted "
        f"{_WEIGHTED_OOF_AUC:.3f}**.\n"
    )
    if learned_auc >= _LEARNED_OOF_AUC - _HOLD_BAND:
        print(
            f"**HOLDS.** Held-out learned AUC {learned_auc:.4f} is within "
            f"{_HOLD_BAND:.2f} of the {_LEARNED_OOF_AUC:.3f} OOF baseline: "
            "separation generalizes to the wild, middle-heavy sample, so "
            "threshold-based triage is viable as a scaling lever.\n"
        )
    else:
        print(
            f"**COLLAPSES.** Held-out learned AUC {learned_auc:.4f} falls more "
            f"than {_HOLD_BAND:.2f} below the {_LEARNED_OOF_AUC:.3f} OOF "
            "baseline: the in-vault separation does NOT generalize off-vault "
            "(the 'worse on unknown' worry is confirmed). Triage thresholds "
            "tuned on the vault cannot be trusted on wild pairs.\n"
        )

    print("## Threshold sweeps\n")
    for arm in result.arms:
        print(f"### {arm.name}\n")
        _print_sweep_table(result.sweeps[arm.name])
    print(
        f"**Best-F1 learned threshold**: {result.best_learned.threshold:.2f} "
        f"(precision {result.best_learned.precision:.4f}, recall "
        f"{result.best_learned.recall:.4f}, F1 {result.best_learned.f1:.4f}).\n"
    )

    print("## Disagreements at the best learned threshold\n")
    print(
        f"At the best-F1 learned threshold "
        f"({result.best_learned.threshold:.2f}). ⚠️ marks pairs in the "
        f"below-{_LOW_REGION:.2f} score region.\n"
    )
    _print_dump("False accepts (score ≥ threshold, gold = no_match)", result.false_accepts)
    _print_dump("False rejects (score < threshold, gold = match)", result.false_rejects)

    triage = result.triage
    print("## Triage viability (learned arm)\n")
    decided = triage.auto_accept + triage.auto_reject
    decided_pct = 100.0 * decided / triage.total if triage.total else 0.0
    residual_pct = 100.0 * triage.residual / triage.total if triage.total else 0.0
    print(
        f"- **Auto-ACCEPT** threshold T_hi = {triage.accept_threshold:.2f} "
        f"(zero false-accepts above it): {triage.auto_accept} pairs\n"
        f"- **Auto-REJECT** threshold T_lo = {triage.reject_threshold:.2f} "
        f"(zero false-rejects below it): {triage.auto_reject} pairs\n"
        f"- **Auto-decided**: {decided}/{triage.total} ({decided_pct:.1f}%)\n"
        f"- **Residual human middle**: {triage.residual}/{triage.total} "
        f"({residual_pct:.1f}%)\n"
    )
    print(
        "> **Caveat.** The TEST set is a STRATIFIED, deliberately middle-heavy "
        "sample, so the auto-decidable fraction here is a LOWER bound — it is "
        "NOT the production rate. Real acquired pairs skew toward the easy "
        "tails, where a far larger share auto-decides.\n"
    )

    print("## Reproduction\n")
    print("```")
    print("pdm run python scripts/separation_wild_test.py \\")
    print("    > docs/findings/separation_wild_test_2026-06-17.md")
    print("```")
    print(
        "\nThis script touches nothing under `src/` or `data/`, overwrites no "
        "artifact, and is deterministic (fixed seed, `n_jobs=1`).\n"
    )


def _parse_limit() -> int | None:
    """Parse the optional ``--limit N`` smoke flag from argv."""
    parser = ArgumentParser(description="Wild-distribution separation test (issue #84).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap the TEST pairs scored (smoke; report is flagged)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    limit: int | None = args.limit
    return limit


def main() -> None:
    """Run the wild separation test and print the markdown report to stdout."""
    limit = _parse_limit()
    _progress(f"script started (limit={limit})")
    entries = current_entries(_VAULT_PATH)
    result = run_separation(entries, limit)
    _progress(
        f"done: scored={result.n_test_scored} learned_auc={_auc(result.arms[0]):.4f} "
        f"weighted_auc={_auc(result.arms[1]):.4f}"
    )
    _print_report(result)
    _progress("report written")


if __name__ == "__main__":
    main()
