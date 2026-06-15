"""Vault-driven matcher evaluator producing P/R, AUC, AP, and a sweep.

Replaces the prior CSV-driven driver. The ground truth is now the
project's append-only ``label_vault.jsonl``: every line is a human
verdict on one ``(marc_control_id, nypl_uuid)`` pair, and the latest
entry for a pair wins (the same "current_entries" semantics that
``build-queue`` already honors). The vault is the only source of
*labeled negatives* the project has, which is what makes
threshold-independent metrics (AUC, average precision) actually
computable; the prior ground-truth CSV held only confirmed matches.

Two passes share the same opened LMDB lookup and IDF table:

* **Pass A** — pair-level scoring. For every non-``unsure`` vault entry
  the (MARC, CCE) pair is materialized and scored via
  :func:`pd_groundtruth.vault_pair_resolver.make_pair_scorer`. The
  ``(calibrated_score, label)`` list feeds :func:`roc_auc`,
  :func:`average_precision`, and :func:`threshold_sweep` in the metrics
  module. ``label`` is ``1`` for ``match`` and ``0`` for ``no_match``.
* **Pass B** — per-MARC linkage P/R. For every MARC that has at least
  one current ``match`` verdict in the vault, the matcher's top
  prediction is compared against that verdict's ``nypl_uuid``. Counts
  collapse into precision (correct top / had a top) and recall
  (correct top / had ground truth).

The vault is small (~300 entries today), so the eval runs single-
process. The prior spawn-pool plumbing was dropped: it added pickle and
lifecycle complexity in service of an optimization the corpus does not
need. If the vault ever grows past the point where a sequential scan is
painful, parallelism can be reintroduced — but that is a separate
ticket, not a permanent shape requirement.
"""

from logging import getLogger
from pathlib import Path
from time import perf_counter

from msgspec import Struct

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.eval.metrics import ThresholdPoint
from pd_matcher.eval.metrics import average_precision
from pd_matcher.eval.metrics import roc_auc
from pd_matcher.eval.metrics import threshold_sweep
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)

_VERDICT_MATCH: str = "match"
_VERDICT_UNSURE: str = "unsure"


class EvalReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of one :func:`run_eval` invocation.

    Vault entry counts (``pairs_*``) drive AUC/AP/sweep; per-MARC
    counts (``marcs_*``) drive precision and recall the same way today's
    gate already reads them. ``threshold_sweep`` is reported but never
    gated — it exists for plotting and for picking a future threshold.
    """

    pairs_evaluated: int
    pairs_positive: int
    pairs_negative: int
    pairs_unsure_excluded: int
    marcs_evaluated: int
    marcs_with_matcher_top: int
    marcs_with_correct_top: int
    precision: float
    recall: float
    f1: float
    auc_roc: float
    average_precision: float
    threshold_sweep: tuple[ThresholdPoint, ...]
    elapsed_seconds: float


def _safe_division(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator`` or ``0.0`` when the denominator is zero."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; ``0.0`` when both are zero."""
    total = precision + recall
    if total <= 0.0:
        return 0.0
    return 2.0 * precision * recall / total


def _partition_entries(
    entries: dict[tuple[str, str], VaultEntry],
) -> tuple[list[VaultEntry], int]:
    """Drop ``unsure`` entries; return the rest plus the dropped count."""
    keep: list[VaultEntry] = []
    unsure = 0
    for entry in entries.values():
        if entry.verdict == _VERDICT_UNSURE:
            unsure += 1
            continue
        keep.append(entry)
    return keep, unsure


def _ground_truth_by_marc(
    entries: dict[tuple[str, str], VaultEntry],
) -> dict[str, str]:
    """Return a ``marc_control_id -> nypl_uuid`` map for current ``match`` verdicts.

    ``current_entries`` already collapses the append-only log to one
    entry per ``(marc, uuid)`` pair (latest verdict wins). A MARC may
    still have several entries — one per distinct UUID it has been
    paired with — so this helper keeps only the ``match`` rows. Two
    distinct UUIDs both currently labeled ``match`` for the same MARC
    is a labeling anomaly; the last one encountered while iterating the
    insertion-ordered dict wins. The vault itself is unchanged either
    way.
    """
    gt: dict[str, str] = {}
    for entry in entries.values():
        if entry.verdict != _VERDICT_MATCH:
            continue
        gt[entry.marc_control_id] = entry.nypl_uuid
    return gt


def _run_pass_a(
    entries: list[VaultEntry],
    *,
    marc_by_id: dict[str, MarcRecord],
    lookup: NyplIndexLookup,
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
    pairings: CompiledPairings,
    matching_config: MatchingConfig,
    calibrator: PlattCalibrator | None,
    learned_model_dir: Path | None,
) -> tuple[list[tuple[float, int]], int, int, int, int]:
    """Score every kept vault entry; return scored pairs plus skip totals.

    Returns ``(scored_labels, positives, negatives, missing_in_pool,
    missing_in_index)``. ``positives + negatives == len(scored_labels)``;
    the skip totals account for the rest of ``entries`` whose MARC or
    CCE no longer resolve.
    """
    score_pair = make_pair_scorer(
        matching_config=matching_config,
        pairings=pairings,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        calibrator=calibrator,
        learned_model_dir=learned_model_dir,
    )
    scored: list[tuple[float, int]] = []
    positives = 0
    negatives = 0
    missing_in_pool = 0
    missing_in_index = 0
    for entry in entries:
        marc = marc_by_id.get(entry.marc_control_id)
        if marc is None:
            missing_in_pool += 1
            _LOGGER.warning(
                "eval.vault.marc_not_in_pool marc_control_id=%s nypl_uuid=%s",
                entry.marc_control_id,
                entry.nypl_uuid,
            )
            continue
        cce = lookup.get_registration(entry.nypl_uuid)
        if cce is None:
            missing_in_index += 1
            _LOGGER.warning(
                "eval.vault.cce_not_in_index marc_control_id=%s nypl_uuid=%s",
                entry.marc_control_id,
                entry.nypl_uuid,
            )
            continue
        candidate = score_pair(marc, cce)
        score = candidate.combined.calibrated
        label = 1 if entry.verdict == _VERDICT_MATCH else 0
        if label == 1:
            positives += 1
        else:
            negatives += 1
        scored.append((score, label))
    return scored, positives, negatives, missing_in_pool, missing_in_index


def _run_pass_b(
    ground_truth: dict[str, str],
    *,
    marc_by_id: dict[str, MarcRecord],
    lookup: NyplIndexLookup,
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
    pairings: CompiledPairings,
    matching_config: MatchingConfig,
    calibrator: PlattCalibrator | None,
    learned_model_dir: Path | None,
) -> tuple[int, int, int]:
    """Run per-MARC linkage; return ``(evaluated, with_top, correct_top)``."""
    combiner = build_combiner(matching_config, learned_model_dir=learned_model_dir)
    marcs_evaluated = 0
    marcs_with_matcher_top = 0
    marcs_with_correct_top = 0
    for marc_id, gt_uuid in ground_truth.items():
        marc = marc_by_id.get(marc_id)
        if marc is None:
            _LOGGER.warning(
                "eval.vault.gt_marc_not_in_pool marc_control_id=%s nypl_uuid=%s",
                marc_id,
                gt_uuid,
            )
            continue
        marcs_evaluated += 1
        result = match_record(
            marc,
            lookup=lookup,
            config=matching_config,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            calibrator=calibrator,
            combiner=combiner,
            pairings=pairings,
        )
        if result.best is None:
            continue
        marcs_with_matcher_top += 1
        if result.best.nypl_uuid == gt_uuid:
            marcs_with_correct_top += 1
    return marcs_evaluated, marcs_with_matcher_top, marcs_with_correct_top


def run_eval(
    *,
    vault_path: Path,
    pool_path: Path,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    calibrator: PlattCalibrator | None = None,
    learned_model_dir: Path | None = None,
) -> EvalReport:
    """Evaluate the matcher pipeline against the vault.

    Args:
        vault_path: Append-only JSONL label vault
            (``data/label_vault.jsonl``).
        pool_path: Root of the gitignored MARC candidate pool
            (``data/candidates``); shards live under
            ``<pool>/<lang>/*.xml``.
        index_path: LMDB env produced by ``pd-matcher index build``.
        matching_config: Active :class:`MatchingConfig`. The
            ``min_combined_score`` floor is applied during Pass B exactly
            as the production matcher applies it.
        pairing_config: Active :class:`PairingConfig`; compiled once
            and reused across both passes.
        calibrator: Optional Platt calibrator threaded through both
            passes; ``None`` uses the linear pass-through (raw / 100)
            from :class:`~pd_matcher.match.combiners.weighted_mean.WeightedMeanCombiner`.
        learned_model_dir: Directory holding the learned-model artifact when
            ``matching_config.scorer == "learned"``; threaded into both
            passes. ``None`` on the default weighted-mean path.

    Returns:
        A populated :class:`EvalReport`.
    """
    started = perf_counter()
    raw = current_entries(vault_path)
    kept_entries, unsure = _partition_entries(raw)
    ground_truth = _ground_truth_by_marc(raw)
    needed_marc_ids = {entry.marc_control_id for entry in kept_entries} | set(ground_truth)
    marc_by_id = build_marc_index(pool_path, needed_marc_ids)
    pairings = compile_pairings(pairing_config)
    with NyplIndexLookup(index_path) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
        scored, positives, negatives, missing_pool, missing_index = _run_pass_a(
            kept_entries,
            marc_by_id=marc_by_id,
            lookup=lookup,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            pairings=pairings,
            matching_config=matching_config,
            calibrator=calibrator,
            learned_model_dir=learned_model_dir,
        )
        marcs_evaluated, marcs_with_top, marcs_correct = _run_pass_b(
            ground_truth,
            marc_by_id=marc_by_id,
            lookup=lookup,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            pairings=pairings,
            matching_config=matching_config,
            calibrator=calibrator,
            learned_model_dir=learned_model_dir,
        )
    pairs_evaluated = positives + negatives
    if scored:
        auc = roc_auc(scored)
        ap = average_precision(scored)
        sweep = threshold_sweep(scored)
    else:
        auc = 0.0
        ap = 0.0
        sweep = ()
    precision = _safe_division(marcs_correct, marcs_with_top)
    recall = _safe_division(marcs_correct, marcs_evaluated)
    if missing_pool or missing_index:
        _LOGGER.info(
            "eval.vault.skipped missing_in_pool=%d missing_in_index=%d",
            missing_pool,
            missing_index,
        )
    return EvalReport(
        pairs_evaluated=pairs_evaluated,
        pairs_positive=positives,
        pairs_negative=negatives,
        pairs_unsure_excluded=unsure,
        marcs_evaluated=marcs_evaluated,
        marcs_with_matcher_top=marcs_with_top,
        marcs_with_correct_top=marcs_correct,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        auc_roc=auc,
        average_precision=ap,
        threshold_sweep=sweep,
        elapsed_seconds=perf_counter() - started,
    )


__all__ = [
    "EvalReport",
    "run_eval",
]
