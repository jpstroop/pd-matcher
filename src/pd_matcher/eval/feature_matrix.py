"""Build a labeled feature matrix from the vault for ML-style diagnostics.

This module is a research instrument, not part of the production pipeline.
It loads every non-``unsure`` vault entry, resolves each pair to its MARC
record (via the candidate pool) and CCE registration (via the LMDB index),
runs the existing scoring pipeline through :func:`pd_matcher.match.pipeline.
_score_candidate`, and extracts a fixed-shape numeric feature vector from the
resulting :class:`Evidence` stream — one row per labeled pair.

The returned matrix is intended as input to a small gradient-boosted model
(LightGBM) trained as a *diagnostic instrument*: feature importance,
SHAP-style contributions, and the gap between LightGBM's predicted
probability and the production combined score per pair. None of those
quantities are deployed; they exist to inform the next round of hand-tuning
and the question "is the weighted-mean combiner leaving signal on the
table?".

Feature columns are emitted in a deterministic order: one normalized score
per scorer followed by one ``_skipped`` flag per scorer. The ordering is
exposed alongside the matrix so callers (notably the diagnostic script and
its tests) can rename and rank columns without hardcoding indices.
"""

from logging import getLogger
from pathlib import Path

from msgspec import Struct
from numpy import asarray
from numpy import float64
from numpy import int64
from numpy import zeros
from numpy.typing import NDArray

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)

_VERDICT_MATCH: str = "match"
_VERDICT_UNSURE: str = "unsure"

SCORER_ORDER: tuple[str, ...] = (
    "title.token_set",
    "name.author",
    "name.publisher",
    "year.delta",
    "edition.compat",
    "lccn.exact",
    "isbn.exact",
    "extent.page_count",
    "volume.compat",
)


class FeatureMatrixRow(Struct, frozen=True, forbid_unknown_fields=True):
    """One labeled row of the feature matrix with provenance for inspection.

    ``feature_values`` is the per-pair feature vector in the same order as
    :func:`feature_column_names`. The MARC and CCE title projections are kept
    short so the disagreement table in the findings document fits on a line.
    """

    pair_id: int
    marc_control_id: str
    nypl_uuid: str
    verdict: str
    combined_score: float
    marc_title: str
    cce_title: str
    feature_values: tuple[float, ...]


def feature_column_names() -> list[str]:
    """Return the deterministic feature column order.

    The first ``len(SCORER_ORDER)`` columns are normalized scorer scores in
    ``SCORER_ORDER``; the next ``len(SCORER_ORDER)`` columns are the matching
    ``_skipped`` flags (``1.0`` when the scorer reported ``skipped``, else
    ``0.0``). Splitting the flag out as its own feature lets a tree-based
    model learn "score X means something different when it was actually
    computed vs. when the scorer fell back to its skipped default".
    """
    names: list[str] = []
    for scorer in SCORER_ORDER:
        names.append(scorer)
    for scorer in SCORER_ORDER:
        names.append(f"{scorer}__skipped")
    return names


def _partition_entries(
    entries: dict[tuple[str, str], VaultEntry],
) -> list[VaultEntry]:
    """Drop ``unsure`` entries; preserve insertion order for determinism."""
    kept: list[VaultEntry] = []
    for entry in entries.values():
        if entry.verdict == _VERDICT_UNSURE:
            continue
        kept.append(entry)
    return kept


def _evidence_by_scorer(evidences: tuple[Evidence, ...]) -> dict[str, Evidence]:
    """Index winning Evidence by scorer name for O(1) lookup."""
    return {evidence.scorer: evidence for evidence in evidences}


def _feature_row(evidences: tuple[Evidence, ...]) -> tuple[float, ...]:
    """Project the winning-Evidence tuple into a fixed-shape feature vector.

    The pipeline emits exactly one Evidence per scorer name in
    :data:`SCORER_ORDER`, so the lookup is keyed by name. A missing scorer
    is a pipeline bug and surfaces as a :class:`KeyError` rather than being
    silently papered over with sentinel zeros — a diagnostic feature matrix
    that lies about which scorers ran would be worse than failing loudly.
    """
    by_name = _evidence_by_scorer(evidences)
    scores: list[float] = []
    flags: list[float] = []
    for scorer in SCORER_ORDER:
        evidence = by_name[scorer]
        scores.append(evidence.normalized)
        flags.append(1.0 if evidence.skipped else 0.0)
    return tuple(scores + flags)


def _marc_title_for_row(marc: MarcRecord) -> str:
    """Return the MARC main title (245a) for the disagreement-table column.

    ``title_main`` is required by the MARC parser: any record reaching this
    function has a non-empty 245a, so no fallback to the composite
    ``title`` field is needed.
    """
    return marc.title_main


def extract_feature_matrix(
    *,
    vault_path: Path,
    pool_path: Path,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
) -> tuple[NDArray[float64], NDArray[int64], list[str], tuple[FeatureMatrixRow, ...]]:
    """Build the labeled feature matrix from the vault and the index.

    Args:
        vault_path: Append-only JSONL label vault (same path
            :func:`pd_matcher.eval.ground_truth.run_eval` consumes).
        pool_path: Root of the gitignored MARC candidate pool with
            ``<pool>/<lang>/*.xml`` shards.
        index_path: LMDB env directory produced by
            ``pd-matcher index build``.
        matching_config: Active :class:`MatchingConfig`; controls scorer
            behavior and the year window.
        pairing_config: Active :class:`PairingConfig`; compiled once.

    Returns:
        ``(X, y, feature_names, rows)`` where ``X`` is shape ``(n, k)`` of
        ``float64`` features, ``y`` is shape ``(n,)`` of ``int64`` labels
        (``1`` for ``match``, ``0`` for ``no_match``), ``feature_names``
        is the column order, and ``rows`` is parallel to ``X``/``y`` with
        provenance for disagreement tables. Skipped entries (MARC absent
        from pool, CCE absent from index) emit a WARNING and are excluded
        from ``X``/``y``.
    """
    raw = current_entries(vault_path)
    kept = _partition_entries(raw)
    needed_marc_ids = {entry.marc_control_id for entry in kept}
    marc_by_id = build_marc_index(pool_path, needed_marc_ids)
    pairings = compile_pairings(pairing_config)
    column_names = feature_column_names()

    rows: list[FeatureMatrixRow] = []
    labels: list[int] = []
    with NyplIndexLookup(index_path) as lookup:
        idf = build_idf_table(lookup)
        score_pair = make_pair_scorer(
            matching_config=matching_config,
            pairings=pairings,
            idf=idf,
            calibrator=None,
        )
        for pair_id, entry in enumerate(kept):
            marc = marc_by_id.get(entry.marc_control_id)
            if marc is None:
                _LOGGER.warning(
                    "eval.vault.marc_not_in_pool marc_control_id=%s nypl_uuid=%s",
                    entry.marc_control_id,
                    entry.nypl_uuid,
                )
                continue
            cce = lookup.get_registration(entry.nypl_uuid)
            if cce is None:
                _LOGGER.warning(
                    "eval.vault.cce_not_in_index marc_control_id=%s nypl_uuid=%s",
                    entry.marc_control_id,
                    entry.nypl_uuid,
                )
                continue
            candidate = score_pair(marc, cce)
            feature_values = _feature_row(candidate.evidence)
            rows.append(
                FeatureMatrixRow(
                    pair_id=pair_id,
                    marc_control_id=entry.marc_control_id,
                    nypl_uuid=entry.nypl_uuid,
                    verdict=entry.verdict,
                    combined_score=candidate.combined.calibrated,
                    marc_title=_marc_title_for_row(marc),
                    cce_title=cce.title,
                    feature_values=feature_values,
                )
            )
            labels.append(1 if entry.verdict == _VERDICT_MATCH else 0)

    if not rows:
        empty_x = zeros((0, len(column_names)), dtype=float64)
        empty_y = zeros((0,), dtype=int64)
        return empty_x, empty_y, column_names, ()

    matrix = asarray([row.feature_values for row in rows], dtype=float64)
    label_array = asarray(labels, dtype=int64)
    return matrix, label_array, column_names, tuple(rows)


__all__ = [
    "SCORER_ORDER",
    "FeatureMatrixRow",
    "extract_feature_matrix",
    "feature_column_names",
]
