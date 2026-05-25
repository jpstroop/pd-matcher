"""Build-queue-side glue around :mod:`pd_groundtruth.vault_pair_resolver`.

Resolves every current vault entry into a ready-to-insert pair *before* the
matcher runs so :func:`pd_groundtruth.build_queue.build_queue` can hand the
pre-scored pairs to its writer (which inserts them outside the per-stratum
caps) and exclude the vault MARC records from the per-language reservoir.
Kept separate from :mod:`pd_groundtruth.build_queue` to keep that module
focused on the writer + orchestration.
"""

from collections.abc import Callable
from logging import getLogger
from pathlib import Path

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import band_of
from pd_groundtruth.vault_pair_resolver import ResolvedVaultPair
from pd_groundtruth.vault_pair_resolver import ResolveSummary
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_groundtruth.vault_pair_resolver import resolve_vault_pairs
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)


def _make_vault_pair_builder() -> Callable[
    [MarcRecord, IndexedNyplRegRecord, CandidateMatch], PairInsert
]:
    """Build a closure that projects scored vault pairs into ``PairInsert`` rows.

    Returning a closure keeps
    :func:`pd_groundtruth.vault_pair_resolver.resolve_vault_pairs` signature
    unchanged and lets the builder import :func:`_build_pair_insert` lazily
    to dodge the circular dependency on :mod:`pd_groundtruth.build_queue`.
    """
    from pd_groundtruth.build_queue import _build_pair_insert
    from pd_groundtruth.build_queue import _language_of

    def _build(
        marc: MarcRecord,
        cce: IndexedNyplRegRecord,
        candidate: CandidateMatch,
    ) -> PairInsert:
        score = candidate.combined.calibrated
        return _build_pair_insert(
            marc,
            cce,
            candidate.evidence,
            language=_language_of(marc),
            score=score,
            band=band_of(score),
            source=SOURCE_BANDED,
            evidence_sources=candidate.evidence_sources,
        )

    return _build


def _load_vault_filtered(
    vault_path: Path,
    requeue_verdicts: frozenset[str],
) -> dict[tuple[str, str], VaultEntry]:
    """Return ``current_entries(vault_path)`` minus any verdict in ``requeue_verdicts``.

    Drops entries whose verdict is in ``requeue_verdicts`` so those pairs
    re-enter the labeling queue on rebuild. Each requeue verdict logs at INFO
    with the dropped count (zero is logged too — re-queuing a verdict that
    has no entries in the vault is a no-op, not an error).
    """
    raw = current_entries(vault_path)
    if not requeue_verdicts:
        return raw
    dropped_by_verdict: dict[str, int] = dict.fromkeys(requeue_verdicts, 0)
    filtered: dict[tuple[str, str], VaultEntry] = {}
    for key, entry in raw.items():
        if entry.verdict in requeue_verdicts:
            dropped_by_verdict[entry.verdict] = dropped_by_verdict.get(entry.verdict, 0) + 1
            continue
        filtered[key] = entry
    for verdict in sorted(requeue_verdicts):
        _LOGGER.info(
            "vault.requeue verdict=%s dropped=%d",
            verdict,
            dropped_by_verdict.get(verdict, 0),
        )
    return filtered


def resolve_vault_for_build(
    *,
    vault_path: Path,
    pool: Path,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    requeue_verdicts: frozenset[str] = frozenset(),
) -> tuple[list[ResolvedVaultPair], ResolveSummary]:
    """Resolve every current vault entry into a ready-to-insert pair.

    Loads the vault, materializes each entry's MARC record by streaming the
    pool shards, opens the LMDB index briefly to look up the matching CCE
    registration, and scores the pair via the matcher's per-pair routine so
    the persisted row carries real ``(score, band, evidence)``. Entries whose
    MARC is no longer in the pool or whose CCE is no longer in the index are
    skipped with a WARNING; the vault file itself is never modified.

    When ``requeue_verdicts`` is non-empty, vault entries whose verdict
    matches are *not* pre-applied — they re-enter the labeling queue as if
    they had never been labeled. Default empty preserves today's behavior of
    carrying every vault verdict forward.

    Returns:
        ``(resolved, summary)`` where ``resolved`` is empty for an empty or
        missing vault, and ``summary`` carries diagnostic counts that
        ``build_queue`` logs at the start of the run.
    """
    vault = _load_vault_filtered(vault_path, requeue_verdicts)
    if not vault:
        _LOGGER.info("vault loaded: 0 entries; 0 MARC records resolved from pool; 0 missing")
        return [], ResolveSummary(resolved=0, missing_in_pool=0, missing_in_index=0)
    needed_marc_ids = {marc_id for marc_id, _uuid in vault}
    marc_by_id = build_marc_index(pool, needed_marc_ids)
    pairings = compile_pairings(pairing_config)
    score_pair = make_pair_scorer(
        matching_config=matching_config,
        pairings=pairings,
        idf=idf,
        calibrator=calibrator,
    )
    with NyplIndexLookup(index_path) as lookup:
        build_pair = _make_vault_pair_builder()
        resolved, summary = resolve_vault_pairs(
            vault=vault,
            marc_lookup=marc_by_id.get,
            cce_lookup=lookup.get_registration,
            score_pair=score_pair,
            build_pair=build_pair,
        )
    _LOGGER.info(
        "vault loaded: %d entries; %d MARC records resolved from pool; %d missing from pool",
        len(vault),
        summary.resolved,
        summary.missing_in_pool,
    )
    if summary.missing_in_index:
        _LOGGER.info(
            "vault.cce_not_in_index count=%d (entries skipped this build)",
            summary.missing_in_index,
        )
    return resolved, summary


__all__ = [
    "resolve_vault_for_build",
]
