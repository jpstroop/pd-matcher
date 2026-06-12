"""One-shot backfill of vault-labeled pairs that are missing from a review DB.

The label vault is the durable source of truth for human verdicts. Since the
``build-queue`` carryover fix (jpstroop/pd-matcher#33), the routine rebuild
already includes every vault MARC that's still in the candidate pool, so this
command is rarely needed in normal operation. It remains available as a
recovery tool for cases where a queue was built without vault carryover (for
example via a future ``--no-vault`` flag) or where the vault was modified out
of band after a build.

This module backfills missing pairs into an existing ``review.db``: it trusts
the vault (the matcher would not necessarily have proposed this exact pair
from scratch), looks the MARC record up in the candidate pool, looks the CCE
registration up in the LMDB index, scores the *specific* pair using the
matcher's per-pair scoring routine so the row carries a real
``(score, band, evidence)``, and inserts both the ``review_pair`` row and the
existing vault label with the original ``labeled_at``.
"""

from collections.abc import Callable
from logging import getLogger
from pathlib import Path

from msgspec import Struct

from pd_groundtruth.build_queue import _build_pair_insert
from pd_groundtruth.build_queue import _language_of
from pd_groundtruth.build_queue import _load_calibrator
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import band_of
from pd_groundtruth.vault_pair_resolver import IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import CceLookupFn
from pd_groundtruth.vault_pair_resolver import MarcLookupFn
from pd_groundtruth.vault_pair_resolver import ScorePairFn
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_groundtruth.vault_pair_resolver import resolve_vault_pairs
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

BuildPairFn = Callable[[MarcRecord, IndexedNyplRegRecord, CandidateMatch], PairInsert]

_LOGGER = getLogger(__name__)


class BackfillSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`run_backfill` and :func:`vault_into_queue`."""

    backfilled: int
    already_present: int
    missing_in_pool: int
    missing_in_index: int


def _default_build_pair() -> BuildPairFn:
    """Build the default closure that projects scored vault pairs into rows."""

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


def run_backfill(
    *,
    db_path: Path,
    vault: dict[tuple[str, str], VaultEntry],
    marc_lookup: MarcLookupFn,
    cce_lookup: CceLookupFn,
    score_pair: ScorePairFn,
    build_pair: BuildPairFn | None = None,
) -> BackfillSummary:
    """Open ``db_path``, compute the missing set, and backfill it.

    Pure orchestration: every external dependency (MARC pool, CCE index,
    scoring pipeline) is injected so tests can drive the backfill without
    standing up LMDB or a real shard tree.

    Args:
        db_path: Existing ``review.db`` (will be modified in place).
        vault: Latest :class:`VaultEntry` per ``(marc_control_id, nypl_uuid)``
            (as returned by :func:`current_entries`).
        marc_lookup: ``control_id -> MarcRecord | None`` resolver.
        cce_lookup: ``nypl_uuid -> IndexedNyplRegRecord | None`` resolver.
        score_pair: ``(marc, cce) -> CandidateMatch`` scorer.
        build_pair: ``(marc, cce, candidate) -> PairInsert`` projection.
            Defaults to the standard projection; tests can inject a
            simpler builder.
    """
    project = build_pair if build_pair is not None else _default_build_pair()
    with ReviewDb.connect(db_path) as db:
        existing = db.pair_keys()
        missing = {key: entry for key, entry in vault.items() if key not in existing}
        already_present = len(vault) - len(missing)
        if not missing:
            _LOGGER.info(
                "vault.backfill nothing-missing vault_total=%d already_present=%d",
                len(vault),
                already_present,
            )
            return BackfillSummary(
                backfilled=0,
                already_present=already_present,
                missing_in_pool=0,
                missing_in_index=0,
            )
        resolved, summary = resolve_vault_pairs(
            vault=missing,
            marc_lookup=marc_lookup,
            cce_lookup=cce_lookup,
            score_pair=score_pair,
            build_pair=project,
        )
        for resolved_pair in resolved:
            pair_id = db.insert_pair(resolved_pair.pair)
            db.insert_existing_label(
                pair_id=pair_id,
                verdict=resolved_pair.entry.verdict,
                labeled_at=resolved_pair.entry.labeled_at,
                note=resolved_pair.entry.note,
                categories=resolved_pair.entry.categories,
            )
    _LOGGER.info(
        "vault.backfill complete backfilled=%d already_present=%d "
        "missing_in_pool=%d missing_in_index=%d",
        summary.resolved,
        already_present,
        summary.missing_in_pool,
        summary.missing_in_index,
    )
    return BackfillSummary(
        backfilled=summary.resolved,
        already_present=already_present,
        missing_in_pool=summary.missing_in_pool,
        missing_in_index=summary.missing_in_index,
    )


def vault_into_queue(
    *,
    db_path: Path,
    vault_path: Path,
    pool_path: Path,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
) -> BackfillSummary:
    """Backfill missing vault pairs into ``db_path``.

    End-to-end entry point used by the ``vault-into-queue`` Typer command.
    Loads the vault, computes the missing set, opens the LMDB lookup and IDF/
    calibrator caches the same way ``build-queue`` does, materializes the
    needed MARC records by streaming the pool shards once, and inserts one
    ``review_pair`` row plus one ``label`` row per resolvable missing entry.

    Args:
        db_path: Existing ``review.db``.
        vault_path: JSONL label vault.
        pool_path: Root directory whose ``<lang>/*.xml`` shards form the pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        matching_config: Active matcher config (provides scorer weights).
        pairing_config: Active field-pairing config.
    """
    vault = current_entries(vault_path)
    if not vault:
        _LOGGER.info("vault.backfill empty-vault path=%s", vault_path)
        return BackfillSummary(
            backfilled=0,
            already_present=0,
            missing_in_pool=0,
            missing_in_index=0,
        )

    needed_marc_ids = {marc_id for marc_id, _uuid in vault}
    marc_by_id = build_marc_index(pool_path, needed_marc_ids)
    _LOGGER.info(
        "vault.backfill pool-scanned wanted=%d resolved=%d",
        len(needed_marc_ids),
        len(marc_by_id),
    )

    idf_cache_path = index_path.parent / IDF_CACHE_NAME
    idf = load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index_path))
    calibrator = _load_calibrator(index_path.parent)
    pairings = compile_pairings(pairing_config)
    score_pair = _make_pair_scorer(
        matching_config=matching_config,
        pairings=pairings,
        idf=idf,
        calibrator=calibrator,
        learned_model_dir=index_path.parent,
    )

    with NyplIndexLookup(index_path) as lookup:
        return run_backfill(
            db_path=db_path,
            vault=vault,
            marc_lookup=marc_by_id.get,
            cce_lookup=lookup.get_registration,
            score_pair=score_pair,
        )


def _make_pair_scorer(
    *,
    matching_config: MatchingConfig,
    pairings: CompiledPairings,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
    learned_model_dir: Path | None,
) -> ScorePairFn:
    """Local indirection so tests can monkey-patch the scorer factory.

    Delegates to :func:`pd_groundtruth.vault_pair_resolver.make_pair_scorer`.
    """
    return make_pair_scorer(
        matching_config=matching_config,
        pairings=pairings,
        idf=idf,
        calibrator=calibrator,
        learned_model_dir=learned_model_dir,
    )


__all__ = [
    "BackfillSummary",
    "build_marc_index",
    "run_backfill",
    "vault_into_queue",
]
