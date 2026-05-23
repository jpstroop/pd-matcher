"""One-shot backfill of vault-labeled pairs that are missing from a review DB.

The label vault is the durable source of truth for human verdicts. The working
``review.db`` is rebuilt by ``build-queue`` from a fresh stratified sample, so a
vault entry whose ``(marc_control_id, nypl_uuid)`` pair was not sampled this
round becomes invisible in the review UI even though the verdict is preserved
on disk.

This module backfills those missing pairs into an existing ``review.db``: it
trusts the vault (the matcher would not necessarily have proposed this exact
pair from scratch), looks the MARC record up in the candidate pool, looks the
CCE registration up in the LMDB index, scores the *specific* pair using the
matcher's per-pair scoring routine so the row carries a real
``(score, band, evidence)``, and inserts both the ``review_pair`` row and the
existing vault label with the original ``labeled_at``.

A durable fix that always preserves vault MARCs through ``build-queue``
sampling is tracked separately (see ``jpstroop/pd-matcher#33``); this CLI is a
tactical unblock for review sessions in the meantime.
"""

from collections.abc import Callable
from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

from msgspec import Struct
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.pipeline import _score_candidate
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.parsers.marc import iter_marc_records

from pd_groundtruth.build_queue import _build_pair_insert
from pd_groundtruth.build_queue import _language_of
from pd_groundtruth.build_queue import _load_calibrator
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import band_of

_LOGGER = getLogger(__name__)

_IDF_CACHE_NAME: str = "idf.msgpack"

ScorePairFn = Callable[[MarcRecord, IndexedNyplRegRecord], CandidateMatch]
MarcLookupFn = Callable[[str], MarcRecord | None]
CceLookupFn = Callable[[str], IndexedNyplRegRecord | None]


class BackfillSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`run_backfill` and :func:`vault_into_queue`."""

    backfilled: int
    already_present: int
    missing_in_pool: int
    missing_in_index: int


def _iter_pool_shards(pool: Path) -> Iterator[Path]:
    """Yield ``<lang>/*.xml`` shards under ``pool`` in deterministic order."""
    for language_dir in sorted(pool.iterdir()):
        if not language_dir.is_dir():
            continue
        yield from sorted(language_dir.glob("*.xml"))


def build_marc_index(pool: Path, wanted: set[str]) -> dict[str, MarcRecord]:
    """Return a ``control_id -> MarcRecord`` map for every ``wanted`` id in ``pool``.

    Streams each shard once with the existing :func:`iter_marc_records` parser,
    keeping only records whose ``control_id`` is in ``wanted`` so memory stays
    bounded by the size of the missing set rather than the pool. Stops scanning
    early once every wanted id has been resolved.

    Args:
        pool: Root directory whose ``<lang>/*.xml`` shards form the candidate
            pool (mirrors ``build-queue --pool``).
        wanted: The MARC control ids to materialize.

    Returns:
        A dict with one entry per resolved id; missing ids are simply absent.
    """
    if not wanted:
        return {}
    found: dict[str, MarcRecord] = {}
    remaining = set(wanted)
    for shard in _iter_pool_shards(pool):
        for record in iter_marc_records(shard):
            if record.control_id in remaining:
                found[record.control_id] = record
                remaining.discard(record.control_id)
                if not remaining:
                    return found
    return found


def _make_pair_scorer(
    *,
    matching_config: MatchingConfig,
    pairings: CompiledPairings,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
) -> ScorePairFn:
    """Bind the matcher's per-pair scoring routine into a one-arg callable.

    Reuses :func:`pd_matcher.match.pipeline._score_candidate` so the rebuilt
    rows carry the *same* evidence/scores the production matcher would emit if
    it ever proposed the pair (the matcher's candidate retrieval wouldn't
    necessarily surface it from scratch, which is exactly why the vault has to
    be honored verbatim here).
    """
    combiner = WeightedMeanCombiner(config=matching_config)

    def scorer(marc: MarcRecord, candidate: IndexedNyplRegRecord) -> CandidateMatch:
        ctx = _build_context(marc, idf, matching_config)
        return _score_candidate(marc, candidate, ctx, combiner, calibrator, pairings)

    return scorer


def _backfill_missing(
    *,
    db: ReviewDb,
    missing: dict[tuple[str, str], VaultEntry],
    marc_lookup: MarcLookupFn,
    cce_lookup: CceLookupFn,
    score_pair: ScorePairFn,
) -> BackfillSummary:
    """Persist one row + label per resolvable missing vault entry.

    Skips with a WARNING when the MARC record is no longer in the pool (e.g.
    purged by an updated filter or removed upstream) or when the CCE record is
    no longer in the index (NYPL re-issue or index rebuild). Either skip leaves
    the vault entry untouched on disk so it can still be reconciled later.
    """
    backfilled = 0
    missing_in_pool = 0
    missing_in_index = 0
    for (marc_id, nypl_uuid), entry in missing.items():
        marc = marc_lookup(marc_id)
        if marc is None:
            missing_in_pool += 1
            _LOGGER.warning(
                "vault.marc_not_in_pool marc_control_id=%s nypl_uuid=%s",
                marc_id,
                nypl_uuid,
            )
            continue
        cce = cce_lookup(nypl_uuid)
        if cce is None:
            missing_in_index += 1
            _LOGGER.warning(
                "vault.cce_not_in_index marc_control_id=%s nypl_uuid=%s",
                marc_id,
                nypl_uuid,
            )
            continue
        candidate = score_pair(marc, cce)
        score = candidate.combined.calibrated
        pair = _build_pair_insert(
            marc,
            cce,
            candidate.evidence,
            language=_language_of(marc),
            score=score,
            band=band_of(score),
            source=SOURCE_BANDED,
        )
        pair_id = db.insert_pair(pair)
        db.insert_existing_label(
            pair_id=pair_id,
            verdict=entry.verdict,
            labeled_at=entry.labeled_at,
            note=entry.note,
            reasons=entry.reasons,
        )
        backfilled += 1
    return BackfillSummary(
        backfilled=backfilled,
        already_present=0,
        missing_in_pool=missing_in_pool,
        missing_in_index=missing_in_index,
    )


def run_backfill(
    *,
    db_path: Path,
    vault: dict[tuple[str, str], VaultEntry],
    marc_lookup: MarcLookupFn,
    cce_lookup: CceLookupFn,
    score_pair: ScorePairFn,
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
    """
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
        partial = _backfill_missing(
            db=db,
            missing=missing,
            marc_lookup=marc_lookup,
            cce_lookup=cce_lookup,
            score_pair=score_pair,
        )
    _LOGGER.info(
        "vault.backfill complete backfilled=%d already_present=%d "
        "missing_in_pool=%d missing_in_index=%d",
        partial.backfilled,
        already_present,
        partial.missing_in_pool,
        partial.missing_in_index,
    )
    return BackfillSummary(
        backfilled=partial.backfilled,
        already_present=already_present,
        missing_in_pool=partial.missing_in_pool,
        missing_in_index=partial.missing_in_index,
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

    idf_cache_path = index_path.parent / _IDF_CACHE_NAME
    idf = load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index_path))
    calibrator = _load_calibrator(index_path.parent)
    pairings = compile_pairings(pairing_config)
    score_pair = _make_pair_scorer(
        matching_config=matching_config,
        pairings=pairings,
        idf=idf,
        calibrator=calibrator,
    )

    with NyplIndexLookup(index_path) as lookup:
        return run_backfill(
            db_path=db_path,
            vault=vault,
            marc_lookup=marc_by_id.get,
            cce_lookup=lookup.get_registration,
            score_pair=score_pair,
        )


__all__ = [
    "BackfillSummary",
    "build_marc_index",
    "run_backfill",
    "vault_into_queue",
]
