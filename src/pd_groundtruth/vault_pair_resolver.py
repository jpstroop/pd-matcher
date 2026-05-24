"""Shared helpers that materialize vault verdicts into matcher-scored pairs.

Both ``vault-into-queue`` (recovery) and ``build-queue`` (carryover) need the
same primitives: walk the candidate pool to find each vault entry's MARC
record, look the matching CCE registration up in the LMDB index, and run the
matcher's per-pair scoring routine to produce a ``CandidateMatch`` so the
resulting ``review_pair`` row carries real ``(score, band, evidence)``.

Keeping the primitives in one module guarantees the two callers stay in
lockstep: a vault entry resolved here looks identical no matter which command
fed it.
"""

from collections.abc import Callable
from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

from msgspec import Struct

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.review_db import PairInsert
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.pipeline import _score_candidate
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.parsers.marc import iter_marc_records

_LOGGER = getLogger(__name__)

IDF_CACHE_NAME: str = "idf.msgpack"

ScorePairFn = Callable[[MarcRecord, IndexedNyplRegRecord], CandidateMatch]
MarcLookupFn = Callable[[str], MarcRecord | None]
CceLookupFn = Callable[[str], IndexedNyplRegRecord | None]


class ResolvedVaultPair(Struct, frozen=True, forbid_unknown_fields=True):
    """A vault entry paired with its already-scored, ready-to-insert pair.

    Carried across the pickle boundary from the parent (where scoring runs
    single-threaded against the live LMDB lookup) to the writer process.
    """

    entry: VaultEntry
    pair: PairInsert


class ResolveSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`resolve_vault_pairs`."""

    resolved: int
    missing_in_pool: int
    missing_in_index: int


def iter_pool_shards(pool: Path) -> Iterator[Path]:
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
    for shard in iter_pool_shards(pool):
        for record in iter_marc_records(shard):
            if record.control_id in remaining:
                found[record.control_id] = record
                remaining.discard(record.control_id)
                if not remaining:
                    return found
    return found


def make_pair_scorer(
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


def resolve_vault_pairs(
    *,
    vault: dict[tuple[str, str], VaultEntry],
    marc_lookup: MarcLookupFn,
    cce_lookup: CceLookupFn,
    score_pair: ScorePairFn,
    build_pair: Callable[[MarcRecord, IndexedNyplRegRecord, CandidateMatch], PairInsert],
) -> tuple[list[ResolvedVaultPair], ResolveSummary]:
    """Score every vault entry whose MARC + CCE are still available.

    Walks each ``(marc_control_id, nypl_uuid)`` key in ``vault``, looks the
    MARC record up via ``marc_lookup`` and the CCE registration up via
    ``cce_lookup``, scores the specific pair via ``score_pair``, and assembles
    the resulting :class:`PairInsert` with ``build_pair``. Entries whose MARC
    is absent from the pool or whose CCE is absent from the index are logged
    with a WARNING and skipped — the vault file is never modified.

    Args:
        vault: Current vault entries keyed by ``(marc_control_id, nypl_uuid)``.
        marc_lookup: ``control_id -> MarcRecord | None`` resolver.
        cce_lookup: ``nypl_uuid -> IndexedNyplRegRecord | None`` resolver.
        score_pair: ``(marc, cce) -> CandidateMatch`` scorer.
        build_pair: Project the scored result into a :class:`PairInsert`.

    Returns:
        ``(resolved, summary)``. ``resolved`` is empty when ``vault`` is empty.
    """
    resolved: list[ResolvedVaultPair] = []
    missing_in_pool = 0
    missing_in_index = 0
    for (marc_id, nypl_uuid), entry in vault.items():
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
        pair = build_pair(marc, cce, candidate)
        resolved.append(ResolvedVaultPair(entry=entry, pair=pair))
    summary = ResolveSummary(
        resolved=len(resolved),
        missing_in_pool=missing_in_pool,
        missing_in_index=missing_in_index,
    )
    return resolved, summary


__all__ = [
    "IDF_CACHE_NAME",
    "CceLookupFn",
    "MarcLookupFn",
    "ResolveSummary",
    "ResolvedVaultPair",
    "ScorePairFn",
    "build_marc_index",
    "iter_pool_shards",
    "make_pair_scorer",
    "resolve_vault_pairs",
]
