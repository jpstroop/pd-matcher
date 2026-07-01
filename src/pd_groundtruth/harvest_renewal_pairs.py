"""Harvest verified MARC↔renewal training pairs from the vault and index.

The label vault holds human-verified ``(MARC, registration)`` verdicts. When a
verified *registration* match's registration is deterministically joined to a
renewal in the rebuilt index, transitivity yields a *free* verified
``(MARC, renewal)`` positive — a human-verified MARC↔registration link plus a
deterministic registration↔renewal join — with no additional hand-labeling.
This module turns those into a renewal-matcher training set:

1. **Positives.** Every vault entry that is a ``verdict == "match"`` on a
   *registration* pathway (``match_source`` in ``{None, "registration",
   "both"}``; the legacy entries carry ``None``, which the schema-7 migration
   backfills to ``"registration"``) is examined. Its registration is resolved
   in the index by ``nypl_uuid``; when that registration is joined
   (``was_renewed`` with a ``renewal_id``) the joined renewal is fetched and a
   POSITIVE pair ``(MARC, renewal, label="match")`` is emitted.
2. **Hard negatives.** For each positive's MARC the renewal-retrieval arm
   (:meth:`~pd_matcher.index.lookup.NyplIndexLookup.candidates_for_renewal`)
   proposes renewal candidates; they are scored with the same weighted-mean
   renewal scorer :mod:`pd_groundtruth.build_renewal_queue` uses. The
   top-scoring candidates whose renewal id differs from the true joined
   renewal's id are emitted as ``label="no_match"`` — look-alikes the matcher
   must learn to reject.

The harvest is kept strictly SEPARATE from the vault: it is a derived,
verified-by-transitivity set, never merged into the pure hand-labeled ground
truth. Every emitted row carries a ``provenance`` marker so it is never
confused with vault data, and the vault file is only ever read.

The core :func:`harvest_renewal_pairs` is dependency-injected — plain lookup
callables and a score function — so it runs against tiny fixtures without an
LMDB index or a candidate pool. :func:`run_harvest` wires the real resolvers
for the CLI.
"""

from collections.abc import Callable
from collections.abc import Iterable
from logging import getLogger
from pathlib import Path

from msgspec import Struct
from msgspec.json import encode as json_encode

from pd_groundtruth.build_renewal_queue import RenewalScoreFn
from pd_groundtruth.build_renewal_queue import _load_calibrator
from pd_groundtruth.build_renewal_queue import _make_score_fn
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.vault_pair_resolver import AUTHOR_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import PUBLISHER_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import build_marc_index_from_collection
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.idf import load_or_build_author_idf
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.idf import load_or_build_publisher_idf
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord

_LOGGER = getLogger(__name__)

PROVENANCE_POSITIVE: str = "harvested_2a"
PROVENANCE_HARD_NEGATIVE: str = "harvested_hard_negative"

_REGISTRATION_PATHWAYS: frozenset[str] = frozenset({"registration", "both"})

MarcLookupFn = Callable[[str], MarcRecord | None]
RegLookupFn = Callable[[str], IndexedNyplRegRecord | None]
RenewalLookupFn = Callable[[str], NyplRenRecord | None]
RenewalCandidatesFn = Callable[[MarcRecord, int], Iterable[NyplRenRecord]]


class HarvestedPair(Struct, frozen=True, forbid_unknown_fields=True):
    """One harvested MARC↔renewal training row, positive or hard negative.

    Carries exactly what a renewal-pathway trainer consumes: the MARC identifier
    plus the MARC fields the renewal scorer reads (title / author / publisher /
    year), the renewal fields it reads (title / author / claimants / oreg /
    odat), the ``label`` (``"match"`` for a positive, ``"no_match"`` for a hard
    negative), the ``provenance`` marker, and the renewal-arm ``score`` (the
    weighted-mean combiner's calibrated confidence for the pairing — the true
    pair's confidence for a positive, the look-alike's for a negative).
    """

    marc_control_id: str
    marc_title: str | None
    marc_author: str | None
    marc_publisher: str | None
    marc_year: int | None
    renewal_id: str
    renewal_entry_id: str
    renewal_title: str | None
    renewal_author: str | None
    renewal_claimants: str | None
    renewal_oreg: str | None
    renewal_odat: str | None
    label: str
    provenance: str
    score: float


class HarvestSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`harvest_renewal_pairs`.

    ``vault_matches_examined`` is the number of verified registration-pathway
    match entries considered; ``missing_marc`` / ``missing_registration`` /
    ``registration_not_joined`` / ``renewal_missing`` account for the entries
    that dropped out before a positive could be emitted; ``joined`` counts the
    registrations that resolved to a joined renewal (equal to ``positives``);
    ``negatives`` counts the emitted hard negatives.
    """

    vault_matches_examined: int
    missing_marc: int
    missing_registration: int
    registration_not_joined: int
    renewal_missing: int
    joined: int
    positives: int
    negatives: int


def _is_registration_pathway_match(entry: VaultEntry) -> bool:
    """Return whether ``entry`` is a verified registration-pathway match.

    A registration-pathway match is ``verdict == "match"`` whose ``match_source``
    is a registration pathway: ``"registration"``, ``"both"``, or ``None`` (the
    legacy default the schema-7 migration backfills to ``"registration"``). The
    ``"renewal"`` pathway is excluded — those pairs are already MARC↔renewal.
    """
    if entry.verdict != VERDICT_MATCH:
        return False
    return entry.match_source is None or entry.match_source in _REGISTRATION_PATHWAYS


def _marc_author(marc: MarcRecord) -> str | None:
    """Return the MARC author the renewal scorer reads, main or statement."""
    return marc.main_author or marc.statement_of_responsibility


def _positive_pair(marc: MarcRecord, renewal: NyplRenRecord, score: float) -> HarvestedPair:
    """Assemble the POSITIVE harvested pair for one verified MARC↔renewal link."""
    return _pair(marc, renewal, VERDICT_MATCH, PROVENANCE_POSITIVE, score)


def _negative_pair(marc: MarcRecord, renewal: NyplRenRecord, score: float) -> HarvestedPair:
    """Assemble a hard-NEGATIVE harvested pair for one look-alike renewal."""
    return _pair(marc, renewal, VERDICT_NO_MATCH, PROVENANCE_HARD_NEGATIVE, score)


def _pair(
    marc: MarcRecord,
    renewal: NyplRenRecord,
    label: str,
    provenance: str,
    score: float,
) -> HarvestedPair:
    """Project a ``(MARC, renewal)`` pairing into a :class:`HarvestedPair`."""
    return HarvestedPair(
        marc_control_id=marc.control_id,
        marc_title=marc.title,
        marc_author=_marc_author(marc),
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        renewal_id=renewal.id,
        renewal_entry_id=renewal.entry_id,
        renewal_title=renewal.title,
        renewal_author=renewal.author,
        renewal_claimants=renewal.claimants,
        renewal_oreg=renewal.oreg,
        renewal_odat=renewal.odat.isoformat() if renewal.odat is not None else None,
        label=label,
        provenance=provenance,
        score=score,
    )


def _rank_hard_negatives(
    marc: MarcRecord,
    candidates: Iterable[NyplRenRecord],
    score_fn: RenewalScoreFn,
    exclude_renewal_id: str,
    limit: int,
) -> list[tuple[NyplRenRecord, float]]:
    """Return the top ``limit`` scored look-alike renewals, true renewal excluded.

    Every candidate whose id differs from ``exclude_renewal_id`` (the true joined
    renewal) is scored with the renewal arm; the highest-scoring ``limit`` are
    returned as ``(renewal, calibrated_score)`` pairs in descending score order.
    An empty list is returned when ``limit`` is non-positive or no candidate
    survives the exclusion.
    """
    if limit <= 0:
        return []
    scored: list[tuple[NyplRenRecord, float]] = [
        (renewal, score_fn(marc, renewal).calibrated)
        for renewal in candidates
        if renewal.id != exclude_renewal_id
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def harvest_renewal_pairs(
    *,
    entries: Iterable[VaultEntry],
    marc_lookup: MarcLookupFn,
    reg_lookup: RegLookupFn,
    renewal_lookup: RenewalLookupFn,
    renewal_candidates: RenewalCandidatesFn,
    score_fn: RenewalScoreFn,
    window: int,
    negatives_per_positive: int,
) -> tuple[list[HarvestedPair], HarvestSummary]:
    """Harvest verified MARC↔renewal positives and their hard negatives.

    For every verified registration-pathway match entry the MARC is resolved via
    ``marc_lookup`` and the registration via ``reg_lookup``; when the
    registration is joined (``was_renewed`` with a ``renewal_id``) the joined
    renewal is fetched via ``renewal_lookup`` and emitted as a POSITIVE pair.
    Each positive's MARC is then run through ``renewal_candidates`` and scored
    with ``score_fn`` to emit up to ``negatives_per_positive`` hard NEGATIVE
    look-alikes (the true renewal excluded). Rows are emitted positive-first,
    each positive immediately followed by its negatives.

    Args:
        entries: Vault entries to examine; non-registration-pathway and
            non-match entries are ignored.
        marc_lookup: ``control_id -> MarcRecord | None`` resolver.
        reg_lookup: ``nypl_uuid -> IndexedNyplRegRecord | None`` resolver.
        renewal_lookup: ``renewal_id -> NyplRenRecord | None`` resolver.
        renewal_candidates: ``(marc, window) -> renewals`` retrieval callable.
        score_fn: ``(marc, renewal) -> RenewalScore`` renewal-arm scorer.
        window: Inclusive year radius passed to ``renewal_candidates``.
        negatives_per_positive: Hard negatives to emit per positive.

    Returns:
        ``(pairs, summary)`` — every harvested row and the run tallies.
    """
    pairs: list[HarvestedPair] = []
    examined = 0
    missing_marc = 0
    missing_registration = 0
    not_joined = 0
    renewal_missing = 0
    positives = 0
    negatives = 0
    for entry in entries:
        if not _is_registration_pathway_match(entry):
            continue
        examined += 1
        marc = marc_lookup(entry.marc_control_id)
        if marc is None:
            missing_marc += 1
            _LOGGER.warning(
                "harvest.marc_not_found marc_control_id=%s nypl_uuid=%s",
                entry.marc_control_id,
                entry.nypl_uuid,
            )
            continue
        reg = reg_lookup(entry.nypl_uuid)
        if reg is None:
            missing_registration += 1
            _LOGGER.warning(
                "harvest.registration_not_in_index marc_control_id=%s nypl_uuid=%s",
                entry.marc_control_id,
                entry.nypl_uuid,
            )
            continue
        if not reg.was_renewed or reg.renewal_id is None:
            not_joined += 1
            continue
        renewal = renewal_lookup(reg.renewal_id)
        if renewal is None:
            renewal_missing += 1
            _LOGGER.warning(
                "harvest.renewal_not_in_index marc_control_id=%s renewal_id=%s",
                entry.marc_control_id,
                reg.renewal_id,
            )
            continue
        positive_score = score_fn(marc, renewal).calibrated
        pairs.append(_positive_pair(marc, renewal, positive_score))
        positives += 1
        for look_alike, score in _rank_hard_negatives(
            marc,
            renewal_candidates(marc, window),
            score_fn,
            renewal.id,
            negatives_per_positive,
        ):
            pairs.append(_negative_pair(marc, look_alike, score))
            negatives += 1
    summary = HarvestSummary(
        vault_matches_examined=examined,
        missing_marc=missing_marc,
        missing_registration=missing_registration,
        registration_not_joined=not_joined,
        renewal_missing=renewal_missing,
        joined=positives,
        positives=positives,
        negatives=negatives,
    )
    return pairs, summary


def write_harvest(path: Path, pairs: Iterable[HarvestedPair]) -> None:
    """Write ``pairs`` as newline-delimited JSON to ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for pair in pairs:
            handle.write(json_encode(pair))
            handle.write(b"\n")


def _make_renewal_lookup(lookup: NyplIndexLookup) -> RenewalLookupFn:
    """Build a ``renewal.id -> NyplRenRecord | None`` resolver over the index.

    The index keys its renewal store by ``entry_id``, but a joined registration
    records the renewal's distinct ``id`` (the ``renewal_id`` projection). One
    scan of :meth:`~pd_matcher.index.lookup.NyplIndexLookup.iter_renewals` builds
    the ``id -> entry_id`` bridge so a registration's ``renewal_id`` resolves to
    its full renewal record — the only place the renewal's ``odat`` and
    ``entry_id`` are available (the projection baked onto the registration omits
    them). The map holds only the two id strings per renewal, not the records.
    """
    id_to_entry = {record.id: record.entry_id for record in lookup.iter_renewals()}

    def renewal_lookup(renewal_id: str) -> NyplRenRecord | None:
        entry_id = id_to_entry.get(renewal_id)
        if entry_id is None:
            return None
        return lookup.get_renewal(entry_id)

    return renewal_lookup


def _make_renewal_score_fn(
    index_path: Path,
    matching_config: MatchingConfig,
) -> RenewalScoreFn:
    """Build the renewal-arm scorer, loading IDF caches and calibrator by index.

    Mirrors :func:`pd_groundtruth.build_renewal_queue.build_renewal_queue`'s
    wiring: the shared IDF caches beside ``index_path`` feed the weighted-mean
    combiner, and the Platt calibrator (when present) maps raw scores as the
    production pipeline does.
    """
    idf = load_or_build_idf(index_path.parent / IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path))
    author_idf = load_or_build_author_idf(
        index_path.parent / AUTHOR_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    publisher_idf = load_or_build_publisher_idf(
        index_path.parent / PUBLISHER_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    calibrator = _load_calibrator(index_path.parent)
    combiner = build_combiner(matching_config, learned_model_dir=None)
    return _make_score_fn(idf, author_idf, publisher_idf, matching_config, combiner, calibrator)


def run_harvest(
    *,
    vault_path: Path,
    index_path: Path,
    out_path: Path,
    matching_config: MatchingConfig,
    negatives_per_positive: int,
    pool_path: Path | None = None,
    marc_collection_path: Path | None = None,
) -> tuple[list[HarvestedPair], HarvestSummary]:
    """Harvest from the real vault + index and write the training set to disk.

    The entry point the ``harvest-renewal-pairs`` Typer command calls. Reads the
    vault (read-only), resolves each verified-match MARC from one MARC source,
    opens the CCE index, builds the renewal-arm scorer, harvests positives and
    hard negatives via :func:`harvest_renewal_pairs`, and writes the JSONL set.

    Exactly one MARC source must be supplied: ``pool_path`` reads each vault MARC
    from the sharded acquired pool (``<pool>/<lang>/*.xml``);
    ``marc_collection_path`` reads them from a single committed MARCXML
    ``<collection>`` (``data/training/marc.xml``).

    Args:
        vault_path: JSONL label vault, read-only.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination JSONL training set (overwritten).
        matching_config: Active matcher config; supplies the renewal-retrieval
            year window and the scoring weights.
        negatives_per_positive: Hard negatives to emit per positive.
        pool_path: Sharded candidate pool root, mutually exclusive with
            ``marc_collection_path``.
        marc_collection_path: Single MARCXML collection, mutually exclusive with
            ``pool_path``.

    Returns:
        ``(pairs, summary)`` — every harvested row and the run tallies.

    Raises:
        ValueError: If neither or both MARC sources are supplied.
    """
    entries = [entry for entry in iter_entries(vault_path) if _is_registration_pathway_match(entry)]
    needed_marc_ids = {entry.marc_control_id for entry in entries}
    if pool_path is not None and marc_collection_path is None:
        marc_by_id = build_marc_index(pool_path, needed_marc_ids)
    elif pool_path is None and marc_collection_path is not None:
        marc_by_id = build_marc_index_from_collection(marc_collection_path, needed_marc_ids)
    else:
        raise ValueError("provide exactly one of pool_path or marc_collection_path")
    score_fn = _make_renewal_score_fn(index_path, matching_config)
    window = matching_config.year_window
    with NyplIndexLookup(index_path) as lookup:
        pairs, summary = harvest_renewal_pairs(
            entries=entries,
            marc_lookup=marc_by_id.get,
            reg_lookup=lookup.get_registration,
            renewal_lookup=_make_renewal_lookup(lookup),
            renewal_candidates=lookup.candidates_for_renewal,
            score_fn=score_fn,
            window=window,
            negatives_per_positive=negatives_per_positive,
        )
    write_harvest(out_path, pairs)
    return pairs, summary


__all__ = [
    "PROVENANCE_HARD_NEGATIVE",
    "PROVENANCE_POSITIVE",
    "HarvestSummary",
    "HarvestedPair",
    "MarcLookupFn",
    "RegLookupFn",
    "RenewalCandidatesFn",
    "RenewalLookupFn",
    "harvest_renewal_pairs",
    "run_harvest",
    "write_harvest",
]
