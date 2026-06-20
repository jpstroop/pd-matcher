"""Backfill machine-derived fields onto every vault entry (schema 6).

``enrich-vault`` sweeps the label vault and ADDS derived data to each entry
without ever touching a human-entered field (verdict, note, categories,
timestamps, labeler, the ``cce_*``/``marc_*`` identifiers). Per pair it
resolves the MARC record and the CCE registration, copies the CCE-side
``reg_year`` / ``was_renewed`` / ``renewal_year`` off the joined record, scores
the pair through the production matcher's per-scorer Evidence exactly once, and
applies both combiners to that single Evidence stream so the two confidences
are directly comparable to ``1.0``. The enriched vault is rewritten atomically
at ``schema=6`` via :func:`pd_groundtruth.label_vault.upsert_entry`.

The orchestration here is dependency-injected: it takes plain lookup callables
and a combiner pair so it can be exercised against tiny fixtures without an
LMDB index, a candidate pool, or a trained model. The CLI command in
:mod:`pd_groundtruth.cli` wires the real resolvers.
"""

from collections.abc import Callable
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import Literal

from msgspec import Struct

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MatcherScores
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import upsert_entry
from pd_groundtruth.matcher_version import matcher_version as resolve_matcher_version
from pd_groundtruth.vault_pair_resolver import AUTHOR_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import PUBLISHER_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import build_marc_index_from_collection
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import load_or_build_author_idf
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.idf import load_or_build_publisher_idf
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)

_SCORE_PRECISION: int = 4
_LEARNED_SCORER: Literal["learned"] = "learned"

MarcLookupFn = Callable[[str], MarcRecord | None]
CceLookupFn = Callable[[str], IndexedNyplRegRecord | None]
EvidenceScorerFn = Callable[[MarcRecord, IndexedNyplRegRecord], Sequence[Evidence]]


class EnrichReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`enrich_vault`."""

    total_entries: int
    enriched: int
    missing_in_pool: int
    missing_in_index: int
    learned_scored: int


def _score(combiner: Combiner, evidence: Sequence[Evidence]) -> float:
    """Return ``combiner``'s calibrated confidence for ``evidence``, rounded."""
    return round(combiner.combine(evidence).calibrated, _SCORE_PRECISION)


def _matcher_scores(
    evidence: Sequence[Evidence],
    weighted_combiner: Combiner,
    learned_combiner: Combiner | None,
) -> tuple[MatcherScores, bool]:
    """Build a :class:`MatcherScores` by applying both combiners to ``evidence``.

    The learned confidence is ``None`` when no learned combiner is available.
    Returns ``(scores, learned_scored)`` where ``learned_scored`` reports
    whether the learned arm produced a value, for the run summary.
    """
    weighted = _score(weighted_combiner, evidence)
    if learned_combiner is None:
        return MatcherScores(weighted_mean=weighted, learned=None), False
    learned = _score(learned_combiner, evidence)
    return MatcherScores(weighted_mean=weighted, learned=learned), True


def _renewal_year(cce: IndexedNyplRegRecord) -> int | None:
    """Return the renewal-recording year, or ``None`` when the join is absent.

    The renewal year is derived from ``renewal_rdat`` (the renewal-recording
    date copied onto the indexed registration during the index-build join);
    when no renewal joined the registration the date — and the year — are
    ``None``.
    """
    if cce.renewal_rdat is None:
        return None
    return cce.renewal_rdat.year


def _enriched_entry(
    entry: VaultEntry,
    cce: IndexedNyplRegRecord,
    scores: MatcherScores,
    matcher_version: str,
) -> VaultEntry:
    """Return ``entry`` with derived fields filled and ``schema`` bumped to 6.

    Every human-entered field is copied through unchanged; only the
    machine-derived fields and the schema version are written.
    """
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=entry.marc_control_id,
        nypl_uuid=entry.nypl_uuid,
        verdict=entry.verdict,
        note=entry.note,
        labeled_at=entry.labeled_at,
        labeler=entry.labeler,
        marc_identifiers=entry.marc_identifiers,
        cce_regnum=entry.cce_regnum,
        cce_renewal_id=entry.cce_renewal_id,
        cce_renewal_oreg=entry.cce_renewal_oreg,
        categories=entry.categories,
        reg_year=cce.reg_year,
        renewal_year=_renewal_year(cce),
        was_renewed=cce.was_renewed,
        scores=scores,
        matcher_version=matcher_version,
    )


def enrich_vault(
    *,
    vault_path: Path,
    marc_lookup: MarcLookupFn,
    cce_lookup: CceLookupFn,
    score_evidence: EvidenceScorerFn,
    weighted_combiner: Combiner,
    learned_combiner: Combiner | None,
    matcher_version: str,
    dry_run: bool,
) -> EnrichReport:
    """Backfill schema-6 derived fields onto every resolvable vault entry.

    Walks every ``(marc_control_id, nypl_uuid)`` entry, resolves the MARC via
    ``marc_lookup`` and the CCE registration via ``cce_lookup``, scores the pair
    into per-scorer Evidence once via ``score_evidence``, applies both combiners
    to that Evidence, and writes the derived fields back. Entries whose MARC or
    CCE no longer resolves keep their derived fields untouched (left ``None``),
    are counted, and are logged at WARNING. When ``dry_run`` is ``True`` the
    counts are computed and reported but nothing is written.

    Args:
        vault_path: The JSONL vault to enrich in place.
        marc_lookup: ``control_id -> MarcRecord | None`` resolver.
        cce_lookup: ``nypl_uuid -> IndexedNyplRegRecord | None`` resolver.
        score_evidence: ``(marc, cce) -> Evidence`` sequence producer.
        weighted_combiner: Weighted-mean combiner; always present.
        learned_combiner: Learned combiner, or ``None`` when no artifact was
            available (the learned score is then left ``None``).
        matcher_version: Build identifier stamped onto every score.
        dry_run: When ``True``, compute and report but write nothing.

    Returns:
        An :class:`EnrichReport` summarising the run.
    """
    entries = current_entries(vault_path)
    missing_in_pool = 0
    missing_in_index = 0
    enriched: list[VaultEntry] = []
    learned_scored = 0
    for (marc_id, nypl_uuid), entry in entries.items():
        marc = marc_lookup(marc_id)
        if marc is None:
            missing_in_pool += 1
            _LOGGER.warning(
                "enrich.marc_not_in_pool marc_control_id=%s nypl_uuid=%s",
                marc_id,
                nypl_uuid,
            )
            continue
        cce = cce_lookup(nypl_uuid)
        if cce is None:
            missing_in_index += 1
            _LOGGER.warning(
                "enrich.cce_not_in_index marc_control_id=%s nypl_uuid=%s",
                marc_id,
                nypl_uuid,
            )
            continue
        evidence = score_evidence(marc, cce)
        scores, did_learn = _matcher_scores(evidence, weighted_combiner, learned_combiner)
        if did_learn:
            learned_scored += 1
        enriched.append(_enriched_entry(entry, cce, scores, matcher_version))
    if not dry_run:
        for entry in enriched:
            upsert_entry(vault_path, entry)
    return EnrichReport(
        total_entries=len(entries),
        enriched=len(enriched),
        missing_in_pool=missing_in_pool,
        missing_in_index=missing_in_index,
        learned_scored=learned_scored,
    )


def _learned_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` with the learned scorer forced on.

    A copy is needed because the on-disk default selects the weighted-mean
    scorer; ``build_combiner`` keys the learned arm off ``config.scorer``.
    """
    return MatchingConfig(
        title_weight=config.title_weight,
        author_weight=config.author_weight,
        publisher_weight=config.publisher_weight,
        edition_weight=config.edition_weight,
        lccn_weight=config.lccn_weight,
        isbn_weight=config.isbn_weight,
        extent_weight=config.extent_weight,
        volume_weight=config.volume_weight,
        year_window=config.year_window,
        min_combined_score=config.min_combined_score,
        scorer=_LEARNED_SCORER,
    )


def _build_learned_combiner(config: MatchingConfig, model_dir: Path) -> Combiner | None:
    """Return the learned combiner, or ``None`` (with a WARNING) when absent.

    The learned artifact is optional at enrichment time: a fresh clone may not
    have trained one. A missing or stale artifact (``ValueError``) or a missing
    ``lightgbm`` dependency (``ImportError``) yields ``None`` so the run still
    records the weighted-mean score; it never aborts enrichment.
    """
    try:
        return build_combiner(_learned_config(config), learned_model_dir=model_dir)
    except (ValueError, ImportError) as exc:
        _LOGGER.warning(
            "enrich.learned_unavailable model_dir=%s reason=%s",
            model_dir,
            exc,
        )
        return None


def _make_evidence_scorer(
    *,
    matching_config: MatchingConfig,
    index_path: Path,
    pairing_config: PairingConfig,
) -> EvidenceScorerFn:
    """Build the per-pair Evidence producer using the weighted-mean pipeline.

    Local indirection (mirroring ``vault_into_queue``'s ``_make_pair_scorer``)
    so tests can monkey-patch the matcher wiring. The weighted-mean scorer is
    used because per-scorer Evidence is combiner-independent — both combiners
    are applied to the same Evidence downstream.
    """
    pairings = compile_pairings(pairing_config)
    idf = load_or_build_idf(index_path.parent / IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path))
    author_idf = load_or_build_author_idf(
        index_path.parent / AUTHOR_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    publisher_idf = load_or_build_publisher_idf(
        index_path.parent / PUBLISHER_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    score_pair = make_pair_scorer(
        matching_config=matching_config,
        pairings=pairings,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        calibrator=None,
    )

    def score_evidence(marc: MarcRecord, cce: IndexedNyplRegRecord) -> Sequence[Evidence]:
        return score_pair(marc, cce).evidence

    return score_evidence


def run_enrich(
    *,
    vault_path: Path,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    pool_path: Path | None = None,
    marc_collection_path: Path | None = None,
    dry_run: bool = False,
) -> EnrichReport:
    """Resolve resources and enrich every vault entry end-to-end.

    The entry point the ``enrich-vault`` Typer command calls. Resolves each
    vault MARC from one MARC source, opens the CCE index, builds the IDF caches
    and both combiners, and delegates the per-entry sweep to :func:`enrich_vault`.

    Exactly one MARC source must be supplied: ``pool_path`` reads each vault
    MARC from the sharded acquired pool (``<pool>/<lang>/*.xml``);
    ``marc_collection_path`` reads them from a single committed MARCXML
    ``<collection>`` (``data/training/marc.xml``).

    Args:
        vault_path: JSONL label vault to enrich in place.
        index_path: LMDB env produced by ``pd-matcher index build``.
        matching_config: Active matcher config (the weighted-mean default).
        pairing_config: Active field-pairing config.
        pool_path: Sharded candidate pool root, mutually exclusive with
            ``marc_collection_path``.
        marc_collection_path: Single MARCXML collection, mutually exclusive
            with ``pool_path``.
        dry_run: When ``True``, compute and report but write nothing.

    Raises:
        ValueError: If neither or both MARC sources are supplied.
    """
    vault = current_entries(vault_path)
    needed_marc_ids = {marc_id for marc_id, _uuid in vault}
    if pool_path is not None and marc_collection_path is None:
        marc_by_id = build_marc_index(pool_path, needed_marc_ids)
    elif pool_path is None and marc_collection_path is not None:
        marc_by_id = build_marc_index_from_collection(marc_collection_path, needed_marc_ids)
    else:
        raise ValueError("provide exactly one of pool_path or marc_collection_path")

    score_evidence = _make_evidence_scorer(
        matching_config=matching_config,
        index_path=index_path,
        pairing_config=pairing_config,
    )
    weighted_combiner = build_combiner(matching_config, learned_model_dir=None)
    learned_combiner = _build_learned_combiner(matching_config, index_path.parent)
    version = resolve_matcher_version()

    with NyplIndexLookup(index_path) as lookup:
        return enrich_vault(
            vault_path=vault_path,
            marc_lookup=marc_by_id.get,
            cce_lookup=lookup.get_registration,
            score_evidence=score_evidence,
            weighted_combiner=weighted_combiner,
            learned_combiner=learned_combiner,
            matcher_version=version,
            dry_run=dry_run,
        )


__all__ = [
    "CceLookupFn",
    "EnrichReport",
    "EvidenceScorerFn",
    "MarcLookupFn",
    "enrich_vault",
    "run_enrich",
]
