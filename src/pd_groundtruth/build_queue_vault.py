"""Build-queue-side glue around :mod:`pd_groundtruth.vault_pair_resolver`.

Resolves every current vault entry into a ready-to-insert pair *before* the
matcher runs so :func:`pd_groundtruth.build_queue.build_queue` can hand the
pre-scored pairs to its writer (which inserts them outside the per-stratum
caps) and exclude the vault MARC records from the per-language reservoir.
Kept separate from :mod:`pd_groundtruth.build_queue` to keep that module
focused on the writer + orchestration.
"""

from collections.abc import Callable
from datetime import date
from logging import getLogger
from pathlib import Path

from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import band_of
from pd_groundtruth.vault_pair_resolver import ResolvedVaultPair
from pd_groundtruth.vault_pair_resolver import ResolveSummary
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_groundtruth.vault_pair_resolver import resolve_vault_pairs
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.copyright.coverage import Coverage
from pd_matcher.copyright.facts import build_facts
from pd_matcher.copyright.rules import assess
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)


def _make_vault_pair_builder(
    ruleset: CopyrightRuleSet,
    copyright_config: CopyrightAssessmentConfig,
    coverage: Coverage,
) -> Callable[[MarcRecord, IndexedNyplRegRecord, CandidateMatch], PairInsert]:
    """Build a closure that projects scored vault pairs into ``PairInsert`` rows.

    Closes over the active ruleset + assessment config + index coverage
    so each pair's Cornell predicted status (Phase 5 rule engine) is
    materialized at vault resolution time. Returning a closure keeps
    :func:`pd_groundtruth.vault_pair_resolver.resolve_vault_pairs` signature
    unchanged.
    """
    from pd_groundtruth.build_queue import _build_pair_insert
    from pd_groundtruth.build_queue import _language_of

    as_of_year = (
        copyright_config.as_of_year
        if copyright_config.as_of_year is not None
        else date.today().year
    )

    def _build(
        marc: MarcRecord,
        cce: IndexedNyplRegRecord,
        candidate: CandidateMatch,
    ) -> PairInsert:
        score = candidate.combined.calibrated
        synthesized = MatchResult(
            marc_control_id=marc.control_id,
            best=candidate,
            alternates=(),
            candidates_considered=1,
        )
        facts = build_facts(marc, synthesized, as_of_year=as_of_year, matched_nypl=cce)
        assessment = assess(
            facts,
            ruleset,
            coverage=coverage,
            enable_assumptions=copyright_config.enable_assumptions,
        )
        return _build_pair_insert(
            marc,
            cce,
            candidate.evidence,
            language=_language_of(marc),
            score=score,
            band=band_of(score),
            source=SOURCE_BANDED,
            predicted_status=assessment.status.name,
        )

    return _build


def resolve_vault_for_build(
    *,
    vault_path: Path,
    pool: Path,
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    ruleset: CopyrightRuleSet,
    copyright_config: CopyrightAssessmentConfig,
    idf: IdfTable,
    calibrator: PlattCalibrator | None,
) -> tuple[list[ResolvedVaultPair], ResolveSummary]:
    """Resolve every current vault entry into a ready-to-insert pair.

    Loads the vault, materializes each entry's MARC record by streaming the
    pool shards, opens the LMDB index briefly to look up the matching CCE
    registration, and scores the pair via the matcher's per-pair routine so
    the persisted row carries real ``(score, band, evidence)``. Entries whose
    MARC is no longer in the pool or whose CCE is no longer in the index are
    skipped with a WARNING; the vault file itself is never modified.

    Returns:
        ``(resolved, summary)`` where ``resolved`` is empty for an empty or
        missing vault, and ``summary`` carries diagnostic counts that
        ``build_queue`` logs at the start of the run.
    """
    vault = current_entries(vault_path)
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
        build_pair = _make_vault_pair_builder(ruleset, copyright_config, lookup.coverage())
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
