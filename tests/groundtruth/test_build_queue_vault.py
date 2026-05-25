"""Unit tests for the build-queue-side vault pair resolver glue."""

from pd_groundtruth.build_queue_vault import _make_vault_pair_builder
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


def _marc() -> MarcRecord:
    return MarcRecord(
        control_id="ctrl-1",
        title="A Title",
        title_main="A Title",
        main_author="An Author",
        statement_of_responsibility="by An Author",
        publisher="A Publisher",
        publication_year=1953,
        language_code="eng",
    )


def _cce() -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="uuid-1",
        title="CCE Title",
        was_renewed=True,
        regnum="R123",
        reg_year=1953,
        author_name="CCE Author",
        publisher_names=("Pub A",),
        claimants=("Claimant A",),
    )


def _candidate(score: float) -> CandidateMatch:
    evidence = Evidence(
        scorer="title.token_set", score=0.9, max=1.0, skipped=False, decisive=False, features=()
    )
    return CandidateMatch(
        nypl_uuid="uuid-1",
        nypl_year=1953,
        combined=CombinedScore(raw=score * 100.0, calibrated=score),
        evidence=(evidence,),
        losing_evidence=(),
    )


def test_make_vault_pair_builder_projects_scored_candidate_into_pair_insert() -> None:
    builder = _make_vault_pair_builder()
    pair = builder(_marc(), _cce(), _candidate(0.95))
    assert pair.marc_control_id == "ctrl-1"
    assert pair.nypl_uuid == "uuid-1"
    assert pair.language == "eng"
    assert pair.score == 0.95
    assert pair.band == "ge90"
    assert pair.source == SOURCE_BANDED


def test_make_vault_pair_builder_bands_below_when_score_under_threshold() -> None:
    builder = _make_vault_pair_builder()
    pair = builder(_marc(), _cce(), _candidate(0.42))
    assert pair.band == "below"


def test_resolve_vault_for_build_round_trips_entry_into_resolved_pair() -> None:
    """A resolved vault pair carries the originating entry forward verbatim."""
    from pd_groundtruth.label_vault import SCHEMA_VERSION
    from pd_groundtruth.label_vault import MarcIdentifiers
    from pd_groundtruth.label_vault import VaultEntry
    from pd_groundtruth.vault_pair_resolver import ResolvedVaultPair

    builder = _make_vault_pair_builder()
    pair = builder(_marc(), _cce(), _candidate(0.95))
    entry = VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=pair.marc_control_id,
        nypl_uuid=pair.nypl_uuid,
        verdict="match",
        note="seed",
        labeled_at="2026-05-22T10:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )
    resolved = ResolvedVaultPair(entry=entry, pair=pair)
    assert resolved.entry == entry
