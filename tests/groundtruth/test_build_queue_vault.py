"""Unit tests for the build-queue-side vault pair resolver glue."""

from logging import INFO
from pathlib import Path

from pytest import LogCaptureFixture

from pd_groundtruth.build_queue_vault import _load_vault_filtered
from pd_groundtruth.build_queue_vault import _make_vault_pair_builder
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import upsert_entry
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


def _entry(control_id: str, nypl_uuid: str, *, verdict: str) -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        note=None,
        labeled_at="2026-05-22T10:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )


def _seed_vault(path: Path, entries: tuple[VaultEntry, ...]) -> None:
    for entry in entries:
        upsert_entry(path, entry)


def test_load_vault_filtered_default_returns_all_entries(tmp_path: Path) -> None:
    """No requeue verdicts: every vault entry comes through unchanged."""
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _entry("m1", "u1", verdict="match"),
            _entry("m2", "u2", verdict="no_match"),
            _entry("m3", "u3", verdict="unsure"),
        ),
    )
    loaded = _load_vault_filtered(vault_path, frozenset())
    assert set(loaded) == {("m1", "u1"), ("m2", "u2"), ("m3", "u3")}


def test_load_vault_filtered_drops_unsure(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    """``requeue_verdicts={'unsure'}`` drops unsure; logs the dropped count."""
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _entry("m1", "u1", verdict="match"),
            _entry("m2", "u2", verdict="no_match"),
            _entry("m3", "u3", verdict="unsure"),
            _entry("m4", "u4", verdict="unsure"),
        ),
    )
    with caplog.at_level(INFO, logger="pd_groundtruth.build_queue_vault"):
        loaded = _load_vault_filtered(vault_path, frozenset({"unsure"}))
    assert set(loaded) == {("m1", "u1"), ("m2", "u2")}
    assert any(
        "vault.requeue verdict=unsure dropped=2" in record.message for record in caplog.records
    )


def test_load_vault_filtered_drops_multiple_verdicts(tmp_path: Path) -> None:
    """``requeue_verdicts={'unsure', 'no_match'}`` re-queues both; match survives."""
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _entry("m1", "u1", verdict="match"),
            _entry("m2", "u2", verdict="no_match"),
            _entry("m3", "u3", verdict="unsure"),
        ),
    )
    loaded = _load_vault_filtered(vault_path, frozenset({"unsure", "no_match"}))
    assert set(loaded) == {("m1", "u1")}


def test_load_vault_filtered_drops_only_matches_when_requeueing_match(
    tmp_path: Path,
) -> None:
    """``requeue_verdicts={'match'}`` re-queues matches; no_match and unsure survive."""
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _entry("m1", "u1", verdict="match"),
            _entry("m2", "u2", verdict="no_match"),
            _entry("m3", "u3", verdict="unsure"),
        ),
    )
    loaded = _load_vault_filtered(vault_path, frozenset({"match"}))
    assert set(loaded) == {("m2", "u2"), ("m3", "u3")}


def test_load_vault_filtered_logs_zero_for_unmatched_verdict(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Re-queuing a verdict with no vault entries is a no-op (logged, not errored)."""
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _entry("m1", "u1", verdict="match"),
            _entry("m2", "u2", verdict="no_match"),
        ),
    )
    with caplog.at_level(INFO, logger="pd_groundtruth.build_queue_vault"):
        loaded = _load_vault_filtered(vault_path, frozenset({"unsure"}))
    assert set(loaded) == {("m1", "u1"), ("m2", "u2")}
    assert any(
        "vault.requeue verdict=unsure dropped=0" in record.message for record in caplog.records
    )
