"""Unit tests for vault enrichment (schema 6)."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Self

from pytest import MonkeyPatch
from pytest import raises
from typer.testing import CliRunner

from pd_groundtruth import enrich_vault as module
from pd_groundtruth.cli import app
from pd_groundtruth.enrich_vault import EnrichReport
from pd_groundtruth.enrich_vault import enrich_vault
from pd_groundtruth.enrich_vault import run_enrich
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.label_vault import upsert_entry
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_RUNNER = CliRunner()

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_MARCXML_TEMPLATE = (
    '<collection xmlns="{ns}">'
    "<record>"
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">{control_id}</controlfield>'
    '<controlfield tag="008">750101s1953    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245" ind1="0" ind2="0"><subfield code="a">A Title</subfield></datafield>'
    "</record>"
    "</collection>"
)

_VERSION = "deadbee"


class _FakeCombiner:
    """A combiner returning a fixed calibrated score for any Evidence."""

    def __init__(self, calibrated: float) -> None:
        self._calibrated = calibrated

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:
        del evidence
        return CombinedScore(raw=self._calibrated * 100.0, calibrated=self._calibrated)


def _marc(control_id: str = "ctrl-1") -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A Title",
        title_main="A Title",
        publication_year=1953,
        language_code="eng",
    )


def _cce(
    uuid: str = "uuid-1",
    *,
    reg_year: int | None = 1953,
    was_renewed: bool = True,
    renewal_rdat: date | None = date(1981, 4, 1),
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid,
        title="CCE Title",
        was_renewed=was_renewed,
        regnum="R123",
        reg_year=reg_year,
        renewal_rdat=renewal_rdat,
    )


def _evidence() -> tuple[Evidence, ...]:
    return (
        Evidence(
            scorer="title.token_set",
            score=0.9,
            max=1.0,
            skipped=False,
            decisive=False,
            features=(),
        ),
    )


def _entry(
    control_id: str = "ctrl-1",
    nypl_uuid: str = "uuid-1",
    *,
    verdict: str = "match",
    note: str | None = "a human note",
) -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        note=note,
        labeled_at="2026-05-22T10:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn="40012345", oclc=None, isbns=()),
        cce_regnum="A1",
        categories=("generic_title",),
        match_source="renewal",
    )


def _enrich(
    vault_path: Path,
    *,
    marc_lookup: module.MarcLookupFn,
    cce_lookup: module.CceLookupFn,
    learned: float | None = 0.91,
    dry_run: bool = False,
) -> EnrichReport:
    learned_combiner = _FakeCombiner(learned) if learned is not None else None
    return enrich_vault(
        vault_path=vault_path,
        marc_lookup=marc_lookup,
        cce_lookup=cce_lookup,
        score_evidence=lambda _m, _c: _evidence(),
        weighted_combiner=_FakeCombiner(0.84211),
        learned_combiner=learned_combiner,
        matcher_version=_VERSION,
        dry_run=dry_run,
    )


def test_enrich_writes_derived_fields_and_preserves_human_fields(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    original = _entry()
    upsert_entry(vault_path, original)

    report = _enrich(vault_path, marc_lookup=lambda _i: _marc(), cce_lookup=lambda _u: _cce())

    assert report == EnrichReport(
        total_entries=1,
        enriched=1,
        missing_in_pool=0,
        missing_in_index=0,
        learned_scored=1,
    )
    [entry] = list(iter_entries(vault_path))
    assert entry.schema == 8
    assert entry.reg_year == 1953
    assert entry.renewal_year == 1981
    assert entry.was_renewed is True
    assert entry.scores is not None
    assert entry.scores.weighted_mean == 0.8421
    assert entry.scores.learned == 0.91
    assert entry.matcher_version == _VERSION
    assert entry.verdict == original.verdict
    assert entry.note == original.note
    assert entry.labeled_at == original.labeled_at
    assert entry.labeler == original.labeler
    assert entry.marc_identifiers == original.marc_identifiers
    assert entry.cce_regnum == original.cce_regnum
    assert entry.categories == original.categories
    assert entry.match_source == "renewal"


def test_enrich_rounds_scores_to_four_decimals(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry())

    _enrich(vault_path, marc_lookup=lambda _i: _marc(), cce_lookup=lambda _u: _cce())

    [entry] = list(iter_entries(vault_path))
    assert entry.scores is not None
    assert entry.scores.weighted_mean == 0.8421


def test_enrich_leaves_renewal_year_none_when_not_renewed(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry())

    _enrich(
        vault_path,
        marc_lookup=lambda _i: _marc(),
        cce_lookup=lambda _u: _cce(was_renewed=False, renewal_rdat=None),
    )

    [entry] = list(iter_entries(vault_path))
    assert entry.was_renewed is False
    assert entry.renewal_year is None
    assert entry.reg_year == 1953


def test_enrich_learned_absent_sets_learned_none(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry())

    report = _enrich(
        vault_path,
        marc_lookup=lambda _i: _marc(),
        cce_lookup=lambda _u: _cce(),
        learned=None,
    )

    assert report.learned_scored == 0
    [entry] = list(iter_entries(vault_path))
    assert entry.scores is not None
    assert entry.scores.weighted_mean == 0.8421
    assert entry.scores.learned is None


def test_enrich_counts_marc_missing_from_pool(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry())

    report = _enrich(vault_path, marc_lookup=lambda _i: None, cce_lookup=lambda _u: _cce())

    assert report.missing_in_pool == 1
    assert report.enriched == 0
    [entry] = list(iter_entries(vault_path))
    assert entry.scores is None
    assert entry.reg_year is None


def test_enrich_counts_cce_missing_from_index(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry())

    report = _enrich(vault_path, marc_lookup=lambda _i: _marc(), cce_lookup=lambda _u: None)

    assert report.missing_in_index == 1
    assert report.enriched == 0
    [entry] = list(iter_entries(vault_path))
    assert entry.scores is None


def test_enrich_dry_run_reports_but_writes_nothing(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry())
    before = vault_path.read_bytes()

    report = _enrich(
        vault_path,
        marc_lookup=lambda _i: _marc(),
        cce_lookup=lambda _u: _cce(),
        dry_run=True,
    )

    assert report.enriched == 1
    assert vault_path.read_bytes() == before


def _make_collection(path: Path, control_id: str) -> Path:
    path.write_text(
        _MARCXML_TEMPLATE.format(ns=_MARC_NS, control_id=control_id),
        encoding="utf-8",
    )
    return path


def _make_pool(root: Path, control_id: str) -> Path:
    lang_dir = root / "eng"
    lang_dir.mkdir(parents=True)
    _make_collection(lang_dir / "shard_1.xml", control_id)
    return root


class _FakeLookup:
    def __init__(self, table: dict[str, IndexedNyplRegRecord]) -> None:
        self._table = table

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None: ...

    def get_registration(self, uuid: str) -> IndexedNyplRegRecord | None:
        return self._table.get(uuid)


def _patch_wiring(
    monkeypatch: MonkeyPatch,
    learned: _FakeCombiner | None,
    table: dict[str, IndexedNyplRegRecord],
) -> None:
    monkeypatch.setattr(module, "NyplIndexLookup", lambda _path: _FakeLookup(table))
    monkeypatch.setattr(
        module,
        "_make_evidence_scorer",
        lambda **_kwargs: lambda _m, _c: _evidence(),
    )
    monkeypatch.setattr(module, "build_combiner", lambda _config, **_kwargs: _FakeCombiner(0.5))
    monkeypatch.setattr(module, "_build_learned_combiner", lambda _config, _dir: learned)
    monkeypatch.setattr(module, "resolve_matcher_version", lambda: _VERSION)


def test_run_enrich_end_to_end_from_collection(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry("id-1", "uuid-1"))
    collection = _make_collection(tmp_path / "marc.xml", "id-1")
    index_path = tmp_path / "idx" / "cce.lmdb"
    index_path.parent.mkdir(parents=True)
    _patch_wiring(monkeypatch, _FakeCombiner(0.7), {"uuid-1": _cce("uuid-1")})

    report = run_enrich(
        vault_path=vault_path,
        index_path=index_path,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        marc_collection_path=collection,
    )

    assert report.enriched == 1
    [entry] = list(iter_entries(vault_path))
    assert entry.scores is not None
    assert entry.scores.weighted_mean == 0.5
    assert entry.scores.learned == 0.7


def test_run_enrich_end_to_end_from_pool(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry("id-1", "uuid-1"))
    pool = _make_pool(tmp_path / "pool", "id-1")
    index_path = tmp_path / "idx" / "cce.lmdb"
    index_path.parent.mkdir(parents=True)
    _patch_wiring(monkeypatch, None, {"uuid-1": _cce("uuid-1")})

    report = run_enrich(
        vault_path=vault_path,
        index_path=index_path,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        pool_path=pool,
    )

    assert report.enriched == 1
    [entry] = list(iter_entries(vault_path))
    assert entry.scores is not None
    assert entry.scores.learned is None


def test_run_enrich_rejects_both_marc_sources(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry("id-1", "uuid-1"))
    _patch_wiring(monkeypatch, None, {})

    with raises(ValueError, match="exactly one"):
        run_enrich(
            vault_path=vault_path,
            index_path=tmp_path / "cce.lmdb",
            matching_config=_load_default_matching_config(),
            pairing_config=_load_default_pairing_config(),
            pool_path=tmp_path / "pool",
            marc_collection_path=tmp_path / "marc.xml",
        )


def test_run_enrich_rejects_neither_marc_source(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    vault_path = tmp_path / "vault.jsonl"
    upsert_entry(vault_path, _entry("id-1", "uuid-1"))
    _patch_wiring(monkeypatch, None, {})

    with raises(ValueError, match="exactly one"):
        run_enrich(
            vault_path=vault_path,
            index_path=tmp_path / "cce.lmdb",
            matching_config=_load_default_matching_config(),
            pairing_config=_load_default_pairing_config(),
        )


def test_learned_config_forces_learned_scorer() -> None:
    config = _load_default_matching_config()
    learned = module._learned_config(config)
    assert learned.scorer == "learned"
    assert config.scorer == "weighted_mean"
    assert learned.title_weight == config.title_weight


def test_build_learned_combiner_returns_none_and_warns_when_absent(tmp_path: Path) -> None:
    config = module._learned_config(_load_default_matching_config())
    assert module._build_learned_combiner(config, tmp_path) is None


def test_make_evidence_scorer_returns_winning_evidence(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from pd_groundtruth.vault_pair_resolver import ScorePairFn
    from pd_matcher.match.result import CandidateMatch

    captured: dict[str, object] = {}

    def module_candidate(uuid: str) -> CandidateMatch:
        return CandidateMatch(
            nypl_uuid=uuid,
            nypl_year=1953,
            combined=CombinedScore(raw=90.0, calibrated=0.9),
            evidence=_evidence(),
            losing_evidence=(),
        )

    def fake_make_pair_scorer(**kwargs: object) -> ScorePairFn:
        captured.update(kwargs)

        def scorer(marc: MarcRecord, cce: IndexedNyplRegRecord) -> CandidateMatch:
            return module_candidate(cce.uuid)

        return scorer

    monkeypatch.setattr(module, "make_pair_scorer", fake_make_pair_scorer)
    monkeypatch.setattr(module, "load_or_build_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_author_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_publisher_idf", lambda *_a, **_k: object())

    scorer = module._make_evidence_scorer(
        matching_config=_load_default_matching_config(),
        index_path=tmp_path / "cce.lmdb",
        pairing_config=_load_default_pairing_config(),
    )
    evidence = scorer(_marc(), _cce())
    assert evidence == _evidence()
    assert captured["calibrator"] is None


def test_cli_enrich_vault_invokes_run_enrich_and_reports(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    report = EnrichReport(
        total_entries=10,
        enriched=8,
        missing_in_pool=1,
        missing_in_index=1,
        learned_scored=8,
    )
    calls: dict[str, object] = {}

    def fake_run_enrich(**kwargs: object) -> EnrichReport:
        calls.update(kwargs)
        return report

    monkeypatch.setattr("pd_groundtruth.cli.run_enrich", fake_run_enrich)
    result = _RUNNER.invoke(
        app,
        [
            "enrich-vault",
            "--vault",
            str(tmp_path / "vault.jsonl"),
            "--index",
            str(tmp_path / "cce.lmdb"),
        ],
    )

    assert result.exit_code == 0
    assert calls["marc_collection_path"] is not None
    assert calls["pool_path"] is None
    assert calls["dry_run"] is False
    assert "enriched 8/10 entries" in result.stdout
    assert "learned_scored=8" in result.stdout
    assert "1 MARC records not found in pool" in result.stdout
    assert "1 CCE records not found in index" in result.stdout


def test_cli_enrich_vault_pool_overrides_collection(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    report = EnrichReport(
        total_entries=1, enriched=1, missing_in_pool=0, missing_in_index=0, learned_scored=0
    )
    calls: dict[str, object] = {}

    def fake_run_enrich(**kwargs: object) -> EnrichReport:
        calls.update(kwargs)
        return report

    monkeypatch.setattr("pd_groundtruth.cli.run_enrich", fake_run_enrich)
    result = _RUNNER.invoke(
        app,
        [
            "enrich-vault",
            "--vault",
            str(tmp_path / "vault.jsonl"),
            "--index",
            str(tmp_path / "cce.lmdb"),
            "--pool",
            str(tmp_path / "pool"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert calls["pool_path"] == tmp_path / "pool"
    assert calls["marc_collection_path"] is None
    assert calls["dry_run"] is True
    assert "(dry-run) enriched 1/1 entries" in result.stdout


def test_cli_enrich_vault_honors_explicit_log_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    report = EnrichReport(
        total_entries=0, enriched=0, missing_in_pool=0, missing_in_index=0, learned_scored=0
    )
    monkeypatch.setattr("pd_groundtruth.cli.run_enrich", lambda **_kwargs: report)
    target = tmp_path / "explicit.log"
    result = _RUNNER.invoke(
        app,
        [
            "enrich-vault",
            "--vault",
            str(tmp_path / "vault.jsonl"),
            "--index",
            str(tmp_path / "cce.lmdb"),
            "--log-file",
            str(target),
        ],
    )
    assert result.exit_code == 0
    assert target.exists()
