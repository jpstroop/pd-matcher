"""Unit tests for the ``vault-into-queue`` backfill command and helpers."""

from pathlib import Path
from typing import Self
from unittest.mock import patch

from msgspec.json import encode as json_encode
from typer.testing import CliRunner

from pd_groundtruth.cli import app
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import append_entry
from pd_groundtruth.review.field_annotations import JUDGMENT_UNDERSCORED
from pd_groundtruth.review.field_annotations import FieldAnnotation
from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.vault_into_queue import BackfillSummary
from pd_groundtruth.vault_into_queue import _make_pair_scorer
from pd_groundtruth.vault_into_queue import build_marc_index
from pd_groundtruth.vault_into_queue import run_backfill
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.copyright.coverage import LEGACY_COVERAGE
from pd_matcher.copyright.coverage import Coverage
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
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
    '<datafield tag="245" ind1="0" ind2="0"><subfield code="a">{title}</subfield></datafield>'
    "</record>"
    "</collection>"
)


def _marc(control_id: str = "ctrl-1") -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A Title",
        title_main="A Title",
        main_author="An Author",
        statement_of_responsibility="by An Author",
        publisher="A Publisher",
        publication_year=1953,
        language_code="eng",
    )


def _marc_json(control_id: str) -> str:
    return json_encode(
        MarcRecord(
            control_id=control_id,
            title="A Title",
            title_main="A Title",
        )
    ).decode("utf-8")


def _pair(control_id: str, nypl_uuid: str) -> PairInsert:
    return PairInsert(
        language="eng",
        decade=1950,
        score=0.95,
        band="ge90",
        source="banded",
        marc_control_id=control_id,
        marc_json=_marc_json(control_id),
        marc_title="A Title",
        marc_author=None,
        marc_publisher=None,
        marc_year=1953,
        nypl_uuid=nypl_uuid,
        cce_title="CCE Title",
        cce_author=None,
        cce_publishers=None,
        cce_claimants=None,
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json="{}",
    )


def _cce(uuid: str = "uuid-1") -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid,
        title="CCE Title",
        was_renewed=True,
        regnum="R123",
        reg_year=1953,
        author_name="CCE Author",
        publisher_names=("Pub A", "Pub B"),
        claimants=("Claimant A",),
    )


def _evidence(scorer: str, score: float, *, skipped: bool = False) -> Evidence:
    return Evidence(
        scorer=scorer,
        score=score,
        max=1.0,
        skipped=skipped,
        decisive=False,
        features=(),
    )


def _candidate(score: float, uuid: str = "uuid-1") -> CandidateMatch:
    return CandidateMatch(
        nypl_uuid=uuid,
        nypl_year=1953,
        combined=CombinedScore(raw=score * 100.0, calibrated=score),
        evidence=(_evidence("title.token_set", 0.9),),
        losing_evidence=(),
    )


def _vault_entry(
    control_id: str,
    nypl_uuid: str,
    *,
    verdict: str = VERDICT_MATCH,
    labeled_at: str = "2026-05-22T10:00:00+00:00",
    reasons: tuple[str, ...] = (),
    note: str | None = None,
    field_annotations: tuple[FieldAnnotation, ...] = (),
) -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        reasons=reasons,
        note=note,
        labeled_at=labeled_at,
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
        field_annotations=field_annotations,
    )


def _write_shard(path: Path, control_id: str, title: str = "A Title") -> None:
    path.write_text(
        _MARCXML_TEMPLATE.format(ns=_MARC_NS, control_id=control_id, title=title),
        encoding="utf-8",
    )


def _make_pool(root: Path, control_ids_by_lang: dict[str, list[str]]) -> Path:
    for language, ids in control_ids_by_lang.items():
        lang_dir = root / language
        lang_dir.mkdir(parents=True)
        for index, control_id in enumerate(ids, start=1):
            _write_shard(lang_dir / f"shard_{index}.xml", control_id)
    return root


def test_build_marc_index_resolves_wanted_ids_across_shards(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path / "pool", {"eng": ["id-1", "id-2"], "fre": ["id-3"]})
    found = build_marc_index(pool, {"id-1", "id-3"})
    assert set(found.keys()) == {"id-1", "id-3"}
    assert found["id-1"].control_id == "id-1"
    assert found["id-3"].control_id == "id-3"


def test_build_marc_index_returns_partial_when_some_ids_absent(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path / "pool", {"eng": ["id-1"]})
    found = build_marc_index(pool, {"id-1", "id-missing"})
    assert set(found.keys()) == {"id-1"}


def test_build_marc_index_short_circuits_on_empty_request(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    assert build_marc_index(pool, set()) == {}


def test_run_backfill_inserts_pair_and_label_for_missing_vault_entry(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass

    vault = {
        ("ctrl-a", "uuid-a"): _vault_entry(
            "ctrl-a",
            "uuid-a",
            verdict=VERDICT_NO_MATCH,
            note="off",
            reasons=("diff_work",),
            labeled_at="2026-05-22T11:00:00+00:00",
        )
    }
    marc_table = {"ctrl-a": _marc("ctrl-a")}
    cce_table = {"uuid-a": _cce("uuid-a")}

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup=marc_table.get,
        cce_lookup=cce_table.get,
        score_pair=lambda _m, _c: _candidate(0.95, "uuid-a"),
    )

    assert summary == BackfillSummary(
        backfilled=1, already_present=0, missing_in_pool=0, missing_in_index=0
    )
    with ReviewDb.connect(db_path) as db:
        progress = db.progress()
        labels = list(db.iter_current_labels())
    assert progress.total == 1
    assert progress.labeled == 1
    assert progress.no_match == 1
    assert len(labels) == 1
    only = labels[0]
    assert only.marc_control_id == "ctrl-a"
    assert only.nypl_uuid == "uuid-a"
    assert only.verdict == VERDICT_NO_MATCH
    assert only.labeled_at == "2026-05-22T11:00:00+00:00"
    assert only.note == "off"
    assert only.reasons == ("diff_work",)


def test_run_backfill_skips_pairs_already_in_db(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair("ctrl-a", "uuid-a"))

    vault = {("ctrl-a", "uuid-a"): _vault_entry("ctrl-a", "uuid-a")}

    def _fail_marc(_id: str) -> MarcRecord | None:
        raise AssertionError("MARC lookup should not run for already-present pair")

    def _fail_cce(_uuid: str) -> IndexedNyplRegRecord | None:
        raise AssertionError("CCE lookup should not run for already-present pair")

    def _fail_score(_m: MarcRecord, _c: IndexedNyplRegRecord) -> CandidateMatch:
        raise AssertionError("scorer should not run for already-present pair")

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup=_fail_marc,
        cce_lookup=_fail_cce,
        score_pair=_fail_score,
    )
    assert summary == BackfillSummary(
        backfilled=0, already_present=1, missing_in_pool=0, missing_in_index=0
    )
    with ReviewDb.connect(db_path) as db:
        assert db.stratum_counts() == {("eng", "ge90"): 1}


def test_run_backfill_reports_marc_missing_from_pool(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    vault = {("ctrl-gone", "uuid-gone"): _vault_entry("ctrl-gone", "uuid-gone")}

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup=lambda _id: None,
        cce_lookup=lambda _u: _cce("uuid-gone"),
        score_pair=lambda _m, _c: _candidate(0.95),
    )

    assert summary == BackfillSummary(
        backfilled=0, already_present=0, missing_in_pool=1, missing_in_index=0
    )
    with ReviewDb.connect(db_path) as db:
        assert db.stratum_counts() == {}


def test_run_backfill_reports_cce_missing_from_index(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    vault = {("ctrl-a", "uuid-gone"): _vault_entry("ctrl-a", "uuid-gone")}

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup={"ctrl-a": _marc("ctrl-a")}.get,
        cce_lookup=lambda _u: None,
        score_pair=lambda _m, _c: _candidate(0.95),
    )

    assert summary == BackfillSummary(
        backfilled=0, already_present=0, missing_in_pool=0, missing_in_index=1
    )
    with ReviewDb.connect(db_path) as db:
        assert db.stratum_counts() == {}


def test_run_backfill_uses_latest_vault_entry_per_key(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    vault_path = tmp_path / "vault.jsonl"
    append_entry(
        vault_path,
        _vault_entry(
            "ctrl-a",
            "uuid-a",
            verdict=VERDICT_NO_MATCH,
            labeled_at="2026-05-22T09:00:00+00:00",
        ),
    )
    append_entry(
        vault_path,
        _vault_entry(
            "ctrl-a",
            "uuid-a",
            verdict=VERDICT_MATCH,
            labeled_at="2026-05-22T12:00:00+00:00",
        ),
    )
    from pd_groundtruth.label_vault import current_entries

    vault = current_entries(vault_path)
    assert len(vault) == 1

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup={"ctrl-a": _marc("ctrl-a")}.get,
        cce_lookup={"uuid-a": _cce("uuid-a")}.get,
        score_pair=lambda _m, _c: _candidate(0.95, "uuid-a"),
    )

    assert summary.backfilled == 1
    with ReviewDb.connect(db_path) as db:
        labels = list(db.iter_current_labels())
    assert len(labels) == 1
    assert labels[0].verdict == VERDICT_MATCH
    assert labels[0].labeled_at == "2026-05-22T12:00:00+00:00"


def test_run_backfill_assigns_score_and_band_from_scorer(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    vault = {("ctrl-a", "uuid-a"): _vault_entry("ctrl-a", "uuid-a")}

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup={"ctrl-a": _marc("ctrl-a")}.get,
        cce_lookup={"uuid-a": _cce("uuid-a")}.get,
        score_pair=lambda _m, _c: _candidate(0.42, "uuid-a"),
    )

    assert summary.backfilled == 1
    with ReviewDb.connect(db_path) as db:
        row = db.get_pair(1)
    assert row is not None
    assert row.score == 0.42
    assert row.band == "below"
    assert row.source == "banded"


def test_run_backfill_on_empty_vault_reports_zero(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    summary = run_backfill(
        db_path=db_path,
        vault={},
        marc_lookup=lambda _id: None,
        cce_lookup=lambda _u: None,
        score_pair=lambda _m, _c: _candidate(0.95),
    )
    assert summary == BackfillSummary(
        backfilled=0, already_present=0, missing_in_pool=0, missing_in_index=0
    )


def test_run_backfill_when_all_vault_pairs_present_skips_resource_setup(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair("ctrl-a", "uuid-a"))
    vault = {("ctrl-a", "uuid-a"): _vault_entry("ctrl-a", "uuid-a")}

    def _boom_marc(_id: str) -> MarcRecord | None:
        raise AssertionError("should not be called when nothing is missing")

    def _boom_cce(_uuid: str) -> IndexedNyplRegRecord | None:
        raise AssertionError("should not be called when nothing is missing")

    def _boom_score(_m: MarcRecord, _c: IndexedNyplRegRecord) -> CandidateMatch:
        raise AssertionError("should not be called when nothing is missing")

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup=_boom_marc,
        cce_lookup=_boom_cce,
        score_pair=_boom_score,
    )
    assert summary == BackfillSummary(
        backfilled=0, already_present=1, missing_in_pool=0, missing_in_index=0
    )


def test_cli_vault_into_queue_invokes_backfill_and_reports(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    vault_path = tmp_path / "vault.jsonl"
    pool_path = tmp_path / "pool"
    index_path = tmp_path / "nypl.lmdb"

    summary = BackfillSummary(
        backfilled=3, already_present=4, missing_in_pool=1, missing_in_index=2
    )
    with patch("pd_groundtruth.cli.vault_into_queue", return_value=summary) as mock_backfill:
        result = _RUNNER.invoke(
            app,
            [
                "vault-into-queue",
                "--db",
                str(db_path),
                "--vault",
                str(vault_path),
                "--pool",
                str(pool_path),
                "--index",
                str(index_path),
            ],
        )

    assert result.exit_code == 0
    _, kwargs = mock_backfill.call_args
    assert kwargs["db_path"] == db_path
    assert kwargs["vault_path"] == vault_path
    assert kwargs["pool_path"] == pool_path
    assert kwargs["index_path"] == index_path
    assert "backfilled 3 vault pairs" in result.stdout
    assert "1 MARC records not found in pool" in result.stdout
    assert "2 CCE records not found in index" in result.stdout
    assert "4 already present" in result.stdout


def test_cli_vault_into_queue_honors_explicit_log_file(tmp_path: Path) -> None:
    target = tmp_path / "explicit.log"
    summary = BackfillSummary(
        backfilled=0, already_present=0, missing_in_pool=0, missing_in_index=0
    )
    with patch("pd_groundtruth.cli.vault_into_queue", return_value=summary):
        result = _RUNNER.invoke(
            app,
            [
                "vault-into-queue",
                "--db",
                str(tmp_path / "review.db"),
                "--vault",
                str(tmp_path / "vault.jsonl"),
                "--pool",
                str(tmp_path / "pool"),
                "--index",
                str(tmp_path / "nypl.lmdb"),
                "--log-file",
                str(target),
            ],
        )
    assert result.exit_code == 0
    assert target.exists()


def test_vault_into_queue_end_to_end_against_fake_index(tmp_path: Path) -> None:
    from pd_groundtruth import vault_into_queue as module_under_test

    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    vault_path = tmp_path / "vault.jsonl"
    append_entry(vault_path, _vault_entry("id-1", "uuid-1"))
    append_entry(
        vault_path,
        _vault_entry("id-missing", "uuid-2", verdict=VERDICT_NO_MATCH),
    )

    pool = _make_pool(tmp_path / "pool", {"eng": ["id-1"]})
    index_path = tmp_path / "idx" / "nypl.lmdb"
    index_path.parent.mkdir(parents=True)

    cce_table = {"uuid-1": _cce("uuid-1"), "uuid-2": _cce("uuid-2")}

    class _FakeLookup:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None: ...

        def get_registration(self, uuid: str) -> IndexedNyplRegRecord | None:
            return cce_table.get(uuid)

        def coverage(self) -> Coverage:
            return LEGACY_COVERAGE

    with (
        patch.object(module_under_test, "NyplIndexLookup", lambda _path: _FakeLookup()),
        patch.object(module_under_test, "load_or_build_idf", lambda *_a, **_k: object()),
        patch.object(module_under_test, "_load_calibrator", lambda _p: None),
        patch.object(
            module_under_test,
            "_make_pair_scorer",
            lambda **_kwargs: lambda _m, _c: _candidate(0.95),
        ),
    ):
        summary = module_under_test.vault_into_queue(
            db_path=db_path,
            vault_path=vault_path,
            pool_path=pool,
            index_path=index_path,
            matching_config=_load_default_matching_config(),
            pairing_config=_load_default_pairing_config(),
        )

    assert summary.backfilled == 1
    assert summary.missing_in_pool == 1
    assert summary.missing_in_index == 0
    assert summary.already_present == 0
    with ReviewDb.connect(db_path) as db:
        rows = list(db.iter_current_labels())
    assert len(rows) == 1
    assert rows[0].marc_control_id == "id-1"


def test_make_pair_scorer_delegates_to_shared_helper() -> None:
    from pd_matcher.match.idf import IdfTable
    from pd_matcher.match.pairing_compiler import compile_pairings

    idf = IdfTable(
        document_count=1,
        default_idf=1.0,
        source_hash="test",
        language="eng",
        idf={},
    )
    pairings = compile_pairings(_load_default_pairing_config())
    scorer = _make_pair_scorer(
        matching_config=_load_default_matching_config(),
        pairings=pairings,
        idf=idf,
        calibrator=None,
    )
    candidate = scorer(_marc("ctrl-1"), _cce("uuid-1"))
    assert candidate.nypl_uuid == "uuid-1"


def test_run_backfill_carries_field_annotations_into_db(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass

    annotations = (FieldAnnotation(field="author", judgment=JUDGMENT_UNDERSCORED),)
    vault = {
        ("ctrl-a", "uuid-a"): _vault_entry(
            "ctrl-a",
            "uuid-a",
            verdict=VERDICT_NO_MATCH,
            reasons=("diff_work",),
            field_annotations=annotations,
        )
    }

    summary = run_backfill(
        db_path=db_path,
        vault=vault,
        marc_lookup={"ctrl-a": _marc("ctrl-a")}.get,
        cce_lookup={"uuid-a": _cce("uuid-a")}.get,
        score_pair=lambda _m, _c: _candidate(0.95, "uuid-a"),
    )
    assert summary.backfilled == 1

    with ReviewDb.connect(db_path) as db:
        [label] = list(db.iter_current_labels())
    assert label.field_annotations == annotations


def test_vault_into_queue_with_empty_vault_returns_zero_summary(tmp_path: Path) -> None:
    from pd_groundtruth import vault_into_queue as module_under_test

    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass

    summary = module_under_test.vault_into_queue(
        db_path=db_path,
        vault_path=tmp_path / "vault.jsonl",
        pool_path=tmp_path / "pool",
        index_path=tmp_path / "idx" / "nypl.lmdb",
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
    )
    assert summary == BackfillSummary(
        backfilled=0, already_present=0, missing_in_pool=0, missing_in_index=0
    )
