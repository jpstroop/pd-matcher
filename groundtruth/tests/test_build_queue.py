"""Unit tests for the review-queue builder orchestration.

No network, no LMDB, no real matching: the matcher is replaced by a
monkeypatched ``run_match`` and the :class:`StratifyingResultWriter` is
exercised directly with fabricated :class:`MarcRecord`,
:class:`MatchResult`, and :class:`IndexedNyplRegRecord` objects against a
temporary SQLite review database.
"""

from datetime import date
from pathlib import Path
from pickle import loads as pickle_loads

from msgspec.json import decode as json_decode
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.coverage import LEGACY_COVERAGE
from pd_matcher.copyright.coverage import Coverage
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.prepare import read_manifest
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.workers import RunReport
from pytest import MonkeyPatch
from pytest import raises

from pd_groundtruth import build_queue as bq
from pd_groundtruth.build_queue import BuildSummary
from pd_groundtruth.build_queue import StratifyingResultWriter
from pd_groundtruth.build_queue import StratifyingWriterFactory
from pd_groundtruth.build_queue import _decade_of
from pd_groundtruth.build_queue import _evidence_payload
from pd_groundtruth.build_queue import _iso_or_none
from pd_groundtruth.build_queue import _iter_language_dirs
from pd_groundtruth.build_queue import _join
from pd_groundtruth.build_queue import _join_notes
from pd_groundtruth.build_queue import _join_places
from pd_groundtruth.build_queue import _join_prev_regnums
from pd_groundtruth.build_queue import _sample_language
from pd_groundtruth.build_queue import _write_sample_chunks
from pd_groundtruth.build_queue import build_queue
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import BudgetModel
from pd_groundtruth.vault_pair_resolver import ResolvedVaultPair
from pd_groundtruth.vault_pair_resolver import ResolveSummary

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

_MATCHING_CONFIG: MatchingConfig = _load_default_matching_config()
_PAIRING_CONFIG = _load_default_pairing_config()
_RULESET: CopyrightRuleSet = bq.load_default_ruleset()
_COPYRIGHT_CONFIG = CopyrightAssessmentConfig(as_of_year=2024)

_ASSESSMENT = CopyrightAssessment(
    status=CopyrightStatus.PD_REGISTERED_NOT_RENEWED,
    matched_rule_name=None,
    explanation="",
    assumptions=(),
)


def _write_shard(path: Path, control_id: str, title: str) -> None:
    path.write_text(
        _MARCXML_TEMPLATE.format(ns=_MARC_NS, control_id=control_id, title=title),
        encoding="utf-8",
    )


def _marc(control_id: str = "ctrl-1", year: int | None = 1953) -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A Title",
        title_main="A Title",
        main_author="An Author",
        statement_of_responsibility="by An Author",
        publisher="A Publisher",
        publication_year=year,
        language_code="eng",
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


def _cce(uuid: str = "uuid-1") -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid,
        title="CCE Title",
        was_renewed=True,
        regnum="R123",
        reg_year=1953,
        author_name="CCE Author",
        author_place="Cambridge, Mass.",
        author_is_claimant=True,
        edition="2nd ed.",
        publisher_names=("Pub A", "Pub B"),
        publication_places=("New York", "London"),
        claimants=("Claimant A",),
        copies="2c.",
        aff_date=date(1953, 6, 1),
        desc="vi, 200 p.",
        notes=("note one", "note two"),
        new_matter_claimed="added ch. 5",
        copy_date=date(1953, 4, 1),
        notice_date=date(1953, 4, 2),
        lccn="28000854",
        prev_regnums=("A100000", "A200000"),
    )


def _match(score: float, *, uuid: str = "uuid-1") -> MatchResult:
    best = CandidateMatch(
        nypl_uuid=uuid,
        nypl_year=1953,
        combined=CombinedScore(raw=score * 100.0, calibrated=score),
        evidence=(
            _evidence("title.token_set", 0.9),
            _evidence("lccn.exact", 0.0, skipped=True),
        ),
        losing_evidence=(),
    )
    return MatchResult(marc_control_id="ctrl-1", best=best, alternates=(), candidates_considered=3)


def test_decade_of_buckets_years() -> None:
    assert _decade_of(1953) == 1950
    assert _decade_of(1960) == 1960
    assert _decade_of(None) is None


def test_join_collapses_empty_to_none() -> None:
    assert _join(()) is None
    assert _join(("a",)) == "a"
    assert _join(("a", "b")) == "a | b"


def test_join_places_uses_semicolon_separator() -> None:
    assert _join_places(()) is None
    assert _join_places(("NY",)) == "NY"
    assert _join_places(("NY", "London")) == "NY; London"


def test_join_notes_uses_newline_separator() -> None:
    assert _join_notes(()) is None
    assert _join_notes(("one",)) == "one"
    assert _join_notes(("one", "two")) == "one\ntwo"


def test_join_prev_regnums_uses_semicolon_separator() -> None:
    assert _join_prev_regnums(()) is None
    assert _join_prev_regnums(("A100000",)) == "A100000"
    assert _join_prev_regnums(("A100000", "A200000")) == "A100000; A200000"


def test_iso_or_none_formats_date_or_returns_none() -> None:
    assert _iso_or_none(None) is None
    assert _iso_or_none(date(1953, 6, 1)) == "1953-06-01"


def test_evidence_payload_drops_skipped() -> None:
    payload = _evidence_payload(
        (_evidence("title.token_set", 0.9), _evidence("lccn.exact", 0.0, skipped=True))
    )
    assert payload == {"title.token_set": 0.9}


def test_iter_language_dirs_yields_only_subdirs(tmp_path: Path) -> None:
    (tmp_path / "eng").mkdir()
    (tmp_path / "fre").mkdir()
    (tmp_path / "stray.txt").write_text("ignore", encoding="utf-8")
    pairs = list(_iter_language_dirs(tmp_path))
    assert [name for name, _ in pairs] == ["eng", "fre"]
    assert all(path.is_dir() for _, path in pairs)


def test_sample_language_parses_shards_without_lmdb(tmp_path: Path) -> None:
    lang_dir = tmp_path / "eng"
    lang_dir.mkdir()
    _write_shard(lang_dir / "shard_1.xml", "id-1", "First Title")
    _write_shard(lang_dir / "shard_2.xml", "id-2", "Second Title")
    records = _sample_language(lang_dir, sample_per_lang=10, seed=1)
    assert {record.control_id for record in records} == {"id-1", "id-2"}


def test_write_sample_chunks_round_trips_via_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "prepared"
    records = [_marc(control_id="a"), _marc(control_id="b")]
    manifest = _write_sample_chunks(records, out_dir)
    assert manifest.total_records == 2
    assert read_manifest(out_dir).total_records == 2
    with (out_dir / manifest.chunk_files[0]).open("rb") as handle:
        decoded: tuple[MarcRecord, ...] = pickle_loads(handle.read())
    assert [record.control_id for record in decoded] == ["a", "b"]


def test_factory_is_picklable_and_builds_writer(tmp_path: Path) -> None:
    factory = StratifyingWriterFactory(
        db_path=tmp_path / "review.db",
        budget=BudgetModel(caps={("eng", "ge90"): 1}),
        seed=7,
        vault_path=tmp_path / "vault.jsonl",
    )
    restored: StratifyingWriterFactory = pickle_loads(__import__("pickle").dumps(factory))
    writer = restored(tmp_path / "ignored.csv")
    assert isinstance(writer, StratifyingResultWriter)


def test_writer_accepts_banded_until_cap(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 2})
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=1) as writer:
        for index in range(4):
            writer.write(_marc(control_id=f"c{index}"), _match(0.95), _ASSESSMENT, _cce())
    with ReviewDb.connect(db_path) as db:
        counts = db.stratum_counts()
    assert counts[("eng", "ge90")] == 2


def test_writer_preapplies_vault_labels_for_known_pairs(tmp_path: Path) -> None:
    from pd_groundtruth.label_vault import SCHEMA_VERSION
    from pd_groundtruth.label_vault import MarcIdentifiers
    from pd_groundtruth.label_vault import VaultEntry

    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 2})
    vault_entry = VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id="c0",
        nypl_uuid="uuid-1",
        verdict="match",
        reasons=(),
        note="from vault",
        labeled_at="2026-05-22T10:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )
    vault = {("c0", "uuid-1"): vault_entry}
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=1, vault=vault) as writer:
        writer.write(_marc(control_id="c0"), _match(0.95), _ASSESSMENT, _cce())
        writer.write(_marc(control_id="c1"), _match(0.95), _ASSESSMENT, _cce("uuid-other"))

    with ReviewDb.connect(db_path) as db:
        progress = db.progress()
        assert progress.total == 2
        assert progress.labeled == 1
        assert progress.match == 1
        labels = list(db.iter_current_labels())
    assert len(labels) == 1
    assert labels[0].marc_control_id == "c0"
    assert labels[0].verdict == "match"
    assert labels[0].labeled_at == "2026-05-22T10:00:00+00:00"
    assert labels[0].note == "from vault"


def test_writer_preapplies_vault_labels_to_below_sample(tmp_path: Path) -> None:
    from pd_groundtruth.label_vault import SCHEMA_VERSION
    from pd_groundtruth.label_vault import MarcIdentifiers
    from pd_groundtruth.label_vault import VaultEntry

    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "below"): 2})
    vault_entry = VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id="c0",
        nypl_uuid="uuid-1",
        verdict="no_match",
        reasons=("diff_work",),
        note=None,
        labeled_at="2026-05-22T11:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )
    vault = {("c0", "uuid-1"): vault_entry}
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=42, vault=vault) as writer:
        writer.write(_marc(control_id="c0"), _match(0.3), _ASSESSMENT, _cce())
        writer.write(_marc(control_id="c1"), _match(0.3), _ASSESSMENT, _cce("uuid-other"))

    with ReviewDb.connect(db_path) as db:
        labels = list(db.iter_current_labels())
    found = {label.marc_control_id for label in labels}
    assert "c0" in found


def test_writer_persists_snapshot_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 1})
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=1) as writer:
        writer.write(_marc(control_id="c0"), _match(0.95), _ASSESSMENT, _cce())
    with ReviewDb.connect(db_path) as db:
        row = db.next_unlabeled()
    assert row is not None
    assert row.marc_control_id == "c0"
    assert json_decode(row.marc_json.encode("utf-8"), type=MarcRecord).control_id == "c0"
    assert row.cce_title == "CCE Title"
    assert row.cce_publishers == "Pub A | Pub B"
    assert row.cce_was_renewed == 1
    assert row.cce_regnum == "R123"
    assert json_decode(row.evidence_json.encode("utf-8")) == {"title.token_set": 0.9}
    assert row.source == "banded"
    assert row.band == "ge90"
    assert row.decade == 1950
    assert row.cce_edition == "2nd ed."
    assert row.cce_publication_places == "New York; London"
    assert row.cce_author_place == "Cambridge, Mass."
    assert row.cce_author_is_claimant == 1
    assert row.cce_copies == "2c."
    assert row.cce_aff_date == "1953-06-01"
    assert row.cce_desc == "vi, 200 p."
    assert row.cce_notes == "note one\nnote two"
    assert row.cce_new_matter_claimed == "added ch. 5"
    assert row.cce_copy_date == "1953-04-01"
    assert row.cce_notice_date == "1953-04-02"
    assert row.cce_lccn == "28000854"
    assert row.cce_prev_regnums == "A100000; A200000"
    assert row.cce_predicted_status == "PD_REGISTERED_NOT_RENEWED"


def test_writer_persists_renewal_projection_when_matched_nypl_carries_it(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 1})
    cce_with_renewal = IndexedNyplRegRecord(
        uuid="uuid-r",
        title="CCE Title",
        was_renewed=True,
        regnum="R123",
        reg_year=1953,
        renewal_id="R200001",
        renewal_oreg="R123",
        renewal_rdat=date(1981, 6, 1),
        renewal_author="Author A",
        renewal_title="Title A",
        renewal_claimants="Acme Press|PWH",
        renewal_new_matter="added ch. 7",
    )
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=1) as writer:
        writer.write(
            _marc(control_id="cr"), _match(0.95, uuid="uuid-r"), _ASSESSMENT, cce_with_renewal
        )
    with ReviewDb.connect(db_path) as db:
        row = db.next_unlabeled()
    assert row is not None
    assert row.cce_renewal_id == "R200001"
    assert row.cce_renewal_oreg == "R123"
    assert row.cce_renewal_rdat == "1981-06-01"
    assert row.cce_renewal_author == "Author A"
    assert row.cce_renewal_title == "Title A"
    assert row.cce_renewal_claimants == "Acme Press|PWH"
    assert row.cce_renewal_new_matter == "added ch. 7"


def test_writer_persists_predicted_status_from_assessment(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 1})
    in_copyright_assessment = CopyrightAssessment(
        status=CopyrightStatus.IN_COPYRIGHT_REGISTERED_AND_RENEWED,
        matched_rule_name=None,
        explanation="",
        assumptions=(),
    )
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=1) as writer:
        writer.write(_marc(control_id="cs"), _match(0.95), in_copyright_assessment, _cce())
    with ReviewDb.connect(db_path) as db:
        row = db.next_unlabeled()
    assert row is not None
    assert row.cce_predicted_status == "IN_COPYRIGHT_REGISTERED_AND_RENEWED"


def test_writer_below_sample_reservoir_caps_on_close(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "below"): 2})
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=42) as writer:
        for index in range(6):
            writer.write(_marc(control_id=f"c{index}"), _match(0.3), _ASSESSMENT, _cce())
        with ReviewDb.connect(db_path) as mid:
            assert mid.stratum_counts() == {}
    with ReviewDb.connect(db_path) as db:
        counts = db.stratum_counts()
        rows = [db.next_unlabeled()]
    assert counts[("eng", "below")] == 2
    assert rows[0] is not None
    assert rows[0].source == "below_sample"


def test_writer_skips_when_match_or_nypl_is_none(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 5, ("eng", "below"): 5})
    empty = MatchResult(marc_control_id="ctrl-1", best=None, alternates=(), candidates_considered=0)
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=1) as writer:
        writer.write(_marc(), None, _ASSESSMENT, _cce())
        writer.write(_marc(), empty, _ASSESSMENT, _cce())
        writer.write(_marc(), _match(0.95), _ASSESSMENT, None)
    with ReviewDb.connect(db_path) as db:
        assert db.stratum_counts() == {}


def test_writer_logs_fill_at_interval_for_banded_inserts(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(bq, "_FILL_LOG_INTERVAL", 3)
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 5})
    with StratifyingResultWriter(db_path=db_path, budget=budget, seed=1) as writer:
        for index in range(3):
            writer.write(_marc(control_id=f"c{index}"), _match(0.95), _ASSESSMENT, _cce())
    with ReviewDb.connect(db_path) as db:
        assert db.stratum_counts() == {("eng", "ge90"): 3}


def test_writer_exit_with_exception_does_not_commit_or_inject_vault(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 5})
    resolved = (
        ResolvedVaultPair(
            entry=VaultEntry(
                schema=SCHEMA_VERSION,
                marc_control_id="ctrl-a",
                nypl_uuid="uuid-a",
                verdict="match",
                reasons=(),
                note=None,
                labeled_at="2026-05-22T10:00:00+00:00",
                labeler="jpstroop",
                marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
            ),
            pair=PairInsert(
                language="eng",
                decade=1950,
                score=0.95,
                band="ge90",
                source="banded",
                marc_control_id="ctrl-a",
                marc_json="{}",
                marc_title="t",
                marc_author=None,
                marc_publisher=None,
                marc_year=1953,
                nypl_uuid="uuid-a",
                cce_title="t",
                cce_author=None,
                cce_publishers=None,
                cce_claimants=None,
                cce_reg_year=1953,
                cce_was_renewed=True,
                cce_regnum="R1",
                evidence_json="{}",
            ),
        ),
    )
    with (
        raises(RuntimeError, match="boom"),
        StratifyingResultWriter(db_path=db_path, budget=budget, seed=1, vault_pairs=resolved),
    ):
        raise RuntimeError("boom")
    with ReviewDb.connect(db_path) as db:
        assert db.stratum_counts() == {}


def test_writer_requires_context_manager(tmp_path: Path) -> None:
    writer = StratifyingResultWriter(
        db_path=tmp_path / "review.db",
        budget=BudgetModel(caps={}),
        seed=1,
    )
    with raises(RuntimeError, match="not entered"):
        writer.write(_marc(), _match(0.95), _ASSESSMENT, _cce())


def test_build_queue_rejects_zero_workers(tmp_path: Path) -> None:
    with raises(ValueError, match="workers must be >= 1"):
        build_queue(
            pool=tmp_path,
            index_path=tmp_path / "idx",
            out_path=tmp_path / "out.db",
            vault_path=tmp_path / "vault.jsonl",
            budget=BudgetModel(caps={}),
            matching_config=_MATCHING_CONFIG,
            pairing_config=_PAIRING_CONFIG,
            ruleset=_RULESET,
            copyright_config=_COPYRIGHT_CONFIG,
            seed=1,
            workers=0,
            sample_per_lang=10,
        )


def test_build_queue_drives_run_match_and_summarizes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    _write_shard(pool / "eng" / "shard_1.xml", "id-1", "Title One")

    monkeypatch.setattr(bq, "load_or_build_idf", lambda *a, **k: object())
    monkeypatch.setattr(bq, "_load_calibrator", lambda *a, **k: None)

    captured: dict[str, object] = {}

    def _fake_run_match(**kwargs: object) -> RunReport:
        captured.update(kwargs)
        factory = kwargs["writer_factory"]
        assert callable(factory)
        prepared_dir = kwargs["prepared_dir"]
        assert isinstance(prepared_dir, Path)
        assert (prepared_dir / "manifest.json").exists()
        writer = factory(tmp_path / "ignored.csv")
        with writer as active:
            active.write(_marc(control_id="id-1"), _match(0.95), _ASSESSMENT, _cce())
            active.write(_marc(control_id="id-2"), _match(0.3), _ASSESSMENT, _cce("uuid-1"))
        return RunReport(
            records_processed=2,
            records_written=2,
            records_enqueued=2,
            duration_seconds=0.1,
            by_status={},
            interrupted=False,
        )

    monkeypatch.setattr(bq, "run_match", _fake_run_match)

    out_path = tmp_path / "review.db"
    summary = build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=out_path,
        vault_path=tmp_path / "vault.jsonl",
        budget=BudgetModel(caps={("eng", "ge90"): 5, ("eng", "below"): 5}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        ruleset=_RULESET,
        copyright_config=_COPYRIGHT_CONFIG,
        seed=42,
        workers=2,
        sample_per_lang=10,
    )

    assert isinstance(summary, BuildSummary)
    assert summary.records_sampled == 1
    assert summary.records_matched == 2
    assert summary.pairs_written == 2
    assert summary.stratum_counts["eng/ge90"] == 1
    assert summary.stratum_counts["eng/below"] == 1
    floored = captured["matching_config"]
    assert isinstance(floored, MatchingConfig)
    assert floored.min_combined_score == 0.0
    assert _MATCHING_CONFIG.min_combined_score != 0.0


def test_build_queue_threads_log_file_to_run_match(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    _write_shard(pool / "eng" / "shard_1.xml", "id-1", "Title One")
    monkeypatch.setattr(bq, "load_or_build_idf", lambda *a, **k: object())
    monkeypatch.setattr(bq, "_load_calibrator", lambda *a, **k: None)

    seen_log_file: list[Path | None] = []

    def _fake_run_match(**kwargs: object) -> RunReport:
        log_file = kwargs.get("log_file")
        assert log_file is None or isinstance(log_file, Path)
        seen_log_file.append(log_file)
        return RunReport(
            records_processed=1,
            records_written=0,
            records_enqueued=1,
            duration_seconds=0.0,
            by_status={},
            interrupted=False,
        )

    monkeypatch.setattr(bq, "run_match", _fake_run_match)
    target = tmp_path / "queue.log"
    build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=tmp_path / "review.db",
        vault_path=tmp_path / "vault.jsonl",
        budget=BudgetModel(caps={("eng", "ge90"): 1}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        ruleset=_RULESET,
        copyright_config=_COPYRIGHT_CONFIG,
        seed=1,
        workers=1,
        sample_per_lang=10,
        log_file=target,
    )
    assert seen_log_file == [target]


def _vault_entry(
    control_id: str,
    nypl_uuid: str,
    *,
    verdict: str = "match",
    labeled_at: str = "2026-05-22T10:00:00+00:00",
    reasons: tuple[str, ...] = (),
    note: str | None = None,
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
    )


def _pair_for(control_id: str, uuid: str, *, score: float = 0.95, band: str = "ge90") -> PairInsert:
    return PairInsert(
        language="eng",
        decade=1950,
        score=score,
        band=band,
        source="banded",
        marc_control_id=control_id,
        marc_json="{}",
        marc_title="t",
        marc_author=None,
        marc_publisher=None,
        marc_year=1953,
        nypl_uuid=uuid,
        cce_title="CCE",
        cce_author=None,
        cce_publishers=None,
        cce_claimants=None,
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json="{}",
    )


def test_writer_injects_vault_pairs_outside_per_stratum_caps(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    budget = BudgetModel(caps={("eng", "ge90"): 0, ("eng", "below"): 0})

    resolved = (
        ResolvedVaultPair(
            entry=_vault_entry("ctrl-a", "uuid-a", verdict="match"),
            pair=_pair_for("ctrl-a", "uuid-a"),
        ),
        ResolvedVaultPair(
            entry=_vault_entry("ctrl-b", "uuid-b", verdict="no_match", reasons=("diff_work",)),
            pair=_pair_for("ctrl-b", "uuid-b"),
        ),
    )

    with StratifyingResultWriter(
        db_path=db_path, budget=budget, seed=1, vault_pairs=resolved
    ) as writer:
        writer.write(
            _marc(control_id="not-in-vault"), _match(0.95), _ASSESSMENT, _cce("uuid-other")
        )

    with ReviewDb.connect(db_path) as db:
        progress = db.progress()
        labels = list(db.iter_current_labels())
        keys = db.pair_keys()
    assert progress.total == 2
    assert progress.labeled == 2
    assert {label.marc_control_id for label in labels} == {"ctrl-a", "ctrl-b"}
    assert ("not-in-vault", "uuid-other") not in keys


def test_writer_vault_injection_preserves_verdict_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    entry = _vault_entry(
        "ctrl-a",
        "uuid-a",
        verdict="no_match",
        reasons=("diff_work", "diff_edition"),
        note="careful read",
        labeled_at="2026-05-22T11:00:00+00:00",
    )
    pair = _pair_for("ctrl-a", "uuid-a", score=0.42, band="below")
    resolved = (ResolvedVaultPair(entry=entry, pair=pair),)
    with StratifyingResultWriter(
        db_path=db_path,
        budget=BudgetModel(caps={}),
        seed=1,
        vault_pairs=resolved,
    ):
        pass

    with ReviewDb.connect(db_path) as db:
        labels = list(db.iter_current_labels())
    assert len(labels) == 1
    only = labels[0]
    assert only.verdict == "no_match"
    assert only.labeled_at == "2026-05-22T11:00:00+00:00"
    assert only.note == "careful read"
    assert set(only.reasons) == {"diff_work", "diff_edition"}


def test_build_queue_carries_vault_pair_through_rebuild(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from pd_groundtruth.label_vault import append_entry

    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    _write_shard(pool / "eng" / "shard_1.xml", "vault-marc", "Title V")
    _write_shard(pool / "eng" / "shard_2.xml", "other-marc", "Title O")

    vault_path = tmp_path / "vault.jsonl"
    vault_entry = _vault_entry("vault-marc", "vault-uuid", verdict="match")
    append_entry(vault_path, vault_entry)

    captured_resolve: dict[str, object] = {}
    captured_sample_marcs: list[list[str]] = []

    def _fake_resolve(**_kwargs: object) -> tuple[list[ResolvedVaultPair], ResolveSummary]:
        captured_resolve["called"] = True
        return (
            [ResolvedVaultPair(entry=vault_entry, pair=_pair_for("vault-marc", "vault-uuid"))],
            ResolveSummary(resolved=1, missing_in_pool=0, missing_in_index=0),
        )

    monkeypatch.setattr(bq, "load_or_build_idf", lambda *a, **k: object())
    monkeypatch.setattr(bq, "_load_calibrator", lambda *a, **k: None)
    monkeypatch.setattr(bq, "resolve_vault_for_build", _fake_resolve)

    def _fake_run_match(**kwargs: object) -> RunReport:
        factory = kwargs["writer_factory"]
        prepared_dir = kwargs["prepared_dir"]
        assert isinstance(prepared_dir, Path)
        from pickle import load as pickle_load

        with (prepared_dir / "chunk_00000.pkl").open("rb") as handle:
            chunk = pickle_load(handle)
        captured_sample_marcs.append([record.control_id for record in chunk])
        assert callable(factory)
        writer = factory(tmp_path / "ignored.csv")
        with writer:
            pass
        return RunReport(
            records_processed=len(chunk),
            records_written=len(chunk),
            records_enqueued=len(chunk),
            duration_seconds=0.0,
            by_status={},
            interrupted=False,
        )

    monkeypatch.setattr(bq, "run_match", _fake_run_match)

    out_path = tmp_path / "review.db"
    summary = build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=out_path,
        vault_path=vault_path,
        budget=BudgetModel(caps={("eng", "ge90"): 5, ("eng", "below"): 5}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        ruleset=_RULESET,
        copyright_config=_COPYRIGHT_CONFIG,
        seed=42,
        workers=1,
        sample_per_lang=10,
    )

    assert captured_resolve == {"called": True}
    assert summary.vault_resolved == 1
    assert summary.vault_missing_in_pool == 0
    assert summary.vault_missing_in_index == 0
    assert captured_sample_marcs == [["other-marc"]]
    assert summary.records_sampled == 1
    assert summary.pairs_written == 1
    assert summary.stratum_counts["eng/ge90"] == 1
    with ReviewDb.connect(out_path) as db:
        labels = list(db.iter_current_labels())
    assert len(labels) == 1
    assert labels[0].marc_control_id == "vault-marc"
    assert labels[0].verdict == "match"


def test_build_queue_excludes_vault_marcs_from_sample(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    _write_shard(pool / "eng" / "shard_1.xml", "vault-a", "T1")
    _write_shard(pool / "eng" / "shard_2.xml", "vault-b", "T2")
    _write_shard(pool / "eng" / "shard_3.xml", "non-vault", "T3")

    entry_a = _vault_entry("vault-a", "uuid-a")
    entry_b = _vault_entry("vault-b", "uuid-b")

    monkeypatch.setattr(bq, "load_or_build_idf", lambda *a, **k: object())
    monkeypatch.setattr(bq, "_load_calibrator", lambda *a, **k: None)
    monkeypatch.setattr(
        bq,
        "resolve_vault_for_build",
        lambda **_kwargs: (
            [
                ResolvedVaultPair(entry=entry_a, pair=_pair_for("vault-a", "uuid-a")),
                ResolvedVaultPair(entry=entry_b, pair=_pair_for("vault-b", "uuid-b")),
            ],
            ResolveSummary(resolved=2, missing_in_pool=0, missing_in_index=0),
        ),
    )

    seen_marcs: list[list[str]] = []

    def _fake_run_match(**kwargs: object) -> RunReport:
        prepared_dir = kwargs["prepared_dir"]
        assert isinstance(prepared_dir, Path)
        from pickle import load as pickle_load

        with (prepared_dir / "chunk_00000.pkl").open("rb") as handle:
            chunk = pickle_load(handle)
        seen_marcs.append([record.control_id for record in chunk])
        factory = kwargs["writer_factory"]
        assert callable(factory)
        with factory(tmp_path / "ignored.csv"):
            pass
        return RunReport(
            records_processed=len(chunk),
            records_written=0,
            records_enqueued=len(chunk),
            duration_seconds=0.0,
            by_status={},
            interrupted=False,
        )

    monkeypatch.setattr(bq, "run_match", _fake_run_match)

    build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=tmp_path / "review.db",
        vault_path=tmp_path / "vault.jsonl",
        budget=BudgetModel(caps={("eng", "ge90"): 5}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        ruleset=_RULESET,
        copyright_config=_COPYRIGHT_CONFIG,
        seed=1,
        workers=1,
        sample_per_lang=10,
    )
    assert seen_marcs == [["non-vault"]]


class _NullCceLookup:
    def __enter__(self) -> _NullCceLookup:
        return self

    def __exit__(self, *_args: object) -> None: ...

    def get_registration(self, _uuid: str) -> object:
        return None

    def coverage(self) -> Coverage:
        return LEGACY_COVERAGE


def test_build_queue_reports_vault_entries_missing_from_pool(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from pd_groundtruth.label_vault import append_entry

    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    _write_shard(pool / "eng" / "shard_1.xml", "in-pool", "Present")
    vault_path = tmp_path / "vault.jsonl"
    append_entry(vault_path, _vault_entry("missing-marc", "uuid-x"))

    from pd_groundtruth import build_queue_vault as bqv

    monkeypatch.setattr(bq, "load_or_build_idf", lambda *a, **k: object())
    monkeypatch.setattr(bq, "_load_calibrator", lambda *a, **k: None)
    monkeypatch.setattr(bqv, "NyplIndexLookup", lambda _p: _NullCceLookup())

    def _fake_run_match(**kwargs: object) -> RunReport:
        factory = kwargs["writer_factory"]
        assert callable(factory)
        with factory(tmp_path / "ignored.csv"):
            pass
        return RunReport(
            records_processed=1,
            records_written=0,
            records_enqueued=1,
            duration_seconds=0.0,
            by_status={},
            interrupted=False,
        )

    monkeypatch.setattr(bq, "run_match", _fake_run_match)

    summary = build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=tmp_path / "review.db",
        vault_path=vault_path,
        budget=BudgetModel(caps={("eng", "ge90"): 1}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        ruleset=_RULESET,
        copyright_config=_COPYRIGHT_CONFIG,
        seed=1,
        workers=1,
        sample_per_lang=10,
    )
    assert summary.vault_resolved == 0
    assert summary.vault_missing_in_pool == 1
    assert summary.vault_missing_in_index == 0
    with ReviewDb.connect(tmp_path / "review.db") as db:
        assert db.stratum_counts() == {}


def test_build_queue_reports_vault_entry_missing_from_index(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from pd_groundtruth.label_vault import append_entry

    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    _write_shard(pool / "eng" / "shard_1.xml", "have-marc", "Present")
    vault_path = tmp_path / "vault.jsonl"
    append_entry(vault_path, _vault_entry("have-marc", "uuid-gone"))

    from pd_groundtruth import build_queue_vault as bqv

    monkeypatch.setattr(bq, "load_or_build_idf", lambda *a, **k: object())
    monkeypatch.setattr(bq, "_load_calibrator", lambda *a, **k: None)
    monkeypatch.setattr(bqv, "compile_pairings", lambda _pairing: object())
    monkeypatch.setattr(bqv, "make_pair_scorer", lambda **_kw: lambda _m, _c: _match(0.95))
    monkeypatch.setattr(bqv, "NyplIndexLookup", lambda _p: _NullCceLookup())

    def _fake_run_match(**kwargs: object) -> RunReport:
        factory = kwargs["writer_factory"]
        assert callable(factory)
        with factory(tmp_path / "ignored.csv"):
            pass
        return RunReport(
            records_processed=0,
            records_written=0,
            records_enqueued=0,
            duration_seconds=0.0,
            by_status={},
            interrupted=False,
        )

    monkeypatch.setattr(bq, "run_match", _fake_run_match)

    summary = build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=tmp_path / "review.db",
        vault_path=vault_path,
        budget=BudgetModel(caps={("eng", "ge90"): 1}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        ruleset=_RULESET,
        copyright_config=_COPYRIGHT_CONFIG,
        seed=1,
        workers=1,
        sample_per_lang=10,
    )
    assert summary.vault_resolved == 0
    assert summary.vault_missing_in_index == 1
    with ReviewDb.connect(tmp_path / "review.db") as db:
        assert db.stratum_counts() == {}


def test_load_calibrator_returns_none_when_file_absent(tmp_path: Path) -> None:
    from pd_groundtruth.build_queue import _load_calibrator

    assert _load_calibrator(tmp_path) is None


def test_load_calibrator_reads_file_when_present(tmp_path: Path) -> None:
    from pd_matcher.match.combiners.calibrator import PlattCalibrator
    from pd_matcher.match.combiners.calibrator import save_calibrator

    from pd_groundtruth.build_queue import _load_calibrator

    saved = PlattCalibrator(
        a=-0.5, b=0.25, trained_at="2026-05-22T00:00:00+00:00", n_positive=10, n_negative=20
    )
    save_calibrator(saved, tmp_path / "calibrator.msgpack")
    loaded = _load_calibrator(tmp_path)
    assert loaded == saved


def test_build_queue_cleans_up_prepared_dir(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    _write_shard(pool / "eng" / "shard_1.xml", "id-1", "Title One")
    monkeypatch.setattr(bq, "load_or_build_idf", lambda *a, **k: object())
    monkeypatch.setattr(bq, "_load_calibrator", lambda *a, **k: None)

    seen: list[Path] = []

    def _fake_run_match(**kwargs: object) -> RunReport:
        prepared_dir = kwargs["prepared_dir"]
        assert isinstance(prepared_dir, Path)
        seen.append(prepared_dir)
        return RunReport(
            records_processed=1,
            records_written=0,
            records_enqueued=1,
            duration_seconds=0.0,
            by_status={},
            interrupted=False,
        )

    monkeypatch.setattr(bq, "run_match", _fake_run_match)
    build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=tmp_path / "review.db",
        vault_path=tmp_path / "vault.jsonl",
        budget=BudgetModel(caps={("eng", "ge90"): 1}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        ruleset=_RULESET,
        copyright_config=_COPYRIGHT_CONFIG,
        seed=1,
        workers=1,
        sample_per_lang=10,
    )
    assert seen
    assert not seen[0].exists()
