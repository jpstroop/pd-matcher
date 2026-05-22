"""Unit tests for the review-queue builder orchestration.

No network, no LMDB, no real matching: the matcher is replaced by a
monkeypatched ``run_match`` and the :class:`StratifyingResultWriter` is
exercised directly with fabricated :class:`MarcRecord`,
:class:`MatchResult`, and :class:`IndexedNyplRegRecord` objects against a
temporary SQLite review database.
"""

from pathlib import Path
from pickle import loads as pickle_loads

from msgspec.json import decode as json_decode
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.copyright.assessment import CopyrightAssessment
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
from pd_groundtruth.build_queue import _iter_language_dirs
from pd_groundtruth.build_queue import _join
from pd_groundtruth.build_queue import _sample_language
from pd_groundtruth.build_queue import _write_sample_chunks
from pd_groundtruth.build_queue import build_queue
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import BudgetModel

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
        publisher_names=("Pub A", "Pub B"),
        claimants=("Claimant A",),
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
