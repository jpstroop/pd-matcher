"""Unit tests for the review-queue builder orchestration.

No network, no LMDB, no real matching: the matcher and CCE lookup are
replaced with fabricated objects and monkeypatched seams.
"""

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from msgspec.json import encode as json_encode
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pytest import MonkeyPatch
from pytest import raises

from pd_groundtruth import build_queue as bq
from pd_groundtruth.build_queue import BuildSummary
from pd_groundtruth.build_queue import MatcherState
from pd_groundtruth.build_queue import WorkerOutcome
from pd_groundtruth.build_queue import _build_tasks
from pd_groundtruth.build_queue import _decade_of
from pd_groundtruth.build_queue import _evidence_payload
from pd_groundtruth.build_queue import _iter_language_dirs
from pd_groundtruth.build_queue import _join
from pd_groundtruth.build_queue import _match_one
from pd_groundtruth.build_queue import _pair_insert
from pd_groundtruth.build_queue import _pool_match
from pd_groundtruth.build_queue import _run_pool
from pd_groundtruth.build_queue import _sample_language
from pd_groundtruth.build_queue import _Task
from pd_groundtruth.build_queue import build_queue
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import AcceptedPair
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


def _write_shard(path: Path, control_id: str, title: str) -> None:
    path.write_text(
        _MARCXML_TEMPLATE.format(ns=_MARC_NS, control_id=control_id, title=title),
        encoding="utf-8",
    )


_MATCHING_CONFIG: MatchingConfig = _load_default_matching_config()
_PAIRING_CONFIG = _load_default_pairing_config()


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


def test_decade_of_buckets_years() -> None:
    assert _decade_of(1953) == 1950
    assert _decade_of(1960) == 1960
    assert _decade_of(None) is None


def test_join_collapses_empty_to_none() -> None:
    assert _join(()) is None
    assert _join(("a",)) == "a"
    assert _join(("a", "b")) == "a | b"


def test_evidence_payload_maps_scorer_to_score() -> None:
    payload = _evidence_payload((("title", 0.9), ("author", 0.5)))
    assert payload == {"title": 0.9, "author": 0.5}


class _FakeState:
    """Typed ``MatcherState`` stand-in (no LMDB handle).

    ``match_record`` is monkeypatched in tests, so the attribute values are
    never dereferenced — only their presence and types matter for the
    Protocol.
    """

    def __init__(self, lookup: NyplIndexLookup, idf: IdfTable, combiner: Combiner) -> None:
        self.lookup = lookup
        self.idf = idf
        self.combiner = combiner
        self.matching_config = _MATCHING_CONFIG
        self.pairings: CompiledPairings = compile_pairings(_PAIRING_CONFIG)


def _fake_state() -> MatcherState:
    idf = IdfTable(document_count=1, default_idf=1.0, source_hash="h", language="eng", idf={})
    lookup = NyplIndexLookup.__new__(NyplIndexLookup)
    return _FakeState(lookup, idf, WeightedMeanCombiner(config=_MATCHING_CONFIG))


def test_match_one_returns_none_when_no_best(monkeypatch: MonkeyPatch) -> None:
    result = MatchResult(
        marc_control_id="ctrl-1", best=None, alternates=(), candidates_considered=0
    )
    monkeypatch.setattr(bq, "match_record", lambda *a, **k: result)
    assert _match_one("eng", _marc(), _fake_state()) is None


def test_match_one_builds_outcome_dropping_skipped_evidence(monkeypatch: MonkeyPatch) -> None:
    best = CandidateMatch(
        nypl_uuid="uuid-1",
        nypl_year=1953,
        combined=CombinedScore(raw=92.0, calibrated=0.92),
        evidence=(
            _evidence("title.token_set", 0.9),
            _evidence("lccn.exact", 0.0, skipped=True),
        ),
        losing_evidence=(),
    )
    result = MatchResult(
        marc_control_id="ctrl-1", best=best, alternates=(), candidates_considered=5
    )
    monkeypatch.setattr(bq, "match_record", lambda *a, **k: result)
    outcome = _match_one("eng", _marc(), _fake_state())
    assert outcome is not None
    assert outcome.language == "eng"
    assert outcome.marc_control_id == "ctrl-1"
    assert outcome.score == 0.92
    assert outcome.nypl_uuid == "uuid-1"
    payload = bq._MARC_DECODER(outcome.evidence_json, type=dict[str, float])
    assert payload == {"title.token_set": 0.9}


def _outcome(control_id: str, score: float, *, uuid: str = "uuid-1") -> WorkerOutcome:
    return WorkerOutcome(
        language="eng",
        marc_control_id=control_id,
        marc_json=json_encode(_marc(control_id=control_id)),
        score=score,
        nypl_uuid=uuid,
        evidence_json=json_encode({"title.token_set": score}),
    )


def test_pair_insert_snapshots_cce_fields() -> None:
    accepted = AcceptedPair(key="ctrl-1", language="eng", band="ge90", source="banded", score=0.95)
    insert = _pair_insert(accepted, _outcome("ctrl-1", 0.95), _cce())
    assert insert.language == "eng"
    assert insert.band == "ge90"
    assert insert.decade == 1950
    assert insert.marc_title == "A Title"
    assert insert.marc_author == "An Author"
    assert insert.cce_title == "CCE Title"
    assert insert.cce_publishers == "Pub A | Pub B"
    assert insert.cce_claimants == "Claimant A"
    assert insert.cce_was_renewed is True
    assert insert.cce_regnum == "R123"


def test_pair_insert_handles_missing_cce() -> None:
    accepted = AcceptedPair(
        key="ctrl-1", language="eng", band="below", source="below_sample", score=0.3
    )
    insert = _pair_insert(accepted, _outcome("ctrl-1", 0.3), None)
    assert insert.cce_title is None
    assert insert.cce_publishers is None
    assert insert.cce_was_renewed is None
    assert insert.nypl_uuid == "uuid-1"


def test_pair_insert_author_falls_back_to_statement_of_responsibility() -> None:
    marc = MarcRecord(
        control_id="ctrl-2",
        title="T",
        title_main="T",
        statement_of_responsibility="by Someone",
        publication_year=1940,
        language_code="eng",
    )
    outcome = WorkerOutcome(
        language="eng",
        marc_control_id="ctrl-2",
        marc_json=json_encode(marc),
        score=0.8,
        nypl_uuid="uuid-2",
        evidence_json=json_encode({}),
    )
    accepted = AcceptedPair(key="ctrl-2", language="eng", band="b80_90", source="banded", score=0.8)
    insert = _pair_insert(accepted, outcome, None)
    assert insert.marc_author == "by Someone"


class _FakeLookup:
    """Context-managed stand-in for ``NyplIndexLookup`` (no LMDB)."""

    def __init__(self, _path: Path) -> None:
        self._records = {"uuid-1": _cce("uuid-1")}

    def __enter__(self) -> _FakeLookup:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def get_registration(self, uuid: str) -> IndexedNyplRegRecord | None:
        return self._records.get(uuid)


def test_build_queue_rejects_zero_workers(tmp_path: Path) -> None:
    with raises(ValueError, match="workers must be >= 1"):
        build_queue(
            pool=tmp_path,
            index_path=tmp_path / "idx",
            out_path=tmp_path / "out.db",
            budget=BudgetModel(caps={}),
            matching_config=_MATCHING_CONFIG,
            pairing_config=_PAIRING_CONFIG,
            seed=1,
            workers=0,
            sample_per_lang=10,
        )


def test_build_queue_end_to_end_with_fakes(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)

    monkeypatch.setattr(bq, "_refresh_idf_cache", lambda *a, **k: None)

    sampled = [_marc(control_id="ctrl-1"), _marc(control_id="ctrl-2", year=1940)]
    monkeypatch.setattr(bq, "_sample_language", lambda *a, **k: sampled)

    def _fake_pool(tasks: list[bq._Task], **_kwargs: object) -> Iterator[WorkerOutcome]:
        yield _outcome("ctrl-1", 0.95)
        yield _outcome("ctrl-2", 0.3, uuid="uuid-1")

    monkeypatch.setattr(bq, "_run_pool", _fake_pool)
    monkeypatch.setattr(bq, "NyplIndexLookup", _FakeLookup)

    budget = BudgetModel(caps={("eng", "ge90"): 5, ("eng", "below"): 5})
    out_path = tmp_path / "review.db"
    summary = build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=out_path,
        budget=budget,
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        seed=42,
        workers=2,
        sample_per_lang=10,
    )

    assert isinstance(summary, BuildSummary)
    assert summary.records_sampled == 2
    assert summary.records_matched == 2
    assert summary.pairs_written == 2
    assert summary.stratum_counts["eng/ge90"] == 1
    assert summary.stratum_counts["eng/below"] == 1

    with ReviewDb.connect(out_path) as db:
        counts = db.stratum_counts()
    assert counts[("eng", "ge90")] == 1
    assert counts[("eng", "below")] == 1


def test_build_queue_floors_min_combined_score(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    captured: list[MatchingConfig] = []

    monkeypatch.setattr(bq, "_refresh_idf_cache", lambda *a, **k: None)
    monkeypatch.setattr(bq, "_sample_language", lambda *a, **k: [_marc()])

    def _fake_pool(
        tasks: list[bq._Task], *, matching_config: MatchingConfig, **_kwargs: object
    ) -> Iterator[WorkerOutcome]:
        captured.append(matching_config)
        yield _outcome("ctrl-1", 0.95)

    monkeypatch.setattr(bq, "_run_pool", _fake_pool)
    monkeypatch.setattr(bq, "NyplIndexLookup", _FakeLookup)

    build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=tmp_path / "review.db",
        budget=BudgetModel(caps={("eng", "ge90"): 5}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        seed=1,
        workers=1,
        sample_per_lang=10,
    )
    assert captured[0].min_combined_score == 0.0
    assert _MATCHING_CONFIG.min_combined_score != 0.0


def test_build_queue_empty_pool_writes_nothing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    monkeypatch.setattr(bq, "_refresh_idf_cache", lambda *a, **k: None)
    monkeypatch.setattr(bq, "NyplIndexLookup", _FakeLookup)

    summary = build_queue(
        pool=pool,
        index_path=tmp_path / "idx" / "nypl.lmdb",
        out_path=tmp_path / "review.db",
        budget=BudgetModel(caps={}),
        matching_config=_MATCHING_CONFIG,
        pairing_config=_PAIRING_CONFIG,
        seed=1,
        workers=1,
        sample_per_lang=10,
    )
    assert summary.records_sampled == 0
    assert summary.pairs_written == 0


def test_iter_language_dirs_yields_only_subdirs(tmp_path: Path) -> None:
    (tmp_path / "eng").mkdir()
    (tmp_path / "fre").mkdir()
    (tmp_path / "stray.txt").write_text("ignore", encoding="utf-8")
    pairs = list(_iter_language_dirs(tmp_path))
    assert [name for name, _ in pairs] == ["eng", "fre"]
    assert all(path.is_dir() for _, path in pairs)


def test_build_tasks_encodes_round_trippable_marc() -> None:
    tasks = _build_tasks("eng", [_marc(control_id="x")])
    assert len(tasks) == 1
    decoded = bq._MARC_DECODER(tasks[0].marc_json, type=MarcRecord)
    assert decoded.control_id == "x"
    assert tasks[0].language == "eng"


def test_sample_language_parses_shards_without_lmdb(tmp_path: Path) -> None:
    lang_dir = tmp_path / "eng"
    lang_dir.mkdir()
    _write_shard(lang_dir / "shard_1.xml", "id-1", "First Title")
    _write_shard(lang_dir / "shard_2.xml", "id-2", "Second Title")
    records = _sample_language(lang_dir, sample_per_lang=10, seed=1)
    assert {record.control_id for record in records} == {"id-1", "id-2"}


def test_pool_match_raises_when_state_uninitialized(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(bq, "_WORKER_STATE", None)
    task = _Task(language="eng", marc_json=json_encode(_marc()))
    with raises(RuntimeError, match="before _pool_initializer"):
        _pool_match(task)


def test_run_pool_empty_tasks_yields_nothing() -> None:
    outcomes = list(
        _run_pool(
            [],
            index_path=Path("unused"),
            idf_cache_path=Path("unused"),
            matching_config=_MATCHING_CONFIG,
            pairing_config=_PAIRING_CONFIG,
            workers=2,
        )
    )
    assert outcomes == []
