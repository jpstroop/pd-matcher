"""Unit tests for the active-learning orchestration (issue #81).

Exercises the select → dual-score → bucket → write loop and the resource-
resolution entry point with all heavy IO (LMDB, IDF caches, learned artifact)
monkey-patched, so no real data is read and the vault is never touched.
"""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from msgspec.structs import replace
from pytest import MonkeyPatch
from pytest import raises

from pd_groundtruth import active_learning as module
from pd_groundtruth.active_learning import ActiveLearningSummary
from pd_groundtruth.active_learning import build_active_learning_summary
from pd_groundtruth.active_learning import run_active_learning
from pd_groundtruth.active_score import BUCKET_AGREE_HIGH
from pd_groundtruth.active_score import BUCKET_AGREE_LOW
from pd_groundtruth.active_score import BUCKET_INFORMATIVE
from pd_groundtruth.active_score import CandidateScorer
from pd_groundtruth.active_select import RecordSource
from pd_groundtruth.review_db import ReviewDb
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


class _LearnedFromMap:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:
        score = self._scores[evidence[0].scorer]
        return CombinedScore(raw=score * 100.0, calibrated=score)


def _marc(control_id: str, *, language: str = "eng") -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A Title",
        title_main="A Title",
        publication_year=1953,
        language_code=language,
    )


def _cce(uuid: str) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid, title="CCE Title", was_renewed=True, regnum="R1", reg_year=1953
    )


def _candidate(uuid: str, weighted: float) -> CandidateMatch:
    evidence = Evidence(
        scorer=uuid, score=weighted, max=1.0, skipped=False, decisive=False, features=()
    )
    return CandidateMatch(
        nypl_uuid=uuid,
        nypl_year=1953,
        combined=CombinedScore(raw=weighted * 100.0, calibrated=weighted),
        evidence=(evidence,),
        losing_evidence=(),
        evidence_sources=(("245", "title"),),
    )


def _source(records: dict[str, list[MarcRecord]]) -> RecordSource:
    def source(language: str) -> Iterator[MarcRecord]:
        yield from records.get(language, [])

    return source


def _candidate_scorer(
    table: dict[str, list[tuple[IndexedNyplRegRecord, CandidateMatch]]],
) -> CandidateScorer:
    def scorer(marc: MarcRecord) -> Iterator[tuple[IndexedNyplRegRecord, CandidateMatch]]:
        yield from table.get(marc.control_id, [])

    return scorer


def _summary(
    *,
    source: RecordSource,
    candidate_scorer: CandidateScorer,
    learned: Combiner,
    out_path: Path,
    dry_run: bool = False,
    target: int = 10,
) -> ActiveLearningSummary:
    return build_active_learning_summary(
        source=source,
        candidate_scorer=candidate_scorer,
        learned=learned,
        excluded_marc_ids=frozenset(),
        weights={"eng": 1.0},
        target=target,
        seed=1,
        out_path=out_path,
        dry_run=dry_run,
    )


def test_build_summary_writes_only_informative_pairs(tmp_path: Path) -> None:
    out_path = tmp_path / "active.db"
    source = _source({"eng": [_marc("split"), _marc("agree")]})
    table = {
        "split": [(_cce("a"), _candidate("a", 0.9)), (_cce("b"), _candidate("b", 0.4))],
        "agree": [(_cce("c"), _candidate("c", 0.9))],
    }
    learned = _LearnedFromMap({"a": 0.2, "b": 0.95, "c": 0.9})
    summary = _summary(
        source=source, candidate_scorer=_candidate_scorer(table), learned=learned, out_path=out_path
    )
    assert summary.scored == 2
    assert summary.informative() == 1
    assert summary.written == 1
    with ReviewDb.connect(out_path) as db:
        assert db.stratum_counts() == {("eng", BUCKET_INFORMATIVE): 1}
        pair = db.next_unlabeled()
        assert pair is not None
        assert pair.marc_control_id == "split"
        assert pair.nypl_uuid == "a"
        assert pair.audit_note is not None
        assert "disagreement=" in pair.audit_note


def test_build_summary_dry_run_writes_nothing(tmp_path: Path) -> None:
    out_path = tmp_path / "active.db"
    source = _source({"eng": [_marc("split")]})
    table = {"split": [(_cce("a"), _candidate("a", 0.9)), (_cce("b"), _candidate("b", 0.4))]}
    learned = _LearnedFromMap({"a": 0.2, "b": 0.95})
    summary = _summary(
        source=source,
        candidate_scorer=_candidate_scorer(table),
        learned=learned,
        out_path=out_path,
        dry_run=True,
    )
    assert summary.dry_run is True
    assert summary.written == 0
    assert summary.informative() == 1
    assert not out_path.exists()


def test_build_summary_ranks_informative_by_disagreement(tmp_path: Path) -> None:
    out_path = tmp_path / "active.db"
    source = _source({"eng": [_marc("small_gap"), _marc("split")]})
    table = {
        "small_gap": [(_cce("a"), _candidate("a", 0.6))],
        "split": [(_cce("b"), _candidate("b", 0.9)), (_cce("c"), _candidate("c", 0.5))],
    }
    learned = _LearnedFromMap({"a": 0.45, "b": 0.1, "c": 0.95})
    _summary(
        source=source, candidate_scorer=_candidate_scorer(table), learned=learned, out_path=out_path
    )
    with ReviewDb.connect(out_path) as db:
        first = db.next_unlabeled()
    assert first is not None
    assert first.marc_control_id == "split"


def test_bucket_stats_cover_all_three_buckets(tmp_path: Path) -> None:
    out_path = tmp_path / "active.db"
    source = _source({"eng": [_marc("high"), _marc("low"), _marc("split")]})
    table = {
        "high": [(_cce("a"), _candidate("a", 0.95))],
        "low": [(_cce("b"), _candidate("b", 0.1))],
        "split": [(_cce("c"), _candidate("c", 0.9)), (_cce("d"), _candidate("d", 0.4))],
    }
    learned = _LearnedFromMap({"a": 0.95, "b": 0.05, "c": 0.1, "d": 0.95})
    summary = _summary(
        source=source, candidate_scorer=_candidate_scorer(table), learned=learned, out_path=out_path
    )
    counts = {stats.bucket: stats.count for stats in summary.buckets}
    assert counts == {BUCKET_INFORMATIVE: 1, BUCKET_AGREE_HIGH: 1, BUCKET_AGREE_LOW: 1}
    informative = next(s for s in summary.buckets if s.bucket == BUCKET_INFORMATIVE)
    assert informative.max_disagreement > 1.0
    agree_high = next(s for s in summary.buckets if s.bucket == BUCKET_AGREE_HIGH)
    assert agree_high.mean_disagreement == 0.0


def test_informative_count_zero_when_no_informative_bucket(tmp_path: Path) -> None:
    out_path = tmp_path / "active.db"
    source = _source({"eng": [_marc("high")]})
    table = {"high": [(_cce("a"), _candidate("a", 0.95))]}
    learned = _LearnedFromMap({"a": 0.9})
    summary = _summary(
        source=source, candidate_scorer=_candidate_scorer(table), learned=learned, out_path=out_path
    )
    assert summary.informative() == 0


def test_pool_record_source_streams_language_shards(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    ns = "http://www.loc.gov/MARC21/slim"
    record = (
        "<record><leader>00000nam a2200000 a 4500</leader>"
        '<controlfield tag="001">id-1</controlfield>'
        '<controlfield tag="008">750101s1953    xxu           000 0 eng d</controlfield>'
        '<datafield tag="245" ind1="0" ind2="0"><subfield code="a">A Title</subfield></datafield>'
        "</record>"
    )
    (pool / "eng" / "shard.xml").write_text(
        f'<collection xmlns="{ns}">{record}</collection>', encoding="utf-8"
    )
    source = module._pool_record_source(pool)
    records = list(source("eng"))
    assert [r.control_id for r in records] == ["id-1"]


def test_pool_record_source_missing_language_yields_nothing(tmp_path: Path) -> None:
    source = module._pool_record_source(tmp_path / "pool")
    assert list(source("fre")) == []


def test_weighted_config_forces_weighted_mean() -> None:
    learned_cfg = replace(_load_default_matching_config(), scorer="learned")
    assert module._weighted_config(learned_cfg).scorer == "weighted_mean"


def test_weighted_config_passthrough_when_already_weighted() -> None:
    config = _load_default_matching_config()
    assert module._weighted_config(config) is config


def test_learned_config_forces_learned() -> None:
    assert module._learned_config(_load_default_matching_config()).scorer == "learned"


def test_record_without_candidates_buckets_agree_low_not_written(tmp_path: Path) -> None:
    out_path = tmp_path / "active.db"
    source = _source({"eng": [_marc("no_cands")]})
    table: dict[str, list[tuple[IndexedNyplRegRecord, CandidateMatch]]] = {"no_cands": []}
    learned = _LearnedFromMap({})
    summary = _summary(
        source=source, candidate_scorer=_candidate_scorer(table), learned=learned, out_path=out_path
    )
    assert summary.informative() == 0
    assert summary.written == 0
    counts = {stats.bucket: stats.count for stats in summary.buckets}
    assert counts[BUCKET_AGREE_LOW] == 1


class _FakeLookup:
    def __init__(self, candidates: dict[str, list[IndexedNyplRegRecord]]) -> None:
        self._candidates = candidates

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None: ...

    def candidates_for(self, marc: MarcRecord, _window: int) -> Iterator[IndexedNyplRegRecord]:
        yield from self._candidates.get(marc.control_id, [])


def _patch_resources(
    monkeypatch: MonkeyPatch,
    *,
    learned: object,
    candidates: dict[str, list[IndexedNyplRegRecord]],
    weighted_scores: dict[str, float],
) -> None:
    monkeypatch.setattr(module, "load_or_build_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_author_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_publisher_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "compile_pairings", lambda _config: object())
    monkeypatch.setattr(module, "NyplIndexLookup", lambda _path: _FakeLookup(candidates))
    monkeypatch.setattr(module, "build_combiner", lambda _config, **_kwargs: learned)

    def fake_make_pair_scorer(
        **_kwargs: object,
    ) -> Callable[[MarcRecord, IndexedNyplRegRecord], CandidateMatch]:
        def score_pair(_marc: MarcRecord, cce: IndexedNyplRegRecord) -> CandidateMatch:
            return _candidate(cce.uuid, weighted_scores[cce.uuid])

        return score_pair

    monkeypatch.setattr(module, "make_pair_scorer", fake_make_pair_scorer)


def _write_vault(path: Path, marc_id: str, uuid: str) -> None:
    from pd_groundtruth.label_vault import SCHEMA_VERSION
    from pd_groundtruth.label_vault import MarcIdentifiers
    from pd_groundtruth.label_vault import VaultEntry
    from pd_groundtruth.label_vault import upsert_entry

    upsert_entry(
        path,
        VaultEntry(
            schema=SCHEMA_VERSION,
            marc_control_id=marc_id,
            nypl_uuid=uuid,
            verdict="match",
            note=None,
            labeled_at="2026-05-22T10:00:00+00:00",
            labeler="jpstroop",
            marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
        ),
    )


def test_run_active_learning_end_to_end(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    (pool / "eng").mkdir(parents=True)
    ns = "http://www.loc.gov/MARC21/slim"
    record = (
        "<record><leader>00000nam a2200000 a 4500</leader>"
        '<controlfield tag="001">split</controlfield>'
        '<controlfield tag="008">750101s1953    xxu           000 0 eng d</controlfield>'
        '<datafield tag="245" ind1="0" ind2="0"><subfield code="a">A Title</subfield></datafield>'
        "</record>"
    )
    (pool / "eng" / "shard.xml").write_text(
        f'<collection xmlns="{ns}">{record}</collection>', encoding="utf-8"
    )
    index_path = tmp_path / "idx" / "cce.lmdb"
    index_path.parent.mkdir(parents=True)
    out_path = tmp_path / "active.db"
    vault_path = tmp_path / "vault.jsonl"
    _write_vault(vault_path, "already-seen", "uuid-x")

    _patch_resources(
        monkeypatch,
        learned=_LearnedFromMap({"a": 0.1, "b": 0.95}),
        candidates={"split": [_cce("a"), _cce("b")]},
        weighted_scores={"a": 0.9, "b": 0.4},
    )
    summary = run_active_learning(
        pool=pool,
        index_path=index_path,
        out_path=out_path,
        vault_path=vault_path,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        weights={"eng": 1.0},
        target=5,
        seed=1,
        dry_run=False,
    )
    assert summary.selected == 1
    assert summary.informative() == 1
    assert summary.written == 1
    assert out_path.exists()


def test_run_active_learning_aborts_when_learned_missing(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    index_path = tmp_path / "idx" / "cce.lmdb"
    index_path.parent.mkdir(parents=True)

    def fail(_config: MatchingConfig, **_kwargs: object) -> object:
        raise ValueError("scorer is 'learned' but no learned-model artifact was found")

    monkeypatch.setattr(module, "load_or_build_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_author_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_publisher_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "compile_pairings", lambda _config: object())
    monkeypatch.setattr(module, "make_pair_scorer", lambda **_kwargs: lambda _m, _c: None)
    monkeypatch.setattr(module, "build_combiner", fail)

    with raises(ValueError, match="learned-model artifact"):
        run_active_learning(
            pool=tmp_path / "pool",
            index_path=index_path,
            out_path=tmp_path / "active.db",
            vault_path=tmp_path / "vault.jsonl",
            matching_config=_load_default_matching_config(),
            pairing_config=_load_default_pairing_config(),
        )
