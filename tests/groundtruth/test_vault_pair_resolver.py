"""Unit tests for the shared vault-pair resolution helpers.

The helpers exercised here are the ones both ``build-queue`` (carryover) and
``vault-into-queue`` (recovery) build on. ``test_vault_into_queue.py`` still
covers the full backfill end-to-end; this module pins down the helpers in
isolation so they stay stable under future caller changes.
"""

from pathlib import Path

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.vault_pair_resolver import IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import ResolvedVaultPair
from pd_groundtruth.vault_pair_resolver import ResolveSummary
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import iter_pool_shards
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_groundtruth.vault_pair_resolver import resolve_vault_pairs
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

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


def _cce(uuid: str = "uuid-1") -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid,
        title="CCE Title",
        was_renewed=True,
        regnum="R123",
        reg_year=1953,
        author_name="CCE Author",
        publisher_names=("Pub A",),
        claimants=("Claimant A",),
    )


def _candidate(score: float, uuid: str = "uuid-1") -> CandidateMatch:
    evidence = Evidence(scorer="t", score=0.9, max=1.0, skipped=False, decisive=False, features=())
    return CandidateMatch(
        nypl_uuid=uuid,
        nypl_year=1953,
        combined=CombinedScore(raw=score * 100.0, calibrated=score),
        evidence=(evidence,),
        losing_evidence=(),
    )


def _vault_entry(control_id: str, nypl_uuid: str) -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=control_id,
        nypl_uuid=nypl_uuid,
        verdict="match",
        note=None,
        labeled_at="2026-05-22T10:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )


def _stub_pair(control_id: str, uuid: str) -> PairInsert:
    return PairInsert(
        language="eng",
        decade=1950,
        score=0.95,
        band="ge90",
        source="banded",
        marc_control_id=control_id,
        marc_json="{}",
        marc_title="t",
        marc_author=None,
        marc_publisher=None,
        marc_year=1953,
        nypl_uuid=uuid,
        cce_title="CCE Title",
        cce_author=None,
        cce_publishers=None,
        cce_claimants=None,
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json="{}",
    )


def test_idf_cache_name_matches_build_queue_constant() -> None:
    assert IDF_CACHE_NAME == "idf.msgpack"


def test_iter_pool_shards_yields_lang_shards_in_order(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path / "pool", {"eng": ["a", "b"], "fre": ["c"]})
    shards = list(iter_pool_shards(pool))
    assert [shard.parent.name for shard in shards] == ["eng", "eng", "fre"]
    assert [shard.name for shard in shards] == ["shard_1.xml", "shard_2.xml", "shard_1.xml"]


def test_iter_pool_shards_skips_non_directory_entries(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "eng").mkdir()
    _write_shard(pool / "eng" / "shard.xml", "a")
    (pool / "stray.txt").write_text("ignore", encoding="utf-8")
    shards = list(iter_pool_shards(pool))
    assert len(shards) == 1
    assert shards[0].parent.name == "eng"


def test_build_marc_index_resolves_wanted_ids_across_shards(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path / "pool", {"eng": ["id-1", "id-2"], "fre": ["id-3"]})
    found = build_marc_index(pool, {"id-1", "id-3"})
    assert set(found.keys()) == {"id-1", "id-3"}


def test_build_marc_index_returns_partial_when_some_ids_absent(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path / "pool", {"eng": ["id-1"]})
    found = build_marc_index(pool, {"id-1", "id-missing"})
    assert set(found.keys()) == {"id-1"}


def test_build_marc_index_short_circuits_on_empty_request(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    assert build_marc_index(pool, set()) == {}


def test_resolve_vault_pairs_scores_each_resolvable_entry() -> None:
    vault = {
        ("ctrl-a", "uuid-a"): _vault_entry("ctrl-a", "uuid-a"),
        ("ctrl-b", "uuid-b"): _vault_entry("ctrl-b", "uuid-b"),
    }
    marc_table = {"ctrl-a": _marc("ctrl-a"), "ctrl-b": _marc("ctrl-b")}
    cce_table = {"uuid-a": _cce("uuid-a"), "uuid-b": _cce("uuid-b")}

    def _builder(
        marc: MarcRecord, _cce: IndexedNyplRegRecord, candidate: CandidateMatch
    ) -> PairInsert:
        return _stub_pair(marc.control_id, candidate.nypl_uuid)

    resolved, summary = resolve_vault_pairs(
        vault=vault,
        marc_lookup=marc_table.get,
        cce_lookup=cce_table.get,
        score_pair=lambda _m, c: _candidate(0.95, c.uuid),
        build_pair=_builder,
    )
    assert summary == ResolveSummary(resolved=2, missing_in_pool=0, missing_in_index=0)
    assert {resolved_pair.entry.marc_control_id for resolved_pair in resolved} == {
        "ctrl-a",
        "ctrl-b",
    }
    assert all(isinstance(resolved_pair, ResolvedVaultPair) for resolved_pair in resolved)


def test_resolve_vault_pairs_counts_marc_missing_from_pool() -> None:
    vault = {("ctrl-gone", "uuid-a"): _vault_entry("ctrl-gone", "uuid-a")}

    def _builder(
        _m: MarcRecord, _c: IndexedNyplRegRecord, _candidate: CandidateMatch
    ) -> PairInsert:
        raise AssertionError("build_pair should not run when MARC is missing")

    resolved, summary = resolve_vault_pairs(
        vault=vault,
        marc_lookup=lambda _id: None,
        cce_lookup=lambda _u: _cce("uuid-a"),
        score_pair=lambda _m, _c: _candidate(0.95),
        build_pair=_builder,
    )
    assert resolved == []
    assert summary == ResolveSummary(resolved=0, missing_in_pool=1, missing_in_index=0)


def test_resolve_vault_pairs_counts_cce_missing_from_index() -> None:
    vault = {("ctrl-a", "uuid-gone"): _vault_entry("ctrl-a", "uuid-gone")}

    def _builder(
        _m: MarcRecord, _c: IndexedNyplRegRecord, _candidate: CandidateMatch
    ) -> PairInsert:
        raise AssertionError("build_pair should not run when CCE is missing")

    resolved, summary = resolve_vault_pairs(
        vault=vault,
        marc_lookup={"ctrl-a": _marc("ctrl-a")}.get,
        cce_lookup=lambda _u: None,
        score_pair=lambda _m, _c: _candidate(0.95),
        build_pair=_builder,
    )
    assert resolved == []
    assert summary == ResolveSummary(resolved=0, missing_in_pool=0, missing_in_index=1)


def test_make_pair_scorer_produces_candidate_match_via_matcher_pipeline() -> None:
    idf = IdfTable(
        document_count=1,
        default_idf=1.0,
        source_hash="test",
        language="eng",
        idf={},
    )
    pairings = compile_pairings(_load_default_pairing_config())
    scorer = make_pair_scorer(
        matching_config=_load_default_matching_config(),
        pairings=pairings,
        idf=idf,
        calibrator=None,
    )
    candidate = scorer(_marc("ctrl-1"), _cce("uuid-1"))
    assert candidate.nypl_uuid == "uuid-1"
    assert candidate.nypl_year == 1953
    assert candidate.combined.raw >= 0.0


def test_resolve_vault_pairs_handles_empty_vault() -> None:
    resolved, summary = resolve_vault_pairs(
        vault={},
        marc_lookup=lambda _id: None,
        cce_lookup=lambda _u: None,
        score_pair=lambda _m, _c: _candidate(0.95),
        build_pair=lambda _m, _c, _cand: _stub_pair("x", "y"),
    )
    assert resolved == []
    assert summary == ResolveSummary(resolved=0, missing_in_pool=0, missing_in_index=0)
