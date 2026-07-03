"""Tests for :mod:`pd_matcher.match.idf`."""

from pathlib import Path

from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.idf import load_idf_table
from pd_matcher.match.idf import load_or_build_author_idf
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.idf import load_or_build_publisher_idf
from pd_matcher.match.idf import save_idf_table

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _build_tiny_index(root: Path) -> Path:
    reg_dir = root / "reg"
    ren_dir = root / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = root / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def test_build_idf_table_counts_documents_and_emits_token_scores(tmp_path: Path) -> None:
    """The built table records document_count and per-token IDF entries."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        table = build_idf_table(lookup)
    assert table.document_count > 0
    assert table.default_idf > 0.0
    assert table.language == "eng"
    # Every recorded token's IDF is finite and positive.
    assert all(value > 0.0 for value in table.idf.values())


def test_idf_score_falls_back_to_default_for_unknown_tokens(tmp_path: Path) -> None:
    """Unknown tokens get the table's ``default_idf``."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        table = build_idf_table(lookup)
    assert table.score("a-token-that-cannot-possibly-be-in-the-fixture") == table.default_idf


def test_idf_table_roundtrips_via_msgpack(tmp_path: Path) -> None:
    """save + load reproduces the IDF table exactly."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        table = build_idf_table(lookup)
    cache_path = tmp_path / "idf.msgpack"
    save_idf_table(table, cache_path)
    loaded = load_idf_table(cache_path)
    assert loaded == table


def test_load_or_build_idf_creates_cache_when_missing(tmp_path: Path) -> None:
    """When no cache exists, the function builds and writes one."""
    out_path = _build_tiny_index(tmp_path)
    cache_path = tmp_path / "idf.msgpack"
    table = load_or_build_idf(cache_path, lambda: NyplIndexLookup(out_path))
    assert cache_path.exists()
    assert isinstance(table, IdfTable)


def test_load_or_build_idf_reuses_cache_when_source_hash_matches(tmp_path: Path) -> None:
    """When the cached source hash matches, the cache is returned verbatim."""
    out_path = _build_tiny_index(tmp_path)
    cache_path = tmp_path / "idf.msgpack"
    first = load_or_build_idf(cache_path, lambda: NyplIndexLookup(out_path))
    second = load_or_build_idf(cache_path, lambda: NyplIndexLookup(out_path))
    assert first == second


def test_load_or_build_idf_rebuilds_when_language_differs(tmp_path: Path) -> None:
    """Cached language mismatch triggers a rebuild."""
    out_path = _build_tiny_index(tmp_path)
    cache_path = tmp_path / "idf.msgpack"
    load_or_build_idf(cache_path, lambda: NyplIndexLookup(out_path), language="eng")
    rebuilt = load_or_build_idf(cache_path, lambda: NyplIndexLookup(out_path), language="fre")
    assert rebuilt.language == "fre"


def test_load_or_build_idf_rebuilds_when_source_hash_changes(tmp_path: Path) -> None:
    """A stale cache (different ``source_hash``) is rebuilt and overwritten."""
    out_path = _build_tiny_index(tmp_path)
    cache_path = tmp_path / "idf.msgpack"
    fake = IdfTable(
        document_count=1,
        default_idf=1.0,
        source_hash="stale-hash",
        language="eng",
        idf={"x": 1.0},
    )
    save_idf_table(fake, cache_path)
    rebuilt = load_or_build_idf(cache_path, lambda: NyplIndexLookup(out_path))
    assert rebuilt.source_hash != "stale-hash"
    assert rebuilt.document_count > 1


def test_build_author_idf_table_counts_documents_and_emits_token_scores(
    tmp_path: Path,
) -> None:
    """The author table records document_count and per-token IDF entries."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        table = build_author_idf_table(lookup)
    assert table.document_count > 0
    assert table.default_idf > 0.0
    assert table.language == "eng"
    assert all(value > 0.0 for value in table.idf.values())
    assert "smith" in table.idf


def test_build_publisher_idf_table_counts_documents_and_emits_token_scores(
    tmp_path: Path,
) -> None:
    """The publisher table records document_count and per-token IDF entries."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        table = build_publisher_idf_table(lookup)
    assert table.document_count > 0
    assert table.default_idf > 0.0
    assert table.language == "eng"
    assert all(value > 0.0 for value in table.idf.values())
    assert "acme" in table.idf


def test_load_or_build_author_idf_creates_cache_when_missing(tmp_path: Path) -> None:
    """When no author cache exists, the function builds and writes one."""
    out_path = _build_tiny_index(tmp_path)
    cache_path = tmp_path / "author_idf.msgpack"
    table = load_or_build_author_idf(cache_path, lambda: NyplIndexLookup(out_path))
    assert cache_path.exists()
    assert isinstance(table, IdfTable)


def test_load_or_build_publisher_idf_creates_cache_when_missing(tmp_path: Path) -> None:
    """When no publisher cache exists, the function builds and writes one."""
    out_path = _build_tiny_index(tmp_path)
    cache_path = tmp_path / "publisher_idf.msgpack"
    table = load_or_build_publisher_idf(cache_path, lambda: NyplIndexLookup(out_path))
    assert cache_path.exists()
    assert isinstance(table, IdfTable)


def test_iter_registrations_walks_every_record(tmp_path: Path) -> None:
    """The new lookup helper visits every registration in the index."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        seen_uuids = sorted(record.uuid for record in lookup.iter_registrations())
    assert "UUID-0001" in seen_uuids
    assert len(seen_uuids) > 0
