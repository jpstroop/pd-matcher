"""Tests for :mod:`pd_matcher.normalize.publishers`."""

from pathlib import Path

from msgspec import ValidationError
from pytest import raises

from pd_matcher.normalize.publishers import DEFAULT_PUBLISHER_TABLE_PATH
from pd_matcher.normalize.publishers import Imprint
from pd_matcher.normalize.publishers import PublisherEntry
from pd_matcher.normalize.publishers import PublisherTable
from pd_matcher.normalize.publishers import build_alias_index
from pd_matcher.normalize.publishers import get_default_alias_index
from pd_matcher.normalize.publishers import load_publisher_table
from pd_matcher.normalize.publishers import normalize_publisher


def test_bundled_publisher_table_roundtrips() -> None:
    """The shipped data file decodes without error."""
    table = load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH)
    assert table.schema_version == 1
    assert len(table.publishers) > 0


def test_load_publisher_table_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    """``forbid_unknown_fields`` is enforced at the top level."""
    fixture = tmp_path / "bad.json"
    fixture.write_text(
        '{"schema_version": 1, "publishers": [], "extra": "nope"}',
        encoding="utf-8",
    )
    with raises(ValidationError):
        load_publisher_table(fixture)


def test_load_publisher_table_rejects_unknown_publisher_fields(tmp_path: Path) -> None:
    """``forbid_unknown_fields`` is enforced on nested entries."""
    fixture = tmp_path / "bad.json"
    fixture.write_text(
        '{"schema_version": 1, "publishers": [{"canonical": "Acme", "bogus": 1}]}',
        encoding="utf-8",
    )
    with raises(ValidationError):
        load_publisher_table(fixture)


def test_load_publisher_table_rejects_unknown_imprint_fields(tmp_path: Path) -> None:
    """``forbid_unknown_fields`` is enforced on imprints."""
    fixture = tmp_path / "bad.json"
    fixture.write_text(
        '{"schema_version": 1, "publishers": ['
        '{"canonical": "Acme", "imprints": [{"name": "X", "bogus": 1}]}]}',
        encoding="utf-8",
    )
    with raises(ValidationError):
        load_publisher_table(fixture)


def test_normalize_publisher_strips_punctuation_and_stopwords() -> None:
    """A house with a long suffix collapses to its distinguishing tokens."""
    assert normalize_publisher("McGraw-Hill Book Company, Inc.") == "mcgraw hill"


def test_normalize_publisher_empty_input_returns_empty() -> None:
    """Empty input yields the empty string."""
    assert normalize_publisher("") == ""


def test_normalize_publisher_all_stopwords_returns_empty() -> None:
    """A pure-stopword input collapses to the empty string."""
    assert normalize_publisher("The Company & Co., Inc.") == ""


def test_normalize_publisher_collapses_whitespace() -> None:
    """Internal punctuation should not leave double spaces."""
    assert normalize_publisher("Doubleday,  Doran") == "doubleday doran"


def test_normalize_publisher_keeps_alphanumeric_tokens() -> None:
    """Numerics and ampersand-stripped joins survive normalization."""
    assert normalize_publisher("ASTM") == "astm"
    assert normalize_publisher("M.I.T. Press") == "m i t"


def test_build_alias_index_maps_imprint_to_parent_canonical() -> None:
    """``Whittlesey House`` resolves to the McGraw-Hill human canonical."""
    table = load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH)
    index = build_alias_index(table)
    whittlesey = normalize_publisher("Whittlesey House")
    assert whittlesey in index
    assert index[whittlesey] == "McGraw-Hill Book Company"


def test_build_alias_index_maps_alias_to_canonical() -> None:
    """``Aldus Books`` (an imprint) resolves to the Doubleday human canonical."""
    table = load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH)
    index = build_alias_index(table)
    aldus = normalize_publisher("Aldus Books")
    assert index[aldus] == "Doubleday & Company"


def test_build_alias_index_maps_clarendon_to_oxford() -> None:
    """``Clarendon Press`` resolves to the Oxford University Press canonical."""
    table = load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH)
    index = build_alias_index(table)
    clarendon = normalize_publisher("Clarendon Press")
    assert index[clarendon] == "Oxford University Press"


def test_build_alias_index_includes_canonicals_themselves() -> None:
    """Each canonical form maps to its human-readable entry name."""
    table = load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH)
    index = build_alias_index(table)
    for entry in table.publishers:
        canonical_key = normalize_publisher(entry.canonical)
        if canonical_key:
            assert index[canonical_key] == entry.canonical


def test_build_alias_index_skips_empty_canonical_entries() -> None:
    """An entry whose canonical normalizes to empty contributes no keys."""
    table = PublisherTable(
        schema_version=1,
        publishers=(
            PublisherEntry(
                canonical="The & Co.",
                aliases=("Real Name",),
                imprints=(Imprint(name="Real Imprint"),),
            ),
        ),
    )
    index = build_alias_index(table)
    assert index == {}


def test_build_alias_index_skips_empty_alias_keys() -> None:
    """An alias whose normalized form is empty does not poison the index."""
    table = PublisherTable(
        schema_version=1,
        publishers=(
            PublisherEntry(
                canonical="Real House",
                aliases=("The Co.",),
                imprints=(Imprint(name="& Inc."),),
            ),
        ),
    )
    index = build_alias_index(table)
    canonical_key = normalize_publisher("Real House")
    assert index == {canonical_key: "Real House"}


def test_bundled_canonicals_are_unique() -> None:
    """Authoring stayed consistent: no duplicate canonical names."""
    table = load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH)
    canonicals = [entry.canonical for entry in table.publishers]
    assert len(canonicals) == len(set(canonicals))


def test_bundled_imprint_names_are_nonempty() -> None:
    """Every imprint carries a non-empty name after stripping."""
    table = load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH)
    for entry in table.publishers:
        for imprint in entry.imprints:
            assert imprint.name.strip()


def test_get_default_alias_index_is_cached() -> None:
    """Repeated calls return the same dict instance."""
    first = get_default_alias_index()
    second = get_default_alias_index()
    assert first is second


def test_get_default_alias_index_contains_known_pairs() -> None:
    """The cached default index resolves the analytical-pass anchors."""
    index = get_default_alias_index()
    whittlesey = normalize_publisher("Whittlesey House")
    assert index[whittlesey] == "McGraw-Hill Book Company"
