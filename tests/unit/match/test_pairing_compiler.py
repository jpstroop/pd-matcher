"""Tests for :mod:`pd_matcher.match.pairing_compiler`."""

from pytest import raises

from pd_matcher.config.loader import ConfigError
from pd_matcher.config.schemas import FieldSpec
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.config.schemas import PairingSpec
from pd_matcher.match.pairing_compiler import CCE_FIELDS
from pd_matcher.match.pairing_compiler import MARC_FIELDS
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


def _marc(
    *,
    title: str = "Full title",
    title_main: str = "Main",
    main_author: str | None = "Author",
    series_titles: tuple[str, ...] = (),
) -> MarcRecord:
    return MarcRecord(
        control_id="m",
        title=title,
        title_main=title_main,
        main_author=main_author,
        series_titles=series_titles,
    )


def _nypl(
    *,
    title: str = "CCE title",
    author_name: str | None = "CCE author",
    publisher_names: tuple[str, ...] = (),
    claimants: tuple[str, ...] = (),
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title=title,
        was_renewed=False,
        author_name=author_name,
        publisher_names=publisher_names,
        claimants=claimants,
    )


def test_marc_registry_scalar_wraps_none_to_empty_tuple() -> None:
    """A ``None`` scalar raw field yields an empty tuple."""
    assert MARC_FIELDS["main_author"](_marc(main_author=None)) == ()


def test_marc_registry_scalar_wraps_value_to_singleton() -> None:
    """A present scalar raw field yields a 1-tuple."""
    assert MARC_FIELDS["title_main"](_marc(title_main="X")) == ("X",)


def test_marc_registry_passes_tuple_field_through() -> None:
    """A list-valued raw field passes through unchanged."""
    assert MARC_FIELDS["series_titles"](_marc(series_titles=("A", "B"))) == ("A", "B")


def test_cce_registry_passes_tuple_field_through() -> None:
    """A CCE list-valued raw field passes through unchanged."""
    assert CCE_FIELDS["publisher_names"](_nypl(publisher_names=("P1", "P2"))) == ("P1", "P2")


def test_combine_first_returns_first_non_empty() -> None:
    """``first`` skips empties and returns the first non-empty value."""
    cfg = PairingConfig(
        marc_fields={"f": FieldSpec(fields=("series_titles",), combine="first")},
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(PairingSpec(group="title", marc="f", cce="t"),),
    )
    compiled = compile_pairings(cfg)
    accessor = compiled.title[0].marc_accessor
    assert accessor(_marc(series_titles=("", "Second"))) == "Second"


def test_combine_first_returns_none_when_all_empty() -> None:
    """``first`` returns ``None`` when every candidate value is empty."""
    cfg = PairingConfig(
        marc_fields={"f": FieldSpec(fields=("series_titles",), combine="first")},
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(PairingSpec(group="title", marc="f", cce="t"),),
    )
    compiled = compile_pairings(cfg)
    accessor = compiled.title[0].marc_accessor
    assert accessor(_marc(series_titles=("", ""))) is None


def test_combine_join_joins_non_empty_with_separator() -> None:
    """``join`` concatenates non-empty values with the separator."""
    cfg = PairingConfig(
        marc_fields={"f": FieldSpec(fields=("title_main",), combine="first")},
        cce_fields={"c": FieldSpec(fields=("claimants",), combine="join", separator=", ")},
        pairings=(PairingSpec(group="author", marc="f", cce="c"),),
    )
    compiled = compile_pairings(cfg)
    accessor = compiled.author[0].cce_accessor
    assert accessor(_nypl(claimants=("A", "", "B"))) == "A, B"


def test_combine_concat_is_synonym_for_join() -> None:
    """``concat`` behaves identically to ``join``."""
    cfg = PairingConfig(
        marc_fields={"f": FieldSpec(fields=("title_main",), combine="first")},
        cce_fields={"c": FieldSpec(fields=("claimants",), combine="concat")},
        pairings=(PairingSpec(group="author", marc="f", cce="c"),),
    )
    compiled = compile_pairings(cfg)
    accessor = compiled.author[0].cce_accessor
    assert accessor(_nypl(claimants=("A", "B"))) == "A B"


def test_combine_join_returns_none_when_all_empty() -> None:
    """``join`` returns ``None`` when no non-empty value remains."""
    cfg = PairingConfig(
        marc_fields={"f": FieldSpec(fields=("title_main",), combine="first")},
        cce_fields={"c": FieldSpec(fields=("claimants",), combine="join")},
        pairings=(PairingSpec(group="author", marc="f", cce="c"),),
    )
    compiled = compile_pairings(cfg)
    accessor = compiled.author[0].cce_accessor
    assert accessor(_nypl(claimants=())) is None


def test_compile_buckets_pairings_by_group() -> None:
    """Pairings are bucketed into the three scorer groups."""
    cfg = PairingConfig(
        marc_fields={
            "tm": FieldSpec(fields=("title_main",), combine="first"),
            "ma": FieldSpec(fields=("main_author",), combine="first"),
        },
        cce_fields={
            "t": FieldSpec(fields=("title",), combine="first"),
            "an": FieldSpec(fields=("author_name",), combine="first"),
            "pn": FieldSpec(fields=("publisher_names",), combine="join"),
        },
        pairings=(
            PairingSpec(group="title", marc="tm", cce="t"),
            PairingSpec(group="author", marc="ma", cce="an"),
            PairingSpec(group="publisher", marc="ma", cce="pn"),
        ),
    )
    compiled = compile_pairings(cfg)
    assert len(compiled.title) == 1
    assert len(compiled.author) == 1
    assert len(compiled.publisher) == 1


def test_compile_raises_on_unknown_marc_raw_field() -> None:
    """A bad raw name in a marc FieldSpec fails at compile time."""
    cfg = PairingConfig(
        marc_fields={"bad": FieldSpec(fields=("nope",), combine="first")},
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(PairingSpec(group="title", marc="bad", cce="t"),),
    )
    with raises(ConfigError, match="unknown raw MARC field 'nope'"):
        compile_pairings(cfg)


def test_compile_raises_on_unknown_cce_raw_field() -> None:
    """A bad raw name in a cce FieldSpec fails at compile time."""
    cfg = PairingConfig(
        marc_fields={"tm": FieldSpec(fields=("title_main",), combine="first")},
        cce_fields={"bad": FieldSpec(fields=("nope",), combine="first")},
        pairings=(PairingSpec(group="title", marc="tm", cce="bad"),),
    )
    with raises(ConfigError, match="unknown raw CCE field 'nope'"):
        compile_pairings(cfg)


def test_compile_raises_on_unknown_marc_pairing_reference() -> None:
    """A pairing referencing an undefined marc field fails at compile time."""
    cfg = PairingConfig(
        marc_fields={"tm": FieldSpec(fields=("title_main",), combine="first")},
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(PairingSpec(group="title", marc="missing", cce="t"),),
    )
    with raises(ConfigError, match="unknown marc field 'missing'"):
        compile_pairings(cfg)


def test_compile_raises_on_unknown_cce_pairing_reference() -> None:
    """A pairing referencing an undefined cce field fails at compile time."""
    cfg = PairingConfig(
        marc_fields={"tm": FieldSpec(fields=("title_main",), combine="first")},
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(PairingSpec(group="title", marc="tm", cce="missing"),),
    )
    with raises(ConfigError, match="unknown cce field 'missing'"):
        compile_pairings(cfg)
