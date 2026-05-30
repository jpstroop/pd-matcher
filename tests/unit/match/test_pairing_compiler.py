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
    statement_of_responsibility: str | None = None,
    publisher: str | None = None,
) -> MarcRecord:
    return MarcRecord(
        control_id="m",
        title=title,
        title_main=title_main,
        main_author=main_author,
        series_titles=series_titles,
        statement_of_responsibility=statement_of_responsibility,
        publisher=publisher,
    )


def _nypl(
    *,
    title: str = "CCE title",
    author_name: str | None = "CCE author",
    publisher_names: tuple[str, ...] = (),
    claimants: tuple[str, ...] = (),
    renewal_author: str | None = None,
    renewal_title: str | None = None,
    renewal_claimants: str | None = None,
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title=title,
        was_renewed=False,
        author_name=author_name,
        publisher_names=publisher_names,
        claimants=claimants,
        renewal_author=renewal_author,
        renewal_title=renewal_title,
        renewal_claimants=renewal_claimants,
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


def test_cce_registry_exposes_renewal_author() -> None:
    """``renewal_author`` surfaces :attr:`IndexedNyplRegRecord.renewal_author`."""
    assert CCE_FIELDS["renewal_author"](_nypl(renewal_author="RA")) == ("RA",)
    assert CCE_FIELDS["renewal_author"](_nypl(renewal_author=None)) == ()


def test_cce_registry_exposes_renewal_title() -> None:
    """``renewal_title`` surfaces :attr:`IndexedNyplRegRecord.renewal_title`."""
    assert CCE_FIELDS["renewal_title"](_nypl(renewal_title="RT")) == ("RT",)
    assert CCE_FIELDS["renewal_title"](_nypl(renewal_title=None)) == ()


def test_cce_registry_exposes_renewal_claimants() -> None:
    """``renewal_claimants`` surfaces the renewal claimants string."""
    assert CCE_FIELDS["renewal_claimants"](_nypl(renewal_claimants="RC1; RC2")) == ("RC1; RC2",)
    assert CCE_FIELDS["renewal_claimants"](_nypl(renewal_claimants=None)) == ()


def test_compile_renewal_author_pairing_uses_renewal_field() -> None:
    """A pairing referencing ``renewal_author`` reads the renewal field."""
    cfg = PairingConfig(
        marc_fields={"ma": FieldSpec(fields=("main_author",), combine="first")},
        cce_fields={"ra": FieldSpec(fields=("renewal_author",), combine="first")},
        pairings=(PairingSpec(group="author", marc="ma", cce="ra"),),
    )
    compiled = compile_pairings(cfg)
    cce_accessor = compiled.author[0].cce_accessor
    assert cce_accessor(_nypl(renewal_author="Renewal Author")) == "Renewal Author"


def test_compile_renewal_title_pairing_uses_renewal_field() -> None:
    """A pairing referencing ``renewal_title`` reads the renewal field."""
    cfg = PairingConfig(
        marc_fields={"tf": FieldSpec(fields=("title",), combine="first")},
        cce_fields={"rt": FieldSpec(fields=("renewal_title",), combine="first")},
        pairings=(PairingSpec(group="title", marc="tf", cce="rt"),),
    )
    compiled = compile_pairings(cfg)
    cce_accessor = compiled.title[0].cce_accessor
    assert cce_accessor(_nypl(renewal_title="Renewal Title")) == "Renewal Title"


def test_compile_publisher_claimants_pairing_reads_claimants() -> None:
    """A ``publisher`` pairing referencing ``claimants`` reads the claimants list."""
    cfg = PairingConfig(
        marc_fields={"pub": FieldSpec(fields=("publisher",), combine="first")},
        cce_fields={"cl": FieldSpec(fields=("claimants",), combine="join")},
        pairings=(PairingSpec(group="publisher", marc="pub", cce="cl"),),
    )
    compiled = compile_pairings(cfg)
    cce_accessor = compiled.publisher[0].cce_accessor
    assert cce_accessor(_nypl(claimants=("Acme Co", "Sons"))) == "Acme Co Sons"


def test_compile_publisher_sor_title_pairing_reads_sor_and_cce_title() -> None:
    """A ``publisher`` pairing of ``sor ↔ title`` reads MARC SoR and CCE title."""
    cfg = PairingConfig(
        marc_fields={
            "sor": FieldSpec(fields=("statement_of_responsibility",), combine="first"),
        },
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(PairingSpec(group="publisher", marc="sor", cce="t"),),
    )
    compiled = compile_pairings(cfg)
    pairing = compiled.publisher[0]
    assert pairing.marc_accessor(_marc(statement_of_responsibility="by Levy")) == "by Levy"
    assert pairing.cce_accessor(_nypl(title="Annotated translation")) == "Annotated translation"


def test_compile_publisher_sor_publisher_names_pairing_reads_publisher_names() -> None:
    """A ``publisher`` pairing of ``sor ↔ publisher_names`` reads the joined names."""
    cfg = PairingConfig(
        marc_fields={
            "sor": FieldSpec(fields=("statement_of_responsibility",), combine="first"),
        },
        cce_fields={"pn": FieldSpec(fields=("publisher_names",), combine="join")},
        pairings=(PairingSpec(group="publisher", marc="sor", cce="pn"),),
    )
    compiled = compile_pairings(cfg)
    pairing = compiled.publisher[0]
    assert pairing.marc_accessor(_marc(statement_of_responsibility="by Matsumoto")) == (
        "by Matsumoto"
    )
    assert pairing.cce_accessor(_nypl(publisher_names=("Ryozo Matsumoto",))) == "Ryozo Matsumoto"


def test_compile_publisher_sor_claimants_pairing_reads_claimants() -> None:
    """A ``publisher`` pairing of ``sor ↔ claimants`` reads the joined claimants."""
    cfg = PairingConfig(
        marc_fields={
            "sor": FieldSpec(fields=("statement_of_responsibility",), combine="first"),
        },
        cce_fields={"cl": FieldSpec(fields=("claimants",), combine="join")},
        pairings=(PairingSpec(group="publisher", marc="sor", cce="cl"),),
    )
    compiled = compile_pairings(cfg)
    pairing = compiled.publisher[0]
    assert pairing.marc_accessor(_marc(statement_of_responsibility="by Levy")) == "by Levy"
    assert pairing.cce_accessor(_nypl(claimants=("Howard S. Levy",))) == "Howard S. Levy"


def test_compile_renewal_claimants_pairing_uses_renewal_field() -> None:
    """A pairing referencing ``renewal_claimants`` reads the renewal field."""
    cfg = PairingConfig(
        marc_fields={"ma": FieldSpec(fields=("main_author",), combine="first")},
        cce_fields={"rc": FieldSpec(fields=("renewal_claimants",), combine="first")},
        pairings=(PairingSpec(group="author", marc="ma", cce="rc"),),
    )
    compiled = compile_pairings(cfg)
    cce_accessor = compiled.author[0].cce_accessor
    assert cce_accessor(_nypl(renewal_claimants="X; Y")) == "X; Y"


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
