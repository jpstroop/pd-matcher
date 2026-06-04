"""Publisher imprint / alias lookup table backing the publisher scorer.

The shipped ``data/publishers/publisher_imprints.json`` records canonical
publisher names, their commercial aliases, and the imprint-level child names
that historically published under each house. The matcher uses this table to
lift the publisher score on pairs where a MARC record cites an imprint
(``Whittlesey House``) and the CCE record cites the parent
(``McGraw-Hill Book Company``) — a class of mismatch the fuzzy scorer
cannot recover on its own.

The loader returns frozen msgspec Structs; :func:`build_alias_index`
flattens every canonical / alias / imprint name into a normalized lookup
dict whose value is the *normalized canonical*, so callers can compare
both sides after running them through :func:`normalize_publisher`.
"""

from pathlib import Path
from re import compile as re_compile

from msgspec import Struct
from msgspec.json import decode as msgspec_json_decode


class Imprint(Struct, frozen=True, forbid_unknown_fields=True):
    """One imprint published under a parent house.

    Attributes:
        name: Display name of the imprint.
        active: Free-form active-years string (``"1929-1968"``) or ``None``.
        notes: Curator notes describing the imprint, or ``None``.
    """

    name: str
    active: str | None = None
    notes: str | None = None


class PublisherEntry(Struct, frozen=True, forbid_unknown_fields=True):
    """One canonical publisher plus its aliases and imprints.

    Attributes:
        canonical: The preferred display name for the house.
        aliases: Other names the same house has been recorded under.
        imprints: Imprint-level children of the house.
        active: Free-form active-years string for the house, or ``None``.
        notes: Curator notes describing the house, or ``None``.
        sources: Provenance URLs / citation keys for the entry.
    """

    canonical: str
    aliases: tuple[str, ...] = ()
    imprints: tuple[Imprint, ...] = ()
    active: str | None = None
    notes: str | None = None
    sources: tuple[str, ...] = ()


class PublisherTable(Struct, frozen=True, forbid_unknown_fields=True):
    """The whole shipped publisher table.

    Attributes:
        schema_version: Format version stamped on the JSON file.
        publishers: Every :class:`PublisherEntry` in the table.
    """

    schema_version: int
    publishers: tuple[PublisherEntry, ...]


DEFAULT_PUBLISHER_TABLE_PATH: Path = (
    Path(__file__).resolve().parents[3] / "data" / "publishers" / "publisher_imprints.json"
)

_PUBLISHER_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "&",
        "co",
        "co.",
        "company",
        "corp",
        "corp.",
        "corporation",
        "inc",
        "inc.",
        "incorporated",
        "ltd",
        "ltd.",
        "limited",
        "publishing",
        "publishers",
        "publisher",
        "press",
        "publications",
        "publication",
        "pub",
        "pub.",
        "books",
        "book",
        "comp",
        "editorial",
        "editions",
        "verlag",
        "librairie",
        "et",
        "cie",
        "sons",
        "of",
    }
)

_NON_ALNUM = re_compile(r"[^a-z0-9\s]+")
_WHITESPACE = re_compile(r"\s+")


def normalize_publisher(raw: str) -> str:
    """Lower, strip punctuation, drop publisher-stopwords, collapse whitespace.

    Args:
        raw: Source string as it appears in MARC / CCE / the lookup table.

    Returns:
        A whitespace-joined sequence of alphanumeric tokens with the
        bundled stopword set removed. Empty when the input collapses to
        nothing useful (pure punctuation, pure stopwords, or empty).
    """
    lowered = raw.lower()
    no_punct = _NON_ALNUM.sub(" ", lowered)
    tokens = no_punct.split()
    kept = [token for token in tokens if token not in _PUBLISHER_STOPWORDS]
    if not kept:
        return ""
    return _WHITESPACE.sub(" ", " ".join(kept)).strip()


def load_publisher_table(path: Path) -> PublisherTable:
    """Decode a publisher table JSON file into a :class:`PublisherTable`.

    Args:
        path: Filesystem location of the JSON file.

    Returns:
        The parsed :class:`PublisherTable`.

    Raises:
        msgspec.ValidationError: When the file carries unknown fields or
            structurally invalid data.
    """
    return msgspec_json_decode(path.read_bytes(), type=PublisherTable)


def build_alias_index(table: PublisherTable) -> dict[str, str]:
    """Flatten every canonical / alias / imprint name into a lookup dict.

    Each key is the *normalized* form of a name (via
    :func:`normalize_publisher`) and each value is the *normalized*
    canonical of the owning :class:`PublisherEntry`. Empty normalized
    forms are skipped so a stopword-only alias cannot poison the table.

    Args:
        table: A loaded :class:`PublisherTable`.

    Returns:
        A ``{normalized_name: normalized_canonical}`` dict.
    """
    index: dict[str, str] = {}
    for entry in table.publishers:
        canonical_key = normalize_publisher(entry.canonical)
        if not canonical_key:
            continue
        index[canonical_key] = canonical_key
        for alias in entry.aliases:
            alias_key = normalize_publisher(alias)
            if alias_key:
                index[alias_key] = canonical_key
        for imprint in entry.imprints:
            imprint_key = normalize_publisher(imprint.name)
            if imprint_key:
                index[imprint_key] = canonical_key
    return index


_DEFAULT_INDEX: dict[str, str] | None = None


def get_default_alias_index() -> dict[str, str]:
    """Return the cached alias index built from the bundled publisher table.

    The bundled file is read and indexed on first call and cached for the
    lifetime of the process; subsequent calls reuse the cached dict.
    """
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = build_alias_index(load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH))
    return _DEFAULT_INDEX


__all__ = [
    "DEFAULT_PUBLISHER_TABLE_PATH",
    "Imprint",
    "PublisherEntry",
    "PublisherTable",
    "build_alias_index",
    "get_default_alias_index",
    "load_publisher_table",
    "normalize_publisher",
]
