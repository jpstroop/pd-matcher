"""Organization-vs-person detection for cross-field publisher pairing.

The publisher scorer group compares the MARC publisher / statement of
responsibility against several CCE fields. Two of those CCE fields —
``author_name`` and ``claimants`` — are *cross-field* fallbacks: they exist
to catch a **corporate** claimant that is in fact the publisher (claimant
``Knopf`` == MARC publisher ``Knopf``).

That fallback mis-fires whenever the CCE comparand is a **person** rather
than an organization (issue #86): when the CCE record carries no publisher
of its own, a person comparand produces a fabricated ``name.publisher = 0.0``
that drags a genuine match down. A person is never a publisher, so a blank
-publisher cross-field pairing should only consume a CCE comparand that
*looks like an organization*. This module is the single, fully-typed
predicate for that decision. The heuristic is deliberately conservative: a
name is an organization only when it carries an unambiguous corporate /
institutional marker token (``press``, ``inc``, ``university``, ``society``
…). Bare personal names — inverted (``"Bannister, Frank Theodore"``) or
natural (``"Gerald S. Snyder"``) — carry no such marker and are classified
as persons, so the cross-field pairing skips them.
"""

from pd_matcher.normalize.text import tokenize

_ORGANIZATION_MARKERS: frozenset[str] = frozenset(
    {
        "co",
        "company",
        "comp",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "plc",
        "press",
        "publishing",
        "publishers",
        "publisher",
        "publications",
        "publication",
        "pub",
        "books",
        "editorial",
        "editions",
        "verlag",
        "librairie",
        "cie",
        "sons",
        "bros",
        "brothers",
        "university",
        "college",
        "school",
        "institute",
        "institution",
        "association",
        "assn",
        "society",
        "foundation",
        "trust",
        "council",
        "committee",
        "commission",
        "bureau",
        "office",
        "department",
        "division",
        "ministry",
        "agency",
        "museum",
        "library",
        "academy",
        "workshop",
        "conference",
        "congress",
        "church",
        "mission",
        "order",
        "guild",
        "league",
        "club",
        "union",
        "federation",
        "organization",
        "organisation",
        "manufacturing",
        "mfg",
        "industries",
        "products",
        "services",
        "service",
        "bank",
        "international",
        "national",
        "american",
    }
)


def looks_like_organization(name: str | None) -> bool:
    """Return ``True`` when ``name`` carries an organization / imprint marker.

    A name is treated as an organization only when at least one of its
    normalized tokens is an unambiguous corporate or institutional marker
    (:data:`_ORGANIZATION_MARKERS`). Personal names — inverted
    (``"Bannister, Frank"``) or natural (``"Gerald S. Snyder"``) — carry no
    such marker and return ``False``.

    Args:
        name: A raw CCE author / claimant string, or ``None``.

    Returns:
        ``True`` if the name looks like an organization, ``False`` for a
        personal name, an empty string, or ``None``.
    """
    if not name:
        return False
    return any(token in _ORGANIZATION_MARKERS for token in tokenize(name))


__all__ = [
    "looks_like_organization",
]
