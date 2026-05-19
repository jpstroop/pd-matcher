"""Pure predicate primitives consumed by the YAML rule engine.

Each predicate is a function ``(facts, *args) -> bool`` over a
:class:`~pd_matcher.copyright.facts.Facts`. The rule engine resolves the
``predicate:`` name in YAML to a callable in this module (or an
inference wrapper in :mod:`pd_matcher.copyright.inference`).

The first predicate evaluated for every record is :func:`in_pd_by_age`,
the moving wall. The wall advances one year every 1 January because US
copyright duration for pre-1978 works runs 95 years from publication;
expressing that as ``as_of_year - 95`` rather than a hard-coded ``1929``
keeps the engine correct without a yearly code change.

MARC 008 country codes are three-character. The US set is large because
each state and territory has its own code; we ship the canonical list
from the MARC Code List for Countries (Library of Congress) so the
predicate distinguishes US-published from foreign-published records.
"""

from pd_matcher.copyright.facts import Facts

_US_COUNTRY_CODES: frozenset[str] = frozenset(
    {
        "aku",
        "alu",
        "aru",
        "azu",
        "cau",
        "cou",
        "ctu",
        "dcu",
        "deu",
        "flu",
        "gau",
        "hiu",
        "iau",
        "idu",
        "ilu",
        "inu",
        "ksu",
        "kyu",
        "lau",
        "mau",
        "mdu",
        "meu",
        "miu",
        "mnu",
        "mou",
        "msu",
        "mtu",
        "nbu",
        "ncu",
        "ndu",
        "nhu",
        "nju",
        "nmu",
        "nvu",
        "nyu",
        "ohu",
        "oku",
        "oru",
        "pau",
        "riu",
        "scu",
        "sdu",
        "tnu",
        "txu",
        "utu",
        "vau",
        "vtu",
        "wau",
        "wiu",
        "wvu",
        "wyu",
        "xxu",
    }
)

_NO_TREATY_COUNTRY_CODES: frozenset[str] = frozenset(
    {
        "er",
        "et",
        "ir",
        "iq",
        "xe",
        "sm",
    }
)

_DELAYED_URAA_COUNTRY_CODES: frozenset[str] = frozenset(
    {
        "af",
        "ae",
        "ao",
        "bt",
        "bn",
        "kn",
        "ws",
        "sa",
        "vm",
        "ye",
    }
)


def in_pd_by_age(facts: Facts) -> bool:
    """Return ``True`` when ``pub_year`` is more than 95 years before ``as_of_year``.

    This is the moving-wall short-circuit. As of 2026 it fires for
    every US-published work with ``pub_year <= 1930``. Every 1 January
    the wall advances one year automatically once callers pass the new
    ``as_of_year``.
    """
    if facts.pub_year is None:
        return False
    return facts.pub_year < facts.as_of_year - 95


def published_between(facts: Facts, lo: int, hi: int) -> bool:
    """Return ``True`` when ``lo <= pub_year <= hi``."""
    if facts.pub_year is None:
        return False
    return lo <= facts.pub_year <= hi


def published_before(facts: Facts, year: int) -> bool:
    """Return ``True`` when ``pub_year < year``."""
    if facts.pub_year is None:
        return False
    return facts.pub_year < year


def published_on_or_after(facts: Facts, year: int) -> bool:
    """Return ``True`` when ``pub_year >= year``."""
    if facts.pub_year is None:
        return False
    return facts.pub_year >= year


def country_is_us(facts: Facts) -> bool:
    """Return ``True`` when the MARC country code denotes a US jurisdiction."""
    if facts.pub_country_code is None:
        return False
    return facts.pub_country_code.lower() in _US_COUNTRY_CODES


def country_is_foreign(facts: Facts) -> bool:
    """Return ``True`` when the country code is present and non-US."""
    if facts.pub_country_code is None:
        return False
    return facts.pub_country_code.lower() not in _US_COUNTRY_CODES


def country_no_treaty(facts: Facts) -> bool:
    """Return ``True`` when the country has no copyright relations with the US."""
    if facts.pub_country_code is None:
        return False
    return facts.pub_country_code.lower() in _NO_TREATY_COUNTRY_CODES


def country_delayed_uraa(facts: Facts) -> bool:
    """Return ``True`` when the country has a non-1996 URAA restoration date."""
    if facts.pub_country_code is None:
        return False
    return facts.pub_country_code.lower() in _DELAYED_URAA_COUNTRY_CODES


def was_registered(facts: Facts) -> bool:
    """Return :attr:`Facts.was_registered`."""
    return facts.was_registered


def was_renewed(facts: Facts) -> bool:
    """Return :attr:`Facts.was_renewed`."""
    return facts.was_renewed


def match_confidence_at_least(facts: Facts, threshold: float) -> bool:
    """Return ``True`` when the matcher's calibrated confidence meets ``threshold``."""
    return facts.match_confidence >= threshold


__all__ = [
    "country_delayed_uraa",
    "country_is_foreign",
    "country_is_us",
    "country_no_treaty",
    "in_pd_by_age",
    "match_confidence_at_least",
    "published_before",
    "published_between",
    "published_on_or_after",
    "was_registered",
    "was_renewed",
]
