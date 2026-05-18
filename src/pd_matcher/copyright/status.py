"""Enumeration of every leaf of the Cornell public-domain decision matrix.

Each member corresponds to one terminal cell of the matrix at
``https://guides.library.cornell.edu/copyright/publicdomain``. Values are
plain strings (``StrEnum``) so that the assessment serializes naturally to
JSON, CSV, and YAML without bespoke converters.

This project's scope is **published books only**, so only the two
book-relevant Cornell categories are represented:

* Category 2 — works first published in the United States (including the
  US-government-work carve-out).
* Category 3 — works first published outside the United States.

Category 1 (unpublished works), Category 4 (sound recordings), and
Category 5 (architectural works) are intentionally absent.
"""

from enum import StrEnum


class CopyrightStatus(StrEnum):
    """Every terminal cell of the Cornell public-domain matrix in scope."""

    PD_BY_AGE_PRE_95_YEARS = "PD_BY_AGE_PRE_95_YEARS"
    """US-published work whose publication year is more than 95 years
    before today; PD by age alone via the moving wall."""

    PD_US_PUB_NO_NOTICE_1931_1977 = "PD_US_PUB_NO_NOTICE_1931_1977"
    """Category 2: US-published 1931-1977 without copyright notice."""

    PD_US_PUB_REGISTERED_NOT_RENEWED = "PD_US_PUB_REGISTERED_NOT_RENEWED"
    """Category 2: US-published 1931-1963 with notice but the copyright
    was not renewed during its 28th year."""

    PD_US_PUB_NO_REGISTRATION_1978_1989 = "PD_US_PUB_NO_REGISTRATION_1978_1989"
    """Category 2: US-published 1978-1 Mar 1989 without notice and
    without registration within five years."""

    PD_US_GOVERNMENT_WORK = "PD_US_GOVERNMENT_WORK"
    """Category 2: created by a federal officer or employee in their
    official capacity; PD in the US regardless of date."""

    PD_FOREIGN_IN_HOME_COUNTRY_PD_1996 = "PD_FOREIGN_IN_HOME_COUNTRY_PD_1996"
    """Category 3: first published outside the US 1931-1977 and in the
    public domain in its source country on 1 Jan 1996 (URAA baseline)."""

    PD_FOREIGN_NO_TREATY_COUNTRY = "PD_FOREIGN_NO_TREATY_COUNTRY"
    """Category 3: first published in a country with no US copyright
    relations (Eritrea, Ethiopia, Iran, Iraq, Marshall Islands, etc.)."""

    IN_COPYRIGHT_US_PUB_REGISTERED_AND_RENEWED = "IN_COPYRIGHT_US_PUB_REGISTERED_AND_RENEWED"
    """Category 2: US-published 1931-1963 with notice and timely
    renewal; copyright runs 95 years from publication."""

    IN_COPYRIGHT_US_PUB_1964_1977_WITH_NOTICE = "IN_COPYRIGHT_US_PUB_1964_1977_WITH_NOTICE"
    """Category 2: US-published 1964-1977 with notice; automatic
    renewal; copyright runs 95 years from publication."""

    IN_COPYRIGHT_US_PUB_1978_1989_CURED = "IN_COPYRIGHT_US_PUB_1978_1989_CURED"
    """Category 2: US-published 1978-1 Mar 1989 without notice but
    registered within five years (defect cured)."""

    IN_COPYRIGHT_US_PUB_POST_1989 = "IN_COPYRIGHT_US_PUB_POST_1989"
    """Category 2: US-published on or after 1 Mar 1989; notice is no
    longer required; 70 years p.m.a. or corporate terms."""

    IN_COPYRIGHT_PRE_1978_PUBLISHED_1978_2002_FLOOR = (
        "IN_COPYRIGHT_PRE_1978_PUBLISHED_1978_2002_FLOOR"
    )
    """Category 2/3: created pre-1978, first published 1978-2002;
    copyright runs at least until 31 Dec 2047."""

    IN_COPYRIGHT_FOREIGN_URAA_RESTORED = "IN_COPYRIGHT_FOREIGN_URAA_RESTORED"
    """Category 3: foreign work whose US copyright was restored by the
    Uruguay Round Agreements Act on 1 Jan 1996."""

    IN_COPYRIGHT_FOREIGN_POST_1989 = "IN_COPYRIGHT_FOREIGN_POST_1989"
    """Category 3: foreign work first published on or after 1 Mar 1989
    in a country with US copyright relations."""

    UNKNOWN_INSUFFICIENT_DATA = "UNKNOWN_INSUFFICIENT_DATA"
    """The facts available do not let the engine reach a determination
    (e.g. delayed-URAA country, missing publication year)."""

    UNKNOWN_NO_RULE_MATCHED = "UNKNOWN_NO_RULE_MATCHED"
    """No rule in the active ruleset matched the observed facts."""


__all__ = [
    "CopyrightStatus",
]
