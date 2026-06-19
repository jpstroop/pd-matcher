"""Unicode text normalization primitives shared across matcher components.

``normalize_text`` folds Latin ligatures to their ASCII expansions ("œuvre"
becomes "oeuvre"), does NFKD decomposition, strips combining marks (so "café"
and "cafe" collide), lower-cases the result, collapses any run of
non-alphanumeric characters to a single ASCII space, and trims leading and
trailing whitespace. ``tokenize`` builds on it by splitting the normalized
string on whitespace, guaranteeing that every emitted token is non-empty.
Both functions are pure and idempotent — a property the test suite asserts
via Hypothesis to prevent accidental regressions when the normalization
pipeline grows.

The ligature fold precedes NFKD because ``œ`` and ``æ`` have no canonical
decomposition; without the fold, NFKD leaves them intact and the
non-alphanumeric collapse strips them entirely, silently dropping a letter.
"""

from re import compile as re_compile
from unicodedata import category as unicode_category
from unicodedata import normalize as unicode_normalize

_NON_ALPHANUM_RUN = re_compile(r"[^0-9a-z]+")

_LIGATURE_FOLD = str.maketrans(
    {
        "œ": "oe",
        "Œ": "OE",
        "æ": "ae",
        "Æ": "AE",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬆ": "st",
    }
)


def normalize_text(s: str) -> str:
    """Return an NFKD-lowercased, diacritic-stripped, whitespace-collapsed form.

    Args:
        s: Arbitrary input string. Empty input returns empty output.

    Returns:
        A normalized form suitable for tokenization and equality testing.
    """
    if not s:
        return ""
    folded = s.translate(_LIGATURE_FOLD)
    decomposed = unicode_normalize("NFKD", folded)
    stripped = "".join(ch for ch in decomposed if unicode_category(ch) != "Mn")
    lowered = stripped.lower()
    collapsed = _NON_ALPHANUM_RUN.sub(" ", lowered)
    return collapsed.strip()


def tokenize(s: str) -> tuple[str, ...]:
    """Normalize ``s`` and split it on whitespace.

    Args:
        s: Arbitrary input string.

    Returns:
        A tuple of non-empty tokens; the empty tuple if no tokens survive.
    """
    normalized = normalize_text(s)
    if not normalized:
        return ()
    return tuple(normalized.split())


__all__ = [
    "normalize_text",
    "tokenize",
]
