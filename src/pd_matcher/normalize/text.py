"""Unicode text normalization primitives shared across matcher components.

``normalize_text`` does NFKD decomposition, strips combining marks (so "café"
and "cafe" collide), lower-cases the result, collapses any run of
non-alphanumeric characters to a single ASCII space, and trims leading and
trailing whitespace. ``tokenize`` builds on it by splitting the normalized
string on whitespace, guaranteeing that every emitted token is non-empty.
Both functions are pure and idempotent — a property the test suite asserts
via Hypothesis to prevent accidental regressions when the normalization
pipeline grows.
"""

from re import compile as re_compile
from unicodedata import category as unicode_category
from unicodedata import normalize as unicode_normalize

_NON_ALPHANUM_RUN = re_compile(r"[^0-9a-z]+")


def normalize_text(s: str) -> str:
    """Return an NFKD-lowercased, diacritic-stripped, whitespace-collapsed form.

    Args:
        s: Arbitrary input string. Empty input returns empty output.

    Returns:
        A normalized form suitable for tokenization and equality testing.
    """
    if not s:
        return ""
    decomposed = unicode_normalize("NFKD", s)
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
