"""Detect a script mismatch between two pre-normalization title strings.

When the MARC title and the CCE title use different dominant Unicode
scripts (e.g. Latin vs. Hebrew), token-set similarity cannot recover the
pair: the character sets are disjoint and any agreement on a stray
Latin sub-token is coincidental noise. The title scorer consults this
predicate before tokenization so it can emit a zero (counted in the
denominator) instead of pretending the pair is incomparable.

The check returns ``False`` whenever either side has no detectable
script — empty strings, digit-only strings, symbol-only strings — so
records that genuinely lack the cue do not get penalized.
"""

from pd_matcher.normalize.script import dominant_script


def is_script_mismatch(marc_text: str, cce_text: str) -> bool:
    """Return ``True`` when ``marc_text`` and ``cce_text`` use different scripts.

    Args:
        marc_text: The MARC-side text to inspect.
        cce_text: The CCE-side text to inspect.

    Returns:
        ``True`` when both sides have a detectable dominant script and
        the two scripts differ; ``False`` when either side has no
        detectable script or when both sides share the same script.
    """
    marc_script = dominant_script(marc_text)
    cce_script = dominant_script(cce_text)
    if marc_script is None or cce_script is None:
        return False
    return marc_script != cce_script


__all__ = [
    "is_script_mismatch",
]
