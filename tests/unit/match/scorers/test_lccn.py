"""Tests for :mod:`pd_matcher.match.scorers.lccn`."""

from hypothesis import given
from hypothesis import strategies as st

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.lccn import _canonical
from pd_matcher.match.scorers.lccn import score_lccn
from pd_matcher.models import IndexedNyplRegRecord


def _record(lccn: str | None, *, regnum: str | None = None) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title="t",
        was_renewed=False,
        regnum=regnum,
        lccn=lccn,
    )


def test_score_lccn_exact_match_is_decisive(scorer_context: ScorerContext) -> None:
    """Equal canonical IDs produce a decisive Evidence at max score."""
    ev = score_lccn("37013688", _record("37013688"), scorer_context)
    assert ev.decisive is True
    assert ev.score == ev.max
    assert ev.skipped is False


def test_score_lccn_normalises_hyphenated_cce_form(scorer_context: ScorerContext) -> None:
    """The CCE hyphenated display form canonicalises to the 8-digit MARC form."""
    ev = score_lccn("37013688", _record("37-13688"), scorer_context)
    assert ev.decisive is True
    assert ev.score == ev.max


def test_score_lccn_strips_surrounding_whitespace(scorer_context: ScorerContext) -> None:
    """Whitespace around the value is removed before comparison."""
    ev = score_lccn(" 37-13688 ", _record("37013688"), scorer_context)
    assert ev.decisive is True


def test_score_lccn_drops_suffix_after_slash(scorer_context: ScorerContext) -> None:
    """A forward slash and everything after is discarded."""
    ev = score_lccn("75425165", _record("75-425165/M/r842"), scorer_context)
    assert ev.decisive is True


def test_score_lccn_handles_alphabetic_prefix(scorer_context: ScorerContext) -> None:
    """LCCNs with alphabetic prefixes round-trip unchanged when no hyphen."""
    ev = score_lccn("n79018774", _record("n79018774"), scorer_context)
    assert ev.decisive is True


def test_score_lccn_pads_short_suffix_under_alphabetic_prefix(
    scorer_context: ScorerContext,
) -> None:
    """An alphabetic-prefix LCCN with a short numeric suffix pads to 6 digits."""
    ev = score_lccn("n81000021", _record("n81-021"), scorer_context)
    assert ev.decisive is True


def test_score_lccn_unequal_ids_skipped(scorer_context: ScorerContext) -> None:
    """Unequal canonical IDs deliberately skip rather than emit a low score."""
    ev = score_lccn("37013688", _record("47039196"), scorer_context)
    assert ev.skipped is True
    assert ev.decisive is False


def test_score_lccn_ignores_regnum_field(scorer_context: ScorerContext) -> None:
    """A populated regnum that happens to look like an LCCN must not fire."""
    ev = score_lccn("A107434", _record(None, regnum="A107434"), scorer_context)
    assert ev.skipped is True
    assert ev.decisive is False


def test_score_lccn_skipped_when_marc_lccn_missing(scorer_context: ScorerContext) -> None:
    """A None MARC LCCN triggers the skipped path."""
    ev = score_lccn(None, _record("37013688"), scorer_context)
    assert ev.skipped is True


def test_score_lccn_skipped_when_marc_lccn_blank(scorer_context: ScorerContext) -> None:
    """A blank MARC LCCN canonicalises to None and triggers skipped."""
    ev = score_lccn("   ", _record("37013688"), scorer_context)
    assert ev.skipped is True


def test_score_lccn_skipped_when_nypl_lccn_missing(scorer_context: ScorerContext) -> None:
    """A None NYPL LCCN triggers the skipped path."""
    ev = score_lccn("37013688", _record(None), scorer_context)
    assert ev.skipped is True


def test_score_lccn_skipped_when_only_slash_present(scorer_context: ScorerContext) -> None:
    """Input that becomes empty after slash truncation yields skipped."""
    ev = score_lccn("/abc", _record("37013688"), scorer_context)
    assert ev.skipped is True


def test_canonical_returns_none_for_none() -> None:
    """``_canonical(None)`` returns ``None`` without raising."""
    assert _canonical(None) is None


def test_canonical_preserves_long_right_substring() -> None:
    """A right-of-hyphen substring longer than 6 digits is kept verbatim."""
    assert _canonical("12-1234567") == "121234567"


@given(value=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=40))
def test_canonical_is_idempotent(value: str) -> None:
    """``_canonical`` is idempotent: applying it twice equals applying it once."""
    once = _canonical(value)
    twice = _canonical(once)
    assert twice == once
