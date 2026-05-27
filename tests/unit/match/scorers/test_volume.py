"""Tests for :mod:`pd_matcher.match.scorers.volume`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.volume import score_volume
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


def _marc(
    *,
    title: str = "Some title",
    title_main: str = "Some title",
    title_part_number: str | None = None,
    extent: str | None = None,
    publication_date_raw: str | None = None,
) -> MarcRecord:
    return MarcRecord(
        control_id="m",
        title=title,
        title_main=title_main,
        title_part_number=title_part_number,
        extent=extent,
        publication_date_raw=publication_date_raw,
    )


def _cce(
    *,
    title: str = "Some title",
    desc: str | None = None,
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title=title,
        was_renewed=False,
        desc=desc,
    )


def test_score_volume_skipped_when_both_unknown(scorer_context: ScorerContext) -> None:
    """Records with no volume cues yield a skipped Evidence."""
    ev = score_volume(_marc(), _cce(), scorer_context)
    assert ev.skipped is True


def test_score_volume_whole_whole_agreement(scorer_context: ScorerContext) -> None:
    """Two multi-volume statements agree at max score."""
    marc = _marc(extent="5 v.")
    cce = _cce(desc="3 v.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is False
    assert ev.score == 100.0


def test_score_volume_whole_via_volume_range(scorer_context: ScorerContext) -> None:
    """``"v. 1-3"`` classifies as whole on the MARC side."""
    marc = _marc(extent="v. 1-3")
    cce = _cce(desc="2 v.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 100.0


def test_score_volume_whole_via_multivolume_in_one(scorer_context: ScorerContext) -> None:
    """``"3 v. in 1"`` classifies as whole on the MARC side."""
    marc = _marc(extent="3 v. in 1")
    cce = _cce(desc="2 v.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 100.0


def test_score_volume_whole_via_collected_title(scorer_context: ScorerContext) -> None:
    """``"Collected works"`` in the title classifies as whole."""
    marc = _marc(title="Collected works of Smith")
    cce = _cce(desc="5 v.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 100.0


def test_score_volume_part_part_same_number_is_max(scorer_context: ScorerContext) -> None:
    """Both sides marked as ``v. 2`` agree at max score."""
    marc = _marc(title_part_number="2")
    cce = _cce(desc="v. 2")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 100.0


def test_score_volume_part_part_different_numbers_is_partial(
    scorer_context: ScorerContext,
) -> None:
    """``Vol. 1`` vs ``Vol. 2`` scores 25.0 (both part, different number)."""
    marc = _marc(title="Vol. 1 of the History", title_main="History")
    cce = _cce(desc="v. 2")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 25.0


def test_score_volume_whole_vs_part_is_zero(scorer_context: ScorerContext) -> None:
    """A multi-volume MARC vs. a single-volume CCE registration scores 0.0."""
    marc = _marc(extent="5 v.")
    cce = _cce(desc="v. 1")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_volume_part_vs_whole_is_zero(scorer_context: ScorerContext) -> None:
    """The whole↔part penalty is symmetric."""
    marc = _marc(title_part_number="1")
    cce = _cce(desc="3 v.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 0.0


def test_score_volume_part_via_marc_title_pt(scorer_context: ScorerContext) -> None:
    """``Pt. I`` (Roman numeral) in the MARC title classifies as part."""
    marc = _marc(title="Pt. II : the empire")
    cce = _cce(desc="pt. 2")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is False
    # marc_part = "ii", cce_part = "2" => different, so 25.0.
    assert ev.score == 25.0


def test_score_volume_part_via_book_word(scorer_context: ScorerContext) -> None:
    """``Book one`` classifies as part with canonical number ``"1"``."""
    marc = _marc(title="Book One of the chronicles")
    cce = _cce(desc="book 1")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 100.0


def test_score_volume_marc_unknown_cce_part_is_skipped(scorer_context: ScorerContext) -> None:
    """Skipped when MARC side has no cardinality cue."""
    marc = _marc()
    cce = _cce(desc="v. 1")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is True


def test_score_volume_marc_part_cce_unknown_is_skipped(scorer_context: ScorerContext) -> None:
    """Skipped when CCE side has no cardinality cue."""
    marc = _marc(title_part_number="1")
    cce = _cce()
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is True


def test_score_volume_single_volume_extent_does_not_flag_whole(
    scorer_context: ScorerContext,
) -> None:
    """``"1 v."`` is just one volume — not a multi-volume whole."""
    marc = _marc(extent="1 v.")
    cce = _cce(desc="1 v.")
    ev = score_volume(marc, cce, scorer_context)
    # Neither side flags whole or part, so the Evidence is skipped.
    assert ev.skipped is True


def test_score_volume_features_record_classification(scorer_context: ScorerContext) -> None:
    """Per-side classification is exposed via the features tuple."""
    marc = _marc(extent="5 v.")
    cce = _cce(desc="v. 1")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["marc_is_part"] == 0.0
    assert feature_map["cce_is_whole"] == 0.0
    assert feature_map["cce_is_part"] == 1.0


def test_score_volume_collected_title_on_cce_side(scorer_context: ScorerContext) -> None:
    """``"Selected writings"`` on the CCE title classifies as whole."""
    marc = _marc(extent="3 v.")
    cce = _cce(title="Selected writings of Smith")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 100.0


def test_score_volume_part_number_in_marc_extent(scorer_context: ScorerContext) -> None:
    """A part marker in the MARC extent (not just title_part_number) classifies."""
    marc = _marc(extent="vol. 3, 312 p.")
    cce = _cce(desc="v. 3")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.score == 100.0


def test_score_volume_empty_marc_title_does_not_crash(scorer_context: ScorerContext) -> None:
    """An empty MARC title still skips cleanly when no other cues fire."""
    marc = _marc(title="", title_main="")
    cce = _cce()
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is True


def test_score_volume_empty_cce_title_does_not_crash(scorer_context: ScorerContext) -> None:
    """An empty CCE title still skips cleanly when no other cues fire."""
    marc = _marc()
    cce = _cce(title="")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is True


def test_score_volume_marc_bare_v_extent_is_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """AACR2 bare ``"v"`` extent classifies as whole_open."""
    marc = _marc(extent="v")
    cce = _cce(desc="551 p.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is False
    assert ev.score == 0.0
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0
    assert feature_map["marc_is_whole"] == 0.0


def test_score_volume_marc_volumes_extent_is_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """RDA bare ``"volumes"`` extent classifies as whole_open."""
    marc = _marc(extent="volumes")
    cce = _cce(desc="312 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0
    assert ev.score == 0.0


def test_score_volume_marc_v_part_one_is_still_part(scorer_context: ScorerContext) -> None:
    """``"v. 1"`` is a part marker, not the bare-v whole_open sentinel."""
    marc = _marc(extent="v. 1")
    cce = _cce(desc="v. 1")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 0.0
    assert feature_map["marc_is_part"] == 1.0
    assert ev.score == 100.0


def test_score_volume_marc_three_volumes_is_closed_whole(
    scorer_context: ScorerContext,
) -> None:
    """``"3 volumes"`` is a closed multi-volume count, not whole_open."""
    marc = _marc(extent="3 volumes")
    cce = _cce(desc="5 v.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 0.0
    assert feature_map["marc_is_whole"] == 1.0
    assert ev.score == 100.0


def test_score_volume_marc_open_publication_date_is_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """``"[1945-]"`` open-date convention classifies as whole_open."""
    marc = _marc(publication_date_raw="[1945-]")
    cce = _cce(desc="551 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0
    assert ev.score == 0.0


def test_score_volume_marc_open_date_with_trailing_space_is_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """``"[1945- ]"`` (with trailing space) still trips the open-date rule."""
    marc = _marc(publication_date_raw="[1945- ]")
    cce = _cce(desc="312 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0


def test_score_volume_marc_truncated_open_date_is_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """``"[1945-"`` (truncated bracket) is still treated as open."""
    marc = _marc(publication_date_raw="[1945-")
    cce = _cce(desc="312 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0


def test_score_volume_marc_closed_bracketed_date_is_not_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """``"[1945-1950]"`` is a closed range, not the open-date sentinel."""
    marc = _marc(publication_date_raw="[1945-1950]")
    cce = _cce(desc="312 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 0.0


def test_score_volume_marc_plain_year_is_not_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """A plain year ``"1945"`` is not an open-date string."""
    marc = _marc(publication_date_raw="1945")
    cce = _cce(desc="312 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 0.0


def test_score_volume_whole_open_vs_part_is_zero(scorer_context: ScorerContext) -> None:
    """Series-level MARC vs explicit CCE part registration scores 0.0."""
    marc = _marc(extent="v")
    cce = _cce(desc="v. 1")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_volume_whole_open_vs_unknown_with_page_count_is_zero(
    scorer_context: ScorerContext,
) -> None:
    """Series-level MARC vs single-volume CCE (parseable pages) scores 0.0."""
    marc = _marc(extent="v")
    cce = _cce(desc="551 p.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_volume_whole_open_vs_unknown_without_page_count_is_skipped(
    scorer_context: ScorerContext,
) -> None:
    """Series-level MARC with no CCE page-count signal yields skipped."""
    marc = _marc(extent="v")
    cce = _cce(desc=None)
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is True


def test_score_volume_whole_open_vs_whole_is_max(scorer_context: ScorerContext) -> None:
    """Open multipart MARC vs closed multi-volume CCE is compatible."""
    marc = _marc(extent="v")
    cce = _cce(desc="5 v.")
    ev = score_volume(marc, cce, scorer_context)
    assert ev.skipped is False
    assert ev.score == 100.0
