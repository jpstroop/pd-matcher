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
    series_titles: tuple[str, ...] = (),
) -> MarcRecord:
    return MarcRecord(
        control_id="m",
        title=title,
        title_main=title_main,
        title_part_number=title_part_number,
        extent=extent,
        publication_date_raw=publication_date_raw,
        series_titles=series_titles,
    )


def _cce(
    *,
    title: str = "Some title",
    desc: str | None = None,
    notes: tuple[str, ...] = (),
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title=title,
        was_renewed=False,
        desc=desc,
        notes=notes,
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


# ---- Multilingual part-indicator coverage (#62) ------------------------------


def test_score_volume_french_t_abbreviation_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """French ``T. 1`` (tome) on the CCE title fires part classification."""
    marc = _marc(extent="5 v.")
    cce = _cce(title="T. 1: Histoire de la philosophie")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_french_tome_with_roman_numeral_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """French verbose ``tome I`` with Roman numeral fires part classification."""
    marc = _marc(extent="3 v.")
    cce = _cce(title="Tome I: Les origines")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_italian_tomo_agreement_on_both_sides(
    scorer_context: ScorerContext,
) -> None:
    """Italian/Spanish ``tomo 2`` on both sides agrees on same part number."""
    marc = _marc(title="Storia d'Italia, tomo 2")
    cce = _cce(title="Storia, tomo 2")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_part"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 100.0


def test_score_volume_german_bd_abbreviation_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """German ``Bd. 2`` (Band, volume) fires part classification."""
    marc = _marc(extent="5 v.")
    cce = _cce(title="Goethes Werke, Bd. 2")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_german_band_with_roman_numeral_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """German verbose ``Band III`` with Roman numeral fires part classification."""
    marc = _marc(extent="4 v.")
    cce = _cce(desc="Band III, 412 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_german_teil_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """German ``Teil 2`` (part) fires part classification on the CCE side."""
    marc = _marc(extent="3 v.")
    cce = _cce(title="Faust, Teil 2")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_german_tl_abbreviation_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """German ``Tl. 1`` (abbreviated Teil) fires part classification."""
    marc = _marc(extent="2 v.")
    cce = _cce(title="Werke, Tl. 1")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_german_heft_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """German ``Heft 4`` (fascicle) fires part classification."""
    marc = _marc(extent="6 v.")
    cce = _cce(title="Zeitschrift, Heft 4")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_dutch_dl_abbreviation_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """Dutch ``Dl. 1`` (abbreviated Deel) fires part classification."""
    marc = _marc(extent="3 v.")
    cce = _cce(title="Werken, Dl. 1")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_latin_lib_with_roman_numeral_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """Latin ``Lib. III`` (liber) fires part classification."""
    marc = _marc(extent="6 v.")
    cce = _cce(title="Historia naturalis, Lib. III")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_latin_pars_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """Latin ``Pars II`` (part) fires part classification."""
    marc = _marc(extent="4 v.")
    cce = _cce(title="Summa theologica, Pars II")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_french_livre_classifies_as_part(
    scorer_context: ScorerContext,
) -> None:
    """French ``livre 3`` fires part classification."""
    marc = _marc(extent="5 v.")
    cce = _cce(title="Discours, livre 3")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


# ---- False-positive guards: prefixes without numbers stay UNKNOWN ------------


def test_score_volume_author_initials_do_not_classify_as_part(
    scorer_context: ScorerContext,
) -> None:
    """``T. S. Eliot`` is initials, not a French tome marker — no part classification."""
    marc = _marc()
    cce = _cce(title="T. S. Eliot, Selected Poems")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_lm_montgomery_initials_do_not_classify_as_part(
    scorer_context: ScorerContext,
) -> None:
    """``L. M. Montgomery`` is initials, not a French livre marker."""
    marc = _marc()
    cce = _cce(title="L. M. Montgomery, Anne of Green Gables")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_h_g_wells_initials_do_not_classify_as_part(
    scorer_context: ScorerContext,
) -> None:
    """``H. G. Wells`` initials — bare ``H.`` is excluded from the prefix set."""
    marc = _marc()
    cce = _cce(title="H. G. Wells, The Time Machine")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_bd_publisher_name_without_number_does_not_classify_as_part(
    scorer_context: ScorerContext,
) -> None:
    """``Bd.`` in a publisher abbreviation context (no following number) stays UNKNOWN."""
    marc = _marc()
    cce = _cce(title="Werke (Bd. München Verlag)")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


# ---- #82: real whole/part positives ------------------------------------------


def test_score_volume_numbered_subdivision_in_cce_title_is_part(
    scorer_context: ScorerContext,
) -> None:
    """MARC ``"3 v"`` set vs CCE ``"...Vol.1: ..."`` numbered volume scores 0."""
    marc = _marc(
        title="Engineering compendium on radiation shielding",
        title_main="Engineering compendium on radiation shielding",
        extent="3 v",
    )
    cce = _cce(title="Engineering compendium on radiation shielding. Vol.1: Shielding fundamentals")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_volume_part_designator_in_cce_note_is_part(
    scorer_context: ScorerContext,
) -> None:
    """An open MARC set vs a CCE whose note carries ``"Pt.1"`` scores 0."""
    marc = _marc(
        title="Demand for rehabilitation in a labor union population",
        title_main="Demand for rehabilitation in a labor union population",
        extent="v",
    )
    cce = _cce(
        title="Demand for rehabilitation in a labor union population",
        desc="1156 p.",
        notes=("Pt.1: Research report",),
    )
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_closed_set_vs_single_volume_pages_is_zero(
    scorer_context: ScorerContext,
) -> None:
    """MARC ``"16 v"`` set vs a single-volume CCE page count scores 0 (#82 imp 2)."""
    marc = _marc(
        title="Check-list of birds of the world",
        title_main="Check-list of birds of the world",
        extent="16 v",
    )
    cce = _cce(title="Check-list of birds of the world", desc="xviii, 345 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_volume_open_set_vs_partial_range_is_part(
    scorer_context: ScorerContext,
) -> None:
    """Open MARC vs CCE ``"Vol.1-3"`` (range, but open ≠ covered) scores 0."""
    marc = _marc(
        title="The Arab-Israeli conflict",
        title_main="The Arab-Israeli conflict",
        extent="v",
    )
    cce = _cce(title="The Arab-Israeli conflict. Vol.1-3")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_descriptive_title_extension_is_part(
    scorer_context: ScorerContext,
) -> None:
    """Open MARC vs a CCE title that descriptively extends it scores 0 (#82 imp 1)."""
    marc = _marc(
        title="Guide to art museums in the United States",
        title_main="Guide to art museums in the United States",
        extent="v",
    )
    cce = _cce(title="Guide to art museums in the United States, east coast — Washington to Miami.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


# ---- #82: real whole/part negatives (must NOT be penalized) ------------------


def test_score_volume_covering_range_extension_is_whole(
    scorer_context: ScorerContext,
) -> None:
    """Closed ``"2 v"`` MARC vs CCE ``"Vol.1-2"`` (full coverage) is whole↔whole."""
    marc = _marc(
        title="A critical history of English literature",
        title_main="A critical history of English literature",
        extent="2 v",
    )
    cce = _cce(title="A critical history of English literature. Vol.1-2.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 0.0
    assert ev.score == 100.0


def test_score_volume_monographic_series_note_is_suppressed(
    scorer_context: ScorerContext,
) -> None:
    """A series-member note matching MARC ``series_titles`` is not a part (#82 imp 4)."""
    marc = _marc(
        title="Theorie der gewöhnlichen Differentialgleichungen",
        title_main="Theorie der gewöhnlichen Differentialgleichungen",
        extent="xi, 389 p",
        series_titles=("Die Grundlehren der mathematischen Wissenschaften, eine Reihe",),
    )
    cce = _cce(
        title="Theorie der gewöhnlichen Differential-gleichungen",
        desc="389 p.",
        notes=("Die Grundlehren der mathematischen Wissenschaften, Bd.66",),
    )
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_monographic_series_note_without_marc_series_is_part(
    scorer_context: ScorerContext,
) -> None:
    """The same note without a matching MARC series still reads as a part."""
    marc = _marc(extent="5 v.")
    cce = _cce(
        notes=("Die Grundlehren der mathematischen Wissenschaften, Bd.66",),
    )
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


# ---- #82: pagination must not be read as part designators --------------------


def test_score_volume_pagination_in_cce_note_is_not_a_part(
    scorer_context: ScorerContext,
) -> None:
    """CCE notes that are pure pagination do not classify as parts."""
    marc = _marc(extent="5 v.")
    cce = _cce(notes=("324 p., 1 l.", "viii, 682 p."))
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_roman_pagination_desc_is_not_a_part(
    scorer_context: ScorerContext,
) -> None:
    """``"viii, 682 p."`` in the CCE desc is front-matter, not ``v. iii``."""
    marc = _marc(extent="5 v.")
    cce = _cce(desc="viii, 682 p.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0


# ---- #82: title-extension guards ---------------------------------------------


def test_score_volume_title_extension_does_not_fire_for_single_volume_marc(
    scorer_context: ScorerContext,
) -> None:
    """A subtitle variation against a single-volume MARC is not whole/part."""
    marc = _marc(
        title="Guide to art museums in the United States",
        title_main="Guide to art museums in the United States",
        extent="312 p.",
    )
    cce = _cce(title="Guide to art museums in the United States, east coast — Washington to Miami.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 0.0
    assert feature_map["marc_is_whole_open"] == 0.0
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_title_extension_skips_empty_marc_main_title(
    scorer_context: ScorerContext,
) -> None:
    """An empty MARC ``title_main`` cannot be extended (covers the no-tokens branch)."""
    marc = _marc(title="", title_main="", extent="3 v")
    cce = _cce(title="Some unrelated registration with many words", desc=None)
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_whole_open_unrelated_cce_title_is_not_extended(
    scorer_context: ScorerContext,
) -> None:
    """A CCE title that does not contain every MARC token is not an extension."""
    marc = _marc(
        title="Guide to art museums",
        title_main="Guide to art museums",
        extent="v",
    )
    cce = _cce(title="Catalogue of paintings", desc=None)
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_range_extension_against_countless_whole_marc_is_part(
    scorer_context: ScorerContext,
) -> None:
    """A covering-shaped range needs a numeric MARC count; a range-extent MARC is not.

    The MARC is a whole via its own ``"v. 1-3"`` range (no ``N v.`` count), so
    ``_range_covers_marc`` cannot confirm coverage and the CCE stays a part.
    """
    marc = _marc(
        title="A critical history of English literature",
        title_main="A critical history of English literature",
        extent="v. 1-3",
    )
    cce = _cce(title="A critical history of English literature. Vol.1-3.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_collected_title_marc_no_extent_range_extension_is_part(
    scorer_context: ScorerContext,
) -> None:
    """A collected-title MARC (no extent) cannot confirm range coverage → part.

    The MARC is whole via its collected ``title`` but carries no extent, so
    ``_range_covers_marc`` cannot confirm coverage of the CCE range and the
    CCE remains a part. Exercises the empty-extent guard.
    """
    marc = _marc(
        title="Collected poems",
        title_main="Poems",
        extent=None,
    )
    cce = _cce(title="Poems. Vol.1-2")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_note_part_with_nonmatching_series_is_part(
    scorer_context: ScorerContext,
) -> None:
    """A note part whose name does not overlap the MARC series is not suppressed."""
    marc = _marc(extent="5 v.", series_titles=("Loeb Classical Library",))
    cce = _cce(notes=("Goethes Werke, Bd.2",))
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_closed_whole_vs_unknown_no_pagecount_is_skipped(
    scorer_context: ScorerContext,
) -> None:
    """A closed multi-volume MARC vs an unparseable CCE skips (no signal)."""
    marc = _marc(extent="4 v.")
    cce = _cce(desc="unpaged")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True
