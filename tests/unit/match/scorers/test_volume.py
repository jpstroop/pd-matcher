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


# ---- Round-2 detection-quality acceptance cases (#82 / #84) ------------------


def test_score_volume_pair_339_punctuated_bare_v_detects_whole_open(
    scorer_context: ScorerContext,
) -> None:
    """Pair 339: MARC extent ``". v"`` (OCR-noised bare ``v``) + CCE ``Pt. 1``.

    marc 996578393506421 / cce A174027. The detection gap was that the
    punctuated bare-volume extent never classified as a whole; it must now
    read whole_open against a single-part CCE for a 0.0 mismatch.
    """
    marc = _marc(
        title="Religious and secular leadership",
        title_main="Religious and secular leadership",
        extent=". v",
    )
    cce = _cce(title="Religious and secular leadership. Pt. 1")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole_open"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_volume_pair_377_part_in_notes_with_multivolume_desc_is_incompatible(
    scorer_context: ScorerContext,
) -> None:
    """Pair 377: MARC ``5 v. in 10`` whole + CCE desc ``2 v.`` + notes ``Vol.1``.

    marc 996173823506421 / cce A310797. The CCE part designator lives in
    the notes, and the CCE desc looks multi-volume (``2 v.``); containment
    of that single part inside the MARC whole is INCOMPATIBLE, not a set
    match. The old scorer scored this 1.0 (wrong direction); it must be 0.0.
    """
    marc = _marc(
        title="The notebooks of Samuel Taylor Coleridge",
        title_main="The notebooks of Samuel Taylor Coleridge",
        extent="5 v. in 10",
    )
    cce = _cce(title="Notebooks.", desc="2 v.", notes=("Vol.1: 1794-1804, text and notes.",))
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_volume_pair_362_roman_numeral_designator_in_title(
    scorer_context: ScorerContext,
) -> None:
    """Pair 362: MARC ``2 v`` whole + CCE bare Roman-numeral ``I:`` subtitle.

    marc 996391933506421 / cce A190318. The ``I:`` is not preceded by a
    Vol/Pt prefix; the bare-designator detector must still flag it as part.
    """
    marc = _marc(title="Kontakia of Romanos Byzantine melodist", extent="2 v")
    cce = _cce(title="Kontakia of Romanos, Byzantine melodist, I: On the person of Christ")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_pair_395_mid_title_volume_designator(
    scorer_context: ScorerContext,
) -> None:
    """Pair 395: MARC ``15 v`` whole + CCE mid-title ``v. 1``.

    marc 9917915453506421 / cce A19508. The ``v. 1`` sits mid-title
    (``...Othmer. v. 1. A to Anthrimides``); detection must scan the whole
    title, not just a trailing designator.
    """
    marc = _marc(title="Encyclopedia of chemical technology", extent="15 v")
    cce = _cce(
        title=(
            "...chemical technology, ed. by Raymond E. Kirk and "
            "Donald F. Othmer. v. 1. A to Anthrimides."
        ),
        desc="xxiv, 982 p.",
    )
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_covering_range_is_not_flagged_incompatible(
    scorer_context: ScorerContext,
) -> None:
    """The spare case: a CCE ``Vol. 1-2`` RANGE over a ``2 v.`` MARC is the whole set.

    A covering designator range registers the entire set, not a single
    part, so it must NOT be penalised — mis-flagging this is what caused
    the original top-1 regression.
    """
    marc = _marc(title="A critical history", extent="2 v.")
    cce = _cce(title="A critical history. Vol. 1-2.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_whole"] == 1.0
    assert feature_map["cce_is_part"] == 0.0
    assert ev.score == 100.0


def test_score_volume_t_range_in_notes_is_not_part(scorer_context: ScorerContext) -> None:
    """A non-volume-prefixed range ``T.1-3`` is a whole/set, not a single part."""
    marc = _marc(extent="3 v.")
    cce = _cce(title="Histoire, T.1-3.")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_whole"] == 1.0
    assert ev.score == 100.0


def test_score_volume_part_in_notes_classifies_as_part(scorer_context: ScorerContext) -> None:
    """A single ``Pt. 2`` designator in the CCE notes classifies the CCE as a part."""
    marc = _marc(extent="4 v.")
    cce = _cce(notes=("Pt. 2 of the collected edition.",))
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_monographic_series_note_is_suppressed(
    scorer_context: ScorerContext,
) -> None:
    """A ``<series>, Bd.N`` note matching a MARC series title is not a whole/part part."""
    marc = _marc(extent="6 v.", series_titles=("Grundlehren der mathematischen Wissenschaften",))
    cce = _cce(notes=("Grundlehren der mathematischen Wissenschaften, Bd. 66",))
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0


def test_score_volume_bare_designator_ignores_subtitle_colon(
    scorer_context: ScorerContext,
) -> None:
    """A plain ``Title: subtitle`` colon is not a bare volume designator."""
    marc = _marc(extent="5 v.")
    cce = _cce(title="A history: of the modern world")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0


def test_score_volume_title_extension_against_whole_open_is_part(
    scorer_context: ScorerContext,
) -> None:
    """A CCE title strictly extending a whole_open MARC title reads as a part."""
    marc = _marc(title="Guide to art museums", title_main="Guide to art museums", extent="v")
    cce = _cce(title="Guide to art museums in the United States east coast")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0


def test_score_volume_covering_range_via_title_extension_stays_whole(
    scorer_context: ScorerContext,
) -> None:
    """A title-extending CCE whose extension is a covering range stays whole."""
    marc = _marc(title="A critical history", title_main="A critical history", extent="2 v.")
    cce = _cce(title="A critical history of modern art Vol. 1-2")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_whole"] == 1.0
    assert ev.score == 100.0


def test_score_volume_leading_noise_extent_with_count_is_whole(
    scorer_context: ScorerContext,
) -> None:
    """A leading-noise multi-volume extent (``". 5 v"``) still classifies as whole."""
    marc = _marc(extent=". 5 v")
    cce = _cce(desc="v. 1")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_is_whole"] == 1.0
    assert ev.score == 0.0


def test_score_volume_title_extension_skipped_when_marc_title_empty(
    scorer_context: ScorerContext,
) -> None:
    """A whole MARC with an empty title_main cannot drive the title-extension path."""
    marc = _marc(title="", title_main="", extent="3 v.")
    cce = _cce(title="Some extended title with content")
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 0.0
    assert ev.skipped is True


def test_score_volume_notes_skip_non_designator_note_then_match(
    scorer_context: ScorerContext,
) -> None:
    """A leading note with no designator is skipped; a later ``Pt. 1`` note fires."""
    marc = _marc(extent="3 v.")
    cce = _cce(notes=("A general descriptive note.", "Pt. 1 of the set."))
    ev = score_volume(marc, cce, scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["cce_is_part"] == 1.0
    assert ev.score == 0.0
