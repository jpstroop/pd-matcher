"""Unit tests for the pure review view model."""

from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta

from msgspec.json import encode as json_encode

from pd_groundtruth.review.view import CLAIMANT_LABEL
from pd_groundtruth.review.view import RENEWAL_NOT_RENEWED
from pd_groundtruth.review.view import RENEWAL_RENEWED
from pd_groundtruth.review.view import RENEWAL_UNKNOWN
from pd_groundtruth.review.view import _hathitrust_url
from pd_groundtruth.review.view import _publication_date_raw_if_distinct
from pd_groundtruth.review.view import author_is_claimant_label
from pd_groundtruth.review.view import build_card
from pd_groundtruth.review.view import build_labeled_row
from pd_groundtruth.review.view import parse_evidence
from pd_groundtruth.review.view import parse_evidence_sources
from pd_groundtruth.review.view import render_renewal_label
from pd_groundtruth.review_db import CurrentLabelRow
from pd_groundtruth.review_db import LabeledPairRow
from pd_groundtruth.review_db import ReviewPairRow
from pd_matcher.models import MarcRecord


def _marc(
    *,
    title: str = "The Full Title : a subtitle",
    title_main: str = "The Full Title",
    extent: str | None = None,
    publication_place: str | None = None,
    publication_date_raw: str | None = None,
    publication_year: int | None = 1953,
    isbns: tuple[str, ...] = (),
    oclc: str | None = None,
    lccn: str | None = "53001234",
    title_part_number: str | None = None,
    title_part_name: str | None = None,
) -> MarcRecord:
    return MarcRecord(
        control_id="ctrl-1",
        title=title,
        title_main=title_main,
        lccn=lccn,
        oclc=oclc,
        isbns=isbns,
        title_part_number=title_part_number,
        title_part_name=title_part_name,
        main_author="Doe, Jane",
        added_authors=("Roe, Richard",),
        statement_of_responsibility="by Jane Doe",
        edition="2nd ed.",
        publication_place=publication_place,
        publisher="Acme Press",
        publication_date_raw=publication_date_raw,
        publication_year=publication_year,
        extent=extent,
        series_titles=("Acme Studies",),
        language_code="eng",
        country_code="nyu",
    )


def _row(
    marc: MarcRecord,
    *,
    evidence_json: str,
    was_renewed: int | None = 1,
    cce_edition: str | None = None,
    cce_publication_places: str | None = None,
    cce_author_place: str | None = None,
    cce_author_is_claimant: int | None = None,
    cce_copies: str | None = None,
    cce_aff_date: str | None = None,
    cce_desc: str | None = None,
    cce_notes: str | None = None,
    cce_new_matter_claimed: str | None = None,
    cce_copy_date: str | None = None,
    cce_notice_date: str | None = None,
    cce_lccn: str | None = None,
    cce_prev_regnums: str | None = None,
    cce_renewal_id: str | None = None,
    cce_renewal_oreg: str | None = None,
    cce_renewal_rdat: str | None = None,
    cce_renewal_author: str | None = None,
    cce_renewal_title: str | None = None,
    cce_renewal_claimants: str | None = None,
    cce_renewal_new_matter: str | None = None,
    cce_claimants: str | None = "Jane Doe",
    evidence_sources_json: str = "{}",
) -> ReviewPairRow:
    return ReviewPairRow(
        id=7,
        language="eng",
        decade=1950,
        score=0.91,
        band="ge90",
        source="banded",
        marc_control_id=marc.control_id,
        marc_json=json_encode(marc).decode("utf-8"),
        marc_title=marc.title,
        marc_author=marc.main_author,
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        nypl_uuid="uuid-9",
        cce_title="The Full Title",
        cce_author="Jane Doe",
        cce_publishers="Acme Press | Beta Co",
        cce_claimants=cce_claimants,
        cce_reg_year=1953,
        cce_was_renewed=was_renewed,
        cce_regnum="R99",
        evidence_json=evidence_json,
        created_at="2026-05-22T00:00:00+00:00",
        cce_edition=cce_edition,
        cce_publication_places=cce_publication_places,
        cce_author_place=cce_author_place,
        cce_author_is_claimant=cce_author_is_claimant,
        cce_copies=cce_copies,
        cce_aff_date=cce_aff_date,
        cce_desc=cce_desc,
        cce_notes=cce_notes,
        cce_new_matter_claimed=cce_new_matter_claimed,
        cce_copy_date=cce_copy_date,
        cce_notice_date=cce_notice_date,
        cce_lccn=cce_lccn,
        cce_prev_regnums=cce_prev_regnums,
        cce_renewal_id=cce_renewal_id,
        cce_renewal_oreg=cce_renewal_oreg,
        cce_renewal_rdat=cce_renewal_rdat,
        cce_renewal_author=cce_renewal_author,
        cce_renewal_title=cce_renewal_title,
        cce_renewal_claimants=cce_renewal_claimants,
        cce_renewal_new_matter=cce_renewal_new_matter,
        evidence_sources_json=evidence_sources_json,
    )


def test_render_renewal_label_maps_flag() -> None:
    assert render_renewal_label(1) == RENEWAL_RENEWED
    assert render_renewal_label(0) == RENEWAL_NOT_RENEWED
    assert render_renewal_label(None) == RENEWAL_UNKNOWN


def test_parse_evidence_preserves_insertion_order() -> None:
    bars = parse_evidence('{"title.token_set": 1.0, "name.author": 0.5, "year.delta": 0.0}')
    assert [bar.scorer for bar in bars] == ["title.token_set", "name.author", "year.delta"]
    assert [bar.normalized for bar in bars] == [1.0, 0.5, 0.0]


def test_parse_evidence_handles_empty_object() -> None:
    assert parse_evidence("{}") == ()


def test_parse_evidence_sources_decodes_map() -> None:
    assert parse_evidence_sources(
        '{"title.token_set": "title_main ↔ title", "name.publisher": "publisher ↔ author_name"}'
    ) == {
        "title.token_set": "title_main ↔ title",
        "name.publisher": "publisher ↔ author_name",
    }


def test_parse_evidence_sources_returns_empty_for_empty_object() -> None:
    assert parse_evidence_sources("{}") == {}


def test_build_card_evidence_bar_source_set_when_present() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json='{"title.token_set": 0.9, "name.publisher": 0.29}',
            evidence_sources_json=(
                '{"title.token_set": "title_main ↔ title", '
                '"name.publisher": "publisher ↔ author_name"}'
            ),
        )
    )
    bars = {bar.scorer: bar for bar in card.evidence}
    assert bars["title.token_set"].source == "title_main ↔ title"
    assert bars["name.publisher"].source == "publisher ↔ author_name"


def test_build_card_evidence_bar_source_none_when_sources_empty() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json='{"title.token_set": 0.9, "name.publisher": 0.29}',
            evidence_sources_json="{}",
        )
    )
    for bar in card.evidence:
        assert bar.source is None


def test_build_card_evidence_bar_source_none_for_scorer_missing_from_map() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json='{"title.token_set": 0.9, "lccn.exact": 1.0}',
            evidence_sources_json='{"title.token_set": "title_main ↔ title"}',
        )
    )
    bars = {bar.scorer: bar for bar in card.evidence}
    assert bars["title.token_set"].source == "title_main ↔ title"
    assert bars["lccn.exact"].source is None


def test_build_card_exposes_marc_subfields() -> None:
    card = build_card(_row(_marc(), evidence_json='{"title.token_set": 0.9}'))
    assert card.pair_id == 7
    assert card.marc_title == "The Full Title : a subtitle"
    assert card.marc_statement_of_responsibility == "by Jane Doe"
    assert card.marc_main_author == "Doe, Jane"
    assert card.marc_added_authors == ("Roe, Richard",)
    assert card.marc_publisher == "Acme Press"
    assert card.marc_year == 1953
    assert card.marc_edition == "2nd ed."
    assert card.marc_series_titles == ("Acme Studies",)
    assert card.marc_lccn == "53001234"
    assert card.marc_language_code == "eng"
    assert card.marc_country_code == "nyu"


def test_build_card_renders_cce_and_renewal() -> None:
    card = build_card(_row(_marc(), evidence_json="{}", was_renewed=0))
    assert card.cce_title == "The Full Title"
    assert card.cce_publishers == "Acme Press | Beta Co"
    assert card.cce_claimants == "Jane Doe"
    assert card.cce_reg_year == 1953
    assert card.cce_regnum == "R99"
    assert card.cce_renewal_label == RENEWAL_NOT_RENEWED


def test_build_card_carries_evidence_bars() -> None:
    card = build_card(_row(_marc(), evidence_json='{"title.token_set": 1.0, "name.author": 0.25}'))
    assert [(bar.scorer, bar.normalized) for bar in card.evidence] == [
        ("title.token_set", 1.0),
        ("name.author", 0.25),
    ]


def test_build_card_flags_online_resource_extent() -> None:
    marc = _marc(extent="1 online resource (xxi, 406 p.)")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_is_online_resource is True


def test_build_card_does_not_flag_physical_extent() -> None:
    marc = _marc(extent="xxiv, 841 p")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_is_online_resource is False


def test_build_card_does_not_flag_missing_extent() -> None:
    marc = _marc(extent=None)
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_is_online_resource is False


def test_build_card_online_resource_match_is_case_insensitive() -> None:
    for extent in ("Online Resource (1 vol.)", "1 ONLINE RESOURCE (PDF)"):
        marc = _marc(extent=extent)
        card = build_card(_row(marc, evidence_json="{}"))
        assert card.marc_is_online_resource is True, extent


def test_build_card_flags_translation_from_desc() -> None:
    """``cce_is_translation`` fires when the CCE desc carries a translation cue."""
    card = build_card(_row(_marc(), evidence_json="{}", cce_desc="312 p. tr. from the French"))
    assert card.cce_is_translation is True


def test_build_card_flags_translation_from_notes() -> None:
    """``cce_is_translation`` fires when notes carry a translation cue."""
    card = build_card(
        _row(_marc(), evidence_json="{}", cce_notes="Original in German\nEnglish translation")
    )
    assert card.cce_is_translation is True


def test_build_card_flags_translation_from_new_matter_claimed() -> None:
    """``cce_is_translation`` fires from ``cce_new_matter_claimed``."""
    card = build_card(
        _row(_marc(), evidence_json="{}", cce_new_matter_claimed="English translation")
    )
    assert card.cce_is_translation is True


def test_build_card_flags_translation_from_renewal_new_matter() -> None:
    """``cce_is_translation`` fires from ``cce_renewal_new_matter``."""
    card = build_card(
        _row(_marc(), evidence_json="{}", cce_renewal_new_matter="translated from the Russian")
    )
    assert card.cce_is_translation is True


def test_build_card_does_not_flag_unrelated_text() -> None:
    """A CCE record with no translation cues yields ``cce_is_translation=False``."""
    card = build_card(_row(_marc(), evidence_json="{}", cce_desc="312 p. illus."))
    assert card.cce_is_translation is False


def test_author_is_claimant_label_maps_truthy_only() -> None:
    assert author_is_claimant_label(1) == CLAIMANT_LABEL
    assert author_is_claimant_label(0) is None
    assert author_is_claimant_label(None) is None


def test_build_card_projects_all_new_cce_fields() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json="{}",
            cce_edition="2nd ed.",
            cce_publication_places="New York; London",
            cce_author_place="Cambridge, Mass.",
            cce_author_is_claimant=1,
            cce_copies="2c.",
            cce_aff_date="1953-06-01",
            cce_desc="vi, 200 p.",
            cce_notes="note one\nnote two",
            cce_new_matter_claimed="added ch. 5",
            cce_copy_date="1953-04-01",
            cce_notice_date="1953-04-02",
        )
    )
    assert card.cce_edition == "2nd ed."
    assert card.cce_publication_places == ("New York", "London")
    assert card.cce_author_place == "Cambridge, Mass."
    assert card.cce_author_is_claimant is True
    assert card.author_is_claimant_label == CLAIMANT_LABEL
    assert card.cce_copies == "2c."
    assert card.cce_aff_date == date(1953, 6, 1)
    assert card.cce_desc == "vi, 200 p."
    assert card.cce_notes == ("note one", "note two")
    assert card.cce_new_matter_claimed == "added ch. 5"
    assert card.cce_copy_date == date(1953, 4, 1)
    assert card.cce_notice_date == date(1953, 4, 2)


def test_build_card_defaults_new_cce_fields_for_legacy_row() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.cce_edition is None
    assert card.cce_publication_places == ()
    assert card.cce_author_place is None
    assert card.cce_author_is_claimant is False
    assert card.author_is_claimant_label is None
    assert card.cce_copies is None
    assert card.cce_aff_date is None
    assert card.cce_desc is None
    assert card.cce_notes == ()
    assert card.cce_new_matter_claimed is None
    assert card.cce_copy_date is None
    assert card.cce_notice_date is None


def test_build_card_claimant_label_none_when_flag_false() -> None:
    card = build_card(_row(_marc(), evidence_json="{}", cce_author_is_claimant=0))
    assert card.cce_author_is_claimant is False
    assert card.author_is_claimant_label is None


def test_build_card_drops_blank_chunks_in_publication_places() -> None:
    card = build_card(
        _row(_marc(), evidence_json="{}", cce_publication_places="New York;  ; London")
    )
    assert card.cce_publication_places == ("New York", "London")


def test_build_card_single_note_round_trips() -> None:
    card = build_card(_row(_marc(), evidence_json="{}", cce_notes="just one"))
    assert card.cce_notes == ("just one",)


def test_build_card_projects_lccn_and_builds_lccn_url() -> None:
    card = build_card(_row(_marc(), evidence_json="{}", cce_lccn="28000854"))
    assert card.cce_lccn == "28000854"
    assert card.cce_lccn_canonical == "28000854"
    assert card.cce_lccn_url == "https://lccn.loc.gov/28000854"


def test_build_card_projects_canonical_form_for_hyphenated_cce_lccn() -> None:
    card = build_card(_row(_marc(), evidence_json="{}", cce_lccn="28-854"))
    assert card.cce_lccn == "28-854"
    assert card.cce_lccn_canonical == "28000854"
    assert card.cce_lccn_url == "https://lccn.loc.gov/28000854"


def test_build_card_lccn_url_none_when_lccn_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.cce_lccn is None
    assert card.cce_lccn_canonical is None
    assert card.cce_lccn_url is None


def test_build_card_projects_canonical_marc_lccn_equal_to_raw_when_already_canonical() -> None:
    card = build_card(_row(_marc(lccn="53001234"), evidence_json="{}"))
    assert card.marc_lccn == "53001234"
    assert card.marc_lccn_canonical == "53001234"


def test_build_card_projects_canonical_marc_lccn_normalises_hyphenated_form() -> None:
    card = build_card(_row(_marc(lccn="53-1234"), evidence_json="{}"))
    assert card.marc_lccn == "53-1234"
    assert card.marc_lccn_canonical == "53001234"


def test_build_card_projects_marc_lccn_canonical_none_when_absent() -> None:
    card = build_card(_row(_marc(lccn=None), evidence_json="{}"))
    assert card.marc_lccn is None
    assert card.marc_lccn_canonical is None


def test_build_card_projects_prev_regnums_in_order() -> None:
    card = build_card(
        _row(_marc(), evidence_json="{}", cce_prev_regnums="A100000; A200000; A300000")
    )
    assert card.cce_prev_regnums == ("A100000", "A200000", "A300000")


def test_build_card_prev_regnums_empty_tuple_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.cce_prev_regnums == ()


def test_build_card_prev_regnums_drops_blank_chunks() -> None:
    card = build_card(_row(_marc(), evidence_json="{}", cce_prev_regnums="A100000;  ; A200000"))
    assert card.cce_prev_regnums == ("A100000", "A200000")


def test_build_card_projects_publication_place_when_present() -> None:
    marc = _marc(publication_place="New York")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_publication_place == "New York"


def test_build_card_publication_place_none_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.marc_publication_place is None


def test_build_card_projects_extent_when_present() -> None:
    marc = _marc(extent="xxiv, 841 p.")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_extent == "xxiv, 841 p."


def test_build_card_extent_none_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.marc_extent is None


def test_build_card_projects_isbns_when_present() -> None:
    marc = _marc(isbns=("9780000000000", "9781111111111"))
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_isbns == ("9780000000000", "9781111111111")


def test_build_card_isbns_empty_tuple_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.marc_isbns == ()


def test_build_card_projects_oclc_and_url_when_present() -> None:
    marc = _marc(oclc="123456")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_oclc == "123456"
    assert card.marc_oclc_url == "https://www.worldcat.org/oclc/123456"


def test_build_card_oclc_and_url_none_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.marc_oclc is None
    assert card.marc_oclc_url is None


def test_build_card_oclc_url_strips_number_class_prefixes() -> None:
    # WorldCat URLs accept the numeric portion only; the historical
    # ocm/ocn/on prefixes carried in 035 $a must be stripped.
    marc_ocm = _marc(oclc="ocm01637690")
    assert (
        build_card(_row(marc_ocm, evidence_json="{}")).marc_oclc_url
        == "https://www.worldcat.org/oclc/01637690"
    )
    marc_ocn = _marc(oclc="ocn123456789")
    assert (
        build_card(_row(marc_ocn, evidence_json="{}")).marc_oclc_url
        == "https://www.worldcat.org/oclc/123456789"
    )
    marc_on = _marc(oclc="on1234567890")
    assert (
        build_card(_row(marc_on, evidence_json="{}")).marc_oclc_url
        == "https://www.worldcat.org/oclc/1234567890"
    )


def test_hathitrust_url_none_when_all_identifiers_absent() -> None:
    assert _hathitrust_url(None, None, ()) is None


def test_hathitrust_url_uses_oclc_when_only_oclc_present() -> None:
    assert (
        _hathitrust_url("123456", None, ())
        == "https://catalog.hathitrust.org/api/volumes/oclc/123456.html"
    )


def test_hathitrust_url_uses_lccn_when_only_lccn_present() -> None:
    assert (
        _hathitrust_url(None, "53001234", ())
        == "https://catalog.hathitrust.org/api/volumes/lccn/53001234.html"
    )


def test_hathitrust_url_uses_first_isbn_when_only_isbns_present() -> None:
    assert (
        _hathitrust_url(None, None, ("9780000000000", "9781111111111"))
        == "https://catalog.hathitrust.org/api/volumes/isbn/9780000000000.html"
    )


def test_hathitrust_url_oclc_wins_over_lccn_and_isbn() -> None:
    assert (
        _hathitrust_url("123456", "53001234", ("9780000000000",))
        == "https://catalog.hathitrust.org/api/volumes/oclc/123456.html"
    )


def test_hathitrust_url_oclc_wins_when_lccn_also_present() -> None:
    assert (
        _hathitrust_url("123456", "53001234", ())
        == "https://catalog.hathitrust.org/api/volumes/oclc/123456.html"
    )


def test_hathitrust_url_lccn_wins_over_isbn_when_no_oclc() -> None:
    assert (
        _hathitrust_url(None, "53001234", ("9780000000000",))
        == "https://catalog.hathitrust.org/api/volumes/lccn/53001234.html"
    )


def test_hathitrust_url_strips_oclc_number_class_prefix() -> None:
    assert (
        _hathitrust_url("ocm01637690", None, ())
        == "https://catalog.hathitrust.org/api/volumes/oclc/01637690.html"
    )


def test_hathitrust_url_encodes_lccn_space_and_slash() -> None:
    assert (
        _hathitrust_url(None, "n 79/12345", ())
        == "https://catalog.hathitrust.org/api/volumes/lccn/n%2079%2F12345.html"
    )


def test_hathitrust_url_empty_isbn_tuple_is_treated_as_absent() -> None:
    assert _hathitrust_url(None, None, ()) is None


def test_build_card_projects_hathitrust_url_from_oclc() -> None:
    marc = _marc(oclc="123456", lccn=None, isbns=())
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_hathitrust_url == "https://catalog.hathitrust.org/api/volumes/oclc/123456.html"


def test_build_card_projects_hathitrust_url_from_lccn_when_no_oclc() -> None:
    marc = _marc(oclc=None, lccn="53001234", isbns=())
    card = build_card(_row(marc, evidence_json="{}"))
    assert (
        card.marc_hathitrust_url == "https://catalog.hathitrust.org/api/volumes/lccn/53001234.html"
    )


def test_build_card_projects_hathitrust_url_from_isbn_when_no_oclc_or_lccn() -> None:
    marc = _marc(oclc=None, lccn=None, isbns=("9780000000000",))
    card = build_card(_row(marc, evidence_json="{}"))
    assert (
        card.marc_hathitrust_url
        == "https://catalog.hathitrust.org/api/volumes/isbn/9780000000000.html"
    )


def test_build_card_hathitrust_url_none_when_no_identifiers() -> None:
    marc = _marc(oclc=None, lccn=None, isbns=())
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_hathitrust_url is None


def test_build_card_projects_title_part_number_when_present() -> None:
    marc = _marc(title_part_number="Pt. 2")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_title_part_number == "Pt. 2"


def test_build_card_title_part_number_none_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.marc_title_part_number is None


def test_build_card_projects_title_part_name_when_present() -> None:
    marc = _marc(title_part_name="The empire of Sebastopol")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_title_part_name == "The empire of Sebastopol"


def test_build_card_title_part_name_none_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.marc_title_part_name is None


def test_build_card_projects_publication_date_raw_when_distinct_from_year() -> None:
    marc = _marc(publication_date_raw="c1953.", publication_year=1953)
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_publication_date_raw == "c1953."


def test_build_card_publication_date_raw_none_when_equals_year() -> None:
    marc = _marc(publication_date_raw="1953", publication_year=1953)
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_publication_date_raw is None


def test_build_card_publication_date_raw_none_when_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.marc_publication_date_raw is None


def test_publication_date_raw_if_distinct_returns_none_when_raw_absent() -> None:
    marc = _marc(publication_date_raw=None, publication_year=1953)
    assert _publication_date_raw_if_distinct(marc) is None


def test_publication_date_raw_if_distinct_returns_none_when_matches_year() -> None:
    marc = _marc(publication_date_raw="1953", publication_year=1953)
    assert _publication_date_raw_if_distinct(marc) is None


def test_publication_date_raw_if_distinct_returns_raw_when_differs_from_year() -> None:
    marc = _marc(publication_date_raw="[1953?]", publication_year=1953)
    assert _publication_date_raw_if_distinct(marc) == "[1953?]"


def test_publication_date_raw_if_distinct_returns_raw_when_year_missing() -> None:
    marc = _marc(publication_date_raw="n.d.", publication_year=None)
    assert _publication_date_raw_if_distinct(marc) == "n.d."


def _labeled_row(
    *,
    pair_id: int = 1,
    language: str = "eng",
    marc_control_id: str = "ctrl-a",
    marc_title: str | None = "A short title",
    cce_title: str | None = "CCE title",
    verdict: str = "no_match",
    note: str | None = None,
    labeled_at: str = "2026-05-23T11:30:00+00:00",
) -> LabeledPairRow:
    return LabeledPairRow(
        pair_id=pair_id,
        language=language,
        marc_control_id=marc_control_id,
        marc_title=marc_title,
        cce_title=cce_title,
        verdict=verdict,
        note=note,
        labeled_at=labeled_at,
    )


def test_build_labeled_row_truncates_long_titles_with_ellipsis() -> None:
    long_title = "x" * 120
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    row = build_labeled_row(_labeled_row(marc_title=long_title, cce_title=long_title), now)
    assert row.marc_title == long_title
    assert len(row.marc_title_short) == 60
    assert row.marc_title_short.endswith("…")
    assert row.cce_title_short.endswith("…")


def test_build_labeled_row_preserves_short_titles_unchanged() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    row = build_labeled_row(_labeled_row(marc_title="Short", cce_title="Also short"), now)
    assert row.marc_title_short == "Short"
    assert row.cce_title_short == "Also short"


def test_build_labeled_row_handles_null_titles() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    row = build_labeled_row(_labeled_row(marc_title=None, cce_title=None), now)
    assert row.marc_title == ""
    assert row.cce_title == ""
    assert row.marc_title_short == ""
    assert row.cce_title_short == ""


def test_build_labeled_row_handles_null_note() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    row = build_labeled_row(_labeled_row(note=None), now)
    assert row.note == ""
    assert row.note_short == ""


def test_build_labeled_row_preserves_short_note_unchanged() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    row = build_labeled_row(_labeled_row(note="quick check"), now)
    assert row.note == "quick check"
    assert row.note_short == "quick check"


def test_build_labeled_row_truncates_long_note_with_ellipsis() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    long_note = "x" * 200
    row = build_labeled_row(_labeled_row(note=long_note), now)
    assert row.note == long_note
    assert len(row.note_short) == 120
    assert row.note_short.endswith("…")


def test_build_labeled_row_renders_relative_time() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    past = (now - timedelta(hours=3)).isoformat()
    row = build_labeled_row(_labeled_row(labeled_at=past), now)
    assert row.labeled_relative == "3h ago"
    assert row.labeled_at == past


def test_build_card_projects_renewal_details_when_populated() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json="{}",
            cce_renewal_id="R200001",
            cce_renewal_oreg="A111111",
            cce_renewal_rdat="1968-05-15",
            cce_renewal_author="Smith, John",
            cce_renewal_title="A study of widgets",
            cce_renewal_claimants="Acme Press|PWH",
            cce_renewal_new_matter="added ch. 7",
        )
    )
    assert card.cce_renewal_id == "R200001"
    assert card.cce_renewal_oreg == "A111111"
    assert card.cce_renewal_rdat == date(1968, 5, 15)
    assert card.cce_renewal_author == "Smith, John"
    assert card.cce_renewal_title == "A study of widgets"
    assert card.cce_renewal_claimants == "Acme Press|PWH"
    assert card.cce_renewal_new_matter == "added ch. 7"
    assert card.cce_has_renewal_details is True


def test_build_card_renewal_details_absent_for_legacy_row() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.cce_renewal_id is None
    assert card.cce_renewal_rdat is None
    assert card.cce_has_renewal_details is False


def test_build_card_renewal_claimants_differ_when_registration_disagrees() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json="{}",
            cce_claimants="Jane Doe",
            cce_renewal_claimants="Estate of Jane Doe",
        )
    )
    assert card.cce_renewal_claimants_differ is True


def test_build_card_renewal_claimants_match_collapses_whitespace_and_case() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json="{}",
            cce_claimants="Jane Doe",
            cce_renewal_claimants="jane doe",
        )
    )
    assert card.cce_renewal_claimants_differ is False


def test_build_card_renewal_claimants_no_diff_signal_when_either_side_blank() -> None:
    card = build_card(
        _row(
            _marc(),
            evidence_json="{}",
            cce_claimants=None,
            cce_renewal_claimants="Estate of Jane Doe",
        )
    )
    assert card.cce_renewal_claimants_differ is False


def _current_label(
    *,
    pair_id: int = 7,
    verdict: str = "match",
    note: str | None = "looks right",
) -> CurrentLabelRow:
    return CurrentLabelRow(
        pair_id=pair_id,
        marc_control_id="ctrl-1",
        nypl_uuid="uuid-9",
        marc_json="{}",
        verdict=verdict,
        note=note,
        labeled_at="2026-05-23T11:30:00+00:00",
    )


def test_build_card_note_and_verdict_default_none_when_no_current_label() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.note is None
    assert card.current_verdict is None


def test_build_card_projects_current_label_note_and_verdict() -> None:
    card = build_card(
        _row(_marc(), evidence_json="{}"),
        current_label=_current_label(verdict="no_match", note="title collision"),
    )
    assert card.note == "title collision"
    assert card.current_verdict == "no_match"


def test_build_card_current_label_with_no_note_propagates_none() -> None:
    card = build_card(
        _row(_marc(), evidence_json="{}"),
        current_label=_current_label(verdict="unsure", note=None),
    )
    assert card.note is None
    assert card.current_verdict == "unsure"
