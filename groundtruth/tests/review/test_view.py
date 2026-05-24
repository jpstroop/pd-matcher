"""Unit tests for the pure review view model."""

from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta

from msgspec.json import encode as json_encode
from pd_matcher.models import MarcRecord

from pd_groundtruth.review.view import CLAIMANT_LABEL
from pd_groundtruth.review.view import RENEWAL_NOT_RENEWED
from pd_groundtruth.review.view import RENEWAL_RENEWED
from pd_groundtruth.review.view import RENEWAL_UNKNOWN
from pd_groundtruth.review.view import author_is_claimant_label
from pd_groundtruth.review.view import build_card
from pd_groundtruth.review.view import build_labeled_row
from pd_groundtruth.review.view import parse_evidence
from pd_groundtruth.review.view import render_renewal_label
from pd_groundtruth.review_db import LabeledPairRow
from pd_groundtruth.review_db import ReviewPairRow


def _marc(
    *,
    title: str = "The Full Title : a subtitle",
    title_main: str = "The Full Title",
    extent: str | None = None,
) -> MarcRecord:
    return MarcRecord(
        control_id="ctrl-1",
        title=title,
        title_main=title_main,
        lccn="53001234",
        main_author="Doe, Jane",
        added_authors=("Roe, Richard",),
        statement_of_responsibility="by Jane Doe",
        edition="2nd ed.",
        publisher="Acme Press",
        publication_year=1953,
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
        cce_claimants="Jane Doe",
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


def test_build_card_exposes_marc_subfields() -> None:
    card = build_card(_row(_marc(), evidence_json='{"title.token_set": 0.9}'))
    assert card.pair_id == 7
    assert card.marc_title == "The Full Title : a subtitle"
    assert card.marc_title_main == "The Full Title"
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


def test_build_card_omits_title_main_when_equal_to_title() -> None:
    marc = _marc(title="Same Title", title_main="Same Title")
    card = build_card(_row(marc, evidence_json="{}"))
    assert card.marc_title_main is None


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
    assert card.cce_lccn_url == "https://lccn.loc.gov/28000854"


def test_build_card_lccn_url_preserves_human_form() -> None:
    card = build_card(_row(_marc(), evidence_json="{}", cce_lccn="28-854"))
    assert card.cce_lccn == "28-854"
    assert card.cce_lccn_url == "https://lccn.loc.gov/28-854"


def test_build_card_lccn_url_none_when_lccn_absent() -> None:
    card = build_card(_row(_marc(), evidence_json="{}"))
    assert card.cce_lccn is None
    assert card.cce_lccn_url is None


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


def _labeled_row(
    *,
    pair_id: int = 1,
    language: str = "eng",
    marc_control_id: str = "ctrl-a",
    marc_title: str | None = "A short title",
    cce_title: str | None = "CCE title",
    verdict: str = "no_match",
    reason_codes: tuple[str, ...] = (),
    labeled_at: str = "2026-05-23T11:30:00+00:00",
) -> LabeledPairRow:
    return LabeledPairRow(
        pair_id=pair_id,
        language=language,
        marc_control_id=marc_control_id,
        marc_title=marc_title,
        cce_title=cce_title,
        verdict=verdict,
        reason_codes=reason_codes,
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


def test_build_labeled_row_resolves_reason_codes_to_labels() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    row = build_labeled_row(
        _labeled_row(verdict="no_match", reason_codes=("diff_work", "garbled")),
        now,
    )
    assert row.reason_codes == ("diff_work", "garbled")
    assert row.reason_labels == (
        "Different work / title collision",
        "Garbled transcription",
    )


def test_build_labeled_row_falls_back_to_code_when_label_unknown() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    row = build_labeled_row(
        _labeled_row(verdict="match", reason_codes=("legacy_code",)),
        now,
    )
    assert row.reason_labels == ("legacy_code",)


def test_build_labeled_row_renders_relative_time() -> None:
    now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    past = (now - timedelta(hours=3)).isoformat()
    row = build_labeled_row(_labeled_row(labeled_at=past), now)
    assert row.labeled_relative == "3h ago"
    assert row.labeled_at == past
