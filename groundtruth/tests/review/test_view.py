"""Unit tests for the pure review view model."""

from msgspec.json import encode as json_encode
from pd_matcher.models import MarcRecord

from pd_groundtruth.review.view import RENEWAL_NOT_RENEWED
from pd_groundtruth.review.view import RENEWAL_RENEWED
from pd_groundtruth.review.view import RENEWAL_UNKNOWN
from pd_groundtruth.review.view import build_card
from pd_groundtruth.review.view import parse_evidence
from pd_groundtruth.review.view import render_renewal_label
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


def _row(marc: MarcRecord, *, evidence_json: str, was_renewed: int | None = 1) -> ReviewPairRow:
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
