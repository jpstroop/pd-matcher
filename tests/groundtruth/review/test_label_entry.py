"""Unit tests for the label-time vault entry builder."""

from msgspec.json import encode as json_encode

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.review.label_entry import build_label_entry
from pd_groundtruth.review_db import ReviewPairRow
from pd_matcher.models import MarcRecord


def _marc(
    *,
    control_id: str = "ctrl-1",
    lccn: str | None = "53001234",
    oclc: str | None = "0001",
    isbns: tuple[str, ...] = ("9780000000000",),
) -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A Studied Title",
        title_main="A Studied Title",
        lccn=lccn,
        oclc=oclc,
        isbns=isbns,
        main_author="Doe, Jane",
        publisher="Acme Press",
        publication_year=1953,
        language_code="eng",
        country_code="nyu",
    )


def _row(
    marc: MarcRecord | None = None,
    *,
    cce_reg_year: int | None = 1953,
    cce_was_renewed: int | None = 1,
    cce_renewal_rdat: str | None = "1981-04-01",
    cce_regnum: str | None = "R99",
    cce_renewal_id: str | None = "R200001",
    cce_renewal_oreg: str | None = "A111111",
    pairing_type: str = "registration",
) -> ReviewPairRow:
    blob = marc if marc is not None else _marc()
    return ReviewPairRow(
        id=7,
        language="eng",
        decade=1950,
        score=0.91,
        band="ge90",
        source="banded",
        pairing_type=pairing_type,
        marc_control_id=blob.control_id,
        marc_json=json_encode(blob).decode("utf-8"),
        marc_title=blob.title,
        marc_author=blob.main_author,
        marc_publisher=blob.publisher,
        marc_year=blob.publication_year,
        nypl_uuid="uuid-9",
        cce_title="A Studied Title",
        cce_author="Jane Doe",
        cce_publishers="Acme Press",
        cce_claimants="Jane Doe",
        cce_reg_year=cce_reg_year,
        cce_was_renewed=cce_was_renewed,
        cce_regnum=cce_regnum,
        evidence_json="{}",
        created_at="2026-06-20T00:00:00+00:00",
        cce_renewal_id=cce_renewal_id,
        cce_renewal_oreg=cce_renewal_oreg,
        cce_renewal_rdat=cce_renewal_rdat,
    )


def _build(row: ReviewPairRow) -> VaultEntry:
    return build_label_entry(
        row,
        verdict="match",
        note="title agrees",
        labeled_at="2026-06-20T12:00:00+00:00",
        labeler="jpstroop",
        categories=("translation",),
    )


def test_build_label_entry_stamps_static_cce_facts() -> None:
    entry = _build(_row())
    assert entry.reg_year == 1953
    assert entry.renewal_year == 1981
    assert entry.was_renewed is True


def test_build_label_entry_leaves_scores_and_matcher_version_none() -> None:
    entry = _build(_row())
    assert entry.scores is None
    assert entry.matcher_version is None


def test_build_label_entry_writes_current_schema() -> None:
    entry = _build(_row())
    assert entry.schema == SCHEMA_VERSION


def test_build_label_entry_defaults_match_source_to_registration() -> None:
    entry = _build(_row())
    assert entry.match_source == "registration"


def test_build_label_entry_sets_match_source_renewal_for_renewal_pairing() -> None:
    entry = _build(_row(pairing_type="renewal"))
    assert entry.match_source == "renewal"


def test_build_label_entry_copies_human_and_identifier_fields() -> None:
    entry = _build(_row())
    assert entry.marc_control_id == "ctrl-1"
    assert entry.nypl_uuid == "uuid-9"
    assert entry.verdict == "match"
    assert entry.note == "title agrees"
    assert entry.labeled_at == "2026-06-20T12:00:00+00:00"
    assert entry.labeler == "jpstroop"
    assert entry.categories == ("translation",)
    assert entry.cce_regnum == "R99"
    assert entry.cce_renewal_id == "R200001"
    assert entry.cce_renewal_oreg == "A111111"


def test_build_label_entry_extracts_marc_identifiers_from_blob() -> None:
    entry = _build(_row())
    assert entry.marc_identifiers.lccn == "53001234"
    assert entry.marc_identifiers.oclc == "0001"
    assert entry.marc_identifiers.isbns == ("9780000000000",)


def test_build_label_entry_renewal_year_none_when_no_renewal() -> None:
    entry = _build(_row(cce_was_renewed=0, cce_renewal_rdat=None))
    assert entry.renewal_year is None
    assert entry.was_renewed is False


def test_build_label_entry_was_renewed_none_when_status_unknown() -> None:
    entry = _build(_row(cce_was_renewed=None, cce_renewal_rdat=None))
    assert entry.was_renewed is None
    assert entry.renewal_year is None


def test_build_label_entry_passes_through_missing_reg_year() -> None:
    entry = _build(_row(cce_reg_year=None))
    assert entry.reg_year is None
