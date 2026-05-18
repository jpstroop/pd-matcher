"""Tests for :mod:`pd_matcher.output.csv_writer`."""

from csv import DictReader
from datetime import date
from pathlib import Path

from pytest import raises

from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.output.csv_writer import CSV_COLUMNS
from pd_matcher.output.csv_writer import CsvResultWriter


def _marc() -> MarcRecord:
    return MarcRecord(
        control_id="marc-1",
        title="A study of widgets",
        lccn="40012345",
        main_author="Alpha, Alice",
        statement_of_responsibility="by Alice Alpha",
        publisher="Acme Press",
        publication_year=1940,
        country_code="nyu",
        language_code="eng",
    )


def _nypl() -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="UUID-0001",
        title="A study of widgets",
        was_renewed=False,
        regnum="A111111",
        reg_date=date(1940, 5, 10),
        reg_year=1940,
        author_name="Smith, John",
        publisher_names=("Acme Press",),
    )


def _evidence(scorer: str, score: float, *, skipped: bool = False) -> Evidence:
    return Evidence(
        scorer=scorer,
        score=score,
        max=100.0,
        skipped=skipped,
        decisive=False,
        features=(),
    )


def _match() -> MatchResult:
    evidence = (
        _evidence("title.token_set", 88.5),
        _evidence("name.author", 67.2),
        _evidence("name.publisher", 100.0),
    )
    best = CandidateMatch(
        nypl_uuid="UUID-0001",
        nypl_year=1940,
        combined=CombinedScore(raw=85.0, calibrated=0.84),
        evidence=evidence,
        losing_evidence=(),
    )
    return MatchResult(
        marc_control_id="marc-1",
        best=best,
        alternates=(),
        candidates_considered=3,
    )


def _assessment() -> CopyrightAssessment:
    return CopyrightAssessment(
        status=CopyrightStatus.PD_REGISTERED_NOT_RENEWED,
        matched_rule_name="us_registered_not_renewed_1931_1963",
        explanation="ok",
        assumptions=(),
    )


def test_csv_writer_emits_header_and_matched_row(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    with CsvResultWriter(path) as writer:
        writer.write(_marc(), _match(), _assessment(), _nypl())
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row.keys()) == CSV_COLUMNS
    assert row["marc_id"] == "marc-1"
    assert row["marc_title_original"] == "A study of widgets"
    assert row["marc_title_normalized"] == "a study of widgets"
    assert row["marc_title_stemmed"] != ""
    assert row["marc_year"] == "1940"
    assert row["marc_lccn"] == "40012345"
    assert row["match_type"] == "registration"
    assert row["match_source_id"] == "UUID-0001"
    assert row["match_year"] == "1940"
    assert row["match_date"] == "1940-05-10"
    assert row["title_score"] == "88"
    assert row["author_score"] == "67"
    assert row["publisher_score"] == "100"
    assert row["combined_score"] == f"{0.84 * 100.0:.2f}"
    assert row["year_difference"] == "0"
    assert row["copyright_status"] == CopyrightStatus.PD_REGISTERED_NOT_RENEWED.value


def test_csv_writer_emits_blank_match_columns_when_no_match(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    marc = _marc()
    assessment = CopyrightAssessment(
        status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
        matched_rule_name=None,
        explanation="no match",
        assumptions=(),
    )
    with CsvResultWriter(path) as writer:
        writer.write(marc, None, assessment)
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert rows[0]["match_type"] == ""
    assert rows[0]["match_source_id"] == ""
    assert rows[0]["title_score"] == ""
    assert rows[0]["combined_score"] == ""
    assert rows[0]["year_difference"] == ""
    assert rows[0]["copyright_status"] == CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA.value


def test_csv_writer_blanks_when_match_present_but_indexed_record_missing(
    tmp_path: Path,
) -> None:
    """Worker without an indexed record falls back to blank match columns."""
    path = tmp_path / "out.csv"
    with CsvResultWriter(path) as writer:
        writer.write(_marc(), _match(), _assessment(), None)
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert rows[0]["match_source_id"] == ""
    assert rows[0]["title_score"] == ""


def test_csv_writer_handles_none_normalized_value(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    marc = MarcRecord(control_id="m", title="X")
    assessment = CopyrightAssessment(
        status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
        matched_rule_name=None,
        explanation="",
        assumptions=(),
    )
    with CsvResultWriter(path) as writer:
        writer.write(marc, None, assessment)
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert rows[0]["marc_title_normalized"] == "x"
    assert rows[0]["marc_publisher_original"] == ""
    assert rows[0]["marc_year"] == ""


def test_csv_writer_handles_value_normalising_to_empty(tmp_path: Path) -> None:
    """A non-empty input that normalizes to empty (pure punctuation) is preserved."""
    path = tmp_path / "out.csv"
    marc = MarcRecord(control_id="m", title="X", publisher="///")
    assessment = CopyrightAssessment(
        status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
        matched_rule_name=None,
        explanation="",
        assumptions=(),
    )
    with CsvResultWriter(path) as writer:
        writer.write(marc, None, assessment)
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert rows[0]["marc_publisher_original"] == "///"
    assert rows[0]["marc_publisher_normalized"] == ""
    assert rows[0]["marc_publisher_stemmed"] == ""


def test_csv_writer_match_date_falls_back_to_year(tmp_path: Path) -> None:
    """When the matched record has only a year (no full reg_date), match_date is the year."""
    path = tmp_path / "out.csv"
    nypl = IndexedNyplRegRecord(
        uuid="UUID-0002",
        title="Other",
        was_renewed=False,
        reg_year=1942,
    )
    match = MatchResult(
        marc_control_id="marc-1",
        best=CandidateMatch(
            nypl_uuid="UUID-0002",
            nypl_year=1942,
            combined=CombinedScore(raw=70.0, calibrated=0.7),
            evidence=(),
            losing_evidence=(),
        ),
        alternates=(),
        candidates_considered=1,
    )
    with CsvResultWriter(path) as writer:
        writer.write(_marc(), match, _assessment(), nypl)
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert rows[0]["match_date"] == "1942"


def test_csv_writer_match_date_blank_when_no_year(tmp_path: Path) -> None:
    """When the matched record has neither reg_date nor reg_year, match_date is blank."""
    path = tmp_path / "out.csv"
    nypl = IndexedNyplRegRecord(uuid="UUID-X", title="No date", was_renewed=False)
    match = MatchResult(
        marc_control_id="marc-1",
        best=CandidateMatch(
            nypl_uuid="UUID-X",
            nypl_year=None,
            combined=CombinedScore(raw=70.0, calibrated=0.7),
            evidence=(),
            losing_evidence=(),
        ),
        alternates=(),
        candidates_considered=1,
    )
    with CsvResultWriter(path) as writer:
        writer.write(_marc(), match, _assessment(), nypl)
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert rows[0]["match_date"] == ""
    assert rows[0]["match_year"] == ""
    assert rows[0]["year_difference"] == ""


def test_csv_writer_skipped_evidence_returns_empty_score(tmp_path: Path) -> None:
    """Skipped Evidence in best.evidence renders as an empty score column."""
    path = tmp_path / "out.csv"
    evidence = (
        _evidence("title.token_set", 0.0, skipped=True),
        _evidence("name.author", 67.0),
    )
    match = MatchResult(
        marc_control_id="marc-1",
        best=CandidateMatch(
            nypl_uuid="UUID-0001",
            nypl_year=1940,
            combined=CombinedScore(raw=85.0, calibrated=0.85),
            evidence=evidence,
            losing_evidence=(),
        ),
        alternates=(),
        candidates_considered=1,
    )
    with CsvResultWriter(path) as writer:
        writer.write(_marc(), match, _assessment(), _nypl())
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert rows[0]["title_score"] == ""
    assert rows[0]["author_score"] == "67"
    assert rows[0]["publisher_score"] == ""


def test_csv_writer_write_outside_context_raises(tmp_path: Path) -> None:
    writer = CsvResultWriter(tmp_path / "out.csv")
    with raises(RuntimeError, match="not entered"):
        writer.write(
            _marc(),
            None,
            CopyrightAssessment(
                status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
                matched_rule_name=None,
                explanation="",
                assumptions=(),
            ),
        )


def test_csv_writer_exit_without_enter_is_a_noop(tmp_path: Path) -> None:
    """Exiting without first entering the context manager does not raise."""
    writer = CsvResultWriter(tmp_path / "out.csv")
    writer.__exit__(None, None, None)


def test_csv_writer_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "out.csv"
    with CsvResultWriter(path) as writer:
        writer.write(
            _marc(),
            None,
            CopyrightAssessment(
                status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
                matched_rule_name=None,
                explanation="",
                assumptions=(),
            ),
        )
    assert path.exists()
