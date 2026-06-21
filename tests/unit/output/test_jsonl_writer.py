"""Tests for :mod:`pd_matcher.output.jsonl_writer`."""

from datetime import date
from pathlib import Path

from msgspec.json import decode as json_decode
from pytest import raises

from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.output.jsonl_writer import RECORD_FIELDS
from pd_matcher.output.jsonl_writer import JsonlResultWriter


def _read_records(path: Path) -> list[dict[str, str]]:
    """Decode each non-empty JSONL line into a record dict."""
    return [
        json_decode(line, type=dict[str, str])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _marc() -> MarcRecord:
    return MarcRecord(
        control_id="marc-1",
        title="A study of widgets",
        title_main="A study of widgets",
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


def test_jsonl_writer_emits_one_object_per_record(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    with JsonlResultWriter(path) as writer:
        writer.write(_marc(), _match(), _nypl())
    records = _read_records(path)
    assert len(records) == 1
    record = records[0]
    assert tuple(record.keys()) == RECORD_FIELDS
    assert record["marc_id"] == "marc-1"
    assert record["marc_title_original"] == "A study of widgets"
    assert record["marc_title_normalized"] == "a study of widgets"
    assert record["marc_title_stemmed"] != ""
    assert record["marc_year"] == "1940"
    assert record["marc_lccn"] == "40012345"
    assert record["match_type"] == "registration"
    assert record["match_source_id"] == "UUID-0001"
    assert record["match_year"] == "1940"
    assert record["match_date"] == "1940-05-10"
    assert record["title_score"] == "88"
    assert record["author_score"] == "67"
    assert record["publisher_score"] == "100"
    assert record["combined_score"] == f"{0.84 * 100.0:.2f}"
    assert record["year_difference"] == "0"


def test_jsonl_writer_emits_blank_match_fields_when_no_match(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    marc = _marc()
    with JsonlResultWriter(path) as writer:
        writer.write(marc, None)
    records = _read_records(path)
    assert records[0]["match_type"] == ""
    assert records[0]["match_source_id"] == ""
    assert records[0]["title_score"] == ""
    assert records[0]["combined_score"] == ""
    assert records[0]["year_difference"] == ""


def test_jsonl_writer_blanks_when_match_present_but_indexed_record_missing(
    tmp_path: Path,
) -> None:
    """Worker without an indexed record falls back to blank match fields."""
    path = tmp_path / "out.jsonl"
    with JsonlResultWriter(path) as writer:
        writer.write(_marc(), _match(), None)
    records = _read_records(path)
    assert records[0]["match_source_id"] == ""
    assert records[0]["title_score"] == ""


def test_jsonl_writer_handles_none_normalized_value(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    marc = MarcRecord(control_id="m", title="X", title_main="X")
    with JsonlResultWriter(path) as writer:
        writer.write(marc, None)
    records = _read_records(path)
    assert records[0]["marc_title_normalized"] == "x"
    assert records[0]["marc_publisher_original"] == ""
    assert records[0]["marc_year"] == ""


def test_jsonl_writer_handles_value_normalising_to_empty(tmp_path: Path) -> None:
    """A non-empty input that normalizes to empty (pure punctuation) is preserved."""
    path = tmp_path / "out.jsonl"
    marc = MarcRecord(control_id="m", title="X", title_main="X", publisher="///")
    with JsonlResultWriter(path) as writer:
        writer.write(marc, None)
    records = _read_records(path)
    assert records[0]["marc_publisher_original"] == "///"
    assert records[0]["marc_publisher_normalized"] == ""
    assert records[0]["marc_publisher_stemmed"] == ""


def test_jsonl_writer_match_date_falls_back_to_year(tmp_path: Path) -> None:
    """When the matched record has only a year (no full reg_date), match_date is the year."""
    path = tmp_path / "out.jsonl"
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
    with JsonlResultWriter(path) as writer:
        writer.write(_marc(), match, nypl)
    records = _read_records(path)
    assert records[0]["match_date"] == "1942"


def test_jsonl_writer_match_date_blank_when_no_year(tmp_path: Path) -> None:
    """When the matched record has neither reg_date nor reg_year, match_date is blank."""
    path = tmp_path / "out.jsonl"
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
    with JsonlResultWriter(path) as writer:
        writer.write(_marc(), match, nypl)
    records = _read_records(path)
    assert records[0]["match_date"] == ""
    assert records[0]["match_year"] == ""
    assert records[0]["year_difference"] == ""


def test_jsonl_writer_skipped_evidence_returns_empty_score(tmp_path: Path) -> None:
    """Skipped Evidence in best.evidence renders as an empty score field."""
    path = tmp_path / "out.jsonl"
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
    with JsonlResultWriter(path) as writer:
        writer.write(_marc(), match, _nypl())
    records = _read_records(path)
    assert records[0]["title_score"] == ""
    assert records[0]["author_score"] == "67"
    assert records[0]["publisher_score"] == ""


def test_jsonl_writer_write_outside_context_raises(tmp_path: Path) -> None:
    writer = JsonlResultWriter(tmp_path / "out.jsonl")
    with raises(RuntimeError, match="not entered"):
        writer.write(_marc(), None)


def test_jsonl_writer_exit_without_enter_is_a_noop(tmp_path: Path) -> None:
    """Exiting without first entering the context manager does not raise."""
    writer = JsonlResultWriter(tmp_path / "out.jsonl")
    writer.__exit__(None, None, None)


def test_jsonl_writer_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "out.jsonl"
    with JsonlResultWriter(path) as writer:
        writer.write(_marc(), None)
    assert path.exists()


def test_jsonl_writer_emits_multiple_records_one_per_line(tmp_path: Path) -> None:
    """Two writes produce two JSONL lines, each a standalone object."""
    path = tmp_path / "out.jsonl"
    with JsonlResultWriter(path) as writer:
        writer.write(_marc(), _match(), _nypl())
        writer.write(_marc(), None)
    records = _read_records(path)
    assert len(records) == 2
    assert records[0]["match_source_id"] == "UUID-0001"
    assert records[1]["match_source_id"] == ""
