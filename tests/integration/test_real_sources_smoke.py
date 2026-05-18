"""Optional smoke tests that pull the first records from real data sources.

Each test is skipped when its source file is not present, so CI environments
that do not check out the data submodules still pass cleanly. When the data
is present, parsing the first 100 records of each source confirms our
streaming parsers handle the actual shapes the project ingests.
"""

from itertools import islice
from pathlib import Path

from pytest import mark

from pd_matcher.parsers.marc import iter_marc_records
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_records
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_records

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MARC = _REPO_ROOT / "data" / "candidate_marc_file.marcxml"
_NYPL_REG = _REPO_ROOT / "data" / "nypl-reg" / "xml" / "1939" / "1939_v36_n1.xml"
_NYPL_REN = _REPO_ROOT / "data" / "nypl-ren" / "data" / "1950-14A.tsv"


@mark.skipif(not _MARC.exists(), reason="real MARCXML file is not available")
def test_real_marc_first_records_parse() -> None:
    records = list(islice(iter_marc_records(_MARC), 100))
    assert len(records) == 100
    assert all(record.control_id for record in records)
    assert all(record.title for record in records)


@mark.skipif(not _NYPL_REG.exists(), reason="real NYPL registration file is not available")
def test_real_nypl_reg_first_records_parse() -> None:
    records = list(islice(iter_nypl_reg_records(_NYPL_REG), 100))
    assert records
    assert all(r.uuid for r in records)
    assert all(r.title for r in records)


@mark.skipif(not _NYPL_REN.exists(), reason="real NYPL renewal file is not available")
def test_real_nypl_ren_first_records_parse() -> None:
    records = list(islice(iter_nypl_ren_records(_NYPL_REN), 100))
    assert records
    assert all(r.id for r in records)
    assert all(r.entry_id for r in records)
