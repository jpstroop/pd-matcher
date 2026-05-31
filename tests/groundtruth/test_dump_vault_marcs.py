"""Unit tests for the dump-vault-marcs command."""

from pathlib import Path

from lxml.etree import fromstring
from lxml.etree import iterparse
from typer.testing import CliRunner

from pd_groundtruth.cli import app
from pd_groundtruth.dump_vault_marcs import dump_vault_marcs
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import upsert_entry

_RUNNER = CliRunner()

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_RECORD_TAG = f"{{{_MARC_NS}}}record"
_CONTROLFIELD_TAG = f"{{{_MARC_NS}}}controlfield"

_MARCXML_TEMPLATE = '<collection xmlns="{ns}">{records}</collection>'

_RECORD_TEMPLATE = (
    "<record>"
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">{control_id}</controlfield>'
    '<datafield tag="245" ind1="0" ind2="0"><subfield code="a">{title}</subfield></datafield>'
    "</record>"
)


def _write_shard(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        _RECORD_TEMPLATE.format(control_id=control_id, title=title) for control_id, title in records
    )
    path.write_text(
        _MARCXML_TEMPLATE.format(ns=_MARC_NS, records=body),
        encoding="utf-8",
    )


def _entry(marc_control_id: str, nypl_uuid: str, *, verdict: str = "match") -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=marc_control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        note=None,
        labeled_at="2026-05-31T00:00:00+00:00",
        labeler="test",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )


def _control_numbers(path: Path) -> list[str]:
    """Return every 001 controlfield value in the dumped MARCXML, in order."""
    values: list[str] = []
    context = iterparse(str(path), events=("end",), tag=_RECORD_TAG)
    for _event, record in context:
        for child in record.iterchildren(tag=_CONTROLFIELD_TAG):
            if child.get("tag") == "001":
                values.append((child.text or "").strip())
                break
        record.clear()
    return values


def test_writes_only_marcs_referenced_by_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-a", "uuid-a"))
    upsert_entry(vault, _entry("ctrl-c", "uuid-c"))
    _write_shard(
        pool / "eng" / "shard.xml",
        [
            ("ctrl-a", "Wanted A"),
            ("ctrl-b", "Not Wanted"),
            ("ctrl-c", "Wanted C"),
        ],
    )

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.vault_entries == 2
    assert report.distinct_marcs_requested == 2
    assert report.marcs_written == 2
    assert report.marcs_missing == 0
    assert sorted(_control_numbers(tmp_path / "vault_marcs.xml")) == ["ctrl-a", "ctrl-c"]


def test_reports_marcs_missing_from_pool(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-present", "uuid-1"))
    upsert_entry(vault, _entry("ctrl-gone", "uuid-2"))
    _write_shard(pool / "eng" / "shard.xml", [("ctrl-present", "Here")])

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.marcs_written == 1
    assert report.marcs_missing == 1
    assert _control_numbers(tmp_path / "vault_marcs.xml") == ["ctrl-present"]


def test_empty_vault_writes_empty_collection(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    _write_shard(pool / "eng" / "shard.xml", [("ctrl-a", "Anything")])

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.vault_entries == 0
    assert report.distinct_marcs_requested == 0
    assert report.marcs_written == 0
    assert report.marcs_missing == 0
    root = fromstring((tmp_path / "vault_marcs.xml").read_bytes())
    assert root.tag == f"{{{_MARC_NS}}}collection"
    assert list(root.iterchildren(tag=_RECORD_TAG)) == []


def test_unsure_verdict_entries_still_included(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-match", "uuid-m", verdict="match"))
    upsert_entry(vault, _entry("ctrl-unsure", "uuid-u", verdict="unsure"))
    _write_shard(
        pool / "eng" / "shard.xml",
        [
            ("ctrl-match", "M"),
            ("ctrl-unsure", "U"),
        ],
    )

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.marcs_written == 2
    assert sorted(_control_numbers(tmp_path / "vault_marcs.xml")) == [
        "ctrl-match",
        "ctrl-unsure",
    ]


def test_deduplicates_when_same_marc_appears_in_multiple_shards(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-dupe", "uuid-1"))
    _write_shard(pool / "eng" / "shard_one.xml", [("ctrl-dupe", "First")])
    _write_shard(pool / "fre" / "shard_two.xml", [("ctrl-dupe", "Second")])

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.marcs_written == 1
    assert _control_numbers(tmp_path / "vault_marcs.xml") == ["ctrl-dupe"]


def test_stops_early_once_all_wanted_marcs_found(tmp_path: Path) -> None:
    """The walker should not parse shards it doesn't need to."""
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-only", "uuid-1"))
    _write_shard(pool / "eng" / "shard_aaa.xml", [("ctrl-only", "Found")])
    bogus = pool / "zzz" / "shard_bogus.xml"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_text("not even XML", encoding="utf-8")

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.marcs_written == 1
    assert report.marcs_missing == 0


def test_record_with_missing_control_number_is_skipped(tmp_path: Path) -> None:
    """A record lacking a 001 controlfield is ignored, not crashed on."""
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-real", "uuid-1"))
    shard = pool / "eng" / "shard.xml"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        _MARCXML_TEMPLATE.format(
            ns=_MARC_NS,
            records=(
                "<record><leader>00000nam a2200000 a 4500</leader></record>"
                + _RECORD_TEMPLATE.format(control_id="ctrl-real", title="T")
            ),
        ),
        encoding="utf-8",
    )

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.marcs_written == 1
    assert _control_numbers(tmp_path / "vault_marcs.xml") == ["ctrl-real"]


def test_non_001_controlfields_are_passed_over(tmp_path: Path) -> None:
    """A record whose first controlfield is 008 (not 001) still has its 001 found."""
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-real", "uuid-1"))
    shard = pool / "eng" / "shard.xml"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        _MARCXML_TEMPLATE.format(
            ns=_MARC_NS,
            records=(
                "<record>"
                "<leader>00000nam a2200000 a 4500</leader>"
                '<controlfield tag="008">750101s1953    xxu           000 0 eng d</controlfield>'
                '<controlfield tag="001">ctrl-real</controlfield>'
                "</record>"
            ),
        ),
        encoding="utf-8",
    )

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.marcs_written == 1
    assert _control_numbers(tmp_path / "vault_marcs.xml") == ["ctrl-real"]


def test_001_with_empty_text_is_treated_as_missing(tmp_path: Path) -> None:
    """A self-closing or whitespace-only 001 yields None and the record is skipped."""
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-real", "uuid-1"))
    shard = pool / "eng" / "shard.xml"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        _MARCXML_TEMPLATE.format(
            ns=_MARC_NS,
            records=(
                "<record>"
                "<leader>00000nam a2200000 a 4500</leader>"
                '<controlfield tag="001"/>'
                "</record>" + _RECORD_TEMPLATE.format(control_id="ctrl-real", title="T")
            ),
        ),
        encoding="utf-8",
    )

    report = dump_vault_marcs(vault, pool, tmp_path / "vault_marcs.xml")

    assert report.marcs_written == 1
    assert _control_numbers(tmp_path / "vault_marcs.xml") == ["ctrl-real"]


def test_creates_output_parent_directory(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    upsert_entry(vault, _entry("ctrl-a", "uuid-a"))
    _write_shard(pool / "eng" / "shard.xml", [("ctrl-a", "A")])

    out = tmp_path / "nested" / "deeper" / "vault_marcs.xml"
    report = dump_vault_marcs(vault, pool, out)

    assert out.exists()
    assert out.parent.is_dir()
    assert report.marcs_written == 1


def test_cli_dump_vault_marcs_command_writes_collection(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    pool = tmp_path / "pool"
    out = tmp_path / "vault_marcs.xml"
    upsert_entry(vault, _entry("ctrl-x", "uuid-x"))
    _write_shard(pool / "eng" / "shard.xml", [("ctrl-x", "X")])

    result = _RUNNER.invoke(
        app,
        [
            "dump-vault-marcs",
            "--vault",
            str(vault),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--log-file",
            str(tmp_path / "run.log"),
        ],
    )

    assert result.exit_code == 0
    assert "wrote 1 records" in result.stdout
    assert _control_numbers(out) == ["ctrl-x"]
