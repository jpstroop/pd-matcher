"""Unit tests for the publish-linkage command."""

from json import loads
from pathlib import Path

from typer.testing import CliRunner

from pd_groundtruth.cli import app
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import upsert_entry
from pd_groundtruth.publish_linkage import publish_linkage

_RUNNER = CliRunner()


def _entry(
    *,
    marc_control_id: str = "ctrl-1",
    nypl_uuid: str = "uuid-1",
    verdict: str = "match",
    note: str | None = None,
    labeled_at: str = "2026-05-31T12:00:00+00:00",
    labeler: str = "jpstroop",
    lccn: str | None = "40012345",
    oclc: str | None = "ocm12345",
    isbns: tuple[str, ...] = ("9780000000000",),
    cce_regnum: str | None = "A12345",
    cce_renewal_id: str | None = "R67890",
    cce_renewal_oreg: str | None = "A12345",
) -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=marc_control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        note=note,
        labeled_at=labeled_at,
        labeler=labeler,
        marc_identifiers=MarcIdentifiers(lccn=lccn, oclc=oclc, isbns=isbns),
        cce_regnum=cce_regnum,
        cce_renewal_id=cce_renewal_id,
        cce_renewal_oreg=cce_renewal_oreg,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_emits_one_row_per_vault_entry(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(vault, _entry(marc_control_id="m1", nypl_uuid="u1"))
    upsert_entry(vault, _entry(marc_control_id="m2", nypl_uuid="u2", verdict="no_match"))
    upsert_entry(vault, _entry(marc_control_id="m3", nypl_uuid="u3", verdict="unsure"))

    report = publish_linkage(vault, out)

    assert report.rows_written == 3
    assert report.matches == 1
    assert report.no_matches == 1
    assert report.unsures == 1
    rows = _rows(out)
    assert len(rows) == 3


def test_field_order_leads_with_universal_identifiers(tmp_path: Path) -> None:
    """Universal IDs lead; marc_control_id is at the tail."""
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(vault, _entry())

    publish_linkage(vault, out)

    raw = out.read_text(encoding="utf-8").strip()
    expected_prefix = '{"lccn":"40012345","oclc":"ocm12345","isbns":["9780000000000"]'
    assert raw.startswith(expected_prefix)
    assert raw.endswith('"marc_control_id":"ctrl-1"}')


def test_note_is_stripped_from_output(tmp_path: Path) -> None:
    """The labeler's free-text note must not appear in published output."""
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(vault, _entry(note="series-level CCE, see series titled X"))

    publish_linkage(vault, out)

    raw = out.read_text(encoding="utf-8")
    assert "note" not in raw
    assert "series-level" not in raw


def test_null_universal_identifiers_serialize_as_null(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(vault, _entry(lccn=None, oclc=None, isbns=()))

    publish_linkage(vault, out)

    row = _rows(out)[0]
    assert row["lccn"] is None
    assert row["oclc"] is None
    assert row["isbns"] == []


def test_null_cce_renewal_fields_serialize_as_null(tmp_path: Path) -> None:
    """An unrenewed registration has cce_renewal_id and cce_renewal_oreg null."""
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(vault, _entry(cce_renewal_id=None, cce_renewal_oreg=None))

    publish_linkage(vault, out)

    row = _rows(out)[0]
    assert row["cce_regnum"] == "A12345"
    assert row["cce_renewal_id"] is None
    assert row["cce_renewal_oreg"] is None


def test_rows_emitted_in_labeled_at_ascending_order(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(
        vault,
        _entry(marc_control_id="late", nypl_uuid="late", labeled_at="2026-05-31T15:00:00+00:00"),
    )
    upsert_entry(
        vault,
        _entry(marc_control_id="early", nypl_uuid="early", labeled_at="2026-05-31T08:00:00+00:00"),
    )
    upsert_entry(
        vault,
        _entry(
            marc_control_id="middle",
            nypl_uuid="middle",
            labeled_at="2026-05-31T12:00:00+00:00",
        ),
    )

    publish_linkage(vault, out)

    control_ids = [row["marc_control_id"] for row in _rows(out)]
    assert control_ids == ["early", "middle", "late"]


def test_empty_vault_produces_empty_output(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"

    report = publish_linkage(vault, out)

    assert report.rows_written == 0
    assert report.matches == 0
    assert report.no_matches == 0
    assert report.unsures == 0
    assert out.read_text(encoding="utf-8") == ""


def test_creates_output_parent_directory(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "nested" / "deeper" / "published.jsonl"
    upsert_entry(vault, _entry())

    publish_linkage(vault, out)

    assert out.exists()


def test_tmp_file_cleaned_up_on_success(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(vault, _entry())

    publish_linkage(vault, out)

    assert not out.with_name(out.name + ".tmp").exists()


def test_cli_publish_linkage_writes_jsonl(tmp_path: Path) -> None:
    vault = tmp_path / "vault.jsonl"
    out = tmp_path / "published.jsonl"
    upsert_entry(vault, _entry(marc_control_id="m1", nypl_uuid="u1"))
    upsert_entry(vault, _entry(marc_control_id="m2", nypl_uuid="u2", verdict="no_match"))

    result = _RUNNER.invoke(
        app,
        [
            "publish-linkage",
            "--vault",
            str(vault),
            "--out",
            str(out),
            "--log-file",
            str(tmp_path / "run.log"),
        ],
    )

    assert result.exit_code == 0
    assert "wrote 2 rows" in result.stdout
    assert "matches=1" in result.stdout
    assert "no_matches=1" in result.stdout
    rows = _rows(out)
    assert len(rows) == 2
