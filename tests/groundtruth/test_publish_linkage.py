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


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return ``(vault, training, matches)`` paths."""
    return tmp_path / "vault.jsonl", tmp_path / "training.jsonl", tmp_path / "matches.jsonl"


def test_training_file_carries_every_adjudicated_verdict(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(marc_control_id="m1", nypl_uuid="u1"))
    upsert_entry(vault, _entry(marc_control_id="m2", nypl_uuid="u2", verdict="no_match"))
    upsert_entry(vault, _entry(marc_control_id="m3", nypl_uuid="u3", verdict="unsure"))

    report = publish_linkage(vault, training, matches)

    assert report.rows_written == 3
    assert report.matches == 1
    assert report.no_matches == 1
    assert report.unsures == 1
    assert len(_rows(training)) == 3


def test_matches_file_carries_only_match_rows(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(marc_control_id="m1", nypl_uuid="u1", verdict="match"))
    upsert_entry(vault, _entry(marc_control_id="m2", nypl_uuid="u2", verdict="no_match"))
    upsert_entry(vault, _entry(marc_control_id="m3", nypl_uuid="u3", verdict="match"))
    upsert_entry(vault, _entry(marc_control_id="m4", nypl_uuid="u4", verdict="unsure"))

    publish_linkage(vault, training, matches)

    matches_rows = _rows(matches)
    assert len(matches_rows) == 2
    assert {row["verdict"] for row in matches_rows} == {"match"}


def test_matches_file_uses_identical_row_schema_to_training(tmp_path: Path) -> None:
    """A match row appearing in both files is byte-identical."""
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(verdict="match"))

    publish_linkage(vault, training, matches)

    assert training.read_bytes() == matches.read_bytes()


def test_field_order_leads_with_universal_identifiers(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry())

    publish_linkage(vault, training, matches)

    raw = training.read_text(encoding="utf-8").strip()
    expected_prefix = '{"lccn":"40012345","oclc":"ocm12345","isbns":["9780000000000"]'
    assert raw.startswith(expected_prefix)
    assert raw.endswith('"marc_control_id":"ctrl-1"}')


def test_note_is_stripped_from_both_files(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(note="series-level CCE, see series titled X"))

    publish_linkage(vault, training, matches)

    for path in (training, matches):
        raw = path.read_text(encoding="utf-8")
        assert "note" not in raw
        assert "series-level" not in raw


def test_null_universal_identifiers_serialize_as_null(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(lccn=None, oclc=None, isbns=()))

    publish_linkage(vault, training, matches)

    row = _rows(training)[0]
    assert row["lccn"] is None
    assert row["oclc"] is None
    assert row["isbns"] == []


def test_null_cce_renewal_fields_serialize_as_null(tmp_path: Path) -> None:
    """An unrenewed registration has cce_renewal_id and cce_renewal_oreg null."""
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(cce_renewal_id=None, cce_renewal_oreg=None))

    publish_linkage(vault, training, matches)

    row = _rows(training)[0]
    assert row["cce_regnum"] == "A12345"
    assert row["cce_renewal_id"] is None
    assert row["cce_renewal_oreg"] is None


def test_rows_emitted_in_labeled_at_ascending_order(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
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

    publish_linkage(vault, training, matches)

    control_ids = [row["marc_control_id"] for row in _rows(training)]
    assert control_ids == ["early", "middle", "late"]


def test_empty_vault_produces_empty_outputs(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)

    report = publish_linkage(vault, training, matches)

    assert report.rows_written == 0
    assert report.matches == 0
    assert training.read_text(encoding="utf-8") == ""
    assert matches.read_text(encoding="utf-8") == ""


def test_vault_with_no_matches_produces_empty_matches_file(tmp_path: Path) -> None:
    """A vault holding only no_match + unsure verdicts produces an empty matches file."""
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(marc_control_id="m1", nypl_uuid="u1", verdict="no_match"))
    upsert_entry(vault, _entry(marc_control_id="m2", nypl_uuid="u2", verdict="unsure"))

    publish_linkage(vault, training, matches)

    assert len(_rows(training)) == 2
    assert matches.read_text(encoding="utf-8") == ""


def test_creates_output_parent_directories(tmp_path: Path) -> None:
    """Each output file's parent directory is created if missing."""
    vault = tmp_path / "vault.jsonl"
    training = tmp_path / "nested" / "training" / "training.jsonl"
    matches = tmp_path / "other" / "matches" / "matches.jsonl"
    upsert_entry(vault, _entry())

    publish_linkage(vault, training, matches)

    assert training.exists()
    assert matches.exists()


def test_tmp_files_cleaned_up_on_success(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry())

    publish_linkage(vault, training, matches)

    assert not training.with_name(training.name + ".tmp").exists()
    assert not matches.with_name(matches.name + ".tmp").exists()


def test_cli_publish_linkage_writes_both_files(tmp_path: Path) -> None:
    vault, training, matches = _paths(tmp_path)
    upsert_entry(vault, _entry(marc_control_id="m1", nypl_uuid="u1"))
    upsert_entry(vault, _entry(marc_control_id="m2", nypl_uuid="u2", verdict="no_match"))

    result = _RUNNER.invoke(
        app,
        [
            "publish-linkage",
            "--vault",
            str(vault),
            "--training-out",
            str(training),
            "--matches-out",
            str(matches),
            "--log-file",
            str(tmp_path / "run.log"),
        ],
    )

    assert result.exit_code == 0
    assert "wrote 2 rows" in result.stdout
    assert "1 matches also written" in result.stdout
    assert len(_rows(training)) == 2
    assert len(_rows(matches)) == 1
