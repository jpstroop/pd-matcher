"""Unit tests for the ``seed-vault`` Typer command."""

from pathlib import Path

from msgspec.json import encode as json_encode
from typer.testing import CliRunner

from pd_groundtruth.cli import app
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_matcher.models import MarcRecord

_RUNNER = CliRunner()


def _marc_json(control_id: str) -> str:
    return json_encode(
        MarcRecord(
            control_id=control_id,
            title="A Title",
            title_main="A Title",
            lccn="40012345",
            oclc="0001",
            isbns=("9780000000000",),
        )
    ).decode("utf-8")


def _pair(control_id: str, nypl_uuid: str) -> PairInsert:
    return PairInsert(
        language="eng",
        decade=1950,
        score=0.95,
        band="ge90",
        source="banded",
        marc_control_id=control_id,
        marc_json=_marc_json(control_id),
        marc_title="A Title",
        marc_author=None,
        marc_publisher=None,
        marc_year=1953,
        nypl_uuid=nypl_uuid,
        cce_title="CCE Title",
        cce_author=None,
        cce_publishers=None,
        cce_claimants=None,
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json="{}",
    )


def test_seed_vault_dumps_all_current_labels(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    vault_path = tmp_path / "vault.jsonl"
    with ReviewDb.connect(db_path) as db:
        a = db.insert_pair(_pair("ctrl-a", "uuid-a"))
        b = db.insert_pair(_pair("ctrl-b", "uuid-b"))
        db.add_label(a, VERDICT_MATCH)
        db.add_label(b, VERDICT_NO_MATCH, note="off")

    result = _RUNNER.invoke(app, ["seed-vault", "--db", str(db_path), "--vault", str(vault_path)])
    assert result.exit_code == 0
    assert "seeded 2 labels" in result.stdout
    assert "skipped 0 already-present" in result.stdout

    latest = current_entries(vault_path)
    assert set(latest.keys()) == {("ctrl-a", "uuid-a"), ("ctrl-b", "uuid-b")}
    assert latest[("ctrl-b", "uuid-b")].note == "off"
    assert latest[("ctrl-a", "uuid-a")].marc_identifiers.lccn == "40012345"
    assert latest[("ctrl-a", "uuid-a")].marc_identifiers.oclc == "0001"
    assert latest[("ctrl-a", "uuid-a")].marc_identifiers.isbns == ("9780000000000",)


def test_seed_vault_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    vault_path = tmp_path / "vault.jsonl"
    with ReviewDb.connect(db_path) as db:
        pair_id = db.insert_pair(_pair("ctrl-a", "uuid-a"))
        db.add_label(pair_id, VERDICT_MATCH)

    first = _RUNNER.invoke(app, ["seed-vault", "--db", str(db_path), "--vault", str(vault_path)])
    assert first.exit_code == 0
    assert "seeded 1 labels" in first.stdout

    second = _RUNNER.invoke(app, ["seed-vault", "--db", str(db_path), "--vault", str(vault_path)])
    assert second.exit_code == 0
    assert "seeded 0 labels" in second.stdout
    assert "skipped 1 already-present" in second.stdout

    assert len(list(vault_path.read_text(encoding="utf-8").splitlines())) == 1


def test_seed_vault_appends_when_relabeled(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    vault_path = tmp_path / "vault.jsonl"
    with ReviewDb.connect(db_path) as db:
        pair_id = db.insert_pair(_pair("ctrl-a", "uuid-a"))
        db.add_label(pair_id, VERDICT_MATCH)

    first = _RUNNER.invoke(app, ["seed-vault", "--db", str(db_path), "--vault", str(vault_path)])
    assert first.exit_code == 0

    with ReviewDb.connect(db_path) as db:
        db.add_label(1, VERDICT_NO_MATCH, note="changed mind")

    second = _RUNNER.invoke(app, ["seed-vault", "--db", str(db_path), "--vault", str(vault_path)])
    assert second.exit_code == 0
    assert "seeded 1 labels" in second.stdout

    latest = current_entries(vault_path)
    assert latest[("ctrl-a", "uuid-a")].verdict == "no_match"
    assert latest[("ctrl-a", "uuid-a")].note == "changed mind"
    lines = vault_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_seed_vault_on_empty_db_writes_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    vault_path = tmp_path / "vault.jsonl"
    with ReviewDb.connect(db_path):
        pass

    result = _RUNNER.invoke(app, ["seed-vault", "--db", str(db_path), "--vault", str(vault_path)])
    assert result.exit_code == 0
    assert "seeded 0 labels" in result.stdout
    assert not vault_path.exists()
