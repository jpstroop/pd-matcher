"""Unit tests for the typed SQLite review-database wrapper."""

from pathlib import Path

from pytest import raises

from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.review_db import VERDICT_UNSURE
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb


def _pair(
    *,
    language: str = "eng",
    band: str = "ge90",
    source: str = "banded",
    control_id: str = "ctrl-1",
    nypl_uuid: str = "uuid-1",
    score: float = 0.95,
    was_renewed: bool | None = True,
) -> PairInsert:
    return PairInsert(
        language=language,
        decade=1950,
        score=score,
        band=band,
        source=source,
        marc_control_id=control_id,
        marc_json='{"control_id": "ctrl-1"}',
        marc_title="A Title",
        marc_author="An Author",
        marc_publisher="A Publisher",
        marc_year=1953,
        nypl_uuid=nypl_uuid,
        cce_title="CCE Title",
        cce_author="CCE Author",
        cce_publishers="Pub A | Pub B",
        cce_claimants="Claimant A",
        cce_reg_year=1953,
        cce_was_renewed=was_renewed,
        cce_regnum="R12345",
        evidence_json='{"title.token_set": 0.91}',
    )


def test_connect_creates_schema_and_returns_empty_counts(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        assert db.stratum_counts() == {}


def test_insert_pair_returns_incrementing_ids(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        assert first == 1
        assert second == 2


def test_stratum_counts_groups_by_language_and_band(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair(language="eng", band="ge90", control_id="a"))
        db.insert_pair(_pair(language="eng", band="ge90", control_id="b"))
        db.insert_pair(_pair(language="fre", band="below", control_id="c"))
        counts = db.stratum_counts()
    assert counts[("eng", "ge90")] == 2
    assert counts[("fre", "below")] == 1


def test_was_renewed_stored_as_integer_flag(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair(control_id="t", nypl_uuid="u-t", was_renewed=True))
        db.insert_pair(_pair(control_id="f", nypl_uuid="u-f", was_renewed=False))
        db.insert_pair(_pair(control_id="n", nypl_uuid="u-n", was_renewed=None))
        renewed = db.next_unlabeled()
        assert renewed is not None
        assert renewed.cce_was_renewed == 1
        db.add_label(renewed.id, VERDICT_MATCH)
        falsy = db.next_unlabeled()
        assert falsy is not None
        assert falsy.cce_was_renewed == 0
        db.add_label(falsy.id, VERDICT_NO_MATCH)
        none_row = db.next_unlabeled()
        assert none_row is not None
        assert none_row.cce_was_renewed is None


def test_next_unlabeled_returns_lowest_id_unlabeled(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        row = db.next_unlabeled()
        assert row is not None
        assert row.id == first
        assert row.marc_control_id == "a"


def test_next_unlabeled_skips_labeled_pairs(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(first, VERDICT_MATCH)
        row = db.next_unlabeled()
        assert row is not None
        assert row.id == second


def test_next_unlabeled_returns_none_when_all_labeled(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        db.add_label(pair_id, VERDICT_MATCH)
        assert db.next_unlabeled() is None


def test_next_unlabeled_filters_by_language_and_band(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair(language="eng", band="ge90", control_id="a", nypl_uuid="u-a"))
        fre_id = db.insert_pair(
            _pair(language="fre", band="below", control_id="b", nypl_uuid="u-b")
        )
        row = db.next_unlabeled(language="fre")
        assert row is not None
        assert row.id == fre_id
        banded = db.next_unlabeled(band="below")
        assert banded is not None
        assert banded.id == fre_id
        none_row = db.next_unlabeled(language="ger")
        assert none_row is None


def test_add_label_keeps_history_across_relabels(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        first_label = db.add_label(pair_id, VERDICT_MATCH, note="initial")
        second_label = db.add_label(pair_id, VERDICT_NO_MATCH, note="corrected")
        assert second_label > first_label
        assert db.next_unlabeled() is None


def test_add_label_rejects_invalid_verdict(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        with raises(ValueError, match="invalid verdict"):
            db.add_label(pair_id, "maybe")


def test_commit_persists_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "review.db"
    with ReviewDb.connect(path) as db:
        db.insert_pair(_pair())
        db.commit()
    with ReviewDb.connect(path) as reopened:
        assert reopened.stratum_counts()[("eng", "ge90")] == 1


def test_get_pair_returns_row_or_none(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair(control_id="a"))
        row = db.get_pair(pair_id)
        assert row is not None
        assert row.id == pair_id
        assert row.marc_control_id == "a"
        assert db.get_pair(9999) is None


def test_progress_empty_db_is_all_zero(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        counts = db.progress()
    assert counts.total == 0
    assert counts.labeled == 0
    assert counts.remaining == 0
    assert counts.match == 0
    assert counts.no_match == 0
    assert counts.unsure == 0
    assert counts.by_language == ()


def test_progress_counts_current_verdicts_and_remaining(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(language="eng", control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(language="eng", control_id="b", nypl_uuid="u-b"))
        db.insert_pair(_pair(language="fre", control_id="c", nypl_uuid="u-c"))
        db.add_label(a, VERDICT_MATCH)
        db.add_label(b, VERDICT_UNSURE)
        counts = db.progress()
    assert counts.total == 3
    assert counts.labeled == 2
    assert counts.remaining == 1
    assert counts.match == 1
    assert counts.unsure == 1
    assert counts.no_match == 0


def test_progress_relabel_uses_latest_verdict(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        db.add_label(pair_id, VERDICT_MATCH)
        db.add_label(pair_id, VERDICT_NO_MATCH)
        counts = db.progress()
    assert counts.labeled == 1
    assert counts.match == 0
    assert counts.no_match == 1


def test_progress_per_language_totals_and_labeled(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        eng_a = db.insert_pair(_pair(language="eng", control_id="a", nypl_uuid="u-a"))
        db.insert_pair(_pair(language="eng", control_id="b", nypl_uuid="u-b"))
        db.insert_pair(_pair(language="fre", control_id="c", nypl_uuid="u-c"))
        db.add_label(eng_a, VERDICT_MATCH)
        by_language = {lang.language: lang for lang in db.progress().by_language}
    assert by_language["eng"].total == 2
    assert by_language["eng"].labeled == 1
    assert by_language["fre"].total == 1
    assert by_language["fre"].labeled == 0


def test_round_trip_preserves_all_columns(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair())
        row = db.next_unlabeled()
    assert row is not None
    assert row.language == "eng"
    assert row.decade == 1950
    assert row.marc_title == "A Title"
    assert row.cce_publishers == "Pub A | Pub B"
    assert row.cce_regnum == "R12345"
    assert row.evidence_json == '{"title.token_set": 0.91}'
    assert row.created_at != ""


def test_previous_labeled_none_when_nothing_labeled(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        assert db.previous_labeled() is None


def test_previous_labeled_returns_most_recent_action(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(first, VERDICT_MATCH)
        db.add_label(second, VERDICT_NO_MATCH)
        back = db.previous_labeled()
        assert back is not None
        assert back.id == second


def test_previous_labeled_chains_backward_with_before(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(first, VERDICT_MATCH)
        db.add_label(second, VERDICT_NO_MATCH)
        step_back = db.previous_labeled(before=second)
        assert step_back is not None
        assert step_back.id == first
        assert db.previous_labeled(before=first) is None


def test_previous_labeled_follows_relabel_to_front(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(first, VERDICT_MATCH)
        db.add_label(second, VERDICT_NO_MATCH)
        db.add_label(first, VERDICT_UNSURE)
        back = db.previous_labeled()
        assert back is not None
        assert back.id == first


def test_previous_labeled_respects_language_filter(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        eng = db.insert_pair(_pair(language="eng", control_id="a", nypl_uuid="u-a"))
        fre = db.insert_pair(_pair(language="fre", control_id="b", nypl_uuid="u-b"))
        db.add_label(eng, VERDICT_MATCH)
        db.add_label(fre, VERDICT_MATCH)
        back = db.previous_labeled(language="eng")
        assert back is not None
        assert back.id == eng


def test_add_label_stores_reason_and_note(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        db.add_label(pair, VERDICT_NO_MATCH, note="looks off", reason="diff_work")
    with ReviewDb.connect(tmp_path / "review.db") as db:
        assert db.reason_counts() == {(VERDICT_NO_MATCH, "diff_work"): 1}


def test_reason_counts_uses_only_current_label(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        db.add_label(pair, VERDICT_NO_MATCH, reason="diff_work")
        db.add_label(pair, VERDICT_UNSURE, reason="multiple_candidates")
        assert db.reason_counts() == {(VERDICT_UNSURE, "multiple_candidates"): 1}


def test_reason_counts_ignores_null_reasons(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(first, VERDICT_MATCH)
        db.add_label(second, VERDICT_NO_MATCH, reason="garbled")
        assert db.reason_counts() == {(VERDICT_NO_MATCH, "garbled"): 1}


def test_init_schema_migrates_label_table_missing_reason(tmp_path: Path) -> None:
    from sqlite3 import connect as sqlite_connect

    db_path = tmp_path / "legacy.db"
    legacy = sqlite_connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE label (
            id INTEGER PRIMARY KEY,
            pair_id INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            note TEXT,
            labeled_at TEXT NOT NULL
        );
        INSERT INTO label (pair_id, verdict, note, labeled_at)
        VALUES (1, 'match', NULL, '2026-01-01T00:00:00+00:00');
        """
    )
    legacy.commit()
    legacy.close()

    with ReviewDb.connect(db_path) as db:
        columns = {row[1] for row in db._conn.execute("PRAGMA table_info(label)")}
        assert "reason" in columns
        existing = db._conn.execute("SELECT COUNT(*) FROM label").fetchone()[0]
        assert existing == 1
