"""Unit tests for the typed SQLite review-database wrapper."""

from pathlib import Path

from pytest import raises

from pd_groundtruth.review_db import PAIRING_REGISTRATION
from pd_groundtruth.review_db import SORT_ASC
from pd_groundtruth.review_db import SORT_DESC
from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.review_db import VERDICT_UNSURE
from pd_groundtruth.review_db import LabelFilters
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


def test_next_unlabeled_excludes_supplied_pair_ids(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        third = db.insert_pair(_pair(control_id="c", nypl_uuid="u-c"))
        row = db.next_unlabeled(exclude_pair_ids=(first,))
        assert row is not None
        assert row.id == second
        row = db.next_unlabeled(exclude_pair_ids=(first, second))
        assert row is not None
        assert row.id == third
        assert db.next_unlabeled(exclude_pair_ids=(first, second, third)) is None


def test_next_unlabeled_exclude_combines_with_language_filter(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        eng_a = db.insert_pair(_pair(language="eng", control_id="a", nypl_uuid="u-a"))
        eng_b = db.insert_pair(_pair(language="eng", control_id="b", nypl_uuid="u-b"))
        db.insert_pair(_pair(language="fre", control_id="c", nypl_uuid="u-c"))
        row = db.next_unlabeled(language="eng", exclude_pair_ids=(eng_a,))
        assert row is not None
        assert row.id == eng_b


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
        assert second_label.label_id > first_label.label_id
        assert second_label.labeled_at != ""
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


def test_round_trip_preserves_extended_cce_columns(tmp_path: Path) -> None:
    extended = PairInsert(
        language="eng",
        decade=1950,
        score=0.91,
        band="ge90",
        source="banded",
        marc_control_id="ctrl-x",
        marc_json="{}",
        marc_title="t",
        marc_author="a",
        marc_publisher="p",
        marc_year=1953,
        nypl_uuid="uuid-x",
        cce_title="CCE Title",
        cce_author="CCE Author",
        cce_publishers="Pub A | Pub B",
        cce_claimants="Claimant A",
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R12345",
        evidence_json="{}",
        cce_edition="2nd ed.",
        cce_publication_places="New York; London",
        cce_author_place="Cambridge, Mass.",
        cce_author_is_claimant=True,
        cce_copies="2c.",
        cce_aff_date="1953-06-01",
        cce_desc="vi, 200 p.",
        cce_notes="note one\nnote two",
        cce_new_matter_claimed="added ch. 5",
        cce_copy_date="1953-04-01",
        cce_notice_date="1953-04-02",
        cce_lccn="28000854",
        cce_prev_regnums="A100000; A200000",
    )
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(extended)
        row = db.next_unlabeled()
    assert row is not None
    assert row.cce_edition == "2nd ed."
    assert row.cce_publication_places == "New York; London"
    assert row.cce_author_place == "Cambridge, Mass."
    assert row.cce_author_is_claimant == 1
    assert row.cce_copies == "2c."
    assert row.cce_aff_date == "1953-06-01"
    assert row.cce_desc == "vi, 200 p."
    assert row.cce_notes == "note one\nnote two"
    assert row.cce_new_matter_claimed == "added ch. 5"
    assert row.cce_copy_date == "1953-04-01"
    assert row.cce_notice_date == "1953-04-02"
    assert row.cce_lccn == "28000854"
    assert row.cce_prev_regnums == "A100000; A200000"


def test_extended_cce_columns_default_to_null(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair())
        row = db.next_unlabeled()
    assert row is not None
    assert row.cce_edition is None
    assert row.cce_publication_places is None
    assert row.cce_author_place is None
    assert row.cce_author_is_claimant is None
    assert row.cce_copies is None
    assert row.cce_aff_date is None
    assert row.cce_desc is None
    assert row.cce_notes is None
    assert row.cce_new_matter_claimed is None
    assert row.cce_copy_date is None
    assert row.cce_notice_date is None
    assert row.cce_lccn is None
    assert row.cce_prev_regnums is None


def test_cce_author_is_claimant_stored_as_integer_flag(tmp_path: Path) -> None:
    def _build(value: bool | None, control_id: str, uuid: str) -> PairInsert:
        base = _pair(control_id=control_id, nypl_uuid=uuid)
        return PairInsert(
            language=base.language,
            decade=base.decade,
            score=base.score,
            band=base.band,
            source=base.source,
            marc_control_id=base.marc_control_id,
            marc_json=base.marc_json,
            marc_title=base.marc_title,
            marc_author=base.marc_author,
            marc_publisher=base.marc_publisher,
            marc_year=base.marc_year,
            nypl_uuid=base.nypl_uuid,
            cce_title=base.cce_title,
            cce_author=base.cce_author,
            cce_publishers=base.cce_publishers,
            cce_claimants=base.cce_claimants,
            cce_reg_year=base.cce_reg_year,
            cce_was_renewed=base.cce_was_renewed,
            cce_regnum=base.cce_regnum,
            evidence_json=base.evidence_json,
            cce_author_is_claimant=value,
        )

    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_build(True, "t", "u-t"))
        db.insert_pair(_build(False, "f", "u-f"))
        db.insert_pair(_build(None, "n", "u-n"))
        truthy = db.next_unlabeled()
        assert truthy is not None
        assert truthy.cce_author_is_claimant == 1
        db.add_label(truthy.id, VERDICT_MATCH)
        falsy = db.next_unlabeled()
        assert falsy is not None
        assert falsy.cce_author_is_claimant == 0
        db.add_label(falsy.id, VERDICT_MATCH)
        none_row = db.next_unlabeled()
        assert none_row is not None
        assert none_row.cce_author_is_claimant is None


def test_init_schema_adds_extended_cce_columns_to_legacy_pair_table(tmp_path: Path) -> None:
    from sqlite3 import connect as sqlite_connect

    db_path = tmp_path / "legacy_pair.db"
    legacy = sqlite_connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE review_pair (
            id INTEGER PRIMARY KEY,
            language TEXT NOT NULL,
            decade INTEGER,
            score REAL NOT NULL,
            band TEXT NOT NULL,
            source TEXT NOT NULL,
            marc_control_id TEXT NOT NULL,
            marc_json TEXT NOT NULL,
            marc_title TEXT,
            marc_author TEXT,
            marc_publisher TEXT,
            marc_year INTEGER,
            nypl_uuid TEXT NOT NULL,
            cce_title TEXT,
            cce_author TEXT,
            cce_publishers TEXT,
            cce_claimants TEXT,
            cce_reg_year INTEGER,
            cce_was_renewed INTEGER,
            cce_regnum TEXT,
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO review_pair (
            language, decade, score, band, source, marc_control_id, marc_json,
            marc_title, marc_author, marc_publisher, marc_year, nypl_uuid,
            cce_title, cce_author, cce_publishers, cce_claimants, cce_reg_year,
            cce_was_renewed, cce_regnum, evidence_json, created_at
        ) VALUES (
            'eng', 1950, 0.9, 'ge90', 'banded', 'ctrl-legacy', '{}',
            't', 'a', 'p', 1953, 'uuid-legacy',
            'CCE', 'CCE Author', 'Pub', 'Claim', 1953,
            1, 'R1', '{}', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    legacy.commit()
    legacy.close()

    with ReviewDb.connect(db_path) as db:
        columns = {row[1] for row in db._conn.execute("PRAGMA table_info(review_pair)")}
        assert "cce_edition" in columns
        assert "cce_publication_places" in columns
        assert "cce_author_place" in columns
        assert "cce_author_is_claimant" in columns
        assert "cce_copies" in columns
        assert "cce_aff_date" in columns
        assert "cce_desc" in columns
        assert "cce_notes" in columns
        assert "cce_new_matter_claimed" in columns
        assert "cce_copy_date" in columns
        assert "cce_notice_date" in columns
        assert "cce_lccn" in columns
        assert "cce_prev_regnums" in columns
        row = db.get_pair(1)
    assert row is not None
    assert row.cce_edition is None
    assert row.cce_author_is_claimant is None
    assert row.cce_lccn is None
    assert row.cce_prev_regnums is None


def test_get_current_label_returns_none_for_unlabeled_pair(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        assert db.get_current_label(pair_id) is None


def test_get_current_label_returns_none_for_missing_pair(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        assert db.get_current_label(9999) is None


def test_get_current_label_returns_latest_verdict_and_note(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        db.add_label(pair_id, VERDICT_MATCH, note="first take")
        db.add_label(pair_id, VERDICT_NO_MATCH, note="changed my mind")
        current = db.get_current_label(pair_id)
    assert current is not None
    assert current.verdict == VERDICT_NO_MATCH
    assert current.note == "changed my mind"
    assert current.pair_id == pair_id
    assert current.marc_control_id == "a"
    assert current.nypl_uuid == "u-a"


def test_get_current_label_isolated_to_requested_pair_id(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(a, VERDICT_MATCH, note="for a")
        db.add_label(b, VERDICT_NO_MATCH, note="for b")
        current_a = db.get_current_label(a)
        current_b = db.get_current_label(b)
    assert current_a is not None
    assert current_a.note == "for a"
    assert current_b is not None
    assert current_b.note == "for b"


def _raise_after_insert(db_path: Path) -> None:
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        raise RuntimeError("boom")


def test_connect_context_manager_swallows_commit_when_exception_raised(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with raises(RuntimeError, match="boom"):
        _raise_after_insert(db_path)
    with ReviewDb.connect(db_path) as db:
        assert db.stratum_counts() == {}


def test_add_label_stores_note(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        db.add_label(pair, VERDICT_NO_MATCH, note="looks off")
    with ReviewDb.connect(tmp_path / "review.db") as db:
        [label] = list(db.iter_current_labels())
        assert label.note == "looks off"
        assert label.verdict == VERDICT_NO_MATCH


def test_iter_current_labels_yields_only_latest_per_pair(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair(control_id="ctrl-a", nypl_uuid="uuid-a"))
        db.add_label(pair_id, VERDICT_MATCH)
        db.add_label(pair_id, VERDICT_NO_MATCH, note="changed")
    with ReviewDb.connect(tmp_path / "review.db") as db:
        labels = list(db.iter_current_labels())
    assert len(labels) == 1
    assert labels[0].verdict == VERDICT_NO_MATCH
    assert labels[0].note == "changed"
    assert labels[0].marc_control_id == "ctrl-a"
    assert labels[0].nypl_uuid == "uuid-a"


def test_iter_current_labels_skips_unlabeled_pairs(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(control_id="ctrl-a", nypl_uuid="uuid-a"))
        db.insert_pair(_pair(control_id="ctrl-b", nypl_uuid="uuid-b"))
        db.add_label(a, VERDICT_MATCH)
    with ReviewDb.connect(tmp_path / "review.db") as db:
        labels = list(db.iter_current_labels())
    assert [label.marc_control_id for label in labels] == ["ctrl-a"]


def test_insert_existing_label_preserves_supplied_timestamp(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        db.insert_existing_label(
            pair_id=pair_id,
            verdict=VERDICT_MATCH,
            labeled_at="2024-01-01T00:00:00+00:00",
            note="from vault",
        )
    with ReviewDb.connect(tmp_path / "review.db") as db:
        labels = list(db.iter_current_labels())
    assert labels[0].labeled_at == "2024-01-01T00:00:00+00:00"
    assert labels[0].note == "from vault"


def test_insert_existing_label_rejects_invalid_verdict(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        with raises(ValueError, match="invalid verdict"):
            db.insert_existing_label(
                pair_id=pair_id,
                verdict="bogus",
                labeled_at="2024-01-01T00:00:00+00:00",
            )


def test_iter_labeled_pairs_returns_latest_label_per_pair(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        db.add_label(pair_id, VERDICT_MATCH)
        db.add_label(pair_id, VERDICT_NO_MATCH, note="later")
        rows = db.iter_labeled_pairs()
    assert len(rows) == 1
    assert rows[0].pair_id == pair_id
    assert rows[0].verdict == VERDICT_NO_MATCH
    assert rows[0].note == "later"


def test_iter_labeled_pairs_excludes_unlabeled_rows(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        labeled = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(labeled, VERDICT_MATCH)
        rows = db.iter_labeled_pairs()
    assert [row.pair_id for row in rows] == [labeled]


def test_iter_labeled_pairs_filters_by_verdict(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.add_label(a, VERDICT_MATCH)
        db.add_label(b, VERDICT_NO_MATCH)
        rows = db.iter_labeled_pairs(LabelFilters(verdict=VERDICT_MATCH))
    assert [row.pair_id for row in rows] == [a]


def test_iter_labeled_pairs_filters_by_language(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        eng = db.insert_pair(_pair(language="eng", control_id="a", nypl_uuid="u-a"))
        fre = db.insert_pair(_pair(language="fre", control_id="b", nypl_uuid="u-b"))
        db.add_label(eng, VERDICT_MATCH)
        db.add_label(fre, VERDICT_MATCH)
        rows = db.iter_labeled_pairs(LabelFilters(language="fre"))
    assert [row.pair_id for row in rows] == [fre]


def test_iter_labeled_pairs_filters_by_substring_against_marc_title(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        match_id = db.insert_pair(_pair(control_id="ctrl-x", nypl_uuid="u-a"))
        db.insert_pair(_pair(control_id="ctrl-y", nypl_uuid="u-b"))
        db.add_label(match_id, VERDICT_MATCH)
        db.add_label(2, VERDICT_MATCH)
        rows = db.iter_labeled_pairs(LabelFilters(q="A TITLE"))
    assert {row.pair_id for row in rows} == {match_id, 2}


def test_iter_labeled_pairs_filters_by_substring_against_control_id(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        target = db.insert_pair(_pair(control_id="needle-1", nypl_uuid="u-a"))
        db.insert_pair(_pair(control_id="other-2", nypl_uuid="u-b"))
        db.add_label(target, VERDICT_MATCH)
        db.add_label(2, VERDICT_MATCH)
        rows = db.iter_labeled_pairs(LabelFilters(q="needle"))
    assert [row.pair_id for row in rows] == [target]


def test_iter_labeled_pairs_pagination_slices_results(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        for i in range(5):
            pair_id = db.insert_pair(_pair(control_id=f"ctrl-{i}", nypl_uuid=f"u-{i}"))
            db.add_label(pair_id, VERDICT_MATCH)
        first_page = db.iter_labeled_pairs(page_size=2, page=1)
        second_page = db.iter_labeled_pairs(page_size=2, page=2)
        third_page = db.iter_labeled_pairs(page_size=2, page=3)
    assert len(first_page) == 2
    assert len(second_page) == 2
    assert len(third_page) == 1
    seen = {row.pair_id for row in first_page + second_page + third_page}
    assert seen == {1, 2, 3, 4, 5}


def test_iter_labeled_pairs_rejects_invalid_page_args(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        with raises(ValueError, match="page must be >= 1"):
            db.iter_labeled_pairs(page=0)
        with raises(ValueError, match="page_size must be >= 1"):
            db.iter_labeled_pairs(page_size=0)


def test_iter_labeled_pairs_default_sort_is_descending(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        oldest = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        middle = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        newest = db.insert_pair(_pair(control_id="c", nypl_uuid="u-c"))
        db.insert_existing_label(oldest, VERDICT_MATCH, "2024-01-01T00:00:00+00:00")
        db.insert_existing_label(middle, VERDICT_MATCH, "2024-06-01T00:00:00+00:00")
        db.insert_existing_label(newest, VERDICT_MATCH, "2024-12-01T00:00:00+00:00")
        rows = db.iter_labeled_pairs()
    assert [row.pair_id for row in rows] == [newest, middle, oldest]


def test_iter_labeled_pairs_sort_asc_returns_oldest_first(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        oldest = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        middle = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        newest = db.insert_pair(_pair(control_id="c", nypl_uuid="u-c"))
        db.insert_existing_label(oldest, VERDICT_MATCH, "2024-01-01T00:00:00+00:00")
        db.insert_existing_label(middle, VERDICT_MATCH, "2024-06-01T00:00:00+00:00")
        db.insert_existing_label(newest, VERDICT_MATCH, "2024-12-01T00:00:00+00:00")
        rows = db.iter_labeled_pairs(LabelFilters(sort=SORT_ASC))
    assert [row.pair_id for row in rows] == [oldest, middle, newest]


def test_iter_labeled_pairs_sort_desc_explicit_matches_default(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        db.insert_existing_label(a, VERDICT_MATCH, "2024-01-01T00:00:00+00:00")
        db.insert_existing_label(b, VERDICT_MATCH, "2024-12-01T00:00:00+00:00")
        default_rows = db.iter_labeled_pairs()
        explicit_rows = db.iter_labeled_pairs(LabelFilters(sort=SORT_DESC))
    assert [row.pair_id for row in default_rows] == [row.pair_id for row in explicit_rows]


def test_iter_labeled_pairs_sort_breaks_ties_by_pair_id(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        first = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        second = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        third = db.insert_pair(_pair(control_id="c", nypl_uuid="u-c"))
        shared = "2024-06-01T00:00:00+00:00"
        db.insert_existing_label(first, VERDICT_MATCH, shared)
        db.insert_existing_label(second, VERDICT_MATCH, shared)
        db.insert_existing_label(third, VERDICT_MATCH, shared)
        desc = db.iter_labeled_pairs()
        asc = db.iter_labeled_pairs(LabelFilters(sort=SORT_ASC))
    assert [row.pair_id for row in desc] == [third, second, first]
    assert [row.pair_id for row in asc] == [first, second, third]


def test_iter_labeled_pairs_rejects_invalid_sort(tmp_path: Path) -> None:
    with (
        ReviewDb.connect(tmp_path / "review.db") as db,
        raises(ValueError, match="invalid sort"),
    ):
        db.iter_labeled_pairs(LabelFilters(sort="sideways"))


def test_count_labeled_pairs_matches_iter_results_across_filters(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(language="eng", control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(language="eng", control_id="b", nypl_uuid="u-b"))
        c = db.insert_pair(_pair(language="fre", control_id="c", nypl_uuid="u-c"))
        db.add_label(a, VERDICT_MATCH)
        db.add_label(b, VERDICT_NO_MATCH)
        db.add_label(c, VERDICT_MATCH)
        for filters in (
            LabelFilters(),
            LabelFilters(verdict=VERDICT_MATCH),
            LabelFilters(language="eng"),
            LabelFilters(verdict=VERDICT_MATCH, language="eng"),
            LabelFilters(q="title"),
        ):
            assert db.count_labeled_pairs(filters) == len(db.iter_labeled_pairs(filters)), filters


def test_round_trip_preserves_renewal_details(tmp_path: Path) -> None:
    extended = PairInsert(
        language="eng",
        decade=1950,
        score=0.91,
        band="ge90",
        source="banded",
        marc_control_id="ctrl-x",
        marc_json="{}",
        marc_title="t",
        marc_author="a",
        marc_publisher="p",
        marc_year=1953,
        nypl_uuid="uuid-x",
        cce_title="CCE Title",
        cce_author="CCE Author",
        cce_publishers="Pub A | Pub B",
        cce_claimants="Claimant A",
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R12345",
        evidence_json="{}",
        cce_renewal_id="R200001",
        cce_renewal_oreg="A111111",
        cce_renewal_rdat="1968-05-15",
        cce_renewal_author="Smith, John",
        cce_renewal_title="A study of widgets",
        cce_renewal_claimants="Acme Press|PWH",
        cce_renewal_new_matter="added ch. 7",
    )
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(extended)
        row = db.next_unlabeled()
    assert row is not None
    assert row.cce_renewal_id == "R200001"
    assert row.cce_renewal_oreg == "A111111"
    assert row.cce_renewal_rdat == "1968-05-15"
    assert row.cce_renewal_author == "Smith, John"
    assert row.cce_renewal_title == "A study of widgets"
    assert row.cce_renewal_claimants == "Acme Press|PWH"
    assert row.cce_renewal_new_matter == "added ch. 7"


def test_renewal_details_default_to_null(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair())
        row = db.next_unlabeled()
    assert row is not None
    assert row.cce_renewal_id is None
    assert row.cce_renewal_oreg is None
    assert row.cce_renewal_rdat is None
    assert row.cce_renewal_author is None
    assert row.cce_renewal_title is None
    assert row.cce_renewal_claimants is None
    assert row.cce_renewal_new_matter is None


def test_init_schema_adds_renewal_columns_to_legacy_pair_table(
    tmp_path: Path,
) -> None:
    from sqlite3 import connect as sqlite_connect

    db_path = tmp_path / "legacy_pair.db"
    legacy = sqlite_connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE review_pair (
            id INTEGER PRIMARY KEY,
            language TEXT NOT NULL,
            decade INTEGER,
            score REAL NOT NULL,
            band TEXT NOT NULL,
            source TEXT NOT NULL,
            marc_control_id TEXT NOT NULL,
            marc_json TEXT NOT NULL,
            marc_title TEXT,
            marc_author TEXT,
            marc_publisher TEXT,
            marc_year INTEGER,
            nypl_uuid TEXT NOT NULL,
            cce_title TEXT,
            cce_author TEXT,
            cce_publishers TEXT,
            cce_claimants TEXT,
            cce_reg_year INTEGER,
            cce_was_renewed INTEGER,
            cce_regnum TEXT,
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO review_pair (
            language, decade, score, band, source, marc_control_id, marc_json,
            marc_title, marc_author, marc_publisher, marc_year, nypl_uuid,
            cce_title, cce_author, cce_publishers, cce_claimants, cce_reg_year,
            cce_was_renewed, cce_regnum, evidence_json, created_at
        ) VALUES (
            'eng', 1950, 0.9, 'ge90', 'banded', 'ctrl-legacy', '{}',
            't', 'a', 'p', 1953, 'uuid-legacy',
            'CCE', 'CCE Author', 'Pub', 'Claim', 1953,
            1, 'R1', '{}', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    legacy.commit()
    legacy.close()

    with ReviewDb.connect(db_path) as db:
        columns = {row[1] for row in db._conn.execute("PRAGMA table_info(review_pair)")}
        assert "cce_renewal_id" in columns
        assert "cce_renewal_oreg" in columns
        assert "cce_renewal_rdat" in columns
        assert "cce_renewal_author" in columns
        assert "cce_renewal_title" in columns
        assert "cce_renewal_claimants" in columns
        assert "cce_renewal_new_matter" in columns
        row = db.get_pair(1)
    assert row is not None
    assert row.cce_renewal_id is None
    assert row.cce_renewal_new_matter is None


def test_round_trip_preserves_evidence_sources_json(tmp_path: Path) -> None:
    sources = '{"title.token_set": "title_main ↔ title"}'
    pair = PairInsert(
        language="eng",
        decade=1950,
        score=0.91,
        band="ge90",
        source="banded",
        marc_control_id="ctrl-s",
        marc_json="{}",
        marc_title="t",
        marc_author="a",
        marc_publisher="p",
        marc_year=1953,
        nypl_uuid="uuid-s",
        cce_title="CCE",
        cce_author="A",
        cce_publishers="P",
        cce_claimants="C",
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json='{"title.token_set": 0.91}',
        evidence_sources_json=sources,
    )
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(pair)
        row = db.next_unlabeled()
    assert row is not None
    assert row.evidence_sources_json == sources


def test_evidence_sources_json_defaults_to_empty_object(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair())
        row = db.next_unlabeled()
    assert row is not None
    assert row.evidence_sources_json == "{}"


def test_init_schema_adds_evidence_sources_json_to_legacy_pair_table(tmp_path: Path) -> None:
    from sqlite3 import connect as sqlite_connect

    db_path = tmp_path / "legacy_sources.db"
    legacy = sqlite_connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE review_pair (
            id INTEGER PRIMARY KEY,
            language TEXT NOT NULL,
            decade INTEGER,
            score REAL NOT NULL,
            band TEXT NOT NULL,
            source TEXT NOT NULL,
            marc_control_id TEXT NOT NULL,
            marc_json TEXT NOT NULL,
            marc_title TEXT,
            marc_author TEXT,
            marc_publisher TEXT,
            marc_year INTEGER,
            nypl_uuid TEXT NOT NULL,
            cce_title TEXT,
            cce_author TEXT,
            cce_publishers TEXT,
            cce_claimants TEXT,
            cce_reg_year INTEGER,
            cce_was_renewed INTEGER,
            cce_regnum TEXT,
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO review_pair (
            language, decade, score, band, source, marc_control_id, marc_json,
            marc_title, marc_author, marc_publisher, marc_year, nypl_uuid,
            cce_title, cce_author, cce_publishers, cce_claimants, cce_reg_year,
            cce_was_renewed, cce_regnum, evidence_json, created_at
        ) VALUES (
            'eng', 1950, 0.9, 'ge90', 'banded', 'ctrl-legacy', '{}',
            't', 'a', 'p', 1953, 'uuid-legacy',
            'CCE', 'A', 'P', 'C', 1953,
            1, 'R1', '{}', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    legacy.commit()
    legacy.close()

    with ReviewDb.connect(db_path) as db:
        columns = {row[1] for row in db._conn.execute("PRAGMA table_info(review_pair)")}
        assert "evidence_sources_json" in columns
        row = db.get_pair(1)
    assert row is not None
    assert row.evidence_sources_json == "{}"


def test_round_trip_preserves_audit_note(tmp_path: Path) -> None:
    note = "you=match · learned=0.20 · weighted=0.85 · [model-vs-model]"
    pair = PairInsert(
        language="eng",
        decade=1950,
        score=0.20,
        band="model-vs-model",
        source="banded",
        marc_control_id="ctrl-a",
        marc_json="{}",
        marc_title="t",
        marc_author="a",
        marc_publisher="p",
        marc_year=1953,
        nypl_uuid="uuid-a",
        cce_title="CCE",
        cce_author="A",
        cce_publishers="P",
        cce_claimants="C",
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json="{}",
        audit_note=note,
    )
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(pair)
        row = db.next_unlabeled()
    assert row is not None
    assert row.audit_note == note


def test_audit_note_defaults_to_null(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        db.insert_pair(_pair())
        row = db.next_unlabeled()
    assert row is not None
    assert row.audit_note is None


def test_init_schema_adds_audit_note_to_legacy_pair_table(tmp_path: Path) -> None:
    from sqlite3 import connect as sqlite_connect

    db_path = tmp_path / "legacy_audit.db"
    legacy = sqlite_connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE review_pair (
            id INTEGER PRIMARY KEY,
            language TEXT NOT NULL,
            decade INTEGER,
            score REAL NOT NULL,
            band TEXT NOT NULL,
            source TEXT NOT NULL,
            marc_control_id TEXT NOT NULL,
            marc_json TEXT NOT NULL,
            marc_title TEXT,
            marc_author TEXT,
            marc_publisher TEXT,
            marc_year INTEGER,
            nypl_uuid TEXT NOT NULL,
            cce_title TEXT,
            cce_author TEXT,
            cce_publishers TEXT,
            cce_claimants TEXT,
            cce_reg_year INTEGER,
            cce_was_renewed INTEGER,
            cce_regnum TEXT,
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO review_pair (
            language, decade, score, band, source, marc_control_id, marc_json,
            marc_title, marc_author, marc_publisher, marc_year, nypl_uuid,
            cce_title, cce_author, cce_publishers, cce_claimants, cce_reg_year,
            cce_was_renewed, cce_regnum, evidence_json, created_at
        ) VALUES (
            'eng', 1950, 0.9, 'ge90', 'banded', 'ctrl-legacy', '{}',
            't', 'a', 'p', 1953, 'uuid-legacy',
            'CCE', 'A', 'P', 'C', 1953,
            1, 'R1', '{}', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    legacy.commit()
    legacy.close()

    with ReviewDb.connect(db_path) as db:
        columns = {row[1] for row in db._conn.execute("PRAGMA table_info(review_pair)")}
        assert "audit_note" in columns
        row = db.get_pair(1)
    assert row is not None
    assert row.audit_note is None


def test_add_label_defaults_categories_to_empty_tuple(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        db.add_label(pair_id, VERDICT_MATCH)
        rows = db.iter_labeled_pairs()
    assert rows[0].categories == ()


def test_add_label_round_trips_categories(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        db.add_label(
            pair_id,
            VERDICT_MATCH,
            categories=("translation", "ocr_confusion"),
        )
        rows = db.iter_labeled_pairs()
    assert rows[0].categories == ("translation", "ocr_confusion")


def test_insert_existing_label_round_trips_categories(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair_id = db.insert_pair(_pair())
        db.insert_existing_label(
            pair_id=pair_id,
            verdict=VERDICT_MATCH,
            labeled_at="2024-01-01T00:00:00+00:00",
            categories=("generic_title",),
        )
        rows = db.iter_labeled_pairs()
    assert rows[0].categories == ("generic_title",)


def test_iter_labeled_pairs_filters_to_single_category(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        c = db.insert_pair(_pair(control_id="c", nypl_uuid="u-c"))
        db.add_label(a, VERDICT_MATCH, categories=("translation",))
        db.add_label(b, VERDICT_NO_MATCH, categories=("ocr_confusion",))
        db.add_label(c, VERDICT_MATCH)
        rows = db.iter_labeled_pairs(LabelFilters(categories=("translation",)))
    assert {row.pair_id for row in rows} == {a}


def test_iter_labeled_pairs_categories_filter_uses_or_semantics(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        c = db.insert_pair(_pair(control_id="c", nypl_uuid="u-c"))
        d = db.insert_pair(_pair(control_id="d", nypl_uuid="u-d"))
        db.add_label(a, VERDICT_MATCH, categories=("translation",))
        db.add_label(b, VERDICT_NO_MATCH, categories=("ocr_confusion",))
        db.add_label(c, VERDICT_MATCH, categories=("translation", "ocr_confusion"))
        db.add_label(d, VERDICT_MATCH, categories=("generic_title",))
        rows = db.iter_labeled_pairs(LabelFilters(categories=("translation", "ocr_confusion")))
    assert {row.pair_id for row in rows} == {a, b, c}


def test_count_labeled_pairs_matches_iter_under_categories_filter(tmp_path: Path) -> None:
    with ReviewDb.connect(tmp_path / "review.db") as db:
        a = db.insert_pair(_pair(control_id="a", nypl_uuid="u-a"))
        b = db.insert_pair(_pair(control_id="b", nypl_uuid="u-b"))
        c = db.insert_pair(_pair(control_id="c", nypl_uuid="u-c"))
        db.add_label(a, VERDICT_MATCH, categories=("translation",))
        db.add_label(b, VERDICT_NO_MATCH, categories=("ocr_confusion",))
        db.add_label(c, VERDICT_MATCH)
        filters = LabelFilters(categories=("translation", "ocr_confusion"))
        assert db.count_labeled_pairs(filters) == len(db.iter_labeled_pairs(filters))


def test_decoding_unknown_category_raises(tmp_path: Path) -> None:
    from msgspec import ValidationError

    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        pair_id = db.insert_pair(_pair())
        db.add_label(pair_id, VERDICT_MATCH)
        db._conn.execute(
            "UPDATE label SET categories = ? WHERE pair_id = ?",
            ('["not_a_real_category"]', pair_id),
        )
    with ReviewDb.connect(db_path) as db, raises(ValidationError):
        db.iter_labeled_pairs()


def test_decoding_empty_categories_text_collapses_to_empty_tuple(tmp_path: Path) -> None:
    """A legacy/empty ``categories`` TEXT decodes to ``()`` rather than crashing.

    The column defaults to ``'[]'`` but the safety branch in ``_decode_categories``
    returns ``()`` when the stored text is empty so a hand-edited row or a
    pre-default value does not trip the JSON decoder.
    """
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        pair_id = db.insert_pair(_pair())
        db.add_label(pair_id, VERDICT_MATCH)
        db._conn.execute(
            "UPDATE label SET categories = '' WHERE pair_id = ?",
            (pair_id,),
        )
    with ReviewDb.connect(db_path) as db:
        rows = db.iter_labeled_pairs()
    assert rows[0].categories == ()


def test_pairing_type_defaults_to_registration(tmp_path: Path) -> None:
    """A pair inserted without ``pairing_type`` round-trips as ``registration``."""
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        pair_id = db.insert_pair(_pair())
    with ReviewDb.connect(db_path) as db:
        row = db.get_pair(pair_id)
    assert row is not None
    assert row.pairing_type == PAIRING_REGISTRATION


def test_legacy_db_without_pairing_type_column_is_backfilled(tmp_path: Path) -> None:
    """A pre-``pairing_type`` DB gains the column on connect with the default."""
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        pair_id = db.insert_pair(_pair())
        db._conn.execute("ALTER TABLE review_pair DROP COLUMN pairing_type")
    with ReviewDb.connect(db_path) as db:
        row = db.get_pair(pair_id)
    assert row is not None
    assert row.pairing_type == PAIRING_REGISTRATION
