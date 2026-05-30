"""Route smoke tests for the FastAPI review UI (``webui`` marker).

These exercise the thin route + template layer end-to-end against a temporary
``review.db`` via :class:`fastapi.testclient.TestClient`. They are deselected
from the default ``pdm run pytest`` (the web layer is excluded from coverage)
and run via ``pdm run webui``.
"""

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
from msgspec.json import encode as json_encode
from pytest import fixture
from pytest import mark

from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.review.app import create_app
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_matcher.models import MarcRecord

pytestmark = mark.webui


def _pair(
    *,
    language: str,
    control_id: str,
    nypl_uuid: str,
    extent: str | None = None,
    renewal_id: str | None = "R200001",
    renewal_oreg: str | None = "A111111",
    renewal_rdat: str | None = "1968-05-15",
    renewal_author: str | None = "Smith, John",
    renewal_title: str | None = "A study of widgets",
    renewal_claimants: str | None = "Estate of Jane Doe|PWH",
    renewal_new_matter: str | None = "added chapter 7",
    evidence_sources_json: str = "{}",
    lccn: str | None = "40012345",
    oclc: str | None = "0001",
    isbns: tuple[str, ...] = ("9780000000000",),
    cce_desc: str = "vi, 200 p.",
    cce_lccn: str | None = "28000854",
) -> PairInsert:
    marc = MarcRecord(
        control_id=control_id,
        title="A Studied Title",
        title_main="A Studied Title",
        lccn=lccn,
        oclc=oclc,
        isbns=isbns,
        title_part_number="Pt. 2",
        title_part_name="The empire of Sebastopol",
        main_author="Doe, Jane",
        edition="2nd ed.",
        publication_place="New York",
        publisher="Acme Press",
        publication_date_raw="c1953.",
        publication_year=1953,
        extent=extent if extent is not None else "xxiv, 841 p.",
        language_code=language,
        country_code="nyu",
    )
    return PairInsert(
        language=language,
        decade=1950,
        score=0.93,
        band="ge90",
        source="banded",
        marc_control_id=control_id,
        marc_json=json_encode(marc).decode("utf-8"),
        marc_title=marc.title,
        marc_author=marc.main_author,
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        nypl_uuid=nypl_uuid,
        cce_title="A Studied Title",
        cce_author="Jane Doe",
        cce_publishers="Acme Press",
        cce_claimants="Jane Doe",
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R12345",
        evidence_json='{"title.token_set": 1.0, "name.author": 0.8}',
        cce_edition="2nd ed.",
        cce_publication_places="New York; London",
        cce_author_place="Cambridge, Mass.",
        cce_author_is_claimant=True,
        cce_copies="2c.",
        cce_aff_date="1953-06-01",
        cce_desc=cce_desc,
        cce_notes="first note\nsecond note",
        cce_new_matter_claimed="added chapter 5",
        cce_copy_date="1953-04-01",
        cce_notice_date="1953-04-02",
        cce_lccn=cce_lccn,
        cce_prev_regnums="A100000; A200000",
        cce_renewal_id=renewal_id,
        cce_renewal_oreg=renewal_oreg,
        cce_renewal_rdat=renewal_rdat,
        cce_renewal_author=renewal_author,
        cce_renewal_title=renewal_title,
        cce_renewal_claimants=renewal_claimants,
        cce_renewal_new_matter=renewal_new_matter,
        evidence_sources_json=evidence_sources_json,
    )


@fixture
def vault_path(tmp_path: Path) -> Path:
    return tmp_path / "vault.jsonl"


@fixture
def client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair(language="eng", control_id="eng-1", nypl_uuid="u-eng-1"))
        db.insert_pair(_pair(language="fre", control_id="fre-1", nypl_uuid="u-fre-1"))
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def skip_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair(language="eng", control_id="s-1", nypl_uuid="u-s-1"))
        db.insert_pair(_pair(language="eng", control_id="s-2", nypl_uuid="u-s-2"))
        db.insert_pair(_pair(language="eng", control_id="s-3", nypl_uuid="u-s-3"))
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def ebook_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="eng-ebook-1",
                nypl_uuid="u-eng-ebook-1",
                extent="1 online resource (xxi, 406 p.)",
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def series_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="eng-series-1",
                nypl_uuid="u-eng-series-1",
                extent="v",
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def translation_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="eng-tr-1",
                nypl_uuid="u-eng-tr-1",
                cce_desc="312 p. tr. from the French",
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def no_renewal_details_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="np-1",
                nypl_uuid="u-np-1",
                renewal_id=None,
                renewal_oreg=None,
                renewal_rdat=None,
                renewal_author=None,
                renewal_title=None,
                renewal_claimants=None,
                renewal_new_matter=None,
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def no_identifiers_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="ni-1",
                nypl_uuid="u-ni-1",
                lccn=None,
                oclc=None,
                isbns=(),
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def empty_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def hyphenated_lccn_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="hyp-1",
                nypl_uuid="u-hyp-1",
                cce_lccn="37-13688",
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def evidence_source_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="es-1",
                nypl_uuid="u-es-1",
                evidence_sources_json=(
                    '{"title.token_set": "title_main ↔ title", '
                    '"name.author": "main_author ↔ author_name"}'
                ),
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def nav_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    """Five pairs with one pre-labeled so ``/pair/5`` is reachable as a jump.

    Cookie-based navigation tests need a way to demonstrate the "jump"
    transition where the visited pair is *not* adjacent to the cursor in the
    trail. Having pair 5 pre-labeled lets a test walk 1→2→3 by submitting
    verdicts and then jump to 5 via direct URL, exercising the truncate-
    forward branch of :func:`nav_history.advance`.
    """
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        for i in range(1, 6):
            db.insert_pair(_pair(language="eng", control_id=f"n-{i}", nypl_uuid=f"u-n-{i}"))
        db.add_label(5, "match")
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def labels_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        eng_match = db.insert_pair(_pair(language="eng", control_id="eng-m", nypl_uuid="u-eng-m"))
        eng_no = db.insert_pair(_pair(language="eng", control_id="eng-n", nypl_uuid="u-eng-n"))
        fre_match = db.insert_pair(_pair(language="fre", control_id="fre-m", nypl_uuid="u-fre-m"))
        db.add_label(eng_match, "match")
        db.add_label(eng_no, "no_match", note="weird publisher")
        db.add_label(fre_match, "match")
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def pagination_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        for i in range(110):
            pair_id = db.insert_pair(_pair(language="eng", control_id=f"c-{i}", nypl_uuid=f"u-{i}"))
            db.add_label(pair_id, "match")
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def sorted_labels_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    """Three labeled pairs with known, well-separated ``labeled_at`` timestamps.

    Used to assert table ordering across both ``sort=desc`` (the default,
    newest-first) and ``sort=asc`` without relying on wall-clock ordering.
    """
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        oldest = db.insert_pair(_pair(language="eng", control_id="old", nypl_uuid="u-old"))
        middle = db.insert_pair(_pair(language="eng", control_id="mid", nypl_uuid="u-mid"))
        newest = db.insert_pair(_pair(language="eng", control_id="new", nypl_uuid="u-new"))
        db.insert_existing_label(oldest, "match", "2024-01-01T00:00:00+00:00")
        db.insert_existing_label(middle, "match", "2024-06-01T00:00:00+00:00")
        db.insert_existing_label(newest, "match", "2024-12-01T00:00:00+00:00")
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


def test_index_renders_a_card(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "A Studied Title" in response.text
    assert "title.token_set" in response.text
    assert "Renewed" in response.text


def test_index_language_filter_selects_french(client: TestClient) -> None:
    response = client.get("/", params={"language": "fre"})
    assert response.status_code == 200
    assert "fre-1" in response.text


def test_label_writes_row_and_redirects(client: TestClient) -> None:
    card = client.get("/")
    assert "pair #1" in card.text
    response = client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match", "language": "eng"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/?language=eng"
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert ">1<" in stats.text


def test_label_advances_to_next_pair(client: TestClient) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "no_match"}, follow_redirects=False)
    nxt = client.get("/")
    assert nxt.status_code == 200
    assert "fre-1" in nxt.text


def test_pair_route_renders_specific_pair(client: TestClient) -> None:
    response = client.get("/pair/2")
    assert response.status_code == 200
    assert "fre-1" in response.text


def test_pair_route_404_for_missing(client: TestClient) -> None:
    response = client.get("/pair/9999")
    assert response.status_code == 404
    assert "not found" in response.text


def test_stats_route_renders_progress(client: TestClient) -> None:
    response = client.get("/stats")
    assert response.status_code == 200
    assert "labeled" in response.text
    assert "By language" in response.text


def test_stats_route_omits_reason_and_annotation_sections(client: TestClient) -> None:
    response = client.get("/stats")
    assert response.status_code == 200
    assert "Reasons" not in response.text
    assert "Field annotations" not in response.text


def test_first_get_sets_nav_cookie_with_no_back_or_forward(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="back-link"' not in response.text
    assert 'id="forward-link"' not in response.text
    assert "pdm_nav" in response.cookies


def test_back_link_targets_previously_viewed_pair_after_label(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    nxt = client.get("/")
    assert 'id="back-link"' in nxt.text
    assert "/pair/1" in nxt.text


def test_back_link_preserves_filter(skip_client: TestClient) -> None:
    skip_client.get("/", params={"language": "eng"})
    skip_client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match", "language": "eng"},
        follow_redirects=False,
    )
    nxt = skip_client.get("/", params={"language": "eng"})
    assert "/pair/1?language=eng" in nxt.text


def test_card_links_marc_control_id_to_princeton_catalog(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="https://catalog.princeton.edu/catalog/eng-1"' in response.text


def test_card_renders_note_textarea_not_chips_or_annotation_grid(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<textarea name="note"' in response.text
    assert 'class="chip' not in response.text
    assert "annotation-cell" not in response.text
    assert "Different work" not in response.text
    assert "field annotations" not in response.text


def test_label_persists_note_to_db_and_vault(client: TestClient, vault_path: Path) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "note": "title collision"},
        follow_redirects=False,
    )
    [entry] = list(iter_entries(vault_path))
    assert entry.verdict == "no_match"
    assert entry.note == "title collision"


def test_label_blank_note_collapses_to_none(client: TestClient, vault_path: Path) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match", "note": "   "},
        follow_redirects=False,
    )
    [entry] = list(iter_entries(vault_path))
    assert entry.note is None


def test_empty_queue_page_when_all_labeled(client: TestClient) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.post("/label", data={"pair_id": "2", "verdict": "match"}, follow_redirects=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "All done" in response.text


def test_ebook_badge_renders_for_online_resource(ebook_client: TestClient) -> None:
    response = ebook_client.get("/")
    assert response.status_code == 200
    assert '<span class="badge-ebook">E-book reprint</span>' in response.text


def test_ebook_badge_absent_for_physical_pairs(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "E-book reprint" not in response.text
    assert '<span class="badge-ebook">' not in response.text


def test_series_badge_renders_for_bare_v_extent(series_client: TestClient) -> None:
    response = series_client.get("/")
    assert response.status_code == 200
    assert '<span class="badge-series">Open multipart set</span>' in response.text


def test_series_badge_absent_for_closed_monograph_pairs(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Open multipart set" not in response.text
    assert '<span class="badge-series">' not in response.text


def test_translation_badge_renders_when_cue_present(translation_client: TestClient) -> None:
    response = translation_client.get("/")
    assert response.status_code == 200
    assert '<span class="badge-translation">Translation</span>' in response.text


def test_translation_badge_absent_for_non_translation_pairs(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Translation" not in response.text
    assert '<span class="badge-translation">' not in response.text


def test_label_appends_to_vault(client: TestClient, vault_path: Path) -> None:
    assert not vault_path.exists()
    response = client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert vault_path.exists()
    entries = list(iter_entries(vault_path))
    assert len(entries) == 1
    entry = entries[0]
    assert entry.marc_control_id == "eng-1"
    assert entry.nypl_uuid == "u-eng-1"
    assert entry.verdict == "match"
    assert entry.labeler == "jpstroop"
    assert entry.schema == 4
    assert entry.marc_identifiers.lccn == "40012345"
    assert entry.marc_identifiers.oclc == "0001"
    assert entry.marc_identifiers.isbns == ("9780000000000",)
    assert entry.cce_regnum == "R12345"
    assert entry.cce_renewal_id == "R200001"
    assert entry.cce_renewal_oreg == "A111111"


def test_label_appends_cce_fields_with_nulls_when_no_renewal(
    no_renewal_details_client: TestClient, vault_path: Path
) -> None:
    response = no_renewal_details_client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    [entry] = list(iter_entries(vault_path))
    assert entry.cce_regnum == "R12345"
    assert entry.cce_renewal_id is None
    assert entry.cce_renewal_oreg is None


def test_label_appends_one_line_per_post(client: TestClient, vault_path: Path) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    after_first = vault_path.read_text(encoding="utf-8")
    client.post("/label", data={"pair_id": "2", "verdict": "no_match"}, follow_redirects=False)
    after_second = vault_path.read_text(encoding="utf-8")
    assert after_second.startswith(after_first)
    assert len(after_second.splitlines()) == 2


def test_label_relabel_preserves_history(client: TestClient, vault_path: Path) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "note": "second look"},
        follow_redirects=False,
    )
    history = list(iter_entries(vault_path))
    assert [event.verdict for event in history] == ["match", "no_match"]
    latest = current_entries(vault_path)
    assert latest[("eng-1", "u-eng-1")].verdict == "no_match"
    assert latest[("eng-1", "u-eng-1")].note == "second look"


def test_label_db_timestamp_matches_vault_timestamp(
    client: TestClient, vault_path: Path, tmp_path: Path
) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "note": "hmm"},
        follow_redirects=False,
    )
    [vault_entry] = list(iter_entries(vault_path))
    with ReviewDb.connect(tmp_path / "review.db") as db:
        [label] = list(db.iter_current_labels())
    assert label.labeled_at == vault_entry.labeled_at
    assert label.note == vault_entry.note == "hmm"


def test_skip_query_excludes_single_pair(skip_client: TestClient) -> None:
    response = skip_client.get("/", params={"skip": 1})
    assert response.status_code == 200
    assert "s-2" in response.text
    assert "s-1" not in response.text


def test_skip_query_excludes_multiple_pairs(skip_client: TestClient) -> None:
    response = skip_client.get("/", params=[("skip", 1), ("skip", 2)])
    assert response.status_code == 200
    assert "s-3" in response.text
    assert "s-1" not in response.text
    assert "s-2" not in response.text


def test_skip_query_empty_returns_all_done(skip_client: TestClient) -> None:
    response = skip_client.get("/", params=[("skip", 1), ("skip", 2), ("skip", 3)])
    assert response.status_code == 200
    assert "All done" in response.text


def test_label_redirect_does_not_carry_skip_state(skip_client: TestClient) -> None:
    response = skip_client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    nxt = skip_client.get("/")
    assert "s-2" in nxt.text


def test_card_renders_skip_url_with_current_pair_id_appended(skip_client: TestClient) -> None:
    response = skip_client.get("/", params={"skip": 1})
    assert response.status_code == 200
    assert 'skipUrl = "/?skip=1&skip=2"' in response.text


def test_card_skip_url_includes_filters(skip_client: TestClient) -> None:
    response = skip_client.get("/", params={"language": "eng"})
    assert response.status_code == 200
    assert 'skipUrl = "/?language=eng&skip=1"' in response.text


def test_labels_route_with_no_labels_shows_empty_message(empty_client: TestClient) -> None:
    response = empty_client.get("/labels")
    assert response.status_code == 200
    assert "No labels yet" in response.text
    assert "<table" not in response.text


def test_labels_route_renders_table_for_each_label(labels_client: TestClient) -> None:
    response = labels_client.get("/labels")
    assert response.status_code == 200
    assert 'href="/pair/1"' in response.text
    assert 'href="/pair/2"' in response.text
    assert 'href="/pair/3"' in response.text
    assert "verdict-match" in response.text
    assert "verdict-no_match" in response.text


def test_labels_route_shows_note_column_with_recorded_text(labels_client: TestClient) -> None:
    response = labels_client.get("/labels")
    assert response.status_code == 200
    assert ">note<" in response.text
    assert "weird publisher" in response.text


def test_labels_route_omits_reason_and_annotation_columns(labels_client: TestClient) -> None:
    response = labels_client.get("/labels")
    assert response.status_code == 200
    assert ">reasons<" not in response.text
    assert ">annotations<" not in response.text


def test_labels_route_filter_by_verdict_narrows_rows(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"verdict": "no_match"})
    assert response.status_code == 200
    assert 'href="/pair/2"' in response.text
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/3"' not in response.text


def test_labels_route_filter_by_language_narrows_rows(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"language": "fre"})
    assert response.status_code == 200
    assert 'href="/pair/3"' in response.text
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/2"' not in response.text


def test_labels_route_substring_search_matches_marc_title(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"q": "studied"})
    assert response.status_code == 200
    assert 'href="/pair/1"' in response.text
    response = labels_client.get("/labels", params={"q": "no-such-thing"})
    assert response.status_code == 200
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/2"' not in response.text


def test_labels_route_combined_filters_and_together(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"verdict": "no_match", "language": "eng"})
    assert response.status_code == 200
    assert 'href="/pair/2"' in response.text
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/3"' not in response.text


def test_labels_route_pair_id_links_to_pair_detail(labels_client: TestClient) -> None:
    response = labels_client.get("/labels")
    assert response.status_code == 200
    assert 'href="/pair/1"' in response.text


def test_labels_route_pagination_caps_rows_and_disables_next_on_last_page(
    pagination_client: TestClient,
) -> None:
    page_two = pagination_client.get("/labels", params={"page": 2})
    assert page_two.status_code == 200
    body_rows = page_two.text.split("<tbody>")[1].split("</tbody>")[0]
    assert body_rows.count("<tr>") == 10
    assert "page 2 of 2" in page_two.text
    assert '<span class="disabled">Next' in page_two.text


def test_labels_route_pagination_disables_prev_on_first_page(
    pagination_client: TestClient,
) -> None:
    page_one = pagination_client.get("/labels")
    assert page_one.status_code == 200
    body_rows = page_one.text.split("<tbody>")[1].split("</tbody>")[0]
    assert body_rows.count("<tr>") == 100
    assert '<span class="disabled">' in page_one.text
    assert "Prev" in page_one.text


def test_labels_route_shows_filter_summary_when_active(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"verdict": "match"})
    assert response.status_code == 200
    assert "Showing 2 of 3 labels" in response.text
    assert "verdict=match" in response.text
    assert "Clear filters" in response.text


def test_labels_route_nav_link_present_on_other_pages(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/labels"' in response.text


def _labels_table_pair_ids(html: str) -> list[str]:
    """Extract the ordered pair-link hrefs from a ``/labels`` table body."""
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    parts: list[str] = []
    for chunk in body.split('href="/pair/')[1:]:
        parts.append(chunk.split('"', 1)[0])
    return parts


def test_labels_route_default_sort_is_desc(sorted_labels_client: TestClient) -> None:
    response = sorted_labels_client.get("/labels")
    assert response.status_code == 200
    assert _labels_table_pair_ids(response.text) == ["3", "2", "1"]


def test_labels_route_sort_asc_reorders_oldest_first(sorted_labels_client: TestClient) -> None:
    response = sorted_labels_client.get("/labels", params={"sort": "asc"})
    assert response.status_code == 200
    assert _labels_table_pair_ids(response.text) == ["1", "2", "3"]


def test_labels_route_garbage_sort_falls_back_to_default(
    sorted_labels_client: TestClient,
) -> None:
    response = sorted_labels_client.get("/labels", params={"sort": "garbage"})
    assert response.status_code == 200
    assert _labels_table_pair_ids(response.text) == ["3", "2", "1"]


def test_labels_route_explicit_desc_renders_without_error(
    sorted_labels_client: TestClient,
) -> None:
    response = sorted_labels_client.get("/labels", params={"sort": "desc"})
    assert response.status_code == 200
    assert _labels_table_pair_ids(response.text) == ["3", "2", "1"]


def test_labels_route_renders_sort_select_with_asc_selected(
    sorted_labels_client: TestClient,
) -> None:
    response = sorted_labels_client.get("/labels", params={"sort": "asc"})
    assert response.status_code == 200
    assert '<select name="sort">' in response.text
    assert '<option value="asc" selected>' in response.text
    assert '<option value="desc">' in response.text


def test_labels_route_renders_sort_select_with_desc_selected_by_default(
    sorted_labels_client: TestClient,
) -> None:
    response = sorted_labels_client.get("/labels")
    assert response.status_code == 200
    assert '<option value="desc" selected>' in response.text
    assert '<option value="asc">' in response.text


def test_labels_route_sort_preserved_in_pagination_links(
    pagination_client: TestClient,
) -> None:
    response = pagination_client.get("/labels", params={"sort": "asc"})
    assert response.status_code == 200
    assert "sort=asc" in response.text
    assert "page=2" in response.text


def test_labels_route_clear_filters_link_preserves_non_default_sort(
    labels_client: TestClient,
) -> None:
    response = labels_client.get("/labels", params={"verdict": "match", "sort": "asc"})
    assert response.status_code == 200
    assert 'href="/labels?sort=asc">Clear filters</a>' in response.text


def test_labels_route_clear_filters_link_omits_default_sort(
    labels_client: TestClient,
) -> None:
    response = labels_client.get("/labels", params={"verdict": "match"})
    assert response.status_code == 200
    assert 'href="/labels">Clear filters</a>' in response.text


def test_labels_route_sort_alone_is_not_active_filter(
    sorted_labels_client: TestClient,
) -> None:
    response = sorted_labels_client.get("/labels", params={"sort": "asc"})
    assert response.status_code == 200
    assert "Clear filters" not in response.text


def test_card_renders_extended_cce_fields(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Cambridge, Mass." in response.text
    assert "Author is claimant" in response.text
    assert "2nd ed." in response.text
    assert "New York; London" in response.text
    assert "vi, 200 p." in response.text
    assert "added chapter 5" in response.text
    assert "2c." in response.text
    assert "first note" in response.text
    assert "second note" in response.text
    assert "1953-04-01" in response.text
    assert "1953-06-01" in response.text
    assert "1953-04-02" in response.text


def test_card_omits_extended_cce_rows_when_absent(empty_client: TestClient) -> None:
    response = empty_client.get("/")
    assert response.status_code == 200
    assert "author place" not in response.text
    assert "Author is claimant" not in response.text
    assert "new matter claimed" not in response.text
    assert "affidavit date" not in response.text
    assert "lccn.loc.gov" not in response.text
    assert "previous registrations" not in response.text


def test_card_renders_lccn_as_lccn_loc_gov_link(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="https://lccn.loc.gov/28000854"' in response.text
    assert ">28000854<" in response.text


def test_card_renders_hyphenated_cce_lccn_with_canonical_primary_and_raw_parenthetical(
    hyphenated_lccn_client: TestClient,
) -> None:
    response = hyphenated_lccn_client.get("/")
    assert response.status_code == 200
    assert 'href="https://lccn.loc.gov/37013688"' in response.text
    assert ">37013688</a>" in response.text
    assert "<small>(37-13688)</small>" in response.text


def test_card_renders_prev_regnums_row(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "previous registrations" in response.text
    assert "A100000; A200000" in response.text


def test_card_renders_extended_marc_fields(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "publication place" in response.text
    assert ">New York<" in response.text
    assert "date (raw)" in response.text
    assert "c1953." in response.text
    assert ">extent<" in response.text
    assert "xxiv, 841 p." in response.text
    assert ">isbns<" in response.text
    assert "9780000000000" in response.text
    assert ">oclc<" in response.text
    assert 'href="https://www.worldcat.org/oclc/0001"' in response.text
    assert "title parts" in response.text
    assert "Pt. 2: The empire of Sebastopol" in response.text


def test_card_omits_extended_marc_rows_when_absent(empty_client: TestClient) -> None:
    response = empty_client.get("/")
    assert response.status_code == 200
    assert "publication place" not in response.text
    assert "date (raw)" not in response.text
    assert "title parts" not in response.text
    assert ">isbns<" not in response.text
    assert ">oclc<" not in response.text
    assert "worldcat.org" not in response.text


def test_card_renders_hathitrust_link_when_oclc_present(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert ">HathiTrust<" in response.text
    assert 'href="https://catalog.hathitrust.org/api/volumes/oclc/0001.html"' in response.text


def test_card_omits_hathitrust_link_when_no_identifiers(
    no_identifiers_client: TestClient,
) -> None:
    response = no_identifiers_client.get("/")
    assert response.status_code == 200
    assert ">HathiTrust<" not in response.text
    assert "catalog.hathitrust.org" not in response.text


def test_card_renders_renewal_details_block_when_populated(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Renewal details" in response.text
    assert "1968-05-15" in response.text
    assert "Estate of Jane Doe|PWH" in response.text
    assert "added chapter 7" in response.text
    assert "R200001" in response.text
    assert "A111111" in response.text


def test_card_renders_renewal_claimants_diff_marker_when_registration_disagrees(
    client: TestClient,
) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "claimants-diff" in response.text


def test_card_omits_renewal_details_block_when_no_renewal_fields(
    no_renewal_details_client: TestClient,
) -> None:
    response = no_renewal_details_client.get("/")
    assert response.status_code == 200
    assert "Renewal details" not in response.text
    assert "renewal date" not in response.text


def test_card_renders_evidence_source_breadcrumb_when_present(
    evidence_source_client: TestClient,
) -> None:
    response = evidence_source_client.get("/")
    assert response.status_code == 200
    assert "via title_main ↔ title" in response.text
    assert "via main_author ↔ author_name" in response.text
    assert 'class="ev-source"' in response.text
    assert 'class="ev-source-row"' in response.text


def test_card_omits_evidence_source_breadcrumb_when_absent(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'class="ev-source"' not in response.text
    assert 'class="ev-source-row"' not in response.text


def test_pair_route_pre_fills_note_textarea_from_current_label(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "note": "title collision"},
        follow_redirects=False,
    )
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert "title collision</textarea>" in response.text


def test_pair_route_renders_empty_textarea_when_no_current_label(client: TestClient) -> None:
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert "></textarea>" in response.text


def test_pair_route_pre_fills_latest_note_after_relabel(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match", "note": "first take"},
        follow_redirects=False,
    )
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "note": "second take"},
        follow_redirects=False,
    )
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert "second take</textarea>" in response.text
    assert "first take</textarea>" not in response.text


def test_pair_route_marks_current_verdict_button(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match"},
        follow_redirects=False,
    )
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert 'class="no current"' in response.text
    assert 'class="match"' in response.text
    assert 'class="unsure"' in response.text


def test_pair_route_marks_match_current_verdict_button(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match"},
        follow_redirects=False,
    )
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert 'class="match current"' in response.text


def test_pair_route_marks_unsure_current_verdict_button(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "unsure"},
        follow_redirects=False,
    )
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert 'class="unsure current"' in response.text


def test_pair_route_no_current_class_when_unlabeled(client: TestClient) -> None:
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert " current" not in response.text


def test_forward_link_appears_after_back_step_through_history(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.get("/")
    back = client.get("/pair/1")
    assert back.status_code == 200
    assert 'id="forward-link"' in back.text
    assert 'href="/pair/2"' in back.text


def test_forward_link_absent_at_session_tail(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    tail = client.get("/")
    assert 'id="forward-link"' not in tail.text


def test_forward_link_preserves_filter(skip_client: TestClient) -> None:
    skip_client.get("/", params={"language": "eng"})
    skip_client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match", "language": "eng"},
        follow_redirects=False,
    )
    skip_client.get("/", params={"language": "eng"})
    back = skip_client.get("/pair/1", params={"language": "eng"})
    assert 'id="forward-link"' in back.text
    assert 'href="/pair/2?language=eng"' in back.text


def test_index_no_forward_link_at_session_head(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="forward-link"' not in response.text


def test_back_link_stable_across_relabel(skip_client: TestClient) -> None:
    skip_client.get("/")
    skip_client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    skip_client.get("/")
    skip_client.post("/label", data={"pair_id": "2", "verdict": "match"}, follow_redirects=False)
    skip_client.get("/")
    skip_client.post("/label", data={"pair_id": "1", "verdict": "no_match"}, follow_redirects=False)
    response = skip_client.get("/pair/3")
    assert response.status_code == 200
    assert 'id="back-link"' in response.text
    assert 'href="/pair/2' in response.text


def test_card_keybinding_hint_mentions_forward_when_available(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.get("/")
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert ">f</strong>" in response.text


def test_card_keybinding_handler_includes_forward_keys(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.get("/")
    response = client.get("/pair/1")
    assert response.status_code == 200
    assert 'case "f": case "ArrowRight":' in response.text
    assert "goForward()" in response.text


def test_back_then_forward_round_trip(skip_client: TestClient) -> None:
    skip_client.get("/")
    skip_client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    skip_client.get("/")
    skip_client.post("/label", data={"pair_id": "2", "verdict": "match"}, follow_redirects=False)
    skip_client.get("/")
    back = skip_client.get("/pair/2")
    assert 'id="back-link"' in back.text
    assert 'href="/pair/1' in back.text
    assert 'id="forward-link"' in back.text
    assert 'href="/pair/3' in back.text
    forward = skip_client.get("/pair/3")
    assert 'id="forward-link"' not in forward.text
    assert 'id="back-link"' in forward.text
    assert 'href="/pair/2' in forward.text


def test_jump_truncates_forward_branch(nav_client: TestClient) -> None:
    nav_client.get("/")
    nav_client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    nav_client.get("/")
    nav_client.post("/label", data={"pair_id": "2", "verdict": "match"}, follow_redirects=False)
    nav_client.get("/")
    back = nav_client.get("/pair/2")
    assert 'id="forward-link"' in back.text
    assert 'href="/pair/3' in back.text
    jump = nav_client.get("/pair/5")
    assert 'id="forward-link"' not in jump.text
    assert 'id="back-link"' in jump.text
    assert 'href="/pair/2' in jump.text


def test_refresh_does_not_grow_trail(client: TestClient) -> None:
    client.get("/")
    first = client.get("/")
    second = client.get("/")
    assert 'id="back-link"' not in first.text
    assert 'id="back-link"' not in second.text


def test_empty_queue_back_link_sourced_from_cookie(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.get("/")
    client.post("/label", data={"pair_id": "2", "verdict": "match"}, follow_redirects=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "All done" in response.text
    assert 'id="back-link"' in response.text
    assert "/pair/1" in response.text


def test_malformed_cookie_falls_back_to_empty_history(client: TestClient) -> None:
    client.cookies.set("pdm_nav", "not-json")
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="back-link"' not in response.text


def test_back_and_forward_share_a_flex_nav_row(skip_client: TestClient) -> None:
    """Both nav links sit in a single ``.nav-row`` flex container."""
    skip_client.get("/")
    skip_client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    skip_client.get("/")
    skip_client.post("/label", data={"pair_id": "2", "verdict": "match"}, follow_redirects=False)
    skip_client.get("/")
    response = skip_client.get("/pair/2")
    assert response.status_code == 200
    assert '<div class="nav-row">' in response.text
    nav_block = response.text.split('<div class="nav-row">')[1].split("</div>", 1)[0]
    assert 'id="back-link"' in nav_block
    assert 'id="forward-link"' in nav_block


def test_nav_row_omitted_when_neither_back_nor_forward(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<div class="nav-row">' not in response.text


def test_nav_row_present_with_only_back_link(client: TestClient) -> None:
    client.get("/")
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    response = client.get("/")
    assert response.status_code == 200
    assert '<div class="nav-row">' in response.text
    assert 'id="back-link"' in response.text
    assert 'id="forward-link"' not in response.text


def test_nav_cookie_is_session_cookie_no_max_age_or_expires(client: TestClient) -> None:
    response = client.get("/")
    set_cookie = response.headers.get("set-cookie", "").lower()
    assert "pdm_nav=" in set_cookie
    assert "max-age" not in set_cookie
    assert "expires" not in set_cookie
    assert "httponly" in set_cookie
    assert "samesite=strict" in set_cookie


def test_card_renders_filter_bar_with_language_and_band_selects(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<form class="filters-bar" method="get" action="/">' in response.text
    assert '<select name="language">' in response.text
    assert '<select name="band">' in response.text
    for choice in ("eng", "fre", "ger", "spa", "ita"):
        assert f'<option value="{choice}">{choice}</option>' in response.text
    for choice in ("ge90", "b80_90", "b70_80", "b60_70", "below"):
        assert f'<option value="{choice}">{choice}</option>' in response.text


def test_card_filter_bar_renders_b60_70_between_b70_80_and_below(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    band_section = response.text.split('<select name="band">')[1].split("</select>")[0]
    b70_index = band_section.index('value="b70_80"')
    b60_index = band_section.index('value="b60_70"')
    below_index = band_section.index('value="below"')
    assert b70_index < b60_index < below_index


def test_card_filter_bar_marks_selected_b60_70_band(client: TestClient) -> None:
    response = client.get("/", params={"band": "b60_70"})
    assert response.status_code == 200
    assert '<option value="b60_70" selected>b60_70</option>' in response.text


def test_card_filter_bar_marks_selected_language_and_band(client: TestClient) -> None:
    response = client.get("/", params={"language": "fre", "band": "b70_80"})
    assert response.status_code == 200
    assert '<option value="fre" selected>fre</option>' in response.text
    assert '<option value="b70_80" selected>b70_80</option>' in response.text


def test_card_filter_bar_shows_clear_link_when_filters_active(client: TestClient) -> None:
    response = client.get("/", params={"language": "fre"})
    assert response.status_code == 200
    assert '<a class="clear" href="/">Clear</a>' in response.text


def test_card_filter_bar_omits_clear_link_when_no_filters(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<a class="clear" href="/">Clear</a>' not in response.text


def test_pair_route_renders_filter_bar_with_language_selected(client: TestClient) -> None:
    response = client.get("/pair/1", params={"language": "eng"})
    assert response.status_code == 200
    assert '<form class="filters-bar" method="get" action="/">' in response.text
    assert '<option value="eng" selected>eng</option>' in response.text


def test_empty_queue_renders_filter_bar(empty_client: TestClient) -> None:
    response = empty_client.get("/")
    assert response.status_code == 200
    assert "All done" in response.text
    assert '<form class="filters-bar" method="get" action="/">' in response.text
    assert '<select name="language">' in response.text
    assert '<select name="band">' in response.text


def test_empty_queue_filter_bar_marks_selected_filters(empty_client: TestClient) -> None:
    response = empty_client.get("/", params={"language": "eng", "band": "below"})
    assert response.status_code == 200
    assert '<option value="eng" selected>eng</option>' in response.text
    assert '<option value="below" selected>below</option>' in response.text
    assert '<a class="clear" href="/">Clear</a>' in response.text
